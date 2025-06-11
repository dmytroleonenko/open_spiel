#!/usr/bin/env python3
"""
Test the exact 5 failing tests in parallel with coverage to reproduce the issue.
"""
import subprocess
import os
import time
import multiprocessing as mp
import tempfile
from typing import Tuple

def run_test_with_coverage(test_case: str, worker_id: int, temp_dir: str) -> Tuple[bool, str, float]:
    """Run a single test case with coverage in a subprocess."""
    start_time = time.time()
    
    # Create unique coverage file for this worker
    coverage_file = os.path.join(temp_dir, f".coverage.worker_{worker_id}")
    
    # Set environment for this worker
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_ENABLE_X64"] = "false"
    env["COVERAGE_FILE"] = coverage_file
    
    cmd = [
        "python", "-m", "pytest",
        test_case,
        "--cov=open_spiel.python.algorithms.muzero_jax",
        "--cov-report=",  # Don't generate report yet
        "-p", "no:xdist",
        "-p", "no:testmon",
        "-v",
        "--override-ini=addopts=",
        "--tb=short"
    ]
    
    try:
        print(f"[Worker {worker_id}] Starting: {test_case.split('::')[-1]}")
        result = subprocess.run(
            cmd, 
            env=env, 
            timeout=300,  # 5 min timeout
            capture_output=True, 
            text=True
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[Worker {worker_id}] ✅ {test_case.split('::')[-1]} passed in {duration:.1f}s")
            return True, "PASSED", duration
        else:
            print(f"[Worker {worker_id}] ❌ {test_case.split('::')[-1]} failed in {duration:.1f}s")
            if result.stderr:
                print(f"[Worker {worker_id}] STDERR: {result.stderr[-800:]}")
            if result.stdout:
                print(f"[Worker {worker_id}] STDOUT: {result.stdout[-400:]}")
            return False, "FAILED", duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[Worker {worker_id}] ⏰ {test_case.split('::')[-1]} TIMED OUT after {duration:.1f}s")
        return False, "TIMEOUT", duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"[Worker {worker_id}] 💥 {test_case.split('::')[-1]} crashed after {duration:.1f}s: {e}")
        return False, "CRASHED", duration

def worker_process(worker_id: int, test_case: str, temp_dir: str, result_queue: mp.Queue):
    """Worker process function."""
    try:
        success, status, duration = run_test_with_coverage(test_case, worker_id, temp_dir)
        result_queue.put((worker_id, test_case, success, status, duration))
    except Exception as e:
        print(f"[Worker {worker_id}] Process error: {e}")
        result_queue.put((worker_id, test_case, False, "PROCESS_ERROR", 0))

def main():
    """Run the 5 failing tests in parallel to reproduce the issue."""
    # Change to project root
    project_root = "/Users/dleonenko/open_spiel"
    os.chdir(project_root)
    
    # The 5 test cases that fail in the coverage runner
    failing_tests = [
        "open_spiel/python/algorithms/muzero_jax/tests/test_orchestrator_integration.py::TestActorLearnerIntegration::test_actor_buffer_interaction",
        "open_spiel/python/algorithms/muzero_jax/tests/test_orchestrator_integration.py::TestActorLearnerIntegration::test_checkpoint_saving_loading", 
        "open_spiel/python/algorithms/muzero_jax/tests/test_orchestrator_integration.py::TestActorLearnerIntegration::test_learner_buffer_interaction",
        "open_spiel/python/algorithms/muzero_jax/tests/self_play/test_actor.py::TestActor::test_actor_decision_recurrent_function_creation",
        "open_spiel/python/algorithms/muzero_jax/tests/test_orchestrator_integration.py::TestOrchestrationWorkflow::test_workflow_initialization"
    ]
    
    print(f"Running the 5 failing tests in parallel with coverage to reproduce the issue...")
    print("Tests:")
    for i, test in enumerate(failing_tests, 1):
        print(f"  {i}. {test.split('::')[-1]}")
    
    with tempfile.TemporaryDirectory(prefix="failing_combo_coverage_") as temp_dir:
        print(f"\nTemporary coverage directory: {temp_dir}")
        
        # Set up result collection
        result_queue = mp.Queue()
        workers = []
        
        # Start all workers (one per test)
        for worker_id, test_case in enumerate(failing_tests, 1):
            worker = mp.Process(
                target=worker_process,
                args=(worker_id, test_case, temp_dir, result_queue),
                name=f"failing-worker-{worker_id}"
            )
            worker.start()
            workers.append(worker)
            print(f"Started worker {worker_id}: {test_case.split('::')[-1]}")
        
        # Collect results
        results = []
        for _ in range(len(failing_tests)):
            try:
                worker_id, test_case, success, status, duration = result_queue.get(timeout=600)
                results.append((worker_id, test_case, success, status, duration))
                test_name = test_case.split('::')[-1]
                print(f"Worker {worker_id} ({test_name}): {status} ({duration:.1f}s)")
            except Exception as e:
                print(f"Error collecting result: {e}")
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        # Summary
        print(f"\n{'='*60}")
        print("FAILING COMBINATION TEST RESULTS")
        print(f"{'='*60}")
        
        successful = [r for r in results if r[2]]
        failed = [r for r in results if not r[2]]
        
        print(f"✅ Successful: {len(successful)}/{len(failing_tests)}")
        print(f"❌ Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailed tests:")
            for worker_id, test_case, success, status, duration in failed:
                test_name = test_case.split('::')[-1]
                print(f"  Worker {worker_id} ({test_name}): {status} ({duration:.1f}s)")
        
        # Analysis
        if failed:
            print(f"\n🚨 ISSUE REPRODUCED: {len(failed)}/{len(failing_tests)} tests failed when run in parallel with coverage")
            print("This confirms that these specific tests have resource contention when run together")
            
            # Check if it's the checkpoint manager tests
            checkpoint_failures = [r for r in failed if "checkpoint" in r[1] or "buffer_interaction" in r[1]]
            if checkpoint_failures:
                print(f"📝 Note: {len(checkpoint_failures)} failures involve checkpoint/buffer tests - likely Orbax async cleanup issues")
        else:
            print(f"\n✅ All tests passed when run in parallel with coverage")
            print("The resource contention may be intermittent or require more stress")

if __name__ == "__main__":
    main() 