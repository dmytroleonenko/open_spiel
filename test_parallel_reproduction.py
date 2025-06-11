#!/usr/bin/env python3
"""
Minimal reproduction of parallel test execution failures with coverage.
This script runs a single test case multiple times in parallel to isolate the issue.
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
        print(f"[Worker {worker_id}] Starting test")
        result = subprocess.run(
            cmd, 
            env=env, 
            timeout=300,  # 5 min timeout
            capture_output=True, 
            text=True
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[Worker {worker_id}] ✅ Test passed in {duration:.1f}s")
            return True, "PASSED", duration
        else:
            print(f"[Worker {worker_id}] ❌ Test failed in {duration:.1f}s")
            if result.stderr:
                print(f"[Worker {worker_id}] STDERR: {result.stderr[-800:]}")
            return False, "FAILED", duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[Worker {worker_id}] ⏰ Test TIMED OUT after {duration:.1f}s")
        return False, "TIMEOUT", duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"[Worker {worker_id}] 💥 Test crashed after {duration:.1f}s: {e}")
        return False, "CRASHED", duration

def worker_process(worker_id: int, test_case: str, temp_dir: str, result_queue: mp.Queue):
    """Worker process function."""
    try:
        success, status, duration = run_test_with_coverage(test_case, worker_id, temp_dir)
        result_queue.put((worker_id, success, status, duration))
    except Exception as e:
        print(f"[Worker {worker_id}] Process error: {e}")
        result_queue.put((worker_id, False, "PROCESS_ERROR", 0))

def main():
    """Run the same test multiple times in parallel to reproduce failures."""
    # Change to project root
    project_root = "/Users/dleonenko/open_spiel"
    os.chdir(project_root)
    
    # Test case that sometimes fails in parallel execution
    test_case = "open_spiel/python/algorithms/muzero_jax/tests/test_orchestrator_integration.py::TestActorLearnerIntegration::test_actor_buffer_interaction"
    
    # Run same test 8 times in parallel (like the coverage runner)
    num_workers = 8
    print(f"Running test case {num_workers} times in parallel with coverage...")
    print(f"Test: {test_case}")
    
    with tempfile.TemporaryDirectory(prefix="repro_coverage_") as temp_dir:
        print(f"Temporary coverage directory: {temp_dir}")
        
        # Set up result collection
        result_queue = mp.Queue()
        workers = []
        
        # Start all workers
        for worker_id in range(1, num_workers + 1):
            worker = mp.Process(
                target=worker_process,
                args=(worker_id, test_case, temp_dir, result_queue),
                name=f"repro-worker-{worker_id}"
            )
            worker.start()
            workers.append(worker)
            print(f"Started worker {worker_id}")
        
        # Collect results
        results = []
        for _ in range(num_workers):
            try:
                worker_id, success, status, duration = result_queue.get(timeout=600)
                results.append((worker_id, success, status, duration))
                print(f"Worker {worker_id}: {status} ({duration:.1f}s)")
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
        print("REPRODUCTION RESULTS")
        print(f"{'='*60}")
        
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        print(f"✅ Successful: {len(successful)}/{num_workers}")
        print(f"❌ Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailed workers:")
            for worker_id, success, status, duration in failed:
                print(f"  Worker {worker_id}: {status} ({duration:.1f}s)")
        
        # If any failures, the issue is reproduced
        if failed:
            print(f"\n🚨 ISSUE REPRODUCED: Same test fails {len(failed)}/{num_workers} times in parallel with coverage")
            print("This confirms the parallel execution + coverage instrumentation issue")
        else:
            print(f"\n✅ No failures reproduced - may need more iterations or different test")

if __name__ == "__main__":
    main() 