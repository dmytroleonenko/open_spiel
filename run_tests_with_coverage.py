#!/usr/bin/env python3
"""
Run MuZero JAX tests in parallel with custom worker management for memory efficiency and coverage.

This script automatically detects test files and runs them with parallel workers where each worker
handles one test file and exits (preventing memory accumulation). Coverage data is collected
without race conditions using sequential merging.

Key Features:
1. **Auto-detection**: Automatically discovers all individual test cases using pytest --collect-only
2. **One-test-case-per-worker**: Each worker process handles exactly one test case then exits
   (prevents memory leaks and Metal GPU resource contention, maximizes worker utilization)
3. **Process naming**: Workers are named 'muzero-wN-test_file::test_method' for easy monitoring
4. **Race-free coverage**: Each worker writes to separate coverage files, then merged sequentially
5. **Memory efficient**: Forces CPU execution and prevents pytest plugin autoloading
6. **Parallel execution**: Configurable number of workers (default: min(cpu_count//2, 4))

Usage:
    python run_tests_with_coverage.py                           # Use defaults
    python run_tests_with_coverage.py --num-workers 6           # Use 6 workers
    python run_tests_with_coverage.py --test-dir other/tests/   # Use different test directory

Requirements:
    - setproctitle package for better process naming (automatically installed)
    - coverage package for coverage collection
    - pytest for test execution
"""
import subprocess
import os
import sys
import glob
import time
import multiprocessing as mp
import threading
import tempfile
import shutil
import argparse
from pathlib import Path
from queue import Empty
from typing import List, Tuple, Dict, Optional

# Global timeout setting
GLOBAL_TIMEOUT = 600


def set_process_title(title: str):
    """Set the process title for visibility in process monitors."""
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except ImportError:
        # Fallback: try to modify argv[0] (less reliable but works on some systems)
        try:
            import ctypes
            import ctypes.util
            
            # Try to set process name on Linux/macOS
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            if hasattr(libc, 'prctl'):
                # Linux
                libc.prctl(15, title.encode(), 0, 0, 0)  # PR_SET_NAME
            elif hasattr(libc, 'pthread_setname_np'):
                # macOS
                libc.pthread_setname_np(title.encode())
        except Exception as e:
            pass  # If all fails, just continue without renaming


def discover_test_cases_from_file(test_file: str) -> List[str]:
    """Discover all test cases from a specific test file."""
    print(f"Discovering test cases from file: {test_file}")
    
    # Use absolute path for test_file
    abs_test_file = os.path.abspath(test_file)
    
    # Use pytest to collect all test cases from the specific file
    cmd = [
        "python", "-m", "pytest",
        abs_test_file,  # Use absolute path to the specific file
        "--collect-only", 
        "--quiet",
        "--tb=no",
        "-p", "no:xdist",
        "-p", "no:testmon",
        "--override-ini=addopts=",  # Clear any addopts from pytest.ini
        "--override-ini=markers="   # Clear markers to avoid issues
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"❌ Failed to collect test cases from {test_file}:")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            print(f"Command: {' '.join(cmd)}")
            return []
        
        # Parse the output to extract test case node IDs
        test_cases = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            
            # Look for lines that contain "::" and appear to be test node IDs
            if "::" in line and not line.startswith("=") and not line.startswith("<"):
                # Skip lines that are just headers or don't contain test methods
                if "test_" in line and not line.endswith("collected"):
                                         # Clean up any extra spaces or formatting
                     test_case = line.strip()
                     
                     # For single file discovery, check if the path is correct
                     file_part = test_case.split("::")[0]
                     rest_part = "::".join(test_case.split("::")[1:])
                     
                     if os.path.exists(file_part):
                         test_cases.append(test_case)
                     else:
                         # If the path is relative, use the original test_file path
                         corrected_case = f"{test_file}::{rest_part}"
                         if os.path.exists(test_file):
                             test_cases.append(corrected_case)
                         else:
                             print(f"⚠️ Skipping test case with non-existent file: {test_case}")
                             print(f"   Tried: {file_part} and {test_file}")
        
        test_cases.sort()  # Consistent ordering
        
        print(f"Discovered {len(test_cases)} test cases from {test_file}")
        for test_case in test_cases:
            print(f"  - {test_case}")
        
        return test_cases
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test collection from {test_file} timed out")
        return []
    except Exception as e:
        print(f"❌ Error collecting test cases from {test_file}: {e}")
        return []


def discover_test_cases(test_dir: str) -> List[str]:
    """Automatically discover all individual test cases in the given directory."""
    print(f"Searching for test cases in: {test_dir}")
    
    # Use absolute path for test_dir
    abs_test_dir = os.path.abspath(test_dir)
    
    # Use pytest to collect all test cases from the test directory
    cmd = [
        "python", "-m", "pytest",
        abs_test_dir,  # Use absolute path
        "--collect-only", 
        "--quiet",
        "--tb=no",
        "-p", "no:xdist",
        "-p", "no:testmon",
        "--override-ini=addopts=",  # Clear any addopts from pytest.ini
        "--override-ini=markers="   # Clear markers to avoid issues
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ Failed to collect test cases:")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            print(f"Command: {' '.join(cmd)}")
            return []
        
        # Parse the output to extract test case node IDs
        test_cases = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            
            # Look for lines that contain "::" and appear to be test node IDs
            if "::" in line and not line.startswith("=") and not line.startswith("<"):
                # Skip lines that are just headers or don't contain test methods
                if "test_" in line and not line.endswith("collected"):
                    # Clean up any extra spaces or formatting
                    test_case = line.strip()
                    
                    # Convert relative path from pytest collection to full path
                    # pytest returns: "tests/training/test_losses.py::test_method"
                    # we need: "open_spiel/python/algorithms/muzero_jax/tests/training/test_losses.py::test_method"
                    if "::" in test_case:
                        file_part = test_case.split("::")[0]
                        rest_part = "::".join(test_case.split("::")[1:])
                        
                        # The key insight: pytest returns paths relative to where we told it to search
                        # If we told it to search "open_spiel/python/algorithms/muzero_jax/tests"
                        # it returns "tests/training/test_X.py"
                        # If we told it to search "open_spiel/python/algorithms/muzero_jax/tests/training"  
                        # it returns "tests/training/test_X.py" (same thing!)
                        
                        # So we need to figure out the base directory that pytest is using
                        # and construct the proper full path
                        
                        # Remove the tests/ prefix if present, then build from project root
                        if file_part.startswith("tests/"):
                            # pytest is showing relative to the muzero_jax directory
                            # we want: open_spiel/python/algorithms/muzero_jax/tests/...
                            relative_from_muzero = file_part  # Keep full relative path
                            full_file_path = f"open_spiel/python/algorithms/muzero_jax/{relative_from_muzero}"
                        else:
                            # Shouldn't happen, but fallback to original logic
                            full_file_path = os.path.join(test_dir, file_part)
                        
                        test_case = f"{full_file_path}::{rest_part}"
                        
                        # Verify the file exists
                        final_file_path = test_case.split("::")[0]
                        if os.path.exists(final_file_path):
                            test_cases.append(test_case)
                        else:
                            print(f"⚠️ Skipping test case with non-existent file: {test_case}")
        
        test_cases.sort()  # Consistent ordering
        
        print(f"Discovered {len(test_cases)} individual test cases:")
        for i, test_case in enumerate(test_cases[:10]):  # Show first 10
            print(f"  - {test_case}")
        if len(test_cases) > 10:
            print(f"  ... and {len(test_cases) - 10} more")
        
        return test_cases
        
    except subprocess.TimeoutExpired:
        print("❌ Test collection timed out")
        return []
    except Exception as e:
        print(f"❌ Error collecting test cases: {e}")
        return []


def run_single_test_case_with_coverage(test_case: str, worker_id: int, temp_dir: str) -> Tuple[bool, str, float, str]:
    """
    Run a single test case with coverage in a dedicated worker process.
    
    Args:
        test_case: Test case node ID (e.g., "path/test_file.py::TestClass::test_method")
        worker_id: Unique worker identifier
        temp_dir: Temporary directory for coverage files
    
    Returns:
        Tuple of (success, test_case, duration, coverage_file_path)
    """
    # Extract test case name for process title
    # Format: "path/test_file.py::TestClass::test_method" -> "test_file::test_method"
    parts = test_case.split("::")
    if len(parts) >= 2:
        file_part = os.path.basename(parts[0]).replace('.py', '')
        test_part = parts[-1]  # Last part is the test method
        test_name = f"{file_part}::{test_part}"
    else:
        test_name = test_case.replace('/', '_').replace('.py', '')
    
    # Truncate if too long for process name
    if len(test_name) > 40:
        test_name = test_name[:37] + "..."
    
    worker_name = f"muzero-w{worker_id}-{test_name}"
    set_process_title(worker_name)
    
    start_time = time.time()
    
    # Create unique coverage file for this worker
    coverage_file = os.path.join(temp_dir, f".coverage.worker_{worker_id}")
    
    # Set environment for this worker
    env = os.environ.copy()
    #env["JAX_PLATFORM_NAME"] = "cpu"  # Force CPU to avoid Metal GPU conflicts
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_ENABLE_X64"] = "false"
    env["COVERAGE_FILE"] = coverage_file
    
    # Simple approach: run pytest from project root like normal
    # No complex path manipulation needed - pytest handles everything
    cmd = [
        "python", "-m", "pytest",
        test_case,  # Use the test case path as-is from pytest collection
        "--cov=open_spiel.python.algorithms.muzero_jax",
        "--cov-report=",  # Don't generate report yet
        "-p", "no:xdist",  # Disable xdist plugin to avoid conflicts
        "-p", "no:testmon",  # Disable testmon plugin to avoid conflicts
        "-v",
        "--override-ini=addopts=",
        "--tb=short"
    ]
    
    try:
        print(f"[Worker {worker_id}] Starting: {test_name}")
        # Run pytest from project root - let it handle imports naturally
        result = subprocess.run(
            cmd, 
            env=env, 
            timeout=GLOBAL_TIMEOUT, 
            capture_output=True, 
            text=True
            # No cwd parameter - runs from current directory (project root)
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[Worker {worker_id}] ✅ {test_name} completed in {duration:.1f}s")
            return True, test_case, duration, coverage_file
        else:
            print(f"[Worker {worker_id}] ❌ {test_name} failed in {duration:.1f}s")
            # Print more detailed error info
            if result.stderr:
                print(f"[Worker {worker_id}] STDERR: {result.stderr[-800:]}")  # More error details
            if result.stdout:
                print(f"[Worker {worker_id}] STDOUT: {result.stdout[-400:]}")  # Show some output
            return False, test_case, duration, coverage_file
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[Worker {worker_id}] ⏰ {test_name} TIMED OUT after {duration:.1f}s")
        print(f"[Worker {worker_id}] 🚨 Test case appears to be hanging - killing process")
        return False, test_case, duration, coverage_file
    except Exception as e:
        duration = time.time() - start_time
        print(f"[Worker {worker_id}] 💥 {test_name} crashed after {duration:.1f}s: {e}")
        return False, test_case, duration, coverage_file


def merge_coverage_files(coverage_files: List[str], final_coverage_file: str) -> bool:
    """Merge multiple coverage files into a single file to avoid race conditions."""
    try:
        # Filter only existing coverage files
        existing_files = [f for f in coverage_files if os.path.exists(f)]
        if not existing_files:
            print("⚠️ No coverage files found to merge")
            return False
            
        print(f"Merging {len(existing_files)} coverage files...")
        
        # Remove existing final coverage file if it exists
        if os.path.exists(final_coverage_file):
            os.remove(final_coverage_file)
        
        # Use coverage combine to merge all worker coverage files
        cmd = ["coverage", "combine"] + existing_files
        env = os.environ.copy()
        env["COVERAGE_FILE"] = final_coverage_file
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Coverage files merged successfully")
            return True
        else:
            print(f"❌ Coverage merge failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"💥 Coverage merge crashed: {e}")
        return False


def worker_process(test_queue: mp.Queue, result_queue: mp.Queue, worker_id: int, temp_dir: str):
    """Worker process that handles one test case and exits."""
    try:
        # Get one test case from the queue
        test_case = test_queue.get(timeout=1)
        
        # Run the test case
        success, test_case_result, duration, coverage_file = run_single_test_case_with_coverage(
            test_case, worker_id, temp_dir
        )
        
        # Report result
        result_queue.put((success, test_case_result, duration, coverage_file))
        
    except Empty:
        # No more tests available
        result_queue.put((None, None, None, None))
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        result_queue.put((False, None, 0, None))


def main(test_dir: str = "open_spiel/python/algorithms/muzero_jax/tests", num_workers: int = 4, specific_tests: List[str] = None, timeout: int = 600, report_slow: int = 100):
    """Main function to coordinate parallel test execution with coverage."""
    
    # Change to the project root
    project_root = "/Users/dleonenko/open_spiel"
    os.chdir(project_root)
    
    # Get test cases - either specific ones or auto-discover
    if specific_tests:
        print(f"Running specific tests: {len(specific_tests)} test cases/files")
        # If specific_tests contains files (no ::), discover test cases from those files
        # If specific_tests contains test cases (with ::), use them directly
        test_cases = []
        for test_spec in specific_tests:
            if "::" in test_spec:
                # It's a specific test case, use directly
                test_cases.append(test_spec)
            else:
                # It's a test file, discover test cases from it
                if os.path.exists(test_spec):
                    file_test_cases = discover_test_cases_from_file(test_spec)
                    test_cases.extend(file_test_cases)
                else:
                    print(f"⚠️ Test file not found: {test_spec}")
        
        if not test_cases:
            print("❌ No valid test cases found from specified tests")
            return False
            
        print(f"Resolved to {len(test_cases)} individual test cases")
    else:
        # Discover test cases automatically from directory
        if not os.path.exists(test_dir):
            print(f"❌ Test directory not found: {test_dir}")
            return False
        
        test_cases = discover_test_cases(test_dir)
        if not test_cases:
            print("❌ No test cases found")
            return False
    
    # Clear existing coverage data
    print("\nClearing existing coverage data...")
    subprocess.run(["coverage", "erase"], capture_output=True)
    
    # Set up parallel execution
    # Limit workers to reasonable defaults: half of CPU cores, max test cases, or user specified
    effective_workers = min(mp.cpu_count() // 1.1, len(test_cases), num_workers)
    print(f"\nRunning {len(test_cases)} test cases with {effective_workers} parallel workers")
    print(f"⏱️  Individual test timeout: {timeout}s")
    
    # Store timeout in global variable to be used by worker processes
    global GLOBAL_TIMEOUT
    GLOBAL_TIMEOUT = timeout
    
    # Create temporary directory for worker coverage files
    with tempfile.TemporaryDirectory(prefix="muzero_coverage_") as temp_dir:
        print(f"Temporary coverage directory: {temp_dir}")
        
        # Set up queues
        test_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Add all test cases to the queue
        for test_case in test_cases:
            test_queue.put(test_case)
        
        # Track results
        results = []
        coverage_files = []
        active_workers = []
        timed_out_tests = []
        slow_tests = []
        
        # Start workers as needed
        worker_id = 0
        tests_completed = 0
        
        while tests_completed < len(test_cases):
            # Start new workers if we have tests and available worker slots
            while len(active_workers) < effective_workers and not test_queue.empty():
                worker_id += 1
                worker = mp.Process(
                    target=worker_process,
                    args=(test_queue, result_queue, worker_id, temp_dir),
                    name=f"muzero-worker-{worker_id}"
                )
                worker.start()
                active_workers.append(worker)
                print(f"Started worker {worker_id}")
            
            # Check for completed workers
            try:
                success, test_case, duration, coverage_file = result_queue.get(timeout=1)
                
                if success is not None:  # Valid result (not timeout indicator)
                    tests_completed += 1
                    results.append((success, test_case, duration))
                    
                    # Track timed out tests separately (close to actual timeout)
                    if duration >= (GLOBAL_TIMEOUT - 1):  # Close to the actual timeout
                        timed_out_tests.append((test_case, duration))
                    
                    # Track slow tests separately
                    if duration >= report_slow:
                        slow_tests.append((test_case, duration))
                    
                    if coverage_file and os.path.exists(coverage_file):
                        coverage_files.append(coverage_file)
                    
                    print(f"Progress: {tests_completed}/{len(test_cases)} test cases completed")
                
                # Clean up finished workers
                active_workers = [w for w in active_workers if w.is_alive()]
                
            except Empty:
                # No results yet, clean up any dead workers
                active_workers = [w for w in active_workers if w.is_alive()]
                continue
        
        # Wait for all workers to finish
        for worker in active_workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        print(f"\nAll tests completed. Merging {len(coverage_files)} coverage files...")
        
        # Merge coverage files sequentially (no race conditions)
        final_coverage_file = os.path.join(project_root, ".coverage")
        if coverage_files:
            merge_coverage_files(coverage_files, final_coverage_file)
        
        # Generate coverage reports
        print("\nGenerating coverage reports...")
        subprocess.run([
            "coverage", "html", 
            "--directory=open_spiel/python/algorithms/muzero_jax/htmlcov"
        ], capture_output=True)
        
        subprocess.run([
            "coverage", "report", "--show-missing"
        ])
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if r[0]]
    failed_tests = [r for r in results if not r[0]]
    
    print(f"✅ Successful test cases: {len(successful_tests)}/{len(test_cases)}")
    print(f"❌ Failed test cases: {len(failed_tests)}")
    
    if timed_out_tests:
        print(f"⏰ Timed out test cases: {len(timed_out_tests)}")
        print("\nTimed out test cases:")
        for test_case, duration in timed_out_tests:
            test_name = test_case.split("::")[-1] if "::" in test_case else test_case
            print(f"  - {test_name} ({duration:.1f}s)")
    
    if slow_tests:
        print(f"\n🐌 Slow test cases (>= {report_slow}s): {len(slow_tests)}")
        print("\nSlow test cases:")
        # Sort by duration, slowest first
        slow_tests.sort(key=lambda x: x[1], reverse=True)
        for test_case, duration in slow_tests:
            # Extract file and function name in file::function format
            if "::" in test_case:
                parts = test_case.split("::")
                if len(parts) >= 2:
                    file_name = parts[0].split("/")[-1].replace(".py", "")  # Get just the filename without path and extension
                    function_name = parts[-1]  # Get the test function name
                    formatted_name = f"{file_name}::{function_name}"
                else:
                    formatted_name = test_case
            else:
                formatted_name = test_case
            print(f"  - {formatted_name} ({duration:.1f}s)")
    
    if failed_tests:
        print("\nFailed test cases:")
        for success, test_case, duration in failed_tests:
            # Show just the test case name part for readability
            test_name = test_case.split("::")[-1] if "::" in test_case else test_case
            print(f"  - {test_name} ({duration:.1f}s)")
    
    # Performance summary
    total_duration = sum(r[2] for r in results)
    avg_duration = total_duration / len(results) if results else 0
    print(f"\nPerformance: {total_duration:.1f}s total, {avg_duration:.1f}s average per test case")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuZero JAX tests in parallel with custom worker management for memory efficiency and coverage.")
    parser.add_argument("--test-dir", type=str, default="open_spiel/python/algorithms/muzero_jax/tests", help="Directory containing test files (used when --specific-tests not provided)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--specific-tests", nargs="+", help="Specific test files or test cases to run (e.g., 'test_file.py' or 'test_file.py::test_method')")
    parser.add_argument("--timeout", type=int, default=600, help="Individual test timeout in seconds (default: 600)")
    parser.add_argument("--report-slow", type=int, default=100, help="Report tests slower than this many seconds (default: 100)")
    args = parser.parse_args()
    
    success = main(test_dir=args.test_dir, num_workers=args.num_workers, specific_tests=args.specific_tests, timeout=args.timeout, report_slow=args.report_slow)
    sys.exit(0 if success else 1) 