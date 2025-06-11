"""
Checkpoint helper for test isolation.

This module provides utilities to ensure complete checkpoint isolation
between test cases when running in parallel or sequential execution.
"""

import tempfile
import shutil
import threading
import time
import os
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestCheckpointHelper:
    """
    Centralized checkpoint management for tests.
    
    Ensures each test gets a completely isolated checkpoint directory
    with proper cleanup and no shared state.
    """
    
    def __init__(self, test_name: Optional[str] = None):
        """Initialize checkpoint helper for a specific test.
        
        Args:
            test_name: Name of the test (auto-detected if None)
        """
        # Get test name from call stack if not provided
        if test_name is None:
            import inspect
            frame = inspect.currentframe()
            try:
                # Go up the stack to find the test function
                while frame:
                    frame = frame.f_back
                    if frame and frame.f_code.co_name.startswith('test_'):
                        test_name = frame.f_code.co_name
                        break
                if test_name is None:
                    test_name = "unknown_test"
            finally:
                del frame  # Prevent reference cycles
        
        self.test_name = test_name
        self.checkpoint_dir: Optional[str] = None
        self._created_dirs: list = []
        
        # Create unique checkpoint directory
        self._create_checkpoint_dir()
    
    def _create_checkpoint_dir(self):
        """Create a unique checkpoint directory for this test."""
        # Include thread ID, process ID, and nanosecond timestamp for uniqueness
        thread_id = threading.get_ident()
        process_id = os.getpid()
        timestamp_ns = time.time_ns()
        
        # Create a highly unique directory name
        unique_suffix = f"{self.test_name}_{process_id}_{thread_id}_{timestamp_ns}"
        
        # Create temporary directory with unique prefix
        self.checkpoint_dir = tempfile.mkdtemp(
            prefix=f"muzero_test_checkpoint_{unique_suffix}_",
            suffix="_isolated"
        )
        
        self._created_dirs.append(self.checkpoint_dir)
        logger.debug(f"Created isolated checkpoint dir: {self.checkpoint_dir}")
    
    def get_checkpoint_dir(self) -> str:
        """Get the checkpoint directory for this test.
        
        Returns:
            Path to the isolated checkpoint directory
        """
        if self.checkpoint_dir is None:
            self._create_checkpoint_dir()
        return self.checkpoint_dir
    
    def create_subdirectory(self, subdir_name: str) -> str:
        """Create a subdirectory within the checkpoint directory.
        
        Args:
            subdir_name: Name of the subdirectory to create
            
        Returns:
            Path to the created subdirectory
        """
        base_dir = self.get_checkpoint_dir()
        subdir_path = os.path.join(base_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path
    
    def cleanup(self):
        """Clean up all created checkpoint directories."""
        for dir_path in self._created_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
                    logger.debug(f"Cleaned up checkpoint dir: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {dir_path}: {e}")
        
        self._created_dirs.clear()
        self.checkpoint_dir = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def get_test_checkpoint_dir(test_name: Optional[str] = None) -> str:
    """
    Convenience function to get an isolated checkpoint directory for a test.
    
    Args:
        test_name: Name of the test (auto-detected if None)
        
    Returns:
        Path to isolated checkpoint directory
        
    Note:
        Caller is responsible for cleanup. Use TestCheckpointHelper context manager
        for automatic cleanup.
    """
    helper = TestCheckpointHelper(test_name)
    return helper.get_checkpoint_dir()


# Global registry to track helpers for pytest fixture cleanup
_active_helpers: Dict[str, TestCheckpointHelper] = {}
_helpers_lock = threading.Lock()


def register_test_checkpoint_helper(test_id: str, helper: TestCheckpointHelper):
    """Register a helper for pytest fixture cleanup."""
    with _helpers_lock:
        _active_helpers[test_id] = helper


def cleanup_test_checkpoint_helper(test_id: str):
    """Cleanup a registered helper."""
    with _helpers_lock:
        helper = _active_helpers.pop(test_id, None)
        if helper:
            helper.cleanup()


def cleanup_all_test_helpers():
    """Emergency cleanup of all active helpers."""
    with _helpers_lock:
        for helper in _active_helpers.values():
            try:
                helper.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup helper: {e}")
        _active_helpers.clear() 