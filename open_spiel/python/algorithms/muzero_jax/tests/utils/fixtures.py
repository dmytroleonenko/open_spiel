"""
Common pytest fixtures for MuZero JAX tests.

This module provides reusable fixtures that ensure proper test isolation,
especially for checkpoint management and resource cleanup.
"""

import pytest
import tempfile
import shutil
import threading
import time
from typing import Generator
from .checkpoint_helper import TestCheckpointHelper, register_test_checkpoint_helper, cleanup_test_checkpoint_helper


@pytest.fixture
def isolated_checkpoint_helper(request) -> Generator[TestCheckpointHelper, None, None]:
    """
    Provides an isolated checkpoint helper for each test.
    
    This fixture ensures complete checkpoint isolation between tests,
    even when running in parallel. Each test gets a unique checkpoint
    directory that is automatically cleaned up after the test.
    
    Usage:
        def test_my_function(isolated_checkpoint_helper):
            checkpoint_dir = isolated_checkpoint_helper.get_checkpoint_dir()
            # Use checkpoint_dir for your test...
    """
    # Create unique test ID for this test instance
    test_id = f"{request.node.nodeid}_{threading.get_ident()}_{time.time_ns()}"
    
    # Create helper with test name from pytest
    test_name = request.node.name
    helper = TestCheckpointHelper(test_name)
    
    # Register for emergency cleanup
    register_test_checkpoint_helper(test_id, helper)
    
    try:
        yield helper
    finally:
        # Clean up this specific helper
        cleanup_test_checkpoint_helper(test_id)


@pytest.fixture
def isolated_checkpoint_dir(request) -> Generator[str, None, None]:
    """
    Convenience fixture that provides just the checkpoint directory path.
    
    This creates an isolated checkpoint directory for the test and automatically
    cleans it up afterward.
    
    Usage:
        def test_my_function(isolated_checkpoint_dir):
            # isolated_checkpoint_dir is a string path
            config.checkpoint_dir = isolated_checkpoint_dir
    """
    # Create unique test ID for this test instance
    test_id = f"{request.node.nodeid}_{threading.get_ident()}_{time.time_ns()}"
    
    # Create helper with test name from pytest
    test_name = request.node.name
    helper = TestCheckpointHelper(test_name)
    
    # Register for emergency cleanup
    register_test_checkpoint_helper(test_id, helper)
    
    try:
        yield helper.get_checkpoint_dir()
    finally:
        # Clean up this specific helper
        cleanup_test_checkpoint_helper(test_id)


@pytest.fixture
def temp_checkpoint_dir(request) -> Generator[str, None, None]:
    """
    Legacy fixture for backward compatibility.
    
    This provides the same interface as the old temp_checkpoint_dir fixture
    but uses the new isolation system underneath.
    """
    with TestCheckpointHelper(request.node.name) as helper:
        yield helper.get_checkpoint_dir() 