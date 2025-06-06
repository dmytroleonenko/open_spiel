"""
Tests for checkpointing utilities.
"""

import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock

from open_spiel.python.algorithms.muzero_jax.utils.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    create_checkpoint_manager
)


class TestCheckpointing:
    """Test suite for checkpointing utilities."""

    def test_save_and_load_checkpoint(self):
        """Test basic checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock model and optimizer states
            model_state = Mock()
            optimizer_state = Mock()
            metadata = {'epoch': 10, 'loss': 0.5}
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(
                checkpoint_dir=temp_dir,
                step=100,
                model_state=model_state,
                optimizer_state=optimizer_state,
                metadata=metadata
            )
            
            # Verify checkpoint file was created
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            loaded_data = load_checkpoint(checkpoint_path)
            
            # Verify loaded data structure
            assert 'step' in loaded_data
            assert 'model_state' in loaded_data
            assert 'optimizer_state' in loaded_data
            assert 'metadata' in loaded_data

    def test_load_nonexistent_checkpoint(self):
        """Test loading a checkpoint that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/checkpoint")

    def test_get_latest_checkpoint(self):
        """Test getting the latest checkpoint from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple checkpoints
            for step in [10, 50, 30]:
                save_checkpoint(
                    checkpoint_dir=temp_dir,
                    step=step,
                    model_state=Mock(),
                    optimizer_state=Mock()
                )
            
            # Get latest checkpoint
            latest = get_latest_checkpoint(temp_dir)
            
            # Should return the checkpoint with highest step number (50)
            assert latest is not None
            assert "checkpoint_50" in latest

    def test_get_latest_checkpoint_empty_dir(self):
        """Test getting latest checkpoint from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            latest = get_latest_checkpoint(temp_dir)
            assert latest is None

    def test_get_latest_checkpoint_nonexistent_dir(self):
        """Test getting latest checkpoint from nonexistent directory."""
        latest = get_latest_checkpoint("/nonexistent/directory")
        assert latest is None

    def test_create_checkpoint_manager(self):
        """Test creating a checkpoint manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_checkpoint_manager(temp_dir, max_to_keep=3)
            
            # Should return a manager object (mock or real)
            assert manager is not None
            assert hasattr(manager, 'save')
            assert hasattr(manager, 'restore')

    def test_checkpoint_manager_save_and_restore(self):
        """Test checkpoint manager save and restore functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_checkpoint_manager(temp_dir, max_to_keep=3)
            
            # Save a checkpoint using the manager
            items = {
                'model_state': Mock(),
                'optimizer_state': Mock(),
                'metadata': {'test': True}
            }
            
            checkpoint_path = manager.save(step=42, items=items)
            assert os.path.exists(checkpoint_path)
            
            # Restore the checkpoint
            restored_data = manager.restore(step=42)
            assert restored_data is not None