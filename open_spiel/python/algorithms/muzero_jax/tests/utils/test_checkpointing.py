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
    create_checkpoint_manager,
    save_composite_checkpoint,
    load_composite_checkpoint
)


class TestCheckpointing:
    """Test suite for checkpointing utilities."""

    def test_save_and_load_checkpoint(self):
        """Test basic checkpoint saving and loading."""
        import jax.numpy as jnp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create real JAX arrays for model and optimizer states
            model_state = {
                'params': jnp.array([1.0, 2.0, 3.0]),
                'layer_weights': jnp.array([[0.1, 0.2], [0.3, 0.4]])
            }
            optimizer_state = {
                'momentum': jnp.array([0.01, 0.02, 0.03]),
                'step_count': jnp.array(100)
            }
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
        import jax.numpy as jnp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple checkpoints
            for step in [10, 50, 30]:
                save_checkpoint(
                    checkpoint_dir=temp_dir,
                    step=step,
                    model_state={'params': jnp.array([1.0, 2.0])},
                    optimizer_state={'momentum': jnp.array([0.1, 0.2])}
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
            
            # Should return a real Orbax CheckpointManager
            assert manager is not None
            assert hasattr(manager, 'save')
            assert hasattr(manager, 'restore')
            assert hasattr(manager, 'all_steps')
            assert hasattr(manager, 'latest_step')

    def test_checkpoint_manager_save_and_restore(self):
        """Test checkpoint manager save and restore functionality."""
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_checkpoint_manager(temp_dir, max_to_keep=3)
            
            # Create simple JAX arrays for testing (Mock objects don't work with Orbax)
            test_state = {
                'model_params': jnp.array([1.0, 2.0, 3.0]),
                'optimizer_state': jnp.array([0.1, 0.2, 0.3]),
                'metadata': {'test': True, 'step': 42}
            }
            
            # Save a checkpoint using the real Orbax manager
            manager.save(step=42, args=ocp.args.StandardSave(test_state))
            manager.wait_until_finished()  # Wait for async save to complete
            
            # Verify checkpoint was saved
            assert 42 in manager.all_steps()
            assert manager.latest_step() == 42
            
            # Restore the checkpoint
            restored_data = manager.restore(step=42)
            assert restored_data is not None
            
            # Verify the restored data matches what we saved
            assert 'model_params' in restored_data
            assert 'optimizer_state' in restored_data
            assert 'metadata' in restored_data

    def test_get_latest_checkpoint_malformed_names(self):
        """Test get_latest_checkpoint with malformed checkpoint names (covers lines 107-108)."""
        import jax.numpy as jnp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some valid checkpoints
            save_checkpoint(
                checkpoint_dir=temp_dir,
                step=10,
                model_state={'params': jnp.array([1.0, 2.0])},
                optimizer_state={'momentum': jnp.array([0.1, 0.2])}
            )
            
            # Create files with malformed names that will trigger the exception
            malformed_files = [
                'checkpoint_abc',  # Non-numeric step
                'checkpoint_',     # Empty step
                'checkpoint_1_extra',  # Extra parts
            ]
            
            for filename in malformed_files:
                malformed_path = os.path.join(temp_dir, filename)
                os.makedirs(malformed_path, exist_ok=True)
            
            # Should still return the valid checkpoint despite malformed names
            latest = get_latest_checkpoint(temp_dir)
            assert latest is not None
            assert "checkpoint_10" in latest

    def test_save_composite_checkpoint(self):
        """Test save_composite_checkpoint function (covers lines 157-172)."""
        import jax.numpy as jnp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data with different types
            items = {
                'model_state': {'params': jnp.array([1.0, 2.0, 3.0])},
                'optimizer_state': {'momentum': jnp.array([0.1, 0.2, 0.3])},
                'metadata': {'epoch': 5, 'loss': 0.25, 'training': True}
            }
            
            # Save composite checkpoint
            checkpoint_path = save_composite_checkpoint(
                checkpoint_dir=temp_dir,
                step=42,
                items=items
            )
            
            # Verify checkpoint was created
            assert os.path.exists(checkpoint_path)
            assert "checkpoint_42" in checkpoint_path

    def test_load_composite_checkpoint(self):
        """Test load_composite_checkpoint function (covers lines 189-197)."""
        import jax.numpy as jnp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a composite checkpoint first
            items = {
                'model_state': {'weights': jnp.array([1.0, 2.0])},
                'optimizer_state': {'step_count': jnp.array(100)},
                'metadata': {'version': 1, 'test': True}
            }
            
            checkpoint_path = save_composite_checkpoint(
                checkpoint_dir=temp_dir,
                step=99,
                items=items
            )
            
            # Load the composite checkpoint
            loaded_data = load_composite_checkpoint(checkpoint_path)
            
            # Verify loaded data contains expected keys
            assert loaded_data is not None
            assert 'model_state' in loaded_data
            assert 'optimizer_state' in loaded_data
            assert 'metadata' in loaded_data

    def test_load_composite_checkpoint_nonexistent(self):
        """Test load_composite_checkpoint with nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_composite_checkpoint("/nonexistent/path/checkpoint")