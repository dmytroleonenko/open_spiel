import pytest
import jax
import jax.numpy as jnp
import tempfile
import os
from unittest.mock import Mock, patch

from trainer_test_utils import (
    key as common_key, 
    cfg_flat as common_cfg_flat,
    make_model, 
    make_cfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner


def test_learner_del_cleanup_error_handling(common_key, common_cfg_flat):
    """Test __del__ method error handling - lines 1123-1124."""
    
    cfg_no_ckpt = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_cleanup", 
        use_ema=False, checkpoint_dir=None
    )
    
    model = make_model(common_key, common_cfg_flat)
    learner = Learner(model, None, cfg_no_ckpt, common_key)
    
    # Test case 1: Checkpoint manager with _closed attribute
    failing_manager = Mock()
    failing_manager._closed = False
    failing_manager.close.side_effect = RuntimeError("Close error")
    learner.checkpoint_manager = failing_manager
    
    # This should handle the exception silently
    learner.__del__()  # Should not raise exception
    
    # Test case 2: Checkpoint manager without _closed attribute  
    failing_manager_no_closed = Mock()
    del failing_manager_no_closed._closed  # Remove _closed attribute
    failing_manager_no_closed.close.side_effect = RuntimeError("Close error")
    learner.checkpoint_manager = failing_manager_no_closed
    
    # This should handle the exception silently (fallback path)
    learner.__del__()  # Should not raise exception
    
    # Test case 3: Import error during logging
    with patch('builtins.__import__', side_effect=ImportError("No logging")):
        failing_manager_import = Mock()
        failing_manager_import._closed = False
        failing_manager_import.close.side_effect = RuntimeError("Close error")
        learner.checkpoint_manager = failing_manager_import
        
        # This should handle the exception silently even with import error
        learner.__del__()  # Should not raise exception


def test_checkpoint_manager_attribute_edge_cases(common_key, common_cfg_flat):
    """Test edge cases for checkpoint manager attribute checking."""
    
    cfg_no_ckpt = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_edge", 
        use_ema=False, checkpoint_dir=None
    )
    
    model = make_model(common_key, common_cfg_flat)
    learner = Learner(model, None, cfg_no_ckpt, common_key)
    
    # Test case 1: No checkpoint_manager attribute at all
    if hasattr(learner, 'checkpoint_manager'):
        delattr(learner, 'checkpoint_manager')
    
    # This should handle missing attribute gracefully
    learner.__del__()  # Should not raise exception
    
    # Test case 2: checkpoint_manager is None
    learner.checkpoint_manager = None
    learner.__del__()  # Should not raise exception
    
    # Test case 3: checkpoint_manager has no close method
    learner.checkpoint_manager = Mock(spec=[])  # No close method
    learner.__del__()  # Should not raise exception


def test_checkpoint_manager_closed_attribute_variations(common_key, common_cfg_flat):
    """Test different _closed attribute scenarios."""
    
    cfg_no_ckpt = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_closed", 
        use_ema=False, checkpoint_dir=None
    )
    
    model = make_model(common_key, common_cfg_flat)
    learner = Learner(model, None, cfg_no_ckpt, common_key)
    
    # Test case 1: _closed is True (already closed)
    closed_manager = Mock()
    closed_manager._closed = True
    learner.checkpoint_manager = closed_manager
    
    learner.__del__()  # Should not call close since already closed
    closed_manager.close.assert_not_called()
    
    # Test case 2: _closed exists but hasattr check fails due to property error
    class ProblematicManager:
        @property  
        def _closed(self):
            raise AttributeError("Cannot access _closed")
        
        def close(self):
            pass
    
    problematic_manager = ProblematicManager()
    learner.checkpoint_manager = problematic_manager
    
    # This should fall back to the no-_closed path
    learner.__del__()  # Should not raise exception 