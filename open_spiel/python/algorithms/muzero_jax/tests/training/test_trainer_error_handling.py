"""Tests for error handling and resource management in trainer.py."""

import unittest
import jax
from unittest.mock import Mock

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_test_utils import (
    make_cfg,
    make_model_from_muzero_config
)


class TestTrainerErrorHandling(unittest.TestCase):
    """Test error handling and resource management in trainer.py."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = make_cfg(
            vsup=0,  # scalar value support
            rsup=0,  # scalar reward support  
            steps=3,  # num_unroll_steps
            proj=False,  # no projection
            suffix="test",
            batch_size=2,
            td_lambda=0.95,
            auto_td_steps=1000,
            lstm_horizon_length=2,
            discount_factor=0.99
        )

    def test_learner_cleanup_exception_handling(self):
        """Test Learner.cleanup method exception handling (lines 1123-1124)."""
        # Create a mock model and learner
        key = jax.random.key(42)
        model = make_model_from_muzero_config(key, self.config)
        
        learner = Learner(
            model=model,
            optimizer_def=None,  # Let it create the default optimizer
            config=self.config,
            rng_key=key
        )
        
        # Create a mock checkpoint manager that will raise an exception on close()
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.close.side_effect = Exception("Mock cleanup error")
        learner.checkpoint_manager = mock_checkpoint_manager
        
        # Call cleanup - should handle exception gracefully (lines 1123-1124)
        learner.cleanup()
        
        # Verify the exception was caught (close was called)
        mock_checkpoint_manager.close.assert_called_once()
        # The checkpoint_manager should still be the mock since the assignment failed
        self.assertEqual(learner.checkpoint_manager, mock_checkpoint_manager)
        
        # Test successful cleanup path as well
        mock_checkpoint_manager_success = Mock()
        learner.checkpoint_manager = mock_checkpoint_manager_success
        
        # This should succeed and set checkpoint_manager to None
        learner.cleanup()
        
        mock_checkpoint_manager_success.close.assert_called_once()
        self.assertIsNone(learner.checkpoint_manager)

    def test_learner_cleanup_exception_during_close(self):
        """Test Learner.cleanup method gracefully handles exceptions during checkpoint manager close."""
        # Create a mock model and learner
        key = jax.random.key(42)
        model = make_model_from_muzero_config(key, self.config)
        
        learner = Learner(
            model=model,
            optimizer_def=None,  # Let it create the default optimizer
            config=self.config,
            rng_key=key
        )
        
        # Create a mock checkpoint manager that will raise an exception on close()
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.close.side_effect = Exception("Mock cleanup error")
        learner.checkpoint_manager = mock_checkpoint_manager
        
        # Call cleanup - should handle exception gracefully
        # The exception should be caught and ignored, but checkpoint_manager stays as is
        # because the assignment is inside the try block that fails
        learner.cleanup()
        
        # Verify the exception was caught (close was called)
        mock_checkpoint_manager.close.assert_called_once()
        # The checkpoint_manager should still be the mock since the assignment failed
        self.assertEqual(learner.checkpoint_manager, mock_checkpoint_manager)
        
        # Test successful cleanup path as well
        mock_checkpoint_manager_success = Mock()
        learner.checkpoint_manager = mock_checkpoint_manager_success
        
        # This should succeed and set checkpoint_manager to None
        learner.cleanup()
        
        mock_checkpoint_manager_success.close.assert_called_once()
        self.assertIsNone(learner.checkpoint_manager)


if __name__ == '__main__':
    unittest.main() 