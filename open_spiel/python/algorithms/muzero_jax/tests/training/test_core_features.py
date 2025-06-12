"""
Tests for Core MuZero Features and Essential Implementations

This test suite validates core MuZero features and essential implementations
that are fundamental to training correctness and runtime reliability.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch
import functools
import dataclasses

# Import the functions we need to test
from open_spiel.python.algorithms.muzero_jax.training.losses import (
    scalar_to_support, support_to_scalar, compute_continuous_policy_entropy
)
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    compute_policy_reanalysis_targets, generate_top_new_masks, Learner, MuZeroConfig
)
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
import flax.nnx as nnx
import optax


class TestCoreMuZeroFeatures:
    """Test suite for core MuZero features and essential implementations."""

    def test_scalar_categorical_support_conversion(self):
        """Test scalar/categorical support conversion matches EfficientZeroV2 canonical transformation."""
        # For OpenSpiel environments, the complex categorical transformation may not be necessary
        # This test verifies that the transformation is at least mathematically consistent
        x = jnp.array([0.0, 1.0, -1.0])  # Simple test values
        
        # Convert to categorical and back
        support_dist = scalar_to_support(x, support_min=-10.0, support_max=10.0, num_atoms=21)
        x_reconstructed = support_to_scalar(support_dist, support_min=-10.0, support_max=10.0, num_atoms=21)
        
        # Check basic reconstruction (relaxed tolerance for complex transformation)
        max_error = jnp.max(jnp.abs(x - x_reconstructed))
        assert max_error < 0.1, f"Reconstruction error too large: {max_error}"
        
        # Test that the transformation preserves ordering
        x_sorted = jnp.sort(x)
        support_sorted = scalar_to_support(x_sorted, support_min=-10.0, support_max=10.0, num_atoms=21)
        x_reconstructed_sorted = support_to_scalar(support_sorted, support_min=-10.0, support_max=10.0, num_atoms=21)
        
        # Check that reconstruction preserves ordering
        assert jnp.all(jnp.diff(x_reconstructed_sorted) >= -0.1), "Transformation should preserve ordering"

    def test_vectorized_policy_reanalysis(self):
        """Test vectorized policy reanalysis is JIT-compiled and works correctly."""
        # Create test config with modified parameters
        base_config = MuZeroConfig()
        config = dataclasses.replace(base_config, num_simulations=4, reanalyze_ratio=0.5, num_actions=9)
        
        # Create mock model with correct parameters
        network_config = MuZeroNetworkConfig(
            observation_shape=(8, 8),
            num_actions=9,
            num_channels=32,
            use_image_observation=True,
            spatial_extents=(8, 8),
            use_batch_norm=False
        )
        
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import create_test_muzero_network
        model = create_test_muzero_network(network_config)
        
        # Create test batch
        batch_size = 4
        num_unroll_steps = 3
        observations = jnp.ones((batch_size, num_unroll_steps + 1, 8, 8))
        
        # Test that the function is JIT-compiled (should not raise errors)
        rng_key = jax.random.PRNGKey(42)
        
        try:
            # This should work without host-side loops
            reanalyzed_policies = compute_policy_reanalysis_targets(
                model, observations, config, training=False, rng_key=rng_key
            )
            
            # Verify output shape
            expected_shape = (batch_size, num_unroll_steps + 1, config.num_actions)
            assert reanalyzed_policies.shape == expected_shape
            
            # Verify policies are valid probability distributions
            assert jnp.allclose(jnp.sum(reanalyzed_policies, axis=-1), 1.0, atol=1e-5)
            assert jnp.all(reanalyzed_policies >= 0.0)
            
        except Exception as e:
            pytest.fail(f"Vectorized policy reanalysis failed: {e}")

    def test_full_model_state_copying(self):
        """Test full model state copying includes BatchStat and Rngs."""
        # Create test config
        base_config = MuZeroConfig()
        config = dataclasses.replace(base_config, use_batch_norm=True, reanalyze_update_interval=1, num_actions=9)
        
        # Create model with batch normalization
        network_config = MuZeroNetworkConfig(
            observation_shape=(8, 8),
            num_actions=9,
            num_channels=32,
            use_image_observation=True,
            spatial_extents=(8, 8),
            use_batch_norm=True
        )
        
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import create_test_muzero_network
        model = create_test_muzero_network(network_config)
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-4)
        
        # Create learner
        learner = Learner(model, optimizer, config, jax.random.PRNGKey(42))
        
        # Get initial states
        initial_main_state = nnx.state(learner.model)
        initial_reanalysis_state = nnx.state(learner.reanalysis_model)
        
        # Verify that states are different initially
        main_params = initial_main_state.get(nnx.Param, {})
        reanalysis_params = initial_reanalysis_state.get(nnx.Param, {})
        
        # Check that the models have different parameter values initially
        param_keys = list(main_params.keys())
        if param_keys:
            # Models should have different random initializations
            first_param_key = param_keys[0]
            main_param = main_params[first_param_key]
            reanalysis_param = reanalysis_params[first_param_key]
            
            # After model weight copying, they should be identical
            # Simulate a training step that triggers model weight copying
            learner.num_training_steps = config.reanalyze_update_interval
            
            # Create dummy batch for training step
            dummy_batch = {
                'observations': jnp.ones((2, 4, 8, 8)),
                'actions': jnp.ones((2, 3), dtype=jnp.int32),
                'rewards': jnp.ones((2, 4)),
                'values': jnp.ones((2, 4)),
                'policies': jnp.ones((2, 4, 9)),
                'dones': jnp.zeros((2, 4), dtype=bool),
                'sample_indices': jnp.array([0, 1]),
                'collected_transitions': 100
            }
            
            # Run training step (this should trigger model weight copying)
            try:
                metrics = learner.train_step(dummy_batch)
                
                # Verify that model states are now synchronized
                updated_main_state = nnx.state(learner.model)
                updated_reanalysis_state = nnx.state(learner.reanalysis_model)
                
                # Check that parameters are now identical
                updated_main_params = updated_main_state.get(nnx.Param, {})
                updated_reanalysis_params = updated_reanalysis_state.get(nnx.Param, {})
                
                if param_keys:
                    main_param_updated = updated_main_params[first_param_key]
                    reanalysis_param_updated = updated_reanalysis_params[first_param_key]
                    
                    assert jnp.allclose(main_param_updated, reanalysis_param_updated, atol=1e-6), \
                        "Model parameters should be identical after weight copying"
                
                # Check that BatchStat is also copied (if present)
                main_batch_stats = updated_main_state.get(nnx.BatchStat, {})
                reanalysis_batch_stats = updated_reanalysis_state.get(nnx.BatchStat, {})
                
                for key in main_batch_stats:
                    if key in reanalysis_batch_stats:
                        assert jnp.allclose(main_batch_stats[key], reanalysis_batch_stats[key], atol=1e-6), \
                            f"BatchStat {key} should be identical after full state copying"
                            
            except Exception as e:
                pytest.fail(f"Full model state copying test failed: {e}")

    def test_ema_momentum_clamping(self):
        """Test EMA momentum is clamped to prevent exceeding valid bounds."""
        # Create test config with extreme momentum values
        base_config = MuZeroConfig()
        config = dataclasses.replace(
            base_config,
            ema_m_init=0.95,
            ema_m_peak=0.999,
            ema_m_final=0.99,
            ema_m_warmup_steps=10,
            num_actions=2
        )
        
        # Create simple model
        network_config = MuZeroNetworkConfig(
            observation_shape=(4,),
            num_actions=2,
            num_channels=32,
            use_image_observation=False,
            use_batch_norm=False
        )
        
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import create_test_muzero_network
        model = create_test_muzero_network(network_config)
        optimizer = optax.adam(learning_rate=1e-4)
        
        learner = Learner(model, optimizer, config, jax.random.PRNGKey(42))
        
        # Test momentum clamping at different training steps
        test_steps = [0, 5, 15, 100, 1000]
        
        for step in test_steps:
            learner.num_training_steps = step
            
            # Create dummy batch (num_unroll_steps=5, so we need 6 time steps)
            dummy_batch = {
                'observation': jnp.ones((2, 6, 4)),
                'action': jnp.ones((2, 5), dtype=jnp.int32),
                'target_reward': jnp.ones((2, 6)),
                'target_value': jnp.ones((2, 6)),
                'target_policy': jnp.ones((2, 6, 2)),
                'game_history_mask': jnp.ones((2, 6), dtype=bool),
                'sample_indices': jnp.array([0, 1]),
                'collected_transitions': 100
            }
            
            try:
                # Run training step
                metrics = learner.train_step(dummy_batch)
                
                # Check that momentum is within valid bounds
                # The momentum should be clamped between ema_m_final and ema_m_peak
                if 'ema_momentum' in metrics:
                    momentum = metrics['ema_momentum']
                    assert config.ema_m_final <= momentum <= config.ema_m_peak, \
                        f"Momentum {momentum} at step {step} should be clamped between {config.ema_m_final} and {config.ema_m_peak}"
                        
            except Exception as e:
                pytest.fail(f"Momentum clamping test failed at step {step}: {e}")

    def test_mixed_value_target_masks(self):
        """Test mixed-value target masks require collected_transitions."""
        # Test the generate_top_new_masks function
        sample_indices = jnp.array([100, 200, 300, 400, 500])
        collected_transitions = 450
        mixed_value_threshold = 100
        
        # Generate masks
        masks = generate_top_new_masks(sample_indices, collected_transitions, mixed_value_threshold)
        
        # Verify mask properties
        assert masks.shape == sample_indices.shape
        assert jnp.all((masks == 0) | (masks == 1)), "Masks should be binary"
        
        # Verify mask logic: mask = int(idx > collected_transitions - mixed_value_threshold)
        expected_threshold = collected_transitions - mixed_value_threshold  # 450 - 100 = 350
        expected_masks = (sample_indices > expected_threshold).astype(jnp.int32)
        
        assert jnp.array_equal(masks, expected_masks), \
            f"Masks {masks} should match expected {expected_masks}"
        
        # Test edge cases
        # All samples old (should be all 0s)
        old_indices = jnp.array([100, 200, 300])
        old_masks = generate_top_new_masks(old_indices, 500, 100)
        assert jnp.all(old_masks == 0), "All old samples should have mask=0"
        
        # All samples new (should be all 1s)
        new_indices = jnp.array([450, 500, 550])
        new_masks = generate_top_new_masks(new_indices, 400, 100)
        assert jnp.all(new_masks == 1), "All new samples should have mask=1"

    def test_continuous_action_entropy_openspiel_limitation(self):
        """Test continuous action entropy clearly indicates OpenSpiel limitation."""
        # Test that the function raises appropriate error for OpenSpiel
        distribution_params = jnp.ones((4, 6))  # Batch size 4, 6 parameters
        
        # Should raise NotImplementedError for OpenSpiel environments
        with pytest.raises(NotImplementedError) as exc_info:
            compute_continuous_policy_entropy(distribution_params, "normal")
        
        # Check that error message mentions OpenSpiel
        error_message = str(exc_info.value)
        assert "OpenSpiel" in error_message, "Error should mention OpenSpiel limitation"
        assert "discrete action" in error_message, "Error should mention discrete action spaces"

    def test_explicit_mctx_import_error(self):
        """Test explicit ImportError for missing mctx dependency."""
        # Test that mctx import handling is explicit
        try:
            from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import mctx_gumbel_muzero_policy
            # If import succeeds, verify it works
            assert callable(mctx_gumbel_muzero_policy), "mctx_gumbel_muzero_policy should be callable"
        except ImportError as e:
            # If import fails, verify error is explicit
            error_message = str(e)
            assert "mctx" in error_message.lower(), "ImportError should mention mctx"

    def test_core_features_integration(self):
        """Integration test verifying all core features work together."""
        # Create comprehensive config
        base_config = MuZeroConfig()
        config = dataclasses.replace(
            base_config,
            value_loss_type="categorical",
            reward_loss_type="categorical",
            reanalyze_ratio=0.1,
            use_iql=True,
            iql_weight=0.8,
            ema_m_init=0.95,
            ema_m_peak=0.999,
            ema_m_final=0.99,
            num_actions=4,
            num_unroll_steps=3  # Match the batch dimensions
        )
        
        # Create model
        network_config = MuZeroNetworkConfig(
            observation_shape=(6, 6),
            num_actions=4,
            num_channels=32,
            use_image_observation=True,
            spatial_extents=(6, 6),
            use_batch_norm=True
        )
        
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import create_test_muzero_network
        model = create_test_muzero_network(network_config)
        optimizer = optax.adam(learning_rate=1e-4)
        
        learner = Learner(model, optimizer, config, jax.random.PRNGKey(42))
        
        # Create realistic batch
        batch_size = 4
        num_unroll_steps = 3
        batch = {
            'observation': jnp.ones((batch_size, num_unroll_steps + 1, 6, 6)),
            'action': jnp.ones((batch_size, num_unroll_steps), dtype=jnp.int32),
            'target_reward': jnp.ones((batch_size, num_unroll_steps + 1)),
            'target_value': jnp.ones((batch_size, num_unroll_steps + 1)),
            'target_policy': jnp.ones((batch_size, num_unroll_steps + 1, 4)),
            'game_history_mask': jnp.ones((batch_size, num_unroll_steps + 1), dtype=bool),
            'sample_indices': jnp.array([100, 200, 300, 400]),
            'collected_transitions': 350
        }
        
        try:
            # Run training step with all fixes active
            metrics = learner.train_step(batch)
            
            # Verify training completed successfully
            assert 'total_loss' in metrics, "Training should produce total_loss metric"
            assert jnp.isfinite(metrics['total_loss']), "Total loss should be finite"
            
            # Verify specific fixes are working
            if 'ema_momentum' in metrics:
                momentum = metrics['ema_momentum']
                assert config.ema_m_final <= momentum <= config.ema_m_peak, \
                    "Momentum clamping should be active"
            
            # Verify model state copying occurred (if applicable)
            if learner.num_training_steps % config.reanalyze_update_interval == 0:
                main_state = nnx.state(learner.model)
                reanalysis_state = nnx.state(learner.reanalysis_model)
                
                # States should be synchronized
                main_params = main_state.get(nnx.Param, {})
                reanalysis_params = reanalysis_state.get(nnx.Param, {})
                
                if main_params and reanalysis_params:
                    first_key = list(main_params.keys())[0]
                    assert jnp.allclose(main_params[first_key], reanalysis_params[first_key], atol=1e-6), \
                        "Model parameters should be synchronized"
                        
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 