"""Tests for adaptive hyperparameter functionality in GAE computation."""

import unittest
import jax
import jax.numpy as jnp
from flax import nnx

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig, compute_gae_value_targets
)
from open_spiel.python.algorithms.muzero_jax.utils.hyperparameter_adapter import (
    HyperparameterAdapter, HyperparameterAdapterConfig
)



class DummyNet(nnx.Module):
    """Minimal dummy network for testing GAE computation."""
    
    def __init__(self, hidden_dim: int = 64, num_actions: int = 4, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
    def initial_inference(self, observations, training=False):
        batch_size = observations.shape[0]
        hidden_state = jnp.zeros((batch_size, self.hidden_dim))
        reward = jnp.zeros((batch_size,))
        value = jnp.zeros((batch_size,))
        policy_logits = jnp.zeros((batch_size, self.num_actions))
        return hidden_state, reward, value, policy_logits
        
    def recurrent_inference(self, hidden_state, action, training=False):
        batch_size = hidden_state.shape[0]
        next_hidden = jnp.zeros_like(hidden_state)
        reward = jnp.zeros((batch_size,))
        value = jnp.zeros((batch_size,))
        policy_logits = jnp.zeros((batch_size, self.num_actions))
        return next_hidden, reward, value, policy_logits


class AdaptiveHyperparametersTest(unittest.TestCase):
    """Test adaptive hyperparameter functionality in GAE computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng_key = jax.random.PRNGKey(42)
        self.batch_size = 4
        self.num_unroll_steps = 5
        self.num_actions = 4
        self.hidden_dim = 64
        
        # Create dummy model
        self.model = DummyNet(
            hidden_dim=self.hidden_dim,
            num_actions=self.num_actions,
            rngs=nnx.Rngs(42)
        )
        
        # Create config with adaptive hyperparameters enabled
        self.config = MuZeroConfig(
            batch_size=self.batch_size,
            num_unroll_steps=self.num_unroll_steps,
            num_actions=self.num_actions,
            td_steps=10,
            td_lambda=0.95,
            use_adaptive_td_steps=True,
            auto_td_steps=1000,  # Small value for testing
            discount_factor=0.99,
            value_target_type="GAE"
        )
        
    def _create_test_data(self, extra_steps=10):
        """Create test data for GAE computation."""
        total_steps = self.num_unroll_steps + 1 + extra_steps
        
        observations = jnp.zeros((self.batch_size, total_steps, 8, 8))
        actions = jnp.zeros((self.batch_size, total_steps - 1), dtype=jnp.int32)
        rewards = jnp.ones((self.batch_size, total_steps)) * 0.1  # Small positive rewards
        dones = jnp.zeros((self.batch_size, total_steps), dtype=jnp.bool_)
        
        return observations, actions, rewards, dones
        
    def test_adaptive_td_steps_affects_gae_computation(self):
        """Test that different adaptive td_steps produce different GAE targets."""
        observations, actions, rewards, dones = self._create_test_data()
        
        # Create sample indices with different ages to trigger different td_steps
        # Older samples (higher indices) should get smaller td_steps
        sample_indices_young = jnp.array([100, 200, 300, 400])  # Young samples
        sample_indices_old = jnp.array([800, 900, 950, 990])    # Old samples
        
        # Compute GAE targets for young samples
        gae_targets_young = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=self.config,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices_young,
            collected_transitions=1000
        )
        
        # Compute GAE targets for old samples
        gae_targets_old = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=self.config,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices_old,
            collected_transitions=1000
        )
        
        # The GAE targets should be different because adaptive td_steps are different
        self.assertFalse(
            jnp.allclose(gae_targets_young, gae_targets_old, atol=1e-6),
            "GAE targets should be different for samples with different ages"
        )
        
    def test_adaptive_td_lambda_affects_gae_computation(self):
        """Test that different adaptive td_lambda values produce different GAE targets."""
        observations, actions, rewards, dones = self._create_test_data()
        
        # Create sample indices with different ages
        sample_indices_young = jnp.array([100, 200, 300, 400])  # Should get higher td_lambda
        sample_indices_old = jnp.array([800, 900, 950, 990])    # Should get lower td_lambda
        
        # Compute GAE targets for young samples
        gae_targets_young = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=self.config,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices_young,
            collected_transitions=1000
        )
        
        # Compute GAE targets for old samples
        gae_targets_old = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=self.config,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices_old,
            collected_transitions=1000
        )
        
        # The GAE targets should be different because adaptive td_lambda are different
        self.assertFalse(
            jnp.allclose(gae_targets_young, gae_targets_old, atol=1e-6),
            "GAE targets should be different for samples with different td_lambda values"
        )
        
    def test_non_adaptive_mode_produces_consistent_results(self):
        """Test that when adaptive mode is disabled, all samples get the same treatment."""
        observations, actions, rewards, dones = self._create_test_data()
        
        # Disable adaptive hyperparameters
        config_non_adaptive = MuZeroConfig(
            batch_size=self.batch_size,
            num_unroll_steps=self.num_unroll_steps,
            num_actions=self.num_actions,
            td_steps=10,
            td_lambda=0.95,
            use_adaptive_td_steps=False,  # Disabled
            discount_factor=0.99,
            value_target_type="GAE"
        )
        
        # Use the same sample indices for both calls
        sample_indices = jnp.array([100, 200, 300, 400])
        
        # Compute GAE targets twice with the same inputs
        gae_targets_1 = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=config_non_adaptive,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices,
            collected_transitions=1000
        )
        
        gae_targets_2 = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=config_non_adaptive,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices,
            collected_transitions=1000
        )
        
        # The GAE targets should be identical when adaptive hyperparameters are disabled
        self.assertTrue(
            jnp.allclose(gae_targets_1, gae_targets_2, atol=1e-8),
            "GAE targets should be identical when adaptive hyperparameters are disabled"
        )
        
    def test_gae_computation_with_varying_td_steps_per_sample(self):
        """Test that GAE computation correctly handles per-sample td_steps."""
        observations, actions, rewards, dones = self._create_test_data()
        
        # Create sample indices that will result in different td_steps for each sample
        sample_indices = jnp.array([100, 500, 800, 950])  # Different ages
        
        # Compute GAE targets
        gae_targets = compute_gae_value_targets(
            model=self.model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=self.config,
            training=False,
            rng_key=self.rng_key,
            sample_indices=sample_indices,
            collected_transitions=1000
        )
        
        # Check that GAE targets have the expected shape
        expected_shape = (self.batch_size, self.num_unroll_steps + 1)
        self.assertEqual(gae_targets.shape, expected_shape)
        
        # Check that GAE targets are finite (no NaN or inf values)
        self.assertTrue(jnp.all(jnp.isfinite(gae_targets)))
        
        # Check that different samples have different GAE targets (due to different td_steps)
        # At least some samples should be different
        all_same = True
        for i in range(1, self.batch_size):
            if not jnp.allclose(gae_targets[0], gae_targets[i], atol=1e-6):
                all_same = False
                break
        
        self.assertFalse(all_same, "At least some samples should have different GAE targets")


if __name__ == "__main__":
    unittest.main() 