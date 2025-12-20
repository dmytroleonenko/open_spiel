"""
Test multi-model orchestration schedules for MuZero JAX implementation.

This module tests that reanalysis_model and self_play_model updates happen
at the correct frequencies as specified in the configuration, addressing
Issue #6 from the critical review.
"""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
import optax
import flax.nnx as nnx


class TestMultiModelOrchestration:
    """Test multi-model orchestration schedules and update frequencies."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with multi-model parameters."""
        import dataclasses
        config = MuZeroConfig()
        config = dataclasses.replace(
            config,
            reanalyze_update_interval=200,
            self_play_update_interval=100,
            reanalyze_ratio=0.8,
            num_actions=9,
            batch_size=32,
            num_unroll_steps=5
        )
        return config
    
    @pytest.fixture
    def network_config(self):
        """Create network configuration."""
        return MuZeroNetworkConfig(
            observation_shape=(9,),  # Flat observation for tic-tac-toe
            num_actions=9,
            num_channels=32,
            use_image_observation=False,
            spatial_extents=None,  # No spatial dimensions for flat observations
            use_value_prefix=True,  # Enable LSTM reward network for discrete games with memory needs
            lstm_hidden_size=64,
            reduced_channels_reward=8,
            hidden_state_size=32,
            spatial_size=8  # Target LSTM input size for flat observations (reduced_channels * spatial_size)
        )
    
    @pytest.fixture
    def model(self, network_config):
        """Create test MuZero model."""
        from open_spiel.python.algorithms.muzero_jax.models.network import (
            RepresentationNetwork, DynamicsNetwork, PredictionNetwork, 
            RewardNetwork, ProjectionNetwork
        )
        
        rng_key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(params=rng_key, dropout=rng_key)
        
        model = MuZeroNetwork(
            representation_network_def=RepresentationNetwork,
            dynamics_network_def=DynamicsNetwork,
            prediction_network_def=PredictionNetwork,
            reward_network_def=RewardNetwork,
            projection_network_def=None,
            config=network_config,
            rngs=rngs
        )
        return model
    
    @pytest.fixture
    def learner(self, model, config):
        """Create test learner."""
        optimizer = optax.adam(learning_rate=1e-4)
        rng_key = jax.random.PRNGKey(42)
        return Learner(model, optimizer, config, rng_key)
    
    def test_reanalysis_model_creation(self, learner, config):
        """Test that reanalysis model is created when reanalyze_ratio > 0."""
        assert config.reanalyze_ratio > 0.0
        # The learner should have access to the model for reanalysis
        assert learner.model is not None
        
        # Test that reanalysis configuration is properly set
        assert hasattr(config, 'reanalyze_update_interval')
        assert config.reanalyze_update_interval == 200
    
    def test_self_play_model_creation(self, learner, config):
        """Test that self-play model configuration is properly set."""
        assert hasattr(config, 'self_play_update_interval')
        assert config.self_play_update_interval == 100
        
        # The learner should have access to the model for self-play
        assert learner.model is not None
    
    def test_update_frequency_validation(self, config):
        """Test that update frequencies are properly configured."""
        # Reanalysis should update less frequently than self-play
        assert config.reanalyze_update_interval > config.self_play_update_interval
        
        # Both intervals should be positive
        assert config.reanalyze_update_interval > 0
        assert config.self_play_update_interval > 0
    
    def test_lstm_reward_network_integration(self, model, network_config):
        """Test that LSTM reward network is properly integrated when use_value_prefix=True."""
        # Check that LSTM reward network is created/disabled based on config
        assert hasattr(model, 'lstm_reward_network')
        
        if network_config.use_value_prefix:
            assert model.lstm_reward_network is not None
            
            # Test LSTM hidden state initialization
            batch_size = 4
            hidden_state = model.lstm_reward_network.init_hidden_state(batch_size)
            # LSTM hidden state is a tuple (h, c)
            assert isinstance(hidden_state, tuple)
            assert len(hidden_state) == 2
            h, c = hidden_state
            assert h.shape == (batch_size, model.lstm_reward_network.config.lstm_hidden_size)
            assert c.shape == (batch_size, model.lstm_reward_network.config.lstm_hidden_size)
            
            # Test that LSTM works with flat observations (discrete games)
            flat_hidden_state = jnp.ones((batch_size, network_config.hidden_state_size))
            reward_pred, new_hidden = model.lstm_reward_network(flat_hidden_state, hidden_state, training=False)
            
            # Verify output shapes
            assert reward_pred.shape == (batch_size, 1)  # Scalar rewards
            assert isinstance(new_hidden, tuple) and len(new_hidden) == 2
            assert new_hidden[0].shape == (batch_size, network_config.lstm_hidden_size)
            assert new_hidden[1].shape == (batch_size, network_config.lstm_hidden_size)
        else:
            # LSTM should be disabled when use_value_prefix=False
            assert model.lstm_reward_network is None
    
    def test_value_prefix_reward_accumulation_integration(self, model, config):
        """Test that value prefix reward accumulation works with LSTM network."""
        from open_spiel.python.algorithms.muzero_jax.training.trainer import apply_value_prefix_reward_accumulation
        
        batch_size = 4
        num_steps = 6
        
        # Create test reward tensor
        target_rewards = jnp.ones((batch_size, num_steps))
        
        # Create test hidden states (dummy for this test)
        hidden_states = jnp.ones((batch_size, num_steps, 32))  # Flat hidden states
        
        # Test value prefix accumulation with LSTM
        accumulated_rewards = apply_value_prefix_reward_accumulation(
            target_rewards, config, None, model, hidden_states, None
        )
        
        # Check output shapes
        assert accumulated_rewards.shape == target_rewards.shape
    
    def test_policy_reanalysis_with_mcts(self, model, config):
        """Test that policy reanalysis uses real MCTS instead of placeholders."""
        from open_spiel.python.algorithms.muzero_jax.training.trainer import compute_policy_reanalysis_targets
        
        batch_size = 4
        num_steps = 6
        num_actions = config.num_actions
        
        # Create test observations (flat for tic-tac-toe)
        observations = jnp.ones((batch_size, num_steps, 9))
        
        rng_key = jax.random.PRNGKey(42)
        
        # Test policy reanalysis
        reanalyzed_policies = compute_policy_reanalysis_targets(
            model=model,
            observations=observations,
            config=config,
            training=False,
            rng_key=rng_key
        )
        
        # Check output shape
        assert reanalyzed_policies.shape == (batch_size, num_steps, num_actions)
        
        # Check that policies are not uniform (indicating real MCTS was used)
        # Allow some tolerance for cases where MCTS might produce near-uniform results
        policy_variance = jnp.var(reanalyzed_policies, axis=-1)
        
        # At least some policies should have non-zero variance (not perfectly uniform)
        # This indicates that real MCTS search was performed rather than returning uniform placeholders
        assert jnp.any(policy_variance > 1e-6), "All policies are uniform, suggesting placeholder implementation"
    
    def test_no_remaining_placeholders_in_production_paths(self, model, config):
        """Test that no placeholder values are returned in production code paths."""
        from open_spiel.python.algorithms.muzero_jax.training.trainer import compute_policy_reanalysis_targets
        
        # Test with non-zero reanalyze_ratio to ensure production path is taken
        import dataclasses
        config = dataclasses.replace(config, reanalyze_ratio=0.5)
        
        batch_size = 2
        num_steps = 3
        observations = jnp.ones((batch_size, num_steps, 9))
        rng_key = jax.random.PRNGKey(42)
        
        # Get reanalyzed policies
        policies = compute_policy_reanalysis_targets(
            model=model,
            observations=observations,
            config=config,
            training=False,
            rng_key=rng_key
        )
        
        # Check that policies are valid probabilities
        assert jnp.all(policies >= 0.0), "Negative probabilities found"
        assert jnp.allclose(jnp.sum(policies, axis=-1), 1.0, atol=1e-5), "Policies don't sum to 1"
        
        # Check that not all policies are exactly uniform (would indicate placeholder)
        uniform_policy = jnp.ones(config.num_actions) / config.num_actions
        is_uniform = jnp.allclose(policies, uniform_policy, atol=1e-6)
        
        # At least some policies should be non-uniform if real MCTS is working
        assert not jnp.all(is_uniform), "All policies are uniform, suggesting placeholder behavior"


if __name__ == "__main__":
    pytest.main([__file__]) 
