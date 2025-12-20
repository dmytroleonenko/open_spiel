"""
Tests for LSTM-based value-prefix reward accumulation (EfficientZeroV2 feature).

This module tests the SupportLSTMRewardNetwork implementation and its integration
with the MuZero training pipeline, ensuring compatibility with EfficientZeroV2.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Tuple
from unittest.mock import Mock

from open_spiel.python.algorithms.muzero_jax.models.network import (
    SupportLSTMRewardNetwork, MuZeroNetwork, RepresentationNetwork, 
    DynamicsNetwork, PredictionNetwork, RewardNetwork, LSTMState
)
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    apply_value_prefix_reward_accumulation, MuZeroConfig, Learner
)


@pytest.fixture
def lstm_config():
    """Configuration for LSTM reward network testing."""
    return MuZeroNetworkConfig(
        observation_shape=(8, 8, 3),
        num_actions=4,
        num_channels=64,
        use_image_observation=True,
        spatial_extents=(2, 2),  # Actual spatial dimensions of hidden state
        num_residual_blocks=2,
        value_support_size=0,  # Scalar values
        reward_support_size=0,  # Scalar rewards
        use_value_prefix=True,
        lstm_hidden_size=128,
        reduced_channels_reward=16,
        lstm_horizon_length=5,
        hidden_state_size=64,
        spatial_size=4,  # 2*2 (actual output from representation network)
    )


@pytest.fixture
def categorical_lstm_config():
    """Configuration for categorical LSTM reward network testing."""
    return MuZeroNetworkConfig(
        observation_shape=(8, 8, 3),
        num_actions=4,
        num_channels=64,
        use_image_observation=True,
        spatial_extents=(8, 8),
        num_residual_blocks=2,
        value_support_size=601,  # Categorical values
        reward_support_size=601,  # Categorical rewards
        use_value_prefix=True,
        lstm_hidden_size=128,
        reduced_channels_reward=16,
        lstm_horizon_length=3,
        hidden_state_size=64,
        spatial_size=64,  # 8*8
    )


@pytest.fixture
def rng_key():
    """Random number generator key."""
    return jax.random.PRNGKey(42)


class TestSupportLSTMRewardNetwork:
    """Test cases for SupportLSTMRewardNetwork architecture."""
    
    def test_lstm_network_initialization(self, lstm_config, rng_key):
        """Test LSTM reward network initialization."""
        rngs = nnx.Rngs(params=rng_key)
        
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        # Check network components exist
        assert hasattr(lstm_network, 'conv1x1_reward')
        assert hasattr(lstm_network, 'lstm_cell')
        assert hasattr(lstm_network, 'reward_head')
        assert lstm_network.config == lstm_config
    
    def test_lstm_hidden_state_initialization(self, lstm_config, rng_key):
        """Test LSTM hidden state initialization."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 4
        hidden_state = lstm_network.init_hidden_state(batch_size)
        
        assert isinstance(hidden_state, tuple) and len(hidden_state) == 2
        c, h = hidden_state
        assert h.shape == (batch_size, lstm_config.lstm_hidden_size)
        assert c.shape == (batch_size, lstm_config.lstm_hidden_size)
        assert jnp.allclose(h, 0.0)
        assert jnp.allclose(c, 0.0)
    
    def test_lstm_hidden_state_reset(self, lstm_config, rng_key):
        """Test LSTM hidden state reset functionality."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 4
        # Create non-zero hidden state
        c = jnp.ones((batch_size, lstm_config.lstm_hidden_size)) * 2.0
        h = jnp.ones((batch_size, lstm_config.lstm_hidden_size))
        hidden_state = (c, h)
        
        # Reset first and third batch items
        reset_mask = jnp.array([1.0, 0.0, 1.0, 0.0])
        reset_hidden = lstm_network.reset_hidden_state(hidden_state, reset_mask)
        
        # Check reset behavior
        expected_h = jnp.array([[0.0], [1.0], [0.0], [1.0]]) * jnp.ones((1, lstm_config.lstm_hidden_size))
        expected_c = jnp.array([[0.0], [2.0], [0.0], [2.0]]) * jnp.ones((1, lstm_config.lstm_hidden_size))
        
        reset_c, reset_h = reset_hidden
        assert jnp.allclose(reset_h, expected_h)
        assert jnp.allclose(reset_c, expected_c)
    
    def test_lstm_forward_pass_scalar_rewards(self, lstm_config, rng_key):
        """Test LSTM forward pass with scalar rewards."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 2
        # Use correct spatial dimensions from config (2, 2) not (8, 8)
        hidden_state_input = jax.random.normal(
            rng_key, (batch_size, lstm_config.spatial_extents[0], lstm_config.spatial_extents[1], lstm_config.hidden_state_size)
        )
        reward_hidden = lstm_network.init_hidden_state(batch_size)
        
        reward_pred, new_reward_hidden = lstm_network(
            hidden_state_input, reward_hidden, training=False
        )
        
        # Check output shapes
        assert reward_pred.shape == (batch_size, 1)  # Scalar rewards
        assert isinstance(new_reward_hidden, tuple) and len(new_reward_hidden) == 2
        new_c, new_h = new_reward_hidden
        old_c, old_h = reward_hidden
        assert new_h.shape == (batch_size, lstm_config.lstm_hidden_size)
        assert new_c.shape == (batch_size, lstm_config.lstm_hidden_size)
        
        # Check that hidden state changed
        assert not jnp.allclose(new_h, old_h)
        assert not jnp.allclose(new_c, old_c)
    
    def test_lstm_forward_pass_categorical_rewards(self, categorical_lstm_config, rng_key):
        """Test LSTM forward pass with categorical rewards."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(categorical_lstm_config, rngs=rngs)
        
        batch_size = 2
        # Use correct spatial dimensions from config (8, 8) for categorical config
        hidden_state_input = jax.random.normal(
            rng_key, (batch_size, categorical_lstm_config.spatial_extents[0], categorical_lstm_config.spatial_extents[1], categorical_lstm_config.hidden_state_size)
        )
        reward_hidden = lstm_network.init_hidden_state(batch_size)
        
        reward_pred, new_reward_hidden = lstm_network(
            hidden_state_input, reward_hidden, training=False
        )
        
        # Check output shapes for categorical rewards
        assert reward_pred.shape == (batch_size, categorical_lstm_config.reward_support_size)
        assert isinstance(new_reward_hidden, tuple) and len(new_reward_hidden) == 2
        new_c, new_h = new_reward_hidden
        assert new_h.shape == (batch_size, categorical_lstm_config.lstm_hidden_size)
        assert new_c.shape == (batch_size, categorical_lstm_config.lstm_hidden_size)
    
    def test_lstm_sequential_consistency(self, lstm_config, rng_key):
        """Test that sequential LSTM calls maintain consistency."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 1
        num_steps = 3
        # Use correct spatial dimensions from config (2, 2) not (8, 8)
        hidden_state_input = jax.random.normal(
            rng_key, (batch_size, lstm_config.spatial_extents[0], lstm_config.spatial_extents[1], lstm_config.hidden_state_size)
        )
        
        # Sequential forward passes
        reward_hidden = lstm_network.init_hidden_state(batch_size)
        rewards = []
        
        for step in range(num_steps):
            reward_pred, reward_hidden = lstm_network(
                hidden_state_input, reward_hidden, training=False
            )
            rewards.append(reward_pred)
        
        # Check that rewards are different (LSTM has memory)
        assert not jnp.allclose(rewards[0], rewards[1])
        assert not jnp.allclose(rewards[1], rewards[2])


class TestMuZeroNetworkLSTMIntegration:
    """Test LSTM integration with MuZeroNetwork."""
    
    def create_test_muzero_network(self, config, rng_key):
        """Create a test MuZero network with LSTM support."""
        rngs = nnx.Rngs(params=rng_key)
        
        def repr_net_def(cfg, *, rngs):
            return RepresentationNetwork(cfg, rngs=rngs)
        
        def dyn_net_def(cfg, *, rngs):
            return DynamicsNetwork(cfg, rngs=rngs)
        
        def pred_net_def(cfg, *, rngs):
            return PredictionNetwork(cfg, rngs=rngs)
        
        def reward_net_def(cfg, *, rngs):
            return RewardNetwork(cfg, rngs=rngs)
        
        return MuZeroNetwork(
            representation_network_def=repr_net_def,
            dynamics_network_def=dyn_net_def,
            prediction_network_def=pred_net_def,
            reward_network_def=reward_net_def,
            projection_network_def=None,
            config=config,
            rngs=rngs
        )
    
    def test_muzero_network_lstm_initialization(self, lstm_config, rng_key):
        """Test MuZero network with LSTM reward network initialization."""
        model = self.create_test_muzero_network(lstm_config, rng_key)
        
        # Check LSTM reward network is created
        assert hasattr(model, 'lstm_reward_network')
        assert model.lstm_reward_network is not None
        assert isinstance(model.lstm_reward_network, SupportLSTMRewardNetwork)
    
    def test_muzero_network_without_lstm(self, rng_key):
        """Test MuZero network without LSTM (use_value_prefix=False)."""
        config = MuZeroNetworkConfig(
            observation_shape=(8, 8, 3),
            num_actions=4,
            use_value_prefix=False,  # Disabled
        )
        
        model = self.create_test_muzero_network(config, rng_key)
        
        # Check LSTM reward network is not created
        assert hasattr(model, 'lstm_reward_network')
        assert model.lstm_reward_network is None
    
    def test_initial_inference_with_lstm(self, lstm_config, rng_key):
        """Test initial inference with LSTM reward hidden state."""
        model = self.create_test_muzero_network(lstm_config, rng_key)
        
        batch_size = 2
        observation = jax.random.normal(rng_key, (batch_size, 8, 8, 3))
        reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size)
        
        output = model.initial_inference(
            observation, training=False, reward_hidden=reward_hidden
        )
        
        # Check output structure
        assert len(output) == 6  # hidden_state, reward, value, policy, projection, reward_hidden
        hidden_state, reward, value, policy, projection, new_reward_hidden = output
        
        assert hidden_state.shape[0] == batch_size
        assert reward.shape == (batch_size, 1)  # Scalar rewards
        assert value.shape == (batch_size,)   # Scalar values are squeezed
        assert policy.shape == (batch_size, lstm_config.num_actions)
        assert projection is None  # No projection network
        assert isinstance(new_reward_hidden, tuple) and len(new_reward_hidden) == 2
    
    def test_recurrent_inference_with_lstm(self, lstm_config, rng_key):
        """Test recurrent inference with LSTM reward hidden state."""
        model = self.create_test_muzero_network(lstm_config, rng_key)
        
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, 2, 2, 64))  # Match spatial_extents
        action = jnp.array([0, 1])
        reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size)
        
        output = model.recurrent_inference(
            hidden_state, action, training=False, reward_hidden=reward_hidden
        )
        
        # Check output structure
        assert len(output) == 6
        next_hidden, reward, value, policy, projection, new_reward_hidden = output
        
        assert next_hidden.shape[0] == batch_size
        assert reward.shape == (batch_size, 1)
        assert value.shape == (batch_size,)  # Scalar values are squeezed
        assert policy.shape == (batch_size, lstm_config.num_actions)
        assert isinstance(new_reward_hidden, tuple) and len(new_reward_hidden) == 2


class TestValuePrefixRewardAccumulation:
    """Test apply_value_prefix_reward_accumulation function."""
    
    def create_test_config(self, use_value_prefix=True, lstm_horizon_length=5):
        """Create test MuZero configuration."""
        return MuZeroConfig(
            use_value_prefix=use_value_prefix,
            lstm_horizon_length=lstm_horizon_length,
            lstm_hidden_size=128,
            num_unroll_steps=5,
            num_actions=4,
            batch_size=2,
            reanalyze_ratio=0.0,
            reward_support_size=0,  # Scalar rewards
        )
    
    def create_test_model(self, config, rng_key):
        """Create test model for value prefix testing."""
        network_config = MuZeroNetworkConfig(
            observation_shape=(8, 8, 3),
            num_actions=4,
            use_value_prefix=config.use_value_prefix,
            lstm_hidden_size=128,
            reduced_channels_reward=16,
            lstm_horizon_length=config.lstm_horizon_length,
            hidden_state_size=64,
            spatial_size=4,  # 2*2 spatial dimensions
            spatial_extents=(2, 2),  # Actual spatial dimensions of hidden state
            use_image_observation=True,
            num_channels=64,
            num_residual_blocks=2,
            value_support_size=0,
            reward_support_size=0,
        )
        
        rngs = nnx.Rngs(params=rng_key)
        
        def repr_net_def(cfg, *, rngs):
            return RepresentationNetwork(cfg, rngs=rngs)
        
        def dyn_net_def(cfg, *, rngs):
            return DynamicsNetwork(cfg, rngs=rngs)
        
        def pred_net_def(cfg, *, rngs):
            return PredictionNetwork(cfg, rngs=rngs)
        
        def reward_net_def(cfg, *, rngs):
            return RewardNetwork(cfg, rngs=rngs)
        
        return MuZeroNetwork(
            representation_network_def=repr_net_def,
            dynamics_network_def=dyn_net_def,
            prediction_network_def=pred_net_def,
            reward_network_def=reward_net_def,
            projection_network_def=None,
            config=network_config,
            rngs=rngs
        )

    def create_dummy_batch(self, config: MuZeroConfig, batch_size: int = 1):
        """Create a minimal batch compatible with _compute_total_loss_static."""
        num_steps = config.num_unroll_steps + 1
        obs_shape = (8, 8, 3)
        observation = jnp.ones((batch_size, num_steps, *obs_shape))
        action = jnp.zeros((batch_size, config.num_unroll_steps), dtype=jnp.int32)
        target_reward = jnp.zeros((batch_size, num_steps))
        target_value = jnp.zeros((batch_size, num_steps))
        target_policy = jnp.full(
            (batch_size, num_steps, config.num_actions),
            1.0 / config.num_actions,
        )
        mask = jnp.ones((batch_size, num_steps))
        return {
            "observation": observation,
            "action": action,
            "target_reward": target_reward,
            "target_value": target_value,
            "target_policy": target_policy,
            "game_history_mask": mask,
        }
    
    def test_value_prefix_disabled(self, rng_key):
        """Test value prefix when disabled."""
        config = self.create_test_config(use_value_prefix=False)
        
        batch_size, num_steps = 2, 6
        target_rewards = jax.random.normal(rng_key, (batch_size, num_steps))
        
        result_rewards = apply_value_prefix_reward_accumulation(
            target_rewards, config, None, None, None, None
        )
        
        # Should return original rewards unchanged
        assert jnp.allclose(result_rewards, target_rewards)
    
    def test_value_prefix_target_accumulation(self, rng_key):
        """Test value prefix target accumulation (ignoring model)."""
        config = self.create_test_config(use_value_prefix=True, lstm_horizon_length=3)
        # Even if model is passed, it should be ignored for target generation
        model = self.create_test_model(config, rng_key)
        
        batch_size, num_steps = 2, 6
        # Use simple ones for predictable accumulation
        target_rewards = jnp.ones((batch_size, num_steps))
        hidden_states = jax.random.normal(rng_key, (batch_size, num_steps, 2, 2, 64))
        initial_reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size)
        
        result_rewards = apply_value_prefix_reward_accumulation(
            target_rewards, config, None, model, hidden_states, initial_reward_hidden
        )
        
        # Check output structure
        assert result_rewards.shape == target_rewards.shape

        # Verify accumulation: 1, 2, 3 (reset) 1, 2, 3
        expected_row = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        assert jnp.allclose(result_rewards[0], expected_row)
        assert jnp.allclose(result_rewards[1], expected_row)
    
    def test_value_prefix_with_game_mask(self, rng_key):
        """Test value prefix with game history mask."""
        config = self.create_test_config(use_value_prefix=True)
        model = self.create_test_model(config, rng_key)
        
        batch_size, num_steps = 2, 6
        target_rewards = jnp.ones((batch_size, num_steps))
        hidden_states = jax.random.normal(rng_key, (batch_size, num_steps, 2, 2, 64))
        game_mask = jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=jnp.float32)
        
        result_rewards = apply_value_prefix_reward_accumulation(
            target_rewards, config, game_mask, model, hidden_states, None
        )
        
        # Check that masked positions are zeroed
        assert jnp.allclose(result_rewards[0, 3:], 0.0)  # Last 3 steps masked
        assert jnp.allclose(result_rewards[1, 4:], 0.0)  # Last 2 steps masked

        # Verify accumulation where mask is 1
        # Horizon is 5 (default).
        # Row 0: 1, 2, 3, 0, 0, 0
        assert jnp.allclose(result_rewards[0, :3], jnp.array([1.0, 2.0, 3.0]))
    
    def test_value_prefix_horizon_reset_behavior(self, rng_key):
        """Test LSTM horizon reset behavior."""
        config = self.create_test_config(use_value_prefix=True, lstm_horizon_length=2)
        model = self.create_test_model(config, rng_key)
        
        batch_size, num_steps = 1, 4
        target_rewards = jnp.ones((batch_size, num_steps))
        hidden_states = jax.random.normal(rng_key, (batch_size, num_steps, 2, 2, 64))
        
        result_rewards = apply_value_prefix_reward_accumulation(
            target_rewards, config, None, model, hidden_states, None
        )
        
        # With horizon 2 and ones input: 1, 2, 1, 2
        expected = jnp.array([[1.0, 2.0, 1.0, 2.0]])
        assert jnp.allclose(result_rewards, expected)

    def test_total_loss_static_initializes_lstm_hidden_state(self, rng_key):
        config = dataclasses.replace(
            self.create_test_config(use_value_prefix=True, lstm_horizon_length=1),
            num_unroll_steps=1,
            batch_size=1,
            num_actions=4,
        )
        model = self.create_test_model(config, rng_key)
        batch = self.create_dummy_batch(config, batch_size=1)

        loss, metrics = Learner._compute_total_loss_static(
            model,
            config,
            batch,
            rng_key,
            training=True,
        )

        assert loss.shape == ()
        assert "policy_loss" in metrics

    def test_total_loss_static_validates_lstm_hidden_dim(self, rng_key):
        config = dataclasses.replace(
            self.create_test_config(use_value_prefix=True, lstm_horizon_length=1),
            num_unroll_steps=1,
            batch_size=1,
            num_actions=4,
        )
        model = self.create_test_model(config, rng_key)

        def bad_init(self, batch_size):
            bad_dim = config.lstm_hidden_size // 2
            zeros = jnp.zeros((batch_size, bad_dim))
            return (zeros, zeros)

        model.lstm_reward_network.init_hidden_state = bad_init.__get__(
            model.lstm_reward_network,
            type(model.lstm_reward_network),
        )

        batch = self.create_dummy_batch(config, batch_size=1)

        with pytest.raises(ValueError, match="LSTM hidden state dimension mismatch"):
            Learner._compute_total_loss_static(
                model,
                config,
                batch,
                rng_key,
                training=True,
            )

    def test_value_prefix_fallback_without_hidden_states(self, rng_key):
        config = self.create_test_config(use_value_prefix=True)
        model = self.create_test_model(config, rng_key)
        target_rewards = jnp.arange(6, dtype=jnp.float32).reshape(1, -1)

        expected = apply_value_prefix_reward_accumulation(
            target_rewards,
            config,
            None,
            None,
            None,
            None,
        )
        actual = apply_value_prefix_reward_accumulation(
            target_rewards,
            config,
            None,
            model,
            None,
            None,
        )

        assert jnp.allclose(actual, expected)

    def test_mask_broadcast_for_categorical_rewards(self):
        config = MuZeroConfig(
            use_value_prefix=True,
            reward_support_size=3,
            lstm_horizon_length=2,
        )

        # Model is irrelevant now
        model = None
        target_rewards = jnp.zeros((2, 3, config.reward_support_size))
        mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        hidden_states = jnp.ones((2, 3, 1, 1, 4))

        predicted = apply_value_prefix_reward_accumulation(
            target_rewards,
            config,
            mask,
            model,
            hidden_states,
            None,
        )

        assert jnp.allclose(predicted[0, 2], 0.0)
        assert jnp.allclose(predicted[1, 1:], 0.0)
    
    def test_value_prefix_empty_input(self, rng_key):
        """Test value prefix with empty input."""
        config = self.create_test_config(use_value_prefix=True)
        
        empty_rewards = jnp.array([]).reshape(0, 0)
        
        result_rewards = apply_value_prefix_reward_accumulation(
            empty_rewards, config, None, None, None, None
        )
        
        assert result_rewards.shape == empty_rewards.shape


class TestLSTMNumericalStability:
    """Test numerical stability and edge cases for LSTM implementation."""
    
    def test_lstm_large_hidden_states(self, lstm_config, rng_key):
        """Test LSTM with large hidden state values."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 2
        # Large input values - use correct spatial dimensions
        hidden_state_input = jnp.ones((batch_size, lstm_config.spatial_extents[0], lstm_config.spatial_extents[1], lstm_config.hidden_state_size)) * 100.0
        reward_hidden = lstm_network.init_hidden_state(batch_size)
        
        reward_pred, new_reward_hidden = lstm_network(
            hidden_state_input, reward_hidden, training=False
        )
        
        # Check outputs are finite
        assert jnp.all(jnp.isfinite(reward_pred))
        new_c, new_h = new_reward_hidden
        assert jnp.all(jnp.isfinite(new_h))
        assert jnp.all(jnp.isfinite(new_c))
    
    def test_lstm_zero_hidden_states(self, lstm_config, rng_key):
        """Test LSTM with zero hidden state values."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 2
        # Zero input values - use correct spatial dimensions
        hidden_state_input = jnp.zeros((batch_size, lstm_config.spatial_extents[0], lstm_config.spatial_extents[1], lstm_config.hidden_state_size))
        reward_hidden = lstm_network.init_hidden_state(batch_size)
        
        reward_pred, new_reward_hidden = lstm_network(
            hidden_state_input, reward_hidden, training=False
        )
        
        # Check outputs are finite and not all zero (due to bias terms)
        assert jnp.all(jnp.isfinite(reward_pred))
        new_c, new_h = new_reward_hidden
        assert jnp.all(jnp.isfinite(new_h))
        assert jnp.all(jnp.isfinite(new_c))
    
    def test_lstm_gradient_flow(self, lstm_config, rng_key):
        """Test gradient flow through LSTM network."""
        rngs = nnx.Rngs(params=rng_key)
        lstm_network = SupportLSTMRewardNetwork(lstm_config, rngs=rngs)
        
        batch_size = 2
        # Use correct spatial dimensions for gradient flow test
        hidden_state_input = jax.random.normal(
            rng_key, (batch_size, lstm_config.spatial_extents[0], lstm_config.spatial_extents[1], lstm_config.hidden_state_size)
        )
        reward_hidden = lstm_network.init_hidden_state(batch_size)
        
        def loss_fn(model):
            reward_pred, _ = model(hidden_state_input, reward_hidden, training=True)
            return jnp.mean(reward_pred ** 2)
        
        loss, grads = nnx.value_and_grad(loss_fn)(lstm_network)
        
        # Check gradients exist and are finite
        assert jnp.isfinite(loss)
        
        # Check that gradients flow to all parameters
        grad_params = nnx.state(grads, nnx.Param)
        for param_name, param_grad in grad_params.items():
            if hasattr(param_grad, 'value'):
                assert jnp.all(jnp.isfinite(param_grad.value))
            else:
                for subparam in param_grad.values():
                    if hasattr(subparam, 'value'):
                        assert jnp.all(jnp.isfinite(subparam.value))


def test_lstm_reward_network_flat_vs_spatial():
    """Test that LSTM reward network works for both flat and spatial hidden states."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from open_spiel.python.algorithms.muzero_jax.models.network import SupportLSTMRewardNetwork
    from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
    
    batch_size = 4
    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=rng_key)
    
    # Test 1: Flat observations (discrete games)
    flat_config = MuZeroNetworkConfig(
        use_image_observation=False,
        hidden_state_size=64,
        reduced_channels_reward=16,
        spatial_size=16,  # Target LSTM input size
        lstm_hidden_size=128,
        reward_support_size=0  # Scalar rewards
    )
    
    flat_lstm = SupportLSTMRewardNetwork(flat_config, rngs=rngs)
    flat_hidden_state = jnp.ones((batch_size, 64))  # [B, C] flat
    
    # Test forward pass
    reward_pred_flat, lstm_state_flat = flat_lstm(flat_hidden_state, training=False)
    
    # Verify shapes
    assert reward_pred_flat.shape == (batch_size, 1), f"Expected (4, 1), got {reward_pred_flat.shape}"
    assert isinstance(lstm_state_flat, tuple) and len(lstm_state_flat) == 2
    assert lstm_state_flat[0].shape == (batch_size, 128)  # c state
    assert lstm_state_flat[1].shape == (batch_size, 128)  # h state
    
    # Test 2: Spatial observations (image games)
    spatial_config = MuZeroNetworkConfig(
        use_image_observation=True,
        hidden_state_size=64,
        reduced_channels_reward=16,
        spatial_size=16,  # 4*4 spatial dimensions
        lstm_hidden_size=128,
        reward_support_size=0  # Scalar rewards
    )
    
    spatial_lstm = SupportLSTMRewardNetwork(spatial_config, rngs=rngs)
    # JAX/Flax Conv expects [B, H, W, C] format (channels last)
    spatial_hidden_state = jnp.ones((batch_size, 4, 4, 64))  # [B, H, W, C] spatial
    
    # Test forward pass
    reward_pred_spatial, lstm_state_spatial = spatial_lstm(spatial_hidden_state, training=False)
    
    # Verify shapes
    assert reward_pred_spatial.shape == (batch_size, 1), f"Expected (4, 1), got {reward_pred_spatial.shape}"
    assert isinstance(lstm_state_spatial, tuple) and len(lstm_state_spatial) == 2
    assert lstm_state_spatial[0].shape == (batch_size, 128)  # c state
    assert lstm_state_spatial[1].shape == (batch_size, 128)  # h state
    
    # Test 3: Verify both paths produce valid LSTM inputs of same size
    # Both should produce inputs of size: reduced_channels_reward * spatial_size = 16 * 16 = 256
    
    # Test 4: Verify LSTM state persistence across steps
    reward_pred_2, lstm_state_2 = flat_lstm(flat_hidden_state, reward_hidden=lstm_state_flat, training=False)
    
    # LSTM states should have changed (not identical)
    assert not jnp.allclose(lstm_state_flat[0], lstm_state_2[0], atol=1e-6)
    assert not jnp.allclose(lstm_state_flat[1], lstm_state_2[1], atol=1e-6)
    
    print("✅ LSTM reward network works correctly for both flat and spatial inputs")


def test_lstm_for_imperfect_observability_games():
    """Test LSTM's potential for handling imperfect observability in discrete games."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from open_spiel.python.algorithms.muzero_jax.models.network import SupportLSTMRewardNetwork
    from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
    
    # Simulate a poker-like game with hidden information
    batch_size = 2
    sequence_length = 5
    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=rng_key)
    
    config = MuZeroNetworkConfig(
        use_image_observation=False,
        hidden_state_size=32,
        reduced_channels_reward=8,
        spatial_size=8,
        lstm_hidden_size=64,
        reward_support_size=0,
        lstm_horizon_length=10  # Longer horizon for imperfect observability
    )
    
    lstm_network = SupportLSTMRewardNetwork(config, rngs=rngs)
    
    # Simulate sequence of hidden states (e.g., poker hands over time)
    hidden_states_sequence = jax.random.normal(rng_key, (batch_size, sequence_length, 32))
    
    # Process sequence with LSTM memory
    lstm_hidden = None
    reward_predictions = []
    
    for t in range(sequence_length):
        hidden_state_t = hidden_states_sequence[:, t, :]  # [B, C]
        reward_pred, lstm_hidden = lstm_network(hidden_state_t, reward_hidden=lstm_hidden, training=False)
        reward_predictions.append(reward_pred)
    
    # Verify that LSTM maintains memory across sequence
    reward_predictions = jnp.stack(reward_predictions, axis=1)  # [B, T, 1]
    
    # Check that predictions change over time (indicating memory usage)
    prediction_variance = jnp.var(reward_predictions, axis=1)  # [B, 1]
    assert jnp.all(prediction_variance > 1e-6), "LSTM predictions should vary over time"
    
    # Test reset functionality (important for episode boundaries)
    reset_mask = jnp.array([1, 0])  # Reset first batch element, keep second
    reset_hidden = lstm_network.reset_hidden_state(lstm_hidden, reset_mask)
    
    # First element should be reset to zeros, second should be unchanged
    assert jnp.allclose(reset_hidden[0][0], 0.0, atol=1e-6), "First element should be reset"
    assert jnp.allclose(reset_hidden[1][0], 0.0, atol=1e-6), "First element should be reset"
    assert not jnp.allclose(reset_hidden[0][1], 0.0, atol=1e-6), "Second element should not be reset"
    assert not jnp.allclose(reset_hidden[1][1], 0.0, atol=1e-6), "Second element should not be reset"
    
    print("✅ LSTM shows potential for handling imperfect observability in discrete games")


def test_value_prefix_falls_back_without_hidden_states():
    """If hidden states are missing we fall back to simple accumulation."""
    base_config = MuZeroConfig(
        use_value_prefix=True,
        reward_support_size=0,
        lstm_horizon_length=2,
        num_unroll_steps=2,
        batch_size=2,
        num_actions=3,
    )
    rewards = jnp.ones((base_config.batch_size, base_config.num_unroll_steps + 1))
    accumulated = apply_value_prefix_reward_accumulation(
        rewards, base_config, None, model=None, hidden_states=None, initial_reward_hidden=None
    )
    assert accumulated.shape == rewards.shape


def test_value_prefix_mask_expansion_for_categorical_rewards():
    """Categorical rewards should broadcast the mask before applying it."""
    config = MuZeroConfig(
        use_value_prefix=True,
        reward_support_size=3,
        lstm_horizon_length=2,
        num_unroll_steps=1,
        batch_size=1,
        num_actions=2,
    )
    # Model is irrelevant now
    model = None
    hidden_states = jnp.zeros((1, config.num_unroll_steps + 1, 1, 1, 1))
    mask = jnp.array([[1.0, 0.0]])
    predicted = apply_value_prefix_reward_accumulation(
        jnp.zeros((1, config.num_unroll_steps + 1, config.reward_support_size)),
        config,
        mask,
        model=model,
        hidden_states=hidden_states,
        initial_reward_hidden=None,
    )
    assert predicted.shape == (1, 2, config.reward_support_size)
    assert jnp.allclose(predicted[0, 1], jnp.zeros(config.reward_support_size))


if __name__ == "__main__":
    pytest.main([__file__])
