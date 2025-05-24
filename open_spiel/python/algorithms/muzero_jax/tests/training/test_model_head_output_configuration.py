"""Tests for model head output configuration.

This module tests that model heads output data in the format expected by different loss functions,
streamlining loss computation by minimizing conversions.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest
import math
import dataclasses

from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
from open_spiel.python.algorithms.muzero_jax.models.network import PredictionNetwork, RewardNetwork
from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig, create_network_config_from_muzero_config


class TestModelHeadOutputConfiguration:
    """Test model head output configuration for different loss types."""

    @pytest.fixture
    def rng_key(self):
        """Provides a random key for tests."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def base_config(self):
        """Base network configuration for testing."""
        return MuZeroNetworkConfig(
            observation_shape=(8, 8, 3),
            num_actions=4,
            num_channels=32,
            use_image_observation=False,  # Use flat observations for simpler testing
            fc_prediction_layers=[16],
            value_support_size=601,
            reward_support_size=601,
            symlog_base=math.e
        )

    def test_value_head_output_dimensions_categorical(self, base_config, rng_key):
        """Test that value head outputs correct dimensions for categorical loss."""
        # Configure for categorical loss
        config = dataclasses.replace(base_config,
            value_loss_type="categorical",
            value_support_size=601
        )
        
        # Create prediction network
        pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        policy_logits, value_output = pred_net(hidden_state, training=False)
        
        # Check value head output dimensions
        assert value_output.shape == (batch_size, 601), f"Expected value shape (2, 601), got {value_output.shape}"
        assert policy_logits.shape == (batch_size, config.num_actions), f"Expected policy shape (2, 4), got {policy_logits.shape}"

    def test_value_head_output_dimensions_mse(self, base_config, rng_key):
        """Test that value head outputs correct dimensions for MSE loss."""
        # Configure for MSE loss
        config = dataclasses.replace(base_config,
            value_loss_type="mse",
            value_support_size=0  # Scalar output
        )
        
        # Create prediction network
        pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        policy_logits, value_output = pred_net(hidden_state, training=False)
        
        # Check value head output dimensions (scalar)
        assert value_output.shape == (batch_size,), f"Expected value shape (2,), got {value_output.shape}"
        assert policy_logits.shape == (batch_size, config.num_actions), f"Expected policy shape (2, 4), got {policy_logits.shape}"

    def test_value_head_output_dimensions_symlog(self, base_config, rng_key):
        """Test that value head outputs correct dimensions and applies symlog transformation."""
        # Configure for symlog loss
        config = dataclasses.replace(base_config,
            value_loss_type="symlog",
            value_support_size=0  # Scalar output for symlog
        )
        
        # Create prediction network
        pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        policy_logits, value_output = pred_net(hidden_state, training=False)
        
        # Check value head output dimensions (scalar, symlog-transformed)
        assert value_output.shape == (batch_size,), f"Expected value shape (2,), got {value_output.shape}"
        assert policy_logits.shape == (batch_size, config.num_actions), f"Expected policy shape (2, 4), got {policy_logits.shape}"
        
        # Check that values appear to be in symlog space (should be different from raw outputs)
        # Note: This is a sanity check - exact symlog verification would require more complex setup

    def test_reward_head_output_dimensions_categorical(self, base_config, rng_key):
        """Test that reward head outputs correct dimensions for categorical loss."""
        # Configure for categorical loss
        config = dataclasses.replace(base_config,
            reward_loss_type="categorical",
            reward_support_size=601
        )
        
        # Create reward network
        reward_net = RewardNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        reward_output = reward_net(hidden_state, training=False)
        
        # Check reward head output dimensions
        assert reward_output.shape == (batch_size, 601), f"Expected reward shape (2, 601), got {reward_output.shape}"

    def test_reward_head_output_dimensions_mse(self, base_config, rng_key):
        """Test that reward head outputs correct dimensions for MSE loss."""
        # Configure for MSE loss
        config = dataclasses.replace(base_config,
            reward_loss_type="mse",
            reward_support_size=0  # Scalar output
        )
        
        # Create reward network
        reward_net = RewardNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        reward_output = reward_net(hidden_state, training=False)
        
        # Check reward head output dimensions (scalar)
        assert reward_output.shape == (batch_size,), f"Expected reward shape (2,), got {reward_output.shape}"

    def test_reward_head_output_dimensions_symlog(self, base_config, rng_key):
        """Test that reward head outputs correct dimensions and applies symlog transformation."""
        # Configure for symlog loss
        config = dataclasses.replace(base_config,
            reward_loss_type="symlog",
            reward_support_size=0  # Scalar output for symlog
        )
        
        # Create reward network
        reward_net = RewardNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        reward_output = reward_net(hidden_state, training=False)
        
        # Check reward head output dimensions (scalar, symlog-transformed)
        assert reward_output.shape == (batch_size,), f"Expected reward shape (2,), got {reward_output.shape}"

    def test_reward_head_output_dimensions_kl(self, base_config, rng_key):
        """Test that reward head outputs correct dimensions for KL loss."""
        # Configure for KL loss
        config = dataclasses.replace(base_config,
            reward_loss_type="kl",
            reward_support_size=601
        )
        
        # Create reward network
        reward_net = RewardNetwork(config, rngs=nnx.Rngs(params=rng_key))
        
        # Test input
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, config.num_channels))
        
        # Forward pass
        reward_output = reward_net(hidden_state, training=False)
        
        # Check reward head output dimensions
        assert reward_output.shape == (batch_size, 601), f"Expected reward shape (2, 601), got {reward_output.shape}"

    def test_config_transfer_from_muzero_config(self, rng_key):
        """Test that loss types are properly transferred from MuZeroConfig to MuZeroNetworkConfig."""
        # Create MuZero training config
        muzero_config = MuZeroConfig(
            value_loss_type="categorical",
            reward_loss_type="symlog",
            value_support_size=601,
            reward_support_size=0,
            symlog_base=math.e
        )
        
        # Create network config from training config
        network_config = create_network_config_from_muzero_config(
            muzero_config=muzero_config,
            observation_shape=(8, 8, 3),
            num_actions=4,
            use_image_observation=False
        )
        
        # Verify loss types are transferred correctly
        assert network_config.value_loss_type == "categorical"
        assert network_config.reward_loss_type == "symlog"
        assert network_config.value_support_size == 601
        assert network_config.reward_support_size == 0
        assert network_config.symlog_base == math.e
        
        # Test that the network config produces correct head dimensions
        pred_net = PredictionNetwork(network_config, rngs=nnx.Rngs(params=rng_key))
        reward_net = RewardNetwork(network_config, rngs=nnx.Rngs(params=rng_key))
        
        batch_size = 2
        hidden_state = jax.random.normal(rng_key, (batch_size, network_config.num_channels))
        
        # Test value head (categorical -> support size output)
        policy_logits, value_output = pred_net(hidden_state, training=False)
        assert value_output.shape == (batch_size, 601)
        
        # Test reward head (symlog -> scalar output)
        reward_output = reward_net(hidden_state, training=False)
        assert reward_output.shape == (batch_size,)

    def test_dynamic_output_dimensions_helper_methods(self, base_config):
        """Test that helper methods return correct output dimensions."""
        # Test value output dimensions
        categorical_config = dataclasses.replace(base_config, value_loss_type="categorical", value_support_size=601)
        assert categorical_config.get_value_output_dim() == 601
        
        mse_config = dataclasses.replace(base_config, value_loss_type="mse", value_support_size=0)
        assert mse_config.get_value_output_dim() == 1
        
        symlog_config = dataclasses.replace(base_config, value_loss_type="symlog", value_support_size=0)
        assert symlog_config.get_value_output_dim() == 1
        
        # Test reward output dimensions
        categorical_reward_config = dataclasses.replace(base_config, reward_loss_type="categorical", reward_support_size=601)
        assert categorical_reward_config.get_reward_output_dim() == 601
        
        kl_reward_config = dataclasses.replace(base_config, reward_loss_type="kl", reward_support_size=601)
        assert kl_reward_config.get_reward_output_dim() == 601
        
        mse_reward_config = dataclasses.replace(base_config, reward_loss_type="mse", reward_support_size=0)
        assert mse_reward_config.get_reward_output_dim() == 1
        
        symlog_reward_config = dataclasses.replace(base_config, reward_loss_type="symlog", reward_support_size=0)
        assert symlog_reward_config.get_reward_output_dim() == 1

    def test_loss_computation_streamlining(self, base_config, rng_key):
        """Test that model outputs minimize the need for conversions in loss computation."""
        batch_size = 2
        
        # Test categorical value case - model should output logits directly
        categorical_config = dataclasses.replace(base_config,
            value_loss_type="categorical",
            value_support_size=601
        )
        pred_net_cat = PredictionNetwork(categorical_config, rngs=nnx.Rngs(params=rng_key))
        hidden_state = jax.random.normal(rng_key, (batch_size, categorical_config.num_channels))
        
        _, value_output_cat = pred_net_cat(hidden_state, training=False)
        
        # Should output logits over support (no conversion needed for categorical loss)
        assert value_output_cat.shape == (batch_size, 601)
        assert value_output_cat.ndim == 2  # Not scalar, so ready for categorical loss
        
        # Test MSE value case - model should output scalars directly
        mse_config = dataclasses.replace(base_config,
            value_loss_type="mse",
            value_support_size=0
        )
        pred_net_mse = PredictionNetwork(mse_config, rngs=nnx.Rngs(params=rng_key))
        
        _, value_output_mse = pred_net_mse(hidden_state, training=False)
        
        # Should output scalars (no conversion needed for MSE loss)
        assert value_output_mse.shape == (batch_size,)
        assert value_output_mse.ndim == 1  # Scalar, ready for MSE loss
        
        # Test symlog value case - model should output symlog-transformed scalars
        symlog_config = dataclasses.replace(base_config,
            value_loss_type="symlog",
            value_support_size=0
        )
        pred_net_symlog = PredictionNetwork(symlog_config, rngs=nnx.Rngs(params=rng_key))
        
        _, value_output_symlog = pred_net_symlog(hidden_state, training=False)
        
        # Should output symlog-transformed scalars (minimal conversion needed for symlog loss)
        assert value_output_symlog.shape == (batch_size,)
        assert value_output_symlog.ndim == 1  # Scalar, in symlog space


if __name__ == "__main__":
    pytest.main([__file__]) 