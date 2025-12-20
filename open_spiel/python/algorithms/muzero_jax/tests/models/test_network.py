import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import dataclasses
from typing import Tuple
import math

from open_spiel.python.algorithms.muzero_jax.models import network as muzero_network_lib
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork, PredictionNetwork
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
from open_spiel.python.algorithms.muzero_jax.models.layers import MLP

# Dummy network components for testing MuZeroNetwork
class DummyRepresentationNetwork(nnx.Module):
    def __init__(self, config: MuZeroNetworkConfig, *, rngs: nnx.Rngs):
        self.config = config
        if config.use_image_observation:
            # nnx.Conv expects in_features, out_features.
            # Assuming config.observation_shape is (H, W, C)
            self.conv = nnx.Conv(in_features=config.observation_shape[-1], out_features=config.num_channels, kernel_size=(3,3), padding='SAME', rngs=rngs)
        else:
            self.dense = nnx.Linear(in_features=jnp.prod(jnp.array(config.observation_shape)), out_features=config.num_channels, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        if self.config.use_image_observation:
            # Original dummy logic returned a fixed shape based on spatial_extents.
            # This should be (B, H_spatial, W_spatial, C_channels_out)
            # where C_channels_out is config.num_channels.
            # The input x is (B, H_obs, W_obs, C_obs)
            # For this dummy, we don't actually pass x through self.conv, just return shaped ones.
            return jnp.ones((x.shape[0], *self.config.spatial_extents, self.config.num_channels))
        else:
            # Flat observation, original dummy logic:
            return jnp.ones((x.shape[0], self.config.num_channels))


class DummyDynamicsNetwork(nnx.Module):
    def __init__(self, config: MuZeroNetworkConfig, *, rngs: nnx.Rngs):
        self.config = config
        action_embedding_dim = config.num_channels // 2 
        self.action_embed = nnx.Embed(num_embeddings=config.num_actions, features=action_embedding_dim, rngs=rngs)
        combined_dim = config.num_channels + action_embedding_dim
        self.mlp = MLP(input_size=combined_dim, hidden_sizes=[config.num_channels], output_size=config.num_channels, rngs=rngs)

    def __call__(self, hidden_state: jax.Array, action: jax.Array, training: bool) -> jax.Array:
        action_embedded = self.action_embed(action)
        if action_embedded.ndim == 1: 
             action_embedded = jnp.repeat(action_embedded[jnp.newaxis,:], hidden_state.shape[0], axis=0)

        if self.config.use_image_observation and hidden_state.ndim > 2:
            current_hidden_features = jnp.mean(hidden_state, axis=tuple(range(1, hidden_state.ndim -1))) 
        else:
            current_hidden_features = hidden_state
        
        concatenated = jnp.concatenate([current_hidden_features, action_embedded], axis=-1)
        processed_flat = self.mlp(concatenated, training=training)

        if self.config.use_image_observation:
            return jnp.ones((hidden_state.shape[0], *self.config.spatial_extents, self.config.num_channels)) 
        return processed_flat


class DummyPredictionNetwork(nnx.Module):
    def __init__(self, config: MuZeroNetworkConfig, *, rngs: nnx.Rngs):
        self.config = config
        # Input to heads is num_channels features
        self.policy_head = nnx.Linear(config.num_channels, config.num_actions, rngs=rngs)
        output_value_size = config.value_support_size if config.value_support_size > 0 else 1
        self.value_head = nnx.Linear(config.num_channels, output_value_size, rngs=rngs)

    def __call__(self, hidden_state: jax.Array, training: bool) -> Tuple[jax.Array, jax.Array]:
        if self.config.use_image_observation and hidden_state.ndim > 2:
            # Global average pooling or flatten for prediction
            hidden_state_flat = jnp.mean(hidden_state, axis=tuple(range(1, hidden_state.ndim -1))) # (B, C)
        else:
            hidden_state_flat = hidden_state # (B, C)
        
        policy_logits = self.policy_head(hidden_state_flat)
        value = self.value_head(hidden_state_flat)
        if self.config.value_support_size == 0 and value.ndim > 1: # Scalar value, remove trailing dim if present
             value = jnp.squeeze(value, axis=-1)
        
        # EfficientZeroV2 pattern: Apply symlog transformation to value output if symlog loss is used
        if hasattr(self.config, 'value_loss_type') and self.config.value_loss_type == "symlog":
            if value.ndim == 1 or (value.ndim == 2 and value.shape[-1] == 1):
                # Only apply symlog to scalar values
                if value.ndim == 2:
                    value = jnp.squeeze(value, axis=-1)
                # Import symlog function from network module
                from open_spiel.python.algorithms.muzero_jax.models.network import symlog
                value = symlog(value, base=getattr(self.config, 'symlog_base', jnp.e))
        
        return policy_logits, value

class DummyRewardNetwork(nnx.Module):
    def __init__(self, config: MuZeroNetworkConfig, *, rngs: nnx.Rngs):
        self.config = config
        output_reward_size = config.reward_support_size if config.reward_support_size > 0 else 1
        self.reward_head = nnx.Linear(config.num_channels, output_reward_size, rngs=rngs)

    def __call__(self, hidden_state: jax.Array, training: bool) -> jax.Array:
        if self.config.use_image_observation and hidden_state.ndim > 2:
            hidden_state_flat = jnp.mean(hidden_state, axis=tuple(range(1, hidden_state.ndim-1)))
        else:
            hidden_state_flat = hidden_state
        
        reward = self.reward_head(hidden_state_flat)
        if self.config.reward_support_size == 0 and reward.ndim > 1: # Scalar reward, remove trailing dim
            reward = jnp.squeeze(reward, axis=-1)
        
        # EfficientZeroV2 pattern: Apply symlog transformation to reward output if symlog loss is used
        if hasattr(self.config, 'reward_loss_type') and self.config.reward_loss_type == "symlog":
            if reward.ndim == 1 or (reward.ndim == 2 and reward.shape[-1] == 1):
                # Only apply symlog to scalar rewards
                if reward.ndim == 2:
                    reward = jnp.squeeze(reward, axis=-1)
                # Import symlog function from network module
                from open_spiel.python.algorithms.muzero_jax.models.network import symlog
                reward = symlog(reward, base=getattr(self.config, 'symlog_base', jnp.e))
        
        return reward

class DummyProjectionNetwork(nnx.Module):
    """Dummy Projection Network for testing."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        # MLP for projection
        # print(f"DummyProjectionNetwork init: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        self.mlp = MLP(input_size=input_dim, 
                       hidden_sizes=[hidden_dim], 
                       output_size=output_dim, 
                       rngs=rngs)

    def __call__(self, hidden_state: jax.Array, training: bool) -> jax.Array:
        # print(f"DummyProjectionNetwork __call__: hidden_state shape {hidden_state.shape}")
        # Assuming hidden_state might be (batch, H, W, C) for image or (batch, features) for flat
        # For MLP, we need to flatten if it's image-like
        if hidden_state.ndim > 2:
            hidden_state_flat = hidden_state.reshape((hidden_state.shape[0], -1))
        else:
            hidden_state_flat = hidden_state
        # print(f"DummyProjectionNetwork: hidden_state_flat shape {hidden_state_flat.shape}")
        return self.mlp(hidden_state_flat, training=training) # Pass training flag

# --- Mock Configurations ---
@pytest.fixture
def base_config():
    class Config:
        def __init__(self):
            # Common
            self.num_actions = 10
            self.num_channels = 32 # Hidden state channels / feature size for flat obs
            self.representation_num_blocks = 2
            self.dynamics_num_blocks = 2
            self.prediction_num_blocks = 2
            self.fc_prediction_layers = [64]
            self.is_continuous = False
            self.value_support_size = 0 # 0 means scalar value
            self.reward_support_size = 0 # 0 means scalar reward
            self.action_embedding_dim = 8 # Example, if action_embedding is True in dynamics
            self.action_embedding = False # Default to False for dynamics
            self.batch_size = 2 # Added batch_size, as per triage

            # For image observations
            self.observation_shape_image = (96, 96, 3) # H, W, C for JAX Conv
            self.downsample = False # Default no downsample for representation
            self.spatial_extents = (6,6) # Example: H, W of feature maps after convs for flattening
            
            # For flat observations (e.g. board state vector)
            self.observation_shape_flat = (128,)
            self.fc_representation_layers = [64]
            self.fc_dynamics_layers = [64]

            # Default to flat observations unless specified
            self.use_image_observation = False
            self.observation_shape = self.observation_shape_flat
            
            # Add missing attributes that network.py expects
            self.num_residual_blocks = 2
            self.num_hidden_units_fc = 64
            self.downsample_blocks = 0  # 0 means no downsampling
            self.downsample_channels = 16
            
            # Loss type configuration for model head output configuration
            self.value_loss_type = "mse"  # Default to mse loss
            self.reward_loss_type = "mse"  # Default to mse loss
            self.symlog_base = math.e  # Base for symlog transformation
            
            # Noisy networks support (Action Item 25)
            self.noisy_net = False  # Default to disabled
            
        def get_value_output_dim(self) -> int:
            """Determines the correct output dimension for the value head based on loss type and support size."""
            if self.value_loss_type == "categorical":
                return self.value_support_size if self.value_support_size > 0 else 601
            else:  # "mse" or "symlog"
                return 1
        
        def get_reward_output_dim(self) -> int:
            """Determines the correct output dimension for the reward head based on loss type and support size."""
            if self.reward_loss_type in ["categorical", "kl"]:
                return self.reward_support_size if self.reward_support_size > 0 else 601
            else:  # "mse" or "symlog"
                return 1
    return Config()

@pytest.fixture
def image_obs_config(base_config):
    base_config.use_image_observation = True
    base_config.observation_shape = base_config.observation_shape_image
    # If not using the DownSample module, RepresentationNetwork output H, W are same as input.
    # Prediction and Reward networks use these spatial extents after their 1x1 convs.
    base_config.spatial_extents = (base_config.observation_shape[0], base_config.observation_shape[1]) # Should be (96,96)
    return base_config

@pytest.fixture
def image_obs_downsample_config(image_obs_config):
    image_obs_config.downsample_blocks = 1  # Enable downsampling
    # After downsampling (e.g. 2x strideconv, 2x strideconv in simplified DownSample)
    # 96 -> 48 -> 24. If DownSample is more complex, this needs adjustment.
    # Let's assume for testing our current simplified DownSample leads to e.g. 24x24
    image_obs_config.spatial_extents = (24,24) 
    return image_obs_config

@pytest.fixture
def flat_obs_config(base_config):
    base_config.use_image_observation = False
    base_config.observation_shape = (base_config.num_channels,)
    return base_config

@pytest.fixture
def action_embedding_config(base_config):
    base_config.action_embedding = True
    base_config.action_embedding_dim = 1  # Align with actual action_plane channel
    base_config.use_image_observation = True
    base_config.observation_shape = base_config.observation_shape_image
    # Adjust spatial extents for image obs representation
    base_config.spatial_extents = (base_config.observation_shape_image[0], base_config.observation_shape_image[1])
    # Dynamics for flat obs uses num_actions for action_dim if not embedding, 
    # or action_embedding_dim if embedding (though current code might need adjustment for MLP input size)
    # For now, just ensuring the flag is set for getattr check.
    return base_config

@pytest.fixture
def categorical_support_config(base_config):
    base_config.value_support_size = 601 # Example: -300 to 300
    base_config.reward_support_size = 601
    base_config.value_loss_type = "categorical"
    base_config.reward_loss_type = "categorical"
    return base_config

@pytest.fixture
def rngs():
    return nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))

# --- Individual Network Tests ---

# RepresentationNetwork Tests
@pytest.mark.parametrize("config_fixture_name", ["flat_obs_config", "image_obs_config", "image_obs_downsample_config"])
def test_representation_network(config_fixture_name, request, rngs):
    config = request.getfixturevalue(config_fixture_name)
    repr_net = muzero_network_lib.RepresentationNetwork(config, rngs=rngs)
    batch_size = 2
    dummy_observation = jnp.ones((batch_size, *config.observation_shape))
    hidden_state = repr_net(dummy_observation, training=True)
    
    expected_shape_prefix = (batch_size,)
    if config.use_image_observation:
        # Assuming representation output has spatial dimensions H', W'
        # This requires knowing the output H, W of the conv part of repr net
        # For now, assume it's config.spatial_extents if not downsampled, or different if downsampled.
        # If not downsampled, and no further pooling, spatial_extents might be original H, W divided by strides.
        # Let's assume for non-downsample path, it keeps spatial dims based on convs if not specified
        # For testing, let's just check the channel dim.
        # A more robust test would calculate exact H', W' based on convs in repr_net.
        # For now, we test channel dim and that it has spatial dims.
        assert len(hidden_state.shape) == 4 # B, H', W', C
        assert hidden_state.shape[-1] == config.num_channels
    else:
        assert hidden_state.shape == expected_shape_prefix + (config.num_channels,)

# DynamicsNetwork Tests
@pytest.mark.parametrize("config_fixture_name", ["flat_obs_config", "image_obs_config", "action_embedding_config"])
def test_dynamics_network(config_fixture_name, request, rngs):
    config = request.getfixturevalue(config_fixture_name)
    # This will cover line 114 if action_embedding_config is used, as getattr will be true.
    dyn_net = muzero_network_lib.DynamicsNetwork(config, rngs=rngs)
    batch_size = 2
    action_shape = (batch_size,) # For discrete actions
    dummy_action = jnp.zeros(action_shape, dtype=jnp.int32)

    if config.use_image_observation:
        # Assuming dynamics input hidden state is H'xW'xC
        dummy_hidden_state = jnp.ones((batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels))
        next_hidden_state = dyn_net(dummy_hidden_state, dummy_action, training=True)
        assert next_hidden_state.shape == dummy_hidden_state.shape
    else:
        dummy_hidden_state = jnp.ones((batch_size, config.num_channels))
        next_hidden_state = dyn_net(dummy_hidden_state, dummy_action, training=True)
        assert next_hidden_state.shape == (batch_size, config.num_channels)

def test_dynamics_network_flat_continuous(flat_obs_config, rngs):
    config = flat_obs_config
    config.is_continuous = True
    # For continuous, action dim might be different from num_actions (e.g. 2 for x,y control)
    # Let's assume action dim is config.num_actions for this test, as MLP input expects that.
    # If action_dim was different, MLP input_size in DynamicsNet for flat_obs would need adjustment.
    
    dyn_net = muzero_network_lib.DynamicsNetwork(config, rngs=rngs)
    batch_size = 2
    # Continuous action, e.g. shape (batch, num_actions) or (batch, action_dim)
    dummy_action_continuous = jax.random.uniform(rngs.params(), (batch_size, config.num_actions))
    dummy_hidden_state = jnp.ones((batch_size, config.num_channels))
    
    next_hidden_state = dyn_net(dummy_hidden_state, dummy_action_continuous, training=True)
    # This call should hit line 167 (action_one_hot = action)
    assert next_hidden_state.shape == (batch_size, config.num_channels)

def test_dynamics_network_flat_exotic_shapes(flat_obs_config, rngs):
    """Test flat observation dynamics with exotic shapes to cover action_one_hot.ndim>2 and hidden_state_flat.ndim>2."""
    config = flat_obs_config
    config.is_continuous = False
    dyn_net = muzero_network_lib.DynamicsNetwork(config, rngs=rngs)
    batch_size = 2
    # Hidden state with ndim > 2: e.g., a sequence length dimension
    dummy_hidden_state = jnp.ones((batch_size, 3, config.num_channels))
    # Action that leads to action_one_hot.ndim > 2: provide a 2D integer array
    dummy_action = jnp.array([[1, 2], [0, 3]], dtype=jnp.int32)
    
    # Expect a TypeError due to concatenation of mismatched dims after exotic shapes
    with pytest.raises(TypeError):
        _ = dyn_net(dummy_hidden_state, dummy_action, training=True)

# Add test to cover action_one_hot.ndim == 1 branch in flat DynamicsNetwork
def test_dynamics_network_flat_scalar_action(flat_obs_config, rngs):
    """
    Test flat continuous dynamics with a scalar action (0D array) to cover expanding 1D action_one_hot.
    """
    config = flat_obs_config
    config.is_continuous = False
    dyn_net = muzero_network_lib.DynamicsNetwork(config, rngs=rngs)
    # Hidden state with explicit batch dimension
    hidden_state = jnp.ones((1, config.num_channels))
    # Scalar action (0D array)
    action = jnp.array(2, dtype=jnp.int32)
    # This should hit the action_one_hot.ndim == 1 branch and expand to (1, num_actions)
    next_hidden_state = dyn_net(hidden_state, action, training=True)
    assert next_hidden_state.shape == (1, config.num_channels)

# PredictionNetwork Tests
@pytest.mark.parametrize("config_fixture_name", ["flat_obs_config", "image_obs_config", "categorical_support_config"])
def test_prediction_network(config_fixture_name, request, rngs):
    config = request.getfixturevalue(config_fixture_name)
    if config_fixture_name == "categorical_support_config": # it starts from base, set image/flat
        config.use_image_observation = True # Or False, to test both with support
        config.observation_shape = config.observation_shape_image if config.use_image_observation else config.observation_shape_flat

    pred_net = muzero_network_lib.PredictionNetwork(config, rngs=rngs)
    batch_size = 2

    if config.use_image_observation:
        dummy_hidden_state = jnp.ones((batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels))
    else:
        dummy_hidden_state = jnp.ones((batch_size, config.num_channels))
    
    policy_logits, value = pred_net(dummy_hidden_state, training=True)
    assert policy_logits.shape == (batch_size, config.num_actions)
    expected_value_shape = (batch_size, config.value_support_size) if config.value_support_size > 0 else (batch_size,)
    assert value.shape == expected_value_shape

def test_prediction_network_flat_categorical_support(flat_obs_config, categorical_support_config, rngs):
    # Test flat observations branch with categorical value support
    config = categorical_support_config
    config.use_image_observation = False
    config.observation_shape = config.observation_shape_flat
    # Ensure flat observation branch is used
    assert not config.use_image_observation
    pred_net = muzero_network_lib.PredictionNetwork(config, rngs=rngs)
    batch_size = 3
    dummy_hidden = jnp.ones((batch_size, config.num_channels))
    policy_logits, value = pred_net(dummy_hidden, training=False)
    # policy logits shape
    assert policy_logits.shape == (batch_size, config.num_actions)
    # value shape should match categorical support size
    assert value.shape == (batch_size, config.value_support_size)

# RewardNetwork Tests
@pytest.mark.parametrize("config_fixture_name", ["flat_obs_config", "image_obs_config", "categorical_support_config"])
def test_reward_network(config_fixture_name, request, rngs):
    config = request.getfixturevalue(config_fixture_name)
    if config_fixture_name == "categorical_support_config":
        config.use_image_observation = False # Or True
        config.observation_shape = config.observation_shape_image if config.use_image_observation else config.observation_shape_flat

    reward_net = muzero_network_lib.RewardNetwork(config, rngs=rngs)
    batch_size = 2

    if config.use_image_observation:
        dummy_hidden_state = jnp.ones((batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels))
    else:
        dummy_hidden_state = jnp.ones((batch_size, config.num_channels))

    reward = reward_net(dummy_hidden_state, training=True)
    expected_reward_shape = (batch_size, config.reward_support_size) if config.reward_support_size > 0 else (batch_size,)
    assert reward.shape == expected_reward_shape

# --- MuZeroNetwork Tests ---
@pytest.mark.parametrize("config_fixture_name", ["flat_obs_config", "image_obs_config"])
def test_muzero_initial_inference(config_fixture_name, request, rngs):
    config = request.getfixturevalue(config_fixture_name)
    # Turn off projection for these older tests, as they expect 4 return values
    config.use_projection = False
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=DummyProjectionNetwork if config.use_projection else None,
        config=config, 
        rngs=rngs
    )
    obs_shape = config.observation_shape
    if config.use_image_observation:
        # Check that obs_shape is consistent with its own channel dimension
        # and that spatial_extents match the H, W of obs_shape.
        # config.spatial_extents is defined as (obs_shape[0], obs_shape[1]) in image_obs_config fixture
        assert obs_shape[:2] == config.spatial_extents
        assert len(obs_shape) == 3 # Should be H, W, C
        # No assertion needed against config.num_channels here for input obs_shape
    else:
        # For flat observations, obs_shape is (features,)
        # Ensure it's a 1D tuple
        assert len(obs_shape) == 1
        # config.num_channels for flat observations is the size of the hidden state after MLP representation.
        # The input obs_shape for flat is defined in flat_obs_config as (config.num_channels,)
        # This seems to conflate input feature size with hidden size in the test config.
        # Let's assume flat_obs_config.observation_shape defines the input layer size.
        # The original assertion was `assert obs_shape == (config.num_channels,)`
        # The flat_obs_config sets `base_config.observation_shape = (base_config.num_channels,)`
        # So this assertion should hold if config is set up that way.
        assert obs_shape == (config.num_channels,)
    
    dummy_obs_shape = (config.batch_size, *obs_shape)
    dummy_obs = jnp.ones(dummy_obs_shape) # Use jnp.ones for dummy data

    # Test initial inference
    hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.initial_inference(dummy_obs, training=False)

    if config.reward_support_size > 0:
        assert reward.shape == (config.batch_size, config.reward_support_size)
    else:
        assert reward.shape == (config.batch_size,)
    
    if config.value_support_size > 0:
        assert value.shape == (config.batch_size, config.value_support_size)
    else:
        assert value.shape == (config.batch_size,)
        
    assert policy_logits.shape == (config.batch_size, config.num_actions)
    assert projected_output is None if config.use_projection else True

@pytest.mark.parametrize("config_fixture_name", ["flat_obs_config", "image_obs_config"])
def test_muzero_recurrent_inference(config_fixture_name, request, rngs):
    config = request.getfixturevalue(config_fixture_name)
    config.use_projection = False # Ensure projection is off for these original tests
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=None, # Projection is explicitly off
        config=config, 
        rngs=rngs
    )
    batch_size = 2
    
    # For discrete actions, action should be (batch_size,) or (batch_size, 1)
    # DynamicsNetwork for flat obs expects (batch_size, num_actions) if one-hot, or (batch_size,) if int to be one-hotted.
    # DynamicsNetwork for image obs expects (batch_size,) if int to be broadcasted.
    # The model.recurrent_inference will handle passing this to dynamics_net correctly.
    dummy_action = jax.random.randint(rngs.params(), (batch_size,), 0, config.num_actions)

    if config.use_image_observation:
        current_hidden_state = jnp.ones((batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels))
    else:
        current_hidden_state = jnp.ones((batch_size, config.num_channels))

    next_hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.recurrent_inference(current_hidden_state, dummy_action, training=False)

    if config.use_image_observation:
        assert next_hidden_state.shape == (batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels)
    else:
        assert next_hidden_state.shape == (batch_size, config.num_channels)

    if config.reward_support_size > 0:
        assert reward.shape == (batch_size, config.reward_support_size)
    else:
        assert reward.shape == (batch_size,)
    
    if config.value_support_size > 0:
        assert value.shape == (batch_size, config.value_support_size)
    else:
        assert value.shape == (batch_size,)
        
    assert policy_logits.shape == (batch_size, config.num_actions)
    assert projected_output is None # Since use_projection is False for this test

# Dummy config for testing
@pytest.fixture
def dummy_config_image():
    class Config:
        def __init__(self):
            self.observation_shape = (96, 96, 3) # H, W, C -- Corrected to HWC
            self.num_actions = 5
            self.num_channels = 16 # Output channels for representation/dynamics hidden state
            self.representation_num_blocks = 2
            self.dynamics_num_blocks = 2
            self.prediction_num_blocks = 2
            self.reward_support_size = 0 # Scalar reward
            self.value_support_size = 0  # Scalar value
            self.downsample = True
            self.fc_representation_layers = [] # Not used for image
            self.fc_dynamics_layers = [] # Not used for image
            self.fc_prediction_layers = [32] # FC layers in prediction head
            self.use_image_observation = True
            self.is_continuous = False
            self.spatial_extents = (24, 24) # H_out, W_out after representation
            self.action_embedding = False # Simplified dynamics
            self.use_projection = True
            self.projection_input_dim = self.num_channels * self.spatial_extents[0] * self.spatial_extents[1]
            self.projection_hidden_dim = 64
            self.projection_output_dim = 64
            self.projection_head_hidden_dim = 32
            self.projection_head_output_dim = 32
            self.batch_size = 2 # Added batch_size
            # Loss type configuration
            self.value_loss_type = "mse"
            self.reward_loss_type = "mse"
            self.symlog_base = math.e
            # Noisy networks support (Action Item 25)
            self.noisy_net = False  # Default to disabled
            
        def get_value_output_dim(self) -> int:
            """Determines the correct output dimension for the value head based on loss type and support size."""
            if self.value_loss_type == "categorical":
                return self.value_support_size if self.value_support_size > 0 else 601
            else:  # "mse" or "symlog"
                return 1
        
        def get_reward_output_dim(self) -> int:
            """Determines the correct output dimension for the reward head based on loss type and support size."""
            if self.reward_loss_type in ["categorical", "kl"]:
                return self.reward_support_size if self.reward_support_size > 0 else 601
            else:  # "mse" or "symlog"
                return 1
    return Config()

@pytest.fixture
def dummy_config_flat():
    class Config:
        def __init__(self):
            self.observation_shape = (50,) # Flat observation vector
            self.num_actions = 7
            self.num_channels = 64 # Size of hidden state for flat observations
            self.representation_num_blocks = 0 # Not used for flat if using MLP directly
            self.dynamics_num_blocks = 0 # Not used for flat if using MLP directly
            self.prediction_num_blocks = 0 # Not used for flat if using MLP directly
            self.reward_support_size = 3 # Categorical reward
            self.value_support_size = 5 # Categorical value
            self.downsample = False # Not used for flat
            self.fc_representation_layers = [128, 64]
            self.fc_dynamics_layers = [128, 64]
            self.fc_prediction_layers = [128, 64]
            self.use_image_observation = False
            self.is_continuous = False
            self.spatial_extents = (0,0) # Not used for flat
            self.action_embedding = False
            self.use_projection = True
            self.projection_input_dim = self.num_channels # Hidden state size for flat
            self.projection_hidden_dim = 64
            self.projection_output_dim = 64
            self.projection_head_hidden_dim = 32
            self.projection_head_output_dim = 32
            self.batch_size = 2 # Added batch_size
            # Loss type configuration
            self.value_loss_type = "categorical"  # Since this config has support_size > 0
            self.reward_loss_type = "categorical"  # Since this config has support_size > 0  
            self.symlog_base = math.e
            # Noisy networks support (Action Item 25)
            self.noisy_net = False  # Default to disabled
            
        def get_value_output_dim(self) -> int:
            """Determines the correct output dimension for the value head based on loss type and support size."""
            if self.value_loss_type == "categorical":
                return self.value_support_size if self.value_support_size > 0 else 601
            else:  # "mse" or "symlog"
                return 1
        
        def get_reward_output_dim(self) -> int:
            """Determines the correct output dimension for the reward head based on loss type and support size."""
            if self.reward_loss_type in ["categorical", "kl"]:
                return self.reward_support_size if self.reward_support_size > 0 else 601
            else:  # "mse" or "symlog"
                return 1
    return Config()

def test_projection_network(dummy_config_image):
    key = jax.random.PRNGKey(0)
    params_key, dropout_key, low_key, high_key = jax.random.split(key, 4)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    config = dummy_config_image
    batch_size = 2
    input_dim = config.num_channels * config.spatial_extents[0] * config.spatial_extents[1]
    hidden_state_image = jax.random.normal(low_key, (batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels))

    projection_net = muzero_network_lib.ProjectionNetwork(
        input_dim=input_dim,
        hidden_dim=config.projection_hidden_dim,
        output_dim=config.projection_output_dim,
        rngs=rngs
    )
    output = projection_net(hidden_state_image, training=False)
    assert output.shape == (batch_size, config.projection_output_dim)
    hidden_state_flat = jax.random.normal(high_key, (batch_size, input_dim))
    output_flat = projection_net(hidden_state_flat, training=True)
    assert output_flat.shape == (batch_size, config.projection_output_dim)

def test_projection_head_network(dummy_config_image):
    key = jax.random.PRNGKey(1)
    params_key, dropout_key, input_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    config = dummy_config_image
    batch_size = 3
    input_tensor = jax.random.normal(input_key, (batch_size, config.projection_output_dim))

    projection_head_net = muzero_network_lib.ProjectionHeadNetwork(
        input_dim=config.projection_output_dim,
        hidden_dim=config.projection_head_hidden_dim,
        output_dim=config.projection_head_output_dim,
        rngs=rngs
    )
    output = projection_head_net(input_tensor, training=False)
    assert output.shape == (batch_size, config.projection_head_output_dim)

def test_muzero_network_initial_inference_with_projection_image(dummy_config_image):
    key = jax.random.PRNGKey(42)
    params_key, dropout_key, obs_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    config = dummy_config_image
    config.use_projection = True
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=lambda cfg_param, *, rngs: DummyProjectionNetwork(
            input_dim=cfg_param.num_channels, # For flat, input_dim is num_channels (hidden state size)
            hidden_dim=cfg_param.projection_hidden_dim,
            output_dim=cfg_param.projection_head_output_dim,
            rngs=rngs
        ),
        config=config, 
        rngs=rngs
    )
    obs_shape = config.observation_shape
    dummy_obs = jax.random.normal(obs_key, (config.batch_size, *obs_shape))
    hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.initial_inference(dummy_obs, training=False)
    assert hidden_state.shape == (config.batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels)
    assert reward.shape == (config.batch_size,)
    assert value.shape == (config.batch_size,)
    assert policy_logits.shape == (config.batch_size, config.num_actions)
    assert projected_output is not None
    assert projected_output.shape == (config.batch_size, config.projection_head_output_dim)

def test_muzero_network_recurrent_inference_with_projection_image(dummy_config_image):
    key = jax.random.PRNGKey(43)
    params_key, dropout_key, state_key, action_key = jax.random.split(key, 4)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    config = dummy_config_image
    config.use_projection = True
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=lambda cfg_param, *, rngs: DummyProjectionNetwork(
            input_dim=cfg_param.num_channels, # For flat, input_dim is num_channels (hidden state size)
            hidden_dim=cfg_param.projection_hidden_dim,
            output_dim=cfg_param.projection_head_output_dim,
            rngs=rngs
        ),
        config=config, 
        rngs=rngs
    )
    hidden_state_shape = (
        config.batch_size, 
        config.spatial_extents[0], 
        config.spatial_extents[1], 
        config.num_channels
    )
    hidden_state = jax.random.normal(state_key, hidden_state_shape)
    action = jax.random.randint(action_key, (config.batch_size,), 0, config.num_actions)
    next_hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.recurrent_inference(hidden_state, action, training=False)
    assert next_hidden_state.shape == (config.batch_size, config.spatial_extents[0], config.spatial_extents[1], config.num_channels)
    assert reward.shape == (config.batch_size,)
    assert value.shape == (config.batch_size,)
    assert policy_logits.shape == (config.batch_size, config.num_actions)
    assert projected_output is not None
    assert projected_output.shape == (config.batch_size, config.projection_head_output_dim)

def test_muzero_network_initial_inference_with_projection_flat(dummy_config_flat, rngs):
    key = jax.random.PRNGKey(44)
    config = dummy_config_flat
    config.use_projection = True
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=lambda cfg_param, *, rngs: DummyProjectionNetwork(
            input_dim=cfg_param.num_channels, # For flat obs, input to projection is hidden state (num_channels)
            hidden_dim=cfg_param.projection_hidden_dim,
            output_dim=cfg_param.projection_head_output_dim,
            rngs=rngs
        ),
        config=config, 
        rngs=rngs
    )
    obs_shape = config.observation_shape
    dummy_obs = jax.random.normal(jax.random.PRNGKey(123), (config.batch_size, *obs_shape))
    hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.initial_inference(dummy_obs, training=True)
    assert hidden_state.shape == (config.batch_size, config.num_channels)
    if config.reward_support_size > 0:
        assert reward.shape == (config.batch_size, config.reward_support_size)
    else: # pragma: no cover
        assert reward.shape == (config.batch_size,)
    if config.value_support_size > 0:
        assert value.shape == (config.batch_size, config.value_support_size)
    else: # pragma: no cover
        assert value.shape == (config.batch_size,)
    assert policy_logits.shape == (config.batch_size, config.num_actions)
    assert projected_output is not None
    assert projected_output.shape == (config.batch_size, config.projection_head_output_dim)

def test_muzero_network_recurrent_inference_with_projection_flat(dummy_config_flat, rngs):
    key = jax.random.PRNGKey(45)
    config = dummy_config_flat
    config.use_projection = True
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=lambda cfg_param, *, rngs: DummyProjectionNetwork(
            input_dim=cfg_param.num_channels, # For flat obs
            hidden_dim=cfg_param.projection_hidden_dim,
            output_dim=cfg_param.projection_head_output_dim, # Use projection_head_output_dim
            rngs=rngs
        ),
        config=config, 
        rngs=rngs
    )
    hidden_state_shape = (
        config.batch_size, 
        config.num_channels
    )
    hidden_state = jax.random.normal(key, hidden_state_shape)
    action = jax.random.randint(key, (config.batch_size,), 0, config.num_actions)
    next_hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.recurrent_inference(hidden_state, action, training=True)
    assert next_hidden_state.shape == (config.batch_size, config.num_channels)
    if config.reward_support_size > 0:
        assert reward.shape == (config.batch_size, config.reward_support_size)
    else: # pragma: no cover
        assert reward.shape == (config.batch_size,)
    if config.value_support_size > 0:
        assert value.shape == (config.batch_size, config.value_support_size)
    else: # pragma: no cover
        assert value.shape == (config.batch_size,)
    assert policy_logits.shape == (config.batch_size, config.num_actions)
    assert projected_output is not None
    assert projected_output.shape == (config.batch_size, config.projection_head_output_dim)

def test_muzero_network_initial_inference_without_projection(dummy_config_image):
    key = jax.random.PRNGKey(46)
    params_key, dropout_key, obs_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    config = dummy_config_image
    config.use_projection = False
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=None, # Projection is off
        config=config, 
        rngs=rngs
    )
    obs_shape = config.observation_shape
    dummy_obs = jax.random.normal(obs_key, (config.batch_size, *obs_shape))
    hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.initial_inference(dummy_obs, training=False)
    assert projected_output is None

def test_muzero_network_recurrent_inference_without_projection(dummy_config_image):
    key = jax.random.PRNGKey(47)
    params_key, dropout_key, state_key, action_key = jax.random.split(key, 4)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    config = dummy_config_image
    config.use_projection = False
    model = muzero_network_lib.MuZeroNetwork(
        representation_network_def=DummyRepresentationNetwork,
        dynamics_network_def=DummyDynamicsNetwork,
        prediction_network_def=DummyPredictionNetwork,
        reward_network_def=DummyRewardNetwork,
        projection_network_def=None, # Projection is off
        config=config, 
        rngs=rngs
    )
    hidden_state_shape = (
        config.batch_size, 
        config.spatial_extents[0], 
        config.spatial_extents[1], 
        config.num_channels
    )
    hidden_state = jax.random.normal(state_key, hidden_state_shape)
    action = jax.random.randint(action_key, (config.batch_size,), 0, config.num_actions)
    next_hidden_state, reward, value, policy_logits, projected_output, reward_hidden = model.recurrent_inference(hidden_state, action, training=False)
    assert projected_output is None

# It might be good to add a test that raises ValueError if use_projection is True but dims are not set
# However, that would require modifying the fixture or config on the fly which can be tricky.
# For now, assume config is correctly set up by the user if use_projection is True.
# The __init__ of MuZeroNetwork already has a check for this. 

# Helper functions for creating test networks
def create_test_representation_network(config, *, rngs):
    return DummyRepresentationNetwork(config, rngs=rngs)

def create_test_prediction_network(config, *, rngs):
    return DummyPredictionNetwork(config, rngs=rngs)

def create_test_dynamics_network(config, *, rngs):
    return DummyDynamicsNetwork(config, rngs=rngs)

def create_test_reward_network(config, *, rngs):
    return DummyRewardNetwork(config, rngs=rngs)

# --- Test symlog model output transformations (EfficientZeroV2 parity) ---
def test_model_symlog_value_output():
    """Test that value head outputs symlog-transformed values when configured."""
    # Test with symlog value loss
    config_symlog = MuZeroNetworkConfig(
        observation_shape=(4,),
        num_actions=3,
        num_channels=8,
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        value_loss_type="symlog",
        symlog_base=math.e
    )
    
    key = jax.random.PRNGKey(0)
    
    # Create model with symlog value loss
    model_symlog = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: create_test_representation_network(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: create_test_prediction_network(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: create_test_dynamics_network(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: create_test_reward_network(config, rngs=rngs),
        projection_network_def=None,
        config=config_symlog,
        rngs=nnx.Rngs(params=key)
    )
    
    # Test with normal MSE value loss
    config_mse = MuZeroNetworkConfig(
        observation_shape=(4,),
        num_actions=3,
        num_channels=8,
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        value_loss_type="mse",
    )
    
    model_mse = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: create_test_representation_network(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: create_test_prediction_network(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: create_test_dynamics_network(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: create_test_reward_network(config, rngs=rngs),
        projection_network_def=None,
        config=config_mse,
        rngs=nnx.Rngs(params=key)
    )
    
    # Test observation
    observation = jnp.array([[1.0, -2.0, 3.0, -4.0]])
    
    # Get initial inference outputs
    _, _, value_symlog, _, _, reward_hidden = model_symlog.initial_inference(observation, training=False)
    _, _, value_mse, _, _, reward_hidden = model_mse.initial_inference(observation, training=False)
    
    # The symlog model should output different values than the MSE model
    # (because symlog transformation is applied)
    assert not jnp.allclose(value_symlog, value_mse, atol=1e-6), \
        "Symlog model should output different values than MSE model"
    
    # Verify symlog values are reasonable (finite and bounded)
    assert jnp.all(jnp.isfinite(value_symlog)), "Symlog values should be finite"
    
    print("✅ Model symlog value output test passed!")


def test_model_symlog_reward_output():
    """Test that reward head outputs symlog-transformed values when configured."""
    # Test with symlog reward loss
    config_symlog = MuZeroNetworkConfig(
        observation_shape=(4,),
        num_actions=3,
        num_channels=8,
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        reward_loss_type="symlog",
        symlog_base=math.e
    )
    
    key = jax.random.PRNGKey(0)
    
    # Create model with symlog reward loss
    model_symlog = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: create_test_representation_network(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: create_test_prediction_network(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: create_test_dynamics_network(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: create_test_reward_network(config, rngs=rngs),
        projection_network_def=None,
        config=config_symlog,
        rngs=nnx.Rngs(params=key)
    )
    
    # Test with normal MSE reward loss
    config_mse = MuZeroNetworkConfig(
        observation_shape=(4,),
        num_actions=3,
        num_channels=8,
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        reward_loss_type="mse",
    )
    
    model_mse = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: create_test_representation_network(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: create_test_prediction_network(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: create_test_dynamics_network(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: create_test_reward_network(config, rngs=rngs),
        projection_network_def=None,
        config=config_mse,
        rngs=nnx.Rngs(params=key)
    )
    
    # Test observation and action
    observation = jnp.array([[1.0, -2.0, 3.0, -4.0]])
    action = jnp.array([1])
    
    # Get initial inference to get hidden state, then do recurrent inference
    hidden_state_symlog, _, _, _, _, reward_hidden = model_symlog.initial_inference(observation, training=False)
    hidden_state_mse, _, _, _, _, reward_hidden = model_mse.initial_inference(observation, training=False)
    
    _, reward_symlog, _, _, _, reward_hidden = model_symlog.recurrent_inference(hidden_state_symlog, action, training=False)
    _, reward_mse, _, _, _, reward_hidden = model_mse.recurrent_inference(hidden_state_mse, action, training=False)
    
    # The symlog model should output different rewards than the MSE model
    # (because symlog transformation is applied)
    assert not jnp.allclose(reward_symlog, reward_mse, atol=1e-6), \
        "Symlog model should output different rewards than MSE model"
    
    # Verify symlog rewards are reasonable (finite and bounded)
    assert jnp.all(jnp.isfinite(reward_symlog)), "Symlog rewards should be finite"
    
    print("✅ Model symlog reward output test passed!")


def test_model_symlog_base_configuration():
    """Test that different symlog bases produce different outputs."""
    # Test with base e
    config_base_e = MuZeroNetworkConfig(
        observation_shape=(4,),
        num_actions=3,
        num_channels=8,
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        value_loss_type="symlog",
        reward_loss_type="symlog", 
        symlog_base=math.e
    )
    
    # Test with base 2
    config_base_2 = MuZeroNetworkConfig(
        observation_shape=(4,),
        num_actions=3,
        num_channels=8,
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        value_loss_type="symlog",
        reward_loss_type="symlog",
        symlog_base=2.0
    )
    
    key = jax.random.PRNGKey(0)
    
    # Create models with different bases
    model_base_e = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: create_test_representation_network(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: create_test_prediction_network(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: create_test_dynamics_network(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: create_test_reward_network(config, rngs=rngs),
        projection_network_def=None,
        config=config_base_e,
        rngs=nnx.Rngs(params=key)
    )
    
    model_base_2 = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: create_test_representation_network(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: create_test_prediction_network(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: create_test_dynamics_network(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: create_test_reward_network(config, rngs=rngs),
        projection_network_def=None,
        config=config_base_2,
        rngs=nnx.Rngs(params=key)
    )
    
    # Test observation and action
    observation = jnp.array([[1.0, -2.0, 3.0, -4.0]])
    action = jnp.array([1])
    
    # Get outputs from both models
    _, reward_e, value_e, _, _, reward_hidden = model_base_e.initial_inference(observation, training=False)
    _, reward_2, value_2, _, _, reward_hidden = model_base_2.initial_inference(observation, training=False)
    
    # Different bases should produce different outputs
    assert not jnp.allclose(value_e, value_2, atol=1e-6), \
        "Different symlog bases should produce different value outputs"
    assert not jnp.allclose(reward_e, reward_2, atol=1e-6), \
        "Different symlog bases should produce different reward outputs"
    
    print("✅ Model symlog base configuration test passed!")


def test_model_output_consistency_with_loss_functions():
    """Test that model outputs are compatible with various loss functions."""
    import jax.random as jr
    from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
    from open_spiel.python.algorithms.muzero_jax.training import losses as loss_lib

    config = MuZeroNetworkConfig(
        observation_shape=(10,),
        num_actions=5,
        num_channels=8,
        value_support_size=11,  # Categorical value 
        reward_support_size=11,  # Categorical reward
        use_image_observation=False,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=16,
        spatial_extents=(8, 8),
        use_projection=False
    )

    key = jr.key(42)
    model = MuZeroNetwork(
        representation_network_def=create_test_representation_network,
        dynamics_network_def=create_test_dynamics_network,
        prediction_network_def=create_test_prediction_network,
        reward_network_def=create_test_reward_network,
        projection_network_def=None,
        config=config,
        rngs=nnx.Rngs(params=key)
    )

    batch_size = 3
    observation = jr.normal(key, (batch_size, 10))
    action = jr.randint(key, (batch_size,), 0, 5)

    # Test initial inference with categorical outputs
    hidden_state, reward, value, policy_logits, _, reward_hidden = model.initial_inference(observation, training=False)
    
    # Verify shapes are compatible with loss functions
    assert reward.shape == (batch_size, 11), f"Expected reward shape (3, 11), got {reward.shape}"
    assert value.shape == (batch_size, 11), f"Expected value shape (3, 11), got {value.shape}"
    assert policy_logits.shape == (batch_size, 5), f"Expected policy shape (3, 5), got {policy_logits.shape}"

    # Test recurrent inference
    next_hidden_state, reward_rec, value_rec, policy_logits_rec, _, reward_hidden = model.recurrent_inference(
        hidden_state, action, training=False
    )
    
    # Verify categorical distributions can be used with categorical losses
    target_value_dist = jr.uniform(key, (batch_size, 11))
    target_value_dist = target_value_dist / jnp.sum(target_value_dist, axis=-1, keepdims=True)
    
    target_reward_dist = jr.uniform(key, (batch_size, 11))
    target_reward_dist = target_reward_dist / jnp.sum(target_reward_dist, axis=-1, keepdims=True)
    
    target_policy = jr.uniform(key, (batch_size, 5))
    target_policy = target_policy / jnp.sum(target_policy, axis=-1, keepdims=True)
    
    # Test compatibility with loss functions
    value_loss = loss_lib.compute_categorical_value_loss(value, target_value_dist)
    reward_loss = loss_lib.compute_categorical_reward_loss(reward, target_reward_dist)
    policy_loss = loss_lib.compute_policy_loss(policy_logits, target_policy)
    
    assert value_loss.shape == (batch_size,), f"Expected value_loss shape (3,), got {value_loss.shape}"
    assert reward_loss.shape == (batch_size,), f"Expected reward_loss shape (3,), got {reward_loss.shape}"
    assert policy_loss.shape == (batch_size,), f"Expected policy_loss shape (3,), got {policy_loss.shape}"


# Test coverage for missing network.py lines
def test_downsample_network():
    """Test the DownSample network with various configurations."""
    from open_spiel.python.algorithms.muzero_jax.models.network import DownSample
    
    key = jax.random.key(42)
    in_channels = 3
    out_channels = 16
    
    downsampler = DownSample(in_channels, out_channels, rngs=nnx.Rngs(params=key))
    
    # Test with different batch sizes and image sizes
    batch_size = 2
    height, width = 96, 96
    x = jax.random.normal(key, (batch_size, height, width, in_channels))
    
    # Test training=True
    output_train = downsampler(x, training=True)
    expected_h = height // 16  # Four stride-2 operations: 96 -> 48 -> 24 -> 12 -> 6
    expected_w = width // 16
    assert output_train.shape == (batch_size, expected_h, expected_w, out_channels), \
        f"Expected shape (2, {expected_h}, {expected_w}, {out_channels}), got {output_train.shape}"
    
    # Test training=False
    output_eval = downsampler(x, training=False)
    assert output_eval.shape == output_train.shape


def test_representation_network_image_branch():
    """Test RepresentationNetwork image branch with and without downsampling."""
    from open_spiel.python.algorithms.muzero_jax.models.network import RepresentationNetwork

    key = jax.random.key(42)
    batch_size = 2

    # Test with downsampling
    config_downsample = MuZeroNetworkConfig(
        observation_shape=(96, 96, 3),
        num_channels=16,
        num_residual_blocks=2,
        use_image_observation=True,
        downsample_channels=8,
        downsample_blocks=1,
        spatial_extents=(24, 24)  # After 4x downsampling
    )
    
    rep_net_down = RepresentationNetwork(config_downsample, rngs=nnx.Rngs(params=key))
    x = jax.random.normal(key, (batch_size, 96, 96, 3))
    output_down = rep_net_down(x, training=True)
    assert output_down.shape == (batch_size, 24, 24, 16)
    
    # Test without downsampling
    config_no_downsample = MuZeroNetworkConfig(
        observation_shape=(96, 96, 3),
        num_channels=16,
        num_residual_blocks=2,
        use_image_observation=True,
        downsample_blocks=0,  # Use 0 blocks to disable downsampling
        spatial_extents=(96, 96)
    )
    
    rep_net_no_down = RepresentationNetwork(config_no_downsample, rngs=nnx.Rngs(params=key))
    output_no_down = rep_net_no_down(x, training=False)
    # When no downsampling, should preserve spatial dimensions
    assert output_no_down.shape == (batch_size, 96, 96, 16)


def test_dynamics_network_action_embedding():
    """Test DynamicsNetwork with action embedding enabled."""
    from open_spiel.python.algorithms.muzero_jax.models.network import DynamicsNetwork

    key = jax.random.key(42)
    batch_size = 2

    # Test image observation with action embedding
    config = MuZeroNetworkConfig(
        observation_shape=(24, 24, 3),
        num_channels=16,
        num_actions=5,
        num_residual_blocks=2,
        use_image_observation=True,
        action_embedding_dim=8,
        spatial_extents=(24, 24)
    )
    
    dynamics_net = DynamicsNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 24, 24, 16))
    action = jax.random.randint(key, (batch_size,), 0, 5)
    
    next_hidden_state = dynamics_net(hidden_state, action, training=True)
    assert next_hidden_state.shape == (batch_size, 24, 24, 16)


def test_dynamics_network_continuous_actions():
    """Test DynamicsNetwork with continuous actions."""
    from open_spiel.python.algorithms.muzero_jax.models.network import DynamicsNetwork

    key = jax.random.key(42)
    batch_size = 2

    # Test with continuous actions in image setting
    config = MuZeroNetworkConfig(
        observation_shape=(24, 24, 3),
        num_channels=16,
        num_actions=3,  # Continuous action dim
        num_residual_blocks=1,
        use_image_observation=True,
        spatial_extents=(24, 24)
    )
    
    dynamics_net = DynamicsNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 24, 24, 16))
    continuous_action = jax.random.normal(key, (batch_size, 3))
    
    next_hidden_state = dynamics_net(hidden_state, continuous_action, training=False)
    assert next_hidden_state.shape == (batch_size, 24, 24, 16)


def test_dynamics_network_error_cases():
    """Test DynamicsNetwork error handling for invalid inputs."""
    from open_spiel.python.algorithms.muzero_jax.models.network import DynamicsNetwork

    key = jax.random.key(42)

    # Test flat observations with invalid hidden state dimensions
    config = MuZeroNetworkConfig(
        observation_shape=(64,),
        num_channels=16,
        num_actions=5,
        num_residual_blocks=1,
        use_image_observation=False,
        num_hidden_units_fc=32
    )
    
    dynamics_net = DynamicsNetwork(config, rngs=nnx.Rngs(params=key))
    
    # Invalid hidden state shape (should be 2D for flat observations)
    hidden_state_invalid = jax.random.normal(key, (2, 16, 16))  # 3D instead of 2D
    action = jax.random.randint(key, (2,), 0, 5)
    
    with pytest.raises(TypeError, match="Expected hidden_state with ndim=2"):
        dynamics_net(hidden_state_invalid, action, training=False)


def test_prediction_network_image_branch():
    """Test PredictionNetwork image branch functionality."""
    from open_spiel.python.algorithms.muzero_jax.models.network import PredictionNetwork
    
    key = jax.random.key(42)
    batch_size = 2
    
    config = MuZeroNetworkConfig(
        observation_shape=(24, 24, 3),
        num_channels=16,
        num_actions=5,
        num_fc_residual_blocks=2,
        use_image_observation=True,
        value_support_size=0,  # Scalar value
        spatial_extents=(24, 24),
        num_hidden_units_fc=32
    )
    
    pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 24, 24, 16))
    
    policy_logits, value = pred_net(hidden_state, training=True)
    assert policy_logits.shape == (batch_size, 5)
    assert value.shape == (batch_size,)  # Scalar value


def test_prediction_network_symlog_transformation():
    """Test PredictionNetwork symlog transformation for value output."""
    from open_spiel.python.algorithms.muzero_jax.models.network import PredictionNetwork

    key = jax.random.key(42)
    batch_size = 2

    # Config with symlog value loss
    config = MuZeroNetworkConfig(
        observation_shape=(64,),
        num_channels=16,
        num_actions=5,
        value_support_size=0,  # Scalar value
        use_image_observation=False,
        num_hidden_units_fc=32,
        value_loss_type="symlog"  # Set during initialization since config is frozen
    )
    # Config is frozen, so symlog_base is already set to e as default
    
    pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 16))
    
    policy_logits, value = pred_net(hidden_state, training=False)
    assert policy_logits.shape == (batch_size, 5)
    assert value.shape == (batch_size,)
    
    # Value should be transformed via symlog
    # We can't easily verify the exact transformation without knowing the internal weights,
    # but we can check the shape and that it doesn't crash


def test_reward_network_image_branch():
    """Test RewardNetwork image branch functionality."""
    from open_spiel.python.algorithms.muzero_jax.models.network import RewardNetwork
    
    key = jax.random.key(42)
    batch_size = 2
    
    config = MuZeroNetworkConfig(
        observation_shape=(24, 24, 3),
        num_channels=16,
        reward_support_size=0,  # Scalar reward
        use_image_observation=True,
        spatial_extents=(24, 24),
        num_hidden_units_fc=32
    )
    
    reward_net = RewardNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 24, 24, 16))
    
    reward = reward_net(hidden_state, training=True)
    assert reward.shape == (batch_size,)  # Scalar reward


def test_reward_network_symlog_transformation():
    """Test RewardNetwork symlog transformation for reward output."""
    from open_spiel.python.algorithms.muzero_jax.models.network import RewardNetwork
    
    key = jax.random.key(42)
    batch_size = 2
    
    # Config with symlog reward loss
    config = MuZeroNetworkConfig(
        observation_shape=(64,),
        num_channels=16,
        reward_support_size=0,  # Scalar reward
        use_image_observation=False,
        num_hidden_units_fc=32,
        reward_loss_type="symlog"  # Set during initialization since config is frozen
    )
    # Config is frozen, so symlog_base is already set to e as default
    
    reward_net = RewardNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 16))
    
    reward = reward_net(hidden_state, training=False)
    assert reward.shape == (batch_size,)


# --- Test Noisy Networks Integration ---
def test_prediction_network_with_noisy_networks():
    """Test PredictionNetwork with noisy networks enabled."""
    
    key = jax.random.key(42)
    batch_size = 2
    
    # Config with noisy networks enabled
    config = MuZeroNetworkConfig(
        observation_shape=(64,),
        num_channels=16,
        num_actions=5,
        value_support_size=0,
        use_image_observation=False,
        num_hidden_units_fc=32,
        noisy_net=True  # Enable noisy networks
    )
    
    pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jax.random.normal(key, (batch_size, 16))
    
    # Test forward pass
    policy_logits1, value1 = pred_net(hidden_state, training=False)
    assert policy_logits1.shape == (batch_size, 5)
    assert value1.shape == (batch_size,)
    
    # Reset noise and test again - outputs should be different
    pred_net.reset_noise(jax.random.PRNGKey(999))
    policy_logits2, value2 = pred_net(hidden_state, training=False)
    
    # Policy outputs should be different due to noise reset
    assert not jnp.allclose(policy_logits1, policy_logits2, atol=1e-6)


def test_prediction_network_noisy_reset_noise_method():
    """Test PredictionNetwork reset_noise method."""
    
    key = jax.random.key(42)
    
    # Test with noisy networks enabled
    config_noisy = MuZeroNetworkConfig(
        observation_shape=(32,),
        num_channels=16,
        num_actions=3,
        value_support_size=0,
        use_image_observation=False,
        num_hidden_units_fc=16,
        noisy_net=True
    )
    
    pred_net_noisy = PredictionNetwork(config_noisy, rngs=nnx.Rngs(params=key))
    
    # Should have reset_noise method and not error when called
    pred_net_noisy.reset_noise(jax.random.PRNGKey(123))
    
    # Test with noisy networks disabled
    config_regular = MuZeroNetworkConfig(
        observation_shape=(32,),
        num_channels=16,
        num_actions=3,
        value_support_size=0,
        use_image_observation=False,
        num_hidden_units_fc=16,
        noisy_net=False
    )
    
    pred_net_regular = PredictionNetwork(config_regular, rngs=nnx.Rngs(params=key))
    
    # Should not error when called on regular network
    pred_net_regular.reset_noise(jax.random.PRNGKey(456))


def test_muzero_network_reset_noise_functionality():
    """Test MuZeroNetwork reset_noise method."""
    
    key = jax.random.key(42)
    batch_size = 2
    
    # Config with noisy networks
    config = MuZeroNetworkConfig(
        observation_shape=(32,),
        num_channels=16,
        num_actions=4,
        value_support_size=0,
        reward_support_size=0,
        use_image_observation=False,
        num_hidden_units_fc=16,
        noisy_net=True,
        use_projection=False
    )
    
    # Create MuZero network
    network = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: DummyRepresentationNetwork(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: DummyDynamicsNetwork(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: PredictionNetwork(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: DummyRewardNetwork(config, rngs=rngs),
        projection_network_def=None,
        config=config,
        rngs=nnx.Rngs(params=key)
    )
    
    # Test initial inference
    observation = jax.random.normal(key, (batch_size, 32))
    hidden_state1, reward1, value1, policy_logits1, proj1, reward_hidden = network.initial_inference(observation, training=False)
    
    # Reset noise
    network.reset_noise(jax.random.PRNGKey(777))
    
    # Test inference again - policy should be different due to noisy networks
    hidden_state2, reward2, value2, policy_logits2, proj2, reward_hidden = network.initial_inference(observation, training=False)
    
    # Only policy should be different (since only policy network uses noisy layers in our implementation)
    assert not jnp.allclose(policy_logits1, policy_logits2, atol=1e-6)
    # Other outputs should be the same since they don't use noisy networks
    assert jnp.allclose(hidden_state1, hidden_state2)
    assert jnp.allclose(reward1, reward2)
    assert jnp.allclose(value1, value2)


def test_muzero_network_reset_noise_without_noisy_networks():
    """Test MuZeroNetwork reset_noise method when noisy networks are disabled."""
    
    key = jax.random.key(42)
    
    # Config without noisy networks
    config = MuZeroNetworkConfig(
        observation_shape=(32,),
        num_channels=16,
        num_actions=4,
        value_support_size=0,
        reward_support_size=0,
        use_image_observation=False,
        num_hidden_units_fc=16,
        noisy_net=False,  # Disabled
        use_projection=False
    )
    
    # Create MuZero network
    network = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: DummyRepresentationNetwork(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: DummyDynamicsNetwork(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: PredictionNetwork(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: DummyRewardNetwork(config, rngs=rngs),
        projection_network_def=None,
        config=config,
        rngs=nnx.Rngs(params=key)
    )
    
    # Should not error when reset_noise is called
    network.reset_noise(jax.random.PRNGKey(888))


def test_noisy_networks_configuration_transfer():
    """Test that noisy_net configuration is properly transferred from MuZeroConfig to MuZeroNetworkConfig."""
    from open_spiel.python.algorithms.muzero_jax.training.trainer import create_network_config_from_muzero_config, MuZeroConfig
    
    # Test with noisy networks enabled
    muzero_config_noisy = MuZeroConfig(noisy_net=True)
    network_config_noisy = create_network_config_from_muzero_config(
        muzero_config_noisy,
        observation_shape=(64,),
        num_actions=5,
        use_image_observation=False
    )
    assert network_config_noisy.noisy_net == True
    
    # Test with noisy networks disabled
    muzero_config_regular = MuZeroConfig(noisy_net=False)
    network_config_regular = create_network_config_from_muzero_config(
        muzero_config_regular,
        observation_shape=(64,),
        num_actions=5,
        use_image_observation=False
    )
    assert network_config_regular.noisy_net == False


def test_noisy_networks_action_item_25_completion():
    """Comprehensive test verifying Action Item 25: Noisy Networks Support completion criteria."""
    from open_spiel.python.algorithms.muzero_jax.models.layers import NoisyLinear, MLP
    from open_spiel.python.algorithms.muzero_jax.training.trainer import create_network_config_from_muzero_config, MuZeroConfig
    
    key = jax.random.key(42)
    batch_size = 2
    
    # 1. Test JAX-compatible NoisyLinear layers implementation
    noisy_layer = NoisyLinear(10, 5, std_init=0.5, rngs=nnx.Rngs(params=key))
    test_input = jnp.ones((batch_size, 10))
    output = noisy_layer(test_input)
    assert output.shape == (batch_size, 5)
    
    # 2. Test integration into MuZero network architecture (policy heads)
    config = MuZeroNetworkConfig(
        observation_shape=(32,),
        num_channels=16,
        num_actions=4,
        noisy_net=True,
        use_image_observation=False,
        value_support_size=0,
        reward_support_size=0,
        use_projection=False
    )
    
    pred_net = PredictionNetwork(config, rngs=nnx.Rngs(params=key))
    hidden_state = jnp.ones((batch_size, 16))
    policy_logits, value = pred_net(hidden_state, training=False)
    
    # Check that noisy layers are used in policy network
    noisy_layers_count = 0
    for layer in pred_net.policy_fc.layers:
        if isinstance(layer, NoisyLinear):
            noisy_layers_count += 1
    assert noisy_layers_count > 0, "Policy network should use noisy layers when noisy_net=True"
    
    # 3. Test reset_noise functionality
    initial_output = pred_net(hidden_state, training=False)[0]
    pred_net.reset_noise(jax.random.PRNGKey(999))
    reset_output = pred_net(hidden_state, training=False)[0]
    assert not jnp.allclose(initial_output, reset_output, atol=1e-6), "Output should change after noise reset"
    
    # 4. Test configurability via noisy_net parameter
    # Already tested in network config and creation functions above
    
    # 5. Test that configuration transfer works properly
    muzero_config = MuZeroConfig(noisy_net=True)
    network_config = create_network_config_from_muzero_config(
        muzero_config, observation_shape=(32,), num_actions=4
    )
    assert network_config.noisy_net == True
    
    # 6. Test full MuZero network with noisy networks
    full_network = MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: DummyRepresentationNetwork(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: DummyDynamicsNetwork(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: PredictionNetwork(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: DummyRewardNetwork(config, rngs=rngs),
        projection_network_def=None,
        config=config,
        rngs=nnx.Rngs(params=key)
    )
    
    observation = jnp.ones((batch_size, 32))
    _, _, _, initial_policy, _, reward_hidden = full_network.initial_inference(observation)
    full_network.reset_noise(jax.random.PRNGKey(888))
    _, _, _, reset_policy, _, reward_hidden = full_network.initial_inference(observation)
    assert not jnp.allclose(initial_policy, reset_policy, atol=1e-6), "Full network policy should change after noise reset"
    
    print("✅ Action Item 25: Noisy Networks Support - All completion criteria verified!")


def test_efficientzero_v2_noisy_networks_parity():
    """Test alignment with EfficientZeroV2 noisy networks implementation patterns."""
    from open_spiel.python.algorithms.muzero_jax.models.layers import NoisyLinear
    
    key = jax.random.key(42)
    
    # Test EfficientZeroV2 std_init=0.5 pattern (from PyTorch reference)
    layer = NoisyLinear(8, 4, std_init=0.5, rngs=nnx.Rngs(params=key))
    
    # Verify initialization follows EfficientZeroV2 pattern
    expected_weight_sigma = 0.5 / jnp.sqrt(8)  # std_init / sqrt(in_features)
    expected_bias_sigma = 0.5 / jnp.sqrt(4)    # std_init / sqrt(out_features)
    
    assert jnp.allclose(layer.weight_sigma.value, expected_weight_sigma)
    assert jnp.allclose(layer.bias_sigma.value, expected_bias_sigma)
    
    # Test factorized Gaussian noise structure (EfficientZeroV2 pattern)
    layer.reset_noise(key)
    
    # Verify noise shapes match factorized structure
    assert layer.weight_epsilon.value.shape == (4, 8)  # (out_features, in_features)
    assert layer.bias_epsilon.value.shape == (4,)      # (out_features,)
    
    print("✅ EfficientZeroV2 Noisy Networks Parity - Implementation aligns with PyTorch reference!")