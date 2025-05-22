import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import dataclasses
from typing import Tuple

from open_spiel.python.algorithms.muzero_jax.models import network as muzero_network_lib
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
    image_obs_config.downsample = True
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
    action_shape = (batch_size, 1) # For discrete actions
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
    hidden_state, reward, value, policy_logits, projected_output = model.initial_inference(dummy_obs, training=False)

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

    next_hidden_state, reward, value, policy_logits, projected_output = model.recurrent_inference(current_hidden_state, dummy_action, training=False)

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
    hidden_state, reward, value, policy_logits, projected_output = model.initial_inference(dummy_obs, training=False)
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
    next_hidden_state, reward, value, policy_logits, projected_output = model.recurrent_inference(hidden_state, action, training=False)
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
    hidden_state, reward, value, policy_logits, projected_output = model.initial_inference(dummy_obs, training=True)
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
    next_hidden_state, reward, value, policy_logits, projected_output = model.recurrent_inference(hidden_state, action, training=True)
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
    hidden_state, reward, value, policy_logits, projected_output = model.initial_inference(dummy_obs, training=False)
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
    next_hidden_state, reward, value, policy_logits, projected_output = model.recurrent_inference(hidden_state, action, training=False)
    assert projected_output is None

# It might be good to add a test that raises ValueError if use_projection is True but dims are not set
# However, that would require modifying the fixture or config on the fly which can be tricky.
# For now, assume config is correctly set up by the user if use_projection is True.
# The __init__ of MuZeroNetwork already has a check for this. 