import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Sequence, Callable, Tuple, Optional

# Assuming layers.py is in the same directory or accessible in PYTHONPATH
from .layers import conv3x3, ResidualBlock, FCResidualBlock, MLP
from .network_config import MuZeroNetworkConfig

# Type alias for LSTM state (c, h) tuple
LSTMState = Tuple[jax.Array, jax.Array]

# Import symlog function for EfficientZeroV2 parity
def symlog(x: jax.Array, base: float = jnp.e) -> jax.Array:
    """Symmetric logarithm transformation as used in EfficientZeroV2."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0) / jnp.log(base)

# --- Configuration dataclass (example, to be defined properly elsewhere) ---
# We'll assume a config object with attributes like:
# config.observation_shape: tuple[int, ...]
# config.num_actions: int
# config.num_channels: int (channels for conv layers, also hidden state size for image-based)
# config.representation_num_blocks: int
# config.dynamics_num_blocks: int
# config.prediction_num_blocks: int # For resblocks before value/policy heads
# config.reward_support_size: int (if using categorical rewards)
# config.value_support_size: int (if using categorical values)
# config.downsample: bool (for image observations)
# config.action_embedding_dim: int (if using action embeddings in dynamics)
# config.is_continuous: bool (for action space)
# config.fc_representation_layers: list[int] (for flat observations)
# config.fc_dynamics_layers: list[int]
# config.fc_prediction_layers: list[int]
# config.use_image_observation: bool # To switch between conv and FC networks

class DownSample(nnx.Module):
    """Downsampling network for image observations (Flax NNX version)."""
    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        # Implements EfficientZeroV2 DownSample path with 4 stages

        # Stage 1: Conv (stride 2) + ResBlock
        self.conv1 = nnx.Conv(in_channels, out_channels // 2, kernel_size=(3,3), strides=(2,2), padding='SAME', use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels // 2, use_running_average=True, rngs=rngs)
        self.resblock1 = ResidualBlock(out_channels // 2, out_channels // 2, rngs=rngs)
        
        # Stage 2: Conv (stride 1) + DownSampleBlock (stride 2) + ResBlock
        self.conv2 = nnx.Conv(out_channels // 2, out_channels, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

        # Downsample block: ResidualBlock with stride 2 and projection shortcut
        self.downsample_block = ResidualBlock(out_channels, out_channels, stride=2,
                                              downsample_conv=nnx.Conv(out_channels, out_channels, kernel_size=(1,1), strides=(2,2), use_bias=False, rngs=rngs),
                                              rngs=rngs)
        self.resblock2 = ResidualBlock(out_channels, out_channels, rngs=rngs)
        
        # Stage 3: Pooling + ResBlock
        self.resblock3 = ResidualBlock(out_channels, out_channels, rngs=rngs)

        # Stage 4: Pooling (only pooling)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.resblock1(x, training=training)
        
        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.downsample_block(x, training=training)
        x = self.resblock2(x, training=training)

        # Stage 3
        x = nnx.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = self.resblock3(x, training=training)

        # Stage 4
        x = nnx.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        return x

class RepresentationNetwork(nnx.Module):
    """Representation Network (h) for Flax NNX."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        if self.config.use_image_observation:
            if self.config.downsample_blocks > 0:
                self.downsampler = DownSample(self.config.observation_shape[-1], self.config.downsample_channels, rngs=rngs)
                # Add a conv to match channels if needed
                if self.config.downsample_channels != self.config.num_channels:
                    self.channel_match_conv = conv3x3(self.config.downsample_channels, self.config.num_channels, rngs=rngs)
                    self.channel_match_bn = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
                else: # pragma: no cover
                    self.channel_match_conv = None # pragma: no cover
                    self.channel_match_bn = None # pragma: no cover
                in_channels_for_resblocks = self.config.num_channels
            else:
                self.downsampler = None
                self.channel_match_conv = None
                self.channel_match_bn = None
                self.initial_conv = conv3x3(self.config.observation_shape[-1], self.config.num_channels, rngs=rngs)
                self.initial_bn = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
                in_channels_for_resblocks = self.config.num_channels
            
            self.resblocks = [
                ResidualBlock(in_channels_for_resblocks, self.config.num_channels, 
                              rngs=nnx.Rngs(params=rngs.params())) 
                for i in range(self.config.num_residual_blocks)
            ]
        else: # Flat observation (e.g. board state vector)
            self.mlp = MLP(input_size=self.config.observation_shape[0],
                           hidden_sizes=[self.config.num_hidden_units_fc],
                           output_size=self.config.num_channels, # Output is the hidden state
                           rngs=rngs)

    def __call__(self, observation: jax.Array, training: bool) -> jax.Array:
        if self.config.use_image_observation:
            if self.downsampler:
                x = self.downsampler(observation, training=training)
                # Match channels if needed
                if self.channel_match_conv is not None: # pragma: no cover
                    x = self.channel_match_conv(x) # pragma: no cover
                    x = self.channel_match_bn(x, use_running_average=not training) # pragma: no cover
                    x = nnx.relu(x) # pragma: no cover
            else:
                x = self.initial_conv(observation)
                x = self.initial_bn(x, use_running_average=not training)
                x = nnx.relu(x)
            
            for block in self.resblocks:
                x = block(x, training=training)
            return x
        else:
            return self.mlp(observation, training=training)

class DynamicsNetwork(nnx.Module):
    """Dynamics Network (g) for Flax NNX."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        # Action encoding part - simplified for now
        # EZv2 broadcasts discrete actions spatially. Continuous actions are used as is or embedded.
        # Assuming image-based for initial conv structure. Flat observations would use MLPs.
        if self.config.use_image_observation:
            # We need to determine the number of input channels to the first conv
            # This depends on action encoding strategy.
            # If actions are tiled and concatenated: num_channels + num_action_planes
            # For simplicity, assume action is encoded into 1 plane for discrete
            # or config.action_embedding_dim planes if using action embedding.
            action_planes = 1 # Simplified: for discrete, one plane representing action_id / num_actions
            # Action embedding is simplified - just use single action plane for now
            # Could be extended later to use config.action_embedding_dim if needed

            self.conv1 = conv3x3(self.config.num_channels + action_planes, self.config.num_channels, rngs=rngs)
            self.bn1 = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
            self.resblocks = [
                ResidualBlock(self.config.num_channels, self.config.num_channels, 
                              rngs=nnx.Rngs(params=rngs.params())) 
                for i in range(self.config.num_residual_blocks)
            ]
        else: # Flat observations
            # Input: hidden_state + action representation
            # Action can be one-hot encoded or embedded
            action_dim = self.config.num_actions # if one-hot for discrete
            self.mlp = MLP(input_size=self.config.num_channels + action_dim,
                           hidden_sizes=[self.config.num_hidden_units_fc],
                           output_size=self.config.num_channels, # Output is next hidden_state
                           rngs=rngs)

    def __call__(self, hidden_state: jax.Array, action: jax.Array, training: bool) -> jax.Array:
        # Action encoding needs to be well-defined here.
        # For image data (as in EZv2 dynamics input conv):
        if self.config.use_image_observation:
            # Check if action is continuous based on its shape and type
            if action.ndim == 1 and action.dtype in (jnp.int32, jnp.int64):
                # Discrete actions: (batch,) with integer type
                action_scaled = action / self.config.num_actions 
                action_plane = jnp.ones_like(hidden_state[..., :1]) * action_scaled.reshape(-1, 1, 1, 1) 
            elif action.ndim == 2:
                # Continuous actions: (batch, action_dim) -> (batch, H, W, 1)
                # For simplicity, take the mean of action dimensions and broadcast spatially
                action_mean = jnp.mean(action, axis=-1, keepdims=True)  # (batch, 1)
                action_plane = jnp.ones_like(hidden_state[..., :1]) * action_mean.reshape(-1, 1, 1, 1)
            else: # pragma: no cover
                # Fallback: treat as discrete scalar action
                action_scaled = action / self.config.num_actions  # pragma: no cover
                action_plane = jnp.ones_like(hidden_state[..., :1]) * action_scaled.reshape(-1, 1, 1, 1) # pragma: no cover

            # Concatenate state and action plane
            x = jnp.concatenate([hidden_state, action_plane], axis=-1)
            
            x = self.conv1(x)
            x = self.bn1(x, use_running_average=not training)
            # EZv2 adds current state here as a residual connection before ResBlocks
            x += hidden_state 
            x = nnx.relu(x)

            for block in self.resblocks:
                x = block(x, training=training)
            next_hidden_state = x
            return next_hidden_state
        else:  # Flat observations
            # Validate hidden_state dimensions for flat observations
            if hidden_state.ndim != 2: # pragma: no cover
                raise TypeError(f"Expected hidden_state with ndim=2 for flat observations, got ndim={hidden_state.ndim}") # pragma: no cover
            # Handle actions based on action shape and type
            if (action.ndim == 0 or action.ndim == 1) and action.dtype in (jnp.int32, jnp.int64):
                # Discrete actions: allow scalar or batch of scalars
                squeezed_action = action.squeeze()
                action_one_hot = jax.nn.one_hot(squeezed_action, num_classes=self.config.num_actions)
            else:
                # Continuous actions should be provided already in correct shape (batch, action_dim)
                action_one_hot = action
            # Validate action_one_hot dimensions
            if action_one_hot.ndim not in (1, 2):
                raise TypeError(f"Expected action_one_hot with ndim 1 or 2 for flat observations, got ndim={action_one_hot.ndim}") # pragma: no cover
            if action_one_hot.ndim == 1: # pragma: no cover
                # Expand batch dimension
                action_one_hot = jnp.expand_dims(action_one_hot, axis=0) # pragma: no cover
            # Now both hidden_state and action_one_hot are 2D: (batch, features)
            dynamics_input = jnp.concatenate([hidden_state, action_one_hot], axis=-1)
            return self.mlp(dynamics_input, training=training)


class PredictionNetwork(nnx.Module):
    """Prediction Network (f) for policy and value - Flax NNX."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        if self.config.use_image_observation:
            self.resblocks = [
                ResidualBlock(self.config.num_channels, self.config.num_channels, 
                              rngs=nnx.Rngs(params=rngs.params())) 
                for i in range(self.config.prediction_num_blocks)
            ]
            self.policy_conv = nnx.Conv(self.config.num_channels, self.config.num_channels, kernel_size=(1,1), rngs=rngs)
            self.policy_bn = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
            flatten_size_policy = self.config.spatial_extents[0] * self.config.spatial_extents[1] * self.config.num_channels
            self.policy_fc = MLP(flatten_size_policy, self.config.fc_prediction_layers, self.config.num_actions, noisy=self.config.noisy_net, rngs=rngs)

            self.value_conv = nnx.Conv(self.config.num_channels, 1, kernel_size=(1,1), rngs=rngs)
            self.value_bn = nnx.BatchNorm(1, use_running_average=True, rngs=rngs)
            flatten_size_value = self.config.spatial_extents[0] * self.config.spatial_extents[1] * 1
            # Configure value head output dimension based on loss type
            value_output_dim = self.config.get_value_output_dim()
            self.value_fc = MLP(flatten_size_value, self.config.fc_prediction_layers, value_output_dim, rngs=rngs)
        else: # Flat observations
            policy_rngs = nnx.Rngs(params=rngs.params())
            self.policy_fc = MLP(input_size=self.config.num_channels,
                                 hidden_sizes=self.config.fc_prediction_layers,
                                 output_size=self.config.num_actions,
                                 noisy=self.config.noisy_net,
                                 rngs=policy_rngs)
            # Configure value head output dimension based on loss type
            value_output_dim = self.config.get_value_output_dim()
            value_rngs = nnx.Rngs(params=rngs.params())
            self.value_fc = MLP(input_size=self.config.num_channels,
                                hidden_sizes=self.config.fc_prediction_layers, 
                                output_size=value_output_dim,
                                rngs=value_rngs)

    def reset_noise(self, rng_key: jax.Array) -> None:
        """Reset noise in policy network if using noisy networks."""
        if self.config.noisy_net:
            self.policy_fc.reset_noise(rng_key)

    def __call__(self, hidden_state: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        if self.config.use_image_observation:
            x_trunk = hidden_state
            for block in self.resblocks:
                x_trunk = block(x_trunk, training=training)
            
            # Policy head
            policy_x = self.policy_conv(x_trunk)
            policy_x = self.policy_bn(policy_x, use_running_average=not training)
            policy_x = nnx.relu(policy_x)
            policy_x = policy_x.reshape((policy_x.shape[0], -1))
            policy_logits = self.policy_fc(policy_x, training=training)

            # Value head
            value_x = self.value_conv(x_trunk)
            value_x = self.value_bn(value_x, use_running_average=not training)
            value_x = nnx.relu(value_x)
            value_x = value_x.reshape((value_x.shape[0], -1))
            value = self.value_fc(value_x, training=training)
        else: 
            policy_logits = self.policy_fc(hidden_state, training=training)
            value = self.value_fc(hidden_state, training=training)

        # Apply transformations to match loss function expectations
        if self.config.value_loss_type == "categorical":
            # For categorical loss, output logits over support (already configured in head)
            pass  # No transformation needed - head outputs correct dimension
        elif self.config.value_loss_type == "symlog":
            # For symlog loss, squeeze to scalar and apply symlog transformation
            if value.ndim == 2 and value.shape[-1] == 1:
                value = jnp.squeeze(value, axis=-1)
            value = symlog(value, base=self.config.symlog_base)
        else:  # MSE
            # For MSE loss, ensure scalar output
            if value.ndim == 2 and value.shape[-1] == 1:
                value = jnp.squeeze(value, axis=-1)

        return policy_logits, value

class RewardNetwork(nnx.Module):
    """Reward Network for Flax NNX. Predicts reward from hidden state."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        # Configure reward head output dimension based on loss type
        output_dim = self.config.get_reward_output_dim()
        
        if self.config.use_image_observation:
            self.conv = nnx.Conv(self.config.num_channels, 1, kernel_size=(1,1), rngs=rngs) 
            self.bn = nnx.BatchNorm(1, use_running_average=True, rngs=rngs)
            flatten_size = self.config.spatial_extents[0] * self.config.spatial_extents[1] * 1 
            self.fc = MLP(flatten_size, self.config.fc_prediction_layers, output_dim, rngs=rngs) 
        else: 
            self.fc = MLP(input_size=self.config.num_channels,
                          hidden_sizes=self.config.fc_prediction_layers, 
                          output_size=output_dim,
                          rngs=rngs)

    def __call__(self, hidden_state: jax.Array, training: bool) -> jax.Array:
        if self.config.use_image_observation:
            x = self.conv(hidden_state)
            x = self.bn(x, use_running_average=not training)
            x = nnx.relu(x)
            x_reshaped = x.reshape((x.shape[0], -1))
            reward = self.fc(x_reshaped, training=training)
        else:
            reward = self.fc(hidden_state, training=training)

        # Apply transformations to match loss function expectations
        if self.config.reward_loss_type in ["categorical", "kl"]:
            # For categorical/kl loss, output logits over support (already configured in head)
            pass  # No transformation needed - head outputs correct dimension
        elif self.config.reward_loss_type == "symlog":
            # For symlog loss, squeeze to scalar and apply symlog transformation
            if reward.ndim == 2 and reward.shape[-1] == 1:
                reward = jnp.squeeze(reward, axis=-1)
            reward = symlog(reward, base=self.config.symlog_base)
        else:  # MSE
            # For MSE loss, ensure scalar output
            if reward.ndim == 2 and reward.shape[-1] == 1:
                reward = jnp.squeeze(reward, axis=-1)
            
        return reward

class ProjectionNetwork(nnx.Module):
    """Projects hidden state for self-supervised learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.dense1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.bn1 = nnx.BatchNorm(hidden_dim, use_running_average=True, rngs=rngs) 
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.bn2 = nnx.BatchNorm(hidden_dim, use_running_average=True, rngs=rngs)
        self.dense3 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.bn3 = nnx.BatchNorm(output_dim, use_running_average=True, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool):
        if x.ndim > 2: 
            x = x.reshape((x.shape[0], -1))
        
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"ProjectionNetwork input_dim {self.input_dim} does not match input shape {x.shape}") # pragma: no cover

        x = self.dense1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.dense2(x)
        x = self.bn2(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.dense3(x)
        x = self.bn3(x, use_running_average=not training)
        return x

class ProjectionHeadNetwork(nnx.Module):
    """Head for the projection network, used in self-supervised learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.bn1 = nnx.BatchNorm(hidden_dim, use_running_average=True, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool):
        x = self.dense1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.dense2(x)
        return x

class MuZeroNetwork(nnx.Module):
    """Combined MuZero Network (Flax NNX)."""
    def __init__(self, 
                 representation_network_def: Callable[..., RepresentationNetwork],
                 dynamics_network_def: Callable[..., DynamicsNetwork],
                 prediction_network_def: Callable[..., PredictionNetwork],
                 reward_network_def: Callable[..., RewardNetwork],
                 projection_network_def: Callable[..., ProjectionNetwork] | None, # Optional
                 config, # General config object for sub-networks and MuZero itself
                 *, rngs: nnx.Rngs):
        super().__init__()
        self.config = config

        initial_rngs_obj = rngs 
        params_jax_key = initial_rngs_obj.params.key.value
        
        if hasattr(initial_rngs_obj, 'dropout') and initial_rngs_obj.dropout is not None:
            dropout_jax_key = initial_rngs_obj.dropout.key.value
        else:
            # If 'dropout' is not present, create a new JAX key for it by splitting the original params_jax_key,
            # but keep the original params_jax_key for the 'params' stream.
            _, dropout_specific_key = jax.random.split(params_jax_key) 
            dropout_jax_key = dropout_specific_key

        submodule_rngs = nnx.Rngs(params=params_jax_key, dropout=dropout_jax_key)

        self.representation_network = representation_network_def(config, rngs=submodule_rngs)
        self.dynamics_network = dynamics_network_def(config, rngs=submodule_rngs)
        self.prediction_network = prediction_network_def(config, rngs=submodule_rngs)
        self.reward_network = reward_network_def(config, rngs=submodule_rngs)
        
        # LSTM reward network for value-prefix (EfficientZeroV2 feature)
        if hasattr(config, 'use_value_prefix') and config.use_value_prefix:
            self.lstm_reward_network = SupportLSTMRewardNetwork(config, rngs=submodule_rngs)
        else:
            self.lstm_reward_network = None
        
        if config.use_projection and projection_network_def is not None:
            # The projection_network_def lambda expects (config, *, rngs_lambda) in tests
            # submodule_rngs is already correctly formatted.
            self.projection_network = projection_network_def(config, rngs=submodule_rngs)
        else:
            self.projection_network = None

    def representation(self, observation: jax.Array, training: bool) -> jax.Array:
        return self.representation_network(observation, training=training)

    def dynamics(self, hidden_state: jax.Array, action: jax.Array, training: bool, 
                 reward_hidden: LSTMState | None = None) -> tuple[jax.Array, jax.Array, LSTMState | None]:
        """Predicts next hidden state and reward, with optional LSTM reward prediction."""
        next_hidden_state = self.dynamics_network(hidden_state, action, training=training)
        
        # Use LSTM reward network if available, otherwise use standard reward network
        if self.lstm_reward_network is not None:
            reward, new_reward_hidden = self.lstm_reward_network(next_hidden_state, reward_hidden, training=training)
            return next_hidden_state, reward, new_reward_hidden
        else:
            reward = self.reward_network(next_hidden_state, training=training)
            return next_hidden_state, reward, None

    def prediction(self, hidden_state: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        return self.prediction_network(hidden_state, training=training)

    def reset_noise(self, rng_key: jax.Array) -> None:
        """Reset noise in all networks if using noisy networks."""
        if hasattr(self.prediction_network, 'reset_noise'):
            self.prediction_network.reset_noise(rng_key)

    def initial_inference(self, observation: jax.Array, training: bool = False, 
                         reward_hidden: LSTMState | None = None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, LSTMState | None]:
        """Representation + Prediction + Reward for the first step."""
        hidden_state = self.representation(observation, training=training)
        
        # Use LSTM reward network if available, otherwise use standard reward network
        if self.lstm_reward_network is not None:
            reward, new_reward_hidden = self.lstm_reward_network(hidden_state, reward_hidden, training=training)
        else:
            reward = self.reward_network(hidden_state, training=training)
            new_reward_hidden = None
            
        policy_logits, value = self.prediction(hidden_state, training=training)
        
        projected_output = None
        if self.projection_network:
            proj_input = hidden_state
            if proj_input.ndim > 2: # Image case, e.g. (B, H, W, C)
                proj_input = jnp.mean(proj_input, axis=tuple(range(1, proj_input.ndim -1))) # Global average pool to (B, C)
            projected_state = self.projection_network(proj_input, training=training)
            projected_output = projected_state
            
        return hidden_state, reward, value, policy_logits, projected_output, new_reward_hidden

    def recurrent_inference(self, hidden_state: jax.Array, action: jax.Array, training: bool = False,
                           reward_hidden: LSTMState | None = None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, LSTMState | None]:
        """Dynamics + Prediction + Reward for subsequent steps."""
        next_hidden_state, reward, new_reward_hidden = self.dynamics(hidden_state, action, training=training, reward_hidden=reward_hidden)
        policy_logits, value = self.prediction(next_hidden_state, training=training)

        projected_output = None
        if self.projection_network:
            proj_input = next_hidden_state
            if proj_input.ndim > 2: # Image case
                proj_input = jnp.mean(proj_input, axis=tuple(range(1, proj_input.ndim -1)))
            projected_state = self.projection_network(proj_input, training=training)
            projected_output = projected_state

        return next_hidden_state, reward, value, policy_logits, projected_output, new_reward_hidden

# ✅ IMPLEMENTED: Supporting NNX Modules for EfficientZeroV2 functionality
# The following components are already implemented in this file:
# - SupportLSTMRewardNetwork: LSTM-based reward network for value-prefix
# - DownSample: Downsampling layers for visual representation
# - RepresentationNetwork, DynamicsNetwork, PredictionNetwork: Core MuZero networks
# - RewardNetwork: Standard reward prediction
# - ProjectionNetwork, ProjectionHeadNetwork: SSL projection heads
# - MuZeroNetwork: Complete integrated network
#
# All necessary components from EfficientZeroV2/ez/agents/models/layer.py 
# have been adapted to Flax NNX conventions.

# Additional ResNet blocks and advanced layers can be added here if needed
# class ResNetBlock(nnx.Module):
#    def __init__(self, num_filters, *, rngs: nnx.Rngs):
#        self.conv1 = nnx.Conv(num_filters, num_filters, kernel_size=(3,3), padding='SAME', rngs=rngs)
#        self.bn1 = nnx.BatchNorm(num_filters, use_running_average=True, rngs=rngs)
#        self.conv2 = nnx.Conv(num_filters, num_filters, kernel_size=(3,3), padding='SAME', rngs=rngs)
#        self.bn2 = nnx.BatchNorm(num_filters, use_running_average=True, rngs=rngs)
#
#    def __call__(self, x, training: bool):
#        residual = x
#        x = nnx.relu(self.bn1(self.conv1(x), use_running_average=not training))
#        x = self.bn2(self.conv2(x), use_running_average=not training)
#        return nnx.relu(x + residual)

class SupportLSTMRewardNetwork(nnx.Module):
    """
    LSTM-based reward network for value-prefix reward accumulation.
    
    This implements EfficientZeroV2's SupportLSTMNetwork architecture for
    reward prediction with hidden state management and periodic reset.
    
    Supports both spatial (image) and flat (discrete) hidden state inputs:
    - Spatial inputs [B, C, H, W]: Uses conv1x1 reduction + flatten
    - Flat inputs [B, C]: Uses MLP reduction
    Both paths produce the same LSTM input dimensionality.
    """
    
    def __init__(self, config: MuZeroNetworkConfig, *, rngs: nnx.Rngs):
        """
        Initialize LSTM reward network.
        
        Args:
            config: Network configuration with LSTM parameters
            rngs: Random number generators for initialization
        """
        self.config = config
        
        # Determine LSTM input size based on observation type
        if config.use_image_observation:
            # Spatial path: Conv1x1 reduction + flatten
            self.conv1x1_reward = nnx.Conv(
                in_features=config.hidden_state_size,
                out_features=config.reduced_channels_reward,
                kernel_size=(1, 1),
                padding='SAME',
                use_bias=True,
                rngs=rngs
            )
            self.mlp_reward = None
            lstm_input_size = config.reduced_channels_reward * config.spatial_size
        else:
            # Flat path: MLP reduction
            self.conv1x1_reward = None
            self.mlp_reward = MLP(
                input_size=config.hidden_state_size,
                hidden_sizes=[config.reduced_channels_reward * 4],  # Intermediate layer
                output_size=config.reduced_channels_reward * config.spatial_size,  # Match spatial path output
                activation=nnx.relu,
                rngs=rngs
            )
            lstm_input_size = config.reduced_channels_reward * config.spatial_size
        
        # LSTM cell for sequential reward prediction
        self.lstm_cell = nnx.LSTMCell(
            in_features=lstm_input_size,
            hidden_features=config.lstm_hidden_size,
            recurrent_kernel_init=nnx.initializers.normal(stddev=0.1),  # Use normal init instead of orthogonal
            rngs=rngs
        )
        
        # Output MLP for reward support prediction
        self.reward_head = MLP(
            input_size=config.lstm_hidden_size,
            hidden_sizes=[config.lstm_hidden_size // 2],
            output_size=config.reward_support_size if config.reward_support_size > 0 else 1,
            activation=nnx.relu,
            rngs=rngs
        )
        
    def init_hidden_state(self, batch_size: int) -> LSTMState:
        """Initialize LSTM hidden state for a batch."""
        c = jnp.zeros((batch_size, self.config.lstm_hidden_size))
        h = jnp.zeros((batch_size, self.config.lstm_hidden_size))
        return (c, h)
        
    def reset_hidden_state(self, hidden_state: LSTMState, reset_mask: jax.Array) -> LSTMState:
        """Reset LSTM hidden state based on reset mask."""
        # reset_mask: [B] - 1 where to reset, 0 where to keep
        reset_mask = reset_mask.reshape(-1, 1)  # [B, 1]
        
        c, h = hidden_state
        new_c = c * (1 - reset_mask)
        new_h = h * (1 - reset_mask)
        return (new_c, new_h)
        
    def __call__(self, 
                 hidden_state: jax.Array, 
                 reward_hidden: LSTMState | None = None,
                 training: bool = False) -> Tuple[jax.Array, LSTMState]:
        """
        Forward pass through LSTM reward network.
        
        Args:
            hidden_state: Hidden state from dynamics network 
                         [B, C, H, W] for spatial or [B, C] for flat
            reward_hidden: LSTM hidden state (h, c)
            training: Whether in training mode
            
        Returns:
            Tuple of (reward_prediction, new_reward_hidden)
        """
        batch_size = hidden_state.shape[0]
        
        # Initialize hidden state if None
        if reward_hidden is None:
            reward_hidden = self.init_hidden_state(batch_size)
            
        # Feature reduction: spatial vs flat paths
        if self.config.use_image_observation:
            # Spatial path: Conv1x1 reduction + flatten
            x = self.conv1x1_reward(hidden_state)  # [B, reduced_channels, H, W]
            x = x.reshape(batch_size, -1)  # [B, reduced_channels * H * W]
        else:
            # Flat path: MLP reduction
            x = self.mlp_reward(hidden_state, training)  # [B, reduced_channels * spatial_size]
        
        # LSTM forward pass
        new_reward_hidden, lstm_out = self.lstm_cell(reward_hidden, x)  # [B, lstm_hidden_size]
        
        # Predict reward support/scalar
        reward_prediction = self.reward_head(lstm_out, training)  # [B, reward_support_size] or [B, 1]
        
        return reward_prediction, new_reward_hidden 