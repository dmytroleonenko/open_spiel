import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Sequence, Callable # For type hinting

# Assuming layers.py is in the same directory or accessible in PYTHONPATH
from .layers import conv3x3, ResidualBlock, FCResidualBlock, MLP

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
        # Simplified version of EZv2 DownSample, focusing on core structure
        # Uses two strided convolutions with ResBlocks
        self.conv1 = nnx.Conv(in_channels, out_channels // 2, kernel_size=(3,3), strides=(2,2), padding='SAME', use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels // 2, use_running_average=True, rngs=rngs)
        # EZv2 uses ModuleList of ResBlocks, here one for simplicity per stage
        self.resblock1 = ResidualBlock(out_channels // 2, out_channels // 2, rngs=rngs)
        
        self.conv2 = nnx.Conv(out_channels // 2, out_channels, kernel_size=(3,3), strides=(2,2), padding='SAME', use_bias=False, rngs=rngs)
        # EZv2 uses a ResBlock with downsample_conv for the second stride, this is a simplified direct conv + resblock
        self.bn2 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)
        self.resblock2 = ResidualBlock(out_channels, out_channels, rngs=rngs)
        
        # EZv2 has further pooling and ResBlocks, omitted for initial brevity

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.resblock1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.resblock2(x, training=training)
        return x

class RepresentationNetwork(nnx.Module):
    """Representation Network (h) for Flax NNX."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        if self.config.use_image_observation:
            if self.config.downsample:
                self.downsampler = DownSample(self.config.observation_shape[-1], self.config.num_channels, rngs=rngs)
                # Input to ResBlocks is now num_channels
                in_channels_for_resblocks = self.config.num_channels
            else:
                self.downsampler = None
                self.initial_conv = conv3x3(self.config.observation_shape[-1], self.config.num_channels, rngs=rngs)
                self.initial_bn = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
                in_channels_for_resblocks = self.config.num_channels
            
            self.resblocks = [
                ResidualBlock(in_channels_for_resblocks, self.config.num_channels, 
                              rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())) 
                for i in range(self.config.representation_num_blocks)
            ]
        else: # Flat observation (e.g. board state vector)
            self.mlp = MLP(input_size=self.config.observation_shape[0],
                           hidden_sizes=self.config.fc_representation_layers,
                           output_size=self.config.num_channels, # Output is the hidden state
                           rngs=rngs)

    def __call__(self, observation: jax.Array, training: bool) -> jax.Array:
        if self.config.use_image_observation:
            if self.downsampler:
                x = self.downsampler(observation, training=training)
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
            if getattr(self.config, 'action_embedding', False):
                # This part needs more careful adaptation from EZv2 if action_embedding is used
                # For now, let's assume a simple concatenation without a separate embedding conv for action here
                action_planes = self.config.action_embedding_dim # Placeholder if we add embedding
                # self.action_embed_conv = nnx.Conv(1, action_planes, kernel_size=(1,1), rngs=...) # Example
                pass # pragma: no cover

            self.conv1 = conv3x3(self.config.num_channels + action_planes, self.config.num_channels, rngs=rngs)
            self.bn1 = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
            self.resblocks = [
                ResidualBlock(self.config.num_channels, self.config.num_channels, 
                              rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())) 
                for i in range(self.config.dynamics_num_blocks)
            ]
        else: # Flat observations
            # Input: hidden_state + action representation
            # Action can be one-hot encoded or embedded
            action_dim = self.config.num_actions # if one-hot for discrete
            self.mlp = MLP(input_size=self.config.num_channels + action_dim,
                           hidden_sizes=self.config.fc_dynamics_layers,
                           output_size=self.config.num_channels, # Output is next hidden_state
                           rngs=rngs)

    def __call__(self, hidden_state: jax.Array, action: jax.Array, training: bool) -> jax.Array:
        # Action encoding needs to be well-defined here.
        # For image data (as in EZv2 dynamics input conv):
        if self.config.use_image_observation:
            if not self.config.is_continuous:
                # Create spatial action plane: (batch, H, W, 1)
                # Scale action_id for numerical stability if it's directly used as values
                action_scaled = action / self.config.num_actions 
                action_plane = jnp.ones_like(hidden_state[..., :1]) * action_scaled.reshape(-1, 1, 1, 1) 
            else:
                # Continuous actions: (batch, action_dim) -> (batch, H, W, action_dim)
                # This part requires careful handling based on how continuous actions are fed.
                # For now, placeholder, assuming action is already appropriately shaped or embedded.
                action_plane = action # pragma: no cover # This is likely incorrect, needs proper spatial broadcasting or embedding
                # If action_embedding is True, action_plane should be processed by an embedding net first.

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
            if hidden_state.ndim != 2:
                raise TypeError(f"Expected hidden_state with ndim=2 for flat observations, got ndim={hidden_state.ndim}")
            # Handle actions
            if not self.config.is_continuous:
                # Discrete actions: allow scalar or batch of scalars
                squeezed_action = action.squeeze()
                action_one_hot = jax.nn.one_hot(squeezed_action, num_classes=self.config.num_actions)
            else:
                # Continuous actions should be provided already in correct shape (batch, action_dim)
                action_one_hot = action
            # Validate action_one_hot dimensions
            if action_one_hot.ndim not in (1, 2):
                raise TypeError(f"Expected action_one_hot with ndim 1 or 2 for flat observations, got ndim={action_one_hot.ndim}") # pragma: no cover
            if action_one_hot.ndim == 1:
                # Expand batch dimension
                action_one_hot = jnp.expand_dims(action_one_hot, axis=0)
            # Now both hidden_state and action_one_hot are 2D: (batch, features)
            dynamics_input = jnp.concatenate([hidden_state, action_one_hot], axis=-1)
            return self.mlp(dynamics_input, training=training)


class PredictionNetwork(nnx.Module):
    """Prediction Network (f) for policy and value - Flax NNX."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        # Common trunk for policy and value (if image-based)
        if self.config.use_image_observation:
            self.resblocks = [
                ResidualBlock(self.config.num_channels, self.config.num_channels, 
                              rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())) 
                for i in range(self.config.prediction_num_blocks)
            ]
            # Policy head
            self.policy_conv = nnx.Conv(self.config.num_channels, self.config.num_channels, kernel_size=(1,1), rngs=rngs) # Reduced channels in EZv2, map to num_channels for now
            self.policy_bn = nnx.BatchNorm(self.config.num_channels, use_running_average=True, rngs=rngs)
            # Flatten size needs to be computed based on output of convs, e.g., H*W*C'
            # Assuming H, W are known (e.g., 6x6 for 96x96 input with 4x downsampling in repr)
            # This needs to be configurable or dynamically calculated.
            # For example, if representation outputs 6x6xnum_channels:
            flatten_size_policy = self.config.spatial_extents[0] * self.config.spatial_extents[1] * self.config.num_channels
            self.policy_fc = MLP(flatten_size_policy, self.config.fc_prediction_layers, self.config.num_actions, rngs=rngs)

            # Value head (similar structure)
            self.value_conv = nnx.Conv(self.config.num_channels, 1, kernel_size=(1,1), rngs=rngs) # EZv2 uses 1 filter for scalar, support_size for categorical
            self.value_bn = nnx.BatchNorm(1, use_running_average=True, rngs=rngs)
            flatten_size_value = self.config.spatial_extents[0] * self.config.spatial_extents[1] * 1 # *1 because 1 filter from value_conv
            
            value_output_dim = self.config.value_support_size if self.config.value_support_size > 0 else 1
            self.value_fc = MLP(flatten_size_value, self.config.fc_prediction_layers, value_output_dim, rngs=rngs)

        else: # Flat observations
            # For flat observations, the hidden state is already 1D (batch, num_channels)
            # We can directly apply MLPs for policy and value heads.
            # Policy head
            policy_rngs = nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())
            self.policy_fc = MLP(input_size=self.config.num_channels,
                                 hidden_sizes=self.config.fc_prediction_layers,
                                 output_size=self.config.num_actions,
                                 rngs=policy_rngs)
            
            # Value head
            value_output_dim = self.config.value_support_size if self.config.value_support_size > 0 else 1
            value_rngs = nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())
            self.value_fc = MLP(input_size=self.config.num_channels,
                                hidden_sizes=self.config.fc_prediction_layers, # Can use same or different FC layers
                                output_size=value_output_dim,
                                rngs=value_rngs)

    def __call__(self, hidden_state: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        if self.config.use_image_observation:
            # Common trunk
            x = hidden_state
            for block in self.resblocks:
                x = block(x, training=training)
            
            # Policy head
            policy_x = self.policy_conv(x)
            policy_x = self.policy_bn(policy_x, use_running_average=not training)
            policy_x = nnx.relu(policy_x)
            policy_x = policy_x.reshape((policy_x.shape[0], -1)) # Flatten
            policy_logits = self.policy_fc(policy_x, training=training)

            # Value head
            value_x = self.value_conv(x)
            value_x = self.value_bn(value_x, use_running_average=not training)
            value_x = nnx.relu(value_x)
            value_x = value_x.reshape((value_x.shape[0], -1)) # Flatten
            value = self.value_fc(value_x, training=training)
        else: # Flat observations
            # No shared trunk of ResBlocks typically for flat, hidden_state is the input to MLPs
            policy_logits = self.policy_fc(hidden_state, training=training)
            value = self.value_fc(hidden_state, training=training)

        if self.config.value_support_size == 0: # Scalar value
            value = jnp.squeeze(value, axis=-1) # Ensure (batch_size,)

        return policy_logits, value

class RewardNetwork(nnx.Module):
    """Reward Network for Flax NNX. Predicts reward from hidden state."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        output_dim = self.config.reward_support_size if self.config.reward_support_size > 0 else 1
        
        if self.config.use_image_observation:
            # Similar to value head in PredictionNetwork for image obs
            self.conv = nnx.Conv(self.config.num_channels, 1, kernel_size=(1,1), rngs=rngs) 
            self.bn = nnx.BatchNorm(1, use_running_average=True, rngs=rngs)
            flatten_size = self.config.spatial_extents[0] * self.config.spatial_extents[1] * 1
            self.fc = MLP(flatten_size, self.config.fc_prediction_layers, output_dim, rngs=rngs) # Using fc_prediction_layers for consistency
        else: # Flat observations
            self.fc = MLP(input_size=self.config.num_channels,
                          hidden_sizes=self.config.fc_prediction_layers, # Reusing for consistency
                          output_size=output_dim,
                          rngs=rngs)

    def __call__(self, hidden_state: jax.Array, training: bool) -> jax.Array:
        if self.config.use_image_observation:
            x = self.conv(hidden_state)
            x = self.bn(x, use_running_average=not training)
            x = nnx.relu(x)
            x = x.reshape((x.shape[0], -1)) # Flatten
            reward = self.fc(x, training=training)
        else:
            reward = self.fc(hidden_state, training=training)

        if self.config.reward_support_size == 0: # Scalar reward
            reward = jnp.squeeze(reward, axis=-1) # Ensure (batch_size,)
        return reward

class MuZeroNetwork(nnx.Module):
    """Combined MuZero Network (Flax NNX)."""
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        
        # Obtain individual JAX keys from the streams for each sub-network
        # by calling the streams. Each call to a stream (e.g., rngs.params()) 
        # advances it and returns a new unique JAX PRNGKey.
        self.representation_net = RepresentationNetwork(
            config, rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())
        )
        self.dynamics_net = DynamicsNetwork(
            config, rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())
        )
        self.prediction_net = PredictionNetwork(
            config, rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())
        )
        self.reward_net = RewardNetwork(
            config, rngs=nnx.Rngs(params=rngs.params(), dropout=rngs.dropout())
        )

    def representation(self, observation: jax.Array, training: bool) -> jax.Array:
        return self.representation_net(observation, training=training)

    def dynamics(self, hidden_state: jax.Array, action: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        """Predicts next hidden state and reward."""
        next_hidden_state = self.dynamics_net(hidden_state, action, training=training)
        # Reward is predicted from the *next* hidden state
        reward = self.reward_net(next_hidden_state, training=training)
        return next_hidden_state, reward

    def prediction(self, hidden_state: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        return self.prediction_net(hidden_state, training=training)

    def initial_inference(self, observation: jax.Array, training: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Representation + Prediction + Reward for the first step."""
        hidden_state = self.representation(observation, training=training)
        # In initial inference, reward is usually considered 0 as no action has been taken to produce it.
        # Or, some variants predict reward from the initial hidden_state.
        # EfficientZeroV2's diagram implies reward is part of dynamics output g(s,a) -> s', r
        # but their model's support_to_scalar(reward_logits) is often called on hidden_state after dynamics.
        # Let's predict reward from the initial hidden state for consistency with recurrent_inference expecting reward from s'.
        reward = self.reward_net(hidden_state, training=training)
        policy_logits, value = self.prediction(hidden_state, training=training)
        return hidden_state, reward, value, policy_logits

    def recurrent_inference(self, hidden_state: jax.Array, action: jax.Array, training: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Dynamics + Prediction + Reward for subsequent steps."""
        next_hidden_state, reward = self.dynamics(hidden_state, action, training=training)
        policy_logits, value = self.prediction(next_hidden_state, training=training)
        return next_hidden_state, reward, value, policy_logits

# TODO: Implement supporting NNX Modules like ResNetBlock, DownSample, SupportNetwork, ProjectionNetwork etc.
# based on EfficientZeroV2/ez/agents/models/layer.py, adapting to Flax NNX conventions.

# Example of a supporting layer (replace with actual implementations)
# class ResNetBlock(nnx.Module):
#    def __init__(self, num_filters, *, rngs: nnx.Rngs):
#        self.conv1 = nnx.Conv(num_filters, num_filters, kernel_size=(3,3), padding='SAME', rngs=rngs)
#        self.bn1 = nnx.BatchNorm(num_filters, use_running_average=True, rngs=rngs) # use_running_average for eval
#        self.conv2 = nnx.Conv(num_filters, num_filters, kernel_size=(3,3), padding='SAME', rngs=rngs)
#        self.bn2 = nnx.BatchNorm(num_filters, use_running_average=True, rngs=rngs)
#
#    def __call__(self, x, training: bool):
#        residual = x
#        x = nnx.relu(self.bn1(self.conv1(x), use_running_average=not training))
#        x = self.bn2(self.conv2(x), use_running_average=not training)
#        return nnx.relu(x + residual) 