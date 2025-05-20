"""JAX/Flax models for AlphaZero.

This module provides JAX/Flax implementations of neural network models commonly
used with AlphaZero, including Multi-Layer Perceptrons (MLP), Convolutional
Neural Networks (Conv2D), and various ResNet architectures.

The ResNet implementations (`ResNet_JAX` and its variants) are based on or
utilize code from the `n2cholas/jax-resnet` repository
(https://github.com/n2cholas/jax-resnet).

Available Model Types (`config.nn_model`):
  - "mlp": A simple Multi-Layer Perceptron.
  - "conv2d": A Convolutional Neural Network with a series of ConvBlocks.
  - "resnet": A generic ResNet that can be configured via `config.resnet_*` fields.
      This allows for flexible ResNet architectures. By default, if these specific
      fields are not populated, it attempts to create a ResNet similar to the
      AlphaGo Zero architecture (e.g., 20 residual blocks with 256 filters,
      configurable via `config.nn_width` and `config.nn_depth`).
  - Specific ResNet Variants:
    - "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200":
      Standard ResNet architectures.
    - "resnet_d18", "resnet_d34", "resnet_d50", "resnet_d101", "resnet_d152", "resnet_d200":
      ResNet-D variants with modified stem and downsampling blocks.
    - "resnext50", "resnext101": ResNeXt architectures with grouped convolutions.
    - "wide_resnet50", "wide_resnet101": Wide ResNet variants.
    - "resnest50fast", "resnest50", "resnest101": ResNeSt variants with Split-Attention networks.

Configuration for Generic "resnet" (`config.nn_model="resnet"`):
  - `config.nn_width`: Number of filters (e.g., 256).
  - `config.nn_depth`: Number of residual blocks (e.g., 20).
  - `config.resnet_depth_config`: List of ints, e.g., `[3, 4, 6, 3]` for ResNet34-like stage sizes.
  - `config.resnet_stem_callable_name`: String name of the stem module constructor (e.g., "ResNetStem").
  - `config.resnet_stem_kwargs`: Dictionary of kwargs for the stem constructor.
  - `config.resnet_block_callable_name`: String name of the block module constructor (e.g., "ResNetBlock").
  - `config.resnet_block_kwargs`: Dictionary of kwargs for the block constructor.

All models output policy logits and a value prediction. They support optional
`legals_mask` to mask policy logits for illegal actions.
Batch Normalization layers use `training=True` during training steps and
`training=False` (i.e., `use_running_average=True`) during inference/evaluation.
"""
import collections
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Callable, Mapping, Tuple
import functools # Added for partial
import sys # For getattr(sys.modules[__name__], ...) if used, or a local dict is safer

# The ResNet_JAX model and its components (ResNetStem, ResNetBlock, ResNetBottleneckBlock, ConvBlock)
# are based on or utilize code from the n2cholas/jax-resnet repository:
# https://github.com/n2cholas/jax-resnet
# Original copyright and license terms apply to those components.
from open_spiel.python.algorithms.alpha_zero_jax.resnet import (
    ResNetStem, ResNetBlock, ResNetBottleneckBlock,
    ResNetDStem, ResNetDBlock, ResNetDBottleneckBlock,
    ResNeStBottleneckBlock # Added ResNeSt variant
)
from open_spiel.python.algorithms.alpha_zero_jax.resnet.common import ConvBlock # Used by Conv2D_JAX
from open_spiel.python.algorithms.alpha_zero_jax.resnet.resnet import STAGE_SIZES, ResNeSt1D50_AZ, ResNet # Import ResNet
from open_spiel.python.algorithms.alpha_zero_jax.resnet.splat import SplAtConv2d # For ResNeSt



class TrainInputJAX(collections.namedtuple(
    "TrainInputJAX", "observation legals_mask policy_target value_target")):
  """Inputs for training the JAX AlphaZero model."""

  @staticmethod
  def stack(train_inputs):
    observation, legals_mask, policy_target, value_target = zip(*train_inputs)
    return TrainInputJAX(
        np.array(observation, dtype=np.float32),
        np.array(legals_mask, dtype=bool),
        np.array(policy_target, dtype=np.float32),  # Assuming policy_target is float for JAX
        np.expand_dims(np.array(value_target, dtype=np.float32), 1)) # Assuming value_target is float for JAX 

class MLP_JAX(nn.Module):
  nn_width: int
  nn_depth: int
  output_size: int

  @nn.compact
  def __call__(self, x: jax.Array, training: bool, legals_mask: Optional[jax.Array] = None):
    # Ensure input is flattened for MLP
    if x.ndim > 2: # (batch_size, features) is expected, so > 2 means needs flattening
        x = x.reshape((x.shape[0], -1))
    # Torso
    for _ in range(self.nn_depth):
      x = nn.Dense(features=self.nn_width)(x)
      x = nn.BatchNorm(use_running_average=not training, name=f"torso_bn_{_}")(x)
      x = nn.relu(x)

    # Policy head
    policy_logits = nn.Dense(features=self.output_size, name="policy_head")(x)

    # Value head
    value_hidden = nn.Dense(features=self.nn_width, name="value_hidden")(x)
    value_hidden = nn.relu(value_hidden)
    value_output = nn.Dense(features=1, name="value_output")(value_hidden)
    value_output = jnp.tanh(value_output) # Use jnp.tanh for JAX 

    if legals_mask is not None:
      policy_logits = jnp.where(legals_mask, policy_logits, -jnp.inf)

    return (policy_logits, value_output)

class ResNet_JAX(nn.Module):
  stem_constructor: Callable[[], nn.Module]
  block_constructor: Callable[..., nn.Module]
  nn_width: int  # Base filter count for the first stage, e.g., 64. Also for value head's hidden layer.
  nn_depth_config: list[int]  # Stage sizes, e.g., [2, 2, 2, 2] for ResNet18 layer count per stage.
  output_size: int # Number of actions for policy head
  expected_input_shape: tuple[int, ...] # Added: e.g., (H, W, C) for the game's observations
  block_kwargs: Optional[Mapping] = None # For expansion, groups, base_width
  norm_cls: Callable[..., nn.Module] = functools.partial(nn.BatchNorm, use_running_average=None, momentum=0.95)

  """A JAX-based ResNet model for AlphaZero.

  Attributes:
    stem_constructor: A callable that returns a Flax nn.Module for the ResNet stem.
    block_constructor: A callable that returns a Flax nn.Module for a ResNet block.
    nn_width: Base filter count for the first stage and for the value head's hidden layer.
    nn_depth_config: A list defining the number of blocks in each ResNet stage.
    output_size: The number of output units for the policy head.
    block_kwargs: Optional dictionary of arguments for the block_constructor (e.g., expansion factor).
    norm_cls: The normalization layer to use (e.g., BatchNorm).

  Note on Torso Output and Head Structure:
    This ResNet_JAX implementation uses global average pooling (jnp.mean) on the output
    of the final ResNet stage before passing it to the policy and value heads.
    This is a common pattern for ResNets in classification tasks.

    The original OpenSpiel TensorFlow AlphaZero reference `Model` (when using ResNet)
    employs a different structure after the main convolutional blocks: it uses a
    1x1 Convolution -> BatchNorm -> ReLU -> Flatten sequence before the final dense
    layers for both policy and value heads, instead of global average pooling.

    This JAX version deviates by using global average pooling, which is a standard
    ResNet design choice. If strict alignment with the TensorFlow reference's specific
    head structure is required, the `__call__` method of this ResNet_JAX module
    would need to be modified to replicate that 1x1 Conv + BN + ReLU + Flatten structure.
  """

  @nn.compact
  def __call__(self, x: jax.Array, training: bool, legals_mask: Optional[jax.Array] = None):
    # The `training` argument to this __call__ method is implicitly used by BatchNorm layers
    # when the model is invoked via `model.apply(..., training=training_status)`.

    # Reshape flat input (Batch, Features) to expected 3D spatial format (Batch, H, W, C)
    # expected_input_shape is (C,H,W) from game.observation_tensor_shape() if CHW layout
    if hasattr(self, 'expected_input_shape') and self.expected_input_shape and x.ndim == 2:
      if np.prod(self.expected_input_shape) == x.shape[1]: # Check if total features match
        # Reshape to (N, C, H, W)
        x = x.reshape((x.shape[0],) + self.expected_input_shape)
      else:
        # This case should ideally not happen if expected_input_shape is correctly set
        # and input observation matches. For robustness, one might log a warning or error.
        # For now, we proceed, but this indicates a potential mismatch.
        pass

    # If input is NCHW (e.g. x.shape = (N,C,H,W)) and model expects NHWC, transpose.
    # self.expected_input_shape stores (C,H,W)
    if x.ndim == 4 and hasattr(self, 'expected_input_shape') and self.expected_input_shape and \
       x.shape[1] == self.expected_input_shape[0] and \
       x.shape[2] == self.expected_input_shape[1] and \
       x.shape[3] == self.expected_input_shape[2]:
      x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
      # After transpose, x.shape is (N, H, W, C) using game's H, W, C

    stem = self.stem_constructor()
    x = stem(x, training=training)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

    current_n_hidden = self.nn_width
    effective_block_kwargs = self.block_kwargs or {}

    for i, num_blocks_in_stage in enumerate(self.nn_depth_config):
      for j in range(num_blocks_in_stage):
        block_strides = (1, 1) # Default strides
        if j == 0 and i > 0:  
          # Only attempt to downsample if spatial dimensions are greater than 1x1
          if x.shape[1] > 1 or x.shape[2] > 1:
            block_strides = (2, 2) # Downsample at the start of a new stage (except the first)
            current_n_hidden *= 2
          else:
            # If already 1x1, do not downsample further, but still increase channels if it's a new stage conceptually
            current_n_hidden *= 2 
            # block_strides remains (1,1)
        
        # Initialize block parameters
        current_block_params = {
            **effective_block_kwargs
        }
        
        block_func_to_check = self.block_constructor
        if isinstance(self.block_constructor, functools.partial):
            block_func_to_check = self.block_constructor.func
        
        # Corrected logic to add n_hidden and strides for all relevant ResNet block types
        # All standard blocks (ResNetBlock, ResNetBottleneckBlock) and their derivatives
        # (ResNetDBlock, ResNetDBottleneckBlock, ResNeStBottleneckBlock) require 'n_hidden' and 'strides'.
        if issubclass(block_func_to_check, (ResNetBlock, ResNetBottleneckBlock)) or \
           block_func_to_check in (ResNetDBlock, ResNetDBottleneckBlock): # DBlock/DBottleneckBlock define n_hidden/strides directly
            current_block_params['n_hidden'] = current_n_hidden
            current_block_params['strides'] = block_strides
        
        block = self.block_constructor(**current_block_params)
        x = block(x, training=training) 

    # DEBUG: Print shape of x before policy/value heads
    # jax.debug.print("ResNet_JAX: Shape of x before policy/value heads: {x_shape}", x_shape=x.shape)

    # TF-style Policy Head for ResNet
    ph = nn.Conv(features=2, kernel_size=(1,1), padding='SAME', name="policy_head_conv1x1")(x)
    ph = nn.BatchNorm(use_running_average=not training, name="policy_head_bn")(ph)
    ph = nn.relu(ph)
    ph = ph.reshape((ph.shape[0], -1))
    policy_logits = nn.Dense(features=self.output_size, name="policy_head_dense")(ph)

    # TF-style Value Head for ResNet
    vh = nn.Conv(features=1, kernel_size=(1,1), padding='SAME', name="value_head_conv1x1")(x)
    vh = nn.BatchNorm(use_running_average=not training, name="value_head_bn")(vh)
    vh = nn.relu(vh)
    vh = vh.reshape((vh.shape[0], -1))
    vh = nn.Dense(features=self.nn_width, name="value_head_dense1")(vh) # Using self.nn_width as per prior TF models (e.g. 256)
    vh = nn.relu(vh)
    value_output = nn.Dense(features=1, name="value_head_dense2")(vh)
    value_output = jnp.tanh(value_output)
    
    if legals_mask is not None:
      policy_logits = jnp.where(legals_mask, policy_logits, -jnp.inf)

    return (policy_logits, value_output)

class Conv2D_JAX(nn.Module):
  nn_width: int  # Number of filters for convolutional layers, also for value head's hidden dense layer
  nn_depth: int  # Number of convolutional layers in the torso
  output_size: int # Number of actions for policy head
  expected_input_shape: tuple[int, ...] # Added: e.g., (H, W, C) for the game's observations

  @nn.compact
  def __call__(self, x: jax.Array, training: bool, legals_mask: Optional[jax.Array] = None):
    if self.expected_input_shape:  # Expected shape is H, W, C
      if x.ndim == 2:  # Input is flat (Batch, Features)
        batch_size = x.shape[0]
        # Ensure product of expected_input_shape matches feature count if flat
        if np.prod(self.expected_input_shape) != x.shape[1]:
          raise ValueError(
              f"Conv2D_JAX: Flat input feature count {x.shape[1]} does not "
              f"match product of expected_input_shape {self.expected_input_shape}."
          )
        x = x.reshape((batch_size,) + self.expected_input_shape)

    # Save the output of the torso before heads are applied
    torso_output = x
    for i in range(self.nn_depth):
      torso_output = ConvBlock(
          n_filters=self.nn_width,
                               kernel_size=(3, 3),
                               strides=(1, 1),
          padding="SAME",
          name=f"torso_conv_block_{i}",
          dim=2 # Explicitly 2D for Conv2D_JAX
          )(torso_output, training=training)

    # Flatten the output for the dense layers
    flat_x = torso_output.reshape((torso_output.shape[0], -1)) # More robust flattening

    # Policy head
    policy_logits_pre_mask = nn.Dense(features=self.output_size, name="policy_head")(flat_x)

    if legals_mask is not None:
      # Ensure legals_mask has the same batch dimension and number of actions
      if legals_mask.shape[0] != policy_logits_pre_mask.shape[0] or \
         legals_mask.shape[1] != policy_logits_pre_mask.shape[1]:
        raise ValueError(
            f"Shape mismatch: policy_logits_pre_mask {policy_logits_pre_mask.shape} "
            f"vs legals_mask {legals_mask.shape}")
      policy_logits = jnp.where(legals_mask, policy_logits_pre_mask, -jnp.inf)
    else:
      # This case should ideally only happen during model initialization or if the game has no concept of legals
      # For safety, let policy_logits be the unmasked logits if legals_mask is None
      # However, this might lead to issues if the model is used in contexts expecting masked logits
      policy_logits = policy_logits_pre_mask

    # Value head
    vh = nn.Conv(features=1, kernel_size=(1,1), padding='SAME', name="value_head_conv1x1")(torso_output)
    vh = nn.BatchNorm(use_running_average=not training, name="value_head_bn")(vh)
    vh = nn.relu(vh)
    vh = vh.reshape((vh.shape[0], -1))
    vh = nn.Dense(features=self.nn_width, name="value_head_dense1")(vh) # Using self.nn_width
    vh = nn.relu(vh)
    value_output = nn.Dense(features=1, name="value_head_dense2")(vh)
    value_output = jnp.tanh(value_output)

    return (policy_logits, value_output)

class SpatialGlobal1DResNetTransformer(nn.Module):
  """Wrapper for a 1D ResNet+Transformer core model (e.g., ResNeSt1D50_AZ)
  that splits a flat input observation tensor into spatial and global features.
  The spatial features are reshaped according to spatial_dims.
  """
  output_size: int # Number of distinct actions
  num_total_observation_features: int # Total number of features in the flat observation vector
  spatial_dims: Tuple[int, ...] # Desired shape of the spatial part, e.g., (Length, Channels) = (24, 8)
  use_transformer_head: bool = False

  @nn.compact
  def __call__(self, x: jax.Array, training: bool, legals_mask: Optional[jax.Array] = None):
    # x is expected to be flat: (Batch, num_total_observation_features), e.g., (B, 200)
    
    num_spatial_flat = int(np.prod(self.spatial_dims))
    num_global_features = self.num_total_observation_features - num_spatial_flat

    if num_global_features < 0:
        raise ValueError(
            f"Total features ({self.num_total_observation_features}) is less than "
            f"required spatial features ({num_spatial_flat} from spatial_dims {self.spatial_dims}). "
            "Check observation_shape and spatial_dims."
        )
    
    if x.shape[1] != self.num_total_observation_features:
        raise ValueError(
            f"Input feature dimension ({x.shape[1]}) does not match "
            f"num_total_observation_features ({self.num_total_observation_features})."
        )

    # Reshape input features
    x_spatial = x[:, :num_spatial_flat].reshape((-1,) + self.spatial_dims)
    
    x_global: Optional[jax.Array] = None
    if num_global_features > 0:
        x_global = x[:, num_spatial_flat:]
    elif num_global_features == 0:
        # If no global features are expected/provided, pass None to the core model.
        # The core ResNet model's __call__ handles global_features=None.
        x_global = None 
        # If the core model strictly requires global features when use_transformer_head=True,
        # this could be an issue. ResNet raises ValueError if use_transformer_head and global_features is None.
        # For now, assume if num_global_features is 0, the user intends no global features for the transformer.
        # A more robust ResNet might make global_features truly optional for the transformer part.
    
    # Filter out use_transformer_head from ResNeSt1D50_AZ.keywords if it exists,
    # to allow explicit override by self.use_transformer_head.
    core_model_base_keywords = {
        k: v for k, v in ResNeSt1D50_AZ.keywords.items() if k != 'use_transformer_head'
    }

    core_model = ResNet(
        **core_model_base_keywords, # Use filtered keywords
        output_size=self.output_size,   # Override output_size
        use_transformer_head=self.use_transformer_head # Explicitly use the wrapper's setting
    )

    # Call the core model with reshaped spatial and global features
    policy_logits, value_output = core_model(
        x_spatial, 
        training=training, 
        legals_mask=legals_mask, 
        global_features=x_global
    )
    
    return policy_logits, value_output

# Import for ConfigJAX and pyspiel.Game will be handled by the calling script.
# For now, assume they are available in the scope or passed correctly.
# import pyspiel # This should be available in the environment
# from open_spiel.python.examples.alpha_zero_jax import ConfigJAX # This is a problematic import path for a library file

def init_flax_model_and_variables(key: jax.random.PRNGKey, config, game): # config type can be ConfigJAX
  """Initializes a Flax model and its variables.

  Args:
    key: JAX PRNG key.
    config: A ConfigJAX object (or similar) containing model parameters.
    game: A pyspiel.Game object.

  Returns:
    A tuple of (model, variables).
  """
  # Added import for logging within the function if not globally available
  # import logging # Assuming logger is passed or configured globally
  # logger = logging.getLogger(__name__) # Example if using standard logging

  # For OpenSpiel's FileLogger, it's usually passed or accessible via a global setup.
  # For this edit, we'll assume a print statement will go to the correct log
  # as the FileLogger in alpha_zero_jax.py captures stdout of processes.


  observation_shape = game.observation_tensor_shape() # This is (H, W, C) or (Features,)
  output_size = game.num_distinct_actions()
  model_type = config.nn_model.lower() # Ensure lowercase for matching


  # For MLP, expected_input_shape isn't strictly needed as it flattens anyway,
  # but let's be consistent if we ever change MLP's flattening logic.
  # For now, MLP handles its own flattening based on x.ndim > 2.
  # So, we only pass expected_input_shape to Conv and ResNet models.

  if model_type == "mlp":
    model = MLP_JAX(nn_width=config.nn_width,
                    nn_depth=config.nn_depth,
                    output_size=output_size)
  elif model_type == "conv2d":
    model = Conv2D_JAX(nn_width=config.nn_width,
                       nn_depth=config.nn_depth,
                         output_size=output_size,
                       expected_input_shape=observation_shape)
  elif model_type == "spatial_global_1dresnet_transformer":
    if not hasattr(config, 'model_spatial_dims') or not config.model_spatial_dims:
        # This model type requires model_spatial_dims to be explicitly set by the user.
        raise ValueError("model_spatial_dims must be provided in config for 'spatial_global_1dresnet_transformer'. Example: --model_spatial_dims=24,8")
    
    num_total_features = int(np.prod(observation_shape)) # Total features from flat game observation
    
    # Ensure config.model_spatial_dims is correctly parsed (e.g., from a comma-separated string list flag)
    # The example script alpha_zero_jax.py already parses FLAGS.model_spatial_dims into a tuple of ints.
    # So, config.model_spatial_dims should already be a tuple of ints here.
    spatial_dims_from_config = config.model_spatial_dims 
    if not isinstance(spatial_dims_from_config, tuple) or not all(isinstance(dim, int) for dim in spatial_dims_from_config):
        raise ValueError(f"config.model_spatial_dims is not a tuple of integers. Got: {spatial_dims_from_config}. Ensure it's correctly parsed from flags (e.g. '24,8').")

    # The use_transformer_head flag is also expected to be in the config.
    # The example script alpha_zero_jax.py sets config.use_transformer_head from FLAGS.use_transformer_head.
    if not hasattr(config, 'use_transformer_head'):
        # Defaulting here, but ideally, it should always be present in the config if this model is chosen.
        print("Warning: config.use_transformer_head not found for spatial_global_1dresnet_transformer. Defaulting to True.")
        use_transformer_head_for_model = True 
    else:
        use_transformer_head_for_model = config.use_transformer_head

    model = SpatialGlobal1DResNetTransformer(
        output_size=output_size,
        num_total_observation_features=num_total_features,
        spatial_dims=spatial_dims_from_config,
        use_transformer_head=use_transformer_head_for_model 
    )
  elif model_type.startswith("resnet") or model_type.startswith("resnext") or model_type.startswith("wide_resnet") or model_type.startswith("resnest") or model_type == "resnest1d50_az":
    if model_type == "resnest1d50_az":
        # This model expects pre-processed (B, 24, 8) spatial input and optional (B, G) global features.
        # If used directly (not via wrapper), the caller must provide correctly shaped input.
        # For general AlphaZero framework, using the wrapper `SpatialGlobal1DResNetTransformer` is preferred for games with flat observations.
        print(f"Warning: Using 'resnest1d50_az' directly. Ensure input is correctly shaped (spatial + global). Consider 'spatial_global_1dresnet_transformer' for games with flat input to be split.")
        
        # Filter out use_transformer_head from ResNeSt1D50_AZ.keywords
        # to allow explicit override by config.use_transformer_head.
        resnest1d50_az_base_keywords = {
            k: v for k, v in ResNeSt1D50_AZ.keywords.items() if k != 'use_transformer_head'
        }

        model = ResNet(
            **resnest1d50_az_base_keywords, # Use filtered keywords
            output_size=output_size,   # Override output_size
            use_transformer_head=config.use_transformer_head # Override use_transformer_head based on config
        )
    elif model_type == "resnet": # Generic 2D ResNet from ResNet_JAX
      stem_callable_name = getattr(config, 'resnet_stem_callable_name', "ResNetStem")
      block_callable_name = getattr(config, 'resnet_block_callable_name', "ResNetBlock")
      stem_kwargs = getattr(config, 'resnet_stem_kwargs', {})
      block_kwargs_from_config = getattr(config, 'resnet_block_kwargs', {})
      
      resolved_stem_constructor = resolve_callable(stem_callable_name, stem_kwargs)
      resolved_block_constructor = resolve_callable(block_callable_name, {}) 

      model = ResNet_JAX(
          stem_constructor=resolved_stem_constructor,
          block_constructor=resolved_block_constructor,
          nn_width=config.nn_width, 
          nn_depth_config=config.resnet_depth_config if hasattr(config, 'resnet_depth_config') and config.resnet_depth_config else [2, 2, 2, 2], 
                         output_size=output_size,
          block_kwargs=block_kwargs_from_config,
          expected_input_shape=observation_shape # ResNet_JAX (2D) uses this for NCHW -> NHWC if needed
      )
    else: # Specific named 2D ResNet variants (resnet18, resnet50, resnest50fast etc.)
      _nn_width = config.nn_width 
      _nn_depth = config.nn_depth 
      is_generic_resnet = model_type == "resnet"
      has_specific_depth_config = hasattr(config, 'resnet_depth_config') and config.resnet_depth_config

      if is_generic_resnet and not has_specific_depth_config:
          _nn_width_for_model = 64 
          num_blocks = _nn_depth
          if num_blocks == 20: 
              depth_config_default = [2, 2, 2, 2] 
              if _nn_width == 256: 
                  block_callable_name_default = "ResNetBottleneckBlock"
                  block_kwargs_default = {"expansion": 4}
              else: 
                  block_callable_name_default = "ResNetBlock"
                  depth_config_default = [5,5,5,5] 
                  block_kwargs_default = {}
          elif num_blocks == 40: 
              depth_config_default = [3,3,3,3] 
              block_callable_name_default = "ResNetBottleneckBlock"
              block_kwargs_default = {"expansion": 4}

          else: 
              depth_config_default = [2,2,2,2] 
              block_callable_name_default = "ResNetBlock"
              block_kwargs_default = {}

          _resnet_depth_config = getattr(config, 'resnet_depth_config', depth_config_default)
          _resnet_stem_callable_name = getattr(config, 'resnet_stem_callable_name', "ResNetStem")
          _resnet_stem_kwargs = getattr(config, 'resnet_stem_kwargs', {"n_hidden": _nn_width_for_model})
          _resnet_block_callable_name = getattr(config, 'resnet_block_callable_name', block_callable_name_default)
          _resnet_block_kwargs = {**block_kwargs_default, **getattr(config, 'resnet_block_kwargs', {})}

          stem_constructor = resolve_callable(_resnet_stem_callable_name, _resnet_stem_kwargs)
          block_constructor = resolve_callable(_resnet_block_callable_name, {}) 

          model = ResNet_JAX(
              stem_constructor=stem_constructor,
              block_constructor=block_constructor,
              nn_width=_nn_width_for_model,
              nn_depth_config=_resnet_depth_config,
                         output_size=output_size,
              block_kwargs=_resnet_block_kwargs,
              expected_input_shape=observation_shape 
          )
      elif model_type == "resnet18":
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          block_constructor = functools.partial(ResNetBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[18]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=block_kwargs_override, expected_input_shape=observation_shape)
      elif model_type == "resnet34":
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          block_constructor = functools.partial(ResNetBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[34]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=block_kwargs_override, expected_input_shape=observation_shape)
      elif model_type == "resnet50":
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          block_constructor = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[50]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=block_kwargs_override, expected_input_shape=observation_shape)
      elif model_type == "resnet101":
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          block_constructor_base = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[101]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          final_block_kwargs = {"expansion": 4, **block_kwargs_override}
          block_constructor = functools.partial(block_constructor_base, **final_block_kwargs) # Apply expansion to partial
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=final_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnet152":
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          block_constructor_base = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[152]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          final_block_kwargs = {"expansion": 4, **block_kwargs_override}
          block_constructor = functools.partial(block_constructor_base, **final_block_kwargs)
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=final_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnet200":
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          block_constructor_base = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[200]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          final_block_kwargs = {"expansion": 4, **block_kwargs_override}
          block_constructor = functools.partial(block_constructor_base, **final_block_kwargs)
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=final_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnet_d18":
          stem_constructor = functools.partial(ResNetDStem, stem_width=64, conv_block_cls=ConvBlock, dim=2)
          block_constructor = functools.partial(ResNetDBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[18]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=block_kwargs_override, expected_input_shape=observation_shape)
      elif model_type == "resnet_d34":
          stem_constructor = functools.partial(ResNetDStem, stem_width=64, conv_block_cls=ConvBlock, dim=2)
          block_constructor = functools.partial(ResNetDBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[34]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=block_kwargs_override, expected_input_shape=observation_shape)
      elif model_type == "resnet_d50":
          stem_constructor = functools.partial(ResNetDStem, stem_width=64, conv_block_cls=ConvBlock, dim=2)
          block_constructor_base = functools.partial(ResNetDBottleneckBlock, conv_block_cls=ConvBlock, dim=2)
          depth_config = STAGE_SIZES[50]
          block_kwargs_override = getattr(config, 'resnet_block_kwargs', {})
          final_block_kwargs = {"expansion": 4, **block_kwargs_override}
          block_constructor = functools.partial(block_constructor_base, **final_block_kwargs)
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=final_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnext50": 
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          _block_kwargs = {"groups": 32, "base_width": 4, "expansion": 4, **getattr(config, 'resnet_block_kwargs', {})}
          block_constructor = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2, **_block_kwargs)
          depth_config = STAGE_SIZES[50]
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnext101": 
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          _block_kwargs = {"groups": 32, "base_width": 8, "expansion": 4, **getattr(config, 'resnet_block_kwargs', {})}
          block_constructor = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2, **_block_kwargs)
          depth_config = STAGE_SIZES[101]
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "wide_resnet50": 
          stem_constructor = functools.partial(ResNetStem, conv_block_cls=ConvBlock, dim=2)
          _block_kwargs = {"expansion": 4, **getattr(config, 'resnet_block_kwargs', {})}
          _nn_width_for_wide = 128 
          block_constructor = functools.partial(ResNetBottleneckBlock, conv_block_cls=ConvBlock, dim=2, **_block_kwargs)
          depth_config = STAGE_SIZES[50]
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=_nn_width_for_wide, nn_depth_config=depth_config, output_size=output_size, block_kwargs=_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnest50fast":
          stem_constructor = functools.partial(ResNetDStem, stem_width=32, conv_block_cls=ConvBlock, dim=2)
          _block_kwargs = {"radix": 1, "groups": 1, "base_width": 64, "expansion": 4, 
                               "conv_block_cls": ConvBlock, 
                               **getattr(config, 'resnet_block_kwargs', {})}
          # ResNeStBottleneckBlock needs dim=2 for this 2D variant
          block_constructor = functools.partial(ResNeStBottleneckBlock, dim=2, **_block_kwargs) 
          depth_config = STAGE_SIZES[50]
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=_block_kwargs, expected_input_shape=observation_shape)
      elif model_type == "resnest50":
          stem_constructor = functools.partial(ResNetDStem, stem_width=64, conv_block_cls=ConvBlock, dim=2)
          _block_kwargs = {"radix": 2, "groups": 1, "base_width": 64, "avg_pool_first": False, "expansion": 4, 
                               "conv_block_cls": ConvBlock, 
                               **getattr(config, 'resnet_block_kwargs', {})}
          block_constructor = functools.partial(ResNeStBottleneckBlock, dim=2, **_block_kwargs)
          depth_config = STAGE_SIZES[50]
          model = ResNet_JAX(stem_constructor=stem_constructor, block_constructor=block_constructor, nn_width=64, nn_depth_config=depth_config, output_size=output_size, block_kwargs=_block_kwargs, expected_input_shape=observation_shape)
      else:
          raise ValueError(f"Unsupported ResNet-family model type: {config.nn_model}")
  else:
    raise ValueError(f"Unsupported model type: {config.nn_model}")

  # Initialize model parameters (variables)
  # Dummy input needs to match the observation_shape for Conv/ResNet,
  # and be flat for MLP if it expects flat.
  # game.observation_tensor_shape() is (H,W,C) or (Features,)
  # model.init expects (Batch, ...features...)
  dummy_input_shape = (1,) + tuple(observation_shape)
  dummy_input = jnp.zeros(dummy_input_shape, dtype=jnp.float32)

  # --- Special handling for resnest1d50_az dummy input --- 
  # This special handling for 'resnest1d50_az' direct use might still be needed if someone uses it directly.
  # However, for 'spatial_global_1dresnet_transformer', the dummy_input should be the flat observation_shape based.
  if model_type == "resnest1d50_az": 
      # The ResNeSt1D50_AZ model instance expects (B, 24, 8) for spatial features if used directly.
      # This path is less common if the SpatialGlobal1DResNetTransformer wrapper is used.
      # For init, we only need to satisfy the `x` (spatial) part.
      # However, ResNet will raise ValueError if use_transformer_head=True and global_features not passed, even for init.
      # This makes direct init of ResNeSt1D50_AZ problematic.
      dummy_input_spatial_shape = (1, 24, 8) # Default for direct use, might need to be configurable or error.
      dummy_global_features_shape = (1, 8) # A guess, if globals are 8. This is problematic.
      
      print(f"Warning: Initializing 'resnest1d50_az' directly. Dummy spatial input set to {dummy_input_spatial_shape}. "
            "Direct init is problematic if its Transformer head requires global features. "
            "Use 'spatial_global_1dresnet_transformer' wrapper instead.")
      
      # To attempt init, we'd need to pass global_features too if transformer is on.
      # This is a hack and shows why the wrapper is better.
      # variables = model.init(key, dummy_input_spatial, training=True, global_features=dummy_global_features)
      # For now, let it use the default dummy_input (which is flat) and it will likely fail if global_features are required.
      # Or, use the specific spatial dummy input, which also might fail for the same reason if global is needed.
      # The code below will use the `dummy_input` which is flat from `observation_shape` which is incorrect for direct ResNeSt1D50_AZ.
      # Let's set dummy_input to spatial here, and accept init might fail due to missing globals for transformer.
      dummy_input = jnp.zeros(dummy_input_spatial_shape, dtype=jnp.float32)


  # Pass legals_mask only if the model's __call__ accepts it.
  # For init, a dummy legals_mask is needed if the model's __call__ signature requires it,
  # even if it's Optional. Let's assume it's always accepted or Flax handles it.
  # A safer approach for init is to call it without optional args if they aren't strictly needed for shape inference.
  # However, if `legals_mask` influences the shape of an intermediate layer (it shouldn't for policy logits),
  # then it would be needed. Policy logits shape is fixed by `output_size`.
  # For now, init call assumes `training=False` and no legals_mask needed for shape inference of params.
  # If BatchStats are created, training=True is better for init.
  # The `mutable=['batch_stats']` in learner's train_step implies batch_stats are collected.
  # So, init with training=True is more consistent if models have BatchNorm.

  # Check if model has batch_stats (i.e., uses BatchNorm)
  # This is a bit indirect. A better way would be for models to declare if they use batch_stats.
  # For now, assume MLP, Conv2D, ResNet *can* have BatchNorm.
  init_args = (dummy_input,)
  init_kwargs = {'training': True}

  if model_type == "resnest1d50_az":
      dummy_input_spatial_shape = (1, 24, 8) # Batch size 1, 24 length, 8 channels
      current_dummy_input_for_init = jnp.zeros(dummy_input_spatial_shape, dtype=jnp.float32)
      init_args = (current_dummy_input_for_init,) # Override init_args for 'x'

      if config.use_transformer_head:
          # Calculate global features based on the actual feature vector length that actors would send.
          initial_state = game.new_initial_state()
          # Check for an optional game-specific method to get a feature array.
          # This allows a game to define a custom transformation of its state
          # into a flat feature vector, potentially differing from the standard
          # observation_tensor(). It provides flexibility for games to tailor
          # their input to models that might expect a specific feature layout
          # (e.g., a combination of spatial and global features for resnest1d50_az).
          if hasattr(game, "state_to_feature_array"):
              feature_vector = game.state_to_feature_array(initial_state)
          else:
              # Fallback to the standard OpenSpiel way of getting the observation tensor.
              # If the initial state is a chance node (e.g., dice roll in backgammon),
              # we need to advance it to a player's turn before getting the observation tensor.
              _state_for_obs = initial_state.clone()
              if _state_for_obs.is_chance_node():
                  # For games like backgammon, the initial state is a chance node (dice roll).
                  # We must apply a chance outcome to reach a player decision node
                  # before an observation tensor can be meaningfully extracted for that player.
                  # Taking the first legal action from the chance node is a common way to proceed.
                  if not _state_for_obs.legal_actions():
                      # This should ideally not happen for a valid game's initial chance node.
                      raise RuntimeError(
                          f"Initial chance node for game {game.get_type().short_name} "
                          "has no legal actions. Cannot determine observation tensor shape."
                      )
                  _state_for_obs.apply_action(_state_for_obs.legal_actions()[0])
              feature_vector = _state_for_obs.observation_tensor()
          
          feature_vector_np = np.array(feature_vector, dtype=np.float32).flatten()
          total_flat_obs_features = feature_vector_np.shape[0]

          fixed_spatial_dims = (24, 8)
          num_spatial_flat_for_model = fixed_spatial_dims[0] * fixed_spatial_dims[1] # 192
          
          num_global_features_for_init = total_flat_obs_features - num_spatial_flat_for_model
          
          if num_global_features_for_init < 0:
              print(f"Warning: Calculated num_global_features_for_init is {num_global_features_for_init} for resnest1d50_az. Using 0. Check total features ({total_flat_obs_features}) from state_to_feature_array and model's spatial assumption ({num_spatial_flat_for_model}).")
              num_global_features_for_init = 0
          
          # # if logger and config.log_level >= DEBUG: # Assuming logger and config are available in this scope or passed
          # #       logger.print(f"ResNeSt1D50_AZ init: total_flat_obs_features={total_flat_obs_features}, num_spatial_flat_for_model={num_spatial_flat_for_model}, num_global_features_for_init={num_global_features_for_init}")

          dummy_global_features_for_init = jnp.zeros((current_dummy_input_for_init.shape[0], num_global_features_for_init), dtype=jnp.float32)
          init_kwargs['global_features'] = dummy_global_features_for_init
  # For spatial_global_1dresnet_transformer, its __call__ handles splitting the flat dummy_input,
  # so it doesn't need global_features passed directly to its own init() call.
  # The `dummy_input` used for it will be the standard flat one.

  variables = model.init(key, *init_args, **init_kwargs)

  return model, variables 


# Helper function to resolve callable from string name
# (This was implicitly assumed to exist or be simple in the original file)
# For the edit, let's define a basic version if not present.
_MODEL_CONSTRUCTOR_REGISTRY = {
    "ResNetStem": ResNetStem,
    "ResNetBlock": ResNetBlock,
    "ResNetBottleneckBlock": ResNetBottleneckBlock,
    "ResNetDStem": ResNetDStem,
    "ResNetDBlock": ResNetDBlock,
    "ResNetDBottleneckBlock": ResNetDBottleneckBlock,
    "ResNeStBottleneckBlock": ResNeStBottleneckBlock,
    # Add other constructors here if needed
}

def resolve_callable(name: str, kwargs: Optional[Mapping] = None) -> Callable[[], nn.Module]:
    constructor = _MODEL_CONSTRUCTOR_REGISTRY.get(name)
    if not constructor:
        # Try to get from sys.modules, less safe
        # constructor = getattr(sys.modules[__name__], name, None)
        raise ValueError(f"Unknown callable name: {name}")
    
    if kwargs:
        return functools.partial(constructor, **kwargs)
    return functools.partial(constructor) 