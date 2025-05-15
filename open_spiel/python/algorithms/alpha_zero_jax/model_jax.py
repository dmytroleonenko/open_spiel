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
from typing import Optional, Callable, Mapping
import functools # Added for partial
import sys # For getattr(sys.modules[__name__], ...) if used, or a local dict is safer

# The ResNet_JAX model and its components (ResNetStem, ResNetBlock, ResNetBottleneckBlock, ConvBlock)
# are based on or utilize code from the n2cholas/jax-resnet repository:
# https://github.com/n2cholas/jax-resnet
# Original copyright and license terms apply to those components.
from open_spiel.python.algorithms.jax_resnet.jax_resnet import (
    ResNetStem, ResNetBlock, ResNetBottleneckBlock,
    ResNetDStem, ResNetDBlock, ResNetDBottleneckBlock,
    ResNeStBottleneckBlock # Added ResNeSt variant
)
from open_spiel.python.algorithms.jax_resnet.jax_resnet.common import ConvBlock # Used by Conv2D_JAX
from open_spiel.python.algorithms.jax_resnet.jax_resnet.resnet import STAGE_SIZES # For ResNet variants
from open_spiel.python.algorithms.jax_resnet.jax_resnet.splat import SplAtConv2d # For ResNeSt



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
  block_kwargs: Optional[Mapping] = None # For expansion, groups, base_width
  norm_cls: Callable[..., nn.Module] = functools.partial(nn.BatchNorm, use_running_average=None, momentum=0.9)

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

    stem = self.stem_constructor()
    x = stem(x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

    current_n_hidden = self.nn_width
    effective_block_kwargs = self.block_kwargs or {}

    for i, num_blocks_in_stage in enumerate(self.nn_depth_config):
      for j in range(num_blocks_in_stage):
        strides = (1, 1)
        if j == 0 and i > 0:  
          strides = (2, 2)
          current_n_hidden *= 2
        
        # Combine n_hidden, strides with other block_kwargs
        current_block_params = {
            'n_hidden': current_n_hidden, 
            'strides': strides,
            **effective_block_kwargs
        }
        block = self.block_constructor(**current_block_params)
        x = block(x) 

    # TF-style Policy Head for ResNet
    ph = nn.Conv(features=2, kernel_size=(1,1), padding='SAME', name="policy_head_conv1x1")(x)
    ph = nn.BatchNorm(use_running_average=not training, name="policy_head_bn")(ph)
    ph = nn.relu(ph)
    ph = nn.Flatten()(ph)
    policy_logits = nn.Dense(features=self.output_size, name="policy_head_dense")(ph)

    # TF-style Value Head for ResNet
    vh = nn.Conv(features=1, kernel_size=(1,1), padding='SAME', name="value_head_conv1x1")(x)
    vh = nn.BatchNorm(use_running_average=not training, name="value_head_bn")(vh)
    vh = nn.relu(vh)
    vh = nn.Flatten()(vh)
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

  @nn.compact
  def __call__(self, x: jax.Array, training: bool, legals_mask: Optional[jax.Array] = None):
    # Torso
    # The `training` flag will be used by ConvBlock for its BatchNorm layers.
    current_filters = self.nn_width
    # Save the output of the torso before heads are applied
    torso_output = x 
    for i in range(self.nn_depth):
      torso_output = ConvBlock(features=current_filters, 
                               kernel_size=(3, 3), 
                               strides=(1, 1), 
                               padding='SAME', 
                               name=f'conv_block_{i}')(torso_output, training=training)
      
    # TF-style Policy Head for Conv2D
    ph = nn.Conv(features=2, kernel_size=(1,1), padding='SAME', name="policy_head_conv1x1")(torso_output)
    ph = nn.BatchNorm(use_running_average=not training, name="policy_head_bn")(ph)
    ph = nn.relu(ph)
    ph = nn.Flatten()(ph)
    policy_logits = nn.Dense(features=self.output_size, name="policy_head_dense")(ph)

    # TF-style Value Head for Conv2D
    vh = nn.Conv(features=1, kernel_size=(1,1), padding='SAME', name="value_head_conv1x1")(torso_output)
    vh = nn.BatchNorm(use_running_average=not training, name="value_head_bn")(vh)
    vh = nn.relu(vh)
    vh = nn.Flatten()(vh)
    vh = nn.Dense(features=self.nn_width, name="value_head_dense1")(vh) # Using self.nn_width
    vh = nn.relu(vh)
    value_output = nn.Dense(features=1, name="value_head_dense2")(vh)
    value_output = jnp.tanh(value_output)

    if legals_mask is not None:
      policy_logits = jnp.where(legals_mask, policy_logits, -jnp.inf)

    return (policy_logits, value_output)

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
  observation_shape = game.observation_tensor_shape()
  # Ensure observation_shape is suitable for dummy_input later.
  # pyspiel might return a list (e.g., [9] or [3,3,1]).
  # MLP_JAX handles flattening internally if needed. ConvNets expect image-like shapes.
  if isinstance(observation_shape, list):
      pass # Current logic relies on dummy_input creation and model internal handling.

  output_size = game.num_distinct_actions()

  # Dictionary of known callable constructors
  known_stem_constructors = {
      "ResNetStem": ResNetStem,
      "ResNetDStem": ResNetDStem,
      # Add other stem constructors if any
  }
  known_block_constructors = {
      "ResNetBlock": ResNetBlock,
      "ResNetBottleneckBlock": ResNetBottleneckBlock,
      "ResNetDBlock": ResNetDBlock,
      "ResNetDBottleneckBlock": ResNetDBottleneckBlock,
      "ResNeStBottleneckBlock": ResNeStBottleneckBlock,
      # Add other block constructors if any
  }

  if config.nn_model == "mlp":
    model = MLP_JAX(nn_width=config.nn_width,
                    nn_depth=config.nn_depth,
                    output_size=output_size)
  elif config.nn_model == "resnet":
    # Try to get specific resnet config fields first
    user_depth_config = getattr(config, 'resnet_depth_config', None)
    user_stem_name = getattr(config, 'resnet_stem_callable_name', None)
    user_block_name = getattr(config, 'resnet_block_callable_name', None)
    
    # Common kwargs, can be used by both user-specified and fallback
    # These should be pulled from config regardless of main path
    stem_kwargs_from_config = getattr(config, 'resnet_stem_kwargs', None) or {}
    block_kwargs_from_config = getattr(config, 'resnet_block_kwargs', None) or {}

    # nn_width is always required for any ResNet variant
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet model (nn_model='resnet' or specific) requires 'nn_width' in config.")
    nn_width_to_use = config.nn_width

    if user_depth_config and user_stem_name and user_block_name:
        # Path 1: User has provided all specific ResNet configurations
        depth_config_final = user_depth_config
        stem_name_final = user_stem_name
        block_name_final = user_block_name
        # stem_kwargs_from_config and block_kwargs_from_config are already fetched and will be used.
    else:
        # Path 2: Fallback to using nn_width and nn_depth for a TF-like ResNet
        if not hasattr(config, 'nn_depth'): # nn_width already checked
            raise ValueError(
                "For nn_model='resnet' fallback (if resnet_depth_config, resnet_stem_callable_name, "
                "and resnet_block_callable_name are not all set), config must provide nn_depth. "
                "nn_width is always required."
            )
        
        num_blocks = config.nn_depth 
        
        depth_config_final = [num_blocks] 
        stem_name_final = "ResNetStem"    
        block_name_final = "ResNetBlock"  
        
        # stem_kwargs_from_config and block_kwargs_from_config (already fetched) will apply to these defaults.
        # Optional: Log this fallback.
        # print(f"AlphaZero JAX: Using nn_model='resnet' fallback with nn_width={nn_width_to_use}, nn_depth={num_blocks}. "
        #       f"Effective config: depth_config={depth_config_final}, stem={stem_name_final}, block={block_name_final}. "
        #       f"Applying stem_kwargs={stem_kwargs_from_config}, block_kwargs={block_kwargs_from_config}")

    # Now, use depth_config_final, stem_name_final, block_name_final, nn_width_to_use,
    # stem_kwargs_from_config, block_kwargs_from_config for instantiation.

    if stem_name_final not in known_stem_constructors:
        raise ValueError(f"Unknown ResNet stem name: {stem_name_final}. Known: {list(known_stem_constructors.keys())}")
    if block_name_final not in known_block_constructors:
        raise ValueError(f"Unknown ResNet block name: {block_name_final}. Known: {list(known_block_constructors.keys())}")

    stem_constructor_base = known_stem_constructors[stem_name_final]
    block_constructor_base = known_block_constructors[block_name_final]

    # Apply stem_kwargs using functools.partial
    stem_constructor_to_use = functools.partial(stem_constructor_base, **stem_kwargs_from_config) if stem_kwargs_from_config else stem_constructor_base
    
    # For block_constructor, ResNet_JAX expects it to be a callable that it will then call with n_hidden, strides, etc.
    # So, the block_kwargs_from_config should be passed to ResNet_JAX's block_kwargs parameter.

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=block_constructor_base, 
                         nn_width=nn_width_to_use,
                         nn_depth_config=depth_config_final,
                         output_size=output_size,
                         block_kwargs=block_kwargs_from_config) # These are expansion, groups, base_width etc.
  elif config.nn_model == "resnet18":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet18 model requires 'nn_width' in config.")
    
    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[18],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnetd18":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNetD18 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetDStem # Uses ResNetDStem
    model_specific_stem_kwargs = {} # ResNetDStem defaults (e.g. stem_width=32) are used unless overridden
    default_block_constructor_base = ResNetDBlock # Uses ResNetDBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[18],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnetd34":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNetD34 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetDStem
    model_specific_stem_kwargs = {} 
    default_block_constructor_base = ResNetDBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[34],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnet34":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet34 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[34],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnet50":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet50 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    # ResNetBottleneckBlock has internal default expansion=4, so model_specific_block_kwargs is empty
    # unless we want to change that default for a "standard" ResNet50.
    model_specific_block_kwargs = {} 

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[50],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnet101":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet101 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[101],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnet152":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet152 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[152],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnet200":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNet200 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[200],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "wideresnet50":
    if not hasattr(config, 'nn_width'):
        raise ValueError("WideResNet50 model requires 'nn_width' in config (e.g., 64 for typical WideResNet50-2x, where nn_width is k*base_width).")
        # Note: jax-resnet's ResNetBottleneckBlock takes `n_hidden` which is output channels of the block *before* expansion.
        # `config.nn_width` (e.g. 64) is passed to ResNet_JAX, then ResNet_JAX sets block's n_hidden.
        # For WideResNet-50-2k, the first stage output is 128 (64*2), bottleneck blocks operate on this.
        # The `expansion` in block_kwargs is for ResNetBottleneckBlock's internal expansion factor.
        # `nn_width` for ResNet_JAX should still be the base for the first stage (e.g., 64), and `k` (width factor)
        # is implicitly handled if `ResNetBottleneckBlock` correctly uses `base_width` or if `nn_width` is scaled.
        # The `jax-resnet` WideResNet typically has `width_factor` (k). `hidden_sizes` = [64 * k, 128 * k, 256 * k, 512 * k]
        # `ResNet_JAX` `nn_width` sets `current_n_hidden` for stage 1. If `k=2`, `nn_width` should be 64.
        # The `expansion=2` in `block_kwargs` might be confusing here.
        # `ResNetBottleneckBlock` default expansion is 4. WideResNet uses `expansion=4` typically.
        # The "width" comes from `n_hidden` (e.g., `64*k`) rather than changing `expansion`.
        # The existing code had `block_kwargs={'expansion': 2}`. This seems non-standard for WRN-50 if it's based on ResNet-50.
        # Standard WRN-50-2 uses expansion=4, and width_factor=2 (doubling channels in each ResNetBlock/Bottleneck).
        # Let's assume the user's previous intent for `{'expansion': 2}` was specific and preserve it as a model-specific default.

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {'expansion': 2} # Preserving previous hardcoded value as a model-specific default

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use, 
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width, 
                         nn_depth_config=STAGE_SIZES[50],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "wideresnet101":
    if not hasattr(config, 'nn_width'):
        raise ValueError("WideResNet101 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {'expansion': 2} # Preserving previous hardcoded value

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width, 
                         nn_depth_config=STAGE_SIZES[101],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnext50":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNeXt50 model requires 'nn_width' in config (e.g., 64).")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {'groups': 32, 'base_width': 4}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width, 
                         nn_depth_config=STAGE_SIZES[50],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnext101":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNeXt101 model requires 'nn_width' in config (e.g., 64).")

    default_stem_constructor_base = ResNetStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetBottleneckBlock
    model_specific_block_kwargs = {'groups': 32, 'base_width': 8}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width, 
                         nn_depth_config=STAGE_SIZES[101],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "conv2d":
    if not hasattr(config, 'nn_width') or not hasattr(config, 'nn_depth'):
        raise ValueError("Conv2D model requires 'nn_width' and 'nn_depth' in config.")
    model = Conv2D_JAX(nn_width=config.nn_width,
                       nn_depth=config.nn_depth,
                       output_size=output_size)
  elif config.nn_model == "resnetd50":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNetD50 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetDStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetDBottleneckBlock
    model_specific_block_kwargs = {} # Relies on ResNetDBottleneckBlock defaults (e.g. expansion=4)

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[50],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnetd101":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNetD101 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetDStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetDBottleneckBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[101],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnetd152":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNetD152 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetDStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetDBottleneckBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[152],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnetd200":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNetD200 model requires 'nn_width' in config.")

    default_stem_constructor_base = ResNetDStem
    model_specific_stem_kwargs = {}
    default_block_constructor_base = ResNetDBottleneckBlock
    model_specific_block_kwargs = {}

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[200],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnest50fast":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNeSt50Fast model requires 'nn_width' in config (e.g., 64).")
    
    default_stem_constructor_base = ResNetDStem
    # ResNeSt50-Fast typically uses stem_width=32 for its ResNetDStem.
    model_specific_stem_kwargs = {'stem_width': 32} 
    default_block_constructor_base = ResNeStBottleneckBlock
    # ResNeSt-Fast variants use avg_pool_first=True. Other ResNeSt params like radix=2, avd=True are defaults in ResNeStBottleneckBlock.
    model_specific_block_kwargs = {'avg_pool_first': True} 
    
    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}
    
    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}
    
    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs: 
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)

    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width,
                         nn_depth_config=STAGE_SIZES[50],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnest50":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNeSt50 model requires 'nn_width' in config (e.g., 64).")

    default_stem_constructor_base = ResNetDStem
    # Standard ResNeSt50 uses ResNetDStem, typically with stem_width=32 (default of ResNetDStem)
    model_specific_stem_kwargs = {'stem_width': 32} 
    default_block_constructor_base = ResNeStBottleneckBlock
    # ResNeStBottleneckBlock defaults: radix=2, groups=1, avd=True, avg_pool_first=False etc.
    # No specific overrides needed here for standard ResNeSt50 unless deviating from those.
    model_specific_block_kwargs = {} 

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}

    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width, 
                         nn_depth_config=STAGE_SIZES[50],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  elif config.nn_model == "resnest101":
    if not hasattr(config, 'nn_width'):
        raise ValueError("ResNeSt101 model requires 'nn_width' in config (e.g., 64).")

    default_stem_constructor_base = ResNetDStem
    # ResNeSt101 typically uses ResNetDStem with stem_width=64.
    model_specific_stem_kwargs = {'stem_width': 64}
    default_block_constructor_base = ResNeStBottleneckBlock
    model_specific_block_kwargs = {} # Rely on ResNeStBottleneckBlock defaults

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs}
    
    stem_constructor_to_use = default_stem_constructor_base
    if final_stem_kwargs:
        stem_constructor_to_use = functools.partial(default_stem_constructor_base, **final_stem_kwargs)
        
    model = ResNet_JAX(stem_constructor=stem_constructor_to_use,
                         block_constructor=default_block_constructor_base,
                         nn_width=config.nn_width, 
                         nn_depth_config=STAGE_SIZES[101],
                         output_size=output_size,
                         block_kwargs=final_block_kwargs)
  else:
    raise ValueError(f"Unsupported model type: {config.nn_model}")

  # Create dummy input (e.g., (1, *obs_shape)). MLP_JAX handles internal flattening if needed.
  dummy_input_shape = (1,) + tuple(observation_shape)
  dummy_input = jnp.zeros(dummy_input_shape, dtype=jnp.float32)

  # Initialize with training=False, as batch_stats are not typically initialized here
  # unless specific initial values are needed. They will be created during the first
  # training "apply" call if `mutable=['batch_stats']` is used.
  variables = model.init(key, dummy_input, training=False)

  return model, variables 