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
  def __call__(self, x: jax.Array, training: bool):
    # Torso
    for _ in range(self.nn_depth):
      x = nn.Dense(features=self.nn_width)(x)
      x = nn.relu(x)

    # Policy head
    policy_logits = nn.Dense(features=self.output_size, name="policy_head")(x)

    # Value head
    value_hidden = nn.Dense(features=self.nn_width, name="value_hidden")(x)
    value_hidden = nn.relu(value_hidden)
    value_output = nn.Dense(features=1, name="value_output")(value_hidden)
    value_output = jnp.tanh(value_output) # Use jnp.tanh for JAX 

    return (policy_logits, value_output)

class ResNet_JAX(nn.Module):
  stem_constructor: Callable[[], nn.Module]
  block_constructor: Callable[..., nn.Module]
  nn_width: int  # Base filter count for the first stage, e.g., 64. Also for value head's hidden layer.
  nn_depth_config: list[int]  # Stage sizes, e.g., [2, 2, 2, 2] for ResNet18 layer count per stage.
  output_size: int # Number of actions for policy head
  block_kwargs: Optional[Mapping] = None # For expansion, groups, base_width

  @nn.compact
  def __call__(self, x: jax.Array, training: bool):
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

    x = jnp.mean(x, axis=(1, 2))
    policy_logits = nn.Dense(features=self.output_size, name="policy_head")(x)
    value_hidden = nn.Dense(features=self.nn_width, name="value_hidden")(x)
    value_hidden = nn.relu(value_hidden)
    value_output = nn.Dense(features=1, name="value_output")(value_hidden)
    value_output = jnp.tanh(value_output)
    return (policy_logits, value_output)

class Conv2D_JAX(nn.Module):
  nn_width: int  # Number of filters for convolutional layers
  nn_depth: int  # Number of convolutional layers in the torso
  output_size: int # Number of actions for policy head

  @nn.compact
  def __call__(self, x: jax.Array, training: bool):
    # Torso
    # The `training` flag will be used by ConvBlock for its BatchNorm layers.
    current_filters = self.nn_width
    for i in range(self.nn_depth):
      # Optionally, increase filters or use different kernel sizes/strides for deeper layers.
      # For simplicity, using same kernel size, padding, and filter count (self.nn_width) for all conv layers here.
      # Stride is 1 to maintain dimensions before flatten, unless downsampling is desired.
      # If input features change (e.g. image channels), first layer might be different.
      # We assume x has appropriate shape (e.g. HxWxC)
      x = ConvBlock(features=current_filters, 
                    kernel_size=(3, 3), 
                    strides=(1, 1), 
                    padding='SAME', 
                    name=f'conv_block_{i}')(x, training=training)
      # Example: Double filters at some point if desired
      # if i == self.nn_depth // 2: current_filters *= 2
      
    # Flatten the output of conv layers before passing to Dense layers for heads
    x = nn.Flatten()(x)

    # Policy head
    policy_logits = nn.Dense(features=self.output_size, name="policy_head")(x)

    # Value head
    # Using self.nn_width for hidden layer size, similar to MLP and ResNet value head (as per plan)
    value_hidden = nn.Dense(features=self.nn_width, name="value_hidden")(x)
    value_hidden = nn.relu(value_hidden)
    value_output = nn.Dense(features=1, name="value_output")(value_hidden)
    value_output = jnp.tanh(value_output)

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
  # Ensure observation_shape is a tuple of integers
  # For games like Tic-Tac-Toe, observation_tensor_shape() might return [9]
  # but Flax expects a shape like (9,)
  if isinstance(observation_shape, list):
      if len(observation_shape) == 1: # e.g. [9] for tic-tac-toe
          # For a flat observation vector, Flax Dense layers expect (batch_size, features)
          # The dummy input later will be (1, *observation_shape), e.g., (1, 9)
          # So, the feature size is just observation_shape[0]
          # No specific shape adjustment needed here for MLP if it flattens anyway or expects flat input.
          pass # Keep as is, dummy_input will handle it.
      elif len(observation_shape) > 1: # e.g. [3, 3, 1] for an image-like observation
          # This is fine for ConvNets. For MLPs, it would typically be flattened.
          # The MLP model defined above takes x and passes it to Dense, which flattens if needed.
          pass 

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
    # Use new ConfigJAX fields for generic ResNet
    depth_config = getattr(config, 'resnet_depth_config', None)
    stem_name = getattr(config, 'resnet_stem_callable_name', None)
    stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    block_name = getattr(config, 'resnet_block_callable_name', None)
    block_kwargs_from_config = getattr(config, 'resnet_block_kwargs', None) or {}

    if not depth_config or not stem_name or not block_name or not hasattr(config, 'nn_width'):
        raise ValueError(
            "For nn_model='resnet', config must provide: nn_width, resnet_depth_config, "
            "resnet_stem_callable_name, resnet_block_callable_name.")

    if stem_name not in known_stem_constructors:
        raise ValueError(f"Unknown resnet_stem_callable_name: {stem_name}. Known: {list(known_stem_constructors.keys())}")
    if block_name not in known_block_constructors:
        raise ValueError(f"Unknown resnet_block_callable_name: {block_name}. Known: {list(known_block_constructors.keys())}")

    stem_constructor_base = known_stem_constructors[stem_name]
    block_constructor_base = known_block_constructors[block_name]

    # Apply kwargs using functools.partial
    stem_constructor = functools.partial(stem_constructor_base, **stem_kwargs) if stem_kwargs else stem_constructor_base
    # For block_constructor, ResNet_JAX expects it to be a callable that it will then call with n_hidden, strides, etc.
    # So, the block_kwargs_from_config should be passed to ResNet_JAX's block_kwargs parameter.

    model = ResNet_JAX(stem_constructor=stem_constructor,
                         block_constructor=block_constructor_base, # Pass the base, ResNet_JAX will apply its own n_hidden, strides
                         nn_width=config.nn_width,
                         nn_depth_config=depth_config,
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
    # ResNeXt50 (32x4d) specifics:
    model_specific_block_kwargs = {'groups': 32, 'base_width': 4} 

    config_stem_kwargs = getattr(config, 'resnet_stem_kwargs', None) or {}
    config_block_kwargs = getattr(config, 'resnet_block_kwargs', None) or {}

    final_stem_kwargs = {**model_specific_stem_kwargs, **config_stem_kwargs}
    final_block_kwargs = {**model_specific_block_kwargs, **config_block_kwargs} # Config can override groups/base_width

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
    # ResNeXt101 (32x8d) specifics used here, from previous hardcoding.
    # Other common is 64x4d. The `base_width` param in `ResNetBottleneckBlock` is `bottleneck_width_per_group`.
    # So `base_width=8` with `groups=32` implies total bottleneck width of 256.
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
    # nn_depth and nn_width from config are used for Conv2D_JAX
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

  # Create a dummy input matching the expected batch size and observation shape.
  # For MLP, if observation_shape is [9], this becomes (1, 9).
  # If observation_shape is [3,3,1], this becomes (1, 3, 3, 1).
  # Flax Dense layers automatically handle flattening if the input is multi-dimensional
  # beyond the batch dimension, but it's common to explicitly flatten for MLPs if input is image-like.
  # The current MLP_JAX does not explicitly flatten, but Dense will work.
  dummy_input_shape = (1,) + tuple(observation_shape)
  dummy_input = jnp.zeros(dummy_input_shape, dtype=jnp.float32)

  # Initialize with training=False, as batch_stats are not typically initialized here
  # unless specific initial values are needed. They will be created during the first
  # training "apply" call if `mutable=['batch_stats']` is used.
  variables = model.init(key, dummy_input, training=False)

  return model, variables 