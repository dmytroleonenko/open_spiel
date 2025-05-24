import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Callable

# Flax NNX equivalent of EfficientZeroV2's conv3x3
def conv3x3(in_channels: int, out_channels: int, stride: int = 1, *, rngs: nnx.Rngs):
    """3x3 convolution with padding, adapted for Flax NNX."""
    return nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding='SAME',  # 'SAME' padding tries to match output size to input size (for stride=1)
        use_bias=False,
        rngs=rngs
    )

class ResidualBlock(nnx.Module):
    """Post Activated Residual block (Flax NNX version)."""
    def __init__(self, in_channels: int, out_channels: int, downsample_conv: nnx.Conv | None = None, stride: int = 1, *, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=out_channels, use_running_average=True, rngs=rngs) # use_running_average for eval mode by default
        self.conv2 = conv3x3(out_channels, out_channels, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=out_channels, use_running_average=True, rngs=rngs)
        self.downsample = downsample_conv

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        identity = x

        out = self.conv1(x)
        # For BatchNorm, use_running_average should be True during inference/evaluation and False during training.
        # The 'training' flag here controls this behavior directly.
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = nnx.relu(out)
        return out

class FCResidualBlock(nnx.Module):
    """Fully Connected Residual block (Flax NNX version)."""
    def __init__(self, input_shape: int, hidden_shape: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(input_shape, hidden_shape, rngs=rngs)
        # For nnx.BatchNorm, feature dimension is the last one for FC layers.
        self.bn1 = nnx.BatchNorm(num_features=hidden_shape, use_running_average=True, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_shape, input_shape, rngs=rngs) # Output matches input for residual
        self.bn2 = nnx.BatchNorm(num_features=input_shape, use_running_average=True, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        identity = x

        out = self.linear1(x)
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)

        out = self.linear2(out)
        out = self.bn2(out, use_running_average=not training)

        out += identity
        out = nnx.relu(out)
        return out

class MLP(nnx.Module):
    """MLP (Multi-Layer Perceptron) class for Flax NNX."""
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, activation: Callable = nnx.relu, output_activation: Callable = lambda x: x, init_zero: bool = False, use_bn: bool = True, *, rngs: nnx.Rngs):
        super().__init__()
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            rngs_l = nnx.Rngs(params=rngs.params()) # Get new key for each layer
            if i < len(sizes) - 2:
                self.layers.append(nnx.Linear(sizes[i], sizes[i+1], rngs=rngs_l))
                if use_bn:
                    self.layers.append(nnx.BatchNorm(sizes[i+1], use_running_average=True, rngs=rngs_l))
                self.layers.append(activation)
            else: # Last layer
                self.layers.append(nnx.Linear(sizes[i], sizes[i+1], rngs=rngs_l))
                self.layers.append(output_activation)

        if init_zero and isinstance(self.layers[-2], nnx.Linear):
            # Find the last Linear layer to initialize its weights and biases to zero
            # This assumes output_activation is identity or similar, so -2 is Linear
            # More robustly, find the last actual Linear layer instance
            last_linear_layer = None
            for layer in reversed(self.layers):
                if isinstance(layer, nnx.Linear):
                    last_linear_layer = layer
                    break
            if last_linear_layer is not None:
                last_linear_layer.kernel.value = jnp.zeros_like(last_linear_layer.kernel.value)
                if last_linear_layer.bias is not None:
                    last_linear_layer.bias.value = jnp.zeros_like(last_linear_layer.bias.value)
            else: # pragma: no cover
                print("Warning: init_zero=True but no Linear layer found as second to last layer in MLP for zero init.")

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        # Temporary print for debugging shapes
        # print(f"MLP input x shape: {x.shape}") 
        # Ensure this print is added before the loop that applies layers.

        for i, layer in enumerate(self.layers):
            # if i == 0: # Before the first Linear layer
            #     print(f"MLP __call__: input x shape to first Linear layer: {x.shape}") # Removed this debug print
            if isinstance(layer, nnx.BatchNorm):
                x = layer(x, use_running_average=not training)
            elif layer == nnx.relu or (hasattr(layer, '__name__') and layer.__name__ == '<lambda>'): # activation functions, check for lambda for identity
                x = layer(x)
            else: # nnx.Linear or other callable modules
                x = layer(x)
        return x

# TODO: Implement mlp function (Flax NNX style, possibly as a class or careful function) 