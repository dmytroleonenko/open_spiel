import flax.nnx as nnx
import jax
import jax.numpy as jnp

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
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int,
                 activation_fn = nnx.relu, # Changed from nn.ELU in reference for common JAX/Flax use
                 output_activation_fn = lambda x: x,
                 init_zero: bool = False,
                 *, rngs: nnx.Rngs):
        super().__init__()
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        # All layers within this MLP will use the same rngs object passed to __init__.
        # NNX RngStreams are stateful and will provide unique keys for each layer.
        for i, (in_s, out_s) in enumerate(zip(sizes[:-1], sizes[1:])):
            if i < len(sizes) - 2: # Hidden layers
                self.layers.append(nnx.Linear(in_s, out_s, rngs=rngs))
                self.layers.append(nnx.BatchNorm(num_features=out_s, use_running_average=True, rngs=rngs))
                self.layers.append(activation_fn) # Store the function itself
            else: # Output layer
                output_layer = nnx.Linear(in_s, out_s, rngs=rngs)
                if init_zero:
                    output_layer = nnx.Linear(in_s, out_s, 
                                              kernel_init=jax.nn.initializers.zeros,
                                              bias_init=jax.nn.initializers.zeros,
                                              rngs=rngs)
                self.layers.append(output_layer)
                self.layers.append(output_activation_fn)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        for layer in self.layers:
            if isinstance(layer, nnx.BatchNorm):
                x = layer(x, use_running_average=not training)
            elif callable(layer) and not isinstance(layer, nnx.Module): # Activation function
                x = layer(x)
            else: # Linear layer
                x = layer(x)
        return x

# TODO: Implement mlp function (Flax NNX style, possibly as a class or careful function) 