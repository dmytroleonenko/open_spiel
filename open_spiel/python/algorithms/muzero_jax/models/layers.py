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
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, activation: Callable = nnx.relu, output_activation: Callable = lambda x: x, init_zero: bool = False, use_bn: bool = True, noisy: bool = False, *, rngs: nnx.Rngs):
        super().__init__()
        self.layers = []
        self.noisy = noisy
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            rngs_l = nnx.Rngs(params=rngs.params()) # Get new key for each layer
            if i < len(sizes) - 2:
                # Use noisy linear layers if requested
                if noisy:
                    self.layers.append(NoisyLinear(sizes[i], sizes[i+1], std_init=0.5, rngs=rngs_l))
                else:
                    self.layers.append(nnx.Linear(sizes[i], sizes[i+1], rngs=rngs_l))
                if use_bn:
                    self.layers.append(nnx.BatchNorm(sizes[i+1], use_running_average=True, rngs=rngs_l))
                self.layers.append(activation)
            else: # Last layer
                # Use noisy linear layers if requested
                if noisy:
                    self.layers.append(NoisyLinear(sizes[i], sizes[i+1], std_init=0.5, rngs=rngs_l))
                else:
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

    def reset_noise(self, rng_key: jax.Array) -> None:
        """Reset noise in all noisy linear layers."""
        if not self.noisy:
            return
            
        # Split key for each noisy layer
        noisy_layers = [layer for layer in self.layers if isinstance(layer, NoisyLinear)]
        if not noisy_layers:
            return
            
        keys = jax.random.split(rng_key, len(noisy_layers))
        for layer, key in zip(noisy_layers, keys):
            layer.reset_noise(key)

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
            else: # nnx.Linear, NoisyLinear, or other callable modules
                x = layer(x)
        return x

class NoisyLinear(nnx.Module):
    """Noisy Linear layer for exploration as in TorchRL NoisyLinear.
    
    Implements factorized Gaussian noise as described in 
    "Noisy Networks for Exploration" (Fortunato et al., 2017).
    
    This layer replaces standard linear layers in policy networks to provide
    structured exploration through learnable noise parameters.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 std_init: float = 0.5,
                 use_bias: bool = True,
                 *, rngs: nnx.Rngs):
        """Initialize NoisyLinear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features  
            std_init: Initial standard deviation for noise parameters
            use_bias: Whether to use bias parameters
            rngs: Random number generators for parameter initialization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.use_bias = use_bias
        
        # Learnable weight parameters (mu and sigma)
        # Weight means (mu_w)
        self.weight_mu = nnx.Param(
            nnx.initializers.lecun_uniform()(
                rngs.params(), (out_features, in_features)
            )
        )
        
        # Weight noise standard deviations (sigma_w)
        bound = 1 / jnp.sqrt(in_features)
        self.weight_sigma = nnx.Param(
            jnp.full((out_features, in_features), std_init / jnp.sqrt(in_features))
        )
        
        if use_bias:
            # Bias means (mu_b) - use normal initialization for 1D bias
            bound = 1 / jnp.sqrt(in_features)
            self.bias_mu = nnx.Param(
                jax.random.uniform(
                    rngs.params(), (out_features,), minval=-bound, maxval=bound
                )
            )
            
            # Bias noise standard deviations (sigma_b)
            self.bias_sigma = nnx.Param(
                jnp.full((out_features,), std_init / jnp.sqrt(out_features))
            )
        else:
            self.bias_mu = None
            self.bias_sigma = None
            
        # Initialize noise variables (will be reset)
        self.weight_epsilon = nnx.Variable(jnp.zeros((out_features, in_features)))
        self.bias_epsilon = nnx.Variable(jnp.zeros((out_features,)) if use_bias else None)
        
        # Reset noise to initial state
        self.reset_noise(rngs.params())
    
    def _scale_noise(self, x: jax.Array) -> jax.Array:
        """Apply factorized Gaussian noise scaling: f(x) = sign(x) * sqrt(|x|)"""
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))
    
    def reset_noise(self, rng_key: jax.Array) -> None:
        """Reset the noise variables with new random values.
        
        Uses factorized Gaussian noise: epsilon_ij = f(epsilon_i) * f(epsilon_j)
        where f(x) = sign(x) * sqrt(|x|) and epsilon ~ N(0,1)
        
        Args:
            rng_key: JAX random key for generating new noise
        """
        key1, key2, key3 = jax.random.split(rng_key, 3)
        
        # Generate factorized noise for weights
        epsilon_in = jax.random.normal(key1, (self.in_features,))
        epsilon_out = jax.random.normal(key2, (self.out_features,))
        
        epsilon_in = self._scale_noise(epsilon_in)
        epsilon_out = self._scale_noise(epsilon_out)
        
        # Factorized noise: outer product
        self.weight_epsilon.value = jnp.outer(epsilon_out, epsilon_in)
        
        if self.use_bias:
            # Bias uses the same epsilon_out for factorized structure
            self.bias_epsilon.value = epsilon_out
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with noisy weights and biases.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Compute noisy weights: w = mu_w + sigma_w * epsilon_w
        weight = self.weight_mu.value + self.weight_sigma.value * self.weight_epsilon.value
        
        # Apply linear transformation
        output = jnp.dot(x, weight.T)
        
        if self.use_bias:
            # Compute noisy bias: b = mu_b + sigma_b * epsilon_b
            bias = self.bias_mu.value + self.bias_sigma.value * self.bias_epsilon.value
            output = output + bias
            
        return output

# TODO: Implement mlp function (Flax NNX style, possibly as a class or careful function) 