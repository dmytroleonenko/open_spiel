import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from open_spiel.python.algorithms.muzero_jax.models.layers import conv3x3, ResidualBlock, FCResidualBlock, MLP
from open_spiel.python.algorithms.muzero_jax.models.network import DownSample

@pytest.fixture
def rngs():
    return nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)) # Add other keys if needed

# Test conv3x3
def test_conv3x3_output_shape(rngs):
    conv_layer = conv3x3(in_channels=3, out_channels=16, stride=1, rngs=rngs)
    # Input shape: (batch_size, height, width, channels)
    # JAX/Flax usually expects channels_last format for Conv
    dummy_input = jnp.ones((1, 32, 32, 3)) 
    output = conv_layer(dummy_input)
    assert output.shape == (1, 32, 32, 16) # Stride 1, SAME padding
    assert hasattr(conv_layer, 'kernel')

def test_conv3x3_output_shape_stride2(rngs):
    conv_layer = conv3x3(in_channels=3, out_channels=16, stride=2, rngs=rngs)
    dummy_input = jnp.ones((1, 32, 32, 3))
    output = conv_layer(dummy_input)
    assert output.shape == (1, 16, 16, 16) # Stride 2, SAME padding halves dimensions
    assert hasattr(conv_layer, 'kernel')

# Test ResidualBlock
@pytest.mark.parametrize("stride, in_channels, out_channels", [
    (1, 16, 16),
    (2, 16, 32), # Test with downsampling (stride 2, different channels)
    (1, 16, 32), # Test with different channels, no downsampling via stride
])
def test_residual_block_output_shape(stride, in_channels, out_channels, rngs):
    downsample_layer = None
    input_h, input_w = 32, 32
    output_h, output_w = input_h // stride, input_w // stride

    if stride != 1 or in_channels != out_channels:
        # Create a downsample layer if stride is not 1 or channels change
        downsample_layer = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            strides=(stride, stride),
            use_bias=False,
            rngs=rngs
        )

    res_block = ResidualBlock(in_channels, out_channels, downsample_conv=downsample_layer, stride=stride, rngs=rngs)
    dummy_input = jnp.ones((1, input_h, input_w, in_channels))
    
    # Test training mode
    output_train = res_block(dummy_input, training=True)
    assert output_train.shape == (1, output_h, output_w, out_channels)

    # Test inference mode
    output_eval = res_block(dummy_input, training=False)
    assert output_eval.shape == (1, output_h, output_w, out_channels)

    # Check that parameters are created
    assert hasattr(res_block.conv1, 'kernel')
    assert hasattr(res_block.bn1, 'scale') 
    assert hasattr(res_block.bn1, 'bias')
    # Batch stats (mean, var) are created/updated during the forward pass when training=True (use_running_average=False)
    # A forward pass in training mode ensures they exist.
    _ = res_block(dummy_input, training=True)
    assert hasattr(res_block.bn1, 'mean')
    assert hasattr(res_block.bn1, 'var')

    if downsample_layer:
         assert hasattr(res_block.downsample, 'kernel')

def test_residual_block_runs_without_downsample(rngs):
    res_block = ResidualBlock(in_channels=16, out_channels=16, stride=1, rngs=rngs)
    dummy_input = jnp.ones((1, 32, 32, 16))
    output = res_block(dummy_input, training=True)
    assert output.shape == (1, 32, 32, 16)
    output_eval = res_block(dummy_input, training=False)
    assert output_eval.shape == (1, 32, 32, 16)

    # Check that parameters are created
    assert hasattr(res_block.conv1, 'kernel')
    assert hasattr(res_block.bn1, 'scale') 
    assert hasattr(res_block.bn1, 'bias')
    # Batch stats (mean, var) are created/updated during the forward pass when training=True (use_running_average=False)
    # A forward pass in training mode ensures they exist.
    _ = res_block(dummy_input, training=True)
    assert hasattr(res_block.bn1, 'mean')
    assert hasattr(res_block.bn1, 'var')

# Test FCResidualBlock
@pytest.mark.parametrize("input_shape, hidden_shape", [
    (64, 128),
    (128, 128),
])
def test_fc_residual_block_output_shape(input_shape, hidden_shape, rngs):
    fc_res_block = FCResidualBlock(input_shape, hidden_shape, rngs=rngs)
    # Input shape: (batch_size, features)
    dummy_input = jnp.ones((1, input_shape))

    # Test training mode
    output_train = fc_res_block(dummy_input, training=True)
    assert output_train.shape == (1, input_shape) # Output shape matches input for residual

    # Test inference mode
    output_eval = fc_res_block(dummy_input, training=False)
    assert output_eval.shape == (1, input_shape)

    # Check that parameters are created
    assert hasattr(fc_res_block.linear1, 'kernel')
    assert hasattr(fc_res_block.linear1, 'bias')
    assert hasattr(fc_res_block.bn1, 'scale')
    assert hasattr(fc_res_block.bn1, 'bias')
    _ = fc_res_block(dummy_input, training=True) # Ensure batch stats are created
    assert hasattr(fc_res_block.bn1, 'mean')
    assert hasattr(fc_res_block.bn1, 'var')

# Test MLP class
@pytest.mark.parametrize("hidden_sizes, init_zero", [
    ([], False), # No hidden layers
    ([128], False),
    ([128, 64], False),
    ([128, 64], True), # Test with zero initialization for output layer
])
def test_mlp_output_shape(hidden_sizes, init_zero, rngs):
    input_size = 32
    output_size = 10
    mlp_model = MLP(input_size, hidden_sizes, output_size, init_zero=init_zero, rngs=rngs)
    dummy_input = jnp.ones((1, input_size)) # Batch size 1

    # Test training mode
    output_train = mlp_model(dummy_input, training=True)
    assert output_train.shape == (1, output_size)

    # Test inference mode
    output_eval = mlp_model(dummy_input, training=False)
    assert output_eval.shape == (1, output_size)

    # Check parameter creation for the first Linear layer and first BatchNorm (if hidden_sizes > 0)
    if mlp_model.layers:
        first_linear = next((layer for layer in mlp_model.layers if isinstance(layer, nnx.Linear)), None)
        first_bn = next((layer for layer in mlp_model.layers if isinstance(layer, nnx.BatchNorm)), None)
        if first_linear:
            assert hasattr(first_linear, 'kernel')
            assert hasattr(first_linear, 'bias')
        if first_bn:
            assert hasattr(first_bn, 'scale')
            assert hasattr(first_bn, 'bias')
            _ = mlp_model(dummy_input, training=True) # Ensure batch stats created for BN
            assert hasattr(first_bn, 'mean')
            assert hasattr(first_bn, 'var')

    # Check zero initialization if enabled
    if init_zero:
        # Assuming the output layer is the second to last element in self.layers 
        # (last is activation_fn)
        output_linear_layer = None
        # Find the last linear layer
        for layer in reversed(mlp_model.layers):
            if isinstance(layer, nnx.Linear):
                output_linear_layer = layer
                break
        
        assert output_linear_layer is not None, "Could not find output linear layer for zero_init check"
        assert isinstance(output_linear_layer, nnx.Linear)
        # .value is needed to get the actual JAX array for parameters in NNX
        assert jnp.all(output_linear_layer.kernel.value == 0)
        assert jnp.all(output_linear_layer.bias.value == 0)

def test_mlp_custom_activations(rngs):
    input_size = 16
    output_size = 4
    mlp_model = MLP(
        input_size, 
        hidden_sizes=[8], 
        output_size=output_size, 
        activation_fn=nnx.sigmoid, 
        output_activation_fn=nnx.tanh,
        rngs=rngs
    )
    dummy_input = jnp.ones((1, input_size))
    output = mlp_model(dummy_input, training=True)
    assert output.shape == (1, output_size)
    # Further tests could check ranges if activations constrain them, e.g., tanh output between -1 and 1
    assert jnp.all(output >= -1) and jnp.all(output <= 1)

# TODO: Add tests for mlp function/class once implemented 

# Add test for DownSample
def test_downsample_output_shape_and_mode(rngs):
    """
    Test DownSample on image inputs, covering both training and evaluation modes.
    """
    in_channels = 3
    out_channels = 8  # Must be divisible by 2 for simplified downsample
    ds = DownSample(in_channels, out_channels, rngs=rngs)
    # Input size divisible by 4 to allow two stride-2 downsamples
    dummy_input = jnp.ones((1, 64, 64, in_channels))
    # Training mode
    out_train = ds(dummy_input, training=True)
    # Expect two stride-2 downsamples: 64 -> 32 -> 16, channels -> out_channels
    assert out_train.shape == (1, 16, 16, out_channels)
    # Evaluation mode
    out_eval = ds(dummy_input, training=False)
    assert out_eval.shape == (1, 16, 16, out_channels) 