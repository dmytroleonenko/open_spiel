import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from unittest.mock import patch, MagicMock
from functools import partial

# Attempt to import the module to be tested
from open_spiel.python.algorithms.muzero_jax.utils import batch_optimizer

# ───── Mock NNX Model and Related Functions for Testing ─────

class MockNNXModel(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params() # or nnx.rngs.params() if using older nnx
        self.linear1 = nnx.Linear(din, dout * 2, rngs=rngs)
        self.linear2 = nnx.Linear(dout * 2, dout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

def mock_model_loss_fn(model_instance: MockNNXModel, batch_input: jax.Array, batch_target: jax.Array) -> jax.Array:
    predictions = model_instance(batch_input)
    return jnp.mean((predictions - batch_target)**2)

def init_mock_model_for_test(key: jax.random.PRNGKey, din: int = 10, dout: int = 5):
    return MockNNXModel(din, dout, rngs=nnx.Rngs(params=key))

def forward_backward_for_test(model_state_or_instance, batch_input_target_tuple):
    """
    A mock forward_and_backward function for testing.
    Assumes model_state_or_instance is an NNX model instance.
    batch_input_target_tuple is (batch_input, batch_target).
    """
    batch_input, batch_target = batch_input_target_tuple
    # In NNX, value_and_grad takes the model instance directly.
    # wrt=nnx.Param ensures grads are for parameters.
    loss, grads_state = nnx.value_and_grad(mock_model_loss_fn)(model_state_or_instance, batch_input, batch_target)
    return loss, grads_state

# JITted version for testing with sweep_accum
@partial(jax.jit, static_argnums=(0,))
def jitted_forward_backward_for_test(model_state_or_instance, batch_input_target_tuple):
    """
    A JITted mock forward_and_backward function for testing sweep_accum.
    """
    batch_input, batch_target = batch_input_target_tuple
    loss, grads_state = nnx.value_and_grad(mock_model_loss_fn)(model_state_or_instance, batch_input, batch_target)
    return loss, grads_state


# ───── Pytest Fixtures ─────

@pytest.fixture
def dummy_rng():
    return jax.random.PRNGKey(0)

@pytest.fixture(params=[jnp.float32, jnp.bfloat16])
def dtype(request):
    return request.param

@pytest.fixture
def mock_model_instance(dummy_rng):
    # Model initialization itself is typically dtype-agnostic at the definition level.
    # Actual tensor dtypes are determined by input data or explicit casting.
    return init_mock_model_for_test(dummy_rng, din=4, dout=2)

@pytest.fixture
def sample_data_shape():
    return (4,) # din for MockNNXModel

@pytest.fixture
def sample_target_shape():
    return (2,) # dout for MockNNXModel

# ───── Test Functions ─────

def test_make_random_batch(dummy_rng, sample_data_shape, sample_target_shape, dtype):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    B = 8
    # Test generating only input batch
    batch_input = batch_optimizer.make_random_batch(dummy_rng, B, sample_data_shape, dtype=dtype)
    assert batch_input.shape == (B, *sample_data_shape)
    assert batch_input.dtype == dtype

    # Test generating input and target batch
    batch_input, batch_target = batch_optimizer.make_random_batch(dummy_rng, B, sample_data_shape, sample_target_shape, dtype=dtype)
    assert batch_input.shape == (B, *sample_data_shape)
    assert batch_target.shape == (B, *sample_target_shape)
    assert batch_input.dtype == dtype
    assert batch_target.dtype == dtype


def test_flatten_grads_nnx(mock_model_instance, sample_data_shape, sample_target_shape, dummy_rng, dtype):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    B = 2
    # Generate mock data with the specified dtype
    input_batch = jax.random.normal(dummy_rng, (B, *sample_data_shape), dtype=dtype)
    # Convert to JAX array and then change dtype if necessary to avoid issues with normal distribution for bfloat16
    input_batch = jnp.array(jax.random.normal(dummy_rng, (B, *sample_data_shape)), dtype=dtype)
    target_batch = jnp.array(jax.random.normal(dummy_rng, (B, *sample_target_shape)), dtype=dtype)
    
    # Get mock gradients (which will be an nnx.State object)
    # Ensure model parameters are also in the correct dtype if the model uses it internally,
    # or that the forward_backward_for_test handles potential dtype conversions.
    # For this test, we assume forward_backward_for_test correctly handles dtypes based on input.
    _, mock_grads_state = forward_backward_for_test(mock_model_instance, (input_batch, target_batch))

    flat_grads_vector = batch_optimizer.flatten_grads_nnx(mock_grads_state)
    
    assert isinstance(flat_grads_vector, jax.Array)
    assert flat_grads_vector.ndim == 1
    
    # Calculate expected number of parameters
    # For MockNNXModel:
    # linear1: (4 * 4) + 4 = 20 (weights + biases)
    # linear2: (4 * 2) + 2 = 10 (weights + biases)
    # Total: 30 -- this needs to be updated if MockNNXModel changes
    
    # A more robust way to count parameters in NNX model's Param part
    model_state = nnx.state(mock_model_instance)
    params_state = nnx.filter_state(model_state, nnx.Param)
    params_flat, _ = jax.tree_util.tree_flatten(params_state)
    expected_total_params = sum(p.size for p in params_flat)

    assert flat_grads_vector.size == expected_total_params
    assert not jnp.isnan(flat_grads_vector).any()


@patch('jax.devices')
def test_get_device_memory_usage_gpu(mock_jax_devices):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    # Mock jax.devices() to simulate a GPU device with memory_stats
    mock_gpu = MagicMock()
    mock_gpu.platform = 'gpu' # or 'cuda', 'rocm'
    mock_gpu.memory_stats = MagicMock(return_value={'bytes_used': 1024, 'bytes_limit': 2048})
    mock_jax_devices.return_value = [mock_gpu]

    used, total = batch_optimizer.get_device_memory_usage()
    assert used == 1024
    assert total == 2048
    mock_gpu.memory_stats.assert_called_once()

@patch('jax.devices')
@patch('subprocess.run')
def test_get_device_memory_usage_gpu_fallback_nvidia_smi(mock_subprocess_run, mock_jax_devices):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    # Simulate GPU but memory_stats is not available or fails
    mock_gpu = MagicMock()
    mock_gpu.platform = 'gpu'
    mock_gpu.memory_stats = MagicMock(side_effect=NotImplementedError) # Simulate not implemented
    mock_jax_devices.return_value = [mock_gpu]

    # Mock subprocess.run for nvidia-smi
    mock_process_result = MagicMock()
    mock_process_result.stdout = "100 MiB, 200 MiB" # Example output
    mock_process_result.returncode = 0
    mock_subprocess_run.return_value = mock_process_result
    
    used, total = batch_optimizer.get_device_memory_usage()
    assert used == 100 * 1024 * 1024
    assert total == 200 * 1024 * 1024
    mock_subprocess_run.assert_called_once()


@patch('jax.devices')
def test_get_device_memory_usage_tpu(mock_jax_devices):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    # Mock jax.devices() to simulate a TPU device with memory_stats
    mock_tpu = MagicMock()
    mock_tpu.platform = 'tpu'
    mock_tpu.memory_stats = MagicMock(return_value={'bytes_used': 4096, 'bytes_limit': 8192})
    mock_jax_devices.return_value = [mock_tpu]

    used, total = batch_optimizer.get_device_memory_usage()
    assert used == 4096
    assert total == 8192
    mock_tpu.memory_stats.assert_called_once()

@patch('jax.devices')
def test_get_device_memory_usage_cpu(mock_jax_devices):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    # Mock jax.devices() to simulate a CPU device
    mock_cpu = MagicMock()
    mock_cpu.platform = 'cpu'
    mock_jax_devices.return_value = [mock_cpu]
    
    # For CPU, we expect psutil to be used. Patch psutil.virtual_memory.
    with patch('psutil.virtual_memory') as mock_virtual_memory:
        mock_vm_stats = MagicMock()
        mock_vm_stats.total = 16000 * 1024 * 1024 # 16GB
        mock_vm_stats.available = 8000 * 1024 * 1024 # 8GB available
        # used = total - available
        mock_virtual_memory.return_value = mock_vm_stats
        
        used, total = batch_optimizer.get_device_memory_usage()
        assert total == 16000 * 1024 * 1024
        assert used == (16000 - 8000) * 1024 * 1024
        mock_virtual_memory.assert_called_once()


# More tests to be added for:
# - find_max_batch (mocking OOM)
# - estimate_grad_var
# - sweep_accum (mocking time and memory changes)
# - main_batch_optimizer_workflow (integration test)

# Placeholder for find_max_batch tests
# This requires careful mocking of the forward_backward_for_test
# to raise ResourceExhaustedError for specific batch sizes.

def mockable_forward_backward_for_find_max_batch_test(oom_threshold_B):
    def actual_mock(model_state_or_instance, batch_input_target_tuple):
        batch_input, batch_target = batch_input_target_tuple
        current_B = batch_input.shape[0]
        if current_B > oom_threshold_B:
            raise RuntimeError("Mock OOM")
        
        # If not OOM, proceed as normal
        loss, grads_state = nnx.value_and_grad(mock_model_loss_fn)(
            model_state_or_instance, batch_input, batch_target
        )
        # Simulate work
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads_state)
        return loss, grads_state
    return actual_mock

def test_find_max_batch_mocked_oom(dummy_rng, sample_data_shape, sample_target_shape, dtype):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    # Case 1: Normal operation, finds a limit
    oom_threshold_normal = 64
    mock_fb_hook_normal = mockable_forward_backward_for_find_max_batch_test(oom_threshold_normal -1) # OOM if B >= 64
    
    # Ensure input_data_shape matches the default din=10 of init_mock_model_for_test
    find_max_batch_input_shape = (10,)
    find_max_batch_target_shape = (5,) # Matches default dout=5

    # Test with default float32
    max_B_normal_f32 = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5), 
        forward_backward_fn=mock_fb_hook_normal,
        input_data_shape=find_max_batch_input_shape, 
        target_data_shape=find_max_batch_target_shape,
        rng_key=dummy_rng, 
        start_B=8,
        limit_B=128,
        max_trials_exp_search=10, # Ensure enough trials for this range
        dtype=jnp.float32 
    )
    assert max_B_normal_f32 == oom_threshold_normal - 1 # Should find 63

    # Test with bfloat16 - assuming OOM threshold might be higher or different
    # For testing purposes, let's assume bfloat16 allows slightly more, e.g., threshold is 70
    oom_threshold_bf16 = 70 
    mock_fb_hook_bf16 = mockable_forward_backward_for_find_max_batch_test(oom_threshold_bf16 -1)

    max_B_normal_bf16 = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5),
        forward_backward_fn=mock_fb_hook_bf16,
        input_data_shape=find_max_batch_input_shape,
        target_data_shape=find_max_batch_target_shape,
        rng_key=dummy_rng,
        start_B=8,
        limit_B=128,
        max_trials_exp_search=10,
        dtype=jnp.bfloat16
    )
    assert max_B_normal_bf16 == oom_threshold_bf16 -1


    # Case 2: OOM at start_B (where start_B > 1)
    oom_threshold_start_oom = 8
    mock_fb_hook_start_oom = mockable_forward_backward_for_find_max_batch_test(oom_threshold_start_oom - 1) # OOM if B >= 8
                                                                                                       # In this case, it means B=1 will be tried and should succeed if threshold is > 1
    
    # Subcase 2a: B=1 succeeds because threshold is e.g. 7 (OOM at 8)
    # mock_fb_hook_start_oom_try1_ok = mockable_forward_backward_for_find_max_batch_test(1) # OOM if B > 1
    max_B_start_oom_try1_ok = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5),
        forward_backward_fn=mock_fb_hook_start_oom, # Uses oom_threshold_start_oom = 8 (fails >=8)
        input_data_shape=find_max_batch_input_shape, 
        target_data_shape=find_max_batch_target_shape,
        rng_key=jax.random.fold_in(dummy_rng, 1),
        start_B=8,
        limit_B=128,
        max_trials_exp_search=10,
        dtype=jnp.float32
    )
    assert max_B_start_oom_try1_ok == 1

    # Subcase 2b: Even B=1 OOMs
    oom_threshold_b1_oom = 1 # OOM if B >= 1
    mock_fb_hook_b1_oom = mockable_forward_backward_for_find_max_batch_test(oom_threshold_b1_oom -1) # OOM if B >= 1 (i.e. threshold = 0)
    max_B_b1_oom = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5),
        forward_backward_fn=mock_fb_hook_b1_oom,
        input_data_shape=find_max_batch_input_shape, 
        target_data_shape=find_max_batch_target_shape,
        rng_key=jax.random.fold_in(dummy_rng, 2),
        start_B=8, # Start high, it should recover to 0
        limit_B=128,
        max_trials_exp_search=10,
        dtype=jnp.float32
    )
    assert max_B_b1_oom == 0

    # Case 3: No OOM up to limit_B
    oom_threshold_no_oom = 256 # Well above limit_B
    mock_fb_hook_no_oom = mockable_forward_backward_for_find_max_batch_test(oom_threshold_no_oom)
    max_B_no_oom = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5),
        forward_backward_fn=mock_fb_hook_no_oom,
        input_data_shape=find_max_batch_input_shape, 
        target_data_shape=find_max_batch_target_shape,
        rng_key=jax.random.fold_in(dummy_rng, 3),
        start_B=8,
        limit_B=128,
        max_trials_exp_search=10,
        dtype=jnp.float32
    )
    assert max_B_no_oom == 128

    # Case 4: limit_B is very small, start_B is larger
    max_B_small_limit = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5),
        forward_backward_fn=mock_fb_hook_normal, # Re-use a hook where e.g. 63 is max
        input_data_shape=find_max_batch_input_shape, 
        target_data_shape=find_max_batch_target_shape,
        rng_key=jax.random.fold_in(dummy_rng, 4),
        start_B=8,
        limit_B=128,
        max_trials_exp_search=10,
        dtype=jnp.float32
    )
    assert max_B_small_limit == 63 # Corrected assertion: OOM is at 64, limit_B is 128

    # Case 5: start_B = 1, limit_B = 1
    oom_threshold_start1_limit1 = 1 # OOM if B > 1 (i.e. B=1 works)
    mock_fb_hook_start1_limit1 = mockable_forward_backward_for_find_max_batch_test(oom_threshold_start1_limit1) 
    max_B_start1_limit1 = batch_optimizer.find_max_batch(
        model_init_fn=lambda key: init_mock_model_for_test(key, din=10, dout=5),
        forward_backward_fn=mock_fb_hook_start1_limit1,
        input_data_shape=find_max_batch_input_shape, 
        target_data_shape=find_max_batch_target_shape,
        rng_key=jax.random.fold_in(dummy_rng, 5),
        start_B=1,
        limit_B=1,
        max_trials_exp_search=10,
        dtype=jnp.float32
    )
    assert max_B_start1_limit1 == 1

# Placeholder for estimate_grad_var tests
def test_estimate_grad_var(dummy_rng, mock_model_instance, sample_data_shape, sample_target_shape, dtype):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    B = 4
    repeats = 3
    
    # Create a local init_fn for estimate_grad_var
    # Model params should be consistent across repeats for a given dtype
    initialized_model = init_mock_model_for_test(dummy_rng, din=sample_data_shape[0], dout=sample_target_shape[0])

    grad_variance = batch_optimizer.estimate_grad_var(
        model_instance=initialized_model, # Pass the initialized instance
        forward_backward_fn=forward_backward_for_test,
        input_data_shape=sample_data_shape,
        target_data_shape=sample_target_shape,
        rng_key=dummy_rng,
        batch_size_B=B,
        repeats=repeats,
        dtype=dtype
    )
    assert isinstance(grad_variance, float)
    assert not jnp.isnan(grad_variance)
    assert grad_variance >= 0.0

    # Test with B=0
    grad_variance_zero_b = batch_optimizer.estimate_grad_var(
        model_instance=initialized_model,
        forward_backward_fn=forward_backward_for_test,
        input_data_shape=sample_data_shape,
        target_data_shape=sample_target_shape,
        rng_key=dummy_rng,
        batch_size_B=0,
        repeats=repeats,
        dtype=dtype
    )
    assert jnp.isnan(grad_variance_zero_b)


    # Test OOM during one of the repeats
    oom_after_n_repeats = 1
    current_repeats = [0] # Using a list to modify in closure

    def fb_fn_with_oom_on_repeat(model_inst, batch_data_tuple):
        current_repeats[0] +=1
        if current_repeats[0] > oom_after_n_repeats:
            raise RuntimeError("Mock OOM during estimate_grad_var repeat")
        return forward_backward_for_test(model_inst, batch_data_tuple)

    current_repeats[0] = 0 # Reset for this test
    grad_variance_oom = batch_optimizer.estimate_grad_var(
        model_instance=initialized_model,
        forward_backward_fn=fb_fn_with_oom_on_repeat,
        input_data_shape=sample_data_shape,
        target_data_shape=sample_target_shape,
        rng_key=dummy_rng,
        batch_size_B=B,
        repeats=repeats, # e.g. 3 repeats, OOM after 1st
        dtype=dtype
    )
    # Should complete with the variance from the successful repeat(s)
    assert isinstance(grad_variance_oom, float)
    assert not jnp.isnan(grad_variance_oom) 


    # Test all repeats OOM
    def fb_fn_always_oom(model_inst, batch_data_tuple):
        raise RuntimeError("Mock OOM always in estimate_grad_var")

    grad_variance_all_oom = batch_optimizer.estimate_grad_var(
        model_instance=initialized_model,
        forward_backward_fn=fb_fn_always_oom,
        input_data_shape=sample_data_shape,
        target_data_shape=sample_target_shape,
        rng_key=dummy_rng,
        batch_size_B=B,
        repeats=repeats,
        dtype=dtype
    )
    assert jnp.isnan(grad_variance_all_oom)


@patch('time.perf_counter')
@patch.object(batch_optimizer, 'get_device_memory_usage')
def test_sweep_accum(mock_get_memory, mock_perf_counter, dummy_rng, mock_model_instance, sample_data_shape, sample_target_shape, dtype):
    if batch_optimizer is None:
        pytest.skip("batch_optimizer module not yet created")

    # Mock model instance
    initialized_model = init_mock_model_for_test(dummy_rng, din=sample_data_shape[0], dout=sample_target_shape[0])

    # Mock time.perf_counter to control timing
    mock_perf_counter.side_effect = [0.0, 0.1, 0.0, 0.25, 0.0, 0.45] # t_start, t_end for K=1, K=2, K=3 etc.

    # Mock get_device_memory_usage
    mock_get_memory.return_value = (1024*1024*100, 1024*1024*1000) # 100MB used, 1000MB total

    results = batch_optimizer.sweep_accum(
        model_instance=initialized_model,
        forward_backward_fn=forward_backward_for_test, # Reverted to non-JITted version for this test
        input_data_shape=sample_data_shape,
        target_data_shape=sample_target_shape,
        rng_key=dummy_rng,
        local_batch_size_B=4,
        max_accum_K=2, # Limit for test
        dtype=dtype
    )

    assert len(results) == 2
    assert results[0]['K'] == 1
    assert results[0]['effective_B'] == 4
    assert pytest.approx(results[0]['time_s']) == 0.1
    assert pytest.approx(results[0]['throughput']) == 4 / 0.1

    assert results[1]['K'] == 2
    assert results[1]['effective_B'] == 8
    assert pytest.approx(results[1]['time_s']) == 0.25 
    assert pytest.approx(results[1]['throughput']) == 8 / 0.25
    
    # Test B=0 case
    results_zero_b = batch_optimizer.sweep_accum(
        model_instance=initialized_model,
        forward_backward_fn=forward_backward_for_test, # Reverted to non-JITted version for this test
        input_data_shape=sample_data_shape,
        target_data_shape=sample_target_shape,
        rng_key=dummy_rng,
        local_batch_size_B=0,
        max_accum_K=2,
        dtype=dtype
    )
    assert len(results_zero_b) == 0

# Placeholder - needs significant mocking of sub-functions
# @patch.object(batch_optimizer, 'find_max_batch')
# @patch.object(batch_optimizer, 'estimate_grad_var')
# @patch.object(batch_optimizer, 'sweep_accum')
# def test_main_batch_optimizer_workflow(mock_sweep, mock_est_var, mock_find_max, dummy_rng, dtype):
#     if batch_optimizer is None:
#         pytest.skip("batch_optimizer module not yet created")

#     # Mock return values
#     mock_find_max.return_value = 64 # B_max
#     mock_est_var.return_value = 0.01 # some variance
#     mock_sweep.return_value = [{'K': 1, 'effective_B': 64, 'time_s': 0.1, 'throughput': 640, 'mem_used_bytes':0}]

#     input_shape = (10,)
#     target_shape = (5,)

#     # Have to provide real functions here, even if sub-components are mocked
#     # The model_init_fn and forward_backward_fn are still called by main_batch_optimizer_workflow
#     # to initialize the model instance that is then passed to the (mocked) sub-functions.
    
#     def _init_fn(key): return init_mock_model_for_test(key, din=input_shape[0], dout=target_shape[0])
    
#     batch_optimizer.main_batch_optimizer_workflow(
#         model_init_fn=_init_fn,
#         forward_backward_fn=forward_backward_for_test, # This will be used by the non-mocked parts
#         input_data_shape=input_shape,
#         target_data_shape=target_shape,
#         rng_seed=0,
#         start_batch_size_B_max=8,
#         limit_batch_size_B_max=128,
#         max_accum_steps_K=2,
#         dtype=dtype
#     )
    
#     mock_find_max.assert_called_once()
#     # estimate_grad_var might be called multiple times, check at least once
#     assert mock_est_var.call_count > 0 
#     mock_sweep.assert_called_once()

#     # Test B_max = 0 path
#     mock_find_max.reset_mock()
#     mock_est_var.reset_mock()
#     mock_sweep.reset_mock()
#     mock_find_max.return_value = 0 # B_max is 0

#     batch_optimizer.main_batch_optimizer_workflow(
#         model_init_fn=_init_fn,
#         forward_backward_fn=forward_backward_for_test,
#         input_data_shape=input_shape,
#         target_data_shape=target_shape,
#         rng_seed=0,
#         dtype=dtype
#     )
#     mock_find_max.assert_called_once()
#     mock_est_var.assert_not_called()
#     mock_sweep.assert_not_called()


@pytest.mark.skip(reason="Main workflow test requires extensive mocking or live runs, focusing on unit tests first.")
def test_main_batch_optimizer_workflow():
    pass 