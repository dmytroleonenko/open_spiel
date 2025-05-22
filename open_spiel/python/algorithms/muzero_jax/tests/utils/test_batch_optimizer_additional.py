import pytest
import jax
import runpy
from open_spiel.python.algorithms.muzero_jax.utils import batch_optimizer

# 1. Tests for NotImplementedError in user-defined hooks

def test_init_muzero_model_and_params_not_implemented():
    with pytest.raises(NotImplementedError):
        batch_optimizer.init_muzero_model_and_params(jax.random.PRNGKey(0))


def test_forward_and_backward_muzero_not_implemented():
    with pytest.raises(NotImplementedError):
        batch_optimizer.forward_and_backward_muzero(None, None)

# 2. Test get_device_memory_usage fallback for unknown platform

class DummyDeviceUnknown:
    platform = 'unknown'
    def memory_stats(self):
        raise NotImplementedError

@pytest.fixture(autouse=True)
def patch_jax_devices(monkeypatch):
    # Patch jax.devices() to return unknown device for CPU branch fallback
    monkeypatch.setattr(batch_optimizer.jax, 'devices', lambda: [DummyDeviceUnknown()])
    yield


def test_get_device_memory_usage_unknown_device():
    used, total = batch_optimizer.get_device_memory_usage()
    assert used == 0
    assert total == 0

# 3. Test example usage block executes without errors and prints expected output

def test_example_usage_runs(capsys):
    # Run module as script to exercise example usage block
    runpy.run_module('open_spiel.python.algorithms.muzero_jax.utils.batch_optimizer', run_name='__main__')
    captured = capsys.readouterr().out
    assert 'Running batch optimizer workflow example with float32' in captured
    assert 'Running batch optimizer workflow example with bfloat16' in captured
    assert 'Max local batch size (B_max)' in captured

# 4. Test full workflow prints and logic using patched functions

def test_main_batch_optimizer_workflow_full(monkeypatch, capsys):
    import jax.numpy as jnp

    # Fake init and forward/backward functions
    def fake_model_init_fn(rng_key):
        print("fake init called")
        return "model"

    def fake_forward_backward_fn(model_instance, batch):
        # Loss is printed, grads ignored
        return 0.123, None

    # Patch find_max_batch, estimate_grad_var, sweep_accum, get_device_memory_usage
    monkeypatch.setattr(batch_optimizer, 'find_max_batch', lambda *args, **kwargs: 4)
    monkeypatch.setattr(batch_optimizer, 'estimate_grad_var', lambda *args, **kwargs: 0.456)
    sample_stats = [
        {'K': 1, 'effective_B': 4, 'time_s': 0.5, 'throughput': 8.0, 'mem_used_bytes': 200},
        {'K': 2, 'effective_B': 8, 'time_s': 1.0, 'throughput': 8.0, 'mem_used_bytes': 300},
    ]
    monkeypatch.setattr(batch_optimizer, 'sweep_accum', lambda *args, **kwargs: sample_stats)
    monkeypatch.setattr(batch_optimizer, 'get_device_memory_usage', lambda: (200, 1000))

    # Run the main workflow
    batch_optimizer.main_batch_optimizer_workflow(
        fake_model_init_fn,
        fake_forward_backward_fn,
        input_data_shape=(2,),
        target_data_shape=(1,),
        rng_seed=0,
        start_batch_size_B_max=2,
        limit_batch_size_B_max=8,
        grad_var_test_batch_size_B=2,
        grad_var_repeats=3,
        max_accum_steps_K=2,
        dtype=jnp.float32
    )
    out = capsys.readouterr().out

    # Verify key prints and summary
    assert "fake init called" in out
    assert "Model initialized." in out
    assert "1) Finding max batch size" in out
    assert "→ Max local batch size (B_max) = 4" in out
    assert "2) Estimating gradient variance" in out
    assert "var(B=2) = 4.560e-01" in out
    assert "var(B=4) = 4.560e-01" in out
    assert "→ Sweet-spot batch size (sweet_spot_B) = 4" in out
    assert "3) Timing & memory at sweet-spot batch size" in out
    assert "4) Sweeping gradient accumulation factors" in out
    assert "=== SUMMARY ===" in out
    assert "Gradient Accumulation Sweep Results" in out
    assert "Recommended based on max throughput" in out 