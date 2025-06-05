import pytest
import jax
import jax.numpy as jnp
import runpy
from open_spiel.python.algorithms.muzero_jax.utils import batch_optimizer
from open_spiel.python.algorithms.muzero_jax.utils.batch_optimizer import (
    OptimizationConfig, BatchOptimizer, ModelType, create_muzero_model_and_params, 
    compute_muzero_loss_and_gradients, create_random_batch
)

# Tests for the class-based implementation

def test_create_muzero_model_and_params_implemented():
    """Test that create_muzero_model_and_params works correctly."""
    config = OptimizationConfig()
    rng_key = jax.random.PRNGKey(0)
    
    # This should work without raising NotImplementedError
    model = create_muzero_model_and_params(config, rng_key)
    
    # Verify it returns a proper model instance
    assert hasattr(model, 'representation')
    assert hasattr(model, 'dynamics')
    assert hasattr(model, 'prediction')
    assert hasattr(model, 'config')


def test_compute_muzero_loss_and_gradients_implemented():
    """Test that compute_muzero_loss_and_gradients works correctly."""
    config = OptimizationConfig()
    rng_key = jax.random.PRNGKey(0)
    
    # Create model and batch
    model = create_muzero_model_and_params(config, rng_key)
    rng_batch, rng_loss = jax.random.split(rng_key)
    batch_data = create_random_batch(rng_batch, 2, config)
    
    # This should work without raising NotImplementedError
    loss, grads = compute_muzero_loss_and_gradients(model, batch_data, rng_loss)
    
    # Verify the outputs
    assert isinstance(loss, jax.Array)
    assert not jnp.isnan(loss)


def test_get_device_memory_usage_fallback():
    """Test get_device_memory_usage fallback for unknown platforms."""
    # Mock devices to simulate unknown platform
    class DummyDeviceUnknown:
        platform = 'unknown'
        def memory_stats(self):
            raise NotImplementedError
    
    # Patch jax.devices() to return unknown device for fallback testing
    import unittest.mock
    with unittest.mock.patch.object(batch_optimizer.jax, 'devices', lambda: [DummyDeviceUnknown()]):
        used, total = batch_optimizer.get_device_memory_usage()
        # Function returns defaults when no JAX memory stats available
        assert used == 1024**3  # 1GB default
        assert total == 8 * 1024**3  # 8GB default


def test_batch_optimizer_class_workflow():
    """Test the new BatchOptimizer class workflow."""
    config = OptimizationConfig(
        model_type=ModelType.MUZERO,
        num_actions=9,
        observation_shape=(27,),
        start_batch_size=2,
        max_batch_size=16,  # Small for testing
        grad_var_repeats=2,  # Few repeats for speed
        max_accum_steps=2,
        enable_jax_smi=False  # Disable for testing
    )
    
    optimizer = BatchOptimizer(config)
    
    # Test individual methods
    rng_key = jax.random.PRNGKey(42)
    max_batch = optimizer.find_max_batch_size(rng_key)
    assert max_batch > 0
    
    # Test that it doesn't crash
    results = optimizer.run_optimization_workflow(rng_seed=42)
    assert results.max_batch_size > 0
    assert results.config == config


def test_optimization_config_advanced_validation():
    """Test advanced validation cases for OptimizationConfig."""
    # Test invalid observation shape
    with pytest.raises(ValueError, match="observation_shape must have positive dimensions"):
        OptimizationConfig(observation_shape=(0, 5))
    
    # Test invalid dtype
    with pytest.raises(ValueError, match="dtype must be float32, bfloat16, or float16"):
        OptimizationConfig(dtype=jnp.int32)
    
    # Test invalid memory warning threshold
    with pytest.raises(ValueError, match="memory_warning_threshold must be between 0 and 1"):
        OptimizationConfig(memory_warning_threshold=1.5)


def test_batch_optimizer_unsupported_model_type():
    """Test BatchOptimizer with unsupported model type."""
    config = OptimizationConfig(model_type=ModelType.GENERIC)
    
    # Should raise ConfigurationError for unsupported model type
    with pytest.raises(batch_optimizer.ConfigurationError, match="Unsupported model type"):
        BatchOptimizer(config)


def test_memory_monitor_functionality():
    """Test MemoryMonitor basic functionality."""
    config = OptimizationConfig(enable_jax_smi=False)  # Disable for testing
    monitor = batch_optimizer.MemoryMonitor(config)
    
    # Test basic properties
    assert not monitor.monitoring
    assert not monitor.jax_smi_available  # Usually false in test environment
    
    # Test context manager
    with monitor.monitoring_context():
        # Should not start monitoring when jax-smi is disabled/unavailable
        assert not monitor.monitoring


def test_accumulation_result_stability():
    """Test AccumulationResult stability property."""
    from open_spiel.python.algorithms.muzero_jax.utils.batch_optimizer import AccumulationResult
    
    # Test stable result (CV < 20%)
    stable_result = AccumulationResult(
        accumulation_steps=2,
        local_batch_size=8,
        effective_batch_size=16,
        mean_time=1.0,
        std_time=0.1,  # 10% CV
        throughput=16.0,
        memory_used_gb=2.0,
        coefficient_of_variation=10.0
    )
    assert stable_result.is_stable
    
    # Test unstable result (CV >= 20%)
    unstable_result = AccumulationResult(
        accumulation_steps=2,
        local_batch_size=8,
        effective_batch_size=16,
        mean_time=1.0,
        std_time=0.3,  # 30% CV
        throughput=16.0,
        memory_used_gb=2.0,
        coefficient_of_variation=30.0
    )
    assert not unstable_result.is_stable


def test_main_demo_runs():
    """Test that the main demo function works with the new implementation."""
    try:
        # Mock the expensive optimization workflow to speed up test
        import unittest.mock
        
        def mock_run_workflow(self, rng_seed=42):
            from open_spiel.python.algorithms.muzero_jax.utils.batch_optimizer import OptimizationResults, AccumulationResult
            recommended = AccumulationResult(
                accumulation_steps=2,
                local_batch_size=8,
                effective_batch_size=16,
                mean_time=1.0,
                std_time=0.1,
                throughput=16.0,
                memory_used_gb=2.0,
                coefficient_of_variation=10.0
            )
            return OptimizationResults(
                config=self.config,
                max_batch_size=32,
                sweet_spot_batch_size=16,
                gradient_variance=0.001,
                accumulation_results=[recommended],
                recommended_config=recommended,
                timing_stats={'mean_time': 0.1}
            )
        
        with unittest.mock.patch.object(BatchOptimizer, 'run_optimization_workflow', mock_run_workflow):
            # This should execute without errors
            runpy.run_module('open_spiel.python.algorithms.muzero_jax.utils.batch_optimizer', run_name='__main__')
        # If we get here, the demo ran successfully
        assert True
    except SystemExit:
        # Normal exit from main function is okay
        assert True
    except Exception as e:
        # Only fail if there's an unexpected error
        pytest.fail(f"Main demo failed with error: {e}") 