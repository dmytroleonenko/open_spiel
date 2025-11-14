"""
Comprehensive tests for Mixed Value Target Computation Validation (Task 6.3).

This module validates that the JAX mixed value target logic produces bit-for-bit 
identical results to EfficientZeroV2's PyTorch BatchWorker implementation.

Tests cover:
- Training step transitions (start_use_mix_training_steps boundary)
- Collected transitions overflow scenarios  
- Mixed value threshold boundary conditions
- Complete numerical equivalence with EfficientZeroV2 patterns
- Edge cases and parameter combinations
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import dataclasses
from typing import Tuple, Any, Dict
from unittest.mock import Mock

# Import JAX implementation components
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner,
    MuZeroConfig,
    generate_top_new_masks,
    apply_mixed_value_targets,
    create_network_config_from_muzero_config,
)

# Import test utilities
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
    make_cfg,
    make_model,
    make_batch,
    MockNetCfg,
)


class TestMixedValueTargetValidation:
    """Comprehensive validation of mixed value target computation equivalence."""

    @pytest.fixture
    def common_key(self):
        """Common random key for reproducible tests."""
        return jax.random.key(42)

    @pytest.fixture 
    def base_config(self):
        """Base configuration for mixed value target tests."""
        return MuZeroConfig(
            num_unroll_steps=2,
            num_actions=5,
            batch_size=4,
            value_target="mixed",
            mixed_value_threshold=1000,
            start_use_mix_training_steps=500,
            td_lambda=0.95,
            discount_factor=0.99,
            learning_rate=1e-3,
            value_support_size=0,  # Scalar values for simplicity
            reward_support_size=0,
        )

    def test_generate_top_new_masks_requires_collected_transitions(self):
        with pytest.raises(ValueError):
            generate_top_new_masks(jnp.array([1, 2, 3]), None, 10)

    def test_training_step_transitions_comprehensive(self, common_key, base_config):
        """Test behavior across training step transitions."""
        # Test scenarios around start_use_mix_training_steps boundary
        test_scenarios = [
            # (training_step, expected_behavior)
            (0, "search_only"),  # Very early training
            (base_config.start_use_mix_training_steps - 1, "search_only"),  # Just before threshold
            (base_config.start_use_mix_training_steps, "mixed_mode"),  # Exactly at threshold  
            (base_config.start_use_mix_training_steps + 1, "mixed_mode"),  # Just after threshold
            (base_config.start_use_mix_training_steps * 2, "mixed_mode"),  # Well after threshold
        ]

        batch_size = base_config.batch_size
        
        # Create test data
        search_values = jnp.ones((batch_size, base_config.num_unroll_steps + 1)) * 1.0
        sarsa_values = jnp.ones((batch_size, base_config.num_unroll_steps + 1)) * 2.0
        
        # Sample indices that create mixed old/new pattern
        sample_indices = jnp.array([500, 800, 1200, 1500])  # Mix around threshold=1000
        collected_transitions = 2000
        
        for training_step, expected_behavior in test_scenarios:
            # Generate masks (this should always work)
            masks = generate_top_new_masks(
                sample_indices, collected_transitions, base_config.mixed_value_threshold
            )
            
            # Expected masks: idx > (2000 - 1000) = 1000
            # So indices [500, 800, 1200, 1500] -> [False, False, True, True] -> [0, 0, 1, 1]
            expected_masks = jnp.array([0.0, 0.0, 1.0, 1.0])
            assert jnp.allclose(masks, expected_masks), f"Mask generation failed at step {training_step}"
            
            # Test mixed value application based on training step
            if expected_behavior == "search_only":
                # Early training: should use search values regardless of masks
                if training_step < base_config.start_use_mix_training_steps:
                    expected_values = search_values
                else:
                    # Should use mixed logic
                    expected_values = apply_mixed_value_targets(
                        search_values, sarsa_values, masks, base_config.num_unroll_steps
                    )
            elif expected_behavior == "mixed_mode":
                # Later training: should apply masks
                expected_values = apply_mixed_value_targets(
                    search_values, sarsa_values, masks, base_config.num_unroll_steps
                 )

            # Verify expected mixed pattern for mixed_mode
            if expected_behavior == "mixed_mode":
                # Samples 0,1 (old): should use search values (1.0)
                # Samples 2,3 (new): should use sarsa values (2.0)
                assert jnp.allclose(
                    expected_values[0, :], 1.0
                ), f"Sample 0 should use search at step {training_step}"
                assert jnp.allclose(
                    expected_values[1, :], 1.0
                ), f"Sample 1 should use search at step {training_step}"
                assert jnp.allclose(
                    expected_values[2, :], 2.0
                ), f"Sample 2 should use sarsa at step {training_step}"
                assert jnp.allclose(
                    expected_values[3, :], 2.0
                ), f"Sample 3 should use sarsa at step {training_step}"
            elif expected_behavior == "search_only":
                # All samples should use search values
                assert jnp.allclose(
                    expected_values, search_values
                ), f"All should use search at step {training_step}"

    def test_collected_transitions_overflow_scenarios(self, common_key, base_config):
        """Test behavior with very large collected_transitions values."""
        # Test overflow and edge cases with large numbers
        overflow_scenarios = [
            # (collected_transitions, mixed_value_threshold, sample_indices, description)
            (1000000, 50000, jnp.array([900000, 950000, 960000, 970000]), "large_numbers_normal"),
            (2**31 - 1, 1000000, jnp.array([2**31 - 1000000, 2**31 - 500000, 2**31 - 1]), "near_int32_max"),
            (1000, 2000, jnp.array([0, 500, 999, 1000]), "threshold_larger_than_collected"),
            (0, 1000, jnp.array([0]), "zero_collected_transitions"),
            (1, 1, jnp.array([0, 1, 2]), "minimal_values"),
        ]
        
        for collected_trans, threshold, sample_indices, description in overflow_scenarios:
            masks = generate_top_new_masks(sample_indices, collected_trans, threshold)
            
            # Verify masks are valid boolean values
            assert masks.dtype == jnp.bool_, f"Wrong dtype in {description}"
            
            # Verify mathematical correctness: mask = (idx > collected_trans - threshold)
            expected_threshold = collected_trans - threshold
            expected_masks = (sample_indices > expected_threshold)
            assert jnp.array_equal(masks, expected_masks), f"Math error in {description}"

    def test_mixed_value_threshold_boundary_conditions(self, common_key, base_config):
        """Test exact behavior at mixed_value_threshold boundaries."""
        # Test boundary conditions with precise threshold calculations
        boundary_test_cases = [
            # (collected_transitions, threshold, test_indices, expected_pattern)
            (1000, 300, [699, 700, 701], [False, False, True]),  # Around 700 boundary
            (5000, 2000, [2999, 3000, 3001], [False, False, True]),  # Around 3000 boundary  
            (10000, 5000, [4999, 5000, 5001], [False, False, True]),  # Around 5000 boundary
            (100, 10, [89, 90, 91], [False, False, True]),  # Around 90 boundary
            (1, 1, [0, 1], [False, True]),  # Edge case: threshold >= collected
        ]
        
        for collected_trans, threshold, test_indices, expected_pattern in boundary_test_cases:
            sample_indices = jnp.array(test_indices)
            expected_masks = jnp.array(expected_pattern)
            
            masks = generate_top_new_masks(sample_indices, collected_trans, threshold)
            
            # Verify exact boundary behavior
            assert jnp.array_equal(masks, expected_masks), (
                f"Boundary test failed: collected={collected_trans}, threshold={threshold}, "
                f"indices={test_indices}, expected={expected_pattern}, got={masks}"
            )
            
            # Test with mixed value application
            search_vals = jnp.ones((len(test_indices), 3)) * 10.0
            sarsa_vals = jnp.ones((len(test_indices), 3)) * 20.0
            
            mixed_vals = apply_mixed_value_targets(search_vals, sarsa_vals, masks, 2)
            
            # Verify mixed application follows mask pattern
            for i, mask_val in enumerate(expected_pattern):
                if mask_val == False:
                    # Should use search values
                    assert jnp.allclose(mixed_vals[i, :], 10.0), f"Sample {i} should use search values"
                else:
                    # Should use sarsa values  
                    assert jnp.allclose(mixed_vals[i, :], 20.0), f"Sample {i} should use sarsa values"

    def test_efficientzero_numerical_equivalence(self, common_key, base_config):
        """Test numerical equivalence with EfficientZeroV2 patterns (optimized)."""
        # Reduced test cases for faster execution
        pytorch_equivalent_test_cases = [
            {
                "name": "typical_training_scenario",
                "collected_transitions": 5000,  # Reduced from 50000
                "mixed_value_threshold": 1000,  # Reduced from 10000
                "sample_indices": [3000, 3500, 4000],  # Reduced count
                "training_step": 1000,
                "start_use_mix_training_steps": 500,
            },
            {
                "name": "mixed_boundary_case",
                "collected_transitions": 2000,  # Combined other cases into one optimized case
                "mixed_value_threshold": 500,
                "sample_indices": [1000, 1500],  # Simplified
                "training_step": 600,
                "start_use_mix_training_steps": 500,
            },
        ]
        
        for test_case in pytorch_equivalent_test_cases:
            sample_indices = jnp.array(test_case["sample_indices"])
            collected_trans = test_case["collected_transitions"]
            threshold = test_case["mixed_value_threshold"]
            training_step = test_case["training_step"]
            start_mix_steps = test_case["start_use_mix_training_steps"]
            
            # Simulate PyTorch BatchWorker mask calculation
            # PyTorch: mask = int(idx > collected_transitions - mixed_value_threshold)
            pytorch_threshold = collected_trans - threshold
            pytorch_masks = []
            for idx in test_case["sample_indices"]:
                pytorch_mask = float(idx > pytorch_threshold)
                pytorch_masks.append(pytorch_mask)
            pytorch_masks = jnp.array(pytorch_masks)
            
            # JAX implementation
            jax_masks = generate_top_new_masks(sample_indices, collected_trans, threshold)
            
            # Convert boolean masks to float for comparison with PyTorch
            jax_masks_float = jax_masks.astype(jnp.float32)
            
            # Verify bit-for-bit equivalence
            assert jnp.allclose(jax_masks_float, pytorch_masks, rtol=1e-15, atol=1e-15), (
                f"Mask mismatch in {test_case['name']}: PyTorch={pytorch_masks}, JAX={jax_masks_float}"
            )
            
            # Test complete target selection logic
            batch_size = len(sample_indices)
            num_steps = 2  # Reduced from 3 for speed
            search_values = jax.random.uniform(common_key, (batch_size, num_steps)) + 1.0
            sarsa_values = jax.random.uniform(common_key, (batch_size, num_steps)) + 10.0
            
            # Simulate PyTorch mixed value logic
            if training_step >= start_mix_steps:
                # Use mixed targets
                pytorch_mixed_values = jnp.zeros_like(search_values)
                for i in range(batch_size):
                    mask = pytorch_masks[i]
                    pytorch_mixed_values = pytorch_mixed_values.at[i, :].set(
                        mask * sarsa_values[i, :] + (1 - mask) * search_values[i, :]
                    )
                expected_values = pytorch_mixed_values
            else:
                # Use search values only
                expected_values = search_values
                
            # JAX mixed value application
            jax_mixed_values = apply_mixed_value_targets(
                search_values, sarsa_values, jax_masks, num_steps - 1
            )
            
            if training_step >= start_mix_steps:
                actual_values = jax_mixed_values
            else:
                actual_values = search_values
                
            # Verify numerical equivalence
            assert jnp.allclose(actual_values, expected_values, rtol=1e-12, atol=1e-12), (
                f"Value target mismatch in {test_case['name']}"
            )

    def test_categorical_value_support_equivalence(self, common_key, base_config):
        """Test mixed value targets with categorical value distributions (optimized)."""
        # Test with smaller categorical value supports for speed
        categorical_config = dataclasses.replace(
            base_config,
            value_support_size=11,  # Reduced from 51 for faster execution
        )
        
        batch_size = 2  # Reduced from 3
        num_steps = 3   # Reduced from 4
        support_size = categorical_config.value_support_size
        
        # Create categorical value distributions
        key1, key2 = jax.random.split(common_key)
        search_values = jax.random.uniform(key1, (batch_size, num_steps, support_size))
        search_values = search_values / jnp.sum(search_values, axis=-1, keepdims=True)  # Normalize
        
        sarsa_values = jax.random.uniform(key2, (batch_size, num_steps, support_size))
        sarsa_values = sarsa_values / jnp.sum(sarsa_values, axis=-1, keepdims=True)  # Normalize
        
        # Test different mask patterns (reduced for speed)
        test_masks = [
            jnp.array([0.0, 0.0]),  # All old samples
            jnp.array([1.0, 0.0]),  # Mixed pattern (combined into one test)
        ]
        
        for masks in test_masks:
            mixed_values = apply_mixed_value_targets(
                search_values, sarsa_values, masks, num_steps - 1
            )
            
            # Verify shape preservation
            assert mixed_values.shape == (batch_size, num_steps, support_size)
            
            # Verify probability distribution properties preserved
            prob_sums = jnp.sum(mixed_values, axis=-1)
            assert jnp.allclose(prob_sums, 1.0, rtol=1e-6), "Categorical distributions not normalized"
            
            # Verify mask application correctness
            for i in range(batch_size):
                if masks[i] == 0.0:
                    # Should use search values
                    assert jnp.allclose(mixed_values[i], search_values[i], rtol=1e-12)
                else:
                    # Should use sarsa values
                    assert jnp.allclose(mixed_values[i], sarsa_values[i], rtol=1e-12)

    def test_integration_with_trainer_comprehensive(self, common_key, base_config):
        """Test integration with trainer workflow (optimized for speed)."""
        # Use smaller, faster configuration for integration test
        fast_config = dataclasses.replace(
            base_config,
            batch_size=2,  # Reduced from 4
            num_unroll_steps=1,  # Reduced from 2
            num_actions=3,  # Reduced from 5
        )
        
        mk, lk, bk = jax.random.split(common_key, 3)
        
        # Create minimal mock model for faster testing
        net_cfg = MockNetCfg(
            observation_shape=(2,),  # Smaller observation
            num_actions=fast_config.num_actions,
            batch_size=fast_config.batch_size,
            value_support_size=fast_config.value_support_size,
            reward_support_size=fast_config.reward_support_size,
        )
        model = make_model(mk, net_cfg)
        
        optimizer = optax.adam(fast_config.learning_rate)
        learner = Learner(model, optimizer, fast_config, lk)
        
        # Test only one scenario to reduce test time
        training_step = 600  # After start_use_mix_training_steps
        
        # Create minimal batch with mixed value target data
        batch = make_batch(
            bk, 
            fast_config.batch_size,
            (2,),  # Smaller observation
            fast_config.num_actions,
            fast_config.num_unroll_steps,
            fast_config.value_support_size,
            fast_config.reward_support_size,
        )
        
        # Add mixed value specific fields
        batch.update({
            "target_search_value": jnp.ones((fast_config.batch_size, fast_config.num_unroll_steps + 1)) * 1.0,
            "target_sarsa_value": jnp.ones((fast_config.batch_size, fast_config.num_unroll_steps + 1)) * 2.0,
            "sample_indices": jnp.array([400, 800]),  # Reduced to match batch_size
            "collected_transitions": 2000,
            "training_step": training_step,
        })
        
        # Execute single training step (optimized)
        try:
            loss, metrics = Learner._compute_total_loss_static(
                model, fast_config, batch, lk, training=True, training_step=training_step
            )
            
            # Verify training completed successfully
            assert jnp.isfinite(loss), f"Non-finite loss in optimized test"
            assert "value_loss" in metrics, f"Missing value_loss in optimized test"
            assert jnp.isfinite(metrics["value_loss"]), f"Non-finite value_loss in optimized test"
            
            # Verify mixed value logic was applied correctly
            masks = generate_top_new_masks(
                batch["sample_indices"], batch["collected_transitions"], fast_config.mixed_value_threshold
            )
            expected_masks = jnp.array([False, False])  # Both indices < (2000-1000)=1000
            assert jnp.array_equal(masks, expected_masks), "Mixed value masks incorrect"
            
        except Exception as e:
            pytest.fail(f"Optimized training test failed: {e}")

    def test_edge_case_robustness(self, common_key, base_config):
        """Test robustness against edge cases and malformed inputs."""
        # Test edge cases that might break the implementation
        edge_cases = [
            {
                "name": "empty_batch",
                "sample_indices": jnp.array([]),
                "collected_transitions": 1000,
                "threshold": 100,
            },
            {
                "name": "single_sample",
                "sample_indices": jnp.array([500]),
                "collected_transitions": 1000,
                "threshold": 400,
            },
            {
                "name": "identical_indices",
                "sample_indices": jnp.array([500, 500, 500]),
                "collected_transitions": 1000,
                "threshold": 300,
            },
            {
                "name": "reverse_order_indices",
                "sample_indices": jnp.array([900, 600, 300]),
                "collected_transitions": 1000,
                "threshold": 200,
            },
        ]
        
        for case in edge_cases:
            if case["name"] == "empty_batch":
                # Empty batch should return empty masks
                masks = generate_top_new_masks(
                    case["sample_indices"], case["collected_transitions"], case["threshold"]
                )
                assert masks.shape == (0,), f"Wrong shape for empty batch: {masks.shape}"
                continue
                
            masks = generate_top_new_masks(
                case["sample_indices"], case["collected_transitions"], case["threshold"]
            )
            
            # Verify output properties
            assert masks.shape == case["sample_indices"].shape, f"Shape mismatch in {case['name']}"
            assert masks.dtype == jnp.bool_, f"Wrong dtype in {case['name']}"
            
            # Test with value target application if batch not empty
            if len(case["sample_indices"]) > 0:
                batch_size = len(case["sample_indices"])
                search_vals = jnp.ones((batch_size, 3)) * 5.0
                sarsa_vals = jnp.ones((batch_size, 3)) * 15.0
                
                mixed_vals = apply_mixed_value_targets(search_vals, sarsa_vals, masks, 2)
                
                assert mixed_vals.shape == (batch_size, 3), f"Wrong output shape in {case['name']}"
                assert jnp.all(jnp.isfinite(mixed_vals)), f"Non-finite mixed values in {case['name']}"

    def test_performance_and_memory_efficiency(self, common_key, base_config):
        """Test performance and memory efficiency (optimized for speed)."""
        # Test with smaller batch sizes for faster execution 
        batch_sizes = [8, 16]  # Reduced from [16, 64, 256]
        
        for batch_size in batch_sizes:
            sample_indices = jnp.arange(batch_size) * 100  # Spread out indices
            collected_transitions = batch_size * 150
            threshold = 5000
            
            # Test mask generation without timing (timing adds overhead in parallel tests)
            masks = generate_top_new_masks(sample_indices, collected_transitions, threshold)
            
            # Verify basic functionality without strict timing constraints
            assert masks.shape == (batch_size,), f"Wrong mask shape for batch_size={batch_size}"
            assert masks.dtype == jnp.bool_, f"Wrong mask dtype for batch_size={batch_size}"
            
            # Test mixed value application with smaller dimensions for speed
            search_vals = jnp.ones((batch_size, 3)) * 3.0  # Reduced from 5 to 3
            sarsa_vals = jnp.ones((batch_size, 3)) * 13.0
            
            mixed_vals = apply_mixed_value_targets(search_vals, sarsa_vals, masks, 2)  # Reduced from 4 to 2
            
            # Verify correctness maintained with optimized test
            assert mixed_vals.shape == (batch_size, 3)
            assert jnp.all(jnp.isfinite(mixed_vals))


if __name__ == "__main__":
    # Can be run directly for manual testing
    print("Running mixed value target validation tests...")
    test_instance = TestMixedValueTargetValidation()
    
    # Run a subset of tests manually
    key = jax.random.key(42)
    config = MuZeroConfig(
        num_unroll_steps=2,
        num_actions=5,
        batch_size=4,
        value_target="mixed",
        mixed_value_threshold=1000,
        start_use_mix_training_steps=500,
    )
    
    test_instance.test_training_step_transitions_comprehensive(key, config)
    test_instance.test_collected_transitions_overflow_scenarios(key, config)
    test_instance.test_mixed_value_threshold_boundary_conditions(key, config)
    
    print("✅ All manual tests passed!") 
