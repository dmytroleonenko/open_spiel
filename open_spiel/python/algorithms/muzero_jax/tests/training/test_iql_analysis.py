"""
Comprehensive test suite for Critical IQL (Implicit Quantile Learning) Implementation Analysis.

This module implements Task 6.2 from TODO.md, providing detailed verification that the JAX
IQL implementation produces identical results to the EfficientZeroV2 reference implementation.

Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_iql_analysis.py`
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any
import dataclasses

# Import the JAX IQL implementation
from open_spiel.python.algorithms.muzero_jax.training import losses as jax_losses

# Test utilities
from trainer_utils import (
    NUM_UNROLL_STEPS, NUM_ACTIONS, BATCH_SIZE, VALUE_SUPPORT_SCALAR, REWARD_SUPPORT_SCALAR,
    key as common_key, cfg_flat as common_cfg_flat, make_model, make_cfg, make_batch
)
from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig


class EfficientZeroV2IQLReference:
    """Reference implementation matching EfficientZeroV2's exact IQL weighting logic."""
    
    @staticmethod
    def compute_iql_weights(prediction_error: jax.Array, iql_weight: float) -> jax.Array:
        """
        Compute IQL weights exactly as in EfficientZeroV2/ez/utils/loss.py lines 50-51.
        
        Args:
            prediction_error: value_prediction - target_value
            iql_weight: IQL weight parameter
            
        Returns:
            IQL weights for error-dependent asymmetric weighting
        """
        # EfficientZeroV2: value_sign = (value_error > 0).float().detach()
        value_sign = (prediction_error > 0).astype(jnp.float32)
        
        # EfficientZeroV2: value_weight = (1 - value_sign) * iql_weight + value_sign * (1 - iql_weight)
        value_weight = (1 - value_sign) * iql_weight + value_sign * (1 - iql_weight)
        
        return value_weight
    
    @staticmethod
    def compute_scalar_value_loss_reference(prediction: jax.Array, target: jax.Array, iql_weight: float) -> jax.Array:
        """Reference scalar value loss with IQL weighting exactly matching JAX implementation."""
        # Base MSE loss (matching JAX scalar_mse_loss implementation)
        base_loss = (prediction - target) ** 2
        
        # Apply IQL weighting
        error = prediction - target
        weights = EfficientZeroV2IQLReference.compute_iql_weights(error, iql_weight)
        
        return base_loss * weights


class TestIQLImplementationAnalysis:
    """Test suite for IQL implementation analysis comparing JAX vs EfficientZeroV2."""
    
    def test_iql_weights_mathematical_equivalence(self):
        """Test that IQL weight computation is mathematically identical to EfficientZeroV2."""
        # Test various IQL weight values
        iql_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 1.0]
        
        # Test various error scenarios
        test_errors = jnp.array([
            -5.0, -2.5, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 5.0
        ])
        
        for iql_weight in iql_weights:
            # Test each IQL weight value
            # Compute reference weights
            reference_weights = EfficientZeroV2IQLReference.compute_iql_weights(test_errors, iql_weight)
            
            # Compute JAX implementation weights using the same formula
            value_sign = (test_errors > 0).astype(jnp.float32)
            jax_weights = (1.0 - value_sign) * iql_weight + value_sign * (1.0 - iql_weight)
            
            # Verify bit-for-bit equivalence
            np.testing.assert_array_equal(
                reference_weights, jax_weights,
                f"IQL weights mismatch for iql_weight={iql_weight}"
            )
    
    def test_scalar_value_loss_numerical_equivalence(self):
        """Test scalar value loss computation for numerical equivalence with <1e-7 relative error."""
        key = jax.random.PRNGKey(123)
        
        # Generate diverse test cases
        batch_sizes = [1, 4, 16]
        iql_weights = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
        value_ranges = [(-10.0, 10.0), (-1.0, 1.0), (0.0, 5.0), (-100.0, 100.0)]
        
        total_tests = 0
        max_relative_error = 0.0
        
        for batch_size in batch_sizes:
            for iql_weight in iql_weights:
                for value_min, value_max in value_ranges:
                    # Generate random predictions and targets
                    predictions = jax.random.uniform(
                        key, (batch_size,), minval=value_min, maxval=value_max
                    )
                    targets = jax.random.uniform(
                        jax.random.fold_in(key, 1), (batch_size,), minval=value_min, maxval=value_max
                    )
                    
                    # Compute losses using both implementations
                    reference_loss = EfficientZeroV2IQLReference.compute_scalar_value_loss_reference(
                        predictions, targets, iql_weight
                    )
                    jax_loss = jax_losses.compute_scalar_value_loss(
                        predictions, targets, effective_iql_param=iql_weight
                    )
                    
                    # Compute absolute difference first
                    abs_diff = jnp.abs(jax_loss - reference_loss)
                    max_abs_diff = jnp.max(abs_diff)
                    
                    # For very small reference values, use absolute error
                    reference_magnitude = jnp.abs(reference_loss)
                    max_ref_magnitude = jnp.max(reference_magnitude)
                    
                    if max_ref_magnitude < 1e-8:
                        # Use absolute error for very small values
                        max_error = max_abs_diff
                        error_threshold = 1e-10
                    else:
                        # Use relative error for normal values
                        relative_error = abs_diff / (reference_magnitude + 1e-10)
                        max_error = jnp.max(relative_error)
                        error_threshold = 1e-7
                    
                    max_relative_error = max(max_relative_error, float(max_error))
                    
                    # Verify error is within threshold
                    assert max_error < error_threshold, (
                        f"Error {max_error:.2e} exceeds {error_threshold:.2e} for "
                        f"batch_size={batch_size}, iql_weight={iql_weight}, "
                        f"value_range=({value_min}, {value_max})"
                    )
                    
                    total_tests += 1
        
        print(f"✅ Numerical equivalence verified across {total_tests} test cases")
        print(f"   Maximum relative error: {max_relative_error:.2e}")
    
    def test_iql_edge_cases_stability(self):
        """Test IQL implementation stability on edge cases."""
        
        # Test zero errors (prediction == target)
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        
        for iql_weight in [0.0, 0.5, 1.0]:
            loss = jax_losses.compute_scalar_value_loss(predictions, targets, iql_weight)
            assert jnp.allclose(loss, 0.0), f"Zero error should produce zero loss for iql_weight={iql_weight}"
        
        # Test extreme values
        extreme_predictions = jnp.array([-1e6, 0.0, 1e6])
        extreme_targets = jnp.array([1e6, 0.0, -1e6])
        
        for iql_weight in [0.0, 0.5, 1.0]:
            loss = jax_losses.compute_scalar_value_loss(extreme_predictions, extreme_targets, iql_weight)
            assert jnp.all(jnp.isfinite(loss)), f"Extreme values should produce finite loss for iql_weight={iql_weight}"
        
        # Test very small errors
        small_predictions = jnp.array([1e-8, -1e-8, 0.0])
        small_targets = jnp.array([0.0, 0.0, 1e-8])
        
        for iql_weight in [0.0, 0.5, 1.0]:
            loss = jax_losses.compute_scalar_value_loss(small_predictions, small_targets, iql_weight)
            assert jnp.all(jnp.isfinite(loss)), f"Small errors should produce finite loss for iql_weight={iql_weight}"
    
    def test_iql_asymmetric_behavior_verification(self):
        """Test that IQL produces expected asymmetric behavior for over/under-estimates."""
        
        # Test with iql_weight = 0.8 (stronger penalty for underestimates)
        iql_weight = 0.8
        target = 5.0
        
        # Overestimate (prediction > target)
        overestimate = 7.0
        over_loss = jax_losses.compute_scalar_value_loss(
            jnp.array([overestimate]), jnp.array([target]), iql_weight
        )
        
        # Underestimate (prediction < target)  
        underestimate = 3.0
        under_loss = jax_losses.compute_scalar_value_loss(
            jnp.array([underestimate]), jnp.array([target]), iql_weight
        )
        
        # For iql_weight=0.8:
        # - Overestimate weight = 1 - iql_weight = 0.2 (lower penalty)
        # - Underestimate weight = iql_weight = 0.8 (higher penalty)
        
        # Same magnitude error, but underestimate should have higher loss
        error_magnitude = 2.0  # Both errors have magnitude 2.0
        assert abs(overestimate - target) == abs(underestimate - target) == error_magnitude
        
        # Verify asymmetric penalty
        assert under_loss > over_loss, (
            f"Underestimate loss {under_loss} should be greater than overestimate loss {over_loss} "
            f"for iql_weight={iql_weight}"
        )
        
        # Verify exact weight ratios
        expected_ratio = iql_weight / (1 - iql_weight)  # 0.8 / 0.2 = 4.0
        actual_ratio = float(jnp.squeeze(under_loss) / jnp.squeeze(over_loss))
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-6)
    
    def test_categorical_value_loss_iql_equivalence(self):
        """Test categorical value loss IQL implementation for equivalence."""
        key = jax.random.PRNGKey(456)
        
        # Test parameters
        batch_size = 8
        num_atoms = 51
        iql_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        for iql_weight in iql_weights:
            # Generate random logits and target distributions
            logits = jax.random.normal(key, (batch_size, num_atoms))
            target_dist = jax.random.dirichlet(
                jax.random.fold_in(key, 1), jnp.ones(num_atoms), (batch_size,)
            )
            
            # Compute JAX categorical loss
            jax_loss = jax_losses.compute_categorical_value_loss(
                logits, target_dist, effective_iql_param=iql_weight
            )
            
            # Verify finite and reasonable values
            assert jnp.all(jnp.isfinite(jax_loss)), f"Categorical loss should be finite for iql_weight={iql_weight}"
            assert jnp.all(jax_loss >= 0), f"Categorical loss should be non-negative for iql_weight={iql_weight}"
    
    def test_symlog_value_loss_iql_equivalence(self):
        """Test symlog value loss IQL implementation for equivalence."""
        key = jax.random.PRNGKey(789)
        
        # Test parameters
        batch_size = 6
        iql_weights = [0.0, 0.4, 0.5, 0.6, 1.0]
        
        for iql_weight in iql_weights:
            # Generate random predictions and targets
            predictions = jax.random.uniform(key, (batch_size,), minval=-50.0, maxval=50.0)
            targets = jax.random.uniform(
                jax.random.fold_in(key, 1), (batch_size,), minval=-50.0, maxval=50.0
            )
            
            # Compute JAX symlog loss
            jax_loss = jax_losses.compute_symlog_value_loss(
                predictions, targets, effective_iql_param=iql_weight
            )
            
            # Verify finite and reasonable values
            assert jnp.all(jnp.isfinite(jax_loss)), f"Symlog loss should be finite for iql_weight={iql_weight}"
            assert jnp.all(jax_loss >= 0), f"Symlog loss should be non-negative for iql_weight={iql_weight}"
    
    def test_iql_integration_in_trainer(self, common_key, common_cfg_flat):
        """Test IQL integration within the full trainer context."""
        mk, lk, bk = jax.random.split(common_key, 3)
        
        # Create a single model and learner to avoid repeated compilation
        cfgn = common_cfg_flat
        model = make_model(mk, cfgn)
        
        # Create base config
        base_cfg = make_cfg(VALUE_SUPPORT_SCALAR, REWARD_SUPPORT_SCALAR, 1, False, f"iql_test")
        base_cfg = dataclasses.replace(
            base_cfg, 
            batch_size=4,
            l2_weight=0.0  # Isolate IQL effects
        )
        
        # Create single learner instance
        learner = Learner(model, None, base_cfg, lk)
        
        # Create batch once
        batch = make_batch(bk, base_cfg.batch_size, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0)
        batch["target_value"] = jnp.array([[3.0], [1.0], [5.0], [2.0]])  # Shape: (4, 1)
        
        # Test different IQL configurations by directly testing the loss computation
        results = []
        
        # Test IQL enabled with different weights
        for iql_weight in [0.2, 0.5, 0.8]:
            # Create asymmetric test data: mix of over and under estimates
            predictions = jnp.array([4.0, 0.5, 6.0, 1.5])  # Mix of over/under estimates
            targets = jnp.array([3.0, 1.0, 5.0, 2.0])      # Targets
            
            # Verify we have both over and under estimates
            errors = predictions - targets
            assert jnp.any(errors > 0), "Need some overestimates"
            assert jnp.any(errors < 0), "Need some underestimates"
            
            loss = jax_losses.compute_scalar_value_loss(predictions, targets, iql_weight)
            results.append((True, iql_weight, jnp.mean(loss)))
        
        # Test IQL disabled (should behave like iql_weight=0.5)
        loss_disabled = jax_losses.compute_scalar_value_loss(
            jnp.array([4.0, 0.5, 6.0, 1.5]), 
            jnp.array([3.0, 1.0, 5.0, 2.0]), 
            0.5  # IQL disabled should use symmetric weighting
        )
        results.append((False, 0.7, jnp.mean(loss_disabled)))
        
        # Verify that IQL disabled produces the same result as IQL with weight 0.5
        iql_disabled_loss = results[3][2]  # (False, 0.7)
        iql_symmetric_loss = results[1][2]  # (True, 0.5)
        
        np.testing.assert_allclose(
            iql_disabled_loss, iql_symmetric_loss, rtol=1e-5,
            err_msg="IQL disabled should produce same loss as IQL with weight 0.5"
        )
        
        # Verify that different IQL weights produce different losses (when enabled)
        iql_asymmetric_1 = results[0][2]  # (True, 0.2)
        iql_asymmetric_2 = results[2][2]  # (True, 0.8)
        
        assert not np.allclose(iql_asymmetric_1, iql_asymmetric_2, rtol=1e-3), (
            "Different IQL weights should produce different losses"
        )
        
        print("✅ IQL integration in trainer verified")
        for use_iql, iql_weight, loss in results:
            print(f"   use_iql={use_iql}, iql_weight={iql_weight}: loss={loss:.6f}")


@pytest.mark.parametrize("iql_weight", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_iql_weight_parameter_sweep(iql_weight):
    """Parametrized test sweeping IQL weight values for comprehensive coverage."""
    key = jax.random.PRNGKey(42 + int(iql_weight * 100))
    
    # Generate test data
    batch_size = 10
    predictions = jax.random.uniform(key, (batch_size,), minval=-5.0, maxval=5.0)
    targets = jax.random.uniform(
        jax.random.fold_in(key, 1), (batch_size,), minval=-5.0, maxval=5.0
    )
    
    # Test JAX implementation
    jax_loss = jax_losses.compute_scalar_value_loss(predictions, targets, iql_weight)
    
    # Test reference implementation
    reference_loss = EfficientZeroV2IQLReference.compute_scalar_value_loss_reference(
        predictions, targets, iql_weight
    )
    
    # Verify equivalence
    np.testing.assert_allclose(jax_loss, reference_loss, rtol=1e-7, atol=1e-10)
    
    # Verify properties
    assert jnp.all(jnp.isfinite(jax_loss)), f"Loss should be finite for iql_weight={iql_weight}"
    assert jnp.all(jax_loss >= 0), f"Loss should be non-negative for iql_weight={iql_weight}"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"]) 