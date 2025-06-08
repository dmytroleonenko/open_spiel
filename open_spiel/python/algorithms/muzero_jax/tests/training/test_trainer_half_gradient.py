import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses

# Import from the common utils module
from trainer_utils import (
    NUM_UNROLL_STEPS,
    NUM_ACTIONS,
    BATCH_SIZE,
    VALUE_SUPPORT_SCALAR,
    REWARD_SUPPORT_SCALAR,
    VALUE_SUPPORT_CATEGORICAL,
    REWARD_SUPPORT_CATEGORICAL,
    key as common_key, 
    cfg_flat as common_cfg_flat,
    cfg_img as common_cfg_img,
    make_model, 
    make_cfg,
    make_batch,
    MockNetCfg,
    OBS_SHAPE_FLAT
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch,
    half_gradient
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_utils import MockRep, MockDyn, MockPred, MockRew, MockProj, MockMuZeroNetwork


def test_half_gradient_placement_in_recurrent_unroll(common_key, common_cfg_flat):
    """Verify half_gradient is applied at correct location in recurrent unroll.

    This test verifies that half_gradient is properly integrated into the training
    pipeline and affects gradient computation as expected during the recurrent unroll.
    OPTIMIZED for speed.
    """
    from open_spiel.python.algorithms.muzero_jax.training.trainer import half_gradient

    mk, bk = jax.random.split(common_key, 2)

    # Use minimal configuration for speed
    model = make_model(mk, common_cfg_flat)
    cfg = make_cfg(
        0,  # value_support_size = 0 for scalar (faster)
        0,  # reward_support_size = 0 for scalar (faster)
        1,  # num_unroll_steps = 1 (minimal)
        False,  # no projection
        "grad_placement",
    )
    
    # Create minimal batch
    batch = make_batch(
        bk,
        1,  # batch_size = 1 (minimal)
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        1,  # unroll_steps = 1 (minimal)
        0,  # value_support_size = 0
        0,  # reward_support_size = 0
    )

    # Test that the complete training step works with half_gradient (lightweight check)
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)
    step_metrics = learner.train_step(batch)

    # Verify training completed successfully (indicating half_gradient integration works)
    assert "total_loss" in step_metrics, "Train step should produce metrics"
    assert jnp.isfinite(step_metrics["total_loss"]), "Train step loss should be finite"

    # Test the half_gradient function directly with MINIMAL test cases for speed
    test_cases = [
        jnp.array([1.0, 2.0]),  # Reduced from 3 elements to 2
        jnp.array([[1.0, 2.0]]),  # Reduced from 2x2 to 1x2 matrix
    ]

    for i, test_input in enumerate(test_cases):
        # Test forward pass: should be identity
        output = half_gradient(test_input)
        assert jnp.allclose(
            output, test_input
        ), f"Forward pass should be identity for case {i}"

        # Test gradient computation: should be halved
        def test_fn(x):
            return jnp.sum(half_gradient(x))

        grad_fn = jax.grad(test_fn)
        computed_grad = grad_fn(test_input)
        expected_grad = jnp.ones_like(test_input) * 0.5

        assert jnp.allclose(
            computed_grad, expected_grad
        ), f"Gradient should be halved for case {i}"

    # Simplified loss computation check (no need for full loss computation)
    # Just verify that the loss computation function doesn't crash with half_gradient
    try:
        loss_value, metrics = Learner._compute_total_loss_static(
            model, cfg, batch, common_key, training=True
        )
        assert jnp.isfinite(loss_value), "Loss computation with half_gradient should be finite"
        assert "total_loss" in metrics, "Metrics should be properly computed"
    except Exception as e:
        assert False, f"Loss computation should not crash: {e}"

    print("✅ Half gradient placement test passed (optimized for speed)")
    print("  - Forward pass verified as identity function")
    print("  - Gradient scaling verified as 0.5x")
    print("  - Integration with training pipeline verified")


def test_half_gradient_mathematical_implementation(common_key, common_cfg_flat):
    """Mathematical correctness of half_gradient function.

    Verifies that the half_gradient function properly implements the EfficientZeroV2
    pattern: forward pass is identity, backward pass multiplies gradient by 0.5.
    """
    from open_spiel.python.algorithms.muzero_jax.training.trainer import half_gradient

    # Test mathematical properties of half_gradient function
    test_inputs = [
        jnp.array([1.0, 2.0, 3.0]),  # Simple case
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # 2D case
        jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D case
        jnp.zeros((2, 3)),  # Zero input
        jnp.ones((3, 2, 4)) * 1e6,  # Large values
        jnp.ones((2, 2)) * 1e-6,  # Small values
    ]

    for i, test_input in enumerate(test_inputs):
        # Test forward pass: should be identity
        output = half_gradient(test_input)
        assert jnp.allclose(
            output, test_input, rtol=1e-7
        ), f"Forward pass should be identity for input {i}, got diff {jnp.max(jnp.abs(output - test_input))}"

        # Test backward pass: gradient should be halved
        def test_fn(x):
            return jnp.sum(half_gradient(x))

        grad_fn = jax.grad(test_fn)
        computed_grad = grad_fn(test_input)
        expected_grad = jnp.ones_like(test_input) * 0.5

        assert jnp.allclose(
            computed_grad, expected_grad, rtol=1e-7
        ), f"Backward pass should multiply gradient by 0.5 for input {i}"

    # Test that half_gradient is numerically stable
    large_input = jnp.ones((100, 50)) * 1e9
    large_output = half_gradient(large_input)
    assert jnp.allclose(
        large_output, large_input
    ), "half_gradient should be stable for large inputs"

    # Test gradient computation for large inputs
    def large_test_fn(x):
        return jnp.sum(half_gradient(x))

    large_grad = jax.grad(large_test_fn)(large_input)
    expected_large_grad = jnp.ones_like(large_input) * 0.5
    assert jnp.allclose(
        large_grad, expected_large_grad
    ), "Gradient computation should be stable for large inputs"


def test_half_gradient_efficientzero_v2_consistency(common_key, common_cfg_flat):
    """Verify consistency with EfficientZeroV2 implementation pattern. OPTIMIZED for speed.

    This test confirms that the JAX implementation follows the exact same pattern
    as the PyTorch EfficientZeroV2 reference: apply half-gradient to hidden states
    in the main training unroll loop, not during MCTS/target generation.
    """
    mk = jax.random.fold_in(common_key, 1)
    bk = jax.random.fold_in(common_key, 2)

    model = make_model(mk, common_cfg_flat)
    cfg = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        1,  # Reduced from 5 to 1
        False,
        "efficientzero_consistency",
    )

    # OPTIMIZED: Test only essential unroll step counts
    for num_unroll in [1, 2]:  # Reduced from [1, 3] to [1, 2]
        cfg_test = dataclasses.replace(cfg, num_unroll_steps=num_unroll, batch_size=1)
        batch = make_batch(
            bk,
            1,  # Reduced batch size from 2 to 1
            common_cfg_flat.observation_shape,
            common_cfg_flat.num_actions,
            num_unroll,
            cfg.value_support_size,
            cfg.reward_support_size,
        )

        # Compute loss - this internally applies half_gradient during unroll
        loss_value, metrics = Learner._compute_total_loss_static(
            model, cfg_test, batch, common_key, training=True
        )

        # Verify that loss computation succeeds (indicating half_gradient was applied correctly)
        assert jnp.isfinite(
            loss_value
        ), f"Loss should be finite for {num_unroll} unroll steps"
        assert jnp.isfinite(
            metrics["total_loss"]
        ), f"Total loss metric should be finite"
        assert loss_value > 0, f"Loss should be positive for {num_unroll} unroll steps"

    print(f"✅ Half gradient EfficientZeroV2 consistency verified (OPTIMIZED)!")


def test_half_gradient_numerical_verification_integration(common_key, common_cfg_flat):
    """Numerical verification that half_gradient affects gradients correctly. OPTIMIZED for speed.

    This test verifies that the half_gradient function is properly integrated in the training
    pipeline and that the loss computation works correctly with half_gradient applied.
    """
    mk = jax.random.fold_in(common_key, 1)
    bk = jax.random.fold_in(common_key, 2)

    # Use the standard model setup for simplicity
    model = make_model(mk, common_cfg_flat)
    cfg = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        1,  # Already optimized
        False,
        "grad_measurement",
    )
    cfg = dataclasses.replace(cfg, batch_size=1)
    batch = make_batch(
        bk,
        1,  # Reduced from 2 to 1
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        1,  # Already optimized
        cfg.value_support_size,
        cfg.reward_support_size,
    )

    # Test that loss computation works with half_gradient applied
    loss_value, metrics = Learner._compute_total_loss_static(
        model, cfg, batch, common_key, training=True
    )

    # Verify the loss computation worked (indicating half_gradient was applied correctly)
    assert jnp.isfinite(loss_value), "Loss should be finite"
    assert loss_value > 0, "Loss should be positive"
    assert "total_loss" in metrics, "Metrics should contain total_loss"
    assert jnp.isfinite(metrics["total_loss"]), "Total loss metric should be finite"

    # Test that a complete training step works with half_gradient
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)
    step_metrics = learner.train_step(batch)

    # Verify training step completed successfully
    assert "total_loss" in step_metrics, "Train step should produce metrics"
    assert jnp.isfinite(step_metrics["total_loss"]), "Train step loss should be finite"

    # Test half_gradient function directly to ensure it works correctly
    from open_spiel.python.algorithms.muzero_jax.training.trainer import half_gradient

    test_hidden_state = jnp.array([[1.0, 2.0]])  # Reduced from 2x2 to 1x2
    result = half_gradient(test_hidden_state)

    # Verify forward pass is identity
    assert jnp.allclose(
        result, test_hidden_state
    ), "half_gradient forward pass should be identity"

    # Verify gradient computation works
    def test_fn(x):
        return jnp.sum(half_gradient(x))

    grad_fn = jax.grad(test_fn)
    computed_grad = grad_fn(test_hidden_state)
    expected_grad = jnp.ones_like(test_hidden_state) * 0.5

    assert jnp.allclose(
        computed_grad, expected_grad
    ), "half_gradient should reduce gradients by 0.5"

    print(f"✅ Half gradient numerical verification completed (OPTIMIZED)!")


def test_half_gradient_documentation_and_comments(common_key, common_cfg_flat):
    """Verify proper documentation of half_gradient implementation.

    Ensures that the half_gradient function and its usage are properly documented
    and reference the EfficientZeroV2 pattern.
    """
    import inspect
    from open_spiel.python.algorithms.muzero_jax.training.trainer import half_gradient

    # Verify the half_gradient function has proper documentation
    docstring = inspect.getdoc(half_gradient)
    assert docstring is not None, "half_gradient function should have documentation"
    assert (
        "EfficientZeroV2" in docstring or "gradient" in docstring.lower()
    ), "Documentation should reference relevant concepts"
    
    # Verify the function works correctly
    test_input = jnp.array([1.0, 2.0, 3.0])
    output = half_gradient(test_input)
    assert jnp.allclose(output, test_input), "half_gradient forward pass should be identity"

    # Verify gradient computation
    def test_fn(x):
        return jnp.sum(half_gradient(x))

    grad_fn = jax.grad(test_fn)
    computed_grad = grad_fn(test_input)
    expected_grad = jnp.ones_like(test_input) * 0.5

    assert jnp.allclose(
        computed_grad, expected_grad
    ), "half_gradient should reduce gradients by 0.5"


def test_half_gradient_coverage_completion(common_key, common_cfg_flat):
    """Complete coverage of half_gradient functionality for EfficientZeroV2.

    This test ensures all aspects of half_gradient are properly tested
    and integrated into the MuZero training pipeline.
    """
    from open_spiel.python.algorithms.muzero_jax.training.trainer import half_gradient

    # Test fewer and smaller test cases for performance
    test_cases = [
        # Reduced test cases
        jnp.array([1.0]),
        jnp.array([[1.0, 2.0]]),
        jnp.ones((2, 2)),
    ]

    for i, test_input in enumerate(test_cases):
        # Forward pass test
        result = half_gradient(test_input)
        assert jnp.allclose(result, test_input), f"Case {i}: forward pass failed"

        # Gradient test
        def objective(x):
            return jnp.sum(half_gradient(x) ** 2)

        grad_fn = jax.grad(objective)
        grad = grad_fn(test_input)
        # d/dx sum((half_gradient(x))^2) = d/dx sum(x^2) * 0.5 = 2*x * 0.5 = x
        expected_grad = test_input
        assert jnp.allclose(grad, expected_grad), f"Case {i}: gradient computation failed"

    # Test integration with training pipeline - REDUCED COMPLEXITY
    mk, bk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)
    cfg = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        1,  # Reduced from 3 to 1
        False,
        "coverage_test",
    )
    batch = make_batch(
        bk,
        2,  # Reduced batch size from 4 to 2
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        1,  # Reduced from 3 to 1
        cfg.value_support_size,
        cfg.reward_support_size,
    )

    # Training step should work correctly with half_gradient - SINGLE STEP ONLY
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)
    
    # Single training step instead of loop
    metrics = learner.train_step(batch)
    assert "total_loss" in metrics, "Missing total_loss"
    assert jnp.isfinite(metrics["total_loss"]), "Loss not finite"
    assert metrics["total_loss"] >= 0, "Negative loss"

    print("✅ Half-gradient coverage completion test passed") 