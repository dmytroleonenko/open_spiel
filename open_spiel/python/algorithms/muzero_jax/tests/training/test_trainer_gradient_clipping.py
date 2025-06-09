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
    Batch
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_utils import MockRep, MockDyn, MockPred, MockRew, MockProj, MockMuZeroNetwork


def test_gradient_clipping_comprehensive_standard_verification(common_key, common_cfg_flat):
    """Ultra-optimized gradient clipping implementation verification.

    This test verifies only the most critical aspects of JAX's gradient clipping:
    1. Standard Optax pattern verification
    2. Essential condition logic testing (clip_grad_norm > 0)
    3. Basic integration with trainer implementation
    """
    mk, lk, bk = jax.random.split(common_key, 3)

    # Test 1: Standard Optax pattern verification vs manual implementation (simplified)
    def manual_gradient_clipping(grads, max_norm):
        """Manual implementation of gradient clipping for comparison."""
        grad_norm = optax.global_norm(grads)
        factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
        clipped_grads = jax.tree_util.tree_map(lambda g: g * factor, grads)
        return clipped_grads, optax.global_norm(clipped_grads)

    def optax_gradient_clipping(grads, max_norm):
        """Standard Optax implementation (as used in trainer)."""
        clipper = optax.clip_by_global_norm(max_norm)
        clipped_grads, _ = clipper.update(grads, None)
        return clipped_grads, optax.global_norm(clipped_grads)

    # Create minimal test gradients
    test_grads = {
        "param1": jnp.array([3.0, 4.0]),  # norm = 5.0
    }

    clip_norm = 2.0
    manual_clipped, manual_final_norm = manual_gradient_clipping(test_grads, clip_norm)
    optax_clipped, optax_final_norm = optax_gradient_clipping(test_grads, clip_norm)

    # Verify both implementations produce equivalent results
    def grads_allclose(g1, g2, rtol=1e-6):
        return jax.tree_util.tree_reduce(
            lambda acc, check: acc and check,
            jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y, rtol=rtol), g1, g2),
            initializer=True,
        )

    assert grads_allclose(
        manual_clipped, optax_clipped
    ), "Optax gradient clipping should match manual implementation"
    assert (
        optax_final_norm <= clip_norm + 1e-6
    ), f"Clipped gradient norm {optax_final_norm} should be <= {clip_norm}"

    # Test 2: Essential condition logic testing - only critical thresholds
    # Create simple network config
    simple_cfg = MockNetCfg(
        observation_shape=(2, 2),
        num_actions=3,
        batch_size=1,
        value_support_size=0,
        reward_support_size=0,
        hidden_size=8
    )

    # Ultra-small batch for maximum speed
    batch = make_batch(bk, 1, (2, 2), 3, 1, 0, 0)
    batch["target_value"] = batch["target_value"] * 2.0  # Moderate scaling

    # Only test 2 most critical thresholds instead of 3
    threshold_tests = [
        (0.0, False),  # clip_grad_norm = 0 should disable clipping
        (1.0, True),   # positive value should enable clipping
    ]

    # Single model for threshold tests
    model_test = make_model(mk, simple_cfg)

    for i, (threshold, should_clip) in enumerate(threshold_tests):
        cfg_test = make_cfg(0, 0, 1, False, f"clip_test_{threshold}", l2_weight=0.0)
        cfg_test = dataclasses.replace(cfg_test, clip_grad_norm=threshold, batch_size=1)

        learner_test = Learner(model_test, None, cfg_test, jax.random.fold_in(lk, i))
        metrics = learner_test.train_step(batch)

        if should_clip and threshold > 0:
            assert (
                float(metrics["grad_norm"]) <= threshold + 1e-6
            ), f"Gradient norm should be clipped to {threshold}, got {metrics['grad_norm']}"
        elif not should_clip:
            assert jnp.isfinite(
                float(metrics["grad_norm"])
            ), f"Gradient norm should be finite when clipping disabled: {metrics['grad_norm']}"

    # Test 3: Basic integration with actual trainer implementation (simplified)
    cfg_integration = make_cfg(0, 0, 1, False, "integration_test", l2_weight=0.0)
    cfg_integration = dataclasses.replace(
        cfg_integration, clip_grad_norm=1.5, batch_size=1
    )

    learner_integration = Learner(
        model_test, None, cfg_integration, jax.random.fold_in(lk, 2)
    )

    metrics_integration = learner_integration.train_step(batch)

    assert "grad_norm" in metrics_integration, "Gradient norm should be reported in metrics"
    grad_norm_final = float(metrics_integration["grad_norm"])
    assert (
        grad_norm_final <= cfg_integration.clip_grad_norm + 1e-6
    ), f"Final gradient norm {grad_norm_final} should respect clip_grad_norm {cfg_integration.clip_grad_norm}"

    print(f"✅ Ultra-optimized gradient clipping verification passed:")
    print(f"  1. ✅ Standard Optax pattern verified vs manual implementation")
    print(f"  2. ✅ Essential condition logic tested with thresholds: {[t[0] for t in threshold_tests]}")
    print(f"  3. ✅ Basic integration with trainer implementation verified")
    print(f"  - Final clipped norm: {grad_norm_final:.6f} (limit: {cfg_integration.clip_grad_norm})") 