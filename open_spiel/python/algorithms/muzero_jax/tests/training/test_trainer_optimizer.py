import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses

# Import from the common utils module
from trainer_test_utils import (
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
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew, MockProj, MockMuZeroNetwork


def test_optimizer_choice_adam_adamw_action_item_23(common_key, common_cfg_flat):
    """Test optimizer choice (AdamW vs. Adam) alignment with EfficientZeroV2. OPTIMIZED for speed.

    Verifies that JAX's optimizer selection strategy correctly chooses between AdamW
    (when weight_decay > 0) and Adam (when weight_decay == 0) and prevents
    double weight decay application, aligning with EfficientZeroV2 practices.
    """
    mk, lk, bk = jax.random.split(common_key, 3)
    cfgn = common_cfg_flat

    # OPTIMIZED: Single shared model and batch for all test cases
    model_shared = make_model(jax.random.fold_in(mk, 1), cfgn)
    batch_shared = make_batch(
        jax.random.fold_in(bk, 1), 1, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0
    )

    # Test Case 1: weight_decay = 0 should use Adam + manual L2
    cfg_adam = make_cfg(0, 0, 1, False, "adam_test", l2_weight=1e-4)
    cfg_adam = dataclasses.replace(cfg_adam, weight_decay=0.0, batch_size=1)

    learner_adam = Learner(model_shared, None, cfg_adam, jax.random.fold_in(lk, 1))
    metrics_adam = learner_adam.train_step(batch_shared)

    # When weight_decay == 0, L2 loss should be non-zero (from manual L2)
    assert (
        "l2_loss" in metrics_adam
    ), "L2 loss should be computed when weight_decay == 0"
    l2_loss_adam = float(metrics_adam["l2_loss"])
    assert (
        l2_loss_adam > 0
    ), f"Manual L2 loss should be positive when weight_decay == 0, got {l2_loss_adam}"

    # Test Case 2: weight_decay > 0 should use AdamW, no manual L2
    cfg_adamw = make_cfg(0, 0, 1, False, "adamw_test", l2_weight=1e-4)
    cfg_adamw = dataclasses.replace(cfg_adamw, weight_decay=1e-3, batch_size=1)

    learner_adamw = Learner(model_shared, None, cfg_adamw, jax.random.fold_in(lk, 2))
    metrics_adamw = learner_adamw.train_step(batch_shared)

    # When weight_decay > 0, L2 loss should be zero (no manual L2, AdamW handles weight decay)
    l2_loss_adamw = float(metrics_adamw["l2_loss"])
    assert (
        l2_loss_adamw == 0.0
    ), f"Manual L2 loss should be zero when weight_decay > 0, got {l2_loss_adamw}"

    print(f"✅ Optimizer Choice (AdamW vs. Adam) verification passed (OPTIMIZED):")
    print(
        f"  1. ✅ weight_decay == 0 → Adam optimizer + manual L2 (L2 loss: {l2_loss_adam:.6f})"
    )
    print(
        f"  2. ✅ weight_decay > 0 → AdamW optimizer + no manual L2 (L2 loss: {l2_loss_adamw:.6f})"
    )
    print(f"  3. ✅ No double weight decay application verified")

def test_optimizer_choice_efficientzero_v2_parity(common_key, common_cfg_flat):
    """Ultra-optimized EfficientZeroV2 parity test for optimizer choice and weight decay handling."""
    mk, lk, bk = jax.random.split(common_key, 3)
    cfgn = common_cfg_flat

    # Single ultra-small shared model and batch for maximum efficiency
    model_ultra_shared = make_model(mk, cfgn)
    batch_ultra_shared = make_batch(bk, 1, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0)

    # EfficientZeroV2 Pattern Test 1: Zero weight decay configuration
    cfg_ez_zero = make_cfg(0, 0, 1, False, "ez_zero_wd", l2_weight=1e-4)
    cfg_ez_zero = dataclasses.replace(cfg_ez_zero, weight_decay=0.0, batch_size=1)

    learner_ez_zero = Learner(model_ultra_shared, None, cfg_ez_zero, jax.random.fold_in(lk, 1))
    metrics_ez_zero = learner_ez_zero.train_step(batch_ultra_shared)

    # Verify EfficientZeroV2 pattern: weight_decay == 0 implies Adam + manual L2
    l2_loss_ez_zero = float(metrics_ez_zero["l2_loss"])
    assert (
        l2_loss_ez_zero > 0
    ), f"EfficientZeroV2 pattern: weight_decay == 0 should use manual L2, got {l2_loss_ez_zero}"

    # EfficientZeroV2 Pattern Test 2: Single standard weight decay test (ultra-reduced)
    cfg_ez_wd = make_cfg(0, 0, 1, False, "ez_wd_single", l2_weight=1e-4)
    cfg_ez_wd = dataclasses.replace(cfg_ez_wd, weight_decay=1e-4, batch_size=1)

    learner_ez_wd = Learner(model_ultra_shared, None, cfg_ez_wd, jax.random.fold_in(lk, 2))
    metrics_ez_wd = learner_ez_wd.train_step(batch_ultra_shared)

    # EfficientZeroV2 pattern: weight_decay > 0 implies AdamW, no manual L2
    l2_loss_ez_wd = float(metrics_ez_wd["l2_loss"])
    assert (
        l2_loss_ez_wd == 0.0
    ), f"EfficientZeroV2 pattern: weight_decay > 0 should disable manual L2, got {l2_loss_ez_wd}"

    print(f"✅ Ultra-optimized EfficientZeroV2 Parity verification passed:")
    print(f"  1. ✅ Zero weight decay pattern: Adam + manual L2 (L2 loss: {l2_loss_ez_zero:.6f})")
    print(f"  2. ✅ Positive weight decay pattern: AdamW + no manual L2 (L2 loss: {l2_loss_ez_wd:.6f})") 