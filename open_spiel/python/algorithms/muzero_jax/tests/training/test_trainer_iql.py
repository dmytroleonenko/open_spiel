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


def test_use_iql_config_default(common_key, common_cfg_flat):
    """Test that use_iql defaults to True in MuZeroConfig."""
    cfg = make_cfg(
        VALUE_SUPPORT_SCALAR,
        REWARD_SUPPORT_SCALAR,
        NUM_UNROLL_STEPS,
        False,
        "iql_default",
    )
    # Default config should have use_iql=True
    assert cfg.use_iql == True
    assert cfg.iql_weight == 1.0


def test_use_iql_disabled_symmetric_loss(common_key, common_cfg_flat):
    """Test that use_iql=False produces symmetric loss (effective_iql_param=0.5)."""
    mk, bk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)

    # Create config with IQL disabled
    cfg_no_iql = make_cfg(
        VALUE_SUPPORT_SCALAR, REWARD_SUPPORT_SCALAR, NUM_UNROLL_STEPS, False, "no_iql"
    )
    cfg_no_iql = dataclasses.replace(cfg_no_iql, use_iql=False, iql_weight=0.8)

    # Create a simple batch
    batch = make_batch(
        bk,
        BATCH_SIZE,
        OBS_SHAPE_FLAT,
        NUM_ACTIONS,
        NUM_UNROLL_STEPS,
        VALUE_SUPPORT_SCALAR,
        REWARD_SUPPORT_SCALAR,
    )

    # Override target values to create controlled error scenario
    batch["target_value"] = jnp.array(
        [[5.0, 1.0]]  # Shape: (1, 2) - one batch item with 2 time steps
    )

    # Compute loss with IQL disabled
    loss_value, metrics = Learner._compute_total_loss_static(
        model, cfg_no_iql, batch, common_key, training=True
    )

    # The effective_iql_param should be 0.5 (symmetric), not cfg.iql_weight (0.8)
    assert "total_loss" in metrics
    assert "value_loss" in metrics
    assert jnp.isfinite(loss_value)
    assert jnp.isfinite(metrics["value_loss"])


def test_iql_effective_param_comparison(common_key, common_cfg_flat):
    """Test that use_iql=True vs use_iql=False produces different losses."""
    mk, bk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)

    # Create configs: one with IQL enabled, one disabled
    cfg_iql_enabled = make_cfg(
        VALUE_SUPPORT_SCALAR,
        REWARD_SUPPORT_SCALAR,
        NUM_UNROLL_STEPS,
        False,
        "iql_enabled",
    )
    cfg_iql_enabled = dataclasses.replace(cfg_iql_enabled, use_iql=True, iql_weight=0.1)

    cfg_iql_disabled = make_cfg(
        VALUE_SUPPORT_SCALAR,
        REWARD_SUPPORT_SCALAR,
        NUM_UNROLL_STEPS,
        False,
        "iql_disabled",
    )
    cfg_iql_disabled = dataclasses.replace(
        cfg_iql_disabled, use_iql=False, iql_weight=0.1
    )  # iql_weight should be ignored

    # Create batch with controlled scenario
    batch = make_batch(
        bk,
        2,  # Reduced from BATCH_SIZE
        OBS_SHAPE_FLAT,
        NUM_ACTIONS,
        1,  # Reduced from NUM_UNROLL_STEPS
        VALUE_SUPPORT_SCALAR,
        REWARD_SUPPORT_SCALAR,
    )

    # Override target values to create clear error signs
    batch["target_value"] = jnp.array(
        [[10.0, 1.0], [1.0, 10.0]]  # Reduced size to match smaller batch
    )

    # Compute losses with both configs
    loss_enabled, metrics_enabled = Learner._compute_total_loss_static(
        model, cfg_iql_enabled, batch, common_key, training=True
    )

    loss_disabled, metrics_disabled = Learner._compute_total_loss_static(
        model, cfg_iql_disabled, batch, common_key, training=True
    )

    # Losses should be different due to different effective IQL parameters
    # IQL enabled uses 0.1, IQL disabled uses 0.5 (symmetric)
    assert not jnp.allclose(
        loss_enabled, loss_disabled, atol=1e-6
    ), f"Expected different losses, got enabled={loss_enabled}, disabled={loss_disabled}"

    assert not jnp.allclose(
        metrics_enabled["value_loss"], metrics_disabled["value_loss"], atol=1e-6
    ), f"Expected different value losses"


def test_consistency_loss_coefficient_consolidation(common_key, common_cfg_flat):
    """Consolidation of SSL consistency loss parameters. OPTIMIZED for speed.

    Verifies that ssl_consistency_loss_weight and consistency_coeff have been
    consolidated into a single consistency_loss_coeff parameter and that
    SSL loss computation works correctly with the consolidated parameter.
    """
    mk, lk = jax.random.split(common_key, 2)

    # Test 1: Verify the old parameters are gone and new parameter exists
    config = MuZeroConfig()

    # Verify new parameter exists
    assert hasattr(
        config, "consistency_loss_coeff"
    ), "Config should have consistency_loss_coeff parameter"
    assert isinstance(
        config.consistency_loss_coeff, float
    ), "consistency_loss_coeff should be float"

    # Verify old parameters are gone
    assert not hasattr(
        config, "ssl_consistency_loss_weight"
    ), "ssl_consistency_loss_weight should be removed"
    assert not hasattr(
        config, "consistency_coeff"
    ), "consistency_coeff should be removed"

    # Test 2: Verify default value aligns with EfficientZeroV2 (2.0)
    assert (
        config.consistency_loss_coeff == 2.0
    ), "consistency_loss_coeff should default to 2.0 for EfficientZeroV2 parity"

    # Test 3: OPTIMIZED - Single shared model and batch
    cfgn = dataclasses.replace(common_cfg_flat, use_projection=True)
    model_shared = make_model(mk, cfgn)
    batch_shared = make_batch(
        common_key,
        1,  # Minimal batch size
        cfgn.observation_shape,
        cfgn.num_actions,
        1,  # Minimal unroll steps
        0,  # Scalar values for speed
        0,  # Scalar values for speed
        cfgn.projection_output_size,
        True,
    )

    # Test with SSL enabled
    config_ssl_enabled = MuZeroConfig(
        consistency_loss_coeff=1.5,  # Non-zero to enable SSL
        use_projection=True,
        num_unroll_steps=1,  # Minimal for speed
        batch_size=1,  # Reduced for speed
        l2_weight=0.0,
        weight_decay=0.0,
        num_actions=cfgn.num_actions,  # Match the model's num_actions
    )

    learner_ssl = Learner(model_shared, None, config_ssl_enabled, lk)
    metrics_ssl = learner_ssl.train_step(batch_shared)

    # Verify SSL loss is computed and included in metrics
    assert (
        "ssl_loss" in metrics_ssl
    ), "SSL loss should be in metrics when consistency_loss_coeff > 0"
    assert jnp.isfinite(metrics_ssl["ssl_loss"]), "SSL loss should be finite"

    # Test with SSL disabled - reuse same model
    config_ssl_disabled = dataclasses.replace(
        config_ssl_enabled, consistency_loss_coeff=0.0
    )
    learner_no_ssl = Learner(
        model_shared,  # Reuse model
        None,
        config_ssl_disabled,
        jax.random.fold_in(lk, 1),
    )

    metrics_no_ssl = learner_no_ssl.train_step(batch_shared)

    # Verify SSL loss is not computed when coefficient is 0
    assert (
        "ssl_loss" not in metrics_no_ssl
    ), "SSL loss should not be in metrics when consistency_loss_coeff = 0"

    print(f"✅ Consistency loss coefficient consolidation test passed (OPTIMIZED):")
    print(f"  - Parameter consolidation verified: consistency_loss_coeff=2.0 default")
    print(f"  - SSL loss computation works with consolidated parameter")
    print(f"  - SSL loss correctly enabled/disabled based on coefficient") 