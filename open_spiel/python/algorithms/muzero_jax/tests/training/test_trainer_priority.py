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


def test_priority_computation_and_batch_indices(common_key, common_cfg_flat):
    """Test that priorities are computed correctly when batch has indices."""
    mk = jax.random.fold_in(common_key, 1)
    bk = jax.random.fold_in(common_key, 2)

    # Create model and learner with priority replay enabled
    model = make_model(mk, common_cfg_flat)
    cfg = dataclasses.replace(
        make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,
            False,
            "priority",
        ),
        use_priority_replay=True,
        min_priority=0.01,
    )
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)

    # Create batch with indices for priority replay
    batch = make_batch(
        bk,
        cfg.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg.num_unroll_steps,
        cfg.value_support_size,
        cfg.reward_support_size,
    )
    batch["indices"] = jnp.array([0, 1])  # Add buffer indices
    batch["weights"] = jnp.array([1.0, 0.5])  # Add importance sampling weights

    # Perform train step
    metrics = learner.train_step(batch)

    # Check that priorities were computed
    assert (
        "priorities" in metrics
    ), "Priorities should be computed when use_priority_replay=True and indices present"
    assert "indices" in metrics, "Indices should be returned in metrics"

    priorities = metrics["priorities"]
    indices = metrics["indices"]

    # Check priority shape and values
    assert priorities.shape == (
        cfg.batch_size,
    ), f"Priorities should have shape {(cfg.batch_size,)}, got {priorities.shape}"
    assert jnp.all(
        priorities >= cfg.min_priority
    ), f"All priorities should be >= min_priority ({cfg.min_priority})"
    assert jnp.array_equal(
        indices, batch["indices"]
    ), "Returned indices should match batch indices"


def test_priority_replay_disabled_no_priority_computation(common_key, common_cfg_flat):
    """Test that priorities are not computed when priority replay is disabled."""
    mk = jax.random.fold_in(common_key, 1)
    bk = jax.random.fold_in(common_key, 2)

    model = make_model(mk, common_cfg_flat)
    cfg = dataclasses.replace(
        make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,
            False,
            "no_priority",
        ),
        use_priority_replay=False,  # Disable priority replay
    )

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)

    # Create batch with indices (but priority replay disabled)
    batch = make_batch(
        bk,
        cfg.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg.num_unroll_steps,
        cfg.value_support_size,
        cfg.reward_support_size,
    )
    batch["indices"] = jnp.array([0, 1])

    # Perform train step
    metrics = learner.train_step(batch)

    # Check that priorities were NOT computed
    assert (
        "priorities" not in metrics
    ), "Priorities should not be computed when use_priority_replay=False"
    assert (
        "indices" not in metrics
    ), "Indices should not be returned when use_priority_replay=False" 