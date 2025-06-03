import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses
import tempfile
import flax.nnx.graph as nnx_graph

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
    MockNetCfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew, MockProj 

def maybe_val(leaf):
    """Extract value from nnx state leaf if it has .value, otherwise return as-is."""
    if hasattr(leaf, 'value'):
        return leaf.value
    return leaf


def test_ema_checkpoint_fallback_edge_cases(common_key, common_cfg_flat):
    """Test additional edge cases for EMA checkpoint fallback scenarios."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test case: Normal EMA checkpoint loading (both save and load with EMA enabled)
        # This verifies that when EMA data IS present in checkpoint, it loads correctly
        cfg = make_cfg(
            0, 0, 1, False, "ema_normal", use_ema=True, checkpoint_dir=temp_dir
        )
        cfg = dataclasses.replace(cfg, checkpoint_frequency=1)

        # Create and train first learner with EMA
        model1 = make_model(common_key, cfg)
        learner1 = Learner(model1, None, cfg, common_key)
        batch = make_batch(common_key, cfg.batch_size, common_cfg_flat.observation_shape, NUM_ACTIONS, 1, 0, 0)
        learner1.train_step(batch)

        # Get state before saving
        saved_online_params = nnx.state(learner1.model, nnx.Param)
        saved_ema_params = learner1.ema_params_state.ema
        saved_target_params = nnx.state(learner1.target_model, nnx.Param)

        # Save checkpoint with EMA components
        learner1.save_checkpoint(force_save=True)

        # Clean up first learner
        if learner1.checkpoint_manager is not None:
            learner1.checkpoint_manager.close()

        # Create second learner with EMA and load checkpoint
        cfg_resume = dataclasses.replace(cfg, resume_from_checkpoint=True)
        model2 = make_model(jax.random.fold_in(common_key, 1), cfg_resume)
        learner2 = Learner(model2, None, cfg_resume, jax.random.fold_in(common_key, 2))

        def params_equal(p1, p2):
            """Check if two parameter trees are equal."""

            def compare_leaf(leaf1, leaf2):
                v1 = maybe_val(leaf1)
                v2 = maybe_val(leaf2)
                return jnp.allclose(v1, v2, rtol=1e-6)

            return jax.tree_util.tree_all(jax.tree_util.tree_map(compare_leaf, p1, p2))

        # Verify all components were loaded correctly
        loaded_online_params = nnx.state(learner2.model, nnx.Param)
        loaded_ema_params = learner2.ema_params_state.ema
        loaded_target_params = nnx.state(learner2.target_model, nnx.Param)

        assert params_equal(
            loaded_online_params, saved_online_params
        ), "Online parameters should be loaded correctly"
        assert params_equal(
            loaded_ema_params, saved_ema_params
        ), "EMA parameters should be loaded correctly"
        assert params_equal(
            loaded_target_params, saved_target_params
        ), "Target model parameters should be loaded correctly"

        # Verify EMA components are properly initialized
        assert learner2.ema_params_state is not None, "EMA state should be loaded"
        assert learner2.target_model is not None, "Target model should be loaded"
        assert learner2.ema_updater is not None, "EMA updater should be initialized"

        # Test that EMA continues to work correctly after loading
        batch2 = make_batch(
            jax.random.fold_in(common_key, 2),
            cfg.batch_size,
            common_cfg_flat.observation_shape,
            NUM_ACTIONS,
            1,
            0,
            0,
        )
        pre_training_ema = learner2.ema_params_state.ema
        learner2.train_step(batch2)
        post_training_ema = learner2.ema_params_state.ema

        if cfg.ema_decay < 1.0:
            params_changed = not params_equal(post_training_ema, pre_training_ema)
            assert params_changed, "EMA should continue updating after checkpoint load"

        # Clean up
        if learner2.checkpoint_manager is not None:
            learner2.checkpoint_manager.close() 