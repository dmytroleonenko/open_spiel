import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock

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
    MockNetCfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_utils import MockRep, MockDyn, MockPred, MockRew 

def test_checkpoint_error_handling_fast(common_key, common_cfg_flat):
    """Fast test for checkpoint error handling edge cases."""
    # Test with no checkpoint manager (should skip gracefully)
    cfg_no_ckpt = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_no_ckpt", 
        use_ema=False, checkpoint_dir=None
    )

    model = make_model(common_key, common_cfg_flat)
    learner_no_ckpt = Learner(model, None, cfg_no_ckpt, common_key)

    # Should not crash and return early
    learner_no_ckpt.save_checkpoint(force_save=True)  # No-op
    success = learner_no_ckpt.load_checkpoint()  # Should return False
    assert success == False
    
    # Test save/load error handling with mocks (no real checkpoint operations)
    failing_manager = Mock()
    failing_manager.latest_step.return_value = 5
    failing_manager.restore.side_effect = ValueError("Mock restore error")
    failing_manager.save.side_effect = RuntimeError("Mock save error")
    
    learner_no_ckpt.checkpoint_manager = failing_manager
    
    with patch("logging.error") as mock_logging_error:
        result = learner_no_ckpt.load_checkpoint()
        learner_no_ckpt.save_checkpoint(force_save=True)

    assert result is False
    assert mock_logging_error.call_count == 2  # Both save and load should log errors 

def test_ema_and_target_network_edge_cases(common_key, common_cfg_flat):
    """Fast test for EMA and target network edge cases."""
    # Test EMA disabled (should be no-op)
    cfg_no_ema = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_no_ema", 
        use_ema=False, checkpoint_dir=None
            )
    
    model = make_model(common_key, common_cfg_flat)
    learner_no_ema = Learner(model, None, cfg_no_ema, common_key)
    
    # EMA is disabled, so no target network should exist
    assert learner_no_ema.target_model is None
    
    # Test cleanup with exception (should not propagate)  
    failing_manager = Mock()
    failing_manager.close.side_effect = RuntimeError("Cleanup error")
    learner_no_ema.checkpoint_manager = failing_manager
    
    # This should not raise an exception
    learner_no_ema.__del__()  # Should silently handle the exception 

def test_squeeze_edge_cases_fast(common_key, common_cfg_flat):
    """Fast test for squeeze operations in loss computations."""
    # Test value loss with (B, 1) shape that triggers squeeze
    class EdgeValuePred(nnx.Module):
        def __init__(self, *, rngs):
            pass
        def __call__(self, h, training):
            policy = jnp.ones((h.shape[0], NUM_ACTIONS)) * 0.1
            value = jnp.ones((h.shape[0], 1))  # B, 1 - triggers squeeze
            return policy, value

    cfg_edge = make_cfg(vsup=0, rsup=0, steps=1, proj=False, suffix="_edge", use_ema=False)

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred_edge = lambda model_config, *, rngs: EdgeValuePred(rngs=rngs)
    rew = lambda model_config, *, rngs: MockRew(common_cfg_flat.hidden_size, 0, rngs=rngs)

    model_edge = MuZeroNetwork(
        rep, dyn, pred_edge, rew, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    batch_edge = make_batch(
        common_key, cfg_edge.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_edge.num_unroll_steps, 
        vsup=0, rsup=0, use_proj=False
    )
    
    # Test both scalar and with extra dimension
    loss_edge, metrics_edge = Learner._compute_total_loss_static(
        model_edge, cfg_edge, batch_edge, common_key, training=True
    )
    assert loss_edge.shape == ()
    assert "value_loss" in metrics_edge 

def test_utility_functions_fast(common_key, common_cfg_flat):
    """Fast test for utility functions."""
    # Test apply_value_prefix_reward_accumulation disabled
    from open_spiel.python.algorithms.muzero_jax.training.trainer import (
        apply_value_prefix_reward_accumulation, generate_top_new_masks
    )

    config_no_prefix = MuZeroConfig(use_value_prefix=False)
    test_rewards = jnp.ones((2, 3))
    result_no_prefix = apply_value_prefix_reward_accumulation(test_rewards, config_no_prefix)
    assert jnp.allclose(result_no_prefix, test_rewards)

    # Test generate_top_new_masks with conversion
    sample_indices = jnp.array([100, 200, 300])
    masks = generate_top_new_masks(sample_indices, 250, 50)
    expected = jnp.array([False, False, True])
    assert jnp.array_equal(masks, expected)
    assert masks.dtype == jnp.bool_  # Masks should be boolean 