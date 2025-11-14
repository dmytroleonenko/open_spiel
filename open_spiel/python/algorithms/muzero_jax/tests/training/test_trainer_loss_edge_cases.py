import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import dataclasses

from trainer_utils import (
    NUM_ACTIONS,
    key as common_key, 
    cfg_flat as common_cfg_flat,
    make_model, 
    make_cfg,
    make_batch,
    MockRep, MockDyn, MockPred, MockRew
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork


def test_value_loss_categorical_squeeze_coverage(common_key, common_cfg_flat):
    """Test value loss with categorical predictions that need squeeze - line 726, 737."""
    
    # Create config for categorical value loss
    cfg_cat = make_cfg(
        vsup=21, rsup=0, steps=1, proj=False, suffix="_cat_val", 
        use_ema=False, value_loss_type="categorical"
    )
    cfg_cat = dataclasses.replace(cfg_cat, value_loss_type="categorical", value_support_size=21)

    # Create model that outputs values with shape (B, 1) to trigger squeeze on line 726
    class CategoricalValuePred(nnx.Module):
        def __init__(self, *, rngs):
            pass
            
        def __call__(self, h, training):
            policy = jnp.ones((h.shape[0], NUM_ACTIONS)) * 0.1
            # Output categorical values with shape (B, 1) to trigger the squeeze
            value = jnp.ones((h.shape[0], 1))  # This will trigger line 726 squeeze
            return policy, value

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred = lambda model_config, *, rngs: CategoricalValuePred(rngs=rngs)
    rew = lambda model_config, *, rngs: MockRew(common_cfg_flat.hidden_size, 0, rngs=rngs)

    model = MuZeroNetwork(
        rep, dyn, pred, rew, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    # Create batch with target values that have shape (B, 1) to trigger line 737 squeeze
    batch = make_batch(
        common_key, cfg_cat.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_cat.num_unroll_steps, 
        vsup=21, rsup=0, use_proj=False
    )
    
    # Modify target values to have shape (B, K+1, 1) to trigger squeeze
    target_values_squeezable = jnp.ones((batch["target_value"].shape[0], batch["target_value"].shape[1], 1))
    batch["target_value"] = target_values_squeezable
    
    # This should trigger the squeeze operations on lines 726 and 737
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_cat, batch, common_key, training=True
    )
    
    assert loss.shape == ()
    assert "value_loss" in metrics


def test_reward_loss_categorical_squeeze_coverage(common_key, common_cfg_flat):
    """Test reward loss with categorical predictions that need squeeze."""
    
    # Create config for categorical reward loss  
    cfg_cat = make_cfg(
        vsup=0, rsup=21, steps=1, proj=False, suffix="_cat_rew", 
        use_ema=False, reward_loss_type="categorical"
    )
    cfg_cat = dataclasses.replace(cfg_cat, reward_loss_type="categorical", reward_support_size=21)

    # Create model where reward prediction outputs (B, 1) to trigger squeeze
    class CategoricalRewardRew(nnx.Module):
        def __init__(self, hidden_size, support_size, *, rngs):
            pass
            
        def __call__(self, h, training):
            # Output rewards with shape (B, 1) to trigger squeeze
            return jnp.ones((h.shape[0], 1))

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred = lambda model_config, *, rngs: MockPred(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, 0, rngs=rngs
    )
    rew = lambda model_config, *, rngs: CategoricalRewardRew(
        common_cfg_flat.hidden_size, 21, rngs=rngs
    )

    model = MuZeroNetwork(
        rep, dyn, pred, rew, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    # Create batch with target rewards that have shape (B, K+1, 1) to trigger squeeze
    batch = make_batch(
        common_key, cfg_cat.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_cat.num_unroll_steps, 
        vsup=0, rsup=21, use_proj=False
    )
    
    # Modify target rewards to have shape (B, K+1, 1) to trigger squeeze
    target_rewards_squeezable = jnp.ones((batch["target_reward"].shape[0], batch["target_reward"].shape[1], 1))
    batch["target_reward"] = target_rewards_squeezable
    
    # This should trigger the squeeze operations for rewards
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_cat, batch, common_key, training=True
    )
    
    assert loss.shape == ()
    assert "reward_loss" in metrics


def test_symlog_reward_squeeze_coverage(common_key, common_cfg_flat):
    """Test symlog reward loss with (B, 1) target shape to trigger squeeze."""
    
    cfg_symlog = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_symlog", 
        use_ema=False, reward_loss_type="symlog"
    )
    cfg_symlog = dataclasses.replace(cfg_symlog, reward_loss_type="symlog")

    model = make_model(common_key, common_cfg_flat)

    batch = make_batch(
        common_key, cfg_symlog.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_symlog.num_unroll_steps, 
        vsup=0, rsup=0, use_proj=False
    )
    
    # Modify target rewards to have shape (B, K+1, 1) to trigger squeeze in symlog loss
    target_rewards_squeezable = jnp.ones((batch["target_reward"].shape[0], batch["target_reward"].shape[1], 1))
    batch["target_reward"] = target_rewards_squeezable
    
    # This should trigger the squeeze operation for symlog rewards
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_symlog, batch, common_key, training=True
    )
    
    assert loss.shape == ()
    assert "reward_loss" in metrics


def test_mse_value_and_reward_squeeze_coverage(common_key, common_cfg_flat):
    """Test MSE loss with (B, 1) predictions to trigger squeeze."""
    
    cfg_mse = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_mse", 
        use_ema=False, value_loss_type="mse", reward_loss_type="mse"
    )
    cfg_mse = dataclasses.replace(cfg_mse, value_loss_type="mse", reward_loss_type="mse")

    # Create model that outputs predictions with shape (B, 1) to trigger squeeze
    class MSEPredWithSqueeze(nnx.Module):
        def __init__(self, *, rngs):
            pass
            
        def __call__(self, h, training):
            policy = jnp.ones((h.shape[0], NUM_ACTIONS)) * 0.1
            # Output value with shape (B, 1) to trigger squeeze in MSE loss
            value = jnp.ones((h.shape[0], 1))
            return policy, value

    class MSERewardWithSqueeze(nnx.Module):
        def __init__(self, hidden_size, support_size, *, rngs):
            pass
            
        def __call__(self, h, training):
            # Output rewards with shape (B, 1) to trigger squeeze in MSE loss
            return jnp.ones((h.shape[0], 1))

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred = lambda model_config, *, rngs: MSEPredWithSqueeze(rngs=rngs)
    rew = lambda model_config, *, rngs: MSERewardWithSqueeze(
        common_cfg_flat.hidden_size, 0, rngs=rngs
    )

    model = MuZeroNetwork(
        rep, dyn, pred, rew, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    batch = make_batch(
        common_key, cfg_mse.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_mse.num_unroll_steps, 
        vsup=0, rsup=0, use_proj=False
    )
    
    # Modify targets to have shape (B, K+1, 1) to trigger squeeze in MSE loss
    target_values_squeezable = jnp.ones((batch["target_value"].shape[0], batch["target_value"].shape[1], 1))
    target_rewards_squeezable = jnp.ones((batch["target_reward"].shape[0], batch["target_reward"].shape[1], 1))
    batch["target_value"] = target_values_squeezable
    batch["target_reward"] = target_rewards_squeezable
    
    # This should trigger squeeze operations in MSE loss paths
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_mse, batch, common_key, training=True
    )
    
    assert loss.shape == ()
    assert "value_loss" in metrics
    assert "reward_loss" in metrics


def test_loss_static_accepts_positional_args(common_key, common_cfg_flat):
    """Legacy positional args should still be honored for training flags."""
    cfg = make_cfg(
        vsup=0,
        rsup=0,
        steps=1,
        proj=False,
        suffix="_posargs",
        use_ema=False,
    )
    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key,
        cfg.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg.num_unroll_steps,
        vsup=0,
        rsup=0,
        use_proj=False,
    )

    loss, metrics = Learner._compute_total_loss_static(
        model,
        cfg,
        batch,
        common_key,
        False,
        7,
    )

    assert loss.shape == ()
    assert "policy_loss" in metrics


def test_loss_static_raises_on_time_dimension_mismatch(common_key, common_cfg_flat):
    """Time-dimension mismatches should raise a clear ValueError."""
    cfg = make_cfg(
        vsup=0,
        rsup=0,
        steps=2,
        proj=False,
        suffix="_timedim",
        use_ema=False,
    )
    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key,
        cfg.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg.num_unroll_steps,
        vsup=0,
        rsup=0,
        use_proj=False,
    )
    # Drop the final timestep from the mask to force a mismatch
    batch["game_history_mask"] = batch["game_history_mask"][:, :-1]

    with pytest.raises(ValueError, match="Time-dimension mismatch"):
        Learner._compute_total_loss_static(
            model,
            cfg,
            batch,
            common_key,
            training=True,
        )
