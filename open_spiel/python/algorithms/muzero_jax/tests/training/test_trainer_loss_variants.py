import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import dataclasses
from unittest.mock import patch, Mock

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
    Batch,
    create_muzero_config_for_game
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_utils import MockRep, MockDyn, MockPred, MockRew


def test_create_muzero_config_for_game():
    """Test create_muzero_config_for_game function - line 2052 area."""
    # Test creating config for a specific game
    config = create_muzero_config_for_game("tic_tac_toe")
    assert config.num_actions == 9  # Tic-tac-toe has 9 actions
    
    # Test with config overrides
    config_custom = create_muzero_config_for_game(
        "tic_tac_toe", 
        learning_rate=1e-5,
        batch_size=64
    )
    assert config_custom.num_actions == 9
    assert config_custom.learning_rate == 1e-5
    assert config_custom.batch_size == 64


def test_value_prefix_lstm_reset_coverage(common_key, common_cfg_flat):
    """Test coverage for value prefix LSTM reset - around line 650."""
    cfg_value_prefix = make_cfg(
        vsup=0, rsup=0, steps=3, proj=False, suffix="_value_prefix",
        use_ema=False, use_value_prefix=True, lstm_horizon_length=2
    )
    cfg_value_prefix = dataclasses.replace(cfg_value_prefix, use_value_prefix=True)

    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key, cfg_value_prefix.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_value_prefix.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Test loss computation with value prefix enabled
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_value_prefix, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_squeeze_edge_cases_value_reward(common_key, common_cfg_flat):
    """Test squeeze operations for various tensor shapes - lines 726, 737."""
    class EdgeNetworks:
        def __init__(self, mode='scalar_reward', pred_shape='B1', target_shape='B1'):
            self.mode = mode
            self.pred_shape = pred_shape
            self.target_shape = target_shape

        def make_pred_net(self):
            pred_shape = self.pred_shape
            class ScalarRewardPred(nnx.Module):
                def __init__(self, *, rngs):
                    pass
                def __call__(self, h, training):
                    policy = jnp.ones((h.shape[0], NUM_ACTIONS)) * 0.1
                    value = jnp.ones((h.shape[0], 1)) if pred_shape == 'B1' else jnp.ones((h.shape[0],))
                    return policy, value
            return ScalarRewardPred

        def make_rew_net(self):
            pred_shape = self.pred_shape
            class ScalarReward(nnx.Module):
                def __init__(self, *, rngs):
                    pass
                def __call__(self, h, training):
                    return jnp.ones((h.shape[0], 1)) if pred_shape == 'B1' else jnp.ones((h.shape[0],))
            return ScalarReward

    # Test with B,1 shaped predictions that trigger squeeze
    edge_nets = EdgeNetworks('scalar_reward', 'B1', 'B1')
    cfg_edge = make_cfg(vsup=0, rsup=0, steps=1, proj=False, suffix="_edge", use_ema=False)

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred_edge = lambda model_config, *, rngs: edge_nets.make_pred_net()(rngs=rngs)
    rew_edge = lambda model_config, *, rngs: edge_nets.make_rew_net()(rngs=rngs)

    model_edge = MuZeroNetwork(
        rep, dyn, pred_edge, rew_edge, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    batch_edge = make_batch(
        common_key, cfg_edge.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_edge.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Test loss computation with squeezable tensors
    loss, metrics = Learner._compute_total_loss_static(
        model_edge, cfg_edge, batch_edge, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_symlog_loss_paths(common_key, common_cfg_flat):
    """Test symlog loss computation paths that might not be covered."""
    cfg_symlog = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_symlog", 
        use_ema=False, value_loss_type="symlog", reward_loss_type="symlog"
    )

    # Create config with symlog
    cfg_symlog = dataclasses.replace(
        cfg_symlog,
        value_loss_type="symlog",
        reward_loss_type="symlog",
        use_symlog=True
    )

    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key, cfg_symlog.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_symlog.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Test symlog loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_symlog, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_gae_value_target_type_coverage(common_key, common_cfg_flat):
    """Test GAE value target type selection - around line 548."""
    cfg_gae = make_cfg(
        vsup=0, rsup=0, steps=2, proj=False, suffix="_gae",
        use_ema=False, value_target_type="GAE", value_target="sarsa"
    )
    
    # Create config with GAE value target type
    cfg_gae = dataclasses.replace(
        cfg_gae,
        value_target_type="GAE",
        value_target="sarsa",
        use_adaptive_td_steps=True
    )

    model = make_model(common_key, common_cfg_flat)
    
    # Create batch with extra observations for GAE
    batch_size = cfg_gae.batch_size
    num_unroll_steps = cfg_gae.num_unroll_steps
    extra_steps = 3  # Extra steps for GAE
    obs_shape = common_cfg_flat.observation_shape
    
    # Create base batch
    batch = make_batch(
        common_key, batch_size, obs_shape, common_cfg_flat.num_actions,
        num_unroll_steps, vsup=0, rsup=0, use_proj=False
    )
    
    # Add extra observations, actions, rewards, dones for GAE with correct shapes
    total_steps = num_unroll_steps + 1 + extra_steps  # K+1+extra = 2+1+3 = 6
    extra_obs = jnp.ones((batch_size, total_steps) + obs_shape)
    extra_actions = jnp.ones((batch_size, total_steps - 1), dtype=jnp.int32)  # B, 5
    extra_rewards = jnp.ones((batch_size, total_steps))  # B, 6
    extra_dones = jnp.zeros((batch_size, total_steps))  # B, 6
    
    batch['extra_observations'] = extra_obs
    batch['extra_actions'] = extra_actions
    batch['extra_rewards'] = extra_rewards
    batch['extra_dones'] = extra_dones
    batch['sample_indices'] = jnp.array([100, 200, 150, 300])[:batch_size]  # For adaptive td_lambda
    batch['collected_transitions'] = 500

    # Test loss computation with GAE
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_gae, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_value_target_search_mode(common_key, common_cfg_flat):
    """Test value target selection in search mode."""
    cfg_search = make_cfg(
        vsup=0, rsup=0, steps=2, proj=False, suffix="_search",
        use_ema=False, value_target="search"
    )
    
    cfg_search = dataclasses.replace(cfg_search, value_target="search")

    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key, cfg_search.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_search.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )
    
    # Add search values
    batch['target_search_value'] = batch['target_value'] + 0.1  # Different from target_value

    # Test loss computation with search values
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_search, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_multiple_value_heads_coverage(common_key, common_cfg_flat):
    """Test multiple value heads (v_num > 1) - around line 682."""
    cfg_multi_v = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_multi_v",
        use_ema=False, v_num=2
    )
    
    cfg_multi_v = dataclasses.replace(cfg_multi_v, v_num=2)

    # Create model that outputs multiple value heads
    class MultiValuePred(nnx.Module):
        def __init__(self, *, rngs):
            pass
        def __call__(self, h, training):
            policy = jnp.ones((h.shape[0], NUM_ACTIONS)) * 0.1
            value = jnp.ones((h.shape[0], 2))  # 2 value heads
            return policy, value

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred_multi = lambda model_config, *, rngs: MultiValuePred(rngs=rngs)
    rew = lambda model_config, *, rngs: MockRew(common_cfg_flat.hidden_size, 0, rngs=rngs)

    model_multi = MuZeroNetwork(
        rep, dyn, pred_multi, rew, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    batch = make_batch(
        common_key, cfg_multi_v.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_multi_v.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Test loss computation with multiple value heads
    loss, metrics = Learner._compute_total_loss_static(
        model_multi, cfg_multi_v, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_policy_reanalysis_exception_coverage(common_key, common_cfg_flat):
    """Test policy reanalysis exception handling - around line 598."""
    cfg_reanalysis = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_reanalysis",
        use_ema=False, reanalyze_ratio=0.5
    )
    
    cfg_reanalysis = dataclasses.replace(cfg_reanalysis, reanalyze_ratio=0.5)

    # Create a model that works normally but has a patched reanalysis function
    model = make_model(common_key, common_cfg_flat)

    batch = make_batch(
        common_key, cfg_reanalysis.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_reanalysis.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Patch the reanalysis function to raise an exception
    def mock_reanalysis_that_fails(*args, **kwargs):
        raise ValueError("Simulated reanalysis failure")

    # Test that exception during reanalysis is handled gracefully
    with patch('open_spiel.python.algorithms.muzero_jax.training.trainer.compute_policy_reanalysis_targets', 
               side_effect=mock_reanalysis_that_fails):
        with patch('warnings.warn') as mock_warn:
            loss, metrics = Learner._compute_total_loss_static(
                model, cfg_reanalysis, batch, common_key, training=True
            )
            assert loss.shape == ()
            assert jnp.isfinite(loss)
            # Should have warned about reanalysis failure
            mock_warn.assert_called()


def test_kl_reward_loss_type(common_key, common_cfg_flat):
    """Test KL reward loss type - covers reward loss computation branches."""
    cfg_kl = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_kl",
        use_ema=False, reward_loss_type="kl"
    )
    
    cfg_kl = dataclasses.replace(cfg_kl, reward_loss_type="kl")

    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key, cfg_kl.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_kl.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Test KL reward loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_kl, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_entropy_loss_coverage(common_key, common_cfg_flat):
    """Test entropy loss computation - around line 908."""
    cfg_entropy = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_entropy",
        use_ema=False, entropy_coeff=0.01
    )
    
    cfg_entropy = dataclasses.replace(
        cfg_entropy, 
        entropy_coeff=0.01,
        action_type="discrete",
        distribution_type="categorical"
    )

    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        common_key, cfg_entropy.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_entropy.num_unroll_steps,
        vsup=0, rsup=0, use_proj=False
    )

    # Test entropy loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg_entropy, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_ssl_projection_loss_coverage(common_key, common_cfg_flat):
    """Test SSL projection loss - around line 896."""
    cfg_ssl = make_cfg(
        vsup=0, rsup=0, steps=2, proj=True, suffix="_ssl",
        use_ema=False, consistency_loss_coeff=2.0
    )
    
    cfg_ssl = dataclasses.replace(
        cfg_ssl,
        use_projection=True,
        consistency_loss_coeff=2.0
    )

    # Create projection function
    def proj_net(model_config, *, rngs):
        class ProjNet(nnx.Module):
            def __init__(self, *, rngs):
                pass
            def __call__(self, h, training):
                return jnp.ones((h.shape[0], 128))  # Projection dimension
        return ProjNet(rngs=rngs)

    # Use proper model creation with projection net
    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred = lambda model_config, *, rngs: MockPred(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, 0, rngs=rngs
    )
    rew = lambda model_config, *, rngs: MockRew(common_cfg_flat.hidden_size, 0, rngs=rngs)

    model_ssl = MuZeroNetwork(
        rep, dyn, pred, rew, proj_net, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    batch = make_batch(
        common_key, cfg_ssl.batch_size, common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions, cfg_ssl.num_unroll_steps,
        vsup=0, rsup=0, use_proj=True
    )

    # Test SSL projection loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model_ssl, cfg_ssl, batch, common_key, training=True
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert 'ssl_loss' in metrics or 'total_loss' in metrics 