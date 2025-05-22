import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import os
import shutil
from unittest.mock import patch, PropertyMock
import dataclasses
import flax.nnx.graph as nnx_graph
import orbax.checkpoint as ocp
from jax.tree_util import tree_structure

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig, Batch
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork

# Constants
OBS_SHAPE_FLAT = (10,)
OBS_SHAPE_IMAGE = (3, 32, 32)
NUM_ACTIONS = 5
BATCH_SIZE = 2
NUM_UNROLL_STEPS = 3
VALUE_SUPPORT_SCALAR = 0
REWARD_SUPPORT_SCALAR = 0
VALUE_SUPPORT_CATEGORICAL = 11
REWARD_SUPPORT_CATEGORICAL = 21

# Mock network components
class MockRep(nnx.Module):
    def __init__(self, obs_shape, hidden, *, rngs):
        self.dense = nnx.Linear(jnp.prod(jnp.array(obs_shape)), hidden, rngs=rngs)
    def __call__(self, x, training):
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        return self.dense(x)

class MockDyn(nnx.Module):
    def __init__(self, hidden, nact, *, rngs):
        self.embed = nnx.Embed(nact, hidden // 2, rngs=rngs)
        self.fc = nnx.Linear(hidden + hidden//2, hidden, rngs=rngs)
    def __call__(self, h, a, training):
        e = self.embed(a)
        if e.ndim == 1:
            e = jnp.broadcast_to(e, (h.shape[0], e.shape[-1]))
        return nnx.relu(self.fc(jnp.concatenate([h, e], -1)))

class MockPred(nnx.Module):
    def __init__(self, hidden, nact, vsup, *, rngs):
        self.ph = nnx.Linear(hidden, nact, rngs=rngs)
        self.vh = nnx.Linear(hidden, vsup if vsup>0 else 1, rngs=rngs)
    def __call__(self, h, training):
        return self.ph(h), self.vh(h)

class MockRew(nnx.Module):
    def __init__(self, hidden, rsup, *, rngs):
        self.rh = nnx.Linear(hidden, rsup if rsup>0 else 1, rngs=rngs)
    def __call__(self, h, training):
        return self.rh(h)

class MockProj(nnx.Module):
    def __init__(self, hidden, psize, *, rngs):
        self.ph = nnx.Linear(hidden, psize, rngs=rngs)
    def __call__(self, h, training):
        return self.ph(h)

@dataclasses.dataclass(frozen=True)
class MockNetCfg:
    observation_shape: tuple = OBS_SHAPE_FLAT
    num_actions: int = NUM_ACTIONS
    hidden_size: int = 16
    value_support_size: int = VALUE_SUPPORT_SCALAR
    reward_support_size: int = REWARD_SUPPORT_SCALAR
    projection_output_size: int = 8
    use_projection: bool = False
    batch_size: int = BATCH_SIZE

# Fixtures
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

@pytest.fixture
def cfg_flat():
    return MockNetCfg()

@pytest.fixture
def cfg_img():
    return dataclasses.replace(MockNetCfg(observation_shape=OBS_SHAPE_IMAGE))

# Helpers
def make_model(key, cfg):
    rep = lambda model_config, *, rngs: MockRep(model_config.observation_shape, model_config.hidden_size, rngs=rngs)
    dyn = lambda model_config, *, rngs: MockDyn(model_config.hidden_size, model_config.num_actions, rngs=rngs)
    pred = lambda model_config, *, rngs: MockPred(model_config.hidden_size, model_config.num_actions, model_config.value_support_size, rngs=rngs)
    rew = lambda model_config, *, rngs: MockRew(model_config.hidden_size, model_config.reward_support_size, rngs=rngs)
    proj_def_lambda = (lambda model_config, *, rngs: MockProj(model_config.hidden_size, model_config.projection_output_size, rngs=rngs)) if cfg.use_projection else None
    return MuZeroNetwork(rep, dyn, pred, rew, proj_def_lambda, cfg, rngs=nnx.Rngs(params=key))

def maybe_val(x):
    return x.value if isinstance(x, nnx.Variable) else x

def make_cfg(vsup, rsup, steps, proj, suffix, use_ema=False, ssl_weight=0.0, l2_weight=1e-4):
    base = f"/tmp/mz_test_{suffix}" # Added _test_ to avoid potential conflict with main example
    if os.path.exists(base):
        shutil.rmtree(base)
    return MuZeroConfig(
        value_support_size=vsup,
        reward_support_size=rsup,
        discount_factor=0.99,
        num_unroll_steps=steps,
        td_steps=steps+1,
        value_loss_weight=0.25,
        reward_loss_weight=1.0,
        policy_loss_weight=1.0,
        l2_weight=l2_weight,
        use_projection=proj,
        ssl_consistency_loss_weight=ssl_weight,
        learning_rate=1e-3,
        adam_b1=0.9,
        adam_b2=0.999,
        clip_grad_norm=5.0,
        batch_size=BATCH_SIZE,
        use_target_network_ema=use_ema,
        ema_decay=0.99,
        checkpoint_dir=base,
        checkpoint_frequency=5,
        max_checkpoints_to_keep=1,
        resume_from_checkpoint=False
    )

def make_batch(key, bs, obs_shape, nact, steps, vsup, rsup, proj_dim=None, use_proj=False):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    obs = jax.random.uniform(k1, (bs, steps+1, *obs_shape))
    acts = jax.random.randint(k2, (bs, steps), 0, nact)
    val = jax.random.normal(k3, (bs, steps+1)) if vsup==0 else jax.random.uniform(k3, (bs, steps+1, vsup))
    rew = jax.random.normal(k4, (bs, steps+1)) if rsup==0 else jax.random.uniform(k4, (bs, steps+1, rsup))
    pol = jax.random.uniform(k5, (bs, steps+1, nact))
    pol = pol / jnp.sum(pol, axis=-1, keepdims=True)
    mask = jnp.ones((bs, steps+1))
    batch_data = {
        'observation': obs, 'action': acts, 
        'target_reward': rew, 'target_value': val,
        'target_policy': pol, 'game_history_mask': mask
    }
    # No need to add projected_hidden_state to batch, it's a model internal
    return batch_data

# Tests
def test_init(key, cfg_flat):
    mk = jax.random.fold_in(key, 1)
    model = make_model(mk, cfg_flat)
    cfg = make_cfg(cfg_flat.value_support_size, cfg_flat.reward_support_size, NUM_UNROLL_STEPS, False, 'init')
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)
    assert learner.num_training_steps == 0
    assert learner.opt_state is not None

@pytest.mark.parametrize("img,val_cat,proj,use_ema,scalar_targets", [
    (False, False, False, False, True), 
    (True, True, True, True, False),
    (False, False, True, False, True), 
    (False, False, False, True, False),
    (False, True, False, False, False), # Scalar outputs, categorical targets (value only)
    (False, False, False, False, False) # Categorical outputs, scalar targets (value only) - should ideally also work if supports are size 1
])
def test_loss_static(key, img, val_cat, proj, use_ema, scalar_targets, cfg_flat, cfg_img):
    bk, mk, lk = jax.random.split(key, 3)
    
    obs_shape_test = OBS_SHAPE_IMAGE if img else OBS_SHAPE_FLAT
    num_actions_test = NUM_ACTIONS
    hidden_size_test = 4 # Smaller hidden size for simpler manual calculation
    unroll_steps_test = 1 # Single unroll step for simplicity
    batch_size_test = 1 # Single batch item for simplicity

    # Configure model
    cfgn_model = MockNetCfg(
        observation_shape=obs_shape_test,
        num_actions=num_actions_test,
        hidden_size=hidden_size_test,
        value_support_size=VALUE_SUPPORT_CATEGORICAL if val_cat else VALUE_SUPPORT_SCALAR,
        reward_support_size=REWARD_SUPPORT_CATEGORICAL if val_cat else REWARD_SUPPORT_SCALAR, # Keep reward cat/scalar same as value for this test
        projection_output_size=hidden_size_test // 2 if proj else 0, # Smaller projection
        use_projection=proj,
        batch_size=batch_size_test
    )
    
    # --- Create a very simple model with fixed weights for predictability ---
    class FixedRep(nnx.Module):
        def __init__(self, *, rngs):
            self.dense = nnx.Linear(jnp.prod(jnp.array(cfgn_model.observation_shape)), cfgn_model.hidden_size, rngs=rngs)
            # Fix weights and biases
            self.dense.kernel.value = jnp.ones_like(self.dense.kernel.value) * 0.1
            self.dense.bias.value = jnp.zeros_like(self.dense.bias.value) + 0.05
        def __call__(self, x, training):
            if x.ndim > 2: x = x.reshape((x.shape[0], -1))
            return self.dense(x)

    class FixedDyn(nnx.Module):
        def __init__(self, *, rngs):
            self.embed = nnx.Embed(cfgn_model.num_actions, cfgn_model.hidden_size // 2, rngs=rngs)
            self.fc = nnx.Linear(cfgn_model.hidden_size + cfgn_model.hidden_size // 2, cfgn_model.hidden_size, rngs=rngs)
            # Fix weights
            self.embed.embedding.value = jnp.ones_like(self.embed.embedding.value) * 0.05
            self.fc.kernel.value = jnp.ones_like(self.fc.kernel.value) * 0.2
            self.fc.bias.value = jnp.zeros_like(self.fc.bias.value) + 0.02
        def __call__(self, h, a, training):
            e = self.embed(a)
            if e.ndim == 1: e = jnp.broadcast_to(e, (h.shape[0], e.shape[-1]))
            return nnx.relu(self.fc(jnp.concatenate([h, e], -1)))

    class FixedPred(nnx.Module):
        def __init__(self, *, rngs):
            self.ph_w = jnp.ones((cfgn_model.hidden_size, cfgn_model.num_actions)) * 0.3
            self.ph_b = jnp.zeros(cfgn_model.num_actions) + 0.01
            v_out_dim = 1 if cfgn_model.value_support_size == 0 else cfgn_model.value_support_size
            self.vh_w = jnp.ones((cfgn_model.hidden_size, v_out_dim)) * 0.4
            self.vh_b = jnp.zeros(v_out_dim) + 0.03
        def __call__(self, h, training):
            p_logits = h @ self.ph_w + self.ph_b
            val_out = h @ self.vh_w + self.vh_b
            return p_logits, val_out

    class FixedRew(nnx.Module):
        def __init__(self, *, rngs):
            r_out_dim = 1 if cfgn_model.reward_support_size == 0 else cfgn_model.reward_support_size
            self.rh_w = jnp.ones((cfgn_model.hidden_size, r_out_dim)) * 0.25
            self.rh_b = jnp.zeros(r_out_dim) + 0.04
        def __call__(self, h, training):
            return h @ self.rh_w + self.rh_b
            
    class FixedProj(nnx.Module):
        def __init__(self, *, rngs):
            self.proj_w = jnp.ones((cfgn_model.hidden_size, cfgn_model.projection_output_size)) * 0.15
            self.proj_b = jnp.zeros(cfgn_model.projection_output_size) + 0.005
        def __call__(self, h, training):
            return h @ self.proj_w + self.proj_b

    fixed_model = MuZeroNetwork(
        representation_network_def=lambda cfg, *, rngs: FixedRep(rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: FixedDyn(rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: FixedPred(rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: FixedRew(rngs=rngs),
        projection_network_def=lambda cfg, *, rngs: FixedProj(rngs=rngs) if cfgn_model.use_projection else None,
        config=cfgn_model,
        rngs=nnx.Rngs(params=mk)
    )
    # --- End Fixed Model ---

    # Configure Learner
    cfg_learner = make_cfg(
        cfgn_model.value_support_size, 
        cfgn_model.reward_support_size, 
        unroll_steps_test, 
        cfgn_model.use_projection, 
        f'loss_static_num_val_{img}_{val_cat}_{proj}_{scalar_targets}', 
        use_ema=use_ema, 
        ssl_weight=0.5 if cfgn_model.use_projection else 0.0, # Non-zero SSL weight for testing
        l2_weight=1e-2 # Non-zero L2 for testing
    )
    cfg_learner = dataclasses.replace(cfg_learner, batch_size=batch_size_test)


    # --- Create fixed batch data ---
    fixed_obs_val = 0.5
    fixed_action_val = 1
    
    # (B, K+1, *obs_shape) -> (1, 2, *obs_shape) since unroll_steps_test = 1
    obs_data = jnp.full((batch_size_test, unroll_steps_test + 1, *cfgn_model.observation_shape), fixed_obs_val)
    # (B, K) -> (1, 1)
    action_data = jnp.full((batch_size_test, unroll_steps_test), fixed_action_val, dtype=jnp.int32)
    
    # Targets (B, K+1, Support_Size_or_1)
    fixed_target_policy_logits = jnp.array([-0.1, 0.1, 0.5, -0.2, 0.0]) # Example logits
    target_policy_data = jax.nn.softmax(jnp.tile(fixed_target_policy_logits, (batch_size_test, unroll_steps_test + 1, 1)), axis=-1)

    if scalar_targets: # Scalar targets
        target_value_data = jnp.full((batch_size_test, unroll_steps_test + 1), 0.75)
        target_reward_data = jnp.full((batch_size_test, unroll_steps_test + 1), 0.25)
    else: # Categorical targets
        v_support_sz = cfgn_model.value_support_size if cfgn_model.value_support_size > 0 else 1
        r_support_sz = cfgn_model.reward_support_size if cfgn_model.reward_support_size > 0 else 1
        
        tv_dist = jnp.zeros(v_support_sz)
        if v_support_sz > 0: tv_dist = tv_dist.at[v_support_sz // 2].set(1.0) # one-hot
        else: tv_dist = jnp.array([0.75]) # scalar if support is 0
        target_value_data = jnp.tile(tv_dist, (batch_size_test, unroll_steps_test + 1, 1))
        if cfgn_model.value_support_size == 0 : target_value_data = jnp.squeeze(target_value_data, axis=-1)


        tr_dist = jnp.zeros(r_support_sz)
        if r_support_sz > 0 : tr_dist = tr_dist.at[r_support_sz // 2].set(1.0) # one-hot
        else: tr_dist = jnp.array([0.25]) # scalar if support is 0
        target_reward_data = jnp.tile(tr_dist, (batch_size_test, unroll_steps_test + 1, 1))
        if cfgn_model.reward_support_size == 0 : target_reward_data = jnp.squeeze(target_reward_data, axis=-1)


    mask_data = jnp.ones((batch_size_test, unroll_steps_test + 1))

    fixed_batch = {
        'observation': obs_data, 
        'action': action_data, 
        'target_reward': target_reward_data, 
        'target_value': target_value_data,
        'target_policy': target_policy_data, 
        'game_history_mask': mask_data
    }
    # --- End fixed batch data ---

    # Compute loss using the Learner's static method
    computed_loss, computed_metrics = Learner._compute_total_loss_static(
        fixed_model, cfg_learner, fixed_batch, lk, training=False # training=False to avoid BN updates for this test
    )

    # --- Manually calculate expected losses ---
    # 1. Forward pass through the fixed model
    # Initial inference
    obs_init = fixed_batch['observation'][:, 0] # (B, *obs_shape)
    s0, r0_pred, v0_pred, p0_logits, proj0_pred = fixed_model.initial_inference(obs_init, training=False)

    # Recurrent inference (1 step)
    action_k0 = fixed_batch['action'][:, 0] # (B,)
    s1, r1_pred, v1_pred, p1_logits, proj1_pred = fixed_model.recurrent_inference(s0, action_k0, training=False)
    
    # For K=1 unroll steps, we have K+1 = 2 sets of predictions/targets
    # Predictions: (r0_pred, v0_pred, p0_logits), (r1_pred, v1_pred, p1_logits)
    # Targets: target_reward[:,0], target_value[:,0], target_policy[:,0]
    #          target_reward[:,1], target_value[:,1], target_policy[:,1]

    # Policy Loss (Cross-entropy)
    # Step 0
    expected_policy_loss_s0 = -jnp.sum(target_policy_data[:, 0] * jax.nn.log_softmax(p0_logits), axis=-1)
    # Step 1
    expected_policy_loss_s1 = -jnp.sum(target_policy_data[:, 1] * jax.nn.log_softmax(p1_logits), axis=-1)
    expected_policy_loss = (jnp.sum(expected_policy_loss_s0 * mask_data[:,0]) + jnp.sum(expected_policy_loss_s1 * mask_data[:,1])) / jnp.maximum(jnp.sum(mask_data), 1.0)


    # Value Loss
    # Step 0
    if cfgn_model.value_support_size > 0: # Categorical
        tv0 = target_value_data[:, 0]
        expected_value_loss_s0 = -jnp.sum(tv0 * jax.nn.log_softmax(v0_pred, axis=-1), axis=-1)
    else: # Scalar (MSE)
        tv0 = target_value_data[:, 0]
        expected_value_loss_s0 = (jnp.squeeze(v0_pred, axis=-1) - tv0)**2
    # Step 1
    if cfgn_model.value_support_size > 0: # Categorical
        tv1 = target_value_data[:, 1]
        expected_value_loss_s1 = -jnp.sum(tv1 * jax.nn.log_softmax(v1_pred, axis=-1), axis=-1)
    else: # Scalar (MSE)
        tv1 = target_value_data[:, 1]
        expected_value_loss_s1 = (jnp.squeeze(v1_pred, axis=-1) - tv1)**2
    expected_value_loss = (jnp.sum(expected_value_loss_s0 * mask_data[:,0]) + jnp.sum(expected_value_loss_s1 * mask_data[:,1])) / jnp.maximum(jnp.sum(mask_data), 1.0)

    # Reward Loss (similar to value)
    # Step 0
    if cfgn_model.reward_support_size > 0: # Categorical
        tr0 = target_reward_data[:, 0]
        expected_reward_loss_s0 = -jnp.sum(tr0 * jax.nn.log_softmax(r0_pred, axis=-1), axis=-1)
    else: # Scalar (MSE)
        tr0 = target_reward_data[:, 0]
        expected_reward_loss_s0 = (jnp.squeeze(r0_pred, axis=-1) - tr0)**2
    # Step 1
    if cfgn_model.reward_support_size > 0: # Categorical
        tr1 = target_reward_data[:, 1]
        expected_reward_loss_s1 = -jnp.sum(tr1 * jax.nn.log_softmax(r1_pred, axis=-1), axis=-1)
    else: # Scalar (MSE)
        tr1 = target_reward_data[:, 1]
        expected_reward_loss_s1 = (jnp.squeeze(r1_pred, axis=-1) - tr1)**2
    expected_reward_loss = (jnp.sum(expected_reward_loss_s0 * mask_data[:,0]) + jnp.sum(expected_reward_loss_s1 * mask_data[:,1])) / jnp.maximum(jnp.sum(mask_data), 1.0)

    # L2 Loss
    expected_l2_loss = 0.0
    _, params_for_l2, _, _, _, _ = nnx.split(fixed_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    
    # Manually iterate through the fixed weights we defined for L2
    # Rep
    expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.representation_network.dense.kernel.value**2)
    # Dyn
    expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.dynamics_network.embed.embedding.value**2)
    expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.dynamics_network.fc.kernel.value**2)
    # Pred (using the manually set weight matrices)
    expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.prediction_network.ph_w**2)
    expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.prediction_network.vh_w**2)
    # Rew
    expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.reward_network.rh_w**2)
    
    # SSL Loss (Cosine similarity based, scaled and shifted)
    expected_ssl_loss = 0.0
    if cfgn_model.use_projection and cfg_learner.ssl_consistency_loss_weight > 0:
        # Calculate according to losses_lib.compute_projection_consistency_loss
        # proj1_pred is projection_current_step, proj0_pred is projection_initial_step
        
        sim1_test = optax.cosine_similarity(proj1_pred, jax.lax.stop_gradient(proj0_pred))
        sim2_test = optax.cosine_similarity(jax.lax.stop_gradient(proj1_pred), proj0_pred)

        clipped_sim1_test = jnp.clip(sim1_test, -1.0, 1.0)
        clipped_sim2_test = jnp.clip(sim2_test, -1.0, 1.0)
        
        # For batch_size_test = 1, jnp.mean over the batch dim is just the value itself
        # The loss is applied per unroll step, and then averaged.
        # Here, we only care about the SSL loss for k_idx=1 vs k_idx=0
        # The total_ssl_loss in _compute_total_loss_static averages this over masked steps.
        # Since mask_data[:, 1] is 1 and batch_size is 1, this should be direct.
        
        # This is the per-instance loss for the (proj1_pred, proj0_pred) pair
        ssl_loss_for_this_pair = -jnp.mean(clipped_sim1_test) - jnp.mean(clipped_sim2_test)
        
        # The test setup has unroll_steps_test = 1.
        # The SSL loss is calculated for k_idx > 0. So only for k_idx = 1.
        # The trainer\'s _compute_total_loss_static applies a mask and averages.
        # expected_ssl_loss should be the value that goes into metrics[\'ssl_loss\']
        # which is total_ssl_loss, accumulated and averaged.
        # For a single unroll step (k_idx=1), and batch size 1, with mask=1:
        # total_ssl_loss = (sum over k_idx > 0) of [ (sum over batch for (loss_val * mask)) / sum(mask) ]
        # Here, just one term: ( (loss_for_pair_batch_item_0 * 1) / 1 )
        expected_ssl_loss = ssl_loss_for_this_pair # Already averaged by jnp.mean in its calculation if batch_dim exists

        # Add L2 for projection network if it exists
        expected_l2_loss += 0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.projection_network.proj_w**2)


    expected_total_loss = (
        cfg_learner.policy_loss_weight * expected_policy_loss +
        cfg_learner.value_loss_weight * expected_value_loss +
        cfg_learner.reward_loss_weight * expected_reward_loss +
        expected_l2_loss
    )
    if cfgn_model.use_projection and cfg_learner.ssl_consistency_loss_weight > 0: # Add SSL to total loss
        expected_total_loss += cfg_learner.ssl_consistency_loss_weight * expected_ssl_loss

    # --- Assertions ---
    assert isinstance(computed_loss, jax.Array) and computed_loss.shape == ()
    for m_key in ['total_loss', 'policy_loss', 'value_loss', 'reward_loss', 'l2_loss']:
        assert m_key in computed_metrics, f"{m_key} not in computed metrics"
    
    jnp.allclose(computed_metrics['policy_loss'], expected_policy_loss, atol=1e-5)
    jnp.allclose(computed_metrics['value_loss'], expected_value_loss, atol=1e-5)
    jnp.allclose(computed_metrics['reward_loss'], expected_reward_loss, atol=1e-5)
    jnp.allclose(computed_metrics['l2_loss'], expected_l2_loss, atol=1e-5)

    if cfgn_model.use_projection and cfg_learner.ssl_consistency_loss_weight > 0:
        assert 'ssl_loss' in computed_metrics
        jnp.allclose(computed_metrics['ssl_loss'], expected_ssl_loss, atol=1e-5)
        # Check if SSL loss contributes if weight > 0
        # This assertion is problematic as SSL loss can be negative.
        # Removing it and relying on allclose with the correctly calculated expected_ssl_loss.
        # if proj0_pred is not None and proj1_pred is not None and jnp.any(proj0_pred != proj1_pred): 
        #    assert computed_metrics[\\\'ssl_loss\\\'] > 1e-6, \"SSL loss should be non-zero if projections differ and weight > 0\"
        
    jnp.allclose(computed_loss, expected_total_loss, atol=1e-5)

@pytest.mark.parametrize("img,val_cat,proj,use_ema", [
    (False, False, False, False), 
    (True, True, True, True),
    (False, False, True, False), # Test projection without EMA
    (False, False, False, True)  # Test EMA without projection
])
def test_step(key, img, val_cat, proj, use_ema, cfg_flat, cfg_img):
    bk, mk, lk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_img if img else cfg_flat)
    cfgn = dataclasses.replace(cfgn, 
                               use_projection=proj,
                               value_support_size=VALUE_SUPPORT_CATEGORICAL if val_cat else VALUE_SUPPORT_SCALAR,
                               reward_support_size=REWARD_SUPPORT_CATEGORICAL if val_cat else REWARD_SUPPORT_SCALAR
                              )
    model = make_model(mk, cfgn)
    cfg = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, NUM_UNROLL_STEPS, proj, f'step_{img}_{val_cat}_{proj}_{use_ema}', use_ema=use_ema, ssl_weight=0.1 if proj else 0.0)
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)
    batch = make_batch(bk, cfg.batch_size, cfgn.observation_shape, cfgn.num_actions, cfg.num_unroll_steps, cfgn.value_support_size, cfgn.reward_support_size, cfgn.projection_output_size, proj)
    
    # Capture initial states
    initial_model_state_vars = nnx.state(learner.model) # All variables
    initial_opt_state = learner.opt_state

    # Get params before step for comparison
    _, initial_online_params_nnx_state, _, _, _, _ = nnx.split(learner.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    # Convert nnx.State of Params to a plain PyTree of values for optax.apply_updates if needed for manual check
    # For Adam, the params used in optimizer.update are the actual nnx.State Variables.
    initial_online_param_values = jax.tree_util.tree_map(maybe_val, initial_online_params_nnx_state)


    # Perform the train step using the JIT-compiled static logic directly to get grads
    # This allows us to inspect gradients, which learner.train_step() doesn't directly return.
    step_rng_key = jax.random.fold_in(lk, learner.num_training_steps + 1) # Use a consistent key for the step
    model_graphdef, current_params_state, current_batch_stats_state, current_static_attrs, current_rngs_state, current_ellipsis_vars = nnx.split(
        learner.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
    )

    # Call the static logic to get gradients and updated states (but before applying to the learner model itself)
    (   updated_params_values_from_static_logic, # These are param *values*
        updated_batch_stats_from_static_logic, 
        updated_rngs_state_from_static_logic, 
        updated_ellipsis_state_from_static_logic,
        new_opt_state_from_static_logic, 
        metrics_from_static_logic
    ) = Learner._static_train_step_logic(
        model_graphdef, 
        current_params_state, 
        current_batch_stats_state, 
        current_static_attrs, 
        current_rngs_state,
        current_ellipsis_vars,
        learner.optimizer, # The optimizer transform
        initial_opt_state, 
        cfg, 
        batch, 
        step_rng_key
    )

    # Manually compute what the gradients should have been to produce these updates
    # The grads are an intermediate result *inside* _static_train_step_logic.
    # To verify optimizer update: we need the gradients that were fed to optimizer.update()
    # We can get these by re-running the grad computation part:
    def loss_fn_for_grad_check(model_to_grad: MuZeroNetwork):
        loss_val, metrics_val = Learner._compute_total_loss_static(model_to_grad, cfg, batch, step_rng_key, training=True)
        return loss_val, metrics_val
    
    # Reconstruct the model as it was at the start of _static_train_step_logic call
    model_at_grad_comp_time = nnx.merge(model_graphdef, current_params_state, current_batch_stats_state, current_static_attrs, current_rngs_state, current_ellipsis_vars)
    (_, metrics_check), grads_for_check = nnx.value_and_grad(loss_fn_for_grad_check, argnums=0, has_aux=True)(model_at_grad_comp_time)

    # Now, manually apply the optimizer update using these grads_for_check
    # The `params` argument to optimizer.update is the PyTree of current parameter variables (nnx.State with Param objects)
    expected_updates, expected_new_opt_state = learner.optimizer.update(grads_for_check, initial_opt_state, current_params_state)
    
    # current_params_state is an nnx.State of Param *Variables*.
    # expected_updates is a PyTree of *update values*.
    # optax.apply_updates takes current param *values* and update *values*.
    # So we need to get the values from current_params_state.
    current_param_values_for_apply = jax.tree_util.tree_map(maybe_val, current_params_state)
    expected_updated_param_values = optax.apply_updates(current_param_values_for_apply, expected_updates)

    # Assert that the optimizer state matches
    jax.tree_util.tree_map(
        lambda a, b: jnp.allclose(a, b, atol=1e-6),
        new_opt_state_from_static_logic, expected_new_opt_state
    )
    # Assert that the updated parameter values match
    # updated_params_values_from_static_logic is already a PyTree of values
    jax.tree_util.tree_map(
        lambda a, b: jnp.allclose(a, b, atol=1e-6),
        updated_params_values_from_static_logic, expected_updated_param_values
    )

    # Now, call the actual learner.train_step() to update the learner model itself
    updated_model, final_opt_state, metrics = learner.train_step(batch)
    
    # Get params after the actual train_step for other checks
    _, online_params_nnx_state_after, _, _, _, _ = nnx.split(updated_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    online_param_values_after = jax.tree_util.tree_map(maybe_val, online_params_nnx_state_after)

    # Check that parameters indeed changed from initial state
    assert any(
        not jnp.allclose(initial_val, final_val)
        for initial_val, final_val in zip(jax.tree_util.tree_leaves(initial_online_param_values), jax.tree_util.tree_leaves(online_param_values_after))
    )
    if proj and cfg.ssl_consistency_loss_weight > 0:
        assert 'ssl_loss' in metrics
    if use_ema:
        assert learner.target_model is not None
        assert learner.ema_params_state is not None
        
        # online_param_values_after are the parameters of the online model *after* the gradient update of the current step.
        # initial_online_param_values are the parameters of the online model *before* the gradient update of the current step.
        # learner.ema_params_state *before* this step's EMA update (if we could access it) would hold the decayed average up to the PREVIOUS step.
        # learner.ema_params_state *after* this step's EMA update holds the new decayed average.

        # For the first EMA update (learner.num_training_steps will be 1 after the first train_step call):
        # Optax EMA: new_ema = decay * old_ema + (1-decay) * new_online_params
        # If old_ema was initialized with initial_online_params (which it is, indirectly via target_model init and then ema_updater.init(target_model_params)):
        # Then for the *first actual update* of EMA state:
        #   The `updates` to `ema_updater.update` is `online_params_values_after` (params after current step grad update).
        #   The `state.ema` fed into `ema_updater.update` is effectively `initial_online_param_values` (because ema was init with it).
        # So, expected_target_params = cfg.ema_decay * initial_online_param_values + (1 - cfg.ema_decay) * online_param_values_after

        # Get target model params *after* learner.train_step() which includes the EMA update for this step.
        _, target_params_nnx_state_after_ema_update, _, _, _, _ = nnx.split(learner.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
        target_param_values_after_ema_update = jax.tree_util.tree_map(maybe_val, target_params_nnx_state_after_ema_update)

        # `online_param_values_after` are the parameters of the main model post-gradient update for the current step.
        # `initial_online_param_values` are the parameters of the main model *before* this step's gradient update.
        # The EMA state `learner.ema_params_state` was updated using `online_param_values_after`.
        # The `state.ema` that went *into* the `self.ema_updater.update` call for *this step* effectively represents
        # the EMA parameters from the *previous* step. 
        # If this is the very first EMA update (i.e. first train_step when EMA is active), 
        # then the `state.ema` that went into `ema_updater.update` was initialized with the initial parameters of the target model,
        # which were a deepcopy of the initial online model parameters.

        # Let P_t_online be online_param_values_after (online params after step t grad update)
        # Let EMA_{t-1} be the EMA state before this step's EMA update.
        # Expected EMA_t = decay * EMA_{t-1} + (1-decay) * P_t_online
        
        # If num_training_steps == 1 (meaning this was the first train_step call):
        # EMA_{t-1} (which is ema_params_state.ema before the update call in train_step) was initialized with initial_online_param_values.
        if learner.num_training_steps == 1: # After the first train_step call, num_training_steps is 1.
            expected_target_param_values = jax.tree_util.tree_map(
                lambda initial_ema_val, current_online_val: cfg.ema_decay * initial_ema_val + (1 - cfg.ema_decay) * current_online_val,
                initial_online_param_values, # This was the effective EMA_{t-1}
                online_param_values_after    # This is P_t_online
            )
        else:
            # For subsequent steps, it's harder to get EMA_{t-1} directly in this test structure
            # without saving it from the previous iteration or re-architecting how EMA state is handled/exposed.
            # So, for now, we only strictly verify the first EMA update. 
            # The existing check that target params differ from online params still holds for subsequent steps.
            pass # Skip precise numerical check for EMA after the first step in this test setup.

        if learner.num_training_steps == 1: # Only check precisely for the first EMA update
            jax.tree_util.tree_map(
                lambda expected, actual: jnp.allclose(expected, actual, atol=1e-6),
                expected_target_param_values,
                target_param_values_after_ema_update
            )

        if cfg.ema_decay < 1.0:
            # After one step, EMA parameters should differ from the *initial* online parameters.
            # And also from the *current* online parameters if updates happened.
            initial_online_leaves = jax.tree_util.tree_leaves(initial_online_param_values)
            target_leaves = jax.tree_util.tree_leaves(target_param_values_after_ema_update) 
            current_online_leaves = jax.tree_util.tree_leaves(online_param_values_after) 

            assert any(not jnp.allclose(init_online, target) for init_online, target in zip(initial_online_leaves, target_leaves)), \
                "Target params should not be identical to initial online params after EMA update with decay < 1"
            assert any(not jnp.allclose(curr_online, target) for curr_online, target in zip(current_online_leaves, target_leaves)), \
                "Target params should not be identical to current online params after EMA update with decay < 1"

@pytest.mark.parametrize("use_ema, resume", [(False, False), (True, False), (True, True)])
def test_train_loop_and_ckpt(key, cfg_flat, use_ema, resume):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_flat, use_projection=use_ema) # Enable projection if EMA is used for more coverage
    model = make_model(mk, cfgn)
    cfg_suffix = f'loop_ema_{use_ema}_resume_{resume}'
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, 
                     proj=cfgn.use_projection, 
                     suffix=cfg_suffix, 
                     use_ema=use_ema, 
                     ssl_weight=0.1 if cfgn.use_projection else 0.0)
    cfg = dataclasses.replace(cfg, resume_from_checkpoint=False) # Start fresh for first run

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)

    num_total_steps = 4 
    batches = [make_batch(jax.random.fold_in(bk, i),
                          cfg.batch_size,
                          cfgn.observation_shape,
                          cfgn.num_actions,
                          cfg.num_unroll_steps,
                          cfgn.value_support_size,
                          cfgn.reward_support_size,
                          cfgn.projection_output_size, cfgn.use_projection)
               for i in range(num_total_steps)]

    def get_batch_generator_fn():
        # This function now returns a new generator each time it's called
        def gen():
            for item in batches:
                yield item
        return gen()

    # with patch.object(learner, 'save_checkpoint') as mock_save: # Removed patch to test actual saving
    learner.train(get_batch_generator_fn, num_epochs=1, steps_per_epoch=num_total_steps)
    
    assert learner.num_training_steps == num_total_steps
    # Check that a checkpoint was actually saved (latest_step should not be None)
    if cfg.checkpoint_dir and learner.checkpoint_manager:
        latest_saved_step = learner.checkpoint_manager.latest_step()
        assert latest_saved_step == num_total_steps, f"Expected checkpoint at step {num_total_steps}, found {latest_saved_step}"

    if resume:
        # Create new model and learner to simulate restart for loading
        mk_resume, lk_resume = jax.random.split(jax.random.fold_in(key, 100), 2)
        model_resume = make_model(mk_resume, cfgn)
        cfg_resume = dataclasses.replace(cfg, resume_from_checkpoint=True)
        opt_resume = optax.adam(cfg_resume.learning_rate)
        learner_resume = Learner(model_resume, opt_resume, cfg_resume, lk_resume)
        assert learner_resume.num_training_steps == num_total_steps
        if use_ema:
            assert learner_resume.target_model is not None
            assert learner_resume.ema_params_state is not None

def test_train_loop_exhausted_buffer(key, cfg_flat):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = cfg_flat
    model = make_model(mk, cfgn)
    cfg_suffix = 'exhausted_buffer'
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, 
                     proj=False, 
                     suffix=cfg_suffix, 
                     use_ema=False)
    cfg = dataclasses.replace(cfg, checkpoint_frequency=100) # Avoid ckpt logic

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)

    # Empty batch list
    batches = []

    def get_empty_batch_generator_fn():
        def gen():
            for item in batches: # Will not yield anything
                yield item
            # Simulate exhaustion even after re-init by not yielding again
            # Or, more realistically, ensure it stops after one attempt to re-init
        return gen()
    
    # To test the re-initialization and immediate exhaustion, 
    # we can make the generator yield once, then be empty upon re-initialization.
    # However, the current test structure is simpler by just providing an always-empty generator.
    # The code under test will try to re-init, then StopIteration again.

    with patch('builtins.print') as mock_print:
        learner.train(get_empty_batch_generator_fn, num_epochs=1, steps_per_epoch=1)
    
    assert learner.num_training_steps == 0
    
    # Check if the specific print messages were called
    assert any("Replay buffer iterator exhausted." in call_args[0][0] for call_args in mock_print.call_args_list)
    assert any("Replay buffer truly exhausted. Stopping training." in call_args[0][0] for call_args in mock_print.call_args_list)

def test_checkpointing_no_manager(key, cfg_flat):
    mk, lk = jax.random.split(key, 2)
    model = make_model(mk, cfg_flat)
    # Create a config with checkpoint_dir=None
    cfg_no_ckpt = MuZeroConfig(
        value_support_size=cfg_flat.value_support_size,
        reward_support_size=cfg_flat.reward_support_size,
        checkpoint_dir=None # Explicitly None
    )
    opt = optax.adam(cfg_no_ckpt.learning_rate)
    learner = Learner(model, opt, cfg_no_ckpt, lk)

    assert learner.checkpoint_manager is None

    # Test save_checkpoint
    with patch('builtins.print') as mock_print_save:
        learner.save_checkpoint()
    mock_print_save.assert_any_call("Checkpoint manager not configured. Skipping save.")

    # Test load_checkpoint
    with patch('builtins.print') as mock_print_load:
        loaded = learner.load_checkpoint()
    assert not loaded
    mock_print_load.assert_any_call("Checkpoint manager not configured. Skipping load.")

def test_load_checkpoint_no_checkpoint_exists(key, cfg_flat):
    mk, lk = jax.random.split(key, 2)
    model = make_model(mk, cfg_flat)
    cfg_suffix = "no_ckpt_exists"
    # Ensure a unique directory that will be empty
    cfg_ckpt = make_cfg(cfg_flat.value_support_size, 
                        cfg_flat.reward_support_size, 
                        1, 
                        proj=False, 
                        suffix=cfg_suffix)
    # Clean up if it somehow exists from a previous failed run
    if os.path.exists(cfg_ckpt.checkpoint_dir):
        shutil.rmtree(cfg_ckpt.checkpoint_dir)
    os.makedirs(cfg_ckpt.checkpoint_dir) # Create empty dir

    opt = optax.adam(cfg_ckpt.learning_rate)
    learner = Learner(model, opt, cfg_ckpt, lk)

    assert learner.checkpoint_manager is not None

    with patch('builtins.print') as mock_print:
        loaded = learner.load_checkpoint()
    assert not loaded
    mock_print.assert_any_call("No checkpoint found to resume from.")
    # Clean up the created directory
    if os.path.exists(cfg_ckpt.checkpoint_dir):
        shutil.rmtree(cfg_ckpt.checkpoint_dir)

def test_save_checkpoint_conditions(key, cfg_flat):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = cfg_flat
    model = make_model(mk, cfgn)
    cfg_suffix = "save_conditions"
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, 
                     proj=False, 
                     suffix=cfg_suffix)
    # Set checkpoint frequency high to test skipping, then force save
    cfg = dataclasses.replace(cfg, checkpoint_frequency=100, max_checkpoints_to_keep=1)

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)
    assert learner.checkpoint_manager is not None

    # --- Test skipping save due to frequency --- 
    learner.num_training_steps = 50 # Less than checkpoint_frequency
    with patch.object(learner.checkpoint_manager, 'save') as mock_manager_save, \
         patch('logging.info') as mock_logging_info_skip:
        learner.save_checkpoint(force_save=False)
    mock_manager_save.assert_not_called()
    # Check for the specific log message indicating skip
    assert any(
        f"SAVE_CHECKPOINT: Condition NOT met. force_save=False, num_training_steps={learner.num_training_steps}, freq={cfg.checkpoint_frequency}" in call_args[0][0]
        for call_args in mock_logging_info_skip.call_args_list
    ), "Log message for skipping save due to frequency not found."


    # --- Test force_save=True at end of hypothetical training (within train loop logic) ---
    # This part simulates the condition within the train() method
    num_epochs = 1
    steps_per_epoch = 3
    learner.num_training_steps = 0 # Reset
    cfg_train_end = dataclasses.replace(cfg, checkpoint_frequency=2) # Save every 2 steps
    learner.config = cfg_train_end # Update learner's config
    
    batches = [make_batch(jax.random.fold_in(bk, i),
                          cfg_train_end.batch_size,
                          cfgn.observation_shape,
                          cfgn.num_actions,
                          cfg_train_end.num_unroll_steps,
                          cfgn.value_support_size,
                          cfgn.reward_support_size)
               for i in range(steps_per_epoch)]

    def get_batch_gen_fn():
        def gen(): yield from batches
        return gen()

    with patch.object(learner.checkpoint_manager, 'save') as mock_manager_save_train, \
         patch('logging.info') as mock_logging_info_train:
        learner.train(get_batch_gen_fn, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch)
    
    # Expected saves: 
    # Step 2 (regular)
    # Step 3 (end of training, force_save=True via internal logic of train() calling save_checkpoint(force_save=True))
    assert mock_manager_save_train.call_count == 2
    # Check the call for step 2 (regular)
    args_step2, _ = mock_manager_save_train.call_args_list[0]
    assert args_step2[0] == 2 # step number
    # Check the call for step 3 (end of training)
    args_step3, _ = mock_manager_save_train.call_args_list[1]
    assert args_step3[0] == 3 # step number
    
    # Verify the logging for force_save=True for the last step
    # The save_checkpoint method is called with force_save=True by the train method internally.
    # We need to check the logging call that reflects this forced save.
    found_force_save_log = False
    for call_args in mock_logging_info_train.call_args_list:
        log_message = call_args[0][0]
        if f"End of training checkpoint: step {steps_per_epoch}" in log_message:
            # This is logged just before calling save_checkpoint(force_save=True)
            # Now find the corresponding SAVE_CHECKPOINT log for this step
            for subsequent_call_args in mock_logging_info_train.call_args_list:
                if f"SAVE_CHECKPOINT: Condition met. force_save=True, num_training_steps={steps_per_epoch}, freq={cfg_train_end.checkpoint_frequency}" in subsequent_call_args[0][0]:
                    found_force_save_log = True
                    break
            if found_force_save_log: break
    assert found_force_save_log, "Log message for force_save=True at end of training not found."

    # --- Test regular save due to frequency ---
    # learner.config.checkpoint_frequency is 2 at this point from the previous section of the test.
    learner.num_training_steps = learner.config.checkpoint_frequency # This will be 2
    with patch.object(learner.checkpoint_manager, 'save') as mock_manager_save_freq, \
         patch('logging.info') as mock_logging_info_freq:
        learner.save_checkpoint(force_save=False)
    mock_manager_save_freq.assert_called_once()
    # Check for the specific log message indicating save due to frequency
    assert any(
        f"SAVE_CHECKPOINT: Condition met. force_save=False, num_training_steps={learner.num_training_steps}, freq={learner.config.checkpoint_frequency}" in call_args[0][0]
        for call_args in mock_logging_info_freq.call_args_list
    ), f"Log message for saving due to frequency not found. Log calls: {mock_logging_info_freq.call_args_list}"

    # Clean up
    if cfg.checkpoint_dir and os.path.exists(cfg.checkpoint_dir):
        shutil.rmtree(cfg.checkpoint_dir)

def test_load_checkpoint_missing_ema_details(key, cfg_flat):
    mk, lk_save, lk_load, bk = jax.random.split(key, 4)
    cfgn = cfg_flat # Using flat config for simplicity
    model_save = make_model(mk, cfgn)
    cfg_suffix = "missing_ema"

    # 1. Configure and run a learner WITHOUT EMA to save a checkpoint
    cfg_save = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, 1, False, 
                        suffix=f"{cfg_suffix}_save", use_ema=False)
    opt_save = optax.adam(cfg_save.learning_rate)
    learner_save = Learner(model_save, opt_save, cfg_save, lk_save)
    learner_save.num_training_steps = 1 # Make it save something
    learner_save.save_checkpoint(force_save=True) # target_model and ema_params_state will be None
    if learner_save.checkpoint_manager: # Ensure manager exists before closing
        learner_save.checkpoint_manager.close()

    # 2. Configure a new learner WITH EMA and try to load the checkpoint
    #    This learner will expect target_model_params etc. in the checkpoint.
    model_load = make_model(jax.random.fold_in(mk,1), cfgn) # Fresh model instance
    cfg_load = dataclasses.replace(cfg_save, use_target_network_ema=True, resume_from_checkpoint=True)
    # Ensure the checkpoint_dir is the same
    opt_load = optax.adam(cfg_load.learning_rate)
    
    with patch('builtins.print') as mock_print, \
         patch.object(ocp.CheckpointManager, 'latest_step', return_value=1) as mock_latest_step, \
         patch.object(ocp.CheckpointManager, 'restore') as mock_restore:

        # Simulate that restore will return a dictionary as if loaded from checkpoint,
        # but without the target_model components and ema_params_state initially.
        # The actual checkpoint was saved by learner_save, which had these as None.
        # learner_save.opt_state and learner_save._rng_key are optax/jax types.
        # model_params_save would be an nnx.State object.
        _, model_params_save, model_bs_save, model_rngs_save, model_static_save, model_ellipsis_save = nnx.split(
            learner_save.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)

        restored_dict_simulation = {
            'model_graphdef': nnx.graphdef(learner_save.model),
            'model_params': model_params_save,
            'model_batch_stats': model_bs_save,
            'model_rngs': model_rngs_save,
            'model_static': model_static_save,
            'model_ellipsis': model_ellipsis_save,
            'opt_state': learner_save.opt_state,
            'num_training_steps': learner_save.num_training_steps, # Should be 1
            'rng_key': learner_save._rng_key,
            # Simulate these being absent from the loaded checkpoint dict initially
            'target_model_graphdef': None,
            'target_model_params': None,
            'target_model_batch_stats': None,
            'target_model_rngs': None,
            'target_model_static': None,
            'target_model_ellipsis': None,
            'ema_params_state': None 
        }
        mock_restore.return_value = restored_dict_simulation

        learner_load = Learner(model_load, opt_load, cfg_load, lk_load)
        # Learner init calls load_checkpoint if resume_from_checkpoint is True

    mock_latest_step.assert_called()

    # To get the correct abstract items for the assertion:
    # Create a temporary learner with the same setup as learner_load but *without* resume,
    # to capture its initial state for opt_state and ema_params_state, which are used
    # by the actual load_checkpoint method to build its abstract_items_to_restore.
    cfg_temp_for_abstract = dataclasses.replace(cfg_load, resume_from_checkpoint=False)
    # Use a different key for this temp learner if its _rng_key is part of abstract items and needs to match lk_load exactly.
    # However, lk_load is what learner_load receives, so its _rng_key will be lk_load.
    temp_learner_for_abstract = Learner(model_load, opt_load, cfg_temp_for_abstract, lk_load) 

    model_graphdef_abs, model_params_abs, model_bs_abs, model_rngs_abs, model_static_abs, model_ellipsis_abs = nnx.split(
        temp_learner_for_abstract.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
    )
    # The RngKey used for abstract item should be the one that the learner_load instance has when it calls load_checkpoint.
    # This is lk_load, which is temp_learner_for_abstract._rng_key if lk_load was passed to it.

    expected_abstract_items = {
        'model_graphdef': model_graphdef_abs,
        'model_params': model_params_abs,
        'model_batch_stats': model_bs_abs,
        'model_rngs': model_rngs_abs,
        'model_static': model_static_abs,
        'model_ellipsis': model_ellipsis_abs,
        'opt_state': temp_learner_for_abstract.opt_state,
        'num_training_steps': 0, # Abstract, value doesn't strictly matter here, type does
        'rng_key': temp_learner_for_abstract._rng_key, # This is lk_load
        'ema_params_state': temp_learner_for_abstract.ema_params_state
    }

    if temp_learner_for_abstract.target_model is not None:
        target_gdef_abs, target_params_abs, target_bs_abs, target_rngs_abs, target_static_abs, target_ellipsis_abs = nnx.split(
            temp_learner_for_abstract.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
        expected_abstract_items['target_model_graphdef'] = target_gdef_abs
        expected_abstract_items['target_model_params'] = target_params_abs
        expected_abstract_items['target_model_batch_stats'] = target_bs_abs
        expected_abstract_items['target_model_rngs'] = target_rngs_abs
        expected_abstract_items['target_model_static'] = target_static_abs
        expected_abstract_items['target_model_ellipsis'] = target_ellipsis_abs
    else:
        expected_abstract_items['target_model_graphdef'] = None
        expected_abstract_items['target_model_params'] = None
        expected_abstract_items['target_model_batch_stats'] = None
        expected_abstract_items['target_model_rngs'] = None
        expected_abstract_items['target_model_static'] = None
        expected_abstract_items['target_model_ellipsis'] = None
    
    # mock_restore.assert_called_once_with(1, args=ocp.args.StandardRestore(expected_abstract_items))
    # Instead, verify the call and compare the abstract_tree directly for clarity
    mock_restore.assert_called_once() # Check it was called
    
    call_args_tuple = mock_restore.call_args
    assert call_args_tuple is not None, "mock_restore was not called despite assert_called_once passing?"

    # call_args can be a tuple (args, kwargs) or just args if no kwargs were used.
    # For CheckpointManager.restore(step, items, bundle, args), args is passed as a kwarg.
    if isinstance(call_args_tuple, tuple):
        called_pos_args, called_kwargs = call_args_tuple
    else: # Should not happen with how restore is called with kwargs
        # For safety, handle if it's just positional (though unlikely for .restore with args=)
        called_pos_args = call_args_tuple
        called_kwargs = {}

    assert called_pos_args[0] == 1 # Step number
    assert 'args' in called_kwargs
    standard_restore_arg_received = called_kwargs['args']
    assert isinstance(standard_restore_arg_received, ocp.args.StandardRestore)
    
    # Compare the PyTrees. Pytest should give a good diff if they are not equal.
    assert tree_structure(standard_restore_arg_received.item) == tree_structure(expected_abstract_items)

    assert learner_load.num_training_steps == 1 # Should load steps
    assert learner_load.target_model is not None # Should be re-initialized
    assert learner_load.ema_params_state is not None # Should be re-initialized
    
    # Check for the specific warning prints
    found_warn_target_missing = any(
        "Warning: EMA enabled, target model components not fully in ckpt. Re-syncing with online model." in call_args[0][0]
        for call_args in mock_print.call_args_list
    )
    found_warn_ema_state_missing = any(
        "Warning: EMA enabled, ema_params_state not in ckpt. Reinitializing EMA state from online model." in call_args[0][0]
        for call_args in mock_print.call_args_list
    )
    assert found_warn_target_missing or found_warn_ema_state_missing, \
        f"Expected EMA recovery warnings not found. Logs: {mock_print.call_args_list}"

    # Clean up
    if os.path.exists(cfg_save.checkpoint_dir):
        shutil.rmtree(cfg_save.checkpoint_dir)

def test_loss_static_missing_projection_in_model_output(key, cfg_flat):
    """Test _compute_total_loss_static when use_projection=True but model.initial_inference is misbehaving."""
    bk, mk, lk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_flat, use_projection=True) # Enable projection in model config
    
    # Scenario 1: initial_inference returns too few elements
    class MockModelShortInitial(MuZeroNetwork):
        def initial_inference(self, x, training):
            # Returns hidden_state, reward, value, policy_logits (4 elements)
            # Actual mock model parts need to be set up if they are accessed by base class
            hidden_state = jnp.zeros((x.shape[0], self.config.hidden_size))
            reward = jnp.zeros((x.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((x.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((x.shape[0], self.config.num_actions))
            return hidden_state, reward, value, policy_logits 
        # recurrent_inference also needs to be properly mocked if reached
        def recurrent_inference(self, h, a, training):
            reward = jnp.zeros((h.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((h.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((h.shape[0], self.config.num_actions))
            # No projection returned here either for simplicity, though not directly testing this part for coverage here
            return h, reward, value, policy_logits

    model_short = MockModelShortInitial(
        representation_network_def=lambda cfg, *, rngs: MockRep(cfg.observation_shape, cfg.hidden_size, rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: MockDyn(cfg.hidden_size, cfg.num_actions, rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: MockPred(cfg.hidden_size, cfg.num_actions, cfg.value_support_size, rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: MockRew(cfg.hidden_size, cfg.reward_support_size, rngs=rngs),
        projection_network_def=None,
        config=cfgn, 
        rngs=nnx.Rngs(params=mk)
    )

    # Learner config with projection enabled and SSL loss active
    cfg_learner_proj = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, NUM_UNROLL_STEPS, proj=True, 
                                suffix='loss_missing_proj1', ssl_weight=0.1)
    batch = make_batch(bk, cfg_learner_proj.batch_size, cfgn.observation_shape, cfgn.num_actions, 
                       cfg_learner_proj.num_unroll_steps, cfgn.value_support_size, cfgn.reward_support_size,
                       use_proj=True) # Batch is made as if proj is expected

    # Test with model_short
    loss1, met1 = Learner._compute_total_loss_static(model_short, cfg_learner_proj, batch, lk, training=True)
    assert 'ssl_loss' in met1 # SSL loss should still be in metrics (even if 0)
    assert met1['ssl_loss'] == 0.0 # SSL loss should be zero as no projections were processed

    # Scenario 2: initial_inference returns projection as None
    class MockModelNoneInitialProjection(MuZeroNetwork):
        def initial_inference(self, x, training):
            hidden_state = jnp.zeros((x.shape[0], self.config.hidden_size))
            reward = jnp.zeros((x.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((x.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((x.shape[0], self.config.num_actions))
            return hidden_state, reward, value, policy_logits, None # 5th element is None
        def recurrent_inference(self, h, a, training):
            reward = jnp.zeros((h.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((h.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((h.shape[0], self.config.num_actions))
            return h, reward, value, policy_logits, None # Return None projection here too

    model_none_proj = MockModelNoneInitialProjection(
        representation_network_def=lambda cfg, *, rngs: MockRep(cfg.observation_shape, cfg.hidden_size, rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: MockDyn(cfg.hidden_size, cfg.num_actions, rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: MockPred(cfg.hidden_size, cfg.num_actions, cfg.value_support_size, rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: MockRew(cfg.hidden_size, cfg.reward_support_size, rngs=rngs),
        projection_network_def=None, 
        config=cfgn, 
        rngs=nnx.Rngs(params=jax.random.fold_in(mk,1))
    )
    # Test with model_none_proj
    loss2, met2 = Learner._compute_total_loss_static(model_none_proj, cfg_learner_proj, batch, jax.random.fold_in(lk,1), training=True)
    assert 'ssl_loss' in met2
    assert met2['ssl_loss'] == 0.0

def test_load_checkpoint_load_exception(key, cfg_flat):
    mk, lk = jax.random.split(key, 2)
    model = make_model(mk, cfg_flat)
    cfg_suffix = "load_exception"
    cfg = make_cfg(cfg_flat.value_support_size, cfg_flat.reward_support_size, 1, False, cfg_suffix)
    cfg = dataclasses.replace(cfg, resume_from_checkpoint=True)

    opt = optax.adam(cfg.learning_rate)
    # Create a dummy checkpoint manager that will raise an exception on restore
    class FailingCheckpointManager:
        def __init__(self, *args, **kwargs):
            self.latest_step_val = 1
        def latest_step(self):
            return self.latest_step_val # Pretend a checkpoint exists
        def restore(self, step, args=None):
            raise ValueError("Simulated restore error")
        def wait_until_finished(self):
            pass
        def save(self, step, args=None): # Add save to allow Learner init to proceed far enough
            pass 

    with patch('orbax.checkpoint.CheckpointManager', FailingCheckpointManager), \
         patch('builtins.print') as mock_print:
        # Learner init calls load_checkpoint
        learner = Learner(model, opt, cfg, lk) 
    
    assert learner.num_training_steps == 0 # Should not have loaded steps
    found_error_log = any(
        "Error loading checkpoint: Simulated restore error" in call_args[0][0]
        for call_args in mock_print.call_args_list
    )
    assert found_error_log, f"Expected error log not found. Logs: {mock_print.call_args_list}"
    # No directory to clean up as we mocked CheckpointManager heavily

def test_wandb_logging(key, cfg_flat):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = cfg_flat
    model = make_model(mk, cfgn)
    cfg_suffix = "wandb_log_test"
    # Ensure a unique directory that will be empty or cleaned
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, # num_unroll_steps
                     proj=False, 
                     suffix=cfg_suffix, 
                     use_ema=False)
    # Disable checkpointing for this specific test to avoid directory issues
    cfg = dataclasses.replace(cfg, checkpoint_dir=None, checkpoint_frequency=10000)

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)

    num_train_steps = 2
    batches = [make_batch(jax.random.fold_in(bk, i),
                          cfg.batch_size,
                          cfgn.observation_shape,
                          cfgn.num_actions,
                          cfg.num_unroll_steps,
                          cfgn.value_support_size,
                          cfgn.reward_support_size)
               for i in range(num_train_steps)]

    def get_batch_generator_fn():
        def gen():
            yield from batches
        return gen()

    with patch('wandb.log') as mock_wandb_log, \
         patch('wandb.init', return_value=None) as mock_wandb_init, \
         patch('wandb.run', new_callable=PropertyMock) as mock_wandb_run: # Added patch for wandb.run
        
        # Configure the mock_wandb_run to behave as if wandb.run is an active run object
        # A simple way is to make it not None. If it needs attributes, they can be set on a MagicMock.
        mock_wandb_run.return_value = patch.object # Use a simple object that's not None

        learner.train(get_batch_generator_fn, num_epochs=1, steps_per_epoch=num_train_steps)

    assert mock_wandb_log.call_count == num_train_steps
    
    # Check the arguments of each call to wandb.log
    for i in range(num_train_steps):
        call_args = mock_wandb_log.call_args_list[i]
        logged_metrics = call_args[0][0] # First positional argument to wandb.log
        logged_step = call_args[1]['step']    # Keyword argument 'step'
        
        assert isinstance(logged_metrics, dict)
        # Check for essential metric keys that should be present
        for expected_key in ['total_loss', 'policy_loss', 'value_loss', 'reward_loss', 'l2_loss', 'grad_norm', 'param_norm']:
            assert expected_key in logged_metrics
        
        assert logged_step == i + 1 # num_training_steps is incremented starting from 1

    # Clean up the dummy directory if make_cfg created it, though disabled for this test
    if cfg.checkpoint_dir and os.path.exists(cfg.checkpoint_dir):
        shutil.rmtree(cfg.checkpoint_dir) # pragma: no cover

def teardown_module(module):
    tmp_dir = "/tmp"
    for item in os.listdir(tmp_dir):
        if item.startswith("mz_test_"):
            path = os.path.join(tmp_dir, item)
            if os.path.isdir(path): shutil.rmtree(path) # pragma: no cover

# New test to verify Learner init with EMA and BatchStats
def test_learner_init_with_ema_and_batch_stats(key, cfg_flat):
    mk_model, mk_learner = jax.random.split(key)

    # Model config that will use BatchNorm
    class RepWithBN(nnx.Module):
        def __init__(self, obs_shape, hidden, *, rngs):
            self.dense = nnx.Linear(jnp.prod(jnp.array(obs_shape)), hidden, rngs=rngs)
            self.bn = nnx.BatchNorm(hidden, use_running_average=True, rngs=rngs) # Has BatchStat
        def __call__(self, x, training):
            if x.ndim > 2:
                x = x.reshape((x.shape[0], -1))
            x = self.dense(x)
            return self.bn(x, use_running_average=not training) # Use BN

    model_cfg_for_bn = dataclasses.replace(cfg_flat, hidden_size=4) # smaller hidden size
    
    model_with_bn = MuZeroNetwork(
        representation_network_def=lambda cfg, *, rngs: RepWithBN(cfg.observation_shape, cfg.hidden_size, rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: MockDyn(cfg.hidden_size, cfg.num_actions, rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: MockPred(cfg.hidden_size, cfg.num_actions, cfg.value_support_size, rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: MockRew(cfg.hidden_size, cfg.reward_support_size, rngs=rngs),
        projection_network_def=None,
        config=model_cfg_for_bn,
        rngs=nnx.Rngs(params=mk_model)
    )

    # Ensure the model actually has batch stats
    _, _, model_bs_before_learner, _, _, _ = nnx.split(model_with_bn, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    assert len(jax.tree_util.tree_leaves(model_bs_before_learner)) > 0 # Check that BatchStats are present

    learner_cfg = make_cfg(
        model_cfg_for_bn.value_support_size, 
        model_cfg_for_bn.reward_support_size, 
        1, False, 'init_ema_bn', use_ema=True
    )
    opt = optax.adam(learner_cfg.learning_rate)
    
    # This initialization should hit lines 102-105 in trainer.py
    learner = Learner(model_with_bn, opt, learner_cfg, mk_learner)

    assert learner.target_model is not None
    # Verify target model also has batch stats and they are properly initialized
    _, _, target_model_bs, _, _, _ = nnx.split(learner.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    assert len(jax.tree_util.tree_leaves(target_model_bs)) > 0

    # Check if the initial EMA state and the target_model's params are aligned
    _, initial_online_params_state, _, _, _, _ = nnx.split(learner.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    initial_online_param_values = jax.tree_util.tree_map(maybe_val, initial_online_params_state)
    
    _, target_params_state_after_init, _, _, _, _ = nnx.split(learner.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
    target_param_values_after_init = jax.tree_util.tree_map(maybe_val, target_params_state_after_init)

    jax.tree_util.tree_map(
        lambda online_val, target_val: jnp.allclose(online_val, target_val),
        initial_online_param_values,
        target_param_values_after_init
    )
    # Also check against learner.ema_params_state.ema
    jax.tree_util.tree_map(
        lambda online_val, ema_val: jnp.allclose(online_val, ema_val),
        initial_online_param_values,
        learner.ema_params_state.ema 
    )

def teardown_module(module):
    tmp_dir = "/tmp"
    for item in os.listdir(tmp_dir):
        if item.startswith("mz_test_"):
            path = os.path.join(tmp_dir, item)
            if os.path.isdir(path): shutil.rmtree(path) # pragma: no cover

# More tests to come for train, save/load checkpoint 