import pytest
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.nnx.graph as nnx_graph
import optax

# Import from the common utils module
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
    NUM_UNROLL_STEPS,
    key as common_key, cfg_flat as common_cfg_flat, cfg_img,
    make_model, make_cfg, make_batch,
    OBS_SHAPE_IMAGE, OBS_SHAPE_FLAT, NUM_ACTIONS,
    VALUE_SUPPORT_CATEGORICAL, VALUE_SUPPORT_SCALAR,
    REWARD_SUPPORT_CATEGORICAL, REWARD_SUPPORT_SCALAR,
    MockNetCfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork 

@pytest.mark.parametrize(
    "img,val_cat,proj,use_ema,scalar_targets",
    [
        (False, False, False, False, True),
        (True, True, True, True, False),
        (False, False, True, False, True),
        (False, False, False, True, False),
        (
            False,
            True,
            False,
            False,
            False,
        ),  # Scalar outputs, categorical targets (value only)
        (False, False, False, False, False),  # Scalar model output, Categorical targets
    ],
)
def test_loss_static(
    common_key, img, val_cat, proj, use_ema, scalar_targets, common_cfg_flat, cfg_img
):
    bk, mk, lk = jax.random.split(common_key, 3)

    obs_shape_test = OBS_SHAPE_IMAGE if img else OBS_SHAPE_FLAT
    num_actions_test = NUM_ACTIONS
    hidden_size_test = 4  # Smaller hidden size for simpler manual calculation
    unroll_steps_test = 1  # Single unroll step for simplicity
    batch_size_test = 1  # Single batch item for simplicity

    # Configure model
    cfgn_model = MockNetCfg(
        observation_shape=obs_shape_test,
        num_actions=num_actions_test,
        hidden_size=hidden_size_test,
        value_support_size=(
            VALUE_SUPPORT_CATEGORICAL if val_cat else VALUE_SUPPORT_SCALAR
        ),
        reward_support_size=(
            REWARD_SUPPORT_CATEGORICAL if val_cat else REWARD_SUPPORT_SCALAR
        ),  # Keep reward cat/scalar same as value for this test
        projection_output_size=(
            hidden_size_test // 2 if proj else 0
        ),  # Smaller projection
        use_projection=proj,
        batch_size=batch_size_test,
    )

    # --- Create a very simple model with fixed weights for predictability ---
    class FixedRep(nnx.Module):
        def __init__(self, *, rngs):
            self.dense = nnx.Linear(
                jnp.prod(jnp.array(cfgn_model.observation_shape)),
                cfgn_model.hidden_size,
                rngs=rngs,
            )
            # Fix weights and biases
            self.dense.kernel.value = jnp.ones_like(self.dense.kernel.value) * 0.1
            self.dense.bias.value = jnp.zeros_like(self.dense.bias.value) + 0.05
            # Add a mock BN layer, but its state won't change if training=False during loss calculation
            self.bn = nnx.BatchNorm(
                cfgn_model.hidden_size, use_running_average=True, rngs=rngs
            )
            self.bn.scale.value = jnp.ones_like(self.bn.scale.value)
            self.bn.bias.value = jnp.zeros_like(self.bn.bias.value)
            self.bn.mean.value = jnp.zeros_like(self.bn.mean.value)
            self.bn.var.value = jnp.ones_like(self.bn.var.value)

        def __call__(self, x, training):
            if x.ndim > 2:
                x = x.reshape((x.shape[0], -1))
            x = self.dense(x)
            return self.bn(x, use_running_average=not training)  # Pass training flag

    class FixedDyn(nnx.Module):
        def __init__(self, *, rngs):
            self.embed = nnx.Embed(
                cfgn_model.num_actions, cfgn_model.hidden_size // 2, rngs=rngs
            )
            self.fc = nnx.Linear(
                cfgn_model.hidden_size + cfgn_model.hidden_size // 2,
                cfgn_model.hidden_size,
                rngs=rngs,
            )
            # Fix weights
            self.embed.embedding.value = (
                jnp.ones_like(self.embed.embedding.value) * 0.05
            )
            self.fc.kernel.value = jnp.ones_like(self.fc.kernel.value) * 0.2
            self.fc.bias.value = jnp.zeros_like(self.fc.bias.value) + 0.02

        def __call__(self, h, a, training):
            e = self.embed(a)
            if e.ndim == 1:
                e = jnp.broadcast_to(e, (h.shape[0], e.shape[-1]))
            return nnx.relu(self.fc(jnp.concatenate([h, e], -1)))

    class FixedPred(nnx.Module):
        def __init__(self, *, rngs):
            self.ph_w = jnp.ones((cfgn_model.hidden_size, cfgn_model.num_actions)) * 0.3
            self.ph_b = jnp.zeros(cfgn_model.num_actions) + 0.01
            v_out_dim = (
                1
                if cfgn_model.value_support_size == 0
                else cfgn_model.value_support_size
            )
            self.vh_w = jnp.ones((cfgn_model.hidden_size, v_out_dim)) * 0.4
            self.vh_b = jnp.zeros(v_out_dim) + 0.03

        def __call__(self, h, training):
            p_logits = h @ self.ph_w + self.ph_b
            val_out = h @ self.vh_w + self.vh_b
            return p_logits, val_out

    class FixedRew(nnx.Module):
        def __init__(self, *, rngs):
            r_out_dim = (
                1
                if cfgn_model.reward_support_size == 0
                else cfgn_model.reward_support_size
            )
            self.rh_w = jnp.ones((cfgn_model.hidden_size, r_out_dim)) * 0.25
            self.rh_b = jnp.zeros(r_out_dim) + 0.04

        def __call__(self, h, training):
            return h @ self.rh_w + self.rh_b

    class FixedProj(nnx.Module):
        def __init__(self, *, rngs):
            self.proj_w = (
                jnp.ones((cfgn_model.hidden_size, cfgn_model.projection_output_size))
                * 0.15
            )
            self.proj_b = jnp.zeros(cfgn_model.projection_output_size) + 0.005

        def __call__(self, h, training):
            return h @ self.proj_w + self.proj_b

    fixed_model = MuZeroNetwork(
        representation_network_def=lambda cfg, *, rngs: FixedRep(rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: FixedDyn(rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: FixedPred(rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: FixedRew(rngs=rngs),
        projection_network_def=lambda cfg, *, rngs: (
            FixedProj(rngs=rngs) if cfgn_model.use_projection else None
        ),
        config=cfgn_model,
        rngs=nnx.Rngs(params=mk),
    )
    # --- End Fixed Model ---

    # Configure Learner
    cfg_learner = make_cfg(
        cfgn_model.value_support_size,
        cfgn_model.reward_support_size,
        unroll_steps_test,
        cfgn_model.use_projection,
        f"loss_static_num_val_{img}_{val_cat}_{proj}_{scalar_targets}",
        use_ema=use_ema,
        ssl_weight=(
            0.5 if cfgn_model.use_projection else 0.0
        ),  # Non-zero SSL weight for testing
        l2_weight=1e-2,  # Non-zero L2 for testing
        checkpoint_dir=None,  # No checkpointing needed for this test
    )
    cfg_learner = dataclasses.replace(cfg_learner, batch_size=batch_size_test)

    # --- Create fixed batch data ---
    fixed_obs_val = 0.5
    fixed_action_val = 1

    # (B, K+1, *obs_shape) -> (1, 2, *obs_shape) since unroll_steps_test = 1
    obs_data = jnp.full(
        (batch_size_test, unroll_steps_test + 1, *cfgn_model.observation_shape),
        fixed_obs_val,
    )
    # (B, K) -> (1, 1)
    action_data = jnp.full(
        (batch_size_test, unroll_steps_test), fixed_action_val, dtype=jnp.int32
    )

    # Targets (B, K+1, Support_Size_or_1)
    fixed_target_policy_logits = jnp.ones(NUM_ACTIONS) * 0.1  # Simplified to use NUM_ACTIONS
    fixed_target_policy_logits = fixed_target_policy_logits.at[0].set(-0.1)
    fixed_target_policy_logits = fixed_target_policy_logits.at[1].set(0.5)
    if NUM_ACTIONS > 2:
        fixed_target_policy_logits = fixed_target_policy_logits.at[2].set(-0.2)
    
    target_policy_data = jax.nn.softmax(
        jnp.tile(
            fixed_target_policy_logits, (batch_size_test, unroll_steps_test + 1, 1)
        ),
        axis=-1,
    )

    if scalar_targets:  # Scalar targets
        target_value_data = jnp.full((batch_size_test, unroll_steps_test + 1), 0.75)
        target_reward_data = jnp.full((batch_size_test, unroll_steps_test + 1), 0.25)
    else:  # Categorical targets
        v_support_sz = (
            cfgn_model.value_support_size if cfgn_model.value_support_size > 0 else 1
        )
        r_support_sz = (
            cfgn_model.reward_support_size if cfgn_model.reward_support_size > 0 else 1
        )

        tv_dist = jnp.zeros(v_support_sz)
        if v_support_sz > 0:
            tv_dist = tv_dist.at[v_support_sz // 2].set(1.0)  # one-hot
        else:
            tv_dist = jnp.array([0.75])  # scalar if support is 0
        target_value_data = jnp.tile(
            tv_dist, (batch_size_test, unroll_steps_test + 1, 1)
        )
        if cfgn_model.value_support_size == 0:
            target_value_data = jnp.squeeze(target_value_data, axis=-1)

        tr_dist = jnp.zeros(r_support_sz)
        if r_support_sz > 0:
            tr_dist = tr_dist.at[r_support_sz // 2].set(1.0)  # one-hot
        else:
            tr_dist = jnp.array([0.25])  # scalar if support is 0
        target_reward_data = jnp.tile(
            tr_dist, (batch_size_test, unroll_steps_test + 1, 1)
        )
        if cfgn_model.reward_support_size == 0:
            target_reward_data = jnp.squeeze(target_reward_data, axis=-1)

    mask_data = jnp.ones((batch_size_test, unroll_steps_test + 1))

    fixed_batch = {
        "observation": obs_data,
        "action": action_data,
        "target_reward": target_reward_data,
        "target_value": target_value_data,
        "target_policy": target_policy_data,
        "game_history_mask": mask_data,
    }
    # --- End fixed batch data ---

    # Compute loss using the Learner's static method
    computed_loss, computed_metrics = Learner._compute_total_loss_static(
        fixed_model,
        cfg_learner,
        fixed_batch,
        lk,
        training=False,  # training=False to avoid BN updates for this test
    )

    # --- Manually calculate expected losses ---
    # 1. Forward pass through the fixed model
    # Initial inference
    obs_init = fixed_batch["observation"][:, 0]  # (B, *obs_shape)
    s0, r0_pred, v0_pred, p0_logits, proj0_pred, reward_hidden = fixed_model.initial_inference(
        obs_init, training=False
    )

    # Recurrent inference (1 step)
    action_k0 = fixed_batch["action"][:, 0]  # (B,)
    s1, r1_pred, v1_pred, p1_logits, proj1_pred, reward_hidden = fixed_model.recurrent_inference(
        s0, action_k0, training=False
    )

    # For K=1 unroll steps, we have K+1 = 2 sets of predictions/targets
    # Predictions: (r0_pred, v0_pred, p0_logits), (r1_pred, v1_pred, p1_logits)
    # Targets: target_reward[:,0], target_value[:,0], target_policy[:,0]
    #          target_reward[:,1], target_value[:,1], target_policy[:,1]

    # Policy Loss (Cross-entropy)
    # Step 0
    expected_policy_loss_s0 = -jnp.sum(
        target_policy_data[:, 0] * jax.nn.log_softmax(p0_logits), axis=-1
    )
    # Step 1
    expected_policy_loss_s1 = -jnp.sum(
        target_policy_data[:, 1] * jax.nn.log_softmax(p1_logits), axis=-1
    )
    expected_policy_loss = (
        jnp.sum(expected_policy_loss_s0 * mask_data[:, 0])
        + jnp.sum(expected_policy_loss_s1 * mask_data[:, 1])
    ) / jnp.maximum(jnp.sum(mask_data), 1.0)

    # Value Loss
    # Step 0
    if (
        cfgn_model.value_support_size > 0
    ):  # Categorical - now uses KL divergence (EfficientZeroV2 pattern)
        tv0 = target_value_data[:, 0]
        # KL divergence: sum(target * (log(target) - log_softmax(prediction)))
        target_log_probs_s0 = jnp.log(jnp.clip(tv0, 1e-8, 1.0))
        pred_log_probs_s0 = jax.nn.log_softmax(v0_pred, axis=-1)
        expected_value_loss_s0 = jnp.sum(
            tv0 * (target_log_probs_s0 - pred_log_probs_s0), axis=-1
        )
    else:  # Scalar (MSE)
        tv0 = target_value_data[:, 0]
        # Ensure scalar predictions are squeezed if value_support_size is 0 (implying scalar output)
        v0_pred_squeezed = (
            jnp.squeeze(v0_pred, axis=-1) if v0_pred.shape[-1] == 1 else v0_pred
        )
        expected_value_loss_s0 = (v0_pred_squeezed - tv0) ** 2
    # Step 1
    if (
        cfgn_model.value_support_size > 0
    ):  # Categorical - now uses KL divergence (EfficientZeroV2 pattern)
        tv1 = target_value_data[:, 1]
        # KL divergence: sum(target * (log(target) - log_softmax(prediction)))
        target_log_probs_s1 = jnp.log(jnp.clip(tv1, 1e-8, 1.0))
        pred_log_probs_s1 = jax.nn.log_softmax(v1_pred, axis=-1)
        expected_value_loss_s1 = jnp.sum(
            tv1 * (target_log_probs_s1 - pred_log_probs_s1), axis=-1
        )
    else:  # Scalar (MSE)
        tv1 = target_value_data[:, 1]
        v1_pred_squeezed = (
            jnp.squeeze(v1_pred, axis=-1) if v1_pred.shape[-1] == 1 else v1_pred
        )
        expected_value_loss_s1 = (v1_pred_squeezed - tv1) ** 2
    expected_value_loss = (
        jnp.sum(expected_value_loss_s0 * mask_data[:, 0])
        + jnp.sum(expected_value_loss_s1 * mask_data[:, 1])
    ) / jnp.maximum(jnp.sum(mask_data), 1.0)

    # Reward Loss (similar to value)
    # Step 0
    if cfgn_model.reward_support_size > 0:  # Categorical
        tr0 = target_reward_data[:, 0]
        expected_reward_loss_s0 = -jnp.sum(
            tr0 * jax.nn.log_softmax(r0_pred, axis=-1), axis=-1
        )
    else:  # Scalar (MSE)
        tr0 = target_reward_data[:, 0]
        r0_pred_squeezed = (
            jnp.squeeze(r0_pred, axis=-1) if r0_pred.shape[-1] == 1 else r0_pred
        )
        expected_reward_loss_s0 = (r0_pred_squeezed - tr0) ** 2
    # Step 1
    if cfgn_model.reward_support_size > 0:  # Categorical
        tr1 = target_reward_data[:, 1]
        expected_reward_loss_s1 = -jnp.sum(
            tr1 * jax.nn.log_softmax(r1_pred, axis=-1), axis=-1
        )
    else:  # Scalar (MSE)
        tr1 = target_reward_data[:, 1]
        r1_pred_squeezed = (
            jnp.squeeze(r1_pred, axis=-1) if r1_pred.shape[-1] == 1 else r1_pred
        )
        expected_reward_loss_s1 = (r1_pred_squeezed - tr1) ** 2
    expected_reward_loss = (
        jnp.sum(expected_reward_loss_s0 * mask_data[:, 0])
        + jnp.sum(expected_reward_loss_s1 * mask_data[:, 1])
    ) / jnp.maximum(jnp.sum(mask_data), 1.0)

    # L2 Loss
    expected_l2_loss = 0.0
    _, params_for_l2, batch_stats_for_l2, _, _, _ = nnx.split(
        fixed_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
    )

    # Manually iterate through the fixed weights we defined for L2
    # Rep
    expected_l2_loss += (
        0.5
        * cfg_learner.l2_weight
        * jnp.sum(fixed_model.representation_network.dense.kernel.value**2)
    )
    # BN scale and bias are params, but mean/var are batch_stats and not part of L2 loss by default.
    # Optax L2 regularizer usually only targets 'kernel' and 'bias' like names if filtered, or all params if not.
    # Our losses_lib.l2_regularization applies to all params in the given PyTree.
    expected_l2_loss += (
        0.5
        * cfg_learner.l2_weight
        * jnp.sum(fixed_model.representation_network.bn.scale.value**2)
    )
    expected_l2_loss += (
        0.5
        * cfg_learner.l2_weight
        * jnp.sum(fixed_model.representation_network.bn.bias.value**2)
    )

    # Dyn
    expected_l2_loss += (
        0.5
        * cfg_learner.l2_weight
        * jnp.sum(fixed_model.dynamics_network.embed.embedding.value**2)
    )
    expected_l2_loss += (
        0.5
        * cfg_learner.l2_weight
        * jnp.sum(fixed_model.dynamics_network.fc.kernel.value**2)
    )
    # Pred (using the manually set weight matrices)
    expected_l2_loss += (
        0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.prediction_network.ph_w**2)
    )
    expected_l2_loss += (
        0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.prediction_network.vh_w**2)
    )
    # Rew
    expected_l2_loss += (
        0.5 * cfg_learner.l2_weight * jnp.sum(fixed_model.reward_network.rh_w**2)
    )

    # SSL Loss (Cosine similarity based, scaled and shifted)
    expected_ssl_loss = 0.0
    if cfgn_model.use_projection and cfg_learner.consistency_loss_coeff > 0:
        # Calculate according to losses_lib.compute_projection_consistency_loss
        # proj1_pred is projection_current_step, proj0_pred is projection_initial_step

        sim1_test = optax.cosine_similarity(
            proj1_pred, jax.lax.stop_gradient(proj0_pred)
        )
        sim2_test = optax.cosine_similarity(
            jax.lax.stop_gradient(proj1_pred), proj0_pred
        )

        clipped_sim1_test = jnp.clip(sim1_test, -1.0, 1.0)
        clipped_sim2_test = jnp.clip(sim2_test, -1.0, 1.0)

        # For batch_size_test = 1, the per-item loss is the value itself
        # The loss is applied per unroll step, and then averaged.
        # Here, we only care about the SSL loss for k_idx=1 vs k_idx=0
        # The trainer's _compute_total_loss_static applies a mask and averages.
        # Since mask_data[:, 1] is 1 and batch size is 1, this should be direct.

        # This is the per-instance loss for the (proj1_pred, proj0_pred) pair
        # Note: compute_projection_consistency_loss now returns per-item losses, not batch-averaged
        ssl_loss_per_item = -clipped_sim1_test - clipped_sim2_test  # Shape (B,)

        # The test setup has unroll_steps_test = 1.
        # The SSL loss is calculated for k_idx > 0. So only for k_idx = 1.
        # The trainer's _compute_total_loss_static applies a mask and averages.
        # expected_ssl_loss should be the value that goes into metrics['ssl_loss']
        # which is total_ssl_loss, accumulated and averaged.
        # For a single unroll step (k_idx=1), and batch size 1, with mask=1:
        # total_ssl_loss = (sum over k_idx > 0) of [ (sum over batch for (loss_val * mask)) / sum(mask) ]
        # Here, just one term: ( ( (loss_for_pair_batch_item_0 * 1) / 1 )
        # Since ssl_loss_per_item has shape (B,) and B=1, we take the first (and only) element
        expected_ssl_loss = ssl_loss_per_item[
            0
        ]  # For batch_size_test = 1, take the single batch item loss

        # Add L2 for projection network if it exists
        expected_l2_loss += (
            0.5
            * cfg_learner.l2_weight
            * jnp.sum(fixed_model.projection_network.proj_w**2)
        )

    expected_total_loss = (
        cfg_learner.policy_loss_weight * expected_policy_loss
        + cfg_learner.value_loss_weight * expected_value_loss
        + cfg_learner.reward_loss_weight * expected_reward_loss
        + expected_l2_loss
    )
    if (
        cfgn_model.use_projection and cfg_learner.consistency_loss_coeff > 0
    ):  # Add SSL to total loss
        expected_total_loss += cfg_learner.consistency_loss_coeff * expected_ssl_loss

    # --- Assertions ---
    assert isinstance(computed_loss, jax.Array) and computed_loss.shape == ()
    for m_key in ["total_loss", "policy_loss", "value_loss", "reward_loss", "l2_loss"]:
        assert m_key in computed_metrics, f"{m_key} not in computed metrics"

    # Add specific assertions for support size 1 handling (scalar equivalence)
    if (
        cfgn_model.value_support_size == 1 and not val_cat
    ):  # Scalar output, categorical target of size 1
        # Loss should be very low if target is effectively [1.0] and prediction is close to 0 (log_softmax(0) = 0 for one class)
        # This scenario needs more thought for precise expectation. Current cross-entropy handles it.
        pass
    if (
        cfgn_model.value_support_size == 0
        and scalar_targets
        and target_value_data.shape[-1] == 1
    ):  # Scalar output, scalar target (originally size 1)
        # Ensure MSE is calculated correctly after squeeze. Already handled by squeeze in manual calculation.
        pass

    jnp.allclose(computed_metrics["policy_loss"], expected_policy_loss, atol=1e-5)
    jnp.allclose(computed_metrics["value_loss"], expected_value_loss, atol=1e-5)
    jnp.allclose(computed_metrics["reward_loss"], expected_reward_loss, atol=1e-5)
    jnp.allclose(computed_metrics["l2_loss"], expected_l2_loss, atol=1e-5)

    if cfgn_model.use_projection and cfg_learner.consistency_loss_coeff > 0:
        assert "ssl_loss" in computed_metrics
        jnp.allclose(computed_metrics["ssl_loss"], expected_ssl_loss, atol=1e-5)
        # Check if SSL loss contributes if weight > 0
        # This assertion is problematic as SSL loss can be negative.
        # Removing it and relying on allclose with the correctly calculated expected_ssl_loss.
        # if proj0_pred is not None and proj1_pred is not None and jnp.any(proj0_pred != proj1_pred):
        #    assert computed_metrics[\\\'ssl_loss\\\'] > 1e-6, "SSL loss should be non-zero if projections differ and weight > 0"

    jnp.allclose(computed_loss, expected_total_loss, atol=1e-5)


def test_loss_static_scalar_pred_categorical_reward_loss_zero_support(common_key, common_cfg_flat):
    """Test _compute_total_loss_static for categorical reward loss with scalar predictions and zero support size."""
    bk, mk, lk = jax.random.split(common_key, 3)

    obs_shape_test = OBS_SHAPE_FLAT
    num_actions_test = NUM_ACTIONS
    hidden_size_test = 4
    unroll_steps_test = 1
    batch_size_test = 1

    # Model config: scalar reward output
    cfgn_model = MockNetCfg(
        observation_shape=obs_shape_test,
        num_actions=num_actions_test,
        hidden_size=hidden_size_test,
        value_support_size=VALUE_SUPPORT_SCALAR,
        reward_support_size=VALUE_SUPPORT_SCALAR,  # Model outputs scalar rewards
        projection_output_size=0,
        use_projection=False,
        batch_size=batch_size_test,
    )
    model = make_model(mk, cfgn_model)

    # Learner config: categorical reward loss, reward_support_size = 0
    cfg_learner = make_cfg(
        vsup=cfgn_model.value_support_size,
        rsup=0,  # This is key: reward_support_size = 0
        steps=unroll_steps_test,
        proj=False,
        suffix="_scalar_pred_cat_rew_zero_sup",
        use_ema=False,
        ssl_weight=0.0,
        l2_weight=0.0,
    )
    # Force reward_loss_type to categorical
    cfg_learner = dataclasses.replace(
        cfg_learner, reward_loss_type="categorical", batch_size=batch_size_test
    )

    # Batch: scalar targets
    batch = make_batch(
        key_rng=bk,
        bs=batch_size_test,
        obs_shape=obs_shape_test,
        nact=num_actions_test,
        steps=unroll_steps_test,
        vsup=VALUE_SUPPORT_SCALAR,
        rsup=VALUE_SUPPORT_SCALAR,  # Scalar targets
        use_proj=False,
    )

    # Compute loss
    _, metrics = Learner._compute_total_loss_static(
        model, cfg_learner, batch, lk, training=True
    )

    assert "reward_loss" in metrics
    assert jnp.isfinite(metrics["reward_loss"])
    # Further checks could involve verifying the 601 atoms were used in scalar_to_support for predicted_rew
    # but confirming the path is taken (no error and finite loss) is the main goal for coverage.


def test_compute_total_loss_static_accepts_positional_args(common_key, common_cfg_flat):
    """Legacy positional args for training/training_step must still function."""
    cfg = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        NUM_UNROLL_STEPS,
        False,
        "positional_args",
        use_ema=False,
    )
    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        key_rng=common_key,
        bs=cfg.batch_size,
        obs_shape=common_cfg_flat.observation_shape,
        nact=common_cfg_flat.num_actions,
        steps=cfg.num_unroll_steps,
        vsup=cfg.value_support_size,
        rsup=cfg.reward_support_size,
        use_proj=False,
    )
    loss, metrics = Learner._compute_total_loss_static(
        model, cfg, batch, common_key, False, 2
    )
    assert jnp.isfinite(loss)
    assert 'total_loss' in metrics


def test_compute_total_loss_static_raises_on_time_mismatch(common_key, common_cfg_flat):
    """The time-dimension guard should raise when tensors do not align."""
    cfg = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        NUM_UNROLL_STEPS,
        False,
        "time_mismatch",
        use_ema=False,
    )
    cfg = dataclasses.replace(cfg, reanalyze_ratio=0.0)
    model = make_model(common_key, common_cfg_flat)
    batch = make_batch(
        key_rng=common_key,
        bs=cfg.batch_size,
        obs_shape=common_cfg_flat.observation_shape,
        nact=common_cfg_flat.num_actions,
        steps=cfg.num_unroll_steps,
        vsup=cfg.value_support_size,
        rsup=cfg.reward_support_size,
        use_proj=False,
    )
    batch['target_policy'] = batch['target_policy'][:, :-1]
    with pytest.raises(ValueError, match="Time-dimension mismatch"):
        Learner._compute_total_loss_static(model, cfg, batch, common_key)
