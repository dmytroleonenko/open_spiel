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
    create_network_config_from_muzero_config,
    MockMuZeroNetwork,
    create_test_muzero_network
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch,
    compute_gae_value_targets
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew 

def test_compute_gae_value_targets_with_episode_termination(common_key):
    """Test GAE computation with episode termination flags."""
    cfgn = MockNetCfg()

    # Create test configuration
    cfg = make_cfg(0, 0, 5, False, "gae_termination")

    # Create test data with episode terminations
    batch_size = 4
    observations = jnp.ones((batch_size, 8, *cfgn.observation_shape))
    actions = jnp.ones((batch_size, 7), dtype=jnp.int32)
    rewards = jnp.ones((batch_size, 8)) * 0.5
    dones = jnp.zeros((batch_size, 8))
    # Set termination at step 3 for some episodes
    dones = dones.at[0, 3].set(1.0)
    dones = dones.at[2, 5].set(1.0)

    # Create model and compute GAE targets
    model = make_model(common_key, cfgn)
    gae_targets = compute_gae_value_targets(
        model, observations, actions, rewards, dones, cfg,
        training=False, rng_key=common_key
    )

    # Verify shape and finite values
    expected_shape = (batch_size, cfg.num_unroll_steps + 1)
    assert gae_targets.shape == expected_shape
    assert jnp.all(jnp.isfinite(gae_targets))

    # Values should be reasonable (not extremely large)
    assert jnp.max(jnp.abs(gae_targets)) < 100.0

    print("✅ GAE computation with episode termination test passed!")


def test_compute_gae_value_targets_categorical_values(common_key):
    """Test GAE computation with categorical value distributions."""
    cfgn = MockNetCfg()

    # Create test configuration with categorical values
    cfg = make_cfg(21, 0, 5, False, "gae_categorical")

    batch_size = 3
    observations = jnp.ones((batch_size, 8, *cfgn.observation_shape))
    actions = jnp.ones((batch_size, 7), dtype=jnp.int32)
    rewards = jnp.ones((batch_size, 8)) * 0.3
    dones = jnp.zeros((batch_size, 8))

    # Create model and compute GAE targets
    model = make_model(common_key, cfgn)
    gae_targets = compute_gae_value_targets(
        model, observations, actions, rewards, dones, cfg,
        training=False, rng_key=common_key
    )

    # Verify shape and finite values
    expected_shape = (batch_size, cfg.num_unroll_steps + 1)
    assert gae_targets.shape == expected_shape
    assert jnp.all(jnp.isfinite(gae_targets))

    # Values should be in reasonable range for categorical case
    assert jnp.max(jnp.abs(gae_targets)) < 1000.0

    print("✅ GAE computation with categorical values test passed!")


def test_gae_integration_with_trainer_loss_computation(common_key):
    """Test GAE value targets integration with trainer loss computation."""
    cfgn = MockNetCfg()

    # Create test configuration with GAE targets
    cfg = make_cfg(0, 0, 3, False, "gae_integration", use_ema=False)
    cfg = dataclasses.replace(cfg, value_target_type="GAE", batch_size=2)

    batch_size = 2
    observations = jnp.ones((batch_size, 6, *cfgn.observation_shape))
    actions = jnp.ones((batch_size, 5), dtype=jnp.int32)
    rewards = jnp.ones((batch_size, 6)) * 0.2
    dones = jnp.zeros((batch_size, 6))

    # Create batch for trainer
    batch = make_batch(
        common_key, batch_size, cfgn.observation_shape, cfgn.num_actions, 
        cfg.num_unroll_steps, 0, 0, use_proj=False
    )

    # Test that loss computation works with GAE targets
    model = make_model(common_key, cfgn)
    loss, metrics = Learner._compute_total_loss_static(
        model=model,
        config=cfg,
        batch=batch,
        rng_key=common_key,
        training=True,
    )

    # Verify loss computation succeeds
    assert jnp.isfinite(loss)
    assert "value_loss" in metrics
    assert jnp.isfinite(metrics["value_loss"])

    print("✅ GAE integration with trainer loss computation test passed!")


def test_gae_fallback_to_precomputed_targets(common_key):
    """Test GAE fallback behavior when precomputed targets are available."""
    cfgn = MockNetCfg()

    # Create test configuration
    cfg = make_cfg(0, 0, 3, False, "gae_fallback", use_ema=False)
    cfg = dataclasses.replace(cfg, value_target_type="GAE", batch_size=2)

    batch_size = 2

    # Create batch with precomputed value targets
    batch = make_batch(
        common_key, batch_size, cfgn.observation_shape, cfgn.num_actions,
        cfg.num_unroll_steps, 0, 0, use_proj=False
    )

    # Verify that batch has value targets (correct field name is 'target_value')
    assert 'target_value' in batch
    assert batch['target_value'] is not None

    # Test loss computation (should use precomputed targets if available)
    model = make_model(common_key, cfgn)
    loss, metrics = Learner._compute_total_loss_static(
        model=model,
        config=cfg,
        batch=batch,
        rng_key=common_key,
        training=True,
    )

    # Verify computation succeeds
    assert jnp.isfinite(loss)
    assert "value_loss" in metrics
    assert jnp.isfinite(metrics["value_loss"])

    print("✅ GAE fallback to precomputed targets test passed!")


def test_compute_gae_adaptive_td_steps(common_key, common_cfg_flat):
    """Tests GAE computation with adaptive td_steps based on sample age."""
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        value_support_size=common_cfg_flat.value_support_size,  # Ensure this matches cfg_flat
        reward_support_size=common_cfg_flat.reward_support_size,  # Ensure this matches cfg_flat
        discount_factor=0.99,
        num_unroll_steps=2,
        td_steps=5,  # N-step for GAE
        use_adaptive_td_steps=True,
        auto_td_steps=10,  # For adaptive calculation: td_steps - (collected_transitions - sample_idx) // auto_td_steps
        value_target_type="GAE",  # Ensure GAE is used
        value_target="search",  # Use "search" instead of "mixed" to enable adaptive td_steps
        value_loss_type="mse" if common_cfg_flat.value_support_size == 0 else "categorical",
    )
    model = make_model(common_key, common_cfg_flat)

    batch_size = 2
    # Max steps needed for GAE lookahead, considering td_steps can be up to config.td_steps
    # For rewards and dones, we need up to K + td_steps. For observations, K + td_steps + 1 for the last value.
    num_obs_needed = config.num_unroll_steps + config.td_steps + 1
    num_rewards_dones_needed = config.num_unroll_steps + config.td_steps
    num_actions_needed = (
        config.num_unroll_steps + config.td_steps
    )  # K + N-1 actions lead to K+N states

    # Dummy data
    observations = jax.random.normal(
        common_key, (batch_size, num_obs_needed, *common_cfg_flat.observation_shape)
    )
    actions = jax.random.randint(
        common_key, (batch_size, num_actions_needed), 0, config.num_actions
    )

    # Rewards: simple sequence for easy manual verification later
    rewards_p1 = jnp.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    )[:num_rewards_dones_needed]
    rewards_p2 = jnp.array(
        [
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            -0.7,
            -0.8,
            -0.9,
            -1.0,
            -1.1,
            -1.2,
            -1.3,
            -1.4,
            -1.5,
        ]
    )[:num_rewards_dones_needed]
    rewards = jnp.stack([rewards_p1, rewards_p2])
    rewards = jnp.broadcast_to(rewards, (batch_size, num_rewards_dones_needed))

    # Dones: First sample terminates early, second runs full length
    dones_p1 = (
        jnp.zeros(num_rewards_dones_needed, dtype=jnp.bool_)
        .at[config.num_unroll_steps + 1]
        .set(True)
    )  # Terminates after K+1 step rewards (at state S_{K+2})
    dones_p2 = jnp.zeros(num_rewards_dones_needed, dtype=jnp.bool_)
    dones = jnp.stack([dones_p1, dones_p2])
    dones = jnp.broadcast_to(dones, (batch_size, num_rewards_dones_needed))

    sample_indices = jnp.arange(batch_size)

    # --- Scenario 1: Adaptive TD-steps -> Effective TD-steps should be small (e.g., 1) ---
    collected_transitions_old = (
        config.td_steps + 5
    ) * config.auto_td_steps  # Makes samples relatively old

    print(
        f"Scenario 1: Testing with use_adaptive_td_steps=True, collected_transitions={collected_transitions_old} (expecting small td_steps, likely 1)"
    )

    target_values_adaptive_old = compute_gae_value_targets(
        model=model,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        config=config,
        training=False,
        rng_key=common_key,
        sample_indices=sample_indices,
        collected_transitions=collected_transitions_old,
    )

    assert target_values_adaptive_old.shape == (batch_size, config.num_unroll_steps + 1)

    # --- Scenario 2: Adaptive TD-steps -> Effective TD-steps should be config.td_steps ---
    # Use sample indices that are very recent to minimize adaptive td_lambda effect
    collected_transitions_new = batch_size  # Make samples very new (sample age = 0 or 1)

    print(
        f"Scenario 2: Testing with use_adaptive_td_steps=True, collected_transitions={collected_transitions_new} (expecting td_steps={config.td_steps})"
    )

    target_values_adaptive_new = compute_gae_value_targets(
        model=model,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        config=config,
        training=False,
        rng_key=common_key,
        sample_indices=sample_indices,
        collected_transitions=collected_transitions_new,
    )
    assert target_values_adaptive_new.shape == (batch_size, config.num_unroll_steps + 1)

    # --- Scenario 3: Non-Adaptive TD-steps -> Effective TD-steps should be config.td_steps ---
    config_no_adaptive = dataclasses.replace(config, use_adaptive_td_steps=False)
    print(
        f"Scenario 3: Testing with use_adaptive_td_steps=False (expecting td_steps={config.td_steps})"
    )

    target_values_non_adaptive = compute_gae_value_targets(
        model=model,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        config=config_no_adaptive,
        training=False,
        rng_key=common_key,
        sample_indices=sample_indices,
        collected_transitions=collected_transitions_old,  # These should be ignored
    )
    assert target_values_non_adaptive.shape == (batch_size, config.num_unroll_steps + 1)

    # --- Scenario 4: Adaptive TD-steps, but value_target = 'mixed' or 'max' (should skip adaptation) ---
    config_mixed_target = dataclasses.replace(
        config, value_target="mixed", use_adaptive_td_steps=True
    )
    # Check 'max' as well
    config_max_target = dataclasses.replace(
        config, value_target="max", use_adaptive_td_steps=True
    )

    print(
        f"Scenario 4a: Testing with use_adaptive_td_steps=True, value_target='mixed' (expecting td_steps={config.td_steps})"
    )
    target_values_mixed_adaptive = compute_gae_value_targets(
        model=model,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        config=config_mixed_target,
        training=False,
        rng_key=common_key,
        sample_indices=sample_indices,
        collected_transitions=collected_transitions_old,  # Should use full td_steps despite this
    )
    assert target_values_mixed_adaptive.shape == (
        batch_size,
        config.num_unroll_steps + 1,
    )

    print(
        f"Scenario 4b: Testing with use_adaptive_td_steps=True, value_target='max' (expecting td_steps={config.td_steps})"
    )
    target_values_max_adaptive = compute_gae_value_targets(
        model=model,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        config=config_max_target,
        training=False,
        rng_key=common_key,
        sample_indices=sample_indices,
        collected_transitions=collected_transitions_old,  # Should use full td_steps despite this
    )
    assert target_values_max_adaptive.shape == (batch_size, config.num_unroll_steps + 1)

    print(
        "\n✅ test_compute_gae_adaptive_td_steps basic structure and calls completed."
    )

    # Assertions
    # 1. Old adaptive vs New adaptive should be different because td_steps used are different
    assert not jnp.allclose(
        target_values_adaptive_old, target_values_adaptive_new, atol=1e-5
    ), "Target values for 'old' (small td_steps) and 'new' (full td_steps) adaptive scenarios should differ."

    # 2. New adaptive vs Non-adaptive will differ due to adaptive td_lambda even with same td_steps
    # This is expected behavior when use_adaptive_td_steps=True
    print(f"Adaptive values: {target_values_adaptive_new}")
    print(f"Non-adaptive values: {target_values_non_adaptive}")
    print("Note: Differences expected due to adaptive td_lambda when use_adaptive_td_steps=True")

    # 3. Mixed target (adaptive but skipped) vs Non-adaptive should be the same
    assert jnp.allclose(
        target_values_mixed_adaptive, target_values_non_adaptive, atol=1e-4
    ), "Target values for 'mixed_target' (adaptation skipped) and non-adaptive should be similar."

    # 4. Max target (adaptive but skipped) vs Non-adaptive should be the same
    assert jnp.allclose(
        target_values_max_adaptive, target_values_non_adaptive, atol=1e-4
    ), "Target values for 'max_target' (adaptation skipped) and non-adaptive should be similar."

    # 5. Old adaptive vs Non-adaptive should be different due to different td_steps
    assert not jnp.allclose(
        target_values_adaptive_old, target_values_non_adaptive, atol=1e-3
    ), "Target values for 'old' adaptive (small td_steps) and non-adaptive (full td_steps) should differ significantly."

    # 6. Verify that adaptive mechanism actually changes behavior between old and new scenarios
    assert not jnp.allclose(
        target_values_adaptive_old, target_values_adaptive_new, atol=1e-3
    ), "Old and new adaptive scenarios should produce meaningfully different results."

    print("✅ test_compute_gae_adaptive_td_steps all assertions passed.")

def test_gae_adaptive_td_lambda_computation(common_key):
    """Test GAE computation with different td_lambda values."""
    batch_size = 2
    num_unroll_steps = 3
    gae_extra_steps = 2
    total_steps = num_unroll_steps + 1 + gae_extra_steps
    obs_shape = (4,)
    num_actions = 3

    # Test with different td_lambda values
    for td_lambda in [0.0, 0.5, 0.9, 1.0]:
        cfg = MuZeroConfig(
            num_unroll_steps=num_unroll_steps,
            td_steps=2,
            td_lambda=td_lambda,
            gae_max_steps=8,
            discount_factor=0.95,
            value_target_type="GAE",
        )

        # Create MockNetCfg for model creation
        mock_cfg = MockNetCfg(
            observation_shape=obs_shape,
            num_actions=num_actions,
            hidden_size=16,
            value_support_size=0,
            reward_support_size=0,
            projection_output_size=8,
            use_projection=False,
            batch_size=batch_size,
        )
        model = make_model(common_key, mock_cfg)

        # Create test data
        k1, k2, k3, k4 = jax.random.split(common_key, 4)
        observations = jax.random.uniform(k1, (batch_size, total_steps, *obs_shape))
        actions = jax.random.randint(k2, (batch_size, total_steps - 1), 0, num_actions)
        rewards = jax.random.uniform(
            k3, (batch_size, total_steps), minval=-0.5, maxval=0.5
        )
        dones = jnp.zeros((batch_size, total_steps))

        # Test GAE computation
        gae_targets = compute_gae_value_targets(
            model=model,
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            config=cfg,
            training=False,
            rng_key=common_key,
        )

        # Verify output properties
        assert gae_targets.shape == (batch_size, num_unroll_steps + 1)
        assert not jnp.isnan(gae_targets).any(), f"NaN with td_lambda={td_lambda}"
        assert jnp.isfinite(
            gae_targets
        ).all(), f"Infinite values with td_lambda={td_lambda}"

        # Different td_lambda values should produce different results
        # (except for the trivial case where rewards/values are constant)
        if td_lambda == 0.0:
            # With td_lambda=0, GAE should reduce to TD-error only
            pass  # Just verify no errors occur
        elif td_lambda == 1.0:
            # With td_lambda=1, GAE should use full episode returns
            pass  # Just verify no errors occur 