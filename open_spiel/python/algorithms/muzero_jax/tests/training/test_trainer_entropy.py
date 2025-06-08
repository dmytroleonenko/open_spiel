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
    MockNetCfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch,
    get_temperature
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_utils import MockRep, MockDyn, MockPred, MockRew

def test_entropy_regularization_comprehensive(common_key, common_cfg_flat):
    """Test comprehensive entropy regularization functionality."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib

    # Test 1: Discrete action entropy regularization
    cfg_discrete = make_cfg(
        vsup=0, rsup=0, steps=2, proj=False, suffix="_entropy_discrete", use_ema=False
    )
    cfg_discrete = dataclasses.replace(
        cfg_discrete,
        entropy_coeff=0.1,
        action_type="discrete",
        distribution_type="categorical",
    )

    model_discrete = make_model(common_key, common_cfg_flat)
    batch_discrete = make_batch(
        common_key,
        cfg_discrete.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg_discrete.num_unroll_steps,
        vsup=0,
        rsup=0,
        use_proj=False,
    )

    loss_discrete, metrics_discrete = Learner._compute_total_loss_static(
        model=model_discrete,
        config=cfg_discrete,
        batch=batch_discrete,
        rng_key=common_key,
        training=True,
    )

    assert jnp.isfinite(loss_discrete)
    assert "entropy_loss" in metrics_discrete
    assert metrics_discrete["entropy_loss"] > 0.0

    # Test 2: Verify entropy regularization affects total loss (entropy coefficient = 0 vs > 0)
    cfg_no_entropy = dataclasses.replace(cfg_discrete, entropy_coeff=0.0)

    loss_no_entropy, metrics_no_entropy = Learner._compute_total_loss_static(
        model=model_discrete,
        config=cfg_no_entropy,
        batch=batch_discrete,
        rng_key=common_key,
        training=True,
    )

    # With entropy regularization, total loss should be different (typically lower due to entropy bonus)
    assert not jnp.isclose(loss_discrete, loss_no_entropy, rtol=1e-5)
    assert (
        "entropy_loss" not in metrics_no_entropy
    )  # Should not be computed when coeff=0

    # Test 3: Verify entropy functions work correctly in isolation
    # Test discrete entropy
    policy_logits = jax.random.normal(common_key, (4, 6))
    discrete_entropy = losses_lib.compute_policy_entropy_general(
        policy_logits, action_type="discrete", distribution_type="categorical"
    )
    assert discrete_entropy.shape == (4,)
    assert jnp.all(discrete_entropy >= 0.0)

    # Test continuous entropy for normal distribution
    continuous_params = jax.random.normal(common_key, (4, 8))  # 4 actions * 2 params
    continuous_entropy = losses_lib.compute_policy_entropy_general(
        continuous_params, action_type="continuous", distribution_type="normal"
    )
    assert continuous_entropy.shape == (4,)
    assert jnp.all(continuous_entropy > 0.0)

    # Test continuous entropy for squashed normal distribution
    squashed_entropy = losses_lib.compute_policy_entropy_general(
        continuous_params, action_type="continuous", distribution_type="squashed_normal"
    )
    assert squashed_entropy.shape == (4,)
    assert jnp.all(squashed_entropy > 0.0)

    print("✅ Comprehensive entropy regularization functionality verified!")

def test_entropy_mathematical_properties_integration(common_key, common_cfg_flat):
    """Test mathematical properties of entropy integration in trainer."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib

    # Simplified test: just verify that entropy functions work correctly with different distributions
    # Test uniform distribution (maximum entropy)
    uniform_logits = jnp.zeros(
        (4, common_cfg_flat.num_actions)
    )  # All zeros = uniform distribution
    uniform_entropy = losses_lib.compute_policy_entropy(uniform_logits)
    expected_max_entropy = jnp.log(common_cfg_flat.num_actions)
    assert jnp.allclose(uniform_entropy, expected_max_entropy, rtol=0.01)

    # Test deterministic distribution (minimal entropy)
    deterministic_logits = jnp.zeros((4, common_cfg_flat.num_actions))
    deterministic_logits = deterministic_logits.at[:, 0].set(
        10.0
    )  # Very high logit for first action
    deterministic_entropy = losses_lib.compute_policy_entropy(deterministic_logits)

    # Deterministic policy should have much lower entropy than uniform
    assert jnp.all(deterministic_entropy < uniform_entropy * 0.1)

    # Test that entropy is always non-negative
    random_logits = jax.random.normal(common_key, (4, common_cfg_flat.num_actions))
    random_entropy = losses_lib.compute_policy_entropy(random_logits)
    assert jnp.all(random_entropy >= 0.0)

    print("✅ Entropy mathematical properties integration verified!")

def test_entropy_error_handling_integration(common_key, common_cfg_flat):
    """Test error handling for entropy functions in trainer integration."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib

    # Test unsupported action type
    cfg_invalid = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_entropy_invalid", use_ema=False
    )
    cfg_invalid = dataclasses.replace(
        cfg_invalid,
        entropy_coeff=0.1,
        action_type="unsupported",
        distribution_type="categorical",
    )

    model_invalid = make_model(common_key, common_cfg_flat)
    batch_invalid = make_batch(
        common_key,
        cfg_invalid.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg_invalid.num_unroll_steps,
        vsup=0,
        rsup=0,
        use_proj=False,
    )

    # This should raise an error when trying to compute entropy
    try:
        loss_invalid, metrics_invalid = Learner._compute_total_loss_static(
            model=model_invalid,
            config=cfg_invalid,
            batch=batch_invalid,
            rng_key=common_key,
            training=True,
        )
        assert False, "Should have raised ValueError for unsupported action type"
    except ValueError as e:
        assert "action_type" in str(e)

    # Test unsupported distribution type for continuous actions (isolated function test)
    continuous_params = jax.random.normal(common_key, (2, 4))
    try:
        entropy_invalid_dist = losses_lib.compute_policy_entropy_general(
            continuous_params,
            action_type="continuous",
            distribution_type="unsupported_distribution",
        )
        assert (
            False
        ), "Should have raised NotImplementedError for unsupported distribution type"
    except NotImplementedError as e:
        assert "unsupported_distribution" in str(e)

    print("✅ Entropy error handling integration verified!")

def test_get_temperature_coverage(common_key, common_cfg_flat):
    """Test get_temperature function to cover missing lines 1595, 1598."""

    # Test case where training_step >= temperature_decay_steps (line 1595)
    config_temp_decay = MuZeroConfig(
        change_temperature=True,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_decay_steps=1000,
    )

    # Test beyond decay steps - should return final temperature (line 1595)
    temp_final = get_temperature(training_step=1500, config=config_temp_decay)
    assert temp_final == config_temp_decay.temperature_final

    # Test exactly at decay steps - should return final temperature (line 1595)
    temp_at_decay = get_temperature(training_step=1000, config=config_temp_decay)
    assert temp_at_decay == config_temp_decay.temperature_final

    # Test min temperature clipping (line 1598) - though this is redundant with current config
    config_edge = MuZeroConfig(
        change_temperature=True,
        temperature_init=0.5,
        temperature_final=1.0,  # Final > Init to test max() clipping
        temperature_decay_steps=1000,
    )

    temp_mid = get_temperature(training_step=500, config=config_edge)
    assert temp_mid >= config_edge.temperature_final  # Should be clipped by max()

    # Test no temperature change (should return init)
    config_no_change = MuZeroConfig(change_temperature=False, temperature_init=2.0)
    temp_no_change = get_temperature(training_step=5000, config=config_no_change)
    assert temp_no_change == config_no_change.temperature_init

    print("✅ get_temperature function coverage test completed!") 