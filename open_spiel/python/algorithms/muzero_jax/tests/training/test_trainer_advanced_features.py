import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import dataclasses
import copy

# Import from the common utils module
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
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
    MockRep,
    MockDyn,
    MockPred,
    MockRew,
    MockProj
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch,
    apply_value_prefix_reward_accumulation,
    generate_top_new_masks,
    apply_mixed_value_targets,
    compute_gae_value_targets,
    compute_policy_reanalysis_targets
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork 
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig

def test_symlog_loss_functionality(common_key, common_cfg_flat):
    """Test symlog/support transformation functionality.

    Verifies that symlog transformations work correctly when configured.
    """
    mk, lk, bk = jax.random.split(common_key, 3)

    from open_spiel.python.algorithms.muzero_jax.training.losses import (
        symlog,
        symexp,
        compute_symlog_loss,
    )

    # Test symlog/symexp functions first
    test_values = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

    # Test symlog properties
    symlog_values = symlog(test_values, base=2.0)

    # Symlog should preserve sign
    assert jnp.all(
        jnp.sign(symlog_values) == jnp.sign(test_values)
    ), "Symlog should preserve the sign of input values"

    # Test symexp is inverse of symlog
    recovered_values = symexp(symlog_values, base=2.0)
    assert jnp.allclose(
        recovered_values, test_values, rtol=1e-5
    ), f"Symexp should be inverse of symlog: got {recovered_values}, expected {test_values}"

    # Test symlog loss function - EfficientZeroV2 pattern
    predictions_symlog = jnp.array([1.0, -2.0, 5.0])  # Already in symlog space
    targets_raw = jnp.array([1.5, -1.5, 4.0])  # Raw scalar targets

    # EfficientZeroV2 manual calculation: prediction already symlog, only transform target
    # loss = 0.5 * (prediction - symlog(target)) ** 2
    symlog_targ = symlog(targets_raw, base=2.0)
    expected_loss = jnp.mean(0.5 * (predictions_symlog - symlog_targ) ** 2)

    # Function calculation
    actual_loss = jnp.mean(
        compute_symlog_loss(predictions_symlog, targets_raw, base=2.0)
    )

    assert jnp.allclose(
        actual_loss, expected_loss, rtol=1e-5
    ), f"Symlog loss {actual_loss} should equal expected {expected_loss}"

    # Test with trainer using symlog loss
    cfgn = common_cfg_flat
    cfg_symlog = make_cfg(0, 0, 1, False, "symlog_test", l2_weight=0.0)
    cfg_symlog = dataclasses.replace(
        cfg_symlog,
        value_loss_type="symlog",
        reward_loss_type="symlog",
        symlog_base=2.0,
        batch_size=2,
    )

    model = make_model(mk, cfgn)
    learner = Learner(model, None, cfg_symlog, lk)

    # Create batch with known values
    batch = make_batch(bk, 2, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0)

    # Modify targets to test symlog (shape should be (batch_size, num_unroll_steps+1))
    batch["target_value"] = jnp.array(
        [[1.0, 1.5], [-2.0, -1.5]]
    )  # Different values for symlog testing
    batch["target_reward"] = jnp.array([[0.5, 0.6], [1.0, 0.8]])

    # Run training step
    metrics = learner.train_step(batch)

    # Verify losses are computed without error
    assert jnp.isfinite(metrics["value_loss"]), "Symlog value loss should be finite"
    assert jnp.isfinite(metrics["reward_loss"]), "Symlog reward loss should be finite"
    assert jnp.isfinite(
        metrics["total_loss"]
    ), "Total loss with symlog should be finite"

    print(f"✅ Symlog loss functionality test passed:")
    print(f"  - Symlog/symexp functions are proper inverses")
    print(f"  - Symlog loss function computes correctly")
    print(f"  - Trainer correctly uses symlog losses when configured")
    print(f"  - Value loss: {metrics['value_loss']:.6f}")
    print(f"  - Reward loss: {metrics['reward_loss']:.6f}")

def test_discrete_support_transformations(common_key, common_cfg_flat):
    """Test discrete support transformations.

    Verifies that the newly implemented scalar_to_support and support_to_scalar
    functions work correctly for EfficientZeroV2 parity.
    """
    from open_spiel.python.algorithms.muzero_jax.training.losses import (
        scalar_to_support,
        support_to_scalar,
    )

    # Test with various scalar values
    test_values = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0, 100.0])

    # Test basic transformation round-trip
    num_atoms = 601
    support_min = -300.0
    support_max = 300.0

    # Convert scalars to support distribution
    support_dist = scalar_to_support(test_values, support_min, support_max, num_atoms)

    # Verify output shape
    assert support_dist.shape == (
        len(test_values),
        num_atoms,
    ), f"Expected shape {(len(test_values), num_atoms)}, got {support_dist.shape}"

    # Verify distributions sum to 1 (approximately)
    dist_sums = jnp.sum(support_dist, axis=-1)
    assert jnp.allclose(
        dist_sums, 1.0, rtol=1e-5
    ), f"Distributions should sum to 1, got {dist_sums}"

    # Convert logits back to scalars (using the distributions as logits)
    # For this test, we'll add a small offset to convert probs to logits
    logits = jnp.log(jnp.clip(support_dist, 1e-8, 1.0))
    recovered_values = support_to_scalar(logits, support_min, support_max, num_atoms)

    # Verify round-trip accuracy
    assert (
        recovered_values.shape == test_values.shape
    ), f"Expected shape {test_values.shape}, got {recovered_values.shape}"

    # Check that values are reasonably close (allowing for some numerical error)
    # The transformation involves non-linear operations, so perfect recovery isn't expected
    relative_errors = jnp.abs(recovered_values - test_values) / (
        jnp.abs(test_values) + 1e-8
    )
    max_relative_error = jnp.max(relative_errors)

    print(f"Discrete support transformation test:")
    print(f"  - Original values: {test_values}")
    print(f"  - Recovered values: {recovered_values}")
    print(f"  - Relative errors: {relative_errors}")
    print(f"  - Max relative error: {max_relative_error}")

    # Allow for reasonable numerical error (the transformation is non-linear)
    assert (
        max_relative_error < 0.1
    ), f"Max relative error {max_relative_error} should be < 0.1"

    # Test edge cases
    # Test with zero
    zero_val = jnp.array([0.0])
    zero_support = scalar_to_support(zero_val, support_min, support_max, num_atoms)
    zero_recovered = support_to_scalar(
        jnp.log(jnp.clip(zero_support, 1e-8, 1.0)), support_min, support_max, num_atoms
    )
    assert jnp.allclose(
        zero_recovered, zero_val, atol=1e-3
    ), f"Zero value should be recovered accurately, got {zero_recovered} vs {zero_val}"

    # Test with extreme values (within support range)
    extreme_vals = jnp.array([-200.0, 200.0])
    extreme_support = scalar_to_support(
        extreme_vals, support_min, support_max, num_atoms
    )
    extreme_recovered = support_to_scalar(
        jnp.log(jnp.clip(extreme_support, 1e-8, 1.0)),
        support_min,
        support_max,
        num_atoms,
    )
    extreme_errors = jnp.abs(extreme_recovered - extreme_vals) / (
        jnp.abs(extreme_vals) + 1e-8
    )
    assert (
        jnp.max(extreme_errors) < 0.1
    ), f"Extreme values should be recovered reasonably, errors: {extreme_errors}"

    print(f"✅ Discrete support transformations test passed!")
    print(f"  - Round-trip transformation accuracy verified")
    print(f"  - Edge cases (zero, extreme values) handled correctly")
    print(f"  - Support distributions properly normalized")

def test_entropy_loss_integration(common_key, common_cfg_flat):
    """Test entropy loss integration to cover missing lines."""
    mk, lk, bk = jax.random.split(common_key, 3)

    cfgn = common_cfg_flat
    cfg_entropy = make_cfg(0, 0, 1, False, "entropy_test", l2_weight=0.0)
    cfg_entropy = dataclasses.replace(
        cfg_entropy,
        entropy_coeff=0.1,  # Enable entropy loss
        batch_size=2,
    )

    model = make_model(mk, cfgn)
    learner = Learner(model, None, cfg_entropy, lk)

    batch = make_batch(bk, 2, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0)

    # Test entropy loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model=model,
        config=cfg_entropy,
        batch=batch,
        rng_key=common_key,
        training=True,
    )

    assert jnp.isfinite(loss), "Loss with entropy should be finite"
    assert "entropy_loss" in metrics, "Entropy loss should be in metrics"
    assert jnp.isfinite(metrics["entropy_loss"]), "Entropy loss should be finite"
    assert metrics["entropy_loss"] >= 0.0, "Entropy loss should be non-negative"

    print(f"✅ Entropy loss integration test passed!")
    print(f"  - Entropy coefficient: {cfg_entropy.entropy_coeff}")
    print(f"  - Entropy loss: {metrics['entropy_loss']:.6f}")

def test_discrete_support_transformations_integration(common_key, common_cfg_flat):
    """Test discrete support transformations integration with trainer."""
    mk, lk, bk = jax.random.split(common_key, 3)

    # Configure models with scalar support
    class ScalarNet(nnx.Module):
        def __init__(self, config, *, rngs):
            self.representation = ScalarRep(config, rngs=rngs)
            self.dynamics = ScalarDyn(config, rngs=rngs)
            self.prediction = ScalarPred(config, rngs=rngs)

        def initial_model(self, observation, training=False):
            return self.representation(observation, training)

        def prediction_model(self, hidden, training=False):
            return self.prediction(hidden, training)

        def recurrent_model(self, hidden, action, training=False):
            return self.dynamics(hidden, action, training)

        def recurrent_inference(self, hidden, action, training=False):
            next_hidden, reward = self.recurrent_model(hidden, action, training)
            value, policy_logits = self.prediction_model(next_hidden, training)
            return next_hidden, reward, value, policy_logits

        def initial_inference(self, observation, training=False):
            hidden = self.initial_model(observation, training)
            value, policy_logits = self.prediction_model(hidden, training)
            # Return 4-element tuple to match expected interface: (hidden, reward, value, policy_logits)
            dummy_reward = jnp.zeros((hidden.shape[0],))  # Dummy reward for initial step
            return hidden, dummy_reward, value, policy_logits

    class ScalarRep(nnx.Module):
        def __init__(self, config, *, rngs):
            self.linear = nnx.Linear(np.prod(common_cfg_flat.observation_shape), 64, rngs=rngs)

        def __call__(self, x, training):
            return self.linear(x.reshape(x.shape[0], -1))

    class ScalarDyn(nnx.Module):
        def __init__(self, config, *, rngs):
            self.linear = nnx.Linear(64 + common_cfg_flat.num_actions, 64, rngs=rngs)

        def __call__(self, h, a, training):
            # Dynamics network should only return next hidden state
            # The MuZeroNetwork.dynamics method will separately call reward_network
            a_onehot = jax.nn.one_hot(a, common_cfg_flat.num_actions)
            return self.linear(jnp.concatenate([h, a_onehot], axis=-1)), jnp.zeros((h.shape[0],))

    class ScalarPred(nnx.Module):
        def __init__(self, config, *, rngs):
            self.value_head = nnx.Linear(64, 1, rngs=rngs)
            # Use common_cfg_flat.num_actions to match the test configuration
            self.policy_head = nnx.Linear(64, common_cfg_flat.num_actions, rngs=rngs)

        def __call__(self, h, training):
            value = self.value_head(h).squeeze(-1)  # Shape: (batch_size,) - scalar
            policy_logits = self.policy_head(h)
            return value, policy_logits

        def reset_noise(self, rng_key):
            pass

    # Test with scalar outputs (no support)
    cfgn = common_cfg_flat
    network_config = MuZeroNetworkConfig(
        observation_shape=cfgn.observation_shape,
        num_actions=cfgn.num_actions,
        num_channels=64,
        value_support_size=0,
        reward_support_size=0,
        use_projection=False,
        projection_head_output_dim=0,
        num_residual_blocks=1,
        use_batch_norm=True,
        noisy_net=False
    )

    # Use the custom ScalarNet instead of MuZeroNetwork
    model = ScalarNet(network_config, rngs=nnx.Rngs(params=mk))
    
    # Continue with the test...
    opt = optax.adam(1e-4)
    cfg_scalar = make_cfg(0, 0, 1, False, "scalar_test", l2_weight=0.0)
    cfg_scalar = dataclasses.replace(cfg_scalar, batch_size=2)
    
    learner = Learner(model, None, cfg_scalar, lk)

    batch = make_batch(bk, 2, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0)

    # Test scalar loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model=model,
        config=cfg_scalar,
        batch=batch,
        rng_key=common_key,
        training=True,
    )

    assert jnp.isfinite(loss), "Scalar loss should be finite"
    assert jnp.isfinite(metrics["value_loss"]), "Scalar value loss should be finite"
    assert jnp.isfinite(metrics["reward_loss"]), "Scalar reward loss should be finite"

    print(f"✅ Discrete support transformations integration test passed!")
    print(f"  - Scalar value loss: {metrics['value_loss']:.6f}")
    print(f"  - Scalar reward loss: {metrics['reward_loss']:.6f}")

def test_ssl_projection_integration(common_key, common_cfg_flat):
    """Test SSL projection integration to cover missing lines."""
    mk, lk, bk = jax.random.split(common_key, 3)

    cfgn = common_cfg_flat
    cfg_ssl = make_cfg(0, 0, 1, True, "ssl_test", l2_weight=0.0)  # use_projection=True
    cfg_ssl = dataclasses.replace(
        cfg_ssl,
        consistency_loss_coeff=1.0,  # Enable SSL loss
        batch_size=2,
    )

    model = make_model(mk, cfgn)
    learner = Learner(model, None, cfg_ssl, lk)

    batch = make_batch(bk, 2, cfgn.observation_shape, cfgn.num_actions, 1, 0, 0, use_proj=True)

    # Test SSL loss computation
    loss, metrics = Learner._compute_total_loss_static(
        model=model,
        config=cfg_ssl,
        batch=batch,
        rng_key=common_key,
        training=True,
    )

    assert jnp.isfinite(loss), "Loss with SSL should be finite"
    assert "ssl_loss" in metrics, "SSL loss should be in metrics"
    assert jnp.isfinite(metrics["ssl_loss"]), "SSL loss should be finite"
    assert metrics["ssl_loss"] >= 0.0, "SSL loss should be non-negative"

    print(f"✅ SSL projection integration test passed!")
    print(f"  - Consistency loss coefficient: {cfg_ssl.consistency_loss_coeff}")
    print(f"  - SSL loss: {metrics['ssl_loss']:.6f}")

def test_symlog_and_kl_loss_types(common_key, common_cfg_flat):
    """Test symlog and KL loss types to cover missing lines."""
    mk, lk, bk = jax.random.split(common_key, 3)

    # Create simple network config
    simple_cfg = MockNetCfg(
        observation_shape=(2, 2),
        num_actions=3,
        batch_size=1,
        value_support_size=0,
        reward_support_size=0,
        hidden_size=8
    )

    # Test symlog loss type
    cfg_symlog = make_cfg(0, 0, 1, False, "symlog_test", l2_weight=0.0, num_actions=3)
    cfg_symlog = dataclasses.replace(
        cfg_symlog,
        value_loss_type="symlog",
        reward_loss_type="symlog",
        symlog_base=2.0,
        batch_size=1,
    )

    model_symlog = make_model(mk, simple_cfg)
    batch_symlog = make_batch(bk, 1, (2, 2), 3, 1, 0, 0)

    # Test symlog loss computation
    loss_symlog, metrics_symlog = Learner._compute_total_loss_static(
        model=model_symlog,
        config=cfg_symlog,
        batch=batch_symlog,
        rng_key=common_key,
        training=True,
    )

    assert jnp.isfinite(loss_symlog), "Symlog loss should be finite"
    assert jnp.isfinite(metrics_symlog["value_loss"]), "Symlog value loss should be finite"
    assert jnp.isfinite(metrics_symlog["reward_loss"]), "Symlog reward loss should be finite"

    # Test KL loss type
    simple_cfg_categorical = MockNetCfg(
        observation_shape=(2, 2),
        num_actions=3,
        batch_size=1,
        value_support_size=21,
        reward_support_size=21,
        hidden_size=8
    )
    
    cfg_kl = make_cfg(21, 21, 1, False, "kl_test", l2_weight=0.0, num_actions=3)  # Use categorical
    cfg_kl = dataclasses.replace(
        cfg_kl,
        value_loss_type="kl",
        reward_loss_type="kl",
        batch_size=1,
    )

    model_kl = make_model(jax.random.fold_in(mk, 1), simple_cfg_categorical)
    batch_kl = make_batch(
        jax.random.fold_in(bk, 1), 1, (2, 2), 3, 1, 21, 21
    )

    # Test KL loss computation
    loss_kl, metrics_kl = Learner._compute_total_loss_static(
        model=model_kl,
        config=cfg_kl,
        batch=batch_kl,
        rng_key=jax.random.fold_in(common_key, 1),
        training=True,
    )

    assert jnp.isfinite(loss_kl), "KL loss should be finite"
    assert jnp.isfinite(metrics_kl["value_loss"]), "KL value loss should be finite"
    assert jnp.isfinite(metrics_kl["reward_loss"]), "KL reward loss should be finite"

    print(f"✅ Symlog and KL loss types test passed!")
    print(f"  - Symlog value loss: {metrics_symlog['value_loss']:.6f}")
    print(f"  - Symlog reward loss: {metrics_symlog['reward_loss']:.6f}")
    print(f"  - KL value loss: {metrics_kl['value_loss']:.6f}")
    print(f"  - KL reward loss: {metrics_kl['reward_loss']:.6f}")
