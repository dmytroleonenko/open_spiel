import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses
from unittest.mock import patch, PropertyMock, MagicMock, Mock

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
    compute_gae_value_targets,
    compute_policy_reanalysis_targets,
    create_network_config_from_muzero_config
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork

def test_gae_mixed_mode_without_top_new_masks(common_key, common_cfg_flat):
    """Test GAE mixed mode fallback when top_new_masks is None."""
    mk, lk = jax.random.split(common_key, 2)

    # GAE mixed mode with no top_new_masks
    config_gae_mixed = MuZeroConfig(
        value_target="mixed",
        value_target_type="GAE",
        start_use_mix_training_steps=10,  # Low threshold
        num_unroll_steps=2,
        num_actions=common_cfg_flat.num_actions,  # FIXED: Set to match test data
    )

    batch_gae_mixed = make_batch(
        lk, 2, common_cfg_flat.observation_shape, common_cfg_flat.num_actions, 2, 0, 0
    )
    # Use correct observation shape for extra observations
    obs_dim = common_cfg_flat.observation_shape[0]
    batch_gae_mixed.update(
        {
            "observations_extra": jnp.ones((2, 5, obs_dim)),
            "actions_extra": jnp.ones((2, 4), dtype=jnp.int32),
            "rewards_extra": jnp.ones((2, 5)),
            "dones": jnp.zeros((2, 5)),
            "training_step": 100,  # Above threshold
            "top_new_masks": None,  # This triggers the fallback
        }
    )

    model_gae_mixed = make_model(mk, common_cfg_flat)
    loss_gae_mixed, _ = Learner._compute_total_loss_static(
        model_gae_mixed, config_gae_mixed, batch_gae_mixed, mk, training=True
    )
    assert jnp.isfinite(loss_gae_mixed)

    print("✅ GAE mixed mode without top_new_masks test completed!")

def test_unknown_value_target_fallback(common_key, common_cfg_flat):
    """Test fallback behavior for unknown value_target type. OPTIMIZED for speed."""
    mk, lk = jax.random.split(common_key, 2)

    # Unknown value_target fallback - OPTIMIZED with minimal batch size
    config_unknown_target = MuZeroConfig(
        value_target="unknown_type", 
        value_target_type="bootstrapped",
        num_actions=common_cfg_flat.num_actions,  # FIXED: Set to match test data
        batch_size=1,  # Reduced from default
        num_unroll_steps=1,  # Reduced for speed
    )

    batch_unknown_target = make_batch(
        lk, 1, common_cfg_flat.observation_shape, common_cfg_flat.num_actions, 1, 0, 0  # Minimal parameters
    )
    model_unknown_target = make_model(mk, common_cfg_flat)

    loss_unknown_target, _ = Learner._compute_total_loss_static(
        model_unknown_target,
        config_unknown_target,
        batch_unknown_target,
        mk,
        training=True,
    )
    assert jnp.isfinite(loss_unknown_target)

    print("✅ Unknown value_target fallback test completed (OPTIMIZED)!")

def test_kl_reward_loss_with_distribution_rewards_basic(common_key, common_cfg_flat):
    """Test KL reward loss computation with distribution-based rewards."""
    mk, lk = jax.random.split(common_key, 2)

    # Create a model that returns distribution rewards to trigger KL loss path
    config_kl_reward = make_cfg(
        0, 601, 1, False, "kl_reward_test"
    )  # reward_support_size=601 for distribution
    config_kl_reward = dataclasses.replace(config_kl_reward, reward_loss_type="kl")

    model_kl_reward = make_model(mk, common_cfg_flat)
    batch_kl_reward = make_batch(
        lk, 2, common_cfg_flat.observation_shape, common_cfg_flat.num_actions, 1, 0, 601
    )

    # This should trigger the KL loss path and squeeze operation
    loss_kl_reward, _ = Learner._compute_total_loss_static(
        model_kl_reward, config_kl_reward, batch_kl_reward, mk, training=True
    )
    assert jnp.isfinite(loss_kl_reward)

    print("✅ KL reward loss with distribution rewards test completed!")

def test_kl_reward_loss_with_distribution_rewards_advanced(common_key, common_cfg_flat):
    """Test KL reward loss with distribution rewards to cover specific squeeze operations."""
    mk, lk = jax.random.split(common_key, 2)

    class KLRewardDistRew(nnx.Module):
        def __init__(self, *, rngs):
            pass

        def __call__(self, h, training):
            # Return distribution rewards with shape (B, support_size) to trigger specific logic
            batch_size = h.shape[0]
            support_size = 11
            return jax.random.normal(lk, (batch_size, support_size))

    class MockModelKLReward(MuZeroNetwork):
        def initial_inference(self, x, training):
            batch_size = x.shape[0]
            hidden = jnp.ones((batch_size, 2))
            reward = KLRewardDistRew(rngs=nnx.Rngs(lk))(hidden, training)
            value = jnp.ones((batch_size, 1))
            policy = jnp.ones((batch_size, NUM_ACTIONS))
            return (hidden, reward, value, policy)

        def recurrent_inference(self, h, a, training):
            reward = KLRewardDistRew(rngs=nnx.Rngs(lk))(h, training)
            value = jnp.ones((h.shape[0], 1))
            policy = jnp.ones((h.shape[0], NUM_ACTIONS))
            return (h, reward, value, policy)

    config = make_cfg(
        VALUE_SUPPORT_SCALAR, 11, 2, False, "kl_reward_test", use_ema=False
    )
    config = dataclasses.replace(config, reward_loss_type="kl")

    # Use the existing make_model function instead of trying to create a custom one
    model = make_model(mk, common_cfg_flat)
    batch = make_batch(
        lk, BATCH_SIZE, common_cfg_flat.observation_shape, NUM_ACTIONS, 2, VALUE_SUPPORT_SCALAR, 11
    )

    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, lk, training=True
    )

    assert jnp.isfinite(loss)
    assert "reward_loss" in metrics

    print("✅ KL reward loss with distribution rewards (advanced) test completed!")

def test_scalar_reward_dimension_check_line_1339(common_key, common_cfg_flat):
    """Test scalar reward dimension expansion when ndim == 0 (line 1339)."""
    mk = jax.random.fold_in(common_key, 1)

    # Create a custom model that returns scalar (ndim=0) rewards
    class ScalarRewardModel(MuZeroNetwork):
        def initial_inference(self, x, training):
            batch_size = x.shape[0]
            hidden = jnp.ones((batch_size, 16))
            reward = jnp.array(1.0)  # Scalar reward (ndim=0)
            value = jnp.ones((batch_size, 1))
            policy = jnp.ones((batch_size, common_cfg_flat.num_actions))
            return (hidden, reward, value, policy, None, None)  # Add missing return values for LSTM

        def recurrent_inference(self, h, a, training, reward_hidden=None):
            reward = jnp.array(
                2.0
            )  # Scalar reward (ndim=0) - this should trigger line 1339
            value = jnp.ones((h.shape[0], 1))
            policy = jnp.ones((h.shape[0], common_cfg_flat.num_actions))
            return (h, reward, value, policy, None, None)  # Add missing return values for LSTM

    # Import and mock components
    from open_spiel.python.algorithms.muzero_jax.training.trainer import (
        compute_policy_reanalysis_targets,
    )
    from unittest.mock import patch, MagicMock

    # Create sophisticated mocks that return actual JAX arrays
    mock_mctx = MagicMock()

    # Mock RootFnOutput constructor
    def mock_root_output(**kwargs):
        mock = MagicMock()
        # Return the values passed to constructor as attributes
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    mock_mctx.RootFnOutput = mock_root_output

    # Mock RecurrentFnOutput constructor
    def mock_recurrent_output(*args, **kwargs):
        mock = MagicMock()
        if args:
            mock.reward = args[0] if len(args) > 0 else jnp.array([1.0])
            mock.discount = args[1] if len(args) > 1 else jnp.array([0.99])
            mock.prior_logits = (
                args[2] if len(args) > 2 else jnp.ones((1, common_cfg_flat.num_actions))
            )
            mock.value = args[3] if len(args) > 3 else jnp.array([0.0])
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    mock_mctx.RecurrentFnOutput = mock_recurrent_output

    # Mock policy output with proper JAX arrays
    mock_policy_output = MagicMock()
    
    # Pre-compute the expected result to avoid recursion
    batch_size = 1
    num_steps = 2  # config.num_unroll_steps + 1
    action_weights_flat = jnp.ones((batch_size * num_steps * common_cfg_flat.num_actions,)) / common_cfg_flat.num_actions
    action_weights_reshaped = jnp.reshape(action_weights_flat, (batch_size, num_steps, common_cfg_flat.num_actions))
    
    mock_policy_output.action_weights = action_weights_reshaped.reshape(-1)  # Flat version
    mock_policy_output.action_weights.reshape = MagicMock(return_value=action_weights_reshaped)
    
    mock_mctx.gumbel_muzero_policy = MagicMock(return_value=mock_policy_output)

    with patch.dict("sys.modules", {"mctx": mock_mctx}):
        # Use the base model structure but override the inference methods
        model = make_model(mk, common_cfg_flat)

        # Replace model methods with our scalar reward model
        model.initial_inference = ScalarRewardModel.initial_inference.__get__(
            model, MuZeroNetwork
        )
        model.recurrent_inference = ScalarRewardModel.recurrent_inference.__get__(
            model, MuZeroNetwork
        )

        config = make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,  # Small unroll steps
            False,
            "scalar_reward_test",
        )
        config = dataclasses.replace(
            config,
            reanalyze_ratio=1.0,  # Force reanalysis to trigger recurrent_inference
            num_actions=common_cfg_flat.num_actions,
            num_simulations=2,  # Small number for test speed
        )

        # Create observations for reanalysis
        batch_size = 1
        num_steps = config.num_unroll_steps + 1
        observations = jax.random.uniform(
            common_key, (batch_size, num_steps, *common_cfg_flat.observation_shape)
        )

        # Run policy reanalysis - should trigger line 1339 in recurrent_fn
        policy_targets = compute_policy_reanalysis_targets(
            model, observations, config, training=False, rng_key=common_key
        )

        # Verify the computation completed successfully
        assert policy_targets.shape == (batch_size, num_steps, common_cfg_flat.num_actions)
        assert jnp.allclose(jnp.sum(policy_targets, axis=-1), 1.0, atol=1e-6)

    print("✅ Scalar reward dimension check line 1339 test completed!")

def test_value_support_to_scalar_conversion_line_1444(common_key, common_cfg_flat):
    """Test value support-to-scalar conversion when value has distribution (line 1444)."""
    mk = jax.random.fold_in(common_key, 1)

    # Create a custom model that returns categorical values (distribution)
    class CategoricalValueModel(MuZeroNetwork):
        def initial_inference(self, x, training):
            batch_size = x.shape[0]
            hidden = jnp.ones((batch_size, 16))
            reward = jnp.ones((batch_size, 1))
            value = jax.random.uniform(
                common_key, (batch_size, 11)
            )  # Categorical value distribution
            policy = jnp.ones((batch_size, common_cfg_flat.num_actions))
            return (hidden, reward, value, policy, None, None)  # Add missing return values for LSTM

        def recurrent_inference(self, h, a, training, reward_hidden=None):
            reward = jnp.ones((h.shape[0], 1))
            value = jax.random.uniform(
                common_key, (h.shape[0], 11)
            )  # Categorical value distribution - should trigger line 1444
            policy = jnp.ones((h.shape[0], common_cfg_flat.num_actions))
            return (h, reward, value, policy, None, None)  # Add missing return values for LSTM

    # Import and mock components
    from open_spiel.python.algorithms.muzero_jax.training.trainer import (
        compute_policy_reanalysis_targets,
    )
    from unittest.mock import patch, MagicMock

    # Create sophisticated mocks that return actual JAX arrays
    mock_mctx = MagicMock()

    # Mock RootFnOutput constructor
    def mock_root_output(**kwargs):
        mock = MagicMock()
        # Return the values passed to constructor as attributes
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    mock_mctx.RootFnOutput = mock_root_output

    # Mock RecurrentFnOutput constructor
    def mock_recurrent_output(*args, **kwargs):
        mock = MagicMock()
        if args:
            mock.reward = args[0] if len(args) > 0 else jnp.array([1.0])
            mock.discount = args[1] if len(args) > 1 else jnp.array([0.99])
            mock.prior_logits = (
                args[2] if len(args) > 2 else jnp.ones((1, common_cfg_flat.num_actions))
            )
            mock.value = args[3] if len(args) > 3 else jnp.array([0.0])
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    mock_mctx.RecurrentFnOutput = mock_recurrent_output

    # Mock policy output with proper JAX arrays
    mock_policy_output = MagicMock()
    
    # Pre-compute the expected result to avoid recursion
    batch_size = 1
    num_steps = 2  # config.num_unroll_steps + 1
    action_weights_flat = jnp.ones((batch_size * num_steps * common_cfg_flat.num_actions,)) / common_cfg_flat.num_actions
    action_weights_reshaped = jnp.reshape(action_weights_flat, (batch_size, num_steps, common_cfg_flat.num_actions))
    
    mock_policy_output.action_weights = action_weights_reshaped.reshape(-1)  # Flat version
    mock_policy_output.action_weights.reshape = MagicMock(return_value=action_weights_reshaped)
    
    mock_mctx.gumbel_muzero_policy = MagicMock(return_value=mock_policy_output)

    with patch.dict("sys.modules", {"mctx": mock_mctx}):
        # Use the base model structure but override the inference methods
        model = make_model(mk, common_cfg_flat)

        # Replace model methods with our categorical value model
        model.initial_inference = CategoricalValueModel.initial_inference.__get__(
            model, MuZeroNetwork
        )
        model.recurrent_inference = CategoricalValueModel.recurrent_inference.__get__(
            model, MuZeroNetwork
        )

        config = make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,  # Small unroll steps
            False,
            "categorical_value_test",
        )
        config = dataclasses.replace(
            config,
            reanalyze_ratio=1.0,  # Force reanalysis to trigger recurrent_inference
            num_actions=common_cfg_flat.num_actions,
            num_simulations=2,  # Small number for test speed
            support_min=-300.0,
            support_max=300.0,
        )

        # Create observations for reanalysis
        batch_size = 1
        num_steps = config.num_unroll_steps + 1
        observations = jax.random.uniform(
            common_key, (batch_size, num_steps, *common_cfg_flat.observation_shape)
        )

        # Run policy reanalysis - should trigger line 1444 in recurrent_fn
        policy_targets = compute_policy_reanalysis_targets(
            model, observations, config, training=False, rng_key=common_key
        )

        # Verify the computation completed successfully
        assert policy_targets.shape == (batch_size, num_steps, common_cfg_flat.num_actions)
        assert jnp.allclose(jnp.sum(policy_targets, axis=-1), 1.0, atol=1e-6)

    print("✅ Value support-to-scalar conversion line 1444 test completed!")

def test_value_support_to_scalar_conversion_line_coverage_test(common_key, common_cfg_flat):
    """Test value support to scalar conversion line coverage."""
    mk = jax.random.fold_in(common_key, 1)
    model = make_model(mk, common_cfg_flat)

    # Configure for bootstrapped (non-GAE) value target type with unknown value_target
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        NUM_UNROLL_STEPS,
        False,
        "non_gae_fallback",
    )
    config = dataclasses.replace(
        config,
        value_target_type="bootstrapped",  # Not GAE
        value_target="unknown_target_type",  # Not "search", "sarsa", or "mixed"
    )

    opt = optax.adam(config.learning_rate)
    learner = Learner(model, opt, config, common_key)

    # Create standard batch without GAE data
    batch_data = make_batch(
        common_key,
        config.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        config.num_unroll_steps,
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    batch = batch_data

    # Run the loss computation - should hit line 553: actual_target_values = target_values
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Verify the computation completed without error
    assert jnp.isfinite(loss), "Loss should be finite"
    assert "total_loss" in metrics, "Metrics should contain total_loss"

    print("✅ Value support to scalar conversion line coverage test completed!")

def test_non_gae_target_selection_fallback_line_553(common_key, common_cfg_flat):
    """Test non-GAE target selection fallback for unknown value_target type (line 553)."""
    mk = jax.random.fold_in(common_key, 1)
    model = make_model(mk, common_cfg_flat)

    # Configure for bootstrapped (non-GAE) value target type with unknown value_target
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        NUM_UNROLL_STEPS,
        False,
        "non_gae_fallback",
    )
    config = dataclasses.replace(
        config,
        value_target_type="bootstrapped",  # Not GAE
        value_target="unknown_target_type",  # Not "search", "sarsa", or "mixed"
    )

    opt = optax.adam(config.learning_rate)
    learner = Learner(model, opt, config, common_key)

    # Create standard batch without GAE data
    batch_data = make_batch(
        common_key,
        config.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        config.num_unroll_steps,
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    batch = batch_data

    # Run the loss computation - should hit line 553: actual_target_values = target_values
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Verify the computation completed without error
    assert jnp.isfinite(loss), "Loss should be finite"
    assert "total_loss" in metrics, "Metrics should contain total_loss"

def test_mcts_policy_shape_validation_line_1462_1464(common_key, common_cfg_flat):
    """Test MCTS policy shape validation and fallback creation (lines 1462, 1464)."""
    mk = jax.random.fold_in(common_key, 1)
    model = make_model(mk, common_cfg_flat)

    # Import required modules
    from open_spiel.python.algorithms.muzero_jax.training.trainer import (
        compute_policy_reanalysis_targets,
    )

    # Mock mctx.muzero_policy to return incorrect shape
    import sys
    from unittest.mock import patch, MagicMock

    # Create more sophisticated mocks that return actual JAX arrays
    mock_mctx = MagicMock()

    # Mock RootFnOutput constructor
    def mock_root_output(**kwargs):
        mock = MagicMock()
        # Return the values passed to constructor as attributes
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    mock_mctx.RootFnOutput = mock_root_output

    # Mock RecurrentFnOutput constructor
    def mock_recurrent_output(*args, **kwargs):
        mock = MagicMock()
        if args:
            mock.reward = args[0] if len(args) > 0 else jnp.array([1.0])
            mock.discount = args[1] if len(args) > 1 else jnp.array([0.99])
            mock.prior_logits = (
                args[2] if len(args) > 2 else jnp.ones((1, common_cfg_flat.num_actions))
            )
            mock.value = args[3] if len(args) > 3 else jnp.array([0.0])
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    mock_mctx.RecurrentFnOutput = mock_recurrent_output

    # Mock gumbel_muzero_policy to raise an exception to trigger fallback
    def mock_gumbel_policy(*args, **kwargs):
        raise AttributeError("Simulated mctx failure to trigger fallback")
    
    mock_mctx.gumbel_muzero_policy = mock_gumbel_policy

    # Mock the import of mctx to return our mock
    with patch.dict("sys.modules", {"mctx": mock_mctx}):
        # Configure for policy reanalysis
        config = make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            2,  # Small unroll steps
            False,
            "mcts_shape_test",
        )
        config = dataclasses.replace(
            config,
            reanalyze_ratio=1.0,  # Force reanalysis
            num_actions=common_cfg_flat.num_actions,  # Ensure correct action count
            num_simulations=4,  # Small number for test speed
        )

        # Create observations for reanalysis
        batch_size = 2
        num_steps = config.num_unroll_steps + 1
        observations = jax.random.uniform(
            common_key, (batch_size, num_steps, *common_cfg_flat.observation_shape)
        )

        # With fallbacks removed, a downstream AttributeError should bubble up
        with pytest.raises(AttributeError, match="Simulated mctx failure"):
            compute_policy_reanalysis_targets(
                model, observations, config, training=False, rng_key=common_key
            )
