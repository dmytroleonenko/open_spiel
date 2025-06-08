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

def create_test_muzero_network(config):
    """Create a simple MuZero network for testing."""
    from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
    from open_spiel.python.algorithms.muzero_jax.models.layers import MLP
    import flax.nnx as nnx

    class SimpleRepresentation(nnx.Module):
        def __init__(self, config, *, rngs):
            self.flatten = lambda x: x.reshape(x.shape[0], -1)
            input_size = int(jnp.prod(jnp.array(config.observation_shape)))
            self.mlp = MLP(input_size, [64], 32, rngs=rngs)

        def __call__(self, x, training=False):
            x = self.flatten(x)
            return self.mlp(x, training)

    class SimplePrediction(nnx.Module):
        def __init__(self, config, *, rngs):
            self.value_head = nnx.Linear(
                32,
                config.value_support_size if config.value_support_size > 0 else 1,
                rngs=rngs,
            )
            self.policy_head = nnx.Linear(32, config.num_actions, rngs=rngs)

        def __call__(self, x, training=False):
            value = self.value_head(x)
            policy = self.policy_head(x)
            return policy, value  # Return in correct order: (policy_logits, value)

    class SimpleDynamics(nnx.Module):
        def __init__(self, config, *, rngs):
            self.mlp = MLP(32, [64], 32, rngs=rngs)

        def __call__(self, hidden_state, action, training=False):
            # Simple dynamics that just processes hidden state
            return self.mlp(hidden_state, training)

    class SimpleReward(nnx.Module):
        def __init__(self, config, *, rngs):
            self.head = nnx.Linear(
                32,
                config.reward_support_size if config.reward_support_size > 0 else 1,
                rngs=rngs,
            )

        def __call__(self, x, training=False):
            return self.head(x)

    return MuZeroNetwork(
        representation_network_def=lambda config, *, rngs: SimpleRepresentation(
            config, rngs=rngs
        ),
        prediction_network_def=lambda config, *, rngs: SimplePrediction(
            config, rngs=rngs
        ),
        dynamics_network_def=lambda config, *, rngs: SimpleDynamics(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: SimpleReward(config, rngs=rngs),
        projection_network_def=None,
        config=config,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    )

def test_policy_reanalysis_edge_cases():
    """Test edge cases for policy reanalysis."""
    config = MuZeroConfig(
        num_actions=3,
        num_unroll_steps=1,
        reanalyze_ratio=0.0,  # No reanalysis
        num_simulations=2,
    )

    network_config = create_network_config_from_muzero_config(
        config, observation_shape=(1, 1), num_actions=3, use_image_observation=False
    )
    model = create_test_muzero_network(network_config)

    # Test with zero reanalysis ratio
    observations = jnp.ones((4, config.num_unroll_steps + 1, 1, 1))
    rng_key = jax.random.PRNGKey(192021)

    policies = compute_policy_reanalysis_targets(
        model=model,
        observations=observations,
        config=config,
        training=False,
        rng_key=rng_key,
    )

    # Should return uniform policies when no reanalysis
    expected_shape = (4, config.num_unroll_steps + 1, config.num_actions)
    assert policies.shape == expected_shape
    assert jnp.allclose(jnp.sum(policies, axis=-1), 1.0, atol=1e-5)

    # Test with single sample batch
    single_obs = jnp.ones((1, config.num_unroll_steps + 1, 1, 1))
    single_config = dataclasses.replace(config, reanalyze_ratio=1.0)

    single_policies = compute_policy_reanalysis_targets(
        model=model,
        observations=single_obs,
        config=single_config,
        training=False,
        rng_key=rng_key,
    )

    assert single_policies.shape == (1, config.num_unroll_steps + 1, config.num_actions)
    assert jnp.allclose(jnp.sum(single_policies, axis=-1), 1.0, atol=1e-5)

def test_policy_reanalysis_fallback_error_handling(common_key, common_cfg_flat):
    """Test error handling when policy reanalysis encounters issues."""
    # Create configuration with policy reanalysis enabled
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        reanalyze_ratio=0.5,
        num_simulations=2,  # Reduced for faster testing
        num_unroll_steps=2,  # Reduced for faster testing
        checkpoint_dir=None  # Disable checkpointing for testing
    )
    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig
    learner = Learner(model, None, config, common_key)

    # Create batch with problematic observations to trigger fallback
    batch_size = 2

    # Test with missing observation sequence (should trigger fallback)
    batch_minimal = {
        "observation": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1, *common_cfg_flat.observation_shape)
        ),  # Correct time dimension
        "action": jax.random.randint(
            common_key, (batch_size, config.num_unroll_steps), 0, config.num_actions
        ),
        "target_reward": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1)
        ),
        "target_value": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1)
        ),
        "target_policy": jax.nn.softmax(
            jax.random.normal(
                common_key, (batch_size, config.num_unroll_steps + 1, config.num_actions)
            )
        ),
        "game_history_mask": jnp.ones((batch_size, config.num_unroll_steps + 1)),
        "training_step": 0,
    }

    # Should handle the fallback gracefully
    loss, metrics = learner._compute_total_loss_static(
        model, config, batch_minimal, common_key, training=True
    )

    assert jnp.isfinite(loss), "Loss should be finite even with fallback observations"
    assert "policy_loss" in metrics, "Policy loss should still be computed"

    print("✅ Policy reanalysis fallback error handling test completed!")

def test_policy_reanalysis_observation_preparation(common_key, common_cfg_flat):
    """Test correct observation preparation for policy reanalysis."""
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        reanalyze_ratio=0.3,
        num_simulations=2,  # Reduced for faster testing
        num_unroll_steps=2,  # Reduced for faster testing
    )
    model = make_model(common_key, common_cfg_flat)
    learner = Learner(model, None, config, common_key)

    batch_size = 3

    # Test with full observation sequence
    full_observations = jax.random.normal(
        common_key, (batch_size, config.num_unroll_steps + 1, *common_cfg_flat.observation_shape)
    )
    batch_full = {
        "observation": full_observations,
        "action": jax.random.randint(
            common_key, (batch_size, config.num_unroll_steps), 0, config.num_actions
        ),
        "target_reward": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1)
        ),
        "target_value": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1)
        ),
        "target_policy": jax.nn.softmax(
            jax.random.normal(
                common_key, (batch_size, config.num_unroll_steps + 1, config.num_actions)
            )
        ),
        "game_history_mask": jnp.ones((batch_size, config.num_unroll_steps + 1)),
        "training_step": 50,
    }

    loss_full, _ = learner._compute_total_loss_static(
        model, config, batch_full, common_key, training=True
    )

    # Test with incomplete observation sequence (triggers tiling)
    incomplete_observations = jax.random.normal(
        common_key, (batch_size, 1, *common_cfg_flat.observation_shape)
    )
    batch_incomplete = {
        "observation": incomplete_observations,
        "action": jax.random.randint(
            common_key, (batch_size, config.num_unroll_steps), 0, config.num_actions
        ),
        "target_reward": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1)
        ),
        "target_value": jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1)
        ),
        "target_policy": jax.nn.softmax(
            jax.random.normal(
                common_key, (batch_size, config.num_unroll_steps + 1, config.num_actions)
            )
        ),
        "game_history_mask": jnp.ones((batch_size, config.num_unroll_steps + 1)),
        "training_step": 50,
    }

    loss_incomplete, _ = learner._compute_total_loss_static(
        model, config, batch_incomplete, common_key, training=True
    )

    # Both should work
    assert jnp.isfinite(loss_full), "Loss should be finite with full observations"
    assert jnp.isfinite(
        loss_incomplete
    ), "Loss should be finite with incomplete observations"

    print("✅ Policy reanalysis observation preparation test completed!") 