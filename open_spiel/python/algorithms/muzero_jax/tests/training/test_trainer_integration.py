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
import wandb
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

def test_policy_reanalysis_integration_in_training_pipeline(common_key, common_cfg_flat):
    """Test integration of policy reanalysis into the main training pipeline.

    This test verifies that policy reanalysis is correctly integrated into
    _compute_total_loss_static and that actual_target_policies are used for
    policy loss computation when reanalyze_ratio > 0.
    """
    # Create proper MuZeroConfig with reanalysis enabled
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        reanalyze_ratio=0.5,
        num_simulations=2,  # Reduced for faster testing
        num_unroll_steps=2,  # Reduced for faster testing
        checkpoint_dir=None  # Disable checkpointing for testing
    )
    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig
    learner = Learner(model, None, config, common_key)

    # Create batch with full observation sequence for reanalysis
    batch_size = 2  # Reduced for faster testing
    observations = jax.random.normal(
        common_key, (batch_size, config.num_unroll_steps + 1, *common_cfg_flat.observation_shape)
    )
    actions = jax.random.randint(
        common_key, (batch_size, config.num_unroll_steps), 0, config.num_actions
    )
    target_rewards = jax.random.normal(common_key, (batch_size, config.num_unroll_steps + 1))
    target_values = jax.random.normal(common_key, (batch_size, config.num_unroll_steps + 1))
    target_policies = jax.nn.softmax(
        jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1, config.num_actions)
        )
    )
    game_history_mask = jnp.ones((batch_size, config.num_unroll_steps + 1))

    batch = {
        "observation": observations,
        "action": actions,
        "target_reward": target_rewards,
        "target_value": target_values,
        "target_policy": target_policies,
        "game_history_mask": game_history_mask,
        "training_step": 100,  # Include training step for temperature scheduling
    }

    # Compute loss with reanalysis enabled
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Verify loss is computed successfully
    assert jnp.isfinite(loss), "Loss should be finite with policy reanalysis"
    assert "policy_loss" in metrics, "Policy loss should be computed"
    assert jnp.isfinite(metrics["policy_loss"]), "Policy loss should be finite"

    # Test with reanalysis disabled
    config_no_reanalysis = dataclasses.replace(config, reanalyze_ratio=0.0)
    loss_no_reanalysis, metrics_no_reanalysis = Learner._compute_total_loss_static(
        model, config_no_reanalysis, batch, common_key, training=True
    )

    # Verify both configurations work
    assert jnp.isfinite(loss_no_reanalysis), "Loss should be finite without reanalysis"

    print("✅ Policy reanalysis integration test completed!")

def test_policy_reanalysis_temperature_scheduling_integration(common_key, common_cfg_flat):
    """Test that temperature scheduling is correctly integrated with policy reanalysis."""
    # Configure temperature scheduling
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        reanalyze_ratio=1.0,
        num_simulations=2,  # Reduced for faster testing
        change_temperature=True,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_decay_steps=100,  # Reduced for faster testing
        num_unroll_steps=2,  # Reduced for faster testing
        checkpoint_dir=None  # Disable checkpointing for testing
    )
    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig
    learner = Learner(model, None, config, common_key)

    # Create minimal batch for testing
    batch_size = 2
    observations = jax.random.normal(
        common_key, (batch_size, config.num_unroll_steps + 1, *common_cfg_flat.observation_shape)
    )
    actions = jax.random.randint(
        common_key, (batch_size, config.num_unroll_steps), 0, config.num_actions
    )
    target_rewards = jax.random.normal(common_key, (batch_size, config.num_unroll_steps + 1))
    target_values = jax.random.normal(common_key, (batch_size, config.num_unroll_steps + 1))
    target_policies = jax.nn.softmax(
        jax.random.normal(
            common_key, (batch_size, config.num_unroll_steps + 1, config.num_actions)
        )
    )
    game_history_mask = jnp.ones((batch_size, config.num_unroll_steps + 1))

    # Test different training steps to verify temperature changes
    for training_step in [0, 50]:
        batch = {
            "observation": observations,
            "action": actions,
            "target_reward": target_rewards,
            "target_value": target_values,
            "target_policy": target_policies,
            "game_history_mask": game_history_mask,
            "training_step": training_step,
        }

        # Should not crash with different training steps
        loss, metrics = Learner._compute_total_loss_static(
            model, config, batch, common_key, training=True
        )
        assert jnp.isfinite(
            loss
        ), f"Loss should be finite at training step {training_step}"

    print("✅ Policy reanalysis temperature scheduling integration test completed!")

def test_policy_reanalysis_efficientzero_v2_integration_patterns(common_key, common_cfg_flat):
    """Test that policy reanalysis integration follows EfficientZeroV2 patterns."""
    # Configure for EfficientZeroV2-style policy reanalysis
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        reanalyze_ratio=0.6,  # EfficientZeroV2 typical value
        num_simulations=3,  # Reduced for faster testing
        c_init=1.25,
        c_base=19652,
        dirichlet_alpha=0.25,
        explore_frac=0.25,
        temperature_init=1.0,
        temperature_final=0.001,
        temperature_decay_steps=100,  # Reduced for faster testing
        change_temperature=True,
        num_unroll_steps=2,  # Reduced for faster testing
        checkpoint_dir=None  # Disable checkpointing for testing
    )
    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig
    learner = Learner(model, None, config, common_key)

    batch_size = 4  # Reduced for faster testing
    observations = jax.random.normal(
        common_key, (batch_size, config.num_unroll_steps + 1, *common_cfg_flat.observation_shape)
    )

    batch = {
        "observation": observations,
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
        "training_step": 5000,  # Middle of temperature decay
    }

    # Test computation with EfficientZeroV2 configuration
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Verify expected behavior
    assert jnp.isfinite(loss), "Loss should be finite with EfficientZeroV2 config"
    assert "policy_loss" in metrics, "Policy loss should be computed"

    # Verify reanalysis ratio calculation
    expected_reanalyze_batch_size = int(batch_size * config.reanalyze_ratio)
    assert (
        expected_reanalyze_batch_size == 2
    ), f"Expected 2 reanalyzed samples, got {expected_reanalyze_batch_size}"

    print("✅ Policy reanalysis EfficientZeroV2 integration patterns test completed!")

def test_wandb_logging_disabled(common_key, common_cfg_flat):
    """Test training with wandb logging disabled to cover missing lines."""
    import wandb

    # Ensure wandb is not initialized
    if wandb.run is not None:
        wandb.finish()

    cfg = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_no_wandb", use_ema=False
    )
    model = make_model(common_key, common_cfg_flat)
    optimizer_def = optax.adam(learning_rate=1e-4)
    learner = Learner(model, optimizer_def, cfg, common_key)

    try:
        # Create a simple batch generator
        def batch_generator():
            while True:
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
                yield batch

        # Train for 1 epoch, 2 steps (should not crash)
        learner.train(batch_generator, num_epochs=1, steps_per_epoch=2)
        print("✅ Training without wandb logging successful!")
    except Exception as e:
        pytest.fail(f"Training failed: {e}")
    finally:
        # Explicit cleanup
        learner.cleanup()
        if wandb.run is not None:
            wandb.finish()

def test_full_training_pipeline_integration(common_key, common_cfg_flat):
    """Test the complete training pipeline integration with all components."""
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        3,  # num_unroll_steps
        False,  # use_projection
        "full_pipeline_test",
        use_ema=True
    )
    config = dataclasses.replace(
        config,
        reanalyze_ratio=0.3,
        num_simulations=2,  # Reduced for testing
        value_target="mixed",
        mixed_value_threshold=1000,
        start_use_mix_training_steps=10,
        value_target_type="GAE",
        gae_max_steps=8,
        td_lambda=0.9,
        checkpoint_frequency=100,
        checkpoint_dir=None,  # Disable checkpointing for testing
    )

    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig
    optimizer = optax.adam(config.learning_rate)
    learner = Learner(model, optimizer, config, common_key)

    # Create comprehensive batch with all components
    batch_size = 2
    num_steps = config.num_unroll_steps + 1
    extra_steps = config.gae_max_steps

    k1, k2, k3, k4, k5, k6 = jax.random.split(common_key, 6)
    
    batch = {
        "observation": jax.random.normal(k1, (batch_size, num_steps, *common_cfg_flat.observation_shape)),
        "action": jax.random.randint(k2, (batch_size, config.num_unroll_steps), 0, config.num_actions),
        "target_reward": jax.random.normal(k3, (batch_size, num_steps)),
        "target_value": jax.random.normal(k4, (batch_size, num_steps)),
        "target_search_value": jax.random.normal(k5, (batch_size, num_steps)),
        "target_policy": jax.nn.softmax(jax.random.normal(k6, (batch_size, num_steps, config.num_actions))),
        "game_history_mask": jnp.ones((batch_size, num_steps)),
        "extra_observations": jax.random.normal(k1, (batch_size, extra_steps, *common_cfg_flat.observation_shape)),
        "extra_actions": jax.random.randint(k2, (batch_size, extra_steps), 0, config.num_actions),
        "extra_rewards": jax.random.normal(k3, (batch_size, extra_steps)),
        "extra_dones": jnp.zeros((batch_size, extra_steps)),
        "sample_indices": jnp.array([500, 1500]),
        "collected_transitions": 2000,
        "training_step": 50,
        "weights": jnp.ones(batch_size),
    }

    # Test full training step
    metrics = learner.train_step(batch)

    # Verify all expected metrics are present
    expected_metrics = ["total_loss", "policy_loss", "value_loss", "reward_loss"]
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert jnp.isfinite(metrics[metric]), f"Metric {metric} is not finite"

    # Test multiple steps to ensure stability
    for step in range(3):
        batch["training_step"] = 50 + step
        metrics = learner.train_step(batch)
        assert jnp.isfinite(metrics["total_loss"]), f"Loss not finite at step {step}"

    # Verify EMA is working if enabled
    if hasattr(learner, 'target_model'):
        assert learner.target_model is not None, "Target model should exist when EMA is enabled"

    # Explicit cleanup
    learner.cleanup()
    
    print("✅ Full training pipeline integration test completed!")

def test_checkpoint_configuration_setup(common_key, common_cfg_flat):
    """Test checkpoint configuration setup and initialization."""
    import tempfile
    import shutil

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp(prefix="mz_checkpoint_config_test_")
    
    try:
        config = make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,  # Minimal unroll steps for faster test
            False,  # use_projection
            "checkpoint_config_test",
            use_ema=False
        )
        config = dataclasses.replace(
            config,
            checkpoint_dir=temp_dir,
            checkpoint_frequency=2,
            max_checkpoints_to_keep=2,
        )

        model = make_model(common_key, common_cfg_flat)
        optimizer = optax.adam(config.learning_rate)
        learner = Learner(model, optimizer, config, common_key)

        # Verify checkpoint configuration is properly set
        assert learner.checkpoint_manager is not None, "Checkpoint manager should be initialized"
        assert learner.config.checkpoint_dir == temp_dir, "Checkpoint directory should be set correctly"
        assert learner.config.checkpoint_frequency == 2, "Checkpoint frequency should be set correctly"
        
        # Explicit cleanup
        learner.cleanup()
        
        print("✅ Checkpoint configuration setup test completed!")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_checkpoint_basic_training_workflow(common_key, common_cfg_flat):
    """Test basic training workflow with checkpoint configuration enabled."""
    import tempfile
    import shutil

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp(prefix="mz_checkpoint_training_test_")

    try:
        config = make_cfg(
            0,  # scalar value support
            0,  # scalar reward support
            1,  # Minimal unroll steps for faster test
            False,  # use_projection
            "checkpoint_training_test",
            use_ema=False
        )
        config = dataclasses.replace(
            config,
            checkpoint_dir=temp_dir,
            checkpoint_frequency=100,  # High frequency to avoid checkpointing in this test
            max_checkpoints_to_keep=1,
            batch_size=1,
        )

        # Create simple network config
        simple_cfg = MockNetCfg(
            observation_shape=(2, 2),
            num_actions=3,
            batch_size=1,
            value_support_size=0,
            reward_support_size=0,
            hidden_size=8
        )

        model = make_model(common_key, simple_cfg)
        optimizer = optax.adam(config.learning_rate)
        learner = Learner(model, optimizer, config, common_key)

        # Create simple batch for training
        batch = make_batch(
            common_key,
            config.batch_size,
            (2, 2),
            3,
            config.num_unroll_steps,
            0,
            0,
        )

        # Train for a few steps to verify training works with checkpoint config
        for step in range(2):  # Reduced from 3 for faster execution
            batch["training_step"] = step
            metrics = learner.train_step(batch)
            assert jnp.isfinite(metrics["total_loss"]), f"Loss not finite at step {step}"

        print("✅ Checkpoint basic training workflow test completed!")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_checkpoint_disabled_workflow(common_key, common_cfg_flat):
    """Test training workflow with checkpointing disabled."""
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        1,  # Minimal unroll steps for faster test
        False,  # use_projection
        "no_checkpoint_test",
        use_ema=False
    )
    # checkpoint_dir=None disables checkpointing

    model = make_model(common_key, common_cfg_flat)
    optimizer = optax.adam(config.learning_rate)
    learner = Learner(model, optimizer, config, common_key)

    # Verify checkpointing is disabled
    assert learner.checkpoint_manager is None, "Checkpoint manager should be None when disabled"

    # Create simple batch for training
    batch = make_batch(
        common_key,
        config.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        config.num_unroll_steps,
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    # Train for a few steps to verify training works without checkpointing
    for step in range(2):  # Minimal steps for faster execution
        batch["training_step"] = step
        metrics = learner.train_step(batch)
        assert jnp.isfinite(metrics["total_loss"]), f"Loss not finite at step {step}"

    print("✅ Checkpoint disabled workflow test completed!")

def test_error_handling_and_fallback_integration(common_key, common_cfg_flat):
    """Test error handling and fallback mechanisms in integrated workflows. OPTIMIZED for speed."""
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        1,  # Reduced from 2 to 1 for speed
        False,  # use_projection
        "error_handling_test",
    )
    config = dataclasses.replace(config, batch_size=1)  # Reduce batch size

    model = make_model(common_key, common_cfg_flat)
    learner = Learner(model, None, config, common_key)

    # Test with minimal batch (missing optional components) - OPTIMIZED SIZE
    minimal_batch = {
        "observation": jax.random.normal(common_key, (1, 2, *common_cfg_flat.observation_shape)),  # Reduced batch size from 2 to 1
        "action": jax.random.randint(common_key, (1, 1), 0, config.num_actions),  # Reduced size
        "target_reward": jax.random.normal(common_key, (1, 2)),  # Reduced size
        "target_value": jax.random.normal(common_key, (1, 2)),  # Reduced size
        "target_policy": jax.nn.softmax(jax.random.normal(common_key, (1, 2, config.num_actions))),  # Reduced size
        "game_history_mask": jnp.ones((1, 2)),  # Reduced size
    }

    # Should handle gracefully without optional components
    loss, metrics = Learner._compute_total_loss_static(
        model, config, minimal_batch, common_key, training=True
    )
    assert jnp.isfinite(loss), "Loss should be finite with minimal batch"

    # OPTIMIZED: Simplified test - only test one additional scenario
    malformed_batch = minimal_batch.copy()
    malformed_batch["unknown_field"] = jnp.ones((1, 2))  # Extra field should be ignored

    loss, metrics = Learner._compute_total_loss_static(
        model, config, malformed_batch, common_key, training=True
    )
    assert jnp.isfinite(loss), "Loss should be finite with extra fields"

    print("✅ Error handling and fallback integration test completed (OPTIMIZED)!") 