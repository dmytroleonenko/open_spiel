import pytest
import os
import tempfile
import dataclasses
import shutil
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.nnx.graph as nnx_graph
import optax
from unittest.mock import patch

# Import from the common utils module
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
    NUM_UNROLL_STEPS,
    key as common_key, cfg_flat as common_cfg_flat,
    make_model, make_cfg, make_batch, make_model_from_muzero_config
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig 

@pytest.mark.parametrize(
    "use_ema, resume", [(False, False), (True, False), (True, True)]
)
def test_train_loop_and_ckpt(common_key, common_cfg_flat, use_ema, resume):
    mk, lk, bk = jax.random.split(common_key, 3)
    
    # Use minimal configuration for fastest testing
    cfgn = dataclasses.replace(
        common_cfg_flat, 
        use_projection=use_ema,
        hidden_size=4,  # Further reduced to 4 for minimal computation
        batch_size=1,
        observation_shape=(4,),  # Smaller observation space
        projection_output_size=2  # Minimal projection size
    )
    model = make_model(mk, cfgn)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        cfg_suffix = f"fast_ckpt_{use_ema}_{resume}"
        cfg = make_cfg(
            cfgn.value_support_size,
            cfgn.reward_support_size,
            1,  # Minimal unroll steps
            proj=cfgn.use_projection,
            suffix=cfg_suffix,
            use_ema=use_ema,
            ssl_weight=0.01 if cfgn.use_projection else 0.0,  # Reduced SSL weight
            checkpoint_dir=checkpoint_dir,
            batch_size=1,
            checkpoint_frequency=1,
            learning_rate=1e-2,  # Higher learning rate for faster convergence in tests
            clip_grad_norm=1.0,  # Reduced gradient clipping
        )
        cfg = dataclasses.replace(
            cfg, resume_from_checkpoint=False
        )

        opt = optax.adam(cfg.learning_rate)
        learner = Learner(model, opt, cfg, lk)

        # Minimal training steps - just enough to test checkpointing
        num_total_steps = 1  # Reduced to just 1 step
        batches = [
            make_batch(
                jax.random.fold_in(bk, i),
                cfg.batch_size,
                cfgn.observation_shape,
                cfgn.num_actions,
                cfg.num_unroll_steps,
                cfgn.value_support_size,
                cfgn.reward_support_size,
                cfgn.projection_output_size,
                cfgn.use_projection,
            )
            for i in range(num_total_steps)
        ]

        def get_batch_generator_fn():
            def gen():
                for item in batches:
                    yield item
            return gen()

        # Run minimal training
        learner.train(
            get_batch_generator_fn, num_epochs=1, steps_per_epoch=num_total_steps
        )

        assert learner.num_training_steps == num_total_steps

        # Test checkpoint saving with minimal overhead
        if cfg.checkpoint_dir and learner.checkpoint_manager:
            learner.save_checkpoint(force_save=True)
            # Reduced wait time for testing
            learner.checkpoint_manager.wait_until_finished()

            latest_saved_step = learner.checkpoint_manager.latest_step()
            assert latest_saved_step is not None, "Expected checkpoint to be saved"
            assert latest_saved_step == num_total_steps

        if resume:
            # Test checkpoint loading
            mk_resume, lk_resume = jax.random.split(jax.random.fold_in(common_key, 100), 2)
            model_resume = make_model(mk_resume, cfgn)
            cfg_resume = dataclasses.replace(cfg, resume_from_checkpoint=True)
            opt_resume = optax.adam(cfg_resume.learning_rate)
            learner_resume = Learner(model_resume, opt_resume, cfg_resume, lk_resume)
            
            # Basic checkpoint restoration checks
            assert learner_resume.num_training_steps == num_total_steps
            if use_ema:
                assert learner_resume.target_model is not None
                assert learner_resume.ema_params_state is not None

            # Quick cleanup for resume learner
            if learner_resume.checkpoint_manager is not None:
                try:
                    learner_resume.checkpoint_manager.close()
                except Exception:
                    pass
            del learner_resume

        # Quick cleanup
        if learner.checkpoint_manager is not None:
            try:
                learner.checkpoint_manager.close()
                learner.checkpoint_manager = None
            except Exception:
                pass
        del learner


def test_checkpointing_no_manager(common_key, common_cfg_flat):
    mk, lk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)
    # Create a config with checkpoint_dir=None
    cfg_no_ckpt = MuZeroConfig(
        value_support_size=common_cfg_flat.value_support_size,
        reward_support_size=common_cfg_flat.reward_support_size,
        checkpoint_dir=None,  # Explicitly None
    )
    opt = optax.adam(cfg_no_ckpt.learning_rate)
    learner = Learner(model, opt, cfg_no_ckpt, lk)

    assert learner.checkpoint_manager is None

    # Test save_checkpoint
    with patch("builtins.print") as mock_print_save:
        learner.save_checkpoint()
    mock_print_save.assert_any_call("Checkpoint manager not configured. Skipping save.")

    # Test load_checkpoint
    with patch("builtins.print") as mock_print_load:
        loaded = learner.load_checkpoint()
    assert not loaded
    mock_print_load.assert_any_call("Checkpoint manager not configured. Skipping load.")

def test_load_checkpoint_no_checkpoint_exists(common_key, common_cfg_flat):
    mk, lk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        cfg_suffix = "no_ckpt_exists"
        cfg_ckpt = make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,
            proj=False,
            suffix=cfg_suffix,
            checkpoint_dir=checkpoint_dir,
        )

        opt = optax.adam(cfg_ckpt.learning_rate)
        learner = Learner(model, opt, cfg_ckpt, lk)

        assert learner.checkpoint_manager is not None

        with patch("builtins.print") as mock_print:
            loaded = learner.load_checkpoint()
    assert not loaded
    mock_print.assert_any_call("No checkpoint found to resume from.")


def test_extract_step_from_checkpoint_path_invalid_format(tmp_path, common_key, common_cfg_flat, caplog):
    """Learner should warn and return None when checkpoint path lacks a step suffix."""
    mk, lk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)
    cfg = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        steps=1,
        proj=False,
        suffix="extract_step_invalid",
        checkpoint_dir=str(tmp_path),
    )
    learner = Learner(model, optax.adam(cfg.learning_rate), cfg, lk)

    caplog.clear()
    with caplog.at_level("WARNING"):
        extracted = learner._extract_step_from_checkpoint_path(str(tmp_path / "checkpoint_latest"))

    assert extracted is None
    assert any("Could not determine step from checkpoint path" in record.message for record in caplog.records)




def test_load_checkpoint_load_exception(common_key, common_cfg_flat):
    mk, lk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)
    cfg_suffix = "load_exception"

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        cfg = make_cfg(
            common_cfg_flat.value_support_size,
            common_cfg_flat.reward_support_size,
            1,
            False,
            cfg_suffix,
            checkpoint_dir=checkpoint_dir,
        )
        cfg = dataclasses.replace(cfg, resume_from_checkpoint=True)

        opt = optax.adam(cfg.learning_rate)

        # Create a dummy checkpoint manager that will raise an exception on restore
        class FailingCheckpointManager:
            def __init__(self, *args, **kwargs):
                self.latest_step_val = 1

            def latest_step(self):
                return self.latest_step_val  # Pretend a checkpoint exists

            def restore(self, step, args=None):
                raise ValueError("Simulated restore error")

            def wait_until_finished(self):
                pass

            def save(
                self, step, args=None
            ):  # Add save to allow Learner init to proceed far enough
                pass

            def close(self):
                pass

        with patch(
            "orbax.checkpoint.CheckpointManager", FailingCheckpointManager
        ), patch("logging.error") as mock_logging_error:
            # Learner init calls load_checkpoint
            learner = Learner(model, opt, cfg, lk)

        assert learner.num_training_steps == 0  # Should not have loaded steps
        found_error_log = any(
            "Failed to load checkpoint: Simulated restore error" in str(call_args[0][0])
            for call_args in mock_logging_error.call_args_list
        )
        assert (
            found_error_log
        ), f"Expected error log not found. Logs: {[str(call) for call in mock_logging_error.call_args_list]}"

def test_checkpoint_error_handling(common_key, common_cfg_flat):
    """Test error handling in checkpoint save/load operations."""
    cfg_no_checkpoint = make_cfg(
        vsup=0,
        rsup=0,
        steps=1,
        proj=False,
        suffix="_no_checkpoint",
        use_ema=False,
        checkpoint_dir=None,
    )
    model = make_model_from_muzero_config(common_key, cfg_no_checkpoint, common_cfg_flat.observation_shape)
    learner = Learner(model, None, cfg_no_checkpoint, common_key)

    # Test save_checkpoint when checkpoint_manager is None
    learner.save_checkpoint(force_save=True)  # Should print message and return

    # Test load_checkpoint when checkpoint_manager is None
    result = learner.load_checkpoint()  # Should print message and return False
    assert result == False

    # Test __del__ method when checkpoint_manager exists
    cfg_with_checkpoint = make_cfg(
        vsup=0,
        rsup=0,
        steps=1,
        proj=False,
        suffix="_cleanup_test",
        use_ema=False,
        checkpoint_dir="/tmp/test_cleanup",
    )
    model_cleanup = make_model_from_muzero_config(common_key, cfg_with_checkpoint, common_cfg_flat.observation_shape)
    learner_cleanup = Learner(model_cleanup, None, cfg_with_checkpoint, common_key)

    # The __del__ method should be called when the object is destroyed
    # We can't directly test __del__ but we can verify the checkpoint_manager exists
    assert learner_cleanup.checkpoint_manager is not None

    # Clean up manually to avoid issues
    learner_cleanup.checkpoint_manager.close()

def test_ema_checkpoint_fallback_edge_cases(common_key, common_cfg_flat):
    """Test additional edge cases for EMA checkpoint fallback scenarios."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test case: Normal EMA checkpoint loading (both save and load with EMA enabled)
        # This verifies that when EMA data IS present in checkpoint, it loads correctly
        cfg = make_cfg(
            0, 0, 1, False, "ema_normal", use_ema=True, checkpoint_dir=temp_dir
        )
        cfg = dataclasses.replace(cfg, checkpoint_frequency=1)

        # Create and train first learner with EMA
        model1 = make_model_from_muzero_config(common_key, cfg, common_cfg_flat.observation_shape)
        learner1 = Learner(model1, None, cfg, common_key)
        batch = make_batch(common_key, cfg.batch_size, common_cfg_flat.observation_shape, common_cfg_flat.num_actions, 1, 0, 0)
        learner1.train_step(batch)

        # Get state before saving
        saved_online_params = nnx.state(learner1.model, nnx.Param)
        saved_ema_params = learner1.ema_params_state.ema
        saved_target_params = nnx.state(learner1.target_model, nnx.Param)

        # Save checkpoint with EMA components
        learner1.save_checkpoint(force_save=True)

        # Clean up first learner
        if learner1.checkpoint_manager is not None:
            learner1.checkpoint_manager.close()

        # Create second learner with EMA and load checkpoint
        cfg_resume = dataclasses.replace(cfg, resume_from_checkpoint=True)
        model2 = make_model_from_muzero_config(jax.random.fold_in(common_key, 1), cfg_resume, common_cfg_flat.observation_shape)
        learner2 = Learner(model2, None, cfg_resume, jax.random.fold_in(common_key, 2))

        def params_equal(p1, p2):
            """Check if two parameter trees are equal."""

            def compare_leaf(leaf1, leaf2):
                v1 = nnx.value(leaf1) if hasattr(leaf1, 'value') else leaf1
                v2 = nnx.value(leaf2) if hasattr(leaf2, 'value') else leaf2
                return jnp.allclose(v1, v2, rtol=1e-6)

            return jax.tree_util.tree_all(jax.tree_util.tree_map(compare_leaf, p1, p2))

        # Verify all components were loaded correctly
        loaded_online_params = nnx.state(learner2.model, nnx.Param)
        loaded_ema_params = learner2.ema_params_state.ema
        loaded_target_params = nnx.state(learner2.target_model, nnx.Param)

        assert params_equal(
            loaded_online_params, saved_online_params
        ), "Online parameters should be loaded correctly"
        assert params_equal(
            loaded_ema_params, saved_ema_params
        ), "EMA parameters should be loaded correctly"
        assert params_equal(
            loaded_target_params, saved_target_params
        ), "Target model parameters should be loaded correctly"

        # Verify EMA components are properly initialized
        assert learner2.ema_params_state is not None, "EMA state should be loaded"
        assert learner2.target_model is not None, "Target model should be loaded"
        assert learner2.ema_updater is not None, "EMA updater should be initialized"

        # Clean up
        if learner2.checkpoint_manager is not None:
            learner2.checkpoint_manager.close()

def test_save_checkpoint_conditions(common_key, common_cfg_flat):
    mk, lk, bk = jax.random.split(common_key, 3)
    cfgn = common_cfg_flat
    model = make_model(mk, cfgn)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        cfg_suffix = "save_conditions"
        cfg = make_cfg(
            cfgn.value_support_size,
            cfgn.reward_support_size,
            1,
            proj=False,
            suffix=cfg_suffix,
            checkpoint_dir=checkpoint_dir,
        )
        # Set checkpoint frequency high to test skipping, then force save
        cfg = dataclasses.replace(
            cfg, checkpoint_frequency=100, max_checkpoints_to_keep=1
        )

        opt = optax.adam(cfg.learning_rate)
        learner = Learner(model, opt, cfg, lk)
        assert learner.checkpoint_manager is not None

        # --- Test skipping save due to frequency ---
        learner.num_training_steps = 50  # Less than checkpoint_frequency
        with patch.object(
            learner.checkpoint_manager, "save"
        ) as mock_manager_save, patch("logging.info") as mock_logging_info_skip:
            learner.save_checkpoint(force_save=False)
        mock_manager_save.assert_not_called()
        # Check for the specific log message indicating skip
        assert any(
            f"SAVE_CHECKPOINT: Condition NOT met. force_save=False, num_training_steps={learner.num_training_steps}, freq={cfg.checkpoint_frequency}"
            in call_args[0][0]
            for call_args in mock_logging_info_skip.call_args_list
        ), "Log message for skipping save due to frequency not found."

        # --- Test force_save=True at end of hypothetical training (within train loop logic) ---
        # This part simulates the condition within the train() method
        num_epochs = 1
        steps_per_epoch = 3
        learner.num_training_steps = 0  # Reset
        cfg_train_end = dataclasses.replace(
            cfg, checkpoint_frequency=2
        )  # Save every 2 steps
        learner.config = cfg_train_end  # Update learner's config

        batches = [
            make_batch(
                jax.random.fold_in(bk, i),
                cfg_train_end.batch_size,
                cfgn.observation_shape,
                cfgn.num_actions,
                cfg_train_end.num_unroll_steps,
                cfgn.value_support_size,
                cfgn.reward_support_size,
            )
            for i in range(steps_per_epoch)
        ]

        def get_batch_gen_fn():
            def gen():
                yield from batches

            return gen()

        with patch.object(
            learner.checkpoint_manager, "save"
        ) as mock_manager_save_train, patch("logging.info") as mock_logging_info_train:
            learner.train(
                get_batch_gen_fn, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch
            )

        # Expected saves:
        # Step 2 (regular)
        # Step 3 (end of training, force_save=True via internal logic of train() calling save_checkpoint(force_save=True))
        assert mock_manager_save_train.call_count == 2
        # Check the call for step 2 (regular)
        args_step2, kwargs_step2 = mock_manager_save_train.call_args_list[0]
        assert kwargs_step2["step"] == 2  # step number
        # Check the call for step 3 (end of training)
        args_step3, kwargs_step3 = mock_manager_save_train.call_args_list[1]
        assert kwargs_step3["step"] == 3  # step number

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
                    if (
                        f"SAVE_CHECKPOINT: Condition met. force_save=True, num_training_steps={steps_per_epoch}, freq={cfg_train_end.checkpoint_frequency}"
                        in subsequent_call_args[0][0]
                    ):
                        found_force_save_log = True
                        break
                if found_force_save_log:
                    break
        assert (
            found_force_save_log
        ), "Log message for force_save=True at end of training not found."

        # --- Test regular save due to frequency ---
        # learner.config.checkpoint_frequency is 2 at this point from the previous section of the test.
        learner.num_training_steps = learner.config.checkpoint_frequency  # This will be 2
        initial_model_params_before_freq_save, _, _, _, _, _ = nnx.split(
            learner.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )
        initial_opt_state_before_freq_save = nnx.state(learner.optimizer)

        with patch.object(
            learner.checkpoint_manager, "save"
        ) as mock_manager_save_freq, patch("logging.info") as mock_logging_info_freq:
            learner.save_checkpoint(force_save=False)
        mock_manager_save_freq.assert_called_once()
        # Check for the specific log message indicating save due to frequency
        assert any(
            f"SAVE_CHECKPOINT: Condition met. force_save=False, num_training_steps={learner.num_training_steps}, freq={learner.config.checkpoint_frequency}"
            in call_args[0][0]
            for call_args in mock_logging_info_freq.call_args_list
        ), f"Log message for saving due to frequency not found. Log calls: {mock_logging_info_freq.call_args_list}"

        # Clean up
        if cfg.checkpoint_dir and os.path.exists(cfg.checkpoint_dir):
            shutil.rmtree(cfg.checkpoint_dir)
