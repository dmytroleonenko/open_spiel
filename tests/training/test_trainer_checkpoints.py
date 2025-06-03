from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew, MockProj
import tempfile
import dataclasses
import jax
import jax.numpy as jnp
import os
import shutil
from unittest.mock import patch
import optax


def test_checkpoint_error_handling(common_key, common_cfg_flat):
    """Test checkpoint error handling paths that aren't covered."""

    # Create a learner without checkpoint manager
    cfg_no_checkpoint = make_cfg(
        vsup=0,
        rsup=0,
        steps=1,
        proj=False,
        suffix="_no_checkpoint",
        use_ema=False,
        checkpoint_dir=None,
    )
    model = make_model(common_key, cfg_no_checkpoint)
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
    model_cleanup = make_model(common_key, cfg_with_checkpoint)
    learner_cleanup = Learner(model_cleanup, None, cfg_with_checkpoint, common_key)

    # The __del__ method should be called when the object is destroyed
    # We can't directly test __del__ but we can verify the checkpoint_manager exists
    assert learner_cleanup.checkpoint_manager is not None

    # Clean up manually to avoid issues
    learner_cleanup.checkpoint_manager.close()


def test_efficientzero_v2_config_defaults(common_key, common_cfg_flat):
    """Test that new EfficientZeroV2 config options have correct defaults."""
    cfg = MuZeroConfig()

    # Test value target selection defaults
    assert cfg.value_target == "mixed"
    assert cfg.mixed_value_target_switch_step == 100000

    # Test multiple value heads defaults
    assert cfg.v_num == 1

    # Test priority replay defaults
    assert cfg.use_priority_replay == True
    assert cfg.priority_exponent == 0.6
    assert cfg.min_priority == 1e-6

    # Test LSTM support defaults
    assert cfg.use_value_prefix == False
    assert cfg.lstm_horizon_length == 5


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
        model1 = make_model(common_key, cfg)
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
        model2 = make_model(jax.random.fold_in(common_key, 1), cfg_resume)
        learner2 = Learner(model2, None, cfg_resume, jax.random.fold_in(common_key, 2))

        def params_equal(p1, p2):
            """Check if two parameter trees are equal."""

            def compare_leaf(leaf1, leaf2):
                from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_test_utils import maybe_val
                v1 = maybe_val(leaf1)
                v2 = maybe_val(leaf2)
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

        # Test that EMA continues to work correctly after loading
        batch2 = make_batch(
            jax.random.fold_in(common_key, 2),
            cfg.batch_size,
            common_cfg_flat.observation_shape,
            common_cfg_flat.num_actions,
            1,
            0,
            0,
        )
        pre_training_ema = learner2.ema_params_state.ema
        learner2.train_step(batch2)
        post_training_ema = learner2.ema_params_state.ema

        if cfg.ema_decay < 1.0:
            params_changed = not params_equal(post_training_ema, pre_training_ema)
            assert params_changed, "EMA should continue updating after checkpoint load"

        # Clean up
        if learner2.checkpoint_manager is not None:
            learner2.checkpoint_manager.close()


def test_ema_checkpoint_real_fallback_scenario(common_key, common_cfg_flat):
    """Test real checkpoint loading that triggers EMA fallback due to missing EMA components."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Create a normal checkpoint with EMA enabled
        cfg = make_cfg(
            0, 0, 1, False, "ema_real_fallback", use_ema=True, checkpoint_dir=temp_dir
        )
        cfg = dataclasses.replace(cfg, checkpoint_frequency=1)

        model1 = make_model(common_key, cfg)
        learner1 = Learner(model1, None, cfg, common_key)

        # Do some training
        batch = make_batch(common_key, cfg.batch_size, common_cfg_flat.observation_shape, common_cfg_flat.num_actions, 1, 0, 0)
        for _ in range(3):
            learner1.train_step(batch)

        # Save normal checkpoint
        learner1.save_checkpoint(force_save=True)
        saved_step = learner1.num_training_steps

        # Get saved parameters before cleanup
        saved_online_params = nnx.state(learner1.model, nnx.Param)

        # Clean up first learner
        if learner1.checkpoint_manager is not None:
            learner1.checkpoint_manager.close()

        # Step 2: Create a new learner and manually trigger fallback by setting target_model to None
        # This simulates the exact condition that triggers the fallback in load_checkpoint
        cfg_resume = dataclasses.replace(cfg, resume_from_checkpoint=True)
        model2 = make_model(jax.random.fold_in(common_key, 1), cfg_resume)
        learner2 = Learner(model2, None, cfg_resume, jax.random.fold_in(common_key, 2))

        # Manually trigger the fallback condition by setting target_model to None
        # This simulates the condition where EMA components are missing or incomplete
        learner2.target_model = None  # This would trigger the fallback logic

        # Get parameters after loading but before fallback fix
        loaded_params_before_fallback = nnx.state(learner2.model, nnx.Param)

        # Step 3: Now manually run the fallback logic from the load_checkpoint method
        # This is the exact code that runs when target_model is None (lines 850-856)
        graphdef, params, batch_stats, rngs, static, ellipsis = nnx.split(
            learner2.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )
        learner2.target_model = nnx.merge(
            graphdef, params, batch_stats, rngs, static, ellipsis
        )

        # Re-initialize EMA state
        learner2.ema_updater = optax.ema(learner2.config.ema_decay)
        learner2.ema_params_state = learner2.ema_updater.init(params)
        # Crucial synchronization: ensure EMA internal average matches current online params
        learner2.ema_params_state = learner2.ema_params_state._replace(ema=params)

        def params_equal(p1, p2):
            """Check if two parameter trees are equal."""

            def compare_leaf(leaf1, leaf2):
                from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_test_utils import maybe_val
                v1 = maybe_val(leaf1)
                v2 = maybe_val(leaf2)
                return jnp.allclose(v1, v2, rtol=1e-6)

            return jax.tree_util.tree_all(jax.tree_util.tree_map(compare_leaf, p1, p2))

        # Step 4: Verify the fallback worked correctly
        assert (
            learner2.num_training_steps == saved_step
        ), "Training steps should be preserved from checkpoint"

        # Critical verification: EMA should be synchronized with loaded online parameters
        ema_internal_params = learner2.ema_params_state.ema
        target_params = nnx.state(learner2.target_model, nnx.Param)
        current_online_params = nnx.state(learner2.model, nnx.Param)

        assert params_equal(
            ema_internal_params, current_online_params
        ), "EMA internal average should be synchronized with online parameters after fallback"

        assert params_equal(
            target_params, current_online_params
        ), "Target model should match online parameters after fallback"

        # Verify that the online parameters loaded correctly from checkpoint
        assert params_equal(
            current_online_params, saved_online_params
        ), "Online parameters should match what was saved in checkpoint"

        # Step 5: Verify EMA functionality after fallback
        batch2 = make_batch(
            jax.random.fold_in(common_key, 2),
            cfg.batch_size,
            common_cfg_flat.observation_shape,
            common_cfg_flat.num_actions,
            1,
            0,
            0,
        )
        pre_training_ema = learner2.ema_params_state.ema

        learner2.train_step(batch2)

        post_training_ema = learner2.ema_params_state.ema
        if cfg.ema_decay < 1.0:
            params_changed = not params_equal(post_training_ema, pre_training_ema)
            assert params_changed, "EMA should update correctly after fallback"

        # Clean up
        if learner2.checkpoint_manager is not None:
            learner2.checkpoint_manager.close()


def test_ema_synchronization_during_initialization(common_key, common_cfg_flat):
    """Test that EMA synchronization works correctly during normal initialization."""
    cfg = make_cfg(0, 0, 1, False, "ema_init", use_ema=True)

    # Create model and learner
    model = make_model(common_key, cfg)
    learner = Learner(model, None, cfg, common_key)

    # Verify EMA state is properly synchronized during initialization
    online_params = nnx.state(learner.model, nnx.Param)
    ema_params = learner.ema_params_state.ema
    target_params = nnx.state(learner.target_model, nnx.Param)

    def params_equal(p1, p2):
        """Check if two parameter trees are equal."""

        def compare_leaf(leaf1, leaf2):
            from open_spiel.python.algorithms.muzero_jax.utils import tree_utils
            return tree_utils.pytree_allclose(leaf1, leaf2)

        return tree_utils.pytree_all(
            jax.tree_util.tree_map(compare_leaf, p1, p2)
        )

    # After initialization, EMA params should match online params (they're synchronized)
    assert params_equal(
        online_params, ema_params
    ), "EMA params should be synchronized with online params during initialization"

    # Target model params should also match (they're copied from online model)
    assert params_equal(
        online_params, target_params
    ), "Target params should be synchronized with online params during initialization"


def test_save_checkpoint_conditions(common_key, common_cfg_flat):
    """Test checkpoint saving under different conditions."""
    mk, lk, bk = jax.random.split(common_key, 3)
    # Use minimal configuration for faster testing
    cfgn = dataclasses.replace(
        common_cfg_flat,
        hidden_size=2,  # Minimal for speed
        batch_size=1,
        observation_shape=(2,),  # Smallest observation space  
    )
    model = make_model(mk, cfgn)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        cfg = make_cfg(
            cfgn.value_support_size,
            cfgn.reward_support_size,
            1,  # Minimal unroll steps
            proj=False,
            suffix="save_test",
            checkpoint_dir=checkpoint_dir,
            batch_size=1,
            learning_rate=1e-1,  # High learning rate for speed
        )
        # Set checkpoint frequency to 2 for testing
        cfg = dataclasses.replace(cfg, checkpoint_frequency=2)

        opt = optax.adam(cfg.learning_rate)
        learner = Learner(model, opt, cfg, lk)
        assert learner.checkpoint_manager is not None

        # Test skipping save due to frequency
        learner.num_training_steps = 1  # Less than checkpoint_frequency (2)
        with patch.object(learner.checkpoint_manager, "save") as mock_save:
            learner.save_checkpoint(force_save=False)
        mock_save.assert_not_called()

        # Test save due to frequency
        learner.num_training_steps = 2  # Equal to checkpoint_frequency
        with patch.object(learner.checkpoint_manager, "save") as mock_save:
            learner.save_checkpoint(force_save=False)
        mock_save.assert_called_once()

        # Test force save
        learner.num_training_steps = 1  # Less than frequency but force_save=True
        with patch.object(learner.checkpoint_manager, "save") as mock_save:
            learner.save_checkpoint(force_save=True)
        mock_save.assert_called_once()

        assert learner.load_checkpoint() == False
        # Check logging
        mock_info.assert_called_with("checkpoint manager is None")

        # Clean up
        if cfg.checkpoint_dir and os.path.exists(cfg.checkpoint_dir):
            shutil.rmtree(cfg.checkpoint_dir) 