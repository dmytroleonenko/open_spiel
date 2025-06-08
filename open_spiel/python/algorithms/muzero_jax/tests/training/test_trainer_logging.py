import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import os
import shutil
import dataclasses
from unittest.mock import patch, PropertyMock, MagicMock

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

def test_wandb_logging(common_key, common_cfg_flat):
    """Test that wandb logging integration works correctly."""
    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig
    mk, lk, bk = jax.random.split(common_key, 3)
    cfgn = common_cfg_flat
    # Ensure a unique directory that will be empty or cleaned
    cfg = make_cfg(
        cfgn.value_support_size,
        cfgn.reward_support_size,
        1,  # num_unroll_steps
        proj=False,
        suffix="wandb_log_test",
        use_ema=False,
    )
    # Disable checkpointing for this specific test to avoid directory issues
    cfg = dataclasses.replace(cfg, checkpoint_dir=None, checkpoint_frequency=10000)

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)

    num_train_steps = 2
    batches = [
        make_batch(
            jax.random.fold_in(bk, i),
            cfg.batch_size,
            cfgn.observation_shape,
            cfgn.num_actions,
            cfg.num_unroll_steps,
            cfgn.value_support_size,
            cfgn.reward_support_size,
        )
        for i in range(num_train_steps)
    ]

    def get_batch_generator_fn():
        def gen():
            yield from batches

        return gen()

    with patch("wandb.log") as mock_wandb_log, patch(
        "wandb.init", return_value=None
    ) as mock_wandb_init, patch(
        "wandb.run", new_callable=PropertyMock
    ) as mock_wandb_run:  # Added patch for wandb.run

        # Configure the mock_wandb_run to behave as if wandb.run is an active run object
        # A simple way is to make it not None. If it needs attributes, they can be set on a MagicMock.
        mock_wandb_run.return_value = (
            patch.object
        )  # Use a simple object that's not None

        learner.train(
            get_batch_generator_fn, num_epochs=1, steps_per_epoch=num_train_steps
        )

    assert mock_wandb_log.call_count == num_train_steps

    # Check the arguments of each call to wandb.log
    for i in range(num_train_steps):
        call_args = mock_wandb_log.call_args_list[i]
        logged_metrics = call_args[0][0]  # First positional argument to wandb.log
        logged_step = call_args[1]["step"]  # Keyword argument 'step'

        assert isinstance(logged_metrics, dict)
        # Check for essential metric keys that should be present (using actual format from trainer)
        for expected_key in [
            "loss/total",
            "loss/policy",
            "loss/value",
            "loss/reward",
            "loss/l2",
            "metrics/grad_norm",
            "metrics/param_norm",
        ]:
            assert expected_key in logged_metrics

        assert logged_step == i + 1  # num_training_steps is incremented starting from 1

    # Clean up the dummy directory if make_cfg created it, though disabled for this test
    if cfg.checkpoint_dir and os.path.exists(cfg.checkpoint_dir):
        shutil.rmtree(cfg.checkpoint_dir)  # pragma: no cover

def teardown_module(module):
    """Clean up temporary directories created during tests."""
    pass 