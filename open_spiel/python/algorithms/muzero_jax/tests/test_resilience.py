"""Resilience and restart tests for MuZero JAX components."""

import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import optax
import pytest
from omegaconf import DictConfig

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
    make_cfg,
    make_model_from_muzero_config,
)


def _orchestrator_config(base_path: Path) -> DictConfig:
    """Return a minimal DictConfig that writes checkpoints under base_path."""
    config = {
        "exp_config": {"tag": "resilience", "debug": False, "seed": 0},
        "game": {"name": "tic_tac_toe"},
        "training": {
            "training_steps": 4,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "start_transitions": 0,
            "learning_rate": 1e-3,
            "td_steps": 2,
            "discount": 0.99,
            "num_unroll_steps": 1,
            "value_loss_weight": 1.0,
            "policy_loss_weight": 1.0,
            "reward_loss_weight": 1.0,
            "l2_regularization": 1e-4,
            "target_update_interval": 10,
        },
        "mcts": {
            "num_simulations": 2,
            "gumbel_scale": 1.0,
            "temperature_init": 1.0,
            "temperature_final": 0.1,
            "temperature_decay_steps": 100,
            "dirichlet_alpha": 0.3,
            "exploration_fraction": 0.25,
        },
        "replay_buffer": {"capacity": 10, "min_size_to_sample": 1},
        "actors": {"num_actors": 1},
        "bootstrap": {"enabled": True, "min_episodes": 1, "c_puct": 1.25},
        "output": {
            "save_path": str(base_path),
            "log_interval": 1,
            "checkpoint_interval": 1,
        },
        "wandb": {"enabled": False},
        "evaluation": {"enabled": False, "interval": 10},
        "resource_management": {
            "device": "cpu",
            "sequential_training": True,
            "training_phase_steps": 1,
            "selfplay_phase_episodes": 1,
            "max_episodes_without_training": 5,
            "max_training_steps_without_episodes": 5,
        },
        "random_seed": 0,
    }
    return DictConfig(copy.deepcopy(config))


def _make_test_learner(tmp_path: Path, num_steps: int = 3):
    """Create a learner, advance it, and persist a checkpoint."""
    key = jax.random.PRNGKey(0)
    model_key, learner_key, resume_model_key, resume_rng = jax.random.split(key, 4)
    cfg = make_cfg(
        vsup=0,
        rsup=0,
        steps=1,
        proj=False,
        suffix="resilience",
        checkpoint_dir=str(tmp_path),
        checkpoint_frequency=1,
    )
    model = make_model_from_muzero_config(model_key, cfg)
    learner = Learner(model, optax.adam(cfg.learning_rate), cfg, learner_key)
    learner.num_training_steps = num_steps
    checkpoint_path = learner.save_checkpoint(force_save=True)
    learner.checkpoint_manager.wait_until_finished()
    return cfg, checkpoint_path, resume_model_key, resume_rng, num_steps


def test_learner_can_resume_from_explicit_checkpoint(tmp_path):
    """Learner.load_checkpoint should accept explicit checkpoint paths."""
    cfg, checkpoint_path, resume_model_key, resume_rng, num_steps = _make_test_learner(tmp_path)

    resume_cfg = dataclasses.replace(cfg, resume_from_checkpoint=False)
    resume_model = make_model_from_muzero_config(resume_model_key, resume_cfg)
    resume_opt = optax.adam(resume_cfg.learning_rate)
    resume_learner = Learner(resume_model, resume_opt, resume_cfg, resume_rng)

    assert resume_learner.num_training_steps == 0
    assert resume_learner.load_checkpoint(checkpoint_path)
    assert resume_learner.num_training_steps == num_steps


def test_orchestrator_restores_training_progress(tmp_path):
    """Restarting the orchestrator should pick up from the latest checkpoint."""
    first_orchestrator = MuZeroOrchestrator(_orchestrator_config(tmp_path))
    first_orchestrator.training_step = 2
    first_orchestrator.learner.num_training_steps = 2
    first_orchestrator.learner.save_checkpoint(force_save=True)
    first_orchestrator.learner.checkpoint_manager.wait_until_finished()

    resumed_orchestrator = MuZeroOrchestrator(_orchestrator_config(tmp_path))
    assert resumed_orchestrator.training_step == 2
    assert resumed_orchestrator.learner.num_training_steps == 2


def test_actor_reloads_network_parameters_on_resume(tmp_path):
    """Actors should pull the latest parameters when a checkpoint appears."""
    network = MagicMock()
    game_wrapper = MagicMock()
    replay_buffer = MagicMock()
    config = SimpleNamespace(num_actions=3)
    actor = Actor(network, game_wrapper, replay_buffer, config=config, mcts=MagicMock())

    fake_checkpoint = str(tmp_path / "checkpoint_5")
    fake_params = {"params": {"w": 1.0}}

    with patch(
        "open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint",
        return_value=fake_checkpoint,
    ) as mock_get, patch.object(
        actor,
        "load_network_parameters",
        return_value=fake_params,
    ) as mock_load:
        loaded = actor.maybe_load_latest_parameters(str(tmp_path))

    assert loaded
    assert actor.current_params == fake_params["params"]
    mock_get.assert_called_once_with(str(tmp_path))
    mock_load.assert_called_once_with(fake_checkpoint)
