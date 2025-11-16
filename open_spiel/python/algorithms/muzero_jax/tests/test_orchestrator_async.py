"""Tests for the asynchronous MuZero orchestrator runtime."""

import queue
import threading
import time
from pathlib import Path
from unittest import mock

import os
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import (
    ActorWorker,
    MuZeroOrchestrator,
)
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import BootstrapActor
from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner


class _StubActor:
    """Minimal actor stub for ActorWorker unit tests."""

    def __init__(self, label: str):
        self.label = label
        self.play_calls = 0

    def play_episode(self, rng_key):
        self.play_calls += 1
        return {
            'observations': [np.zeros((1,), dtype=np.float32)],
            'actions': [0],
            'rewards': [0.0],
            'policy_targets': [np.array([1.0], dtype=np.float32)],
            'value_targets': [0.0],
        }

    def maybe_load_latest_parameters(self, checkpoint_dir: str):
        return False


def _dummy_episode(game_wrapper):
    """Create a deterministic trajectory for patched actors."""
    obs_shape = game_wrapper.observation_shape
    num_actions = game_wrapper.num_distinct_actions()
    observation = np.zeros(obs_shape, dtype=np.float32)
    policy = np.ones(num_actions, dtype=np.float32) / num_actions
    return {
        'observations': [observation.copy() for _ in range(2)],
        'actions': [0, 1],
        'rewards': [0.0, 0.0],
        'policy_targets': [policy.copy() for _ in range(2)],
        'value_targets': [0.0, 0.0],
    }


def _patched_episode(self, rng_key):
    return _dummy_episode(self.game_wrapper)


def _slow_patched_episode(self, rng_key):
    time.sleep(0.01)
    return _dummy_episode(self.game_wrapper)


def _patched_train_step(self, batch):
    self.num_training_steps += 1
    batch_size = len(batch.get('actions', []))
    return {'total_loss': float(batch_size)}


@pytest.fixture
def async_config(tmp_path: Path) -> DictConfig:
    """Load base Hydra config and tailor it for async smoke tests."""
    # Force CPU backend so xdist/tests don't fight over the Metal device.
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    try:
        import jax
        jax.config.update("jax_platform_name", "cpu")
    except Exception:  # pragma: no cover - fallback if JAX already initialized
        pass

    cfg = OmegaConf.load("open_spiel/python/algorithms/muzero_jax/configs/config.yaml")
    cfg.output.save_path = str(tmp_path / "async_run")
    cfg.output.checkpoint_interval = 100
    cfg.output.log_interval = 1

    cfg.training.training_steps = 4
    cfg.training.batch_size = 2
    cfg.training.start_transitions = 2
    cfg.training.num_unroll_steps = 2
    cfg.training.td_steps = 2
    cfg.training.warmup_steps = 0

    cfg.replay_buffer.capacity = 32
    cfg.replay_buffer.min_size_to_sample = 2
    cfg.replay_buffer.priority_alpha = 0.0
    cfg.replay_buffer.max_trajectory_length = 4

    cfg.actors.num_actors = 2
    cfg.actors.checkpoint_sync_interval = 1

    cfg.bootstrap.enabled = True
    cfg.bootstrap.min_episodes = 2

    cfg.resource_management.device = "cpu"
    cfg.resource_management.sequential_training = False
    cfg.resource_management.concurrent = True
    cfg.resource_management.training_phase_steps = 1
    cfg.resource_management.selfplay_phase_episodes = 1
    cfg.resource_management.actor_queue_capacity = 4
    cfg.resource_management.learner_idle_sleep_ms = 1

    cfg.evaluation.enabled = False
    cfg.wandb.enabled = False
    return cfg


def test_actor_worker_switches_modes_and_stops():
    """ActorWorker should honor bootstrap flag and switch to MuZero actor."""
    trajectory_queue: "queue.Queue" = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    bootstrap_event = threading.Event()
    bootstrap_event.set()
    error_queue: "queue.Queue" = queue.Queue()

    bootstrap_actor = _StubActor("bootstrap")
    muzero_actor = _StubActor("muzero")

    worker = ActorWorker(
        worker_id=0,
        trajectory_queue=trajectory_queue,
        stop_event=stop_event,
        bootstrap_event=bootstrap_event,
        muzero_actor=muzero_actor,
        bootstrap_actor=bootstrap_actor,
        checkpoint_dir="/tmp",
        checkpoint_sync_interval=0,
        rng_seed=0,
        error_queue=error_queue,
    )
    worker.start()

    packet = trajectory_queue.get(timeout=1.5)
    assert packet.actor_type == "bootstrap"

    bootstrap_event.clear()
    packet = None
    for _ in range(5):
        candidate = trajectory_queue.get(timeout=1.5)
        if candidate.actor_type == "muzero":
            packet = candidate
            break
    assert packet is not None, "Actor worker never switched to MuZero mode"
    assert packet.actor_type == "muzero"

    stop_event.set()
    worker.join(timeout=3.0)
    assert error_queue.empty(), "Actor worker should not report errors"


@pytest.mark.parametrize("episode_fn", [_patched_episode, _slow_patched_episode])
def test_async_orchestrator_runs_to_completion(async_config, episode_fn):
    """Concurrent orchestrator should reach target steps and respect buffer bounds."""
    with (
        mock.patch.object(BootstrapActor, "play_episode", autospec=True, side_effect=episode_fn),
        mock.patch.object(Actor, "play_episode", autospec=True, side_effect=episode_fn),
        mock.patch.object(Actor, "maybe_load_latest_parameters", autospec=True, return_value=False),
        mock.patch.object(Learner, "train_step", autospec=True, side_effect=_patched_train_step),
        mock.patch.object(Learner, "wait_for_pending_checkpoints", autospec=True, return_value=None) as wait_mock,
    ):
        orchestrator = MuZeroOrchestrator(async_config)
        start = time.perf_counter()
        orchestrator.run()
        duration = time.perf_counter() - start

    assert orchestrator.training_step == async_config.training.training_steps
    assert orchestrator._buffer_size() <= async_config.replay_buffer.capacity
    wait_mock.assert_called()
    # Slow actor runs should still complete quickly in concurrent mode.
    assert duration < 10.0, "Async orchestrator stalled despite slow actors"
