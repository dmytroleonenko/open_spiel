import threading
import queue

import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import ActorWorker, TrajectoryPacket


class DummyActor:
    def __init__(self):
        self.refreshed = 0

    def refresh_params(self):
        self.refreshed += 1

    def play_episode(self, rng):
        # Immediately stop after one trajectory
        return {
            "observations": np.zeros((1, 1)),
            "actions": np.zeros(1),
            "rewards": np.zeros(1),
            "policy_targets": [np.zeros(1)],
            "value_targets": np.zeros(1),
        }


def test_actor_worker_calls_refresh_params_once():
    traj_queue: queue.Queue[TrajectoryPacket] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    bootstrap_event = threading.Event()
    error_queue: queue.Queue[BaseException] = queue.Queue()

    actor = DummyActor()

    worker = ActorWorker(
        worker_id=0,
        trajectory_queue=traj_queue,
        stop_event=stop_event,
        bootstrap_event=bootstrap_event,
        muzero_actor=actor,
        bootstrap_actor=None,
        checkpoint_dir="",
        checkpoint_sync_interval=0,
        rng_seed=0,
        error_queue=error_queue,
    )
    worker.start()

    # Wait for one trajectory then stop
    packet = traj_queue.get(timeout=2.0)
    assert packet is not None
    stop_event.set()
    worker.join(timeout=2.0)

    assert actor.refreshed >= 1
