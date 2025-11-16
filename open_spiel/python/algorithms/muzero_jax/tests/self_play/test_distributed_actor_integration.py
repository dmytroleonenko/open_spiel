import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import build_replay_buffer
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayServer,
    InMemoryReplayService,
)
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer


def test_remote_buffer_len_gate():
    """Learner gating should see buffer size via adapter len()."""
    backend = InMemoryReplayService(capacity=4, alpha=0.0)
    server = GrpcReplayServer(backend)
    server.start()

    cfg = type("C", (), {})()
    cfg.game = type("G", (), {"name": "tic_tac_toe"})
    cfg.replay_buffer = type(
        "RB",
        (),
        {
            "capacity": 4,
            "min_size_to_sample": 1,
            "max_trajectory_length": 2,
            "priority_alpha": 0.0,
            "remote_enabled": True,
            "rpc_endpoint": server.endpoint,
        },
    )()

    buf = build_replay_buffer(cfg, observation_shape=(3,), num_actions=2, game_obj=None, register_replay_client=None)
    assert len(buf) == 0
    traj = {
        "observations": np.zeros((1, 3), dtype=np.float32),
        "actions": np.array([0], dtype=np.int32),
        "rewards": np.zeros(1, dtype=np.float32),
        "policy_targets": np.zeros((1, 2), dtype=np.float32),
        "value_targets": np.zeros(1, dtype=np.float32),
    }
    buf.add_trajectory(traj)
    assert len(buf) >= 1

    server.stop()
