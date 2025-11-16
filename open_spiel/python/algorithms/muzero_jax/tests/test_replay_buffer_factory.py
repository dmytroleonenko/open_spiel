import pytest
import numpy as np
from omegaconf import OmegaConf

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import build_replay_buffer
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayServer,
    InMemoryReplayService,
    RemoteReplayBufferAdapter,
)


def _base_config():
    return OmegaConf.create(
        {
            "game": {"name": "tic_tac_toe"},
            "replay_buffer": {
                "capacity": 8,
                "min_size_to_sample": 2,
                "max_trajectory_length": 4,
                "priority_alpha": 0.6,
                "remote_enabled": False,
                "rpc_endpoint": "",
            },
        }
    )


def test_build_replay_buffer_local_prioritized():
    cfg = _base_config()
    buf = build_replay_buffer(cfg, observation_shape=(3,), num_actions=2, game_obj=None, register_replay_client=None)
    # PrioritizedTrajectoryBuffer has update_priorities_by_ids; adapter lacks it.
    assert hasattr(buf, "update_priorities_by_ids")


def test_build_replay_buffer_remote_switches_to_adapter():
    backend = InMemoryReplayService(capacity=8, alpha=0.6)
    server = GrpcReplayServer(backend)
    server.start()

    cfg = _base_config()
    cfg.replay_buffer.remote_enabled = True
    cfg.replay_buffer.rpc_endpoint = server.endpoint

    registered = []

    def _register(client):
        registered.append(client)

    buf = build_replay_buffer(
        cfg,
        observation_shape=(3,),
        num_actions=2,
        game_obj=None,
        register_replay_client=_register,
    )
    assert isinstance(buf, RemoteReplayBufferAdapter)
    assert len(registered) == 1
    traj = {
        "observations": np.zeros((1, 3), dtype=np.float32),
        "actions": np.array([0], dtype=np.int32),
        "rewards": np.array([0.0], dtype=np.float32),
        "policy_targets": np.zeros((1, 2), dtype=np.float32),
        "value_targets": np.zeros(1, dtype=np.float32),
    }
    buf.add_trajectory(traj)
    sample, ids, weights = buf.sample_batch_with_ids(batch_size=1, rng_key=None)
    assert len(sample) == 1
    assert weights.shape == (1,)
    assert len(buf) >= 1  # post-append size should reflect data

    registered[0].close()
    server.stop()
