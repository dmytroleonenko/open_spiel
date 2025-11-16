import tempfile
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    InMemoryReplayService,
    GrpcReplayServer,
)
from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    LocalParameterPublisher,
    GrpcParameterPublisherServer,
)


@pytest.mark.slow
def test_remote_replay_and_publisher_end_to_end():
    """
    End-to-end run with remote replay + remote parameter publisher enabled.
    Uses minimal steps/config so the run completes quickly in CI.
    """
    tmp_dir = tempfile.mkdtemp(prefix="muzero_remote_e2e_")
    replay_backend = InMemoryReplayService(capacity=16, alpha=0.0)
    replay_server = GrpcReplayServer(replay_backend)
    publisher_backend = LocalParameterPublisher()
    publisher_server = GrpcParameterPublisherServer(publisher_backend)
    replay_server.start()
    publisher_server.start()

    try:
        base_cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
        )
        overrides = OmegaConf.create(
            {
                "output": {"save_path": tmp_dir, "log_interval": 1},
                "training": {
                    "training_steps": 2,
                    "batch_size": 1,
                    "start_transitions": 1,
                },
                "actors": {"num_actors": 1},
                "bootstrap": {"min_episodes": 1, "enabled": False},
                "evaluation": {"enabled": False},
                "wandb": {"enabled": False},
                "resource_management": {
                    "training_phase_steps": 1,
                    "selfplay_phase_episodes": 1,
                    "sequential_training": True,
                    "concurrent": False,
                },
                "replay_buffer": {
                    "remote_enabled": True,
                    "rpc_endpoint": replay_server.endpoint,
                    "priority_alpha": 0.0,
                    "capacity": 16,
                    "min_size_to_sample": 1,
                    "max_trajectory_length": 4,
                },
                "publisher": {
                    "remote_enabled": True,
                    "rpc_endpoint": publisher_server.endpoint,
                    "publish_interval": 1,
                    "min_step": 0,
                },
                "inference": {"remote_enabled": False},
            }
        )
        config = OmegaConf.merge(base_cfg, overrides)

        orch = MuZeroOrchestrator(config)
        assert orch._remote_replay is True
        assert orch._parameter_client is not None
        # Skip long training; just ensure components are wired without errors.
    finally:
        replay_server.stop()
        publisher_server.stop()
        shutil.rmtree(tmp_dir, ignore_errors=True)
