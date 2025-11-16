"""Orchestrator integration with all services on localhost via gRPC."""

from pathlib import Path
import tempfile

import pytest
from omegaconf import OmegaConf

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    InMemoryReplayService,
    GrpcReplayServer,
)
from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherServer,
    LocalParameterPublisher,
)
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper


def _base_config():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    return OmegaConf.load(cfg_path)


@pytest.mark.timeout(8)
def test_orchestrator_runs_with_localhost_remote_services():
    # Load base config and shrink for speed.
    cfg = _base_config()
    cfg.game.name = "tic_tac_toe"
    cfg.training.batch_size = 1
    cfg.training.training_steps = 2
    cfg.training.warmup_steps = 0
    cfg.training.num_unroll_steps = 1
    cfg.training.td_steps = 1
    cfg.training.start_transitions = 1
    cfg.training.num_actions = 9
    cfg.training.use_projection = False
    cfg.output.log_interval = 1
    cfg.output.checkpoint_interval = 1
    cfg.exp_config.debug = True
    cfg.actors.num_actors = 1
    cfg.actors.episodes_per_actor = 1
    cfg.actors.max_episode_length = 2
    cfg.bootstrap.enabled = False
    cfg.resource_management.training_phase_steps = 1
    cfg.resource_management.selfplay_phase_episodes = 1
    cfg.replay_buffer.capacity = 16
    cfg.replay_buffer.min_size_to_sample = 1
    cfg.replay_buffer.max_trajectory_length = 2
    cfg.replay_buffer.priority_alpha = 0.6
    cfg.replay_buffer.priority_beta_start = 0.4
    cfg.replay_buffer.priority_beta_steps = 1
    cfg.publisher.publish_interval = 1
    cfg.wandb.enabled = False

    # Auto-start services (leave endpoints empty).
    cfg.replay_buffer.remote_enabled = True
    cfg.replay_buffer.rpc_endpoint = ""
    cfg.publisher.remote_enabled = True
    cfg.publisher.rpc_endpoint = ""
    cfg.inference.enable_local_batching = False

    with tempfile.TemporaryDirectory(prefix="muzero_localhost_remote_") as tmpdir:
        cfg.output.save_path = tmpdir + "/"
        orch = MuZeroOrchestrator(cfg)
        metrics = orch.run_training_phase()
        assert metrics is None or isinstance(metrics, dict)
        # Auto-launched endpoints should now be populated for replay/publisher.
        assert cfg.replay_buffer.rpc_endpoint
        assert cfg.publisher.rpc_endpoint
