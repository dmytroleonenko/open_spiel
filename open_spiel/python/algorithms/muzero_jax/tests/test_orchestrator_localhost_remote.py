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
from open_spiel.python.algorithms.muzero_jax.services.inference_client import GrpcInferenceServer
from open_spiel.python.algorithms.muzero_jax.models.network import (
    MuZeroNetwork,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    RewardNetwork,
    ProjectionNetwork,
)
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.training.trainer import create_network_config_from_muzero_config, MuZeroConfig
import jax
import flax.nnx as nnx


def _base_config():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    return OmegaConf.load(cfg_path)


def _tiny_network(game_name: str = "tic_tac_toe"):
    gw = GameWrapper(game_name)
    mu_cfg = MuZeroConfig(
        num_actions=gw.num_distinct_actions(),
        batch_size=1,
        training_steps=1,
        learning_rate=0.1,
        priority_exponent=0.6,
        priority_beta=0.4,
        buffer_size=16,
        support_min=-1,
        support_max=1,
    )
    net_cfg = create_network_config_from_muzero_config(mu_cfg, gw.observation_shape, mu_cfg.num_actions, False)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    return MuZeroNetwork(
        representation_network_def=RepresentationNetwork,
        dynamics_network_def=DynamicsNetwork,
        prediction_network_def=PredictionNetwork,
        reward_network_def=RewardNetwork,
        projection_network_def=ProjectionNetwork if mu_cfg.use_projection else None,
        config=net_cfg,
        rngs=rngs,
    )


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
    cfg.inference.remote_enabled = True
    cfg.inference.rpc_endpoint = ""
    cfg.inference.timeout_s = 1.0

    with tempfile.TemporaryDirectory(prefix="muzero_localhost_remote_") as tmpdir:
        cfg.output.save_path = tmpdir + "/"
        orch = MuZeroOrchestrator(cfg)
        metrics = orch.run_training_phase()
        assert metrics is None or isinstance(metrics, dict)
        # Auto-launched endpoints should now be populated (unless TPU fallback disables remote inference).
        assert cfg.replay_buffer.rpc_endpoint
        assert cfg.publisher.rpc_endpoint
        assert cfg.inference.rpc_endpoint or not cfg.inference.remote_enabled
