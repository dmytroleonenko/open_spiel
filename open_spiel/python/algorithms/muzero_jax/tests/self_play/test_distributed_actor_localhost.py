"""End-to-end actor run against local gRPC services (replay + inference + publisher).

Keeps runtime < 2s by using tiny tic-tac-toe network and single-step episode.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
from open_spiel.python.algorithms.muzero_jax.models.network import (
    MuZeroNetwork,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    RewardNetwork,
    ProjectionNetwork,
)
from open_spiel.python.algorithms.muzero_jax.services.inference_client import LocalInferenceClient
from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherClient,
    GrpcParameterPublisherServer,
    LocalParameterPublisher,
)
from open_spiel.python.algorithms.muzero_jax.services.parameter_client_inference_adapter import (
    ParameterRefreshingInferenceClient,
)
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayClient,
    GrpcReplayServer,
    InMemoryReplayService,
    RemoteReplayBufferAdapter,
)
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig,
    create_network_config_from_muzero_config,
)


def _small_network(cfg: MuZeroConfig, game_name: str = "tic_tac_toe"):
    gw = GameWrapper(game_name)
    obs_shape = gw.observation_shape
    net_cfg = create_network_config_from_muzero_config(cfg, obs_shape, cfg.num_actions, False)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    return MuZeroNetwork(
        representation_network_def=RepresentationNetwork,
        dynamics_network_def=DynamicsNetwork,
        prediction_network_def=PredictionNetwork,
        reward_network_def=RewardNetwork,
        projection_network_def=ProjectionNetwork if cfg.use_projection else None,
        config=net_cfg,
        rngs=rngs,
    )


@pytest.mark.timeout(6)
def test_actor_remote_inference_and_replay_end_to_end():
    cfg = MuZeroConfig(
        num_actions=9,
        batch_size=1,
        training_steps=1,
        learning_rate=0.1,
        priority_exponent=0.6,
        priority_beta=0.4,
        buffer_size=8,
        support_min=-1,
        support_max=1,
    )

    # Spin up services on ephemeral ports.
    replay_backend = InMemoryReplayService(capacity=8, alpha=0.6)
    replay_server = GrpcReplayServer(replay_backend)
    replay_server.start()
    replay_client = GrpcReplayClient(replay_server.endpoint)
    replay_adapter = RemoteReplayBufferAdapter(replay_client, prioritized=True)

    param_backend = LocalParameterPublisher()
    param_server = GrpcParameterPublisherServer(param_backend)
    param_server.start()
    param_client = GrpcParameterPublisherClient(param_server.endpoint, timeout_s=1.0)

    network = _small_network(cfg)
    infer_client = LocalInferenceClient(network)
    wrapped_client = ParameterRefreshingInferenceClient(infer_client, param_client)

    # Publish once so actor refresh doesn't block.
    param_backend.publish({"dummy": 1}, step=1)

    actor = Actor(
        network=None,  # Actor will use inference client path
        mcts=MCTS(num_simulations=2, max_num_considered_actions=9, gumbel_scale=1.0),
        game_wrapper=GameWrapper("tic_tac_toe"),
        replay_buffer=replay_adapter,
        config=cfg,
        inference_client=wrapped_client,
    )
    actor.refresh_params()

    # Run a single fake episode by manually crafting a trajectory; avoid full rollout costs.
    traj = {
        "observations": np.zeros((1, 27), dtype=np.float32),
        "actions": np.array([0], dtype=np.int32),
        "rewards": np.array([0.0], dtype=np.float32),
        "target_values": np.array([0.0], dtype=np.float32),
        "policy_targets": np.full((1, cfg.num_actions), 1.0 / cfg.num_actions, dtype=np.float32),
    }
    replay_adapter.add_trajectory(traj)

    sampled, ids, weights = replay_adapter.sample_batch_with_ids(batch_size=1, rng_key=None)
    assert len(sampled) == 1
    assert ids.shape == (1,)
    assert weights.shape == (1,)

    # Latest params should be visible via publisher client.
    params, step = param_client.latest()
    assert step == 1 and params["dummy"] == 1

    # Clean up
    replay_client.close()
    infer_client.close()
    param_client.close()
    replay_server.stop()
    param_server.stop()
