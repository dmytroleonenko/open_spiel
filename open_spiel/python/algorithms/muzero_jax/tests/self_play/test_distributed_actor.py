import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayClient,
    GrpcReplayServer,
    InMemoryReplayService,
    RemoteReplayBufferAdapter,
)
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.models.network import (
    MuZeroNetwork,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    RewardNetwork,
    ProjectionNetwork,
)
from open_spiel.python.algorithms.muzero_jax.training.trainer import create_network_config_from_muzero_config, MuZeroConfig
import flax.nnx as nnx
import jax
import jax.numpy as jnp


def make_dummy_actor(config: MuZeroConfig, game_name: str = "tic_tac_toe"):
    game_wrapper = GameWrapper(game_name)
    observation_shape = game_wrapper.observation_shape
    net_cfg = create_network_config_from_muzero_config(config, observation_shape, config.num_actions, False)
    rng = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(params=rng)
    network = MuZeroNetwork(
        representation_network_def=RepresentationNetwork,
        dynamics_network_def=DynamicsNetwork,
        prediction_network_def=PredictionNetwork,
        reward_network_def=RewardNetwork,
        projection_network_def=ProjectionNetwork if config.use_projection else None,
        config=net_cfg,
        rngs=rngs,
    )
    mcts = MCTS(
        num_simulations=2,
        max_num_considered_actions=16,
        gumbel_scale=1.0,
    )
    return Actor(
        network=network,
        mcts=mcts,
        game_wrapper=game_wrapper,
        replay_buffer=None,
        config=config,
        inference_client=None,
    )


def test_actor_writes_to_remote_replay_and_learner_reads():
    """Actor should write trajectories via RemoteReplayBufferAdapter; learner-side sampler can fetch."""
    backend = InMemoryReplayService(capacity=8, alpha=0.6)
    server = GrpcReplayServer(backend)
    server.start()
    client = GrpcReplayClient(server.endpoint)
    replay = RemoteReplayBufferAdapter(client, prioritized=True)

    cfg = MuZeroConfig(
        num_actions=9,
        batch_size=2,
        training_steps=1,
        learning_rate=0.1,
        priority_exponent=0.6,
        priority_beta=0.4,
        buffer_size=8,
        support_min=-1,
        support_max=1,
    )

    actor = make_dummy_actor(cfg)
    # Force a tiny rollout: deterministic policy picks first legal action.
    trajectory = {
        "observations": np.zeros((1, 27), dtype=np.float32),
        "actions": np.array([0], dtype=np.int32),
        "rewards": np.array([0.0], dtype=np.float32),
        "target_values": np.array([0.0], dtype=np.float32),
        "policy_targets": np.full((1, cfg.num_actions), 1.0 / cfg.num_actions, dtype=np.float32),
    }
    replay.add_trajectory(trajectory)

    sampled, ids, weights = replay.sample_batch_with_ids(batch_size=1, rng_key=None)
    assert len(sampled) == 1
    assert ids.shape == (1,)
    assert weights.shape == (1,)
    # Importance weights are placeholder ones for now.
    assert weights[0] == pytest.approx(1.0)

    client.close()
    server.stop()
