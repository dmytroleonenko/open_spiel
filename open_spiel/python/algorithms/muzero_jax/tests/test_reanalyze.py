import numpy as np
import jax
import jax.numpy as jnp
import pytest

from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    PrioritizedTrajectoryBuffer,
    TrajectoryBuffer,
)
from open_spiel.python.algorithms.muzero_jax.self_play.reanalyze_worker import ReanalyzeWorker
from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig


class DummyNetwork:
    """Minimal MuZeroNetwork stub for reanalysis tests."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def initial_inference(self, observations, training: bool = False):
        batch = observations.shape[0]
        hidden = jnp.zeros((batch, 4))
        reward = jnp.zeros((batch, 1))
        values = jnp.arange(batch, dtype=jnp.float32).reshape(batch, 1)
        policy_logits = jnp.tile(jnp.linspace(0.1, 1.0, self.num_actions), (batch, 1))
        return hidden, reward, values, policy_logits

    def recurrent_inference(self, hidden_state, action, training: bool = False):
        batch = hidden_state.shape[0]
        next_hidden = hidden_state
        reward = jnp.zeros((batch, 1))
        values = jnp.ones((batch, 1))
        policy_logits = jnp.tile(jnp.linspace(0.1, 1.0, self.num_actions), (batch, 1))
        return next_hidden, reward, values, policy_logits


class DummySupportNetwork(DummyNetwork):
    """Network that emits categorical supports for value targets."""

    def __init__(self, num_actions: int, atoms: jnp.ndarray):
        super().__init__(num_actions)
        self._atoms = atoms

    def initial_inference(self, observations, training: bool = False):
        batch = observations.shape[0]
        hidden = jnp.zeros((batch, 4))
        reward = jnp.zeros((batch, 1))
        values = jnp.tile(self._atoms, (batch, 1))
        policy_logits = jnp.tile(jnp.linspace(0.1, 1.0, self.num_actions), (batch, 1))
        return hidden, reward, values, policy_logits


class DummyFlatValueNetwork(DummyNetwork):
    """Network that emits 1-D value predictions."""

    def initial_inference(self, observations, training: bool = False):
        batch = observations.shape[0]
        hidden = jnp.zeros((batch, 4))
        reward = jnp.zeros((batch, 1))
        values = jnp.linspace(0.0, 1.0, batch)
        policy_logits = jnp.tile(jnp.linspace(0.1, 1.0, self.num_actions), (batch, 1))
        return hidden, reward, values, policy_logits


def _make_trajectory(num_actions: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(length, 4)).astype(np.float32)
    actions = np.arange(length, dtype=np.int32)
    rewards = np.zeros(length, dtype=np.float32)
    target_values = np.zeros(length, dtype=np.float32)
    policy_targets = [np.ones(num_actions, dtype=np.float32) / num_actions for _ in range(length)]
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'target_values': target_values,
        'value_targets': target_values,
        'policy_targets': policy_targets,
    }


def _fake_policy_reanalysis_fn(model, observations, config, training=False, rng_key=None):
    batch, steps = observations.shape[:2]
    num_actions = config.num_actions
    logits = jnp.arange(batch * steps * num_actions, dtype=jnp.float32).reshape(batch, steps, num_actions) + 1.0
    return logits / jnp.sum(logits, axis=-1, keepdims=True)


def test_reanalyze_worker_updates_replay_buffer_targets():
    config = MuZeroConfig(num_actions=3, num_unroll_steps=2)

    buffer = TrajectoryBuffer(
        capacity=2,
        observation_shape=(4,),
        num_actions=config.num_actions,
        max_trajectory_length=10,
    )
    buffer.add_trajectory(_make_trajectory(config.num_actions, 5, seed=1))
    buffer.add_trajectory(_make_trajectory(config.num_actions, 6, seed=2))

    worker = ReanalyzeWorker(
        config=config,
        model=DummyNetwork(config.num_actions),
        replay_buffer=buffer,
        sample_batch_size=2,
        rng_key=jax.random.PRNGKey(0),
        policy_reanalysis_fn=_fake_policy_reanalysis_fn,
    )

    captured_obs = worker._prepare_observation_batch([traj for _, traj in buffer.iter_trajectories()])
    expected_policies = np.array(_fake_policy_reanalysis_fn(worker.model, captured_obs, config, False, None))
    expected_values = np.array(worker._compute_value_targets(captured_obs))

    result = worker.run_once()
    assert result.sampled == 2
    assert result.updated == 2

    steps = config.num_unroll_steps + 1
    for idx, (_, trajectory) in enumerate(buffer.iter_trajectories()):
        stacked_policy = np.stack(trajectory['policy_targets'][:steps], axis=0)
        np.testing.assert_allclose(stacked_policy, expected_policies[idx], atol=1e-6)
        np.testing.assert_allclose(trajectory['value_targets'][:steps], expected_values[idx], atol=1e-6)


def test_reanalyze_worker_updates_priorities():
    config = MuZeroConfig(num_actions=3, num_unroll_steps=2, min_priority=1e-5)

    buffer = PrioritizedTrajectoryBuffer(
        capacity=2,
        observation_shape=(4,),
        num_actions=config.num_actions,
        alpha=0.6,
        max_trajectory_length=10,
    )
    buffer.add_trajectory(_make_trajectory(config.num_actions, 5, seed=3), priority=0.5)
    buffer.add_trajectory(_make_trajectory(config.num_actions, 6, seed=4), priority=0.5)

    worker = ReanalyzeWorker(
        config=config,
        model=DummyNetwork(config.num_actions),
        replay_buffer=buffer,
        sample_batch_size=2,
        rng_key=jax.random.PRNGKey(1),
        policy_reanalysis_fn=_fake_policy_reanalysis_fn,
    )

    captured_obs = worker._prepare_observation_batch([traj for _, traj in buffer.iter_trajectories()])
    expected_policies = np.array(_fake_policy_reanalysis_fn(worker.model, captured_obs, config, False, None))

    old_policies = [
        np.stack(traj['policy_targets'][: config.num_unroll_steps + 1], axis=0)
        for _, traj in buffer.iter_trajectories()
    ]
    ordered_ids = [traj_id for traj_id, _ in buffer.iter_trajectories()]

    worker.run_once()

    for idx, traj_id in enumerate(ordered_ids):
        expected_delta = np.mean(np.abs(expected_policies[idx] - old_policies[idx]))
        priority = buffer.get_priority(traj_id)
        assert pytest.approx(priority, rel=1e-6) == max(expected_delta, config.min_priority)


def test_prepare_observation_batch_pads_short_trajectories():
    config = MuZeroConfig(num_actions=2, num_unroll_steps=3)
    worker = ReanalyzeWorker(
        config=config,
        model=DummyNetwork(config.num_actions),
        replay_buffer=TrajectoryBuffer(capacity=1, observation_shape=(4,), num_actions=2, max_trajectory_length=5),
        sample_batch_size=1,
    )
    obs = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    padded = worker._prepare_observation_batch([{'observations': obs}])
    assert padded.shape == (1, 4, 4)
    np.testing.assert_array_equal(padded[0, -1], obs[-1])


def test_compute_value_targets_support_branch():
    config = MuZeroConfig(num_actions=2, num_unroll_steps=1, support_min=-1, support_max=1)
    atoms = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)  # one-hot for +1 support
    worker = ReanalyzeWorker(
        config=config,
        model=DummySupportNetwork(config.num_actions, atoms),
        replay_buffer=TrajectoryBuffer(capacity=1, observation_shape=(2,), num_actions=2),
        sample_batch_size=1,
    )
    observations = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    values = worker._compute_value_targets(observations)
    np.testing.assert_allclose(values, np.ones((1, 2)), atol=1e-4)


def test_compute_value_targets_vector_branch():
    config = MuZeroConfig(num_actions=2, num_unroll_steps=0)
    worker = ReanalyzeWorker(
        config=config,
        model=DummyFlatValueNetwork(config.num_actions),
        replay_buffer=TrajectoryBuffer(capacity=1, observation_shape=(2,), num_actions=2),
        sample_batch_size=1,
    )
    observations = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    values = worker._compute_value_targets(observations)
    np.testing.assert_allclose(values, np.zeros((1, 1)), atol=1e-5)


def test_run_once_returns_zero_when_buffer_empty():
    config = MuZeroConfig(num_actions=2, num_unroll_steps=1)
    buffer = TrajectoryBuffer(capacity=1, observation_shape=(2,), num_actions=2)
    worker = ReanalyzeWorker(
        config=config,
        model=DummyNetwork(config.num_actions),
        replay_buffer=buffer,
        sample_batch_size=1,
    )
    result = worker.run_once()
    assert result.sampled == 0
    assert result.updated == 0
    assert result.buffer_size == 0
