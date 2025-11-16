import types

import numpy as np

from open_spiel.python.algorithms.muzero_jax import run_muzero_jax as rm


class DummyReplayTwoTuple:
    """Mimics non-prioritized remote adapter returning (trajectories, ids)."""

    def sample_batch(self, batch_size, rng_key=None):
        traj = {
            "observations": np.zeros((1, 1)),
            "actions": np.array([0]),
            "rewards": np.array([0.0]),
            "policy_targets": np.array([[1.0]]),
            "value_targets": np.array([0.0]),
        }
        # return list for trajectories to match expected format
        return [traj], np.array([0])

    # Presence of update_priorities triggers prioritized path in orchestrator
    def update_priorities(self, ids, priorities):
        return None

    def __len__(self):
        return 10


def test_perform_training_step_handles_two_tuple_sample():
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = types.SimpleNamespace(
        training=types.SimpleNamespace(batch_size=1),
        replay_buffer=types.SimpleNamespace(min_size_to_sample=1),
    )
    orch.muzero_config = types.SimpleNamespace(
        start_transitions=1,
        batch_size=1,
        min_priority=1e-6,
        num_unroll_steps=1,
        td_steps=1,
        num_actions=1,
    )
    orch.replay_buffer = DummyReplayTwoTuple()
    orch._buffer_lock = rm.threading.Lock()
    orch.learner = types.SimpleNamespace(train_step=lambda batch: {"priorities": np.array([1.0])})
    orch.training_step = 0
    orch.rng_key = rm.jax.random.PRNGKey(0)
    # minimal game_wrapper stub
    orch.game_wrapper = types.SimpleNamespace(num_distinct_actions=lambda: 1)
    orch._parameter_client = None

    metrics = orch._perform_training_step()
    assert metrics is not None
