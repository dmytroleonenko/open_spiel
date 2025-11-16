import types

import jax
import pytest

import open_spiel.python.algorithms.muzero_jax.run_muzero_jax as rm


class DummyBuffer:
    def __init__(self, sample_out):
        self.sample_out = sample_out
        self.calls = 0

    def __len__(self):
        return 10

    def sample_batch(self, batch_size, rng_key=None):
        self.calls += 1
        return self.sample_out


@pytest.mark.parametrize("sample_out", [
    ([{"obs": 1}], [1], None),  # 3-tuple
    ([{"obs": 1}], [1]),        # 2-tuple
    ([{"obs": 1}],),            # 1-tuple
])
def test_perform_training_step_tuple_outputs(monkeypatch, sample_out):
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = types.SimpleNamespace(
        training=types.SimpleNamespace(batch_size=1),
        output=types.SimpleNamespace(log_interval=1),
        publisher=types.SimpleNamespace(publish_interval=0),
        wandb=types.SimpleNamespace(enabled=False),
    )
    orch.muzero_config = types.SimpleNamespace(min_priority=0.0)
    orch.replay_buffer = DummyBuffer(sample_out)
    import threading
    orch._buffer_lock = threading.Lock()
    orch._min_buffer_before_training = lambda: 1
    orch._maybe_publish_params = lambda force=False: None
    orch._update_priorities = lambda sampled_indices, metrics: None
    orch._maybe_run_periodic_evaluation = lambda: None
    orch._maybe_log_training_metrics = lambda metrics: None
    orch._convert_trajectories_to_batch = lambda *a, **k: {}
    orch.training_step = 0
    orch.learner = types.SimpleNamespace(train_step=lambda batch: {"priorities": [1.0]})
    orch.rng_key = jax.random.PRNGKey(0)

    metrics = orch._perform_training_step()
    assert metrics is not None
    assert orch.replay_buffer.calls == 1


def test_perform_training_step_empty_returns_none(monkeypatch):
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = types.SimpleNamespace(
        training=types.SimpleNamespace(batch_size=1),
        output=types.SimpleNamespace(log_interval=1),
        publisher=types.SimpleNamespace(publish_interval=0),
        wandb=types.SimpleNamespace(enabled=False),
    )
    orch.replay_buffer = DummyBuffer(sample_out=[])
    import threading
    orch._buffer_lock = threading.Lock()
    orch._min_buffer_before_training = lambda: 1
    orch.muzero_config = types.SimpleNamespace(min_priority=0.0)
    orch.learner = types.SimpleNamespace(train_step=lambda batch: {})
    orch._maybe_publish_params = lambda force=False: None
    orch._update_priorities = lambda sampled_indices, metrics: None
    orch._maybe_run_periodic_evaluation = lambda: None
    orch._maybe_log_training_metrics = lambda metrics: None
    orch._convert_trajectories_to_batch = lambda *a, **k: {}
    orch.training_step = 0
    orch.rng_key = jax.random.PRNGKey(0)

    assert orch._perform_training_step() is None


def test_perform_training_step_value_error_returns_none():
    class DummyBuffer:
        def __len__(self):
            return 10

        def sample_batch(self, *a, **k):
            raise ValueError("too small")

    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = types.SimpleNamespace(
        training=types.SimpleNamespace(batch_size=1),
        output=types.SimpleNamespace(log_interval=1),
        publisher=types.SimpleNamespace(publish_interval=0),
        wandb=types.SimpleNamespace(enabled=False),
    )
    orch.replay_buffer = DummyBuffer()
    import threading
    orch._buffer_lock = threading.Lock()
    orch._min_buffer_before_training = lambda: 1
    orch.muzero_config = types.SimpleNamespace(min_priority=0.0)
    orch.learner = types.SimpleNamespace(train_step=lambda batch: {})
    orch._maybe_publish_params = lambda force=False: None
    orch._update_priorities = lambda sampled_indices, metrics: None
    orch._maybe_run_periodic_evaluation = lambda: None
    orch._maybe_log_training_metrics = lambda metrics: None
    orch._convert_trajectories_to_batch = lambda *a, **k: {}
    orch.training_step = 0
    orch.rng_key = jax.random.PRNGKey(0)

    assert orch._perform_training_step() is None
