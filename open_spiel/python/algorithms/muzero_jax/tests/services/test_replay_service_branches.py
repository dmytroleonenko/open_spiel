import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.services import replay_service as rs


def test_eviction_and_sampling_paths():
    service = rs.InMemoryReplayService(capacity=2, alpha=0.5)
    id1 = service.append({"a": 1}, priority=1.0)
    id2 = service.append({"a": 2}, priority=2.0)
    id3 = service.append({"a": 3}, priority=3.0)  # evicts id1
    assert id1 not in service._priorities
    trajs, ids, weights = service.sample(batch_size=2)
    assert len(trajs) == 2
    service.update_priorities(ids, [2.0, 4.0])


def test_timeout_and_error_paths():
    service = rs.InMemoryReplayService(capacity=1, alpha=0.0)
    with pytest.raises(TimeoutError):
        service.sample(batch_size=1, timeout_s=0.1)
    with pytest.raises(ValueError):
        service.update_priorities([1], [1, 2])
    with pytest.raises(KeyError):
        service.get_trajectory(99)
    service.append({"a": 1}, priority=1.0)
    with pytest.raises(KeyError):
        service.update_targets(99, policy_targets=None, value_targets=None)
    with pytest.raises(ValueError):
        service._select_ids(batch_size=5, rng_key=None)


def test_adapter_two_tuple_path(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def sample(self, batch_size):
            self.calls += 1
            return ([{"a": 1}], np.array([0]), np.array([1.0]))

        def append(self, trajectory, priority):
            self.calls += 1

        def size(self):
            return 1

    adapter = rs.RemoteReplayBufferAdapter(FakeClient(), prioritized=False)
    out = adapter.sample_batch(batch_size=1)
    assert isinstance(out, list)


def test_adapter_prioritized_weights():
    class FakeClient:
        def sample(self, batch_size):
            return ([{"a": 1}], np.array([0]), np.array([1.0]))

        def append(self, trajectory, priority):
            pass

        def size(self):
            return 1

    adapter = rs.RemoteReplayBufferAdapter(FakeClient(), prioritized=True)
    trajs, ids, weights = adapter.sample_batch(batch_size=1)
    assert list(weights) == [1.0]


def test_adapter_update_priorities_alias():
    class FakeClient:
        def update_priorities(self, ids, priorities):
            self.called = True

    client = FakeClient()
    adapter = rs.RemoteReplayBufferAdapter(client, prioritized=True)
    adapter.update_priorities([0], [1.0])
    assert getattr(client, "called", False)
