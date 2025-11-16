import pytest
import numpy as np

from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    InMemoryReplayService,
)


def test_sample_timeout_raises():
    service = InMemoryReplayService(capacity=1, alpha=0.0)
    with pytest.raises(TimeoutError):
        service.sample(1, timeout_s=0.01)


def test_select_ids_error():
    service = InMemoryReplayService(capacity=1, alpha=0.0)
    with pytest.raises(ValueError):
        service.sample(0)


def test_update_targets_key_error():
    service = InMemoryReplayService(capacity=1, alpha=0.0)
    with pytest.raises(KeyError):
        service.update_targets(999, policy_targets=[[1.0, 0.0]])


def test_update_priorities_missing_ok():
    service = InMemoryReplayService(capacity=2, alpha=0.0)
    # Missing id is silently ignored
    service.update_priorities([123], [1.0])


def test_deep_copy_preserves_data():
    service = InMemoryReplayService(capacity=2, alpha=0.0)
    tid = service.append(
        {
            "observations": [[1.0]],
            "actions": [0],
            "rewards": [0.0],
            "policy_targets": [[0.1, 0.9]],
            "value_targets": [0.5],
        },
        priority=1.0,
    )
    got = service.get_trajectory(tid)
    got["observations"][0][0] = 9.0
    orig = service.get_trajectory(tid)
    assert orig["observations"][0][0] == 1.0
