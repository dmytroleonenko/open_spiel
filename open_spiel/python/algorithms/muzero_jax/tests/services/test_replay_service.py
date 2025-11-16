import threading
import time

import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayClient,
    GrpcReplayServer,
    InMemoryReplayService,
)


def _make_traj(length: int = 3, obs_dim: int = 4, num_actions: int = 2):
    observations = np.ones((length, obs_dim), dtype=np.float32)
    actions = np.arange(length, dtype=np.int32)
    rewards = np.zeros(length, dtype=np.float32)
    policy_targets = np.full((length, num_actions), 0.5, dtype=np.float32)
    value_targets = np.zeros(length, dtype=np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "policy_targets": policy_targets,
        "value_targets": value_targets,
        "length": length,
    }


def test_local_append_sample_round_trip():
    service = InMemoryReplayService(capacity=8, alpha=0.6)
    ids = [service.append(_make_traj(), priority=prio) for prio in (1.0, 2.0)]

    trajectories, sample_ids, priorities = service.sample(batch_size=2, rng_key=None)

    assert set(sample_ids.tolist()) == set(ids)
    assert priorities.shape == (2,)
    # Returned priorities reflect the updates we supplied.
    assert sorted(priorities.tolist()) == [1.0, 2.0]
    assert len(trajectories) == 2
    for traj in trajectories:
        np.testing.assert_equal(traj["policy_targets"][0], np.array([0.5, 0.5], dtype=np.float32))


def test_update_priorities_affects_state():
    service = InMemoryReplayService(capacity=4, alpha=0.6)
    traj_id = service.append(_make_traj(), priority=1.0)
    service.update_priorities([traj_id], [3.5])

    _, ids, priorities = service.sample(batch_size=1, rng_key=None)
    assert ids[0] == traj_id
    assert np.allclose(priorities[0], 3.5)


@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_grpc_round_trip(alpha):
    backend = InMemoryReplayService(capacity=16, alpha=alpha)
    server = GrpcReplayServer(backend)
    server.start()
    client = GrpcReplayClient(server.endpoint)

    first = client.append(_make_traj(length=2), priority=1.5)
    second = client.append(_make_traj(length=2), priority=2.5)
    assert first != second

    trajectories, ids, priorities = client.sample(batch_size=2)
    assert set(ids.tolist()) == {first, second}
    # Priorities should round-trip even when alpha==0 (they're stored, not used).
    assert sorted(priorities.tolist()) == [1.5, 2.5]
    assert len(trajectories) == 2

    client.update_priorities([first], [7.0])
    _, ids_after, priorities_after = client.sample(batch_size=2)
    # One of them must carry the updated priority.
    assert 7.0 in priorities_after

    client.close()
    server.stop()


def test_sample_blocks_until_data_via_threads():
    service = InMemoryReplayService(capacity=4, alpha=0.6)

    results = {}

    def delayed_append():
        time.sleep(0.05)
        results["id"] = service.append(_make_traj(), priority=4.0)

    thread = threading.Thread(target=delayed_append)
    thread.start()
    trajectories, ids, priorities = service.sample(batch_size=1, rng_key=None)
    thread.join()

    assert ids[0] == results["id"]
    assert priorities[0] == pytest.approx(4.0)
    assert trajectories[0]["actions"].shape[0] == 3
