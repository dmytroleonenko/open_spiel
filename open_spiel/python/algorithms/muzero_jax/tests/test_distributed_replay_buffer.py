import pytest

from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayClient,
    GrpcReplayServer,
    InMemoryReplayService,
    RemoteReplayBufferAdapter,
)


def test_remote_adapter_round_trip_add_and_sample():
    backend = InMemoryReplayService(capacity=8, alpha=0.6)
    server = GrpcReplayServer(backend)
    server.start()
    client = GrpcReplayClient(server.endpoint)
    adapter = RemoteReplayBufferAdapter(client, prioritized=True)

    traj = {
        "observations": [[0.0, 0.0]],
        "actions": [1],
        "rewards": [0.0],
        "policy_targets": [[0.5, 0.5]],
        "value_targets": [0.0],
    }
    adapter.add_trajectory(traj)

    sample, ids, weights = adapter.sample_batch_with_ids(batch_size=1, rng_key=None)
    assert len(sample) == 1
    assert ids.shape == (1,)
    assert weights.shape == (1,)

    fetched = adapter.get_trajectory(int(ids[0]))
    assert fetched["actions"][0] == 1

    adapter.update_trajectory_targets(int(ids[0]), policy_targets=[[0.7, 0.3]])
    updated = adapter.get_trajectory(int(ids[0]))
    assert updated["policy_targets"][0][0] == pytest.approx(0.7)

    adapter.update_priorities_by_ids(ids, [3.0])
    _, _, weights_after = adapter.sample_batch_with_ids(batch_size=1, rng_key=None)
    assert weights_after[0] == pytest.approx(1.0)  # importance weights default to 1

    client.close()
    server.stop()
