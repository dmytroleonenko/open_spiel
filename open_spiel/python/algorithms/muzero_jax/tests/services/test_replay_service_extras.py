import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    InMemoryReplayService,
    RemoteReplayBufferAdapter,
    GrpcReplayServer,
    GrpcReplayClient,
)


def test_replay_service_get_and_update_targets():
    service = InMemoryReplayService(capacity=4, alpha=0.0)
    traj_id = service.append(
        {
            "observations": [[0.0]],
            "actions": [0],
            "rewards": [0.0],
            "policy_targets": [[0.5, 0.5]],
            "value_targets": [0.0],
        },
        priority=1.0,
    )
    got = service.get_trajectory(traj_id)
    np.testing.assert_allclose(got["policy_targets"][0], [0.5, 0.5])

    service.update_targets(traj_id, policy_targets=[[0.2, 0.8]], value_targets=[1.0])
    updated = service.get_trajectory(traj_id)
    np.testing.assert_allclose(updated["policy_targets"][0], [0.2, 0.8])
    np.testing.assert_allclose(updated["value_targets"][0], 1.0)


def test_remote_adapter_importance_weights_and_size():
    backend = InMemoryReplayService(capacity=4, alpha=0.0)
    server = GrpcReplayServer(backend)
    server.start()
    client = GrpcReplayClient(server.endpoint)
    adapter = RemoteReplayBufferAdapter(client, prioritized=True)

    adapter.add_trajectory(
        {
            "observations": [[0.0]],
            "actions": [0],
            "rewards": [0.0],
            "policy_targets": [[1.0, 0.0]],
            "value_targets": [0.0],
        }
    )
    # Force size exception path by monkeypatching client.size to raise
    class DummyClient:
        def size(self):
            raise RuntimeError("boom")
    adapter._client = DummyClient()
    assert len(adapter) >= 1
    adapter._client = client
    assert len(adapter) >= 1
    trajs, ids, weights = adapter.sample_batch_with_ids(batch_size=1, rng_key=None)
    assert weights.shape == (1,)
    assert weights[0] == 1.0
    # Non-prioritized path returns two elements
    trajs2, ids2 = RemoteReplayBufferAdapter(client, prioritized=False).sample_batch_with_ids(
        batch_size=1, rng_key=None
    )
    assert len(trajs2) == 1
    assert ids2.shape == (1,)

    client.close()
    server.stop()


def test_invalid_capacity_and_alpha_errors():
    with pytest.raises(ValueError):
        InMemoryReplayService(capacity=0)
    with pytest.raises(ValueError):
        InMemoryReplayService(capacity=1, alpha=-0.1)


def test_sample_batch_input_validation():
    service = InMemoryReplayService(capacity=2, alpha=0.0)
    with pytest.raises(ValueError):
        service.sample(0)
    service.append(
        {
            "observations": [[0.0]],
            "actions": [0],
            "rewards": [0.0],
            "policy_targets": [[0.5, 0.5]],
            "value_targets": [0.0],
        }
    )
    with pytest.raises(TimeoutError):
        # request more than available; use short timeout to avoid blocking
        service.sample(2, timeout_s=0.01)
