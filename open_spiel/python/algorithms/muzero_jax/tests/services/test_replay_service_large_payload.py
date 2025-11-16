"""Check that large batched RPC responses don't hit gRPC message limits."""

import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.services import replay_service as rs


@pytest.mark.timeout(5)
def test_grpc_replay_handles_large_messages():
    # Construct trajectories whose serialized payload is several MB.
    obs = np.zeros((64, 64), dtype=np.float32)  # 16 KB per obs approx
    trajectory = {
        "observations": obs,
        "actions": np.zeros(64, dtype=np.int32),
        "rewards": np.zeros(64, dtype=np.float32),
        "policy_targets": np.zeros((64, 16), dtype=np.float32),
        "value_targets": np.zeros(64, dtype=np.float32),
    }

    backend = rs.InMemoryReplayService(capacity=200, alpha=0.0)
    server = rs.GrpcReplayServer(backend, max_message_mb=64)
    server.start()

    client = rs.GrpcReplayClient(server.endpoint, timeout_s=2.0, max_message_mb=64)

    for _ in range(128):
        client.append(trajectory, priority=1.0)

    trajs, ids, prios = client.sample(batch_size=64)
    # Should succeed without RESOURCE_EXHAUSTED.
    assert len(trajs) == 64
    assert ids.shape[0] == 64
    assert prios.shape[0] == 64

    client.close()
    server.stop()
