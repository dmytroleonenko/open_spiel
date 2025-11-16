import numpy as np

from open_spiel.python.algorithms.muzero_jax.services.inference_client import (
    LocalBatchingInferenceClient,
)


class DummyNetwork:
    def initial_inference(self, obs, training=False):
        return obs, obs, obs, obs, None, None

    def recurrent_inference(self, hidden, action, training=False):
        return hidden, hidden, hidden, hidden, None, None


def test_local_batching_refresh_noop():
    client = LocalBatchingInferenceClient(DummyNetwork(), batch_size=1, max_wait_ms=1)
    client.refresh_params()  # Should be a no-op
    out = client.initial_inference(np.zeros((1, 1)))
    assert out[0] is not None
    client.close()


def test_grpc_client_refresh_noop(monkeypatch):
    pass  # remote inference removed
