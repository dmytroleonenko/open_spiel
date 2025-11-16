import numpy as np

from open_spiel.python.algorithms.muzero_jax.services.parameter_client_inference_adapter import (
    ParameterRefreshingInferenceClient,
)


class DummyClient:
    def __init__(self):
        self.refreshed = 0
        self.close_called = 0

    def initial_inference(self, obs, training=False):
        return ("init", obs)

    def recurrent_inference(self, hidden, act, training=False):
        return ("rec", hidden, act)

    def refresh_params(self):
        self.refreshed += 1

    def close(self):
        self.close_called += 1


def test_parameter_refreshing_client_passthrough_and_refresh():
    base = DummyClient()
    adapter = ParameterRefreshingInferenceClient(base, publisher_client=None)

    out = adapter.initial_inference(np.array([1.0]))
    assert out[0] == "init"

    out2 = adapter.recurrent_inference(np.array([2.0]), np.array([3.0]))
    assert out2[0] == "rec"

    adapter.refresh_params()
    assert base.refreshed == 1

    adapter.close()
    assert base.close_called == 1
