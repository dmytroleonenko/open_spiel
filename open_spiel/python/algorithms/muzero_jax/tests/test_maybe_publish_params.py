from types import SimpleNamespace

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator


def test_maybe_publish_params_calls_client_when_interval_matches(monkeypatch):
    dummy = SimpleNamespace()
    published = {}

    class DummyNet:
        def get_variables(self):
            return {"a": 1}

    orch = SimpleNamespace()
    orch.network = DummyNet()
    orch.training_step = 4
    orch.config = SimpleNamespace(publisher=SimpleNamespace(publish_interval=2))
    orch._parameter_client = SimpleNamespace(publish=lambda params, step: published.update({"params": params, "step": step}))

    # Bind the helper method from MuZeroOrchestrator onto dummy instance
    MuZeroOrchestrator._maybe_publish_params(orch, force=False)

    assert published["params"] == {"a": 1}
    assert published["step"] == 4


def test_maybe_publish_params_no_get_variables_skips_publish():
    published = {}

    class DummyNet:
        pass

    orch = SimpleNamespace()
    orch.network = DummyNet()
    orch.training_step = 1
    orch.config = SimpleNamespace(publisher=SimpleNamespace(publish_interval=1))
    orch._parameter_client = SimpleNamespace(publish=lambda params, step: published.update({"params": params, "step": step}))

    MuZeroOrchestrator._maybe_publish_params(orch, force=True)
    assert published == {}
