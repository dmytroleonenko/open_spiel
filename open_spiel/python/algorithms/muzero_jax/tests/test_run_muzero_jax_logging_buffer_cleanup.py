import types

import pytest

import open_spiel.python.algorithms.muzero_jax.run_muzero_jax as rm


def _orch(min_buffer=0, log_interval=2):
    cfg = types.SimpleNamespace(
        output=types.SimpleNamespace(log_interval=log_interval, save_path="/tmp/nojit"),
        wandb=types.SimpleNamespace(enabled=False),
    )
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = cfg
    orch.training_step = 2
    orch.total_episodes = 3
    import threading

    orch._buffer_lock = threading.Lock()
    orch.replay_buffer = []
    return orch


def test_maybe_log_training_metrics_interval_respected(monkeypatch):
    orch = _orch(log_interval=5)
    called = {}
    monkeypatch.setattr(rm, "get_cli_verbosity", lambda: 1)

    def fake_log(metrics):
        called["metrics"] = metrics

    monkeypatch.setattr(orch, "log_training_metrics", fake_log)
    orch._maybe_log_training_metrics({"loss": 1.0})
    assert called  # interval divides training_step


def test_should_log_episode_metrics_interval_zero_returns_false(monkeypatch):
    orch = _orch(log_interval=0)
    monkeypatch.setattr(rm, "get_cli_verbosity", lambda: 0)
    assert orch._should_log_episode_metrics() is False


def test_register_clients_ignore_none():
    orch = _orch()
    orch._inference_clients = []
    orch._replay_clients = []
    orch._register_inference_client(None)
    orch._register_replay_client(None)
    assert orch._inference_clients == []
    assert orch._replay_clients == []


def test_cleanup_empty_clients(monkeypatch):
    orch = _orch()
    orch._inference_clients = []
    orch._replay_clients = []
    orch.learner = types.SimpleNamespace(save_checkpoint=lambda force_save=True: None, wait_for_pending_checkpoints=lambda: None)
    orch.config = types.SimpleNamespace(wandb=types.SimpleNamespace(enabled=False))
    monkeypatch.setattr(rm.logger, "info", lambda *a, **k: None)
    orch.cleanup()


def test_maybe_publish_params_interval_skip():
    orch = _orch()
    orch.training_step = 1
    orch._parameter_client = type("C", (), {"publish": lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not publish"))})()
    orch.config.publisher = types.SimpleNamespace(publish_interval=2)
    orch._maybe_publish_params = rm.MuZeroOrchestrator._maybe_publish_params.__get__(orch, rm.MuZeroOrchestrator)
    # Should return early without calling publish
    orch._maybe_publish_params(force=False)


def test_cleanup_closes_replay_clients(monkeypatch):
    orch = _orch()
    calls = {}
    orch._replay_clients = [type("C", (), {"close": lambda self=None: calls.setdefault("closed", 0) or calls.__setitem__("closed", calls.get("closed",0)+1)})()]
    orch._inference_clients = []
    orch.learner = types.SimpleNamespace(save_checkpoint=lambda force_save=True: None, wait_for_pending_checkpoints=lambda: None)
    orch.config = types.SimpleNamespace(wandb=types.SimpleNamespace(enabled=False))
    monkeypatch.setattr(rm.logger, "info", lambda *a, **k: None)
    orch.cleanup()
    assert calls["closed"] == 1


def test_build_replay_buffer_remote_missing_endpoint():
    cfg = types.SimpleNamespace(
        replay_buffer=types.SimpleNamespace(remote_enabled=True, rpc_endpoint="", priority_alpha=0.0, capacity=10),
        game=types.SimpleNamespace(name="tic_tac_toe"),
    )
    game_obj = type("G", (), {"max_game_length": lambda self=None: 1})()
    with pytest.raises(ValueError):
        rm.build_replay_buffer(cfg, observation_shape=(3,), num_actions=3, game_obj=game_obj)


def test_build_publisher_client_missing_endpoint():
    cfg = types.SimpleNamespace(publisher=types.SimpleNamespace(remote_enabled=True, rpc_endpoint=""))
    with pytest.raises(ValueError):
        rm.build_parameter_publisher_client(cfg)
