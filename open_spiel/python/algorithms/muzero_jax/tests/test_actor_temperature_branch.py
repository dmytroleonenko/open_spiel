import numpy as np

from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor


class DummyMCTSResult:
    def __init__(self, logits):
        self.action_weights = logits


def test_select_action_temperature_floor():
    # Build a minimal actor with a stubbed select_action path
    actor = Actor.__new__(Actor)
    actor.temperature = 1e-6
    actor.temperature_threshold = 100
    actor.min_legal_actions = 0

    actor.game_wrapper = type("GW", (), {"legal_actions": lambda self=None: [0, 1]})()
    actor._is_stochastic_mcts = False
    logits = np.array([[0.1, 0.9]], dtype=np.float32)
    result = DummyMCTSResult(logits)
    action, _ = actor._select_action(result, 0)
    assert action == 1  # argmax


def test_refresh_params_calls_client():
    actor = Actor.__new__(Actor)
    actor._parameter_client = object()
    called = {}
    actor.inference_client = type("IC", (), {"refresh_params": lambda self=None: called.setdefault("ok", True)})()
    actor.refresh_params()
    assert called.get("ok") is True
