import argparse
import types
from pathlib import Path
import contextlib

import numpy as np
import jax

import open_spiel.python.algorithms.muzero_jax.scripts.eval_vs_mcts as mod


class DummyActor:
    def __init__(self):
        dummy_game = types.SimpleNamespace()
        dummy_state = types.SimpleNamespace(
            current_player=lambda: 0,
            is_terminal=lambda: True,
            returns=lambda: [1, -1],
            chance_outcomes=lambda: [],
        )
        self.game_wrapper = types.SimpleNamespace(
            reset=lambda: None,
            _state=dummy_state,
            _game=dummy_game,
            legal_actions=lambda: [0],
            step=lambda a: None,
            num_distinct_actions=lambda: 1,
        )
        self.mcts = types.SimpleNamespace(run=lambda **kwargs: types.SimpleNamespace(prior_logits=np.zeros((1, 1)), value=np.zeros((1, 1)), embedding=np.zeros((1, 1))))
        self.network = types.SimpleNamespace(initial_inference=lambda obs, training=False: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), None, None))
        self._is_stochastic_mcts = False
        self.config = types.SimpleNamespace(num_actions=1)

    def _create_recurrent_fn(self):
        return lambda *args, **kwargs: None

    def _select_action(self, root, temperature):
        return 0, None

    def maybe_load_latest_parameters(self, *a, **k):
        return True


class DummyBot:
    def __init__(self):
        self.called = 0

    def step(self, state):
        self.called += 1
        return 0


def test_parse_args_overrides():
    ns = mod.parse_args(["--save-path", "/tmp/foo", "--checkpoint-step", "5", "--games", "1", "--mcts-simulations", "2", "--override", "training.batch_size=8"])
    assert ns.save_path
    assert ns.checkpoint_step == 5
    assert "training.batch_size=8" in ns.override


def test_evaluate_against_mcts_short(monkeypatch):
    args = argparse.Namespace(
        save_path="/tmp/foo",
        checkpoint_step=None,
        games=1,
        mcts_simulations=1,
        seed=0,
        temperature=0.0,
        muzero_player="auto",
        verbose=False,
        override=[],
    )

    dummy_actor = DummyActor()
    dummy_bot = DummyBot()

    # Stub config loader to return minimal cfg
    dummy_cfg = types.SimpleNamespace(game=types.SimpleNamespace(name="tic_tac_toe"))
    monkeypatch.setattr(mod, "initialize_config_dir", lambda **k: contextlib.nullcontext())
    monkeypatch.setattr(mod, "compose", lambda config_name, overrides, return_hydra_config=False: dummy_cfg)

    # Dummy orchestrator with required attrs
    class DummyOrch:
        def __init__(self, cfg):
            self.muzero_config = types.SimpleNamespace(
                num_simulations=1, td_steps=1, discount_factor=0.99, num_actions=1
            )
            self.network = dummy_actor.network
            self.game_wrapper = types.SimpleNamespace(num_distinct_actions=lambda: 1)
            self.checkpoint_dir = "/tmp/fake"
            self.learner = types.SimpleNamespace(load_checkpoint=lambda self, path=None: True)

    monkeypatch.setattr(mod, "MuZeroOrchestrator", DummyOrch)
    monkeypatch.setattr(mod, "Actor", lambda **k: dummy_actor)
    monkeypatch.setattr(mod.pyspiel, "MCTSBot", lambda *a, **k: dummy_bot)
    monkeypatch.setattr(jax.random, "split", lambda key: (key, key))

    # No-op game run, return draw
    monkeypatch.setattr(mod, "run_game", lambda *a, **k: (0, jax.random.PRNGKey(0)))
    results = mod.evaluate(
        cfg_dir=Path("/tmp"),
        cfg_name="config",
        overrides=[],
        save_path=Path("/tmp"),
        checkpoint_step=None,
        games=1,
        baseline_simulations=1,
        seed=0,
        mu_player_mode="auto",
        verbose=False,
    )
    assert results.games == 1
    assert results.draws == 1
