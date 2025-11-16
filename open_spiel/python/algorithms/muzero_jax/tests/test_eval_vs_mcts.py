import contextlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.scripts import eval_vs_mcts as eval_mod


def test_parse_args_round_trip(tmp_path):
    args = eval_mod.parse_args(
        [
            "--save-path",
            str(tmp_path),
            "--games",
            "7",
            "--mcts-simulations",
            "123",
            "--override",
            "training.batch_size=4",
        ]
    )
    assert args.save_path == tmp_path
    assert args.games == 7
    assert args.mcts_simulations == 123
    assert args.override == ["training.batch_size=4"]


def test_evaluate_with_stubbed_dependencies(monkeypatch, tmp_path):
    payoffs = [1, 0, -1, 1]

    def fake_run_game(actor, baseline_bot, mu_player, rng_key, rng_np):
        payoff = payoffs.pop(0)
        return payoff, rng_key

    monkeypatch.setattr(eval_mod, "run_game", fake_run_game)

    monkeypatch.setattr(
        eval_mod,
        "initialize_config_dir",
        lambda **kwargs: contextlib.nullcontext(),
    )

    def fake_compose(config_name, overrides):
        return SimpleNamespace(
            game=SimpleNamespace(name="tic_tac_toe"),
            output=SimpleNamespace(save_path=str(tmp_path)),
            resource_management=SimpleNamespace(),
            evaluation=SimpleNamespace(enabled=False, interval=1),
            actors=SimpleNamespace(num_actors=1),
            exp_config=SimpleNamespace(seed=0),
        )

    monkeypatch.setattr(eval_mod, "compose", fake_compose)

    class StubLearner:
        def __init__(self):
            self.loaded = None

        def load_checkpoint(self, path):
            self.loaded = path
            return True

        def wait_for_pending_checkpoints(self):
            pass

    class StubOrchestrator:
        def __init__(self, cfg):
            self.cfg = cfg
            self.network = object()
            self.game_wrapper = SimpleNamespace(
                num_distinct_actions=lambda: 3,
            )
            self.muzero_config = SimpleNamespace(
                num_simulations=1,
                td_steps=1,
                discount_factor=0.99,
            )
            self.learner = StubLearner()
            self.checkpoint_dir = tmp_path / "ckpts"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(eval_mod, "MuZeroOrchestrator", StubOrchestrator)

    class StubActor:
        def __init__(self, **kwargs):
            self.game_wrapper = SimpleNamespace(_game="stub_game")

        def maybe_load_latest_parameters(self, checkpoint_dir):
            self.loaded_dir = checkpoint_dir

    monkeypatch.setattr(eval_mod, "Actor", StubActor)

    class StubGameWrapper:
        def __init__(self, name):
            self._game = f"game::{name}"

        def reset(self):
            pass

    monkeypatch.setattr(eval_mod, "GameWrapper", StubGameWrapper)

    class StubBaseline:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, state):
            return 0

    monkeypatch.setattr(eval_mod.pyspiel, "MCTSBot", StubBaseline)
    monkeypatch.setattr(
        eval_mod.pyspiel,
        "RandomRolloutEvaluator",
        lambda *args, **kwargs: None,
    )

    result = eval_mod.evaluate(
        cfg_dir=tmp_path,
        cfg_name="config",
        overrides=["training.training_steps=5"],
        save_path=tmp_path,
        checkpoint_step=5,
        games=4,
        baseline_simulations=10,
        seed=0,
        mu_player_mode="auto",
        verbose=True,
    )

    assert result.games == 4
    assert result.muzero_wins == 2
    assert result.baseline_wins == 1
    assert result.draws == 1


class _StubGameWrapper:
    def __init__(self):
        self._state = SimpleNamespace()

    def current_observation(self):
        return [0.0, 1.0]

    def legal_actions(self):
        return [0, 1, 2]

    def reset(self):
        self._state = SimpleNamespace(moves=0, returns=lambda: [1, -1])

    def step(self, action):
        state = getattr(self, "_state")
        if not hasattr(state, "moves"):
            state.moves = 0
        state.moves += 1
        state.is_terminal = lambda: state.moves >= 2
        state.is_chance_node = lambda: False
        state.current_player = lambda: state.moves % 2
        state.returns = lambda: [1, -1]
        self._state = state


def test_select_muzero_action_uses_stub_actor():
    class StubNetwork:
        def initial_inference(self, obs, training):
            logits = jnp.ones((1, 3))
            return (
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                logits,
                None,
                None,
            )

    class StubMCTS:
        def run(self, **kwargs):
            return SimpleNamespace()

    class StubActor:
        def __init__(self):
            self.game_wrapper = SimpleNamespace(
                current_observation=lambda: [0.0],
                legal_actions=lambda: [0, 1],
            )
            self.network = StubNetwork()
            self.config = SimpleNamespace(num_actions=2)
            self.mcts = StubMCTS()
            self._is_stochastic_mcts = False
            self.temperature = 1.0

        def _create_recurrent_fn(self):
            return None

        def _select_action(self, policy_output, temperature_step):
            self._selected = temperature_step
            return jnp.array(1), None

    actor = StubActor()
    rng = jax.random.PRNGKey(0)
    action, new_key = eval_mod._select_muzero_action(actor, rng, 0)
    assert action == 1
    assert isinstance(new_key, jax.Array)
    assert actor.temperature == 0.0


def test_run_game_with_stub_actor(monkeypatch):
    class StubActor:
        def __init__(self):
            state = SimpleNamespace(moves=0)

            def returns():
                return [1, -1]

            def is_terminal():
                return state.moves >= 2

            def current_player():
                return state.moves % 2

            def apply_action(action):
                state.moves += 1

            state.is_terminal = is_terminal
            state.is_chance_node = lambda: False
            state.current_player = current_player
            state.apply_action = apply_action
            state.returns = returns

            def step(action):
                state.apply_action(action)

            self.game_wrapper = SimpleNamespace(
                reset=lambda: None,
                _state=state,
                step=step,
            )

    def fake_select(actor, rng_key, idx):
        return 0, rng_key

    monkeypatch.setattr(eval_mod, "_select_muzero_action", fake_select)

    class StubBaseline:
        def step(self, state):
            state.apply_action(1)
            return 1

    actor = StubActor()
    baseline = StubBaseline()

    rng = jax.random.PRNGKey(0)
    rng_np = np.random.default_rng(0)

    payoff, _ = eval_mod.run_game(
        actor=actor,
        baseline_bot=baseline,
        mu_player=0,
        rng_key=rng,
        rng_np=rng_np,
    )
    assert payoff == 1
