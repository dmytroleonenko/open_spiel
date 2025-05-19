import pytest
try:
    import pyspiel
except ImportError:
    pytest.skip("pyspiel module not available; skipping tests", allow_module_level=True)
import numpy as np
import pyspiel
from open_spiel.python.algorithms.alpha_zero_jax.actor_evaluator_logic import _init_bot

class CountingEvaluator:
    """Evaluator that records whether evaluate() is ever called on a chance node."""
    def __init__(self):
        self.calls = []
    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        actions = state.legal_actions(state.current_player())
        return [(a, 1.0 / len(actions)) for a in actions]
    def evaluate(self, state):
        # Record if this state is a chance node
        self.calls.append(state.is_chance_node())
        # Return dummy zero-sum value
        return np.array([0.0, 0.0], dtype=np.float32)

class DummyConfig:
    async_mode = True
    uct_c = 1.0
    max_simulations = 5
    policy_alpha = 0.3
    policy_epsilon = 0.25
    log_level = 4

@pytest.mark.parametrize("game_name", ["long_narde"])
def test_async_mcts_no_evaluate_on_chance_root(game_name):
    """Test that AsyncMCTSBot does not call evaluate() on a chance root node."""
    game = pyspiel.load_game(game_name)
    evaluator = CountingEvaluator()
    config = DummyConfig()
    bot = _init_bot(config, game, evaluator, evaluation=False, player_id_for_bot=0)
    policy, action = bot.step_with_policy(game.new_initial_state())
    assert evaluator.calls == [], f"AsyncMCTS evaluate called on chance node for game {game_name}" 
