import importlib


def test_eval_vs_mcts_import_and_main_guard():
    # Ensure module imports without executing main
    import open_spiel.python.algorithms.muzero_jax.scripts.eval_vs_mcts as mod
    importlib.reload(mod)
    assert hasattr(mod, "evaluate_against_mcts")
