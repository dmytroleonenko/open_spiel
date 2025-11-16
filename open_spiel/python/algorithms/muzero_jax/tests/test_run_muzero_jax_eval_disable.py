import types

import open_spiel.python.algorithms.muzero_jax.run_muzero_jax as rm


def test_maybe_run_periodic_evaluation_disabled():
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.training_step = 10
    orch._last_eval_step = 0
    orch.config = types.SimpleNamespace(evaluation=types.SimpleNamespace(enabled=False))
    orch._evaluation_enabled = rm.MuZeroOrchestrator._evaluation_enabled.__get__(orch, rm.MuZeroOrchestrator)
    orch._maybe_run_periodic_evaluation()
    assert orch._last_eval_step == 0
