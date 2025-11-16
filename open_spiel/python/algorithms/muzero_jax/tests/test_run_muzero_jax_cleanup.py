from open_spiel.python.algorithms.muzero_jax import run_muzero_jax as rm


def test_cleanup_with_empty_clients(monkeypatch):
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = type("C", (), {"wandb": type("W", (), {"enabled": False})})
    orch._inference_clients = []
    orch._replay_clients = []
    orch.learner = type("L", (), {"save_checkpoint": lambda self, force_save=True: None, "wait_for_pending_checkpoints": lambda self: None})()
    orch.cleanup()  # should not raise
