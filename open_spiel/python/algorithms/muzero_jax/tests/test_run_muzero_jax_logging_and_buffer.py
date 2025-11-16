from types import SimpleNamespace

from omegaconf import OmegaConf

from open_spiel.python.algorithms.muzero_jax import run_muzero_jax as rm


def make_orchestrator_stub(log_interval: int, cli_verbosity: int = 0):
    cfg = OmegaConf.create(
        {
            "output": {"log_interval": log_interval},
            "training": {"batch_size": 2, "start_transitions": 3},
            "replay_buffer": {"min_size_to_sample": 5},
            "game": {"name": "tic_tac_toe"},
            "resource_management": {"training_phase_steps": 1, "selfplay_phase_episodes": 1},
            "actors": {"num_actors": 1},
            "bootstrap": {"enabled": False, "min_episodes": 0},
            "evaluation": {"enabled": False},
            "wandb": {"enabled": False},
            "exp_config": {"seed": 0},
        }
    )
    orch = rm.MuZeroOrchestrator.__new__(rm.MuZeroOrchestrator)
    orch.config = cfg
    orch.muzero_config = SimpleNamespace(start_transitions=3, batch_size=2)
    orch.total_episodes = 4
    orch.training_step = 0
    orch._buffer_lock = rm.threading.Lock()
    orch._inference_clients = []
    orch._replay_clients = []
    rm.CLI_VERBOSITY = cli_verbosity
    return orch


def test_min_buffer_before_training_uses_max():
    orch = make_orchestrator_stub(log_interval=0)
    assert orch._min_buffer_before_training() == 5  # max of 5,2,3


def test_should_log_episode_metrics_interval_zero():
    orch = make_orchestrator_stub(log_interval=0, cli_verbosity=0)
    assert orch._should_log_episode_metrics() is False


def test_should_log_episode_metrics_forced_by_cli():
    orch = make_orchestrator_stub(log_interval=0, cli_verbosity=2)
    assert orch._should_log_episode_metrics() is True
