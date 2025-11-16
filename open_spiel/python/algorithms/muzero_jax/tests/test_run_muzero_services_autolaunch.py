import types

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator


def _minimal_cfg(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("open_spiel/python/algorithms/muzero_jax/configs/config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.output.save_path = str(tmp_path) + "/"
    cfg.game.name = "tic_tac_toe"
    cfg.training.batch_size = 1
    cfg.training.training_steps = 1
    cfg.training.start_transitions = 1
    cfg.actors.num_actors = 0
    cfg.bootstrap.enabled = False
    cfg.evaluation.enabled = False
    cfg.resource_management.sequential_training = True
    cfg.resource_management.concurrent = False
    cfg.replay_buffer.remote_enabled = True
    cfg.replay_buffer.rpc_endpoint = ""
    cfg.publisher.remote_enabled = True
    cfg.publisher.rpc_endpoint = ""
    return cfg


def test_autolaunch_replay_and_publisher(tmp_path):
    cfg = _minimal_cfg(tmp_path)
    orch = MuZeroOrchestrator(cfg)
    try:
        # Endpoints should be auto-populated
        assert cfg.replay_buffer.rpc_endpoint
        assert cfg.publisher.rpc_endpoint
    finally:
        orch.cleanup()
