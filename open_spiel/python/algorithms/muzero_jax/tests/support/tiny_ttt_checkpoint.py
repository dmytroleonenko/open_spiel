"""Utility to create a 1-step Tic-Tac-Toe MuZero checkpoint for fast tests.

We keep runtime tiny (<2s) by disabling almost everything (no wandb, no eval,
sequential mode, start_transitions=1). Tests can import `tiny_ttt_checkpoint`
to obtain a path to the generated checkpoint directory and reuse it across
multiple cases to avoid network stubbing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hydra import initialize_config_dir, compose

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator


_CFG_DIR = Path(__file__).resolve().parents[2] / "configs"


def tiny_ttt_checkpoint(tmpdir: str | None = None) -> Path:
    """Train for a single step and return checkpoint directory path.

    Args:
        tmpdir: optional directory to place outputs; if omitted a temp dir is used.

    Returns:
        Path to directory containing checkpoints/ (with step 1 saved).
    """

    output_dir = Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp())

    overrides = [
        "game.name=tic_tac_toe",
        "resource_management.device=cpu",
        "resource_management.concurrent=false",
        "resource_management.sequential_training=true",
        "training.training_steps=1",
        "training.batch_size=1",
        "+training.start_transitions=1",
        "actors.num_actors=1",
        "evaluation.enabled=false",
        "wandb.enabled=false",
        f"output.save_path={output_dir}",
    ]

    with initialize_config_dir(version_base="1.1", config_dir=str(_CFG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)

    orch = MuZeroOrchestrator(cfg)
    # One sequential training step
    orch.run_training_phase()
    orch._maybe_publish_params(force=True)
    checkpoint_path = orch.learner.save_checkpoint(force_save=True)
    orch.cleanup()

    return Path(checkpoint_path).parent


__all__ = ["tiny_ttt_checkpoint"]
