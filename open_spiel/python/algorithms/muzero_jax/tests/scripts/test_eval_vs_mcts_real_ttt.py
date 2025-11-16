import pathlib

from open_spiel.python.algorithms.muzero_jax.scripts import eval_vs_mcts as ev
from open_spiel.python.algorithms.muzero_jax.tests.support.tiny_ttt_checkpoint import tiny_ttt_checkpoint


def test_eval_vs_mcts_real_ttt(tmp_path):
    ckpt_dir = tiny_ttt_checkpoint(tmpdir=tmp_path)
    argv = [
        "--save-path",
        str(ckpt_dir),
        "--games",
        "1",
        "--mcts-simulations",
        "1",
        "--override",
        "game.name=tic_tac_toe",
    ]
    ev.evaluate_against_mcts(argv)

