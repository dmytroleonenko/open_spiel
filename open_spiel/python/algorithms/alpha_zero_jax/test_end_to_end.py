import os
import tempfile
import shutil
import pytest
try:
    import pyspiel
except ImportError:
    pytest.skip("pyspiel module not available; skipping tests", allow_module_level=True)
import pyspiel
from open_spiel.python.algorithms.alpha_zero_jax.alpha_zero_jax import ConfigJAX, alpha_zero_jax

@pytest.mark.parametrize("game_name", ["tic_tac_toe"])
def test_end_to_end_smoke(tmp_path, game_name):
    # End-to-end smoke test: run one training step on a simple game
    out_dir = tmp_path / "az_output"
    config = ConfigJAX(
        game=game_name,
        path=str(out_dir),
        learning_rate=1e-3,
        weight_decay=0.0,
        train_batch_size=1,
        replay_buffer_size=1,
        replay_buffer_reuse=1,
        max_steps=1,
        checkpoint_freq=0,
        actors=1,
        evaluators=1,
        evaluation_window=1,
        eval_levels=1,
        uct_c=1.0,
        max_simulations=1,
        policy_alpha=0.3,
        policy_epsilon=0.25,
        temperature=1.0,
        temperature_drop=0,
        nn_model="mlp",
        nn_width=1,
        nn_depth=1,
        observation_shape=(),
        output_size=0,
        quiet=True,
        master_seed=0,
        resnet_depth_config=None,
        resnet_stem_callable_name=None,
        resnet_stem_kwargs={},
        resnet_block_callable_name=None,
        resnet_block_kwargs={},
        evaluator_cache_size=1,
        log_level=0,
        remote_evaluator_timeout_ms=1000,
        inference_batch_timeout_ms=1.0,
        inference_batch_size=1,
        async_mode=False,
        async_batch_size=1,
        async_virtual_loss=1,
        async_timeout=1.0,
    )
    # Should run without exception and produce output directory
    alpha_zero_jax(config)
    assert os.path.isdir(str(out_dir)), "Output directory was not created"
    # Check that at least main log file exists
    log_files = [f for f in os.listdir(str(out_dir)) if f.startswith("log-main_alpha_zero_jax")]
    assert log_files, f"Expected main log files in {out_dir}, found none" 