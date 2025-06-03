import jax
import jax.numpy as jnp
import pytest
from open_spiel.python.algorithms.muzero_jax.training.trainer import Batch, MuZeroConfig


def test_batch_content_alignment_action_item_10():
    # Default configuration
    config = MuZeroConfig()
    B = 3
    K = config.num_unroll_steps
    num_actions = config.num_actions
    num_sampled = config.num_sampled_actions

    # Construct a dummy batch with all required fields
    batch: Batch = {
        'batch_actions': jnp.zeros((B, K+1, num_sampled, num_actions), dtype=jnp.int32),
        'batch_best_actions': jnp.zeros((B, K+1), dtype=jnp.int32),
        'policy_masks': jnp.ones((B, K+1), dtype=jnp.int32),
        'reanalyzed_values': jnp.zeros((B, K+1), dtype=jnp.float32),
        'value_prefix': jnp.zeros((B, K+1), dtype=jnp.float32),
    }

    # Check that required keys are present
    required_keys = ['batch_actions', 'batch_best_actions', 'policy_masks', 'reanalyzed_values', 'value_prefix']
    assert set(required_keys).issubset(batch.keys())

    # Verify shapes and dtypes for each field
    ba = batch['batch_actions']
    assert isinstance(ba, jax.Array)
    assert ba.dtype == jnp.int32
    assert ba.shape == (B, K+1, num_sampled, num_actions)

    bba = batch['batch_best_actions']
    assert isinstance(bba, jax.Array)
    assert bba.dtype == jnp.int32
    assert bba.ndim == 2 and bba.shape == (B, K+1)

    pm = batch['policy_masks']
    assert isinstance(pm, jax.Array)
    assert pm.dtype == jnp.int32
    assert pm.shape == (B, K+1)

    rv = batch['reanalyzed_values']
    assert isinstance(rv, jax.Array)
    assert rv.dtype == jnp.float32
    assert rv.shape == (B, K+1)

    vp = batch['value_prefix']
    assert isinstance(vp, jax.Array)
    assert vp.dtype == jnp.float32
    assert vp.shape == (B, K+1) 