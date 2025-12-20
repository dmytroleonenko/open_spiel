
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    generate_top_new_masks,
    apply_mixed_value_targets,
)

def test_generate_top_new_masks():
    """Test mask generation for mixed value targets."""
    # Setup
    batch_size = 5
    sample_indices = jnp.array([100, 200, 800, 900, 950])
    collected_transitions = 1000
    mixed_value_threshold = 200  # Recent if index > 1000 - 200 = 800

    # Threshold = 800.
    # Indices: 100, 200, 800 <= 800 (Old)
    # Indices: 900, 950 > 800 (Recent)

    masks = generate_top_new_masks(sample_indices, collected_transitions, mixed_value_threshold)

    expected_masks = jnp.array([False, False, False, True, True])

    assert jnp.array_equal(masks, expected_masks)

def test_apply_mixed_value_targets():
    """Test mixed value target application."""
    # Setup
    batch_size = 4
    unroll_steps = 2

    # [B, K+1]
    search_values = jnp.full((batch_size, unroll_steps + 1), 1.0)
    sarsa_values = jnp.full((batch_size, unroll_steps + 1), 2.0)

    # Masks: [False, True, False, True] -> [Old, Recent, Old, Recent]
    top_new_masks = jnp.array([False, True, False, True])

    mixed_values = apply_mixed_value_targets(
        search_values, sarsa_values, top_new_masks, unroll_steps
    )

    # Check shape
    assert mixed_values.shape == search_values.shape

    # Old samples (mask=0) should use search values (1.0)
    assert jnp.allclose(mixed_values[0], 1.0)
    assert jnp.allclose(mixed_values[2], 1.0)

    # Recent samples (mask=1) should use sarsa values (2.0)
    assert jnp.allclose(mixed_values[1], 2.0)
    assert jnp.allclose(mixed_values[3], 2.0)

def test_apply_mixed_value_targets_categorical():
    """Test mixed value target application for categorical values."""
    # Setup
    batch_size = 2
    unroll_steps = 1
    support_size = 5

    # [B, K+1, S]
    search_values = jnp.full((batch_size, unroll_steps + 1, support_size), 1.0)
    sarsa_values = jnp.full((batch_size, unroll_steps + 1, support_size), 2.0)

    # Masks: [False, True] -> [Old, Recent]
    top_new_masks = jnp.array([False, True])

    mixed_values = apply_mixed_value_targets(
        search_values, sarsa_values, top_new_masks, unroll_steps
    )

    # Check shape
    assert mixed_values.shape == search_values.shape

    # Old sample (mask=0) -> search
    assert jnp.allclose(mixed_values[0], 1.0)

    # Recent sample (mask=1) -> sarsa
    assert jnp.allclose(mixed_values[1], 2.0)

if __name__ == "__main__":
    pytest.main([__file__])
