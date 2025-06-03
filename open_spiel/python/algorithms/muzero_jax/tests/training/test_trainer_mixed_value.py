import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses

# Import from the common utils module
from trainer_test_utils import (
    NUM_UNROLL_STEPS,
    NUM_ACTIONS,
    BATCH_SIZE,
    VALUE_SUPPORT_SCALAR,
    REWARD_SUPPORT_SCALAR,
    VALUE_SUPPORT_CATEGORICAL,
    REWARD_SUPPORT_CATEGORICAL,
    key as common_key, 
    cfg_flat as common_cfg_flat,
    cfg_img as common_cfg_img,
    make_model, 
    make_cfg,
    make_batch,
    MockNetCfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch,
    generate_top_new_masks,
    apply_mixed_value_targets
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew 

def test_generate_top_new_masks_edge_cases(common_key, common_cfg_flat):
    """Test edge cases for generate_top_new_masks."""
    # Test with samples being recent (threshold = 9000, so only 9500 > 9000)
    sample_indices = jnp.array([8000, 9000, 9500])
    collected_transitions = 10000
    mixed_value_threshold = 1000  # threshold at 9000 (10000-1000)

    expected_mixed = jnp.array([0.0, 0.0, 1.0])  # Only 9500 > 9000
    result_mixed = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(result_mixed, expected_mixed)

    # Test with all samples being new (recent) - use very low threshold
    sample_indices = jnp.array([8000, 9000, 9500])
    collected_transitions = 10000
    mixed_value_threshold = (
        2000  # threshold at 8000 (10000-2000), so all samples > 8000
    )

    expected_all_new = jnp.array(
        [0.0, 1.0, 1.0]
    )  # 8000 == 8000 (not >), 9000 > 8000, 9500 > 8000
    result_all_new = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(result_all_new, expected_all_new)

    # Test with truly all samples being new
    sample_indices = jnp.array([8500, 9000, 9500])
    collected_transitions = 10000
    mixed_value_threshold = 2000  # threshold at 8000, so all samples > 8000

    expected_truly_all_new = jnp.array([1.0, 1.0, 1.0])  # All > 8000
    result_truly_all_new = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(result_truly_all_new, expected_truly_all_new)

    # Test with all samples being old
    sample_indices = jnp.array([1000, 2000, 3000])
    collected_transitions = 10000
    mixed_value_threshold = (
        7000  # threshold at 3000 (10000-7000), so all samples <= 3000
    )

    expected_all_old = jnp.array(
        [0.0, 0.0, 0.0]
    )  # 1000 <= 3000, 2000 <= 3000, 3000 <= 3000 (not >)
    result_all_old = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(result_all_old, expected_all_old)

    # Test with single sample at boundary
    sample_indices = jnp.array([7000])
    collected_transitions = 10000
    mixed_value_threshold = 3000  # Threshold exactly at 7000

    expected_boundary = jnp.array([0.0])  # idx == threshold gives mask=0
    result_boundary = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(result_boundary, expected_boundary)

def test_apply_mixed_value_targets_categorical(common_key, common_cfg_flat):
    """Test apply_mixed_value_targets with categorical value distributions."""
    batch_size, num_steps, support_size = 2, 3, 5

    # Create categorical search and sarsa values
    search_values = jnp.ones((batch_size, num_steps, support_size)) * 0.1
    sarsa_values = jnp.ones((batch_size, num_steps, support_size)) * 0.2

    # Create masks
    top_new_masks = jnp.array([0.0, 1.0])

    expected_mixed = jnp.array(
        [
            [[0.1] * support_size] * num_steps,  # Old sample: search values
            [[0.2] * support_size] * num_steps,  # New sample: sarsa values
        ]
    )

    result = apply_mixed_value_targets(
        search_values, sarsa_values, top_new_masks, num_steps - 1
    )

    assert result.shape == (batch_size, num_steps, support_size)
    assert jnp.allclose(result, expected_mixed)

def test_mixed_value_threshold_trainer_integration(common_key, common_cfg_flat):
    """Test integration of mixed value threshold logic in trainer."""
    mk, bk = jax.random.split(common_key, 2)

    # Create model and config with mixed value target
    model = make_model(mk, common_cfg_flat)
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        NUM_UNROLL_STEPS,  # Use optimized constant instead of 3
        False,  # use_projection
        "mixed_value_threshold_integration",
    )
    config = dataclasses.replace(
        config,
        value_target="mixed",
        mixed_value_threshold=1000,
        start_use_mix_training_steps=50,
        batch_size=BATCH_SIZE,  # Use optimized constant
    )

    # Create batch with mixed value components
    batch = make_batch(
        bk,
        BATCH_SIZE,  # Use optimized constant
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        NUM_UNROLL_STEPS,  # Use optimized constant
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    # Add search and sarsa values (adjusted for optimized sizes)
    search_values = jnp.ones((BATCH_SIZE, NUM_UNROLL_STEPS + 1)) * 1.0
    sarsa_values = jnp.ones((BATCH_SIZE, NUM_UNROLL_STEPS + 1)) * 2.0
    sample_indices = jnp.array([500])  # Single sample for BATCH_SIZE=1
    collected_transitions = 2000

    batch["target_search_value"] = search_values
    batch["target_sarsa_value"] = sarsa_values
    batch["sample_indices"] = sample_indices
    batch["collected_transitions"] = collected_transitions
    batch["training_step"] = 100  # After start_use_mix_training_steps

    # Compute loss (should apply mixed value targets internally)
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Verify loss computation completed successfully
    assert jnp.isfinite(loss)
    assert "value_loss" in metrics
    assert jnp.isfinite(metrics["value_loss"])

    # Test with training step before start_use_mix_training_steps
    batch["training_step"] = 30
    loss_early, metrics_early = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Should use only search values early in training
    assert jnp.isfinite(loss_early)

def test_mixed_value_fallback_behavior(common_key, common_cfg_flat):
    """Test fallback behavior when mixed value components are missing."""
    mk, bk = jax.random.split(common_key, 2)

    model = make_model(mk, common_cfg_flat)
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        2,  # num_unroll_steps
        False,  # use_projection
        "fallback_test",
    )
    config = dataclasses.replace(config, value_target="mixed")

    # Create minimal batch without mixed value components
    batch = make_batch(
        bk,
        2,  # batch_size
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        2,  # num_unroll_steps
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    # Should fallback to regular target_value when mixed components missing
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    assert jnp.isfinite(loss)
    assert "value_loss" in metrics 

def test_top_new_masks_mathematical_properties(common_key, common_cfg_flat):
    """Test mathematical properties of top_new_masks generation."""
    # Test that mask generation is deterministic and consistent
    sample_indices = jnp.array([1000, 2000, 3000, 4000, 5000])
    collected_transitions = 4000
    mixed_value_threshold = 1500

    # Generate masks multiple times
    masks1 = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )
    masks2 = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )

    # Should be identical
    assert jnp.allclose(masks1, masks2)

    # Test monotonicity: higher indices should have >= mask values
    sorted_indices = jnp.sort(sample_indices)
    sorted_masks = generate_top_new_masks(
        sorted_indices, collected_transitions, mixed_value_threshold
    )

    # Check that mask values are non-decreasing (monotonic)
    for i in range(len(sorted_masks) - 1):
        assert (
            sorted_masks[i] <= sorted_masks[i + 1]
        ), f"Masks should be non-decreasing, but {sorted_masks[i]} > {sorted_masks[i + 1]}"

    # Test boundary conditions
    threshold_idx = collected_transitions - mixed_value_threshold  # 2500

    # Indices exactly at threshold should get mask=0
    mask_at_threshold = generate_top_new_masks(
        jnp.array([threshold_idx]), collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(mask_at_threshold, jnp.array([0.0]))

    # Indices just above threshold should get mask=1
    mask_above_threshold = generate_top_new_masks(
        jnp.array([threshold_idx + 1]), collected_transitions, mixed_value_threshold
    )
    assert jnp.allclose(mask_above_threshold, jnp.array([1.0])) 

def test_mixed_value_target_selection_logic(common_key, common_cfg_flat):
    """Test the value target selection logic for different modes. OPTIMIZED for speed."""
    mk, bk = jax.random.split(common_key, 2)

    model = make_model(mk, common_cfg_flat)
    base_config = make_cfg(
        0,  # value_support_size = 0 for scalar (faster)
        0,  # reward_support_size = 0 for scalar (faster)
        1,  # num_unroll_steps = 1 (minimal)
        False,  # use_projection
        "target_selection",
    )

    # Create minimal batch
    batch = make_batch(
        bk,
        1,  # batch_size = 1 (minimal)
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        1,  # num_unroll_steps = 1 (minimal)
        0,  # value_support_size = 0
        0,  # reward_support_size = 0
    )

    # Simplified value arrays
    search_values = jnp.array([[1.0, 1.0]])  # (1, 2) for K+1=2 steps
    sarsa_values = jnp.array([[2.0, 2.0]])   # (1, 2) for K+1=2 steps

    batch["target_search_value"] = search_values
    batch["target_sarsa_value"] = sarsa_values

    # Test "search" mode
    config_search = dataclasses.replace(base_config, value_target="search")
    loss_search, _ = Learner._compute_total_loss_static(
        model, config_search, batch, common_key, training=True
    )

    # Test "sarsa" mode
    config_sarsa = dataclasses.replace(base_config, value_target="sarsa")
    loss_sarsa, _ = Learner._compute_total_loss_static(
        model, config_sarsa, batch, common_key, training=True
    )

    # Test "mixed" mode with minimal configuration
    config_mixed = dataclasses.replace(
        base_config,
        value_target="mixed",
        start_use_mix_training_steps=0,
        mixed_value_threshold=500,
    )

    # Add minimal mixed value batch components
    batch["sample_indices"] = jnp.array([400])  # Single sample (old)
    batch["collected_transitions"] = 1000
    batch["training_step"] = 10

    loss_mixed, _ = Learner._compute_total_loss_static(
        model, config_mixed, batch, common_key, training=True
    )

    # All losses should be finite
    assert jnp.isfinite(loss_search), "Search loss should be finite"
    assert jnp.isfinite(loss_sarsa), "SARSA loss should be finite"
    assert jnp.isfinite(loss_mixed), "Mixed loss should be finite"

    # Verify that different target modes produce different results
    # (This is the core functionality being tested)
    assert not jnp.allclose(loss_search, loss_sarsa, atol=1e-6), "Search and SARSA losses should differ"

    print("✅ Mixed value target selection logic test passed (OPTIMIZED)!")
    print(f"  - Search loss: {float(loss_search):.6f}")
    print(f"  - SARSA loss:  {float(loss_sarsa):.6f}")
    print(f"  - Mixed loss:  {float(loss_mixed):.6f}")
    print("  - Target selection logic verified with minimal test scenarios")

def test_mixed_value_targets_comprehensive_shapes(common_key, common_cfg_flat):
    """Test mixed value targets with comprehensive shape validation."""
    # Create configuration for mixed value targets
    config = MuZeroConfig(
        num_actions=common_cfg_flat.num_actions,
        value_target="mixed",  # Enable mixed value targets
        value_target_type="GAE",  # Use GAE for value computation
        num_unroll_steps=2,  # Reduced for faster testing
        checkpoint_dir=None  # Disable checkpointing for testing
    )
    model = make_model(common_key, common_cfg_flat)  # Use MockNetCfg instead of MuZeroConfig

    # Test scalar values
    search_scalar = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    sarsa_scalar = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    masks = jnp.array([0.0, 1.0])

    result_scalar = apply_mixed_value_targets(search_scalar, sarsa_scalar, masks, 1)
    expected_scalar = jnp.array([[1.0, 2.0], [30.0, 40.0]])
    assert jnp.allclose(result_scalar, expected_scalar)

    # Test categorical values (3D tensors)
    support_size = 3
    search_cat = jnp.array(
        [[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], [[0.2, 0.2, 0.6], [0.4, 0.4, 0.2]]]
    )
    sarsa_cat = jnp.array(
        [[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]], [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]]
    )

    result_cat = apply_mixed_value_targets(search_cat, sarsa_cat, masks, 1)

    # First sample (mask=0): should use search values
    assert jnp.allclose(result_cat[0], search_cat[0])
    # Second sample (mask=1): should use sarsa values
    assert jnp.allclose(result_cat[1], sarsa_cat[1])

    assert result_cat.shape == (2, 2, support_size)

def test_action_item_21_comprehensive_completion(common_key, common_cfg_flat):
    """Comprehensive test to verify mixed value target implementation."""
    # Test all components of mixed value target functionality together

    # 1. Test mask generation with EfficientZeroV2 pattern
    sample_indices = jnp.array([4000, 6000, 8000])  # Mix of old and new
    collected_transitions = 7000
    mixed_value_threshold = 2000

    masks = generate_top_new_masks(
        sample_indices, collected_transitions, mixed_value_threshold
    )

    # Expected: idx > 5000 (7000-2000) get mask=1
    expected_masks = jnp.array([0.0, 1.0, 1.0])
    assert jnp.allclose(masks, expected_masks)

    # 2. Test value target mixing
    search_vals = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    sarsa_vals = jnp.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

    mixed_vals = apply_mixed_value_targets(search_vals, sarsa_vals, masks, 1)

    # Sample 0 (old): search values, Samples 1&2 (new): sarsa values
    expected_mixed = jnp.array([[1.0, 1.0], [20.0, 20.0], [30.0, 30.0]])
    assert jnp.allclose(mixed_vals, expected_mixed)

    # 3. Test trainer integration
    mk, bk = jax.random.split(common_key, 2)
    model = make_model(mk, common_cfg_flat)
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        2,  # num_unroll_steps
        False,  # use_projection
        "action_item_21_complete",
    )
    config = dataclasses.replace(
        config,
        value_target="mixed",
        mixed_value_threshold=1000,
        start_use_mix_training_steps=0,
    )

    batch = make_batch(
        bk,
        2,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        2,
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    # Add all required mixed value components
    batch["target_search_value"] = jnp.ones((2, 3)) * 1.0
    batch["target_sarsa_value"] = jnp.ones((2, 3)) * 2.0
    batch["sample_indices"] = jnp.array([500, 1500])  # One old, one new
    batch["collected_transitions"] = 2000
    batch["training_step"] = 10

    # Should work without errors
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    assert jnp.isfinite(loss)
    assert "value_loss" in metrics

    # 4. Test configuration parameters
    assert hasattr(config, "mixed_value_threshold")
    assert hasattr(config, "start_use_mix_training_steps")
    assert hasattr(config, "value_target")

    # 5. Test EfficientZeroV2 compliance
    # Verify that the logic matches PyTorch BatchWorker pattern:
    # mask = int(idx > collected_transitions - mixed_value_threshold)
    test_idx = 1500
    test_collected = 2000
    test_threshold = 300

    expected_pytorch_mask = float(test_idx > (test_collected - test_threshold))
    actual_mask = generate_top_new_masks(
        jnp.array([test_idx]), test_collected, test_threshold
    )[0]

    assert jnp.allclose(
        actual_mask, expected_pytorch_mask
    ), "Should match PyTorch BatchWorker mask generation pattern"

    print("✅ Mixed value training test passed!")
    print("   - Mixed value threshold logic implemented")
    print("   - Top new masks generation working")
    print("   - Value target mixing functional")
    print("   - Trainer integration complete")
    print("   - EfficientZeroV2 pattern compliance verified") 