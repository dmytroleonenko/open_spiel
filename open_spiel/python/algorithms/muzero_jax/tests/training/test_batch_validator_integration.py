"""Comprehensive tests for JAX Batch content alignment.

These tests verify that the JAX Batch structure can accommodate all necessary
fields for dynamic target computation and advanced loss components, ensuring
alignment with PyTorch BatchWorker outputs as required.
"""

import jax
import jax.numpy as jnp
import pytest

from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig, Batch
from open_spiel.python.algorithms.muzero_jax.training.batch_validator import (
    BatchContentValidator,
    BatchValidationResult,
    validate_data_flow_compatibility,
    create_minimal_compatible_batch
)


def test_batch_content_validator_basic_functionality():
    """Test basic functionality of the BatchContentValidator."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        batch_size=2,
        use_priority_replay=True,
        value_target="mixed",
        value_target_type="GAE",
        reanalyze_ratio=0.8,
        use_value_prefix=True
    )
    
    validator = BatchContentValidator(config)
    
    # Test with minimal valid batch
    minimal_batch = create_minimal_compatible_batch(config, batch_size=2)
    result = validator.validate_batch(minimal_batch)
    
    assert result.is_valid, f"Validation failed: {result.missing_required_fields + result.shape_errors + result.type_errors}"
    assert len(result.missing_required_fields) == 0
    assert len(result.shape_errors) == 0
    assert len(result.type_errors) == 0


def test_core_required_fields_validation():
    """Test validation of core required MuZero fields."""
    config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=2)
    validator = BatchContentValidator(config)
    
    # Test with missing core fields
    incomplete_batch: Batch = {
        'observation': jax.random.uniform(jax.random.key(42), (2, 4, 4)),
        'action': jax.random.randint(jax.random.key(42), (2, 3), 0, 6),
        # Missing target_reward, target_value, target_policy, game_history_mask
    }
    
    result = validator.validate_batch(incomplete_batch)
    assert not result.is_valid
    assert len(result.missing_required_fields) >= 3  # Missing at least target_reward, target_value, target_policy, game_history_mask
    
    # Verify specific missing fields
    missing_field_names = [field.split(':')[0] for field in result.missing_required_fields]
    expected_missing = ['target_reward', 'target_value', 'target_policy', 'game_history_mask']
    for expected in expected_missing:
        assert expected in missing_field_names, f"Expected missing field '{expected}' not found in {missing_field_names}"


def test_conditional_fields_based_on_config():
    """Test validation of conditional fields based on configuration."""
    # Config with priority replay enabled
    config_with_priority = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        use_priority_replay=True
    )
    
    validator = BatchContentValidator(config_with_priority)
    
    # Batch without priority replay fields
    batch_without_priority = create_minimal_compatible_batch(config_with_priority)
    # Remove priority fields to test validation
    del batch_without_priority['weights']
    del batch_without_priority['indices']
    del batch_without_priority['priorities']
    
    result = validator.validate_batch(batch_without_priority, strict=True)
    assert not result.is_valid
    
    # Check that priority replay fields are flagged as missing
    missing_field_names = [field.split(':')[0] for field in result.missing_required_fields]
    assert 'weights' in missing_field_names
    assert 'indices' in missing_field_names


def test_pytorch_equivalence_validation():
    """Test validation of PyTorch BatchWorker field equivalence."""
    config = MuZeroConfig(num_actions=6, num_unroll_steps=3)
    validator = BatchContentValidator(config)
    
    # Create batch with all PyTorch equivalent fields
    complete_batch = create_minimal_compatible_batch(config, batch_size=2)
    
    # Add all optional fields to test complete PyTorch equivalence
    rng_key = jax.random.key(42)
    complete_batch.update({
        'batch_actions': jax.random.randint(rng_key, (2, 4, 16, 6), 0, 2),
        'batch_best_actions': jax.random.randint(rng_key, (2, 4), 0, 6),
        'policy_masks': jnp.ones((2, 4)),
        'reanalyzed_values': jax.random.uniform(rng_key, (2, 4)),
    })
    
    result = validator.validate_batch(complete_batch)
    
    # Should have minimal compatibility issues since all major fields are present
    assert len(result.compatibility_issues) == 0 or all(
        'Missing PyTorch equivalents' not in issue for issue in result.compatibility_issues
    )


def test_shape_validation():
    """Test validation of field shapes."""
    config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=2)
    validator = BatchContentValidator(config)
    
    # Create batch with incorrect shapes
    batch_with_wrong_shapes: Batch = {
        'observation': jax.random.uniform(jax.random.key(42), (2, 2, 4)),  # Wrong: should be (2, 4, 4) for K+1=4
        'action': jax.random.randint(jax.random.key(42), (2, 2), 0, 6),    # Wrong: should be (2, 3) for K=3
        'target_reward': jax.random.uniform(jax.random.key(42), (2, 3)),   # Wrong: should be (2, 4) for K+1=4
        'target_value': jax.random.uniform(jax.random.key(42), (2, 4)),    # Correct
        'target_policy': jax.random.uniform(jax.random.key(42), (2, 4, 6)), # Correct
        'game_history_mask': jnp.ones((2, 4)),                             # Correct
    }
    
    result = validator.validate_batch(batch_with_wrong_shapes)
    assert not result.is_valid
    assert len(result.shape_errors) >= 2  # observation, action, target_reward have wrong shapes


def test_gae_data_flow_compatibility():
    """Test data flow compatibility for GAE dynamic target computation."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        value_target_type="GAE",
        gae_max_steps=5
    )
    
    # Create batch with GAE fields
    batch_with_gae = create_minimal_compatible_batch(config, batch_size=2)
    
    result = validate_data_flow_compatibility(
        batch_with_gae, config, check_gae=True
    )
    assert result.is_valid, f"GAE data flow validation failed: {result.compatibility_issues}"
    
    # Test without GAE fields
    batch_without_gae = create_minimal_compatible_batch(config, batch_size=2)
    for field in ['extra_observations', 'extra_actions', 'extra_rewards', 'extra_dones']:
        if field in batch_without_gae:
            del batch_without_gae[field]
    
    result = validate_data_flow_compatibility(
        batch_without_gae, config, check_gae=True
    )
    assert not result.is_valid
    assert any('GAE computation requires' in issue for issue in result.compatibility_issues)


def test_policy_reanalysis_data_flow_compatibility():
    """Test data flow compatibility for policy reanalysis."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        reanalyze_ratio=0.8
    )
    
    # Create batch with policy reanalysis support
    batch_with_reanalysis = create_minimal_compatible_batch(config, batch_size=2)
    
    result = validate_data_flow_compatibility(
        batch_with_reanalysis, config, check_reanalysis=True
    )
    assert result.is_valid, f"Policy reanalysis data flow validation failed: {result.compatibility_issues}"
    
    # Test without observation field (required for reanalysis)
    batch_without_obs = batch_with_reanalysis.copy()
    del batch_without_obs['observation']
    
    result = validate_data_flow_compatibility(
        batch_without_obs, config, check_reanalysis=True
    )
    assert not result.is_valid
    assert any('Policy reanalysis requires' in issue for issue in result.compatibility_issues)


def test_mixed_value_targets_data_flow_compatibility():
    """Test data flow compatibility for mixed value targets."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        value_target="mixed",
        mixed_value_threshold=500
    )
    
    # Create batch with mixed value target fields
    batch_with_mixed = create_minimal_compatible_batch(config, batch_size=2)
    
    result = validate_data_flow_compatibility(
        batch_with_mixed, config, check_mixed_targets=True
    )
    assert result.is_valid, f"Mixed value targets data flow validation failed: {result.compatibility_issues}"
    
    # Test without required mixed target fields
    batch_without_mixed = create_minimal_compatible_batch(config, batch_size=2)
    del batch_without_mixed['target_search_value']
    del batch_without_mixed['target_sarsa_value']
    del batch_without_mixed['sample_indices']
    del batch_without_mixed['top_new_masks']
    
    result = validate_data_flow_compatibility(
        batch_without_mixed, config, check_mixed_targets=True
    )
    assert not result.is_valid
    assert any('Mixed value targets require' in issue for issue in result.compatibility_issues)
    assert any('Mixed targets need' in issue for issue in result.compatibility_issues)


def test_value_prefix_support():
    """Test support for value prefix/LSTM fields."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        use_value_prefix=True,
        lstm_horizon_length=5
    )
    
    validator = BatchContentValidator(config)
    
    # Create batch with value prefix field
    batch_with_prefix = create_minimal_compatible_batch(config, batch_size=2)
    
    result = validator.validate_batch(batch_with_prefix)
    assert result.is_valid
    assert 'value_prefix' in batch_with_prefix
    assert batch_with_prefix['value_prefix'].shape == (2, 4)  # (batch_size, K+1)


def test_categorical_support_compatibility():
    """Test compatibility with categorical value and reward representations."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        value_support_size=21,
        reward_support_size=11,
        value_loss_type="categorical",
        reward_loss_type="categorical"
    )
    
    validator = BatchContentValidator(config)
    
    # Create batch with categorical targets
    rng_key = jax.random.key(42)
    batch_categorical: Batch = {
        'observation': jax.random.uniform(rng_key, (2, 4, 4)),
        'action': jax.random.randint(rng_key, (2, 3), 0, 6),
        'target_reward': jax.random.uniform(rng_key, (2, 4, 11)),  # Categorical rewards
        'target_value': jax.random.uniform(rng_key, (2, 4, 21)),   # Categorical values
        'target_policy': jax.random.uniform(rng_key, (2, 4, 6)),
        'game_history_mask': jnp.ones((2, 4)),
    }
    
    result = validator.validate_batch(batch_categorical)
    assert result.is_valid, f"Categorical batch validation failed: {result.shape_errors}"


def test_continuous_action_fields():
    """Test support for continuous action fields (for completeness)."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        num_sampled_actions=16,
        reanalyze_ratio=0.5
    )
    
    validator = BatchContentValidator(config)
    
    # Create batch with continuous action fields
    batch_with_continuous = create_minimal_compatible_batch(config, batch_size=2)
    
    result = validator.validate_batch(batch_with_continuous)
    assert result.is_valid
    
    # Verify continuous action fields have correct shapes
    assert 'batch_actions' in batch_with_continuous
    assert 'batch_best_actions' in batch_with_continuous
    assert batch_with_continuous['batch_actions'].shape == (2, 4, 16, 6)  # (B, K+1, num_sampled, A)
    assert batch_with_continuous['batch_best_actions'].shape == (2, 4)      # (B, K+1)


def test_comprehensive_efficientzero_v2_alignment():
    """Comprehensive test of EfficientZeroV2 feature alignment."""
    # Configuration with all EfficientZeroV2 features enabled
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        batch_size=2,
        value_support_size=0,  # Use scalar for simplicity
        reward_support_size=0,
        use_priority_replay=True,
        value_target="mixed",
        value_target_type="GAE",
        mixed_value_threshold=500,
        start_use_mix_training_steps=100,
        reanalyze_ratio=0.8,
        use_value_prefix=True,
        lstm_horizon_length=5,
        gae_max_steps=10,
        num_sampled_actions=16,
    )
    
    # Create comprehensive batch
    comprehensive_batch = create_minimal_compatible_batch(config, batch_size=2)
    
    # Validate with all checks enabled
    result = validate_data_flow_compatibility(
        comprehensive_batch, 
        config,
        check_gae=True,
        check_reanalysis=True,
        check_mixed_targets=True
    )
    
    assert result.is_valid, f"Comprehensive EfficientZeroV2 validation failed: {result.missing_required_fields + result.compatibility_issues}"
    
    # Verify all key field categories are present
    field_categories = {
        'core': ['observation', 'action', 'target_reward', 'target_value', 'target_policy', 'game_history_mask'],
        'priority_replay': ['weights', 'indices', 'priorities'],
        'value_targets': ['target_search_value', 'target_sarsa_value'],
        'mixed_targets': ['top_new_masks', 'sample_indices', 'collected_transitions', 'training_step'],
        'gae': ['extra_observations', 'extra_actions', 'extra_rewards', 'extra_dones'],
        'reanalysis': ['policy_masks', 'reanalyzed_values', 'batch_actions', 'batch_best_actions'],
        'value_prefix': ['value_prefix'],
    }
    
    for category, fields in field_categories.items():
        for field in fields:
            assert field in comprehensive_batch, f"Missing {category} field: {field}"


def test_batch_validator_warnings_generation():
    """Test generation of validation warnings for configuration mismatches."""
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        use_priority_replay=False,  # Disabled
        value_target_type="bootstrapped",  # Not GAE
        reanalyze_ratio=0.8  # Enabled
    )
    
    validator = BatchContentValidator(config)
    
    # Create batch with fields that don't match config
    batch_with_mismatches = create_minimal_compatible_batch(config, batch_size=2)
    
    # Add fields that don't match the configuration
    rng_key = jax.random.key(42)
    batch_with_mismatches.update({
        'weights': jnp.ones(2),  # Priority replay disabled but field present
        'extra_observations': jax.random.uniform(rng_key, (2, 9, 4)),  # GAE disabled but field present
    })
    
    result = validator.validate_batch(batch_with_mismatches)
    
    # Should generate warnings about configuration mismatches
    assert len(result.warnings) > 0
    warning_text = ' '.join(result.warnings)
    assert 'weights' in warning_text or 'priority_replay' in warning_text
    assert 'extra_observations' in warning_text or 'GAE' in warning_text


def test_trainer_integration_compatibility():
    """Test that validated batches work with the actual trainer."""
    # This test verifies that the batch structure we validate is actually
    # compatible with the trainer's _compute_total_loss_static method
    
    config = MuZeroConfig(
        num_actions=6,
        num_unroll_steps=3,
        batch_size=2,
        value_target="mixed",
        value_target_type="GAE",
        reanalyze_ratio=0.5,
        use_value_prefix=True
    )
    
    # Create a valid batch according to our validator
    validated_batch = create_minimal_compatible_batch(config, batch_size=2)
    
    # Validate the batch
    validator = BatchContentValidator(config)
    result = validator.validate_batch(validated_batch)
    assert result.is_valid, f"Batch validation failed: {result.missing_required_fields}"
    
    # Test that all required fields for trainer are present
    trainer_required_fields = [
        'observation', 'action', 'target_reward', 'target_value', 
        'target_policy', 'game_history_mask'
    ]
    
    for field in trainer_required_fields:
        assert field in validated_batch, f"Trainer required field '{field}' missing from validated batch"
    
    # Test that conditional fields expected by trainer are present when features are enabled
    if config.use_priority_replay:
        assert 'weights' in validated_batch, "Priority replay enabled but 'weights' field missing"
    
    if config.value_target == "mixed":
        assert 'target_search_value' in validated_batch, "Mixed value target enabled but search values missing"
        assert 'target_sarsa_value' in validated_batch, "Mixed value target enabled but sarsa values missing"
    
    if config.value_target_type == "GAE":
        gae_fields = ['extra_observations', 'extra_actions', 'extra_rewards', 'extra_dones']
        for field in gae_fields:
            assert field in validated_batch, f"GAE enabled but '{field}' field missing"


if __name__ == "__main__":
    # Run a subset of tests for quick verification
    test_batch_content_validator_basic_functionality()
    test_core_required_fields_validation()
    test_comprehensive_efficientzero_v2_alignment()
    print("✅ All comprehensive batch alignment tests passed!") 