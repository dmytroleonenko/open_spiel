"""Core tests for BatchContentValidator class methods.

Tests core functionality of the BatchContentValidator class including:
- Validator initialization
- Field property validation (types, dtypes)
- Field shape validation
- Configuration-based requirements
- Full validation workflows
"""

import jax
import jax.numpy as jnp
import pytest
from typing import Dict, Any

from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig, Batch
from open_spiel.python.algorithms.muzero_jax.training.batch_validator import (
    BatchContentValidator,
    BatchValidationResult,
    create_minimal_compatible_batch
)


class TestBatchContentValidator:
    """Test the BatchContentValidator class methods."""
    
    def test_validator_initialization(self):
        """Test BatchContentValidator initialization."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        assert validator.config == config
        assert hasattr(validator, 'CORE_REQUIRED_FIELDS')
        assert hasattr(validator, 'CONDITIONAL_FIELDS')
        assert hasattr(validator, 'PYTORCH_EQUIVALENCE')
    
    def test_validate_field_properties_valid_jax_arrays(self):
        """Test _validate_field_properties with valid JAX arrays."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3)
        validator = BatchContentValidator(config)
        
        # Valid JAX array
        field_value = jnp.array([1, 2, 3], dtype=jnp.int32)
        errors = validator._validate_field_properties('action', field_value)
        assert len(errors) == 0
        
        # Valid scalar
        errors = validator._validate_field_properties('some_field', 1.0)
        assert len(errors) == 0
    
    def test_validate_field_properties_invalid_types(self):
        """Test _validate_field_properties with invalid types."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3)
        validator = BatchContentValidator(config)
        
        # Invalid type (list)
        field_value = [1, 2, 3]
        errors = validator._validate_field_properties('action', field_value)
        assert len(errors) == 1
        assert "Expected JAX array or scalar" in errors[0]
        
        # Invalid type (string)
        errors = validator._validate_field_properties('action', "invalid")
        assert len(errors) == 1
        assert "Expected JAX array or scalar" in errors[0]
    
    def test_validate_field_properties_dtype_mismatch(self):
        """Test _validate_field_properties with dtype mismatches."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3)
        validator = BatchContentValidator(config)
        
        # Wrong dtype for action field
        field_value = jnp.array([1, 2, 3], dtype=jnp.float32)  # Should be int32
        errors = validator._validate_field_properties('action', field_value)
        assert len(errors) == 1
        assert "Expected dtype" in errors[0]
        assert "int32" in errors[0]
    
    def test_get_expected_dtype(self):
        """Test _get_expected_dtype method."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3)
        validator = BatchContentValidator(config)
        
        assert validator._get_expected_dtype('action') == jnp.int32
        assert validator._get_expected_dtype('policy_masks') == jnp.float32
        assert validator._get_expected_dtype('indices') == jnp.int32
        assert validator._get_expected_dtype('extra_dones') == jnp.bool_
        assert validator._get_expected_dtype('unknown_field') is None
    
    def test_validate_field_shape_scalars(self):
        """Test _validate_field_shape with scalar values."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3)
        validator = BatchContentValidator(config)
        
        # Scalars should not have shape constraints
        error = validator._validate_field_shape('some_field', 1.0)
        assert error is None
        
        error = validator._validate_field_shape('some_field', 42)
        assert error is None
    
    def test_validate_field_shape_valid_arrays(self):
        """Test _validate_field_shape with correctly shaped arrays."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Valid action shape: (B, K)
        action = jnp.zeros((4, 3), dtype=jnp.int32)
        error = validator._validate_field_shape('action', action)
        assert error is None
        
        # Valid weights shape: (B,)
        weights = jnp.ones((4,))
        error = validator._validate_field_shape('weights', weights)
        assert error is None
    
    def test_validate_field_shape_invalid_arrays(self):
        """Test _validate_field_shape with incorrectly shaped arrays."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Invalid action shape: should be (4, 3) but is (4, 5)
        action = jnp.zeros((4, 5), dtype=jnp.int32)
        error = validator._validate_field_shape('action', action)
        assert error is not None
        assert "Expected shape (4, 3)" in error
        assert "got (4, 5)" in error
    
    def test_get_expected_shape_basic_fields(self):
        """Test _get_expected_shape for basic field types."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Action: (B, K)
        shape = validator._get_expected_shape('action', (4, 3))
        assert shape == (4, 3)
        
        # Weights: (B,)
        shape = validator._get_expected_shape('weights', (4,))
        assert shape == (4,)
        
        # Target policy: (B, K+1, A)
        shape = validator._get_expected_shape('target_policy', (4, 4, 6))
        assert shape == (4, 4, 6)
    
    def test_get_expected_shape_observation_fields(self):
        """Test _get_expected_shape for observation fields with variable shapes."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Valid observation shapes (different observation dimensions)
        shape = validator._get_expected_shape('observation', (4, 4, 5))  # (B, K+1, obs_dim)
        assert shape is None  # Should accept variable observation shapes
        
        shape = validator._get_expected_shape('observation', (4, 4, 84, 84, 3))  # Image obs
        assert shape is None  # Should accept variable observation shapes
        
        # Invalid observation shape (wrong time dimension)
        shape = validator._get_expected_shape('observation', (4, 2, 5))  # Wrong K+1
        assert shape == (4, 4)  # Should expect correct batch and time dimensions
    
    def test_get_expected_shape_special_fields(self):
        """Test _get_expected_shape for special fields like batch_actions and extras."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            num_sampled_actions=8, gae_max_steps=5
        )
        validator = BatchContentValidator(config)
        
        # batch_actions: (B, K+1, num_sampled_actions, A)
        shape = validator._get_expected_shape('batch_actions', (4, 4, 8, 6))
        assert shape == (4, 4, 8, 6)
        
        # extra_actions: (B, K + gae_max_steps)
        shape = validator._get_expected_shape('extra_actions', (4, 8))  # K=3, gae=5 -> 3+5=8
        assert shape == (4, 8)
        
        # extra_observations with GAE steps
        shape = validator._get_expected_shape('extra_observations', (4, 9, 5))  # (B, K+1+gae, obs_dim)
        assert shape is None  # Should accept variable obs dimensions
    
    def test_get_expected_shape_support_dimensions(self):
        """Test _get_expected_shape for fields with optional support dimensions."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Scalar version: (B, K+1)
        shape = validator._get_expected_shape('target_value', (4, 4))
        assert shape == (4, 4)
        
        # Categorical version: (B, K+1, support_size) - should not validate support size
        shape = validator._get_expected_shape('target_value', (4, 4, 601))
        assert shape is None  # Should accept categorical version
    
    def test_is_field_required_priority_replay(self):
        """Test _is_field_required for priority replay fields."""
        config_enabled = MuZeroConfig(use_priority_replay=True)
        config_disabled = MuZeroConfig(use_priority_replay=False)
        
        validator_enabled = BatchContentValidator(config_enabled)
        validator_disabled = BatchContentValidator(config_disabled)
        
        assert validator_enabled._is_field_required('weights', 'use_priority_replay') is True
        assert validator_disabled._is_field_required('weights', 'use_priority_replay') is False
    
    def test_is_field_required_value_targets(self):
        """Test _is_field_required for value target fields."""
        config = MuZeroConfig(value_target='mixed')
        validator = BatchContentValidator(config)
        
        assert validator._is_field_required('target_search_value', 'value_target') is True
        assert validator._is_field_required('target_sarsa_value', 'value_target') is True
        assert validator._is_field_required('top_new_masks', 'value_target') is True
        
        config_search = MuZeroConfig(value_target='search')
        validator_search = BatchContentValidator(config_search)
        assert validator_search._is_field_required('target_search_value', 'value_target') is True
        assert validator_search._is_field_required('top_new_masks', 'value_target') is False
    
    def test_is_field_required_gae(self):
        """Test _is_field_required for GAE fields."""
        config_gae = MuZeroConfig(value_target_type='GAE')
        config_other = MuZeroConfig(value_target_type='bootstrapped')
        
        validator_gae = BatchContentValidator(config_gae)
        validator_other = BatchContentValidator(config_other)
        
        assert validator_gae._is_field_required('extra_observations', 'value_target_type') is True
        assert validator_other._is_field_required('extra_observations', 'value_target_type') is False
    
    def test_is_field_required_reanalysis(self):
        """Test _is_field_required for policy reanalysis fields."""
        config_reanalysis = MuZeroConfig(reanalyze_ratio=0.8)
        config_no_reanalysis = MuZeroConfig(reanalyze_ratio=0.0)
        
        validator_reanalysis = BatchContentValidator(config_reanalysis)
        validator_no_reanalysis = BatchContentValidator(config_no_reanalysis)
        
        assert validator_reanalysis._is_field_required('policy_masks', 'reanalyze_ratio') is True
        assert validator_no_reanalysis._is_field_required('policy_masks', 'reanalyze_ratio') is False
    
    def test_is_field_required_value_prefix(self):
        """Test _is_field_required for value prefix fields."""
        config_prefix = MuZeroConfig(use_value_prefix=True)
        config_no_prefix = MuZeroConfig(use_value_prefix=False)
        
        validator_prefix = BatchContentValidator(config_prefix)
        validator_no_prefix = BatchContentValidator(config_no_prefix)
        
        assert validator_prefix._is_field_required('value_prefix', 'use_value_prefix') is True
        assert validator_no_prefix._is_field_required('value_prefix', 'use_value_prefix') is False
    
    def test_validate_pytorch_equivalence(self):
        """Test _validate_pytorch_equivalence method."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Create a batch with basic required fields
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Should not raise issues for valid batch
        issues = validator._validate_pytorch_equivalence(batch)
        # We expect this to return a list (may be empty)
        assert isinstance(issues, list)
    
    def test_generate_warnings(self):
        """Test _generate_warnings method."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Create a basic batch
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Should return a list of warnings
        warnings = validator._generate_warnings(batch)
        assert isinstance(warnings, list)
    
    def test_full_validation_with_missing_required_fields(self):
        """Test full validate_batch with missing required fields."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Empty batch
        batch = {}
        
        result = validator.validate_batch(batch, strict=False)
        assert not result.is_valid
        assert len(result.missing_required_fields) > 0
        
        # Check that all core required fields are reported missing
        missing_field_names = [field.split(':')[0] for field in result.missing_required_fields]
        for required_field in validator.CORE_REQUIRED_FIELDS:
            assert required_field in missing_field_names
    
    def test_full_validation_with_conditional_fields(self):
        """Test full validate_batch with conditional fields."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            use_priority_replay=True, reanalyze_ratio=0.8
        )
        validator = BatchContentValidator(config)
        
        # Create basic batch without conditional fields
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Remove conditional fields to test validation
        if 'weights' in batch:
            del batch['weights']
        if 'policy_masks' in batch:
            del batch['policy_masks']
        
        result = validator.validate_batch(batch, strict=True)
        
        # Should report missing conditional fields in strict mode
        assert len(result.missing_required_fields) > 0 or len(result.missing_optional_fields) > 0
    
    def test_strict_vs_non_strict_validation(self):
        """Test difference between strict and non-strict validation."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            use_priority_replay=True
        )
        validator = BatchContentValidator(config)
        
        # Create basic batch without priority replay fields
        batch = {
            'observation': jnp.zeros((4, 4, 5)),
            'action': jnp.zeros((4, 3), dtype=jnp.int32),
            'target_reward': jnp.zeros((4, 4)),
            'target_value': jnp.zeros((4, 4)),
            'target_policy': jnp.zeros((4, 4, 6)),
            'game_history_mask': jnp.ones((4, 4)),
            # Missing: weights, indices, priorities
        }
        
        result_non_strict = validator.validate_batch(batch, strict=False)
        result_strict = validator.validate_batch(batch, strict=True)
        
        # Non-strict should be more lenient
        assert len(result_strict.missing_required_fields) >= len(result_non_strict.missing_required_fields)
    
    def test_validate_batch_conditional_field_validation(self):
        """Test that conditional fields are properly validated when present."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            use_priority_replay=True, reanalyze_ratio=0.8
        )
        validator = BatchContentValidator(config)
        
        # Create batch with conditional fields but wrong types/shapes
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Add conditional field with wrong dtype
        batch['weights'] = jnp.array([1, 2, 3, 4], dtype=jnp.int32)  # Should be float32
        batch['policy_masks'] = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)  # Wrong shape and dtype
        
        result = validator.validate_batch(batch, strict=True)
        
        # Should catch type and shape errors
        assert len(result.type_errors) > 0 or len(result.shape_errors) > 0
    
    def test_validate_batch_exercises_full_validation_path(self):
        """Test that exercises the complete validation path including all method calls."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            use_priority_replay=True,
            value_target='mixed',
            value_target_type='GAE',
            reanalyze_ratio=0.8,
            use_value_prefix=True,
            mixed_value_threshold=1000
        )
        validator = BatchContentValidator(config)
        
        # Create a valid batch that will exercise all code paths
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Run full validation - this should exercise:
        # - All core required field validation
        # - All conditional field validation
        # - _validate_field_properties for each field
        # - _validate_field_shape for each field  
        # - _validate_pytorch_equivalence
        # - _generate_warnings
        result = validator.validate_batch(batch, strict=True)
        
        # Should be valid
        assert result.is_valid
        assert isinstance(result.compatibility_issues, list)
        assert isinstance(result.warnings, list) 