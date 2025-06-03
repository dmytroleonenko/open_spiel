"""Utility function tests for batch_validator module.

Tests utility functions and helper classes including:
- validate_data_flow_compatibility function
- create_minimal_compatible_batch function
- BatchValidationResult dataclass
- Custom observation shapes and RNG key handling
"""

import jax
import jax.numpy as jnp
import pytest
from typing import Dict, Any

from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig, Batch
from open_spiel.python.algorithms.muzero_jax.training.batch_validator import (
    BatchContentValidator,
    BatchValidationResult,
    validate_data_flow_compatibility,
    create_minimal_compatible_batch
)


class TestUtilityFunctions:
    """Test utility functions in batch_validator module."""
    
    def test_validate_data_flow_compatibility_basic(self):
        """Test validate_data_flow_compatibility with basic configuration."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        result = validate_data_flow_compatibility(batch, config)
        assert isinstance(result, BatchValidationResult)
    
    def test_validate_data_flow_compatibility_with_checks(self):
        """Test validate_data_flow_compatibility with specific checks enabled."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            value_target_type='GAE', reanalyze_ratio=0.8, value_target='mixed'
        )
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        result = validate_data_flow_compatibility(
            batch, config,
            check_gae=True,
            check_reanalysis=True,
            check_mixed_targets=True
        )
        assert isinstance(result, BatchValidationResult)
    
    def test_create_minimal_compatible_batch_basic(self):
        """Test create_minimal_compatible_batch with basic configuration."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Check basic structure
        assert isinstance(batch, dict)
        assert len(batch) > 0
        
        # Check that all core required fields are present
        validator = BatchContentValidator(config)
        for field in validator.CORE_REQUIRED_FIELDS:
            assert field in batch, f"Missing core field: {field}"
    
    def test_create_minimal_compatible_batch_with_features(self):
        """Test create_minimal_compatible_batch with various features enabled."""
        config = MuZeroConfig(
            num_actions=6, num_unroll_steps=3, batch_size=4,
            use_priority_replay=True,
            value_target='mixed',
            value_target_type='GAE',
            reanalyze_ratio=0.8,
            use_value_prefix=True
        )
        
        batch = create_minimal_compatible_batch(config, batch_size=4)
        
        # Should include conditional fields based on config
        assert 'weights' in batch  # Priority replay
        assert 'top_new_masks' in batch  # Mixed targets
        assert 'policy_masks' in batch  # Reanalysis
        assert 'value_prefix' in batch  # Value prefix
    
    def test_create_minimal_compatible_batch_custom_obs_shape(self):
        """Test create_minimal_compatible_batch with custom observation shape."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        
        # Test different observation shapes
        obs_shapes = [(8,), (84, 84, 4), (10, 10)]
        
        for obs_shape in obs_shapes:
            batch = create_minimal_compatible_batch(config, batch_size=4, obs_shape=obs_shape)
            
            # Check observation shape
            assert batch['observation'].shape[:2] == (4, 4)  # (batch_size, K+1)
            assert batch['observation'].shape[2:] == obs_shape
    
    def test_create_minimal_compatible_batch_with_rng_key(self):
        """Test create_minimal_compatible_batch with custom RNG key."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        rng_key = jax.random.key(12345)
        
        batch1 = create_minimal_compatible_batch(config, batch_size=4, rng_key=rng_key)
        batch2 = create_minimal_compatible_batch(config, batch_size=4, rng_key=rng_key)
        
        # With same RNG key, should get identical results
        assert jnp.allclose(batch1['target_reward'], batch2['target_reward'])
    
    def test_create_minimal_compatible_batch_field_shapes(self):
        """Test that create_minimal_compatible_batch creates correctly shaped fields."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        
        batch = create_minimal_compatible_batch(config, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Validate all field shapes
        for field_name, field_value in batch.items():
            if isinstance(field_value, jax.Array):
                shape_error = validator._validate_field_shape(field_name, field_value)
                assert shape_error is None, f"Shape error for {field_name}: {shape_error}"
    
    def test_create_minimal_compatible_batch_field_dtypes(self):
        """Test that create_minimal_compatible_batch creates correctly typed fields."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        
        batch = create_minimal_compatible_batch(config, batch_size=4)
        validator = BatchContentValidator(config)
        
        # Validate field dtypes
        for field_name, field_value in batch.items():
            if isinstance(field_value, jax.Array):
                dtype_errors = validator._validate_field_properties(field_name, field_value)
                assert len(dtype_errors) == 0, f"Dtype errors for {field_name}: {dtype_errors}"


class TestBatchValidationResult:
    """Test BatchValidationResult dataclass."""
    
    def test_batch_validation_result_creation(self):
        """Test BatchValidationResult creation and attributes."""
        result = BatchValidationResult(
            is_valid=True,
            missing_required_fields=[],
            missing_optional_fields=[],
            shape_errors=[],
            type_errors=[],
            compatibility_issues=[],
            warnings=[]
        )
        
        assert result.is_valid is True
        assert isinstance(result.missing_required_fields, list)
        assert isinstance(result.missing_optional_fields, list)
        assert isinstance(result.shape_errors, list)
        assert isinstance(result.type_errors, list)
        assert isinstance(result.compatibility_issues, list)
        assert isinstance(result.warnings, list)
    
    def test_batch_validation_result_with_errors(self):
        """Test BatchValidationResult with various error types."""
        result = BatchValidationResult(
            is_valid=False,
            missing_required_fields=["field1: description"],
            missing_optional_fields=["field2: description"],
            shape_errors=["shape error"],
            type_errors=["type error"],
            compatibility_issues=["compatibility issue"],
            warnings=["warning"]
        )
        
        assert result.is_valid is False
        assert len(result.missing_required_fields) == 1
        assert len(result.missing_optional_fields) == 1
        assert len(result.shape_errors) == 1
        assert len(result.type_errors) == 1
        assert len(result.compatibility_issues) == 1
        assert len(result.warnings) == 1 