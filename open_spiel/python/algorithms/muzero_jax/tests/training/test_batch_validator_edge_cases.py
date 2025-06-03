"""Edge case tests for batch_validator module.

Tests edge cases and error conditions including:
- Extreme configuration values
- NaN value handling
- Mismatched batch sizes
- Minimal configuration edge cases
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


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_validator_with_extreme_config_values(self):
        """Test validator with extreme configuration values."""
        config = MuZeroConfig(
            num_actions=1000,
            num_unroll_steps=100,
            batch_size=1,
            gae_max_steps=200,
            num_sampled_actions=50
        )
        
        validator = BatchContentValidator(config)
        batch = create_minimal_compatible_batch(config, batch_size=1)
        
        result = validator.validate_batch(batch, strict=True)
        # Should handle extreme values gracefully
        assert isinstance(result, BatchValidationResult)
    
    def test_batch_with_nan_values(self):
        """Test batch validation with NaN values."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4)
        validator = BatchContentValidator(config)
        
        batch = create_minimal_compatible_batch(config, batch_size=4)
        # Introduce NaN values
        batch['target_value'] = jnp.full_like(batch['target_value'], jnp.nan)
        
        result = validator.validate_batch(batch)
        # Should not crash on NaN values
        assert isinstance(result, BatchValidationResult)
    
    def test_batch_with_mismatched_batch_sizes(self):
        """Test batch validation with mismatched batch sizes across fields."""
        config = MuZeroConfig(num_actions=6, num_unroll_steps=3, batch_size=4, use_priority_replay=True)
        validator = BatchContentValidator(config)
        
        batch = create_minimal_compatible_batch(config, batch_size=4)
        # Create mismatched batch size for a required field
        batch['weights'] = jnp.ones((2,))  # Should be (4,)
        
        result = validator.validate_batch(batch)
        assert not result.is_valid
        assert len(result.shape_errors) > 0
    
    def test_empty_config_edge_cases(self):
        """Test with minimal/empty configuration values."""
        config = MuZeroConfig(
            num_actions=1,
            num_unroll_steps=1,
            batch_size=1
        )
        
        validator = BatchContentValidator(config)
        batch = create_minimal_compatible_batch(config, batch_size=1)
        
        result = validator.validate_batch(batch)
        # Should handle minimal configuration
        assert isinstance(result, BatchValidationResult) 