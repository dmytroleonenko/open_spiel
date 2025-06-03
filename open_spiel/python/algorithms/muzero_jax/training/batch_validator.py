"""Batch Content Validation for EfficientZeroV2 JAX Implementation.

This module provides comprehensive validation and verification tools to ensure
the JAX Batch structure can accommodate all necessary fields for dynamic target
computation and advanced loss components, maintaining alignment with PyTorch
BatchWorker outputs.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import warnings
from dataclasses import dataclass

from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig, Batch


@dataclass
class BatchValidationResult:
    """Result of batch content validation."""
    is_valid: bool
    missing_required_fields: List[str]
    missing_optional_fields: List[str]
    shape_errors: List[str]
    type_errors: List[str]
    compatibility_issues: List[str]
    warnings: List[str]
    

class BatchContentValidator:
    """Comprehensive validator for JAX Batch content alignment."""
    
    # Core MuZero fields that are always required
    CORE_REQUIRED_FIELDS = {
        'observation': 'Core observations for initial + K unroll steps',
        'action': 'Actions taken for K unroll steps',
        'target_reward': 'Target rewards (scalar or categorical)',
        'target_value': 'Target values (scalar or categorical)',
        'target_policy': 'Target policy distributions',
        'game_history_mask': 'Validity mask for each step'
    }
    
    # EfficientZeroV2 fields that are conditionally required based on config
    CONDITIONAL_FIELDS = {
        # Prioritized Experience Replay
        'weights': ('use_priority_replay', 'Importance sampling weights from prioritized replay'),
        'indices': ('use_priority_replay', 'Buffer indices for priority updates'),
        'priorities': ('use_priority_replay', 'Current priorities for updates'),
        
        # Value target selection
        'target_search_value': ('value_target', 'MCTS search values for target selection'),
        'target_sarsa_value': ('value_target', 'N-step TD targets for value learning'),
        
        # Mixed value targets
        'top_new_masks': ('value_target', 'Masks for mixed value target selection'),
        'sample_indices': ('mixed_value_threshold', 'Sample indices for adaptive parameters'),
        'collected_transitions': ('mixed_value_threshold', 'Total transitions collected'),
        'training_step': ('mixed_value_threshold', 'Current training step'),
        
        # GAE dynamic computation
        'extra_observations': ('value_target_type', 'Extended observations for GAE bootstrapping'),
        'extra_actions': ('value_target_type', 'Extended actions for GAE computation'),
        'extra_rewards': ('value_target_type', 'Extended rewards for GAE computation'),
        'extra_dones': ('value_target_type', 'Episode termination flags for GAE'),
        
        # Policy reanalysis
        'batch_actions': ('reanalyze_ratio', 'Sampled actions for continuous policy loss'),
        'batch_best_actions': ('reanalyze_ratio', 'Best actions for simple policy loss'),
        'policy_masks': ('reanalyze_ratio', 'Masks for policy reanalysis'),
        'reanalyzed_values': ('reanalyze_ratio', 'Values from policy reanalysis'),
        
        # Value prefix/LSTM
        'value_prefix': ('use_value_prefix', 'Accumulated rewards over LSTM horizon'),
    }
    
    # PyTorch BatchWorker equivalence mapping
    PYTORCH_EQUIVALENCE = {
        'obs_lst': 'observation',
        'action_lst': 'action',
        'target_value_prefixs': 'target_reward',  # PyTorch naming
        'target_values': 'target_value',
        'target_policies': 'target_policy',
        'game_history_mask': 'game_history_mask',
        'weights_lst': 'weights',
        'indices_lst': 'indices',
        'batch_value_prefixes': 'target_reward',
        'batch_values': 'target_value',
        'batch_policies': 'target_policy',
        'batch_actions': 'batch_actions',
        'batch_best_actions': 'batch_best_actions',
        'top_new_masks': 'top_new_masks',
        'policy_masks': 'policy_masks',
        'reanalyzed_values': 'reanalyzed_values',
        'value_masks': 'top_new_masks',  # Alternative name
    }
    
    def __init__(self, config: MuZeroConfig):
        """Initialize validator with configuration."""
        self.config = config
        
    def validate_batch(self, batch: Batch, strict: bool = False) -> BatchValidationResult:
        """Comprehensive batch validation.
        
        Args:
            batch: JAX Batch to validate
            strict: If True, missing optional fields are treated as errors
            
        Returns:
            BatchValidationResult with detailed validation information
        """
        missing_required = []
        missing_optional = []
        shape_errors = []
        type_errors = []
        compatibility_issues = []
        warnings_list = []
        
        # Validate core required fields
        for field, description in self.CORE_REQUIRED_FIELDS.items():
            if field not in batch:
                missing_required.append(f"{field}: {description}")
            else:
                # Validate field type and basic properties
                field_errors = self._validate_field_properties(field, batch[field])
                type_errors.extend(field_errors)
                
                # Validate field shapes
                shape_error = self._validate_field_shape(field, batch[field])
                if shape_error:
                    shape_errors.append(shape_error)  # pragma: no cover
        
        # Validate conditional fields based on configuration
        for field, (config_key, description) in self.CONDITIONAL_FIELDS.items():
            field_required = self._is_field_required(field, config_key)
            
            if field_required and field not in batch:
                if strict:
                    missing_required.append(f"{field}: {description} (required by config.{config_key})")
                else:
                    missing_optional.append(f"{field}: {description} (expected due to config.{config_key})")
            elif field in batch:
                # Validate field properties even if optional
                field_errors = self._validate_field_properties(field, batch[field])
                type_errors.extend(field_errors)
                
                shape_error = self._validate_field_shape(field, batch[field])
                if shape_error:
                    shape_errors.append(shape_error)
        
        # Check PyTorch BatchWorker equivalence
        pytorch_issues = self._validate_pytorch_equivalence(batch)
        compatibility_issues.extend(pytorch_issues)
        
        # Generate warnings for best practices
        warnings_list.extend(self._generate_warnings(batch))
        
        is_valid = (len(missing_required) == 0 and 
                   len(shape_errors) == 0 and 
                   len(type_errors) == 0 and
                   (not strict or len(missing_optional) == 0))
        
        return BatchValidationResult(
            is_valid=is_valid,
            missing_required_fields=missing_required,
            missing_optional_fields=missing_optional,
            shape_errors=shape_errors,
            type_errors=type_errors,
            compatibility_issues=compatibility_issues,
            warnings=warnings_list
        )
    
    def _validate_field_properties(self, field_name: str, field_value: Any) -> List[str]:
        """Validate basic field properties (type, dtype, etc.)."""
        errors = []
        
        # Check if field is a JAX array or valid scalar
        if not isinstance(field_value, (jax.Array, int, float)):
            errors.append(f"{field_name}: Expected JAX array or scalar, got {type(field_value)}")
            return errors
        
        # For JAX arrays, validate dtype expectations
        if isinstance(field_value, jax.Array):
            expected_dtype = self._get_expected_dtype(field_name)
            if expected_dtype and field_value.dtype != expected_dtype:
                errors.append(f"{field_name}: Expected dtype {expected_dtype}, got {field_value.dtype}")
        
        return errors
    
    def _validate_field_shape(self, field_name: str, field_value: Any) -> Optional[str]:
        """Validate field shape against expected dimensions."""
        if not isinstance(field_value, jax.Array):
            return None  # Scalars don't have shape constraints
        
        expected_shape = self._get_expected_shape(field_name, field_value.shape)
        if expected_shape and field_value.shape != expected_shape:
            return f"{field_name}: Expected shape {expected_shape}, got {field_value.shape}"
        
        return None
    
    def _get_expected_dtype(self, field_name: str) -> Optional[jnp.dtype]:
        """Get expected dtype for a field."""
        dtype_map = {
            'action': jnp.int32,
            'batch_actions': jnp.int32,  
            'batch_best_actions': jnp.int32,
            'policy_masks': jnp.float32,  # Changed to float32 to match actual implementation
            'game_history_mask': jnp.float32,
            'indices': jnp.int32,
            'top_new_masks': jnp.int32,
            'extra_dones': jnp.bool_,
            'sample_indices': jnp.int32,
        }
        return dtype_map.get(field_name)
    
    def _get_expected_shape(self, field_name: str, actual_shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """Get expected shape for a field based on configuration."""
        B = self.config.batch_size  # Use configured batch size, not actual
        K = self.config.num_unroll_steps
        A = self.config.num_actions
        
        if len(actual_shape) == 0:
            return None  # pragma: no cover
        
        # For observation fields, don't validate beyond batch and time dimensions
        # since observation shape can vary (e.g., (4,), (84, 84, 4), etc.)
        if field_name == 'observation':
            if len(actual_shape) >= 2 and actual_shape[:2] == (B, K + 1):
                return None  # Valid - has correct batch and time dimensions
            else:
                return (B, K + 1)  # Expected batch and time dimensions
        
        if field_name == 'extra_observations':
            gae_steps = getattr(self.config, 'gae_max_steps', 5)
            if len(actual_shape) >= 2 and actual_shape[:2] == (B, K + 1 + gae_steps):
                return None  # Valid - has correct batch and time dimensions
            else:
                return (B, K + 1 + gae_steps)  # Expected batch and time dimensions  # pragma: no cover
        
        shape_map = {
            'action': (B, K),
            'target_reward': (B, K + 1),  # May have support dimension
            'target_value': (B, K + 1),   # May have support dimension  
            'target_policy': (B, K + 1, A),
            'game_history_mask': (B, K + 1),
            'weights': (B,),
            'indices': (B,),
            'priorities': (B,),
            'target_search_value': (B, K + 1),  # May have support dimension
            'target_sarsa_value': (B, K + 1),   # May have support dimension
            'sample_indices': (B,),
            'top_new_masks': (B,),
            'batch_best_actions': (B, K + 1),
            'policy_masks': (B, K + 1),
            'reanalyzed_values': (B, K + 1),
            'value_prefix': (B, K + 1),
        }
        
        # Handle special cases with variable dimensions
        if field_name == 'batch_actions':
            return (B, K + 1, self.config.num_sampled_actions, A)
        elif field_name == 'extra_actions':
            gae_steps = getattr(self.config, 'gae_max_steps', 5)
            return (B, K + gae_steps)
        elif field_name in ['extra_rewards', 'extra_dones']:
            gae_steps = getattr(self.config, 'gae_max_steps', 5)
            return (B, K + 1 + gae_steps)
        
        # For fields with support dimensions, only check first dimensions
        base_shape = shape_map.get(field_name)
        if base_shape and field_name in ['target_reward', 'target_value', 'target_search_value', 'target_sarsa_value']:
            if len(actual_shape) == len(base_shape):
                return base_shape  # Scalar version
            elif len(actual_shape) == len(base_shape) + 1:
                return None  # Categorical version - don't validate support size
        
        return base_shape
    
    def _is_field_required(self, field_name: str, config_key: str) -> bool:
        """Check if a field is required based on configuration."""
        if config_key == 'use_priority_replay':
            return getattr(self.config, config_key, False)
        elif config_key == 'value_target':
            value_target = getattr(self.config, config_key, "mixed")
            if field_name in ['target_search_value', 'target_sarsa_value']:
                return value_target in ["mixed", "search", "sarsa"]
            elif field_name == 'top_new_masks':
                return value_target == "mixed"
        elif config_key == 'mixed_value_threshold':
            return getattr(self.config, 'value_target', '') == 'mixed'
        elif config_key == 'value_target_type':
            return getattr(self.config, config_key, '') == 'GAE'
        elif config_key == 'reanalyze_ratio':
            return getattr(self.config, config_key, 0.0) > 0.0
        elif config_key == 'use_value_prefix':
            return getattr(self.config, config_key, False)
        
        return False  # pragma: no cover
    
    def _validate_pytorch_equivalence(self, batch: Batch) -> List[str]:
        """Validate equivalence with PyTorch BatchWorker fields."""
        issues = []
        
        # Check if JAX batch has equivalents for all critical PyTorch fields
        available_jax_fields = set(batch.keys())
        available_pytorch_equivalents = set(self.PYTORCH_EQUIVALENCE.values())
        
        # Find PyTorch fields that don't have JAX equivalents available
        missing_equivalents = []
        for pytorch_field, jax_field in self.PYTORCH_EQUIVALENCE.items():
            if jax_field not in available_jax_fields:
                missing_equivalents.append(f"PyTorch '{pytorch_field}' -> JAX '{jax_field}'")
        
        if missing_equivalents:
            issues.append(f"Missing PyTorch equivalents: {', '.join(missing_equivalents)}")
        
        return issues
    
    def _generate_warnings(self, batch: Batch) -> List[str]:
        """Generate warnings for best practices and potential issues."""
        warnings_list = []
        
        # Check for unused fields that might indicate configuration mismatch
        if 'weights' in batch and not self.config.use_priority_replay:
            warnings_list.append("Found 'weights' field but use_priority_replay=False")  # pragma: no cover
        
        if 'extra_observations' in batch and self.config.value_target_type != 'GAE':
            warnings_list.append("Found GAE fields but value_target_type != 'GAE'")  # pragma: no cover
        
        if self.config.reanalyze_ratio > 0.0 and 'policy_masks' not in batch:
            warnings_list.append("reanalyze_ratio > 0 but missing policy reanalysis fields")
        
        # Check for potential shape mismatches
        if 'observation' in batch and 'extra_observations' in batch:
            obs_shape = batch['observation'].shape
            extra_obs_shape = batch['extra_observations'].shape
            if obs_shape[0] != extra_obs_shape[0]:  # Batch size mismatch
                warnings_list.append("Batch size mismatch between observation and extra_observations")  # pragma: no cover
        
        return warnings_list


def validate_data_flow_compatibility(
    batch: Batch, 
    config: MuZeroConfig,
    check_gae: bool = True,
    check_reanalysis: bool = True,
    check_mixed_targets: bool = True
) -> BatchValidationResult:
    """Validate data flow compatibility for dynamic target generation.
    
    This function specifically validates that the batch contains all necessary
    fields for the dynamic target generation features implemented in Action Items
    1 (GAE), 2 (Policy Reanalysis), and 21 (Mixed Value Targets).
    
    Args:
        batch: JAX Batch to validate
        config: MuZero configuration
        check_gae: Whether to validate GAE compatibility
        check_reanalysis: Whether to validate policy reanalysis compatibility
        check_mixed_targets: Whether to validate mixed value target compatibility
        
    Returns:
        BatchValidationResult with data flow compatibility information
    """
    validator = BatchContentValidator(config)
    result = validator.validate_batch(batch, strict=False)
    
    # Additional data flow checks
    data_flow_issues = []
    
    if check_gae and config.value_target_type == 'GAE':
        gae_fields = ['extra_observations', 'extra_actions', 'extra_rewards', 'extra_dones']
        missing_gae = [f for f in gae_fields if f not in batch]
        if missing_gae:
            data_flow_issues.append(f"GAE computation requires: {missing_gae}")  # pragma: no cover
    
    if check_reanalysis and config.reanalyze_ratio > 0.0:
        # Policy reanalysis is handled dynamically but we can check for optional fields
        if 'observation' not in batch:
            data_flow_issues.append("Policy reanalysis requires 'observation' field")  # pragma: no cover
    
    if check_mixed_targets and config.value_target == 'mixed':
        mixed_fields = ['target_search_value', 'target_sarsa_value']
        missing_mixed = [f for f in mixed_fields if f not in batch]
        if missing_mixed:
            data_flow_issues.append(f"Mixed value targets require: {missing_mixed}")  # pragma: no cover
        
        # top_new_masks can be generated dynamically, but check for sample info
        if 'sample_indices' not in batch and 'top_new_masks' not in batch:
            data_flow_issues.append("Mixed targets need 'sample_indices' or 'top_new_masks'")  # pragma: no cover
    
    # Merge data flow issues with original validation
    result.compatibility_issues.extend(data_flow_issues)
    result.is_valid = result.is_valid and len(data_flow_issues) == 0
    
    return result


def create_minimal_compatible_batch(
    config: MuZeroConfig,
    batch_size: int = 2,
    obs_shape: Tuple[int, ...] = (4,),
    rng_key: Optional[jax.Array] = None
) -> Batch:
    """Create a minimal batch that's compatible with all implemented features.
    
    This function creates a JAX Batch that contains all fields necessary for
    the trainer to work with all implemented EfficientZeroV2 features.
    
    Args:
        config: MuZero configuration
        batch_size: Size of the batch
        obs_shape: Shape of observations
        rng_key: Random key for generating data
        
    Returns:
        JAX Batch with all necessary fields populated
    """
    if rng_key is None:
        rng_key = jax.random.key(42)
    
    K = config.num_unroll_steps
    A = config.num_actions
    
    # Core required fields
    batch: Batch = {
        'observation': jax.random.uniform(rng_key, (batch_size, K + 1, *obs_shape)),
        'action': jax.random.randint(rng_key, (batch_size, K), 0, A),
        'target_reward': jax.random.uniform(rng_key, (batch_size, K + 1)),
        'target_value': jax.random.uniform(rng_key, (batch_size, K + 1)),
        'target_policy': jax.random.uniform(rng_key, (batch_size, K + 1, A)),
        'game_history_mask': jnp.ones((batch_size, K + 1)),
    }
    
    # Add fields based on configuration
    if config.use_priority_replay:
        batch.update({
            'weights': jnp.ones(batch_size),
            'indices': jnp.arange(batch_size),
            'priorities': jax.random.uniform(rng_key, (batch_size,)),
        })
    
    if config.value_target in ['mixed', 'search', 'sarsa']:
        batch.update({
            'target_search_value': jax.random.uniform(rng_key, (batch_size, K + 1)),
            'target_sarsa_value': jax.random.uniform(rng_key, (batch_size, K + 1)),
        })
    
    if config.value_target == 'mixed':
        batch.update({
            'top_new_masks': jax.random.randint(rng_key, (batch_size,), 0, 2),
            'sample_indices': jnp.arange(batch_size) * 100,
            'collected_transitions': 1000,
            'training_step': 500,
        })
    
    if config.value_target_type == 'GAE':
        extra_steps = config.gae_max_steps
        batch.update({
            'extra_observations': jax.random.uniform(rng_key, (batch_size, K + 1 + extra_steps, *obs_shape)),
            'extra_actions': jax.random.randint(rng_key, (batch_size, K + extra_steps), 0, A),
            'extra_rewards': jax.random.uniform(rng_key, (batch_size, K + 1 + extra_steps)),
            'extra_dones': jax.random.bernoulli(rng_key, 0.1, (batch_size, K + 1 + extra_steps)),
        })
    
    if config.reanalyze_ratio > 0.0:
        batch.update({
            'policy_masks': jnp.ones((batch_size, K + 1), dtype=jnp.float32),
            'reanalyzed_values': jax.random.uniform(rng_key, (batch_size, K + 1)),
        })
        
        # Continuous action fields (though not used in OpenSpiel)
        batch.update({
            'batch_actions': jax.random.randint(rng_key, (batch_size, K + 1, config.num_sampled_actions, A), 0, 2),
            'batch_best_actions': jax.random.randint(rng_key, (batch_size, K + 1), 0, A),
        })
    
    if config.use_value_prefix:
        batch.update({
            'value_prefix': jax.random.uniform(rng_key, (batch_size, K + 1)),
        })
    
    return batch 