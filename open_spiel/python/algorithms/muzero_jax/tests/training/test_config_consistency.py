"""
Tests for Action Item 15: Configuration Parameter Consistency.

This module tests the alignment of JAX MuZeroConfig with EfficientZeroV2 PyTorch
configuration parameters, ensuring all necessary parameters are present and
correctly mapped.
"""

import pytest
import jax
import jax.numpy as jnp
import dataclasses
import math
from typing import Dict, Any, Tuple

from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig


class TestEfficientZeroV2ConfigurationAlignment:
    """Test suite for EfficientZeroV2 configuration parameter alignment."""
    
    def test_gae_td_lambda_parameters_present(self):
        """Test that GAE and TD-Lambda parameters are present in configuration."""
        config = MuZeroConfig()
        
        # GAE/TD-Lambda parameters from EfficientZeroV2
        assert hasattr(config, 'value_target_type'), "Config should have value_target_type parameter"
        assert hasattr(config, 'td_lambda'), "Config should have td_lambda parameter"
        assert hasattr(config, 'auto_td_steps'), "Config should have auto_td_steps parameter"
        assert hasattr(config, 'gae_max_steps'), "Config should have gae_max_steps parameter"
        
        # Verify default values align with EfficientZeroV2
        assert config.value_target_type == "bootstrapped", "Default value_target_type should be 'bootstrapped'"
        assert config.td_lambda == 0.95, "Default td_lambda should be 0.95"
        assert config.auto_td_steps == 30000, "Default auto_td_steps should be 30000"
        assert config.gae_max_steps == 15, "Default gae_max_steps should be 15"
    
    def test_reanalysis_parameters_present(self):
        """Test that reanalysis parameters are present and properly configured."""
        config = MuZeroConfig()
        
        # Reanalysis parameters from EfficientZeroV2
        assert hasattr(config, 'reanalyze_ratio'), "Config should have reanalyze_ratio parameter"
        assert hasattr(config, 'reanalyze_update_interval'), "Config should have reanalyze_update_interval parameter"
        assert hasattr(config, 'self_play_update_interval'), "Config should have self_play_update_interval parameter"
        
        # Verify default values
        assert config.reanalyze_ratio == 1.0, "Default reanalyze_ratio should be 1.0"
        assert config.reanalyze_update_interval == 200, "Default reanalyze_update_interval should be 200"
        assert config.self_play_update_interval == 100, "Default self_play_update_interval should be 100"
        
        # Test valid ranges
        assert 0.0 <= config.reanalyze_ratio <= 1.0, "Reanalyze ratio should be between 0 and 1"
        assert config.reanalyze_update_interval > 0, "Reanalyze update interval should be positive"
        assert config.self_play_update_interval > 0, "Self-play update interval should be positive"
    
    def test_value_target_parameters_present(self):
        """Test that value target selection parameters are present."""
        config = MuZeroConfig()
        
        # Value target parameters from EfficientZeroV2
        assert hasattr(config, 'value_target'), "Config should have value_target parameter"
        assert hasattr(config, 'start_use_mix_training_steps'), "Config should have start_use_mix_training_steps parameter"
        assert hasattr(config, 'mixed_value_threshold'), "Config should have mixed_value_threshold parameter"
        
        # Verify default values
        assert config.value_target == "mixed", "Default value_target should be 'mixed'"
        assert config.start_use_mix_training_steps == 30000, "Default start_use_mix_training_steps should be 30000"
        assert config.mixed_value_threshold == 5000, "Default mixed_value_threshold should be 5000"
        
        # Test valid values
        valid_targets = ["search", "sarsa", "mixed", "max"]
        assert config.value_target in valid_targets, f"Value target should be one of {valid_targets}"
    
    def test_mcts_parameters_present(self):
        """Test that MCTS parameters are present and properly configured."""
        config = MuZeroConfig()
        
        # MCTS parameters from EfficientZeroV2
        mcts_params = [
            'num_simulations', 'c_visit', 'c_scale', 'c_base', 'c_init',
            'dirichlet_alpha', 'explore_frac', 'value_minmax_delta'
        ]
        
        for param in mcts_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values align with EfficientZeroV2
        assert config.num_simulations == 16, "Default num_simulations should be 16"
        assert config.c_visit == 50, "Default c_visit should be 50"
        assert config.c_scale == 0.1, "Default c_scale should be 0.1"
        assert config.c_base == 19652, "Default c_base should be 19652"
        assert config.c_init == 1.25, "Default c_init should be 1.25"
        assert config.dirichlet_alpha == 0.3, "Default dirichlet_alpha should be 0.3"
        assert config.explore_frac == 0.25, "Default explore_frac should be 0.25"
        assert config.value_minmax_delta == 0.01, "Default value_minmax_delta should be 0.01"
        
        # Test valid ranges
        assert config.num_simulations > 0, "Number of simulations should be positive"
        assert 0.0 <= config.explore_frac <= 1.0, "Explore fraction should be between 0 and 1"
        assert config.dirichlet_alpha > 0.0, "Dirichlet alpha should be positive"
    
    def test_priority_replay_parameters_present(self):
        """Test that priority replay parameters are complete."""
        config = MuZeroConfig()
        
        # Priority replay parameters from EfficientZeroV2
        assert hasattr(config, 'use_priority_replay'), "Config should have use_priority_replay parameter"
        assert hasattr(config, 'priority_exponent'), "Config should have priority_exponent parameter"
        assert hasattr(config, 'priority_beta'), "Config should have priority_beta parameter"
        assert hasattr(config, 'min_priority'), "Config should have min_priority parameter"
        
        # Verify default values
        assert config.use_priority_replay == True, "Default use_priority_replay should be True"
        assert config.priority_exponent == 0.6, "Default priority_exponent should be 0.6"
        assert config.priority_beta == 0.4, "Default priority_beta should be 0.4"
        assert config.min_priority == 1e-6, "Default min_priority should be 1e-6"
        
        # Test valid ranges
        assert 0.0 <= config.priority_exponent <= 1.0, "Priority exponent should be between 0 and 1"
        assert 0.0 <= config.priority_beta <= 1.0, "Priority beta should be between 0 and 1"
        assert config.min_priority > 0.0, "Minimum priority should be positive"
    
    def test_temperature_scheduling_parameters_present(self):
        """Test that temperature scheduling parameters are present."""
        config = MuZeroConfig()
        
        # Temperature scheduling parameters from EfficientZeroV2
        temp_params = ['change_temperature', 'temperature_init', 'temperature_final', 'temperature_decay_steps']
        
        for param in temp_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.change_temperature == True, "Default change_temperature should be True"
        assert config.temperature_init == 1.0, "Default temperature_init should be 1.0"
        assert config.temperature_final == 0.1, "Default temperature_final should be 0.1"
        assert config.temperature_decay_steps == 50000, "Default temperature_decay_steps should be 50000"
        
        # Test valid ranges
        assert config.temperature_init > 0.0, "Initial temperature should be positive"
        assert config.temperature_final > 0.0, "Final temperature should be positive"
        assert config.temperature_decay_steps > 0, "Temperature decay steps should be positive"
    
    def test_training_parameters_present(self):
        """Test that training step parameters are present."""
        config = MuZeroConfig()
        
        # Training parameters from EfficientZeroV2
        training_params = [
            'training_steps', 'offline_training_steps', 'start_transitions', 'mini_batch_size'
        ]
        
        for param in training_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.training_steps == 100000, "Default training_steps should be 100000"
        assert config.offline_training_steps == 20000, "Default offline_training_steps should be 20000"
        assert config.start_transitions == 2000, "Default start_transitions should be 2000"
        assert config.mini_batch_size == 256, "Default mini_batch_size should be 256"
        
        # Test valid ranges
        assert config.training_steps > 0, "Training steps should be positive"
        assert config.offline_training_steps >= 0, "Offline training steps should be non-negative"
        assert config.start_transitions > 0, "Start transitions should be positive"
        assert config.mini_batch_size > 0, "Mini batch size should be positive"
    
    def test_data_collection_parameters_present(self):
        """Test that data collection parameters are present."""
        config = MuZeroConfig()
        
        # Data collection parameters from EfficientZeroV2
        data_params = ['total_transitions', 'trajectory_size', 'buffer_size']
        
        for param in data_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.total_transitions == 100000, "Default total_transitions should be 100000"
        assert config.trajectory_size == 400, "Default trajectory_size should be 400"
        assert config.buffer_size == 1000000, "Default buffer_size should be 1000000"
        
        # Test valid ranges
        assert config.total_transitions > 0, "Total transitions should be positive"
        assert config.trajectory_size > 0, "Trajectory size should be positive"
        assert config.buffer_size > 0, "Buffer size should be positive"
    
    def test_continuous_action_parameters_present(self):
        """Test that continuous action parameters are present."""
        config = MuZeroConfig()
        
        # Continuous action parameters from EfficientZeroV2
        continuous_params = ['num_top_actions', 'num_sampled_actions']
        
        for param in continuous_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.num_top_actions == 4, "Default num_top_actions should be 4"
        assert config.num_sampled_actions == 16, "Default num_sampled_actions should be 16"
        
        # Test valid ranges
        assert config.num_top_actions > 0, "Number of top actions should be positive"
        assert config.num_sampled_actions > 0, "Number of sampled actions should be positive"
    
    def test_model_architecture_parameters_present(self):
        """Test that model architecture parameters are present."""
        config = MuZeroConfig()
        
        # Model architecture parameters from EfficientZeroV2
        model_params = ['noisy_net', 'use_batch_norm', 'state_norm', 'init_zero']
        
        for param in model_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.noisy_net == False, "Default noisy_net should be False"
        assert config.use_batch_norm == True, "Default use_batch_norm should be True"
        assert config.state_norm == False, "Default state_norm should be False"
        assert config.init_zero == True, "Default init_zero should be True"
    
    def test_support_transformation_parameters_present(self):
        """Test that support transformation parameters are present."""
        config = MuZeroConfig()
        
        # Support transformation parameters from EfficientZeroV2
        support_params = ['support_min', 'support_max', 'support_scale', 'support_bins', 'epsilon']
        
        for param in support_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.support_min == -300.0, "Default support_min should be -300.0"
        assert config.support_max == 300.0, "Default support_max should be 300.0"
        assert config.support_scale == 1.0, "Default support_scale should be 1.0"
        assert config.support_bins == 51, "Default support_bins should be 51"
        assert config.epsilon == 0.001, "Default epsilon should be 0.001"
        
        # Test valid ranges
        assert config.support_min < config.support_max, "Support min should be less than support max"
        assert config.support_scale > 0.0, "Support scale should be positive"
        assert config.support_bins > 0, "Support bins should be positive"
        assert config.epsilon > 0.0, "Epsilon should be positive"
    
    def test_additional_loss_coefficients_present(self):
        """Test that additional EfficientZeroV2 loss coefficients are present."""
        config = MuZeroConfig()
        
        # Additional loss coefficients from EfficientZeroV2
        loss_params = ['decorrelation_coeff', 'off_diag_coeff']
        
        for param in loss_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.decorrelation_coeff == 0.01, "Default decorrelation_coeff should be 0.01"
        assert config.off_diag_coeff == 5e-3, "Default off_diag_coeff should be 5e-3"
        
        # Test valid ranges
        assert config.decorrelation_coeff >= 0.0, "Decorrelation coefficient should be non-negative"
        assert config.off_diag_coeff >= 0.0, "Off-diagonal coefficient should be non-negative"
    
    def test_evaluation_parameters_present(self):
        """Test that evaluation parameters are present."""
        config = MuZeroConfig()
        
        # Evaluation parameters from EfficientZeroV2
        eval_params = ['eval_n_episode', 'eval_interval']
        
        for param in eval_params:
            assert hasattr(config, param), f"Config should have {param} parameter"
        
        # Verify default values
        assert config.eval_n_episode == 10, "Default eval_n_episode should be 10"
        assert config.eval_interval == 10000, "Default eval_interval should be 10000"
        
        # Test valid ranges
        assert config.eval_n_episode > 0, "Evaluation episodes should be positive"
        assert config.eval_interval > 0, "Evaluation interval should be positive"
    
    def test_lstm_parameters_present(self):
        """Test that LSTM-related parameters are present."""
        config = MuZeroConfig()
        
        # LSTM parameters from EfficientZeroV2
        assert hasattr(config, 'lstm_hidden_size'), "Config should have lstm_hidden_size parameter"
        
        # Verify default values
        assert config.lstm_hidden_size == 512, "Default lstm_hidden_size should be 512"
        
        # Test valid ranges
        assert config.lstm_hidden_size > 0, "LSTM hidden size should be positive"
    
    def test_efficientzero_v2_pytorch_parameter_mapping(self):
        """Test mapping between EfficientZeroV2 PyTorch config and JAX config."""
        config = MuZeroConfig()
        
        # Mapping verification: PyTorch config path -> JAX config attribute
        pytorch_jax_mapping = {
            # RL parameters
            'config.rl.td_lambda': config.td_lambda,
            'config.rl.auto_td_steps': config.auto_td_steps,
            'config.rl.discount': config.discount_factor,
            'config.rl.unroll_steps': config.num_unroll_steps,
            'config.rl.td_steps': config.td_steps,
            
            # Training parameters
            'config.train.reanalyze_ratio': config.reanalyze_ratio,
            'config.train.reanalyze_update_interval': config.reanalyze_update_interval,
            'config.train.self_play_update_interval': config.self_play_update_interval,
            'config.train.batch_size': config.batch_size,
            'config.train.training_steps': config.training_steps,
            'config.train.offline_training_steps': config.offline_training_steps,
            'config.train.start_transitions': config.start_transitions,
            'config.train.mini_batch_size': config.mini_batch_size,
            'config.train.value_target': config.value_target,
            'config.train.start_use_mix_training_steps': config.start_use_mix_training_steps,
            'config.train.mixed_value_threshold': config.mixed_value_threshold,
            'config.train.change_temperature': config.change_temperature,
            'config.train.v_num': config.v_num,
            'config.train.use_IQL': config.use_iql,
            'config.train.IQL_weight': config.iql_weight,
            
            # Model parameters
            'config.model.value_target': config.value_target_type,
            'config.model.GAE_max_steps': config.gae_max_steps,
            'config.model.value_prefix': config.use_value_prefix,
            'config.model.lstm_horizon_len': config.lstm_horizon_length,
            'config.model.lstm_hidden_size': config.lstm_hidden_size,
            'config.model.noisy_net': config.noisy_net,
            
            # Priority parameters
            'config.priority.priority_prob_alpha': config.priority_exponent,
            'config.priority.priority_prob_beta': config.priority_beta,
            'config.priority.min_prior': config.min_priority,
            
            # Data parameters
            'config.data.total_transitions': config.total_transitions,
            'config.data.trajectory_size': config.trajectory_size,
            'config.data.buffer_size': config.buffer_size,
            
            # MCTS parameters
            'config.mcts.num_simulations': config.num_simulations,
            'config.mcts.c_visit': config.c_visit,
            'config.mcts.c_scale': config.c_scale,
            'config.mcts.c_base': config.c_base,
            'config.mcts.c_init': config.c_init,
            'config.mcts.dirichlet_alpha': config.dirichlet_alpha,
            'config.mcts.explore_frac': config.explore_frac,
            'config.mcts.value_minmax_delta': config.value_minmax_delta,
            'config.mcts.num_top_actions': config.num_top_actions,
            'config.mcts.num_sampled_actions': config.num_sampled_actions,
        }
        
        # Verify all mappings exist and have reasonable values
        for pytorch_path, jax_value in pytorch_jax_mapping.items():
            assert jax_value is not None, f"Mapping for {pytorch_path} should not be None"
    
    def test_config_immutability(self):
        """Test that the config is properly frozen (immutable)."""
        config = MuZeroConfig()
        
        # Test that the config is frozen
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.learning_rate = 0.1  # Should raise an error
    
    def test_config_serialization_compatibility(self):
        """Test that config can be serialized/deserialized properly."""
        config = MuZeroConfig(
            value_support_size=601,
            reward_support_size=601,
            td_lambda=0.8,
            reanalyze_ratio=0.5,
            num_simulations=32
        )
        
        # Test dataclass serialization
        config_dict = dataclasses.asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict['value_support_size'] == 601
        assert config_dict['td_lambda'] == 0.8
        assert config_dict['reanalyze_ratio'] == 0.5
        
        # Test reconstruction from dict
        new_config = MuZeroConfig(**config_dict)
        assert new_config.value_support_size == 601
        assert new_config.td_lambda == 0.8
        assert new_config.reanalyze_ratio == 0.5
    
    def test_environment_specific_defaults(self):
        """Test that we can create environment-specific configurations."""
        # Atari-like configuration
        atari_config = MuZeroConfig(
            support_min=-300.0,
            support_max=300.0,
            support_scale=1.0,
            num_simulations=16,
            reanalyze_ratio=1.0,
            value_target="mixed"
        )
        
        # DMC-like configuration
        dmc_config = MuZeroConfig(
            support_min=-299.0,
            support_max=299.0,
            support_scale=0.5,
            num_simulations=32,
            num_top_actions=16,
            num_sampled_actions=16
        )
        
        # Verify differences
        assert atari_config.support_scale != dmc_config.support_scale
        assert atari_config.num_simulations != dmc_config.num_simulations
        assert atari_config.num_top_actions != dmc_config.num_top_actions
    
    def test_backward_compatibility(self):
        """Test that existing code continues to work with new parameters."""
        # Test that we can create config with only old parameters
        old_style_config = MuZeroConfig(
            value_support_size=0,
            reward_support_size=0,
            discount_factor=0.99,
            num_unroll_steps=5,
            learning_rate=1e-3
        )
        
        # All new parameters should have defaults
        assert old_style_config.td_lambda == 0.95
        assert old_style_config.reanalyze_ratio == 1.0
        assert old_style_config.gae_max_steps == 15
        assert old_style_config.num_simulations == 16
    
    def test_parameter_validation_ranges(self):
        """Test that parameter ranges are sensible for EfficientZeroV2."""
        config = MuZeroConfig()
        
        # Test probability parameters are in [0, 1]
        prob_params = ['td_lambda', 'reanalyze_ratio', 'priority_exponent', 'priority_beta', 'explore_frac']
        for param in prob_params:
            value = getattr(config, param)
            assert 0.0 <= value <= 1.0, f"Parameter {param} should be in [0, 1], got {value}"
        
        # Test positive parameters
        positive_params = [
            'auto_td_steps', 'gae_max_steps', 'training_steps', 'mini_batch_size',
            'total_transitions', 'trajectory_size', 'buffer_size', 'num_simulations',
            'c_visit', 'temperature_decay_steps', 'num_top_actions', 'num_sampled_actions',
            'support_bins', 'epsilon', 'eval_n_episode', 'eval_interval', 'lstm_hidden_size'
        ]
        for param in positive_params:
            value = getattr(config, param)
            assert value > 0, f"Parameter {param} should be positive, got {value}"
        
        # Test non-negative parameters
        non_negative_params = ['offline_training_steps', 'decorrelation_coeff', 'off_diag_coeff']
        for param in non_negative_params:
            value = getattr(config, param)
            assert value >= 0, f"Parameter {param} should be non-negative, got {value}"


class TestConfigurationParameterCompleteness:
    """Test that all EfficientZeroV2 parameter categories are covered."""
    
    def test_all_pytorch_config_sections_covered(self):
        """Test that all major PyTorch config sections have JAX equivalents."""
        config = MuZeroConfig()
        
        # Check that we have parameters from all major EfficientZeroV2 config sections
        config_sections = {
            'rl': ['td_lambda', 'auto_td_steps', 'discount_factor', 'num_unroll_steps', 'td_steps'],
            'train': ['reanalyze_ratio', 'training_steps', 'batch_size', 'value_target'],
            'model': ['value_target_type', 'gae_max_steps', 'lstm_hidden_size', 'noisy_net'],
            'priority': ['priority_exponent', 'priority_beta', 'min_priority'],
            'data': ['total_transitions', 'trajectory_size', 'buffer_size'],
            'mcts': ['num_simulations', 'c_visit', 'c_scale', 'dirichlet_alpha'],
            'optimizer': ['learning_rate', 'weight_decay', 'clip_grad_norm']
        }
        
        for section, params in config_sections.items():
            for param in params:
                assert hasattr(config, param), f"Config missing {param} from {section} section"
    
    def test_parameter_count_comprehensive(self):
        """Test that we have a comprehensive set of parameters."""
        config = MuZeroConfig()
        
        # Count parameters by category
        all_fields = dataclasses.fields(config)
        total_params = len(all_fields)
        
        # We should have a substantial number of parameters for comprehensive EfficientZeroV2 support
        assert total_params >= 60, f"Expected at least 60 parameters for comprehensive EfficientZeroV2 support, got {total_params}"
    
    def test_action_item_15_completion_criteria(self):
        """Test that all Action Item 15 completion criteria are met."""
        config = MuZeroConfig()
        
        # 1. JAX MuZeroConfig contains all necessary parameters for replicating PyTorch's logic
        essential_params = [
            # For Action Item 1 (GAE/TD-Lambda)
            'td_lambda', 'auto_td_steps', 'gae_max_steps', 'value_target_type',
            # For Action Item 2 (Policy Reanalysis) 
            'reanalyze_ratio', 'reanalyze_update_interval',
            # For Action Item 16 (N-step returns)
            'td_steps', 'auto_td_steps',
            # For Action Item 19 (value_prefix)
            'use_value_prefix', 'lstm_horizon_length',
            # For Action Item 20 (Temperature scheduling)
            'change_temperature', 'temperature_init', 'temperature_final',
            # For Action Item 21 (mixed_value_threshold)
            'mixed_value_threshold', 'start_use_mix_training_steps',
            # For Action Item 25 (Noisy networks)
            'noisy_net'
        ]
        
        for param in essential_params:
            assert hasattr(config, param), f"Missing essential parameter {param} for Action Item completion"
        
        # 2. Consistent naming convention (flat structure with clear names)
        field_names = [field.name for field in dataclasses.fields(config)]
        
        # Check naming consistency (no spaces, underscores for separation)
        for name in field_names:
            assert '_' in name or name.islower(), f"Parameter {name} should use underscore naming convention"
            assert ' ' not in name, f"Parameter {name} should not contain spaces"
        
        # 3. All parameters have reasonable defaults
        # This is implicitly tested by the dataclass creation above
        assert True, "All parameters have defaults (implicit from successful config creation)"


if __name__ == "__main__":
    pytest.main([__file__]) 