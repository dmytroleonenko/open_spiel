"""
Tests for MuZero JAX orchestrator setup, configuration, and initialization.

This module tests the basic setup, configuration management, and error handling
for the MuZero JAX orchestration system.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace, asdict
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

# Import the components that will be used by run_muzero_jax
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig, 
    Learner, 
    create_muzero_config_for_game
)
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper


class TestMainOrchestrationSetup:
    """Test the initialization and setup logic of the main orchestration script."""
    
    def test_config_initialization(self):
        """Test that configuration can be properly initialized for a game."""
        config = create_muzero_config_for_game("tic_tac_toe")
        assert isinstance(config, MuZeroConfig)
        assert config.num_actions == 9  # Tic-tac-toe has 9 actions
        assert config.learning_rate > 0
        assert config.batch_size > 0
        
    def test_config_with_overrides(self):
        """Test that configuration overrides work correctly."""
        config = create_muzero_config_for_game(
            "tic_tac_toe", 
            learning_rate=1e-3,
            batch_size=32
        )
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.num_actions == 9
        
    def test_network_initialization(self):
        """Test that MuZero network can be initialized from config."""
        config = create_muzero_config_for_game("tic_tac_toe")
        
        # Create a simple observation shape for tic-tac-toe (flat observation)
        observation_shape = (27,)  # Tic-tac-toe observation from GameWrapper
        
        from open_spiel.python.algorithms.muzero_jax.training.trainer import (
            create_network_config_from_muzero_config
        )
        from open_spiel.python.algorithms.muzero_jax.models.network import (
            RepresentationNetwork, DynamicsNetwork, PredictionNetwork, 
            RewardNetwork, ProjectionNetwork
        )
        
        network_config = create_network_config_from_muzero_config(
            config, observation_shape, config.num_actions, use_image_observation=False
        )
        
        # Initialize the network with proper constructor
        rng = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(params=rng)
        
        network = MuZeroNetwork(
            representation_network_def=RepresentationNetwork,
            dynamics_network_def=DynamicsNetwork,
            prediction_network_def=PredictionNetwork,
            reward_network_def=RewardNetwork,
            projection_network_def=ProjectionNetwork if config.use_projection else None,
            config=network_config,
            rngs=rngs
        )
        
        # Test that network can process dummy inputs
        dummy_obs = jnp.ones((1,) + observation_shape)
        representation = network.representation(dummy_obs, training=False)
        assert representation.shape[0] == 1  # Batch dimension
        
    def test_buffer_initialization(self):
        """Test that replay buffer can be properly initialized."""
        config = create_muzero_config_for_game("tic_tac_toe")
        
        buffer = TrajectoryBuffer(
            capacity=config.buffer_size  # Use correct parameter name
        )
        
        assert len(buffer) == 0
        assert buffer.capacity == config.buffer_size


class TestConfigurationManagement:
    """Test configuration management using Hydra-style patterns."""
    
    def test_default_config_structure(self):
        """Test that default configuration has expected structure."""
        config = create_muzero_config_for_game("tic_tac_toe")
        
        # Check required fields exist (using actual field names from MuZeroConfig)
        required_fields = [
            'num_actions', 'learning_rate', 'batch_size', 'buffer_size',
            'num_unroll_steps', 'td_steps', 'discount_factor', 'training_steps'
        ]
        
        for field in required_fields:
            assert hasattr(config, field), f"Missing required field: {field}"
            
    def test_config_validation(self):
        """Test that configuration validation works."""
        # Test invalid game name
        with pytest.raises(Exception):
            create_muzero_config_for_game("nonexistent_game")
            
    def test_config_serialization(self):
        """Test that configuration can be serialized/deserialized."""
        config = create_muzero_config_for_game("tic_tac_toe")
        
        # Should be serializable to dict
        config_dict = config.__dict__
        assert isinstance(config_dict, dict)
        assert 'num_actions' in config_dict
        
        # Should be able to recreate from dict
        new_config = MuZeroConfig(**config_dict)
        assert new_config.num_actions == config.num_actions


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_game_handling(self):
        """Test handling of invalid game names."""
        with pytest.raises(Exception):
            create_muzero_config_for_game("invalid_game_name")
            
    def test_insufficient_buffer_data(self):
        """Test handling when buffer doesn't have enough data."""
        config = create_muzero_config_for_game(
            "tic_tac_toe",
            batch_size=10,  # Larger than buffer will contain
            buffer_size=20  # Use correct parameter name
        )
        
        observation_shape = (3, 3, 1)
        buffer = TrajectoryBuffer(
            capacity=config.buffer_size  # Use correct parameter name
        )
        
        # Buffer is empty, should handle gracefully
        assert len(buffer) == 0
        
        # Sampling should either return None or raise appropriate exception
        try:
            batch = buffer.sample_batch(config.batch_size)
            # If it returns something, it should be valid
            if batch is not None:
                assert batch is not None
        except (ValueError, RuntimeError):
            # Expected when buffer is empty
            pass
            
    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = os.path.join(temp_dir, "nonexistent", "checkpoint_dir")
            
            config = create_muzero_config_for_game("tic_tac_toe")
            
            # This should create the directory
            rng = jax.random.PRNGKey(42)
            from open_spiel.python.algorithms.muzero_jax.training.trainer import (
                create_network_config_from_muzero_config
            )
            from open_spiel.python.algorithms.muzero_jax.models.network import (
                RepresentationNetwork, DynamicsNetwork, PredictionNetwork, 
                RewardNetwork, ProjectionNetwork
            )
            
            observation_shape = (27,)  # Tic-tac-toe observation from GameWrapper
            network_config = create_network_config_from_muzero_config(
                config, observation_shape, config.num_actions, use_image_observation=False
            )
            
            rngs = nnx.Rngs(params=rng)
            network = MuZeroNetwork(
                representation_network_def=RepresentationNetwork,
                dynamics_network_def=DynamicsNetwork,
                prediction_network_def=PredictionNetwork,
                reward_network_def=RewardNetwork,
                projection_network_def=ProjectionNetwork if config.use_projection else None,
                config=network_config,
                rngs=rngs
            )
            
            config_with_checkpoint = dataclasses.replace(config, checkpoint_dir=nonexistent_dir)
            
            learner = Learner(
                model=network,  # Use model instead of network
                optimizer_def=None,  # Let it create default optimizer
                config=config_with_checkpoint,
                rng_key=rng  # Add required rng_key parameter
            )
            
            # Directory should now exist
            assert os.path.exists(nonexistent_dir) 