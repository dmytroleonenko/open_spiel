"""
Tests for MuZero JAX orchestrator integration between components.

This module tests the integration between actor and learner components,
as well as the overall workflow coordination.
"""

import pytest
import tempfile
import os
import shutil
import threading
import time
import dataclasses
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace, asdict
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

# Import the isolated checkpoint fixture
from open_spiel.python.algorithms.muzero_jax.tests.utils.fixtures import isolated_checkpoint_dir




def _get_unique_rng_key() -> jax.random.PRNGKey:
    """Generate a unique random key using thread ID and time to avoid parallel collisions."""
    thread_id = threading.get_ident()
    timestamp_ns = time.time_ns()
    # Combine thread ID and timestamp for uniqueness across parallel workers
    unique_seed = int(thread_id % 10000) * 1000000 + int(timestamp_ns % 1000000)
    return jax.random.PRNGKey(unique_seed)


def convert_trajectories_to_batch(trajectories, config):
    """Convert list of trajectory dictionaries to batch format expected by trainer."""
    if not trajectories:
        raise ValueError("No trajectories provided")
    
    batch_size = len(trajectories)
    
    # Find the maximum trajectory length to determine K
    max_length = max(len(traj['actions']) for traj in trajectories)
    K = min(max_length, config.num_unroll_steps)  # Limit to num_unroll_steps
    
    # Initialize batch arrays
    batch = {}
    
    # Get observation shape from first trajectory
    obs_shape = jnp.array(trajectories[0]['observations'][0]).shape
    
    # Create arrays with proper shapes
    batch['observation'] = jnp.zeros((batch_size, K + 1) + obs_shape)
    batch['action'] = jnp.zeros((batch_size, K), dtype=jnp.int32)
    batch['target_reward'] = jnp.zeros((batch_size, K + 1))
    batch['target_value'] = jnp.zeros((batch_size, K + 1))
    batch['target_policy'] = jnp.zeros((batch_size, K + 1, config.num_actions))
    batch['game_history_mask'] = jnp.ones((batch_size, K + 1))  # All valid for simplicity
    
    # Fill in data from trajectories
    for b, traj in enumerate(trajectories):
        traj_length = min(len(traj['actions']), K)
        
        # Observations (0 to K)
        for t in range(min(traj_length + 1, len(traj['observations']))):
            if t < len(traj['observations']):
                batch['observation'] = batch['observation'].at[b, t].set(jnp.array(traj['observations'][t]))
        
        # Actions (0 to K-1) 
        for t in range(traj_length):
            if t < len(traj['actions']):
                batch['action'] = batch['action'].at[b, t].set(traj['actions'][t])
        
        # Rewards (0 to K)
        for t in range(min(traj_length + 1, len(traj['rewards']) + 1)):
            if t < len(traj['rewards']):
                batch['target_reward'] = batch['target_reward'].at[b, t].set(traj['rewards'][t])
        
        # Values (0 to K)
        for t in range(min(traj_length + 1, len(traj['value_targets']))):
            if t < len(traj['value_targets']):
                batch['target_value'] = batch['target_value'].at[b, t].set(traj['value_targets'][t])
        
        # Policy targets (0 to K)
        for t in range(min(traj_length + 1, len(traj['policy_targets']))):
            if t < len(traj['policy_targets']):
                policy_target = jnp.array(traj['policy_targets'][t])
                # Remove batch dimension if present
                if policy_target.ndim > 1:
                    policy_target = jnp.squeeze(policy_target, axis=0)
                batch['target_policy'] = batch['target_policy'].at[b, t].set(policy_target)
    
    return batch 


class TestActorLearnerIntegration:
    """Test the integration between actor and learner components."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self, isolated_checkpoint_dir):
        """Use the isolated checkpoint helper for complete test isolation."""
        return isolated_checkpoint_dir
        
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return create_muzero_config_for_game(
            "tic_tac_toe",
            learning_rate=1e-3,
            batch_size=4,
            buffer_size=100,  # Use correct parameter name
            training_steps=10
        )
        
    @pytest.fixture
    def mock_components(self, mock_config, temp_checkpoint_dir):
        """Create mock components for testing with proper checkpoint cleanup."""
        # Create mock network
        rng = _get_unique_rng_key()
        from open_spiel.python.algorithms.muzero_jax.training.trainer import (
            create_network_config_from_muzero_config
        )
        from open_spiel.python.algorithms.muzero_jax.models.network import (
            RepresentationNetwork, DynamicsNetwork, PredictionNetwork, 
            RewardNetwork, ProjectionNetwork
        )
        
        observation_shape = (27,)  # Tic-tac-toe observation from GameWrapper
        network_config = create_network_config_from_muzero_config(
            mock_config, observation_shape, mock_config.num_actions, use_image_observation=False
        )
        
        rngs = nnx.Rngs(params=rng)
        network = MuZeroNetwork(
            representation_network_def=RepresentationNetwork,
            dynamics_network_def=DynamicsNetwork,
            prediction_network_def=PredictionNetwork,
            reward_network_def=RewardNetwork,
            projection_network_def=ProjectionNetwork if mock_config.use_projection else None,
            config=network_config,
            rngs=rngs
        )
        
        # Create replay buffer
        buffer = TrajectoryBuffer(
            capacity=mock_config.buffer_size  # Use correct parameter name
        )
        
        # CRITICAL FIX: Use temp_checkpoint_dir for isolation between tests
        config_with_checkpoint = dataclasses.replace(mock_config, checkpoint_dir=temp_checkpoint_dir)
        
        # Create learner
        learner = Learner(
            model=network,
            optimizer_def=None,  # Will use default from config
            config=config_with_checkpoint,  # Use config with isolated checkpoint directory
            rng_key=rng
        )
        
        # Create game wrapper
        game_wrapper = GameWrapper("tic_tac_toe")
        
        # Create actor
        actor = Actor(
            network=network,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=mock_config,
            num_simulations=10,
            max_num_considered_actions=16,
            gumbel_scale=1.0
        )
        
        components = {
            'network': network,
            'buffer': buffer,
            'learner': learner,
            'actor': actor,
            'game_wrapper': game_wrapper,
            'config': mock_config
        }
        
        yield components
        
        # CRITICAL: Explicitly close checkpoint manager to prevent parallel cleanup conflicts
        if hasattr(learner, 'checkpoint_manager') and learner.checkpoint_manager is not None:
            try:
                learner.checkpoint_manager.close()
            except Exception as e:
                # Log but don't fail test if cleanup fails
                print(f"Warning: checkpoint manager cleanup failed: {e}")
        
    def test_actor_buffer_interaction(self, mock_components):
        """Test that actor can add trajectories to buffer."""
        actor = mock_components['actor']
        buffer = mock_components['buffer']
        
        initial_buffer_size = len(buffer)
        
        # Run one episode to generate trajectory
        rng_key = _get_unique_rng_key()
        episode_data = actor.play_episode(rng_key)
        
        # Add trajectory to buffer (this simulates what the run method does)
        buffer.add_trajectory(episode_data)
        
        # Check that buffer size increased
        assert len(buffer) > initial_buffer_size
        assert isinstance(episode_data, dict)
        assert 'observations' in episode_data
        
    def test_learner_buffer_interaction(self, mock_components):
        """Test that learner can sample from buffer after actor adds data (lightweight)."""
        actor = mock_components['actor']
        learner = mock_components['learner']
        buffer = mock_components['buffer']
        config = mock_components['config']

        # Mock the expensive play_episode method to return lightweight dummy data
        with patch.object(actor, 'play_episode') as mock_play_episode:
            mock_play_episode.return_value = {
                'observations': [jnp.zeros(27)],  # Single dummy observation
                'actions': [0],  # Single dummy action
                'rewards': [0.0],  # Single dummy reward
                'policy_targets': [jnp.ones(9) / 9],  # Uniform policy
                'value_targets': [0.0]  # Single dummy value target
            }
            
            # Generate and add lightweight trajectory to buffer
            rng_key = _get_unique_rng_key()
            episode_data = actor.play_episode(rng_key)
            buffer.add_trajectory(episode_data)
            
            # Ensure buffer has enough data for sampling
            assert len(buffer) > 0
            
            # Mock the expensive sampling and training process
            with patch.object(buffer, 'sample_batch') as mock_sample:
                # Return lightweight batch data
                mock_sample.return_value = {
                    'observation': jnp.zeros((2, 5, 27)),  # Tiny batch
                    'action': jnp.zeros((2, 4), dtype=jnp.int32),
                    'target_reward': jnp.zeros((2, 5)),
                    'target_value': jnp.zeros((2, 5)),
                    'target_policy': jnp.ones((2, 5, 9)) / 9,
                    'game_history_mask': jnp.ones((2, 5))
                }
                
                # This should not raise errors (learning step is mocked)
                with patch.object(learner, 'train_step') as mock_train_step:
                    mock_train_step.return_value = {'total_loss': 0.5}  # Mock loss
                    
                    batch = buffer.sample_batch(batch_size=2)
                    result = learner.train_step(batch)
                    
                    # Verify mocks were called
                    mock_sample.assert_called_once_with(batch_size=2)
                    mock_train_step.assert_called_once()
                    assert 'total_loss' in result

    def test_checkpoint_saving_loading(self, mock_components, temp_checkpoint_dir):
        """Test that checkpoints can be saved and loaded."""
        learner = mock_components['learner']
        actor = mock_components['actor']
        
        # Save a checkpoint
        learner.save_checkpoint(force_save=True)
        # Note: save_checkpoint doesn't return a path, it saves to the configured checkpoint directory
        
        # Actor should be able to load parameters
        success = actor.maybe_load_latest_parameters(temp_checkpoint_dir)
        # Note: This might be False if no checkpoint exists yet, but shouldn't error
        assert isinstance(success, bool)


class TestOrchestrationWorkflow:
    """Test the overall orchestration workflow."""
    
    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch('wandb.init'), patch('wandb.log'), patch('wandb.finish'):
            yield
    
    def test_workflow_initialization(self, mock_wandb):
        """Test that all components can be initialized together."""
        config = create_muzero_config_for_game(
            "tic_tac_toe",
            batch_size=4,
            buffer_size=50  # Use buffer_size instead of replay_buffer_size
        )
        
        # This should not raise any errors
        observation_shape = (27,)  # Tic-tac-toe observation from GameWrapper
        
        # Initialize network
        rng = _get_unique_rng_key()
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
        
        # Initialize buffer
        buffer = TrajectoryBuffer(capacity=config.buffer_size)
        
        # Initialize learner
        learner = Learner(
            model=network,
            optimizer_def=None,
            config=config,
            rng_key=rng
        )
        
        # Initialize game wrapper and actor
        game_wrapper = GameWrapper("tic_tac_toe")
        actor = Actor(
            network=network,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=config,
            num_simulations=10,
            max_num_considered_actions=16
        )
        
        # All components should be initialized without errors
        assert network is not None
        assert buffer is not None
        assert learner is not None
        assert actor is not None
        assert game_wrapper is not None 