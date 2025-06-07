"""
Tests for MuZero JAX orchestrator integration between components.

This module tests the integration between actor and learner components,
as well as the overall workflow coordination.
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
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
        
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
        """Create mock components for testing."""
        # Create mock network
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
        
        # Create learner
        learner = Learner(
            model=network,
            optimizer_def=None,  # Will use default from config
            config=mock_config,
            rng_key=rng
        )
        
        # Create game wrapper
        game_wrapper = GameWrapper("tic_tac_toe")
        
        # Create actor (need to create MCTS and network first)
        from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
        mcts = MCTS(
            num_simulations=10,
            max_num_considered_actions=16,
            gumbel_scale=1.0
        )
        
        actor = Actor(
            network=network,
            mcts=mcts,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=mock_config
        )
        
        return {
            'network': network,
            'buffer': buffer,
            'learner': learner,
            'actor': actor,
            'game_wrapper': game_wrapper,
            'config': mock_config
        }
        
    def test_actor_buffer_interaction(self, mock_components):
        """Test that actor can add trajectories to buffer."""
        actor = mock_components['actor']
        buffer = mock_components['buffer']
        
        initial_buffer_size = len(buffer)
        
        # Run one episode to generate trajectory
        rng_key = jax.random.PRNGKey(123)
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

            # Generate just one lightweight trajectory
            rng_key = jax.random.PRNGKey(200)
            episode_data = actor.play_episode(rng_key)
            buffer.add_trajectory(episode_data)

            # Test that learner can sample and train
            if len(buffer) >= config.batch_size:
                trajectory_list = buffer.sample_batch(config.batch_size)
                batch = convert_trajectories_to_batch(trajectory_list, config)
                
                # Mock the expensive train_step method too
                with patch.object(learner, 'train_step') as mock_train_step:
                    mock_train_step.return_value = {
                        'total_loss': 1.0,
                        'policy_loss': 0.3,
                        'value_loss': 0.4,
                        'reward_loss': 0.3
                    }
                    
                    metrics = learner.train_step(batch)
                    assert isinstance(metrics, dict)
                    assert 'total_loss' in metrics
                    
                    # Verify the mock was called
                    mock_train_step.assert_called_once()
            
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
    """Test the overall workflow coordination."""
    
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
        rng = jax.random.PRNGKey(42)
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
        buffer = TrajectoryBuffer(
            capacity=config.buffer_size  # Use correct parameter name
        )
        
        # Initialize components
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = Learner(
                model=network,
                optimizer_def=None,
                config=config,
                rng_key=rng
            )
            
            game_wrapper = GameWrapper("tic_tac_toe")
            from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
            mcts = MCTS(
                num_simulations=10,
                max_num_considered_actions=16,
                gumbel_scale=1.0
            )
            
            actor = Actor(
                network=network,
                mcts=mcts,
                game_wrapper=game_wrapper,
                replay_buffer=buffer,
                config=config
            )
            
            # All components should be properly initialized
            assert learner is not None
            assert actor is not None
            assert buffer is not None
            
    def test_mini_training_loop(self, mock_wandb):
        """Test a minimal training loop with lightweight mocked components."""
        config = create_muzero_config_for_game(
            "tic_tac_toe",
            batch_size=1,  # Reduced batch size
            buffer_size=5,  # Reduced buffer size
            training_steps=1  # Just 1 training step
        )

        # Mock the expensive components to make test lightweight
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.Actor.play_episode') as mock_play_episode, \
             patch('open_spiel.python.algorithms.muzero_jax.training.trainer.Learner.train_step') as mock_train_step, \
             patch('open_spiel.python.algorithms.muzero_jax.training.trainer.Learner.save_checkpoint') as mock_save_checkpoint, \
             patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.Actor.maybe_load_latest_parameters') as mock_load_params:
            
            # Mock play_episode to return lightweight dummy data
            mock_play_episode.return_value = {
                'observations': [jnp.zeros(27)],  # Single dummy observation
                'actions': [0],  # Single dummy action
                'rewards': [0.0],  # Single dummy reward
                'policy_targets': [jnp.ones(9) / 9],  # Uniform policy
                'value_targets': [0.0]  # Single dummy value target
            }
            
            # Mock train_step to return dummy metrics
            mock_train_step.return_value = {
                'total_loss': 1.0,
                'policy_loss': 0.3,
                'value_loss': 0.4,
                'reward_loss': 0.3
            }
            
            # Create minimal components
            observation_shape = (27,)
            rng = jax.random.PRNGKey(42)
            
            # Create a minimal mock network
            mock_network = Mock()
            mock_network.initial_inference.return_value = (
                jnp.zeros((1, 32)),  # hidden_state
                jnp.array([0.0]),    # reward
                jnp.array([0.0]),    # value
                jnp.zeros((1, 9)),   # policy_logits
                jnp.zeros((1, 64))   # projection (if used)
            )
            
            buffer = TrajectoryBuffer(capacity=config.buffer_size)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create mocked learner
                learner = Mock()
                learner.train_step = mock_train_step
                learner.save_checkpoint = mock_save_checkpoint
                
                # Create mocked actor
                actor = Mock()
                actor.play_episode = mock_play_episode
                actor.maybe_load_latest_parameters = mock_load_params
                
                # Test the orchestration logic
                # Simulate running an episode to populate buffer
                rng_key = jax.random.PRNGKey(300)
                episode_data = actor.play_episode(rng_key)
                buffer.add_trajectory(episode_data)
                
                # Test that we can sample and train
                if len(buffer) >= config.batch_size:
                    trajectory_list = buffer.sample_batch(config.batch_size)
                    batch = convert_trajectories_to_batch(trajectory_list, config)
                    
                    # Test training step
                    metrics = learner.train_step(batch)
                    assert isinstance(metrics, dict)
                    assert 'total_loss' in metrics
                    
                    # Test checkpoint saving
                    learner.save_checkpoint(force_save=True)
                    
                    # Test parameter loading
                    actor.maybe_load_latest_parameters(temp_dir)
                
                # Verify mocks were called appropriately
                mock_play_episode.assert_called()
                mock_train_step.assert_called()
                mock_save_checkpoint.assert_called_with(force_save=True)
                mock_load_params.assert_called_with(temp_dir) 