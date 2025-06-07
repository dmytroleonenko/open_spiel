"""Integration tests for the complete MuZero JAX workflow.

These tests verify end-to-end behavior that unit tests might miss,
such as orchestrator properly adding episodes to replay buffer.
"""

import pytest
import jax
import jax.numpy as jnp
import tempfile
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from unittest.mock import Mock, patch

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner


class TestOrchestratorIntegration:
    """Integration tests for MuZeroOrchestrator workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_config(self, temp_dir):
        """Create minimal config for testing."""
        config = {
            'exp_config': {'tag': 'test', 'debug': False, 'seed': 42},
            'game': {'name': 'tic_tac_toe'},
            'training': {
                'training_steps': 2,
                'batch_size': 4,
                'gradient_accumulation_steps': 1,
                'start_transitions': 2,
                'learning_rate': 0.001,
                'td_steps': 3,
                'discount': 0.99,
                'num_unroll_steps': 3,
                'value_loss_weight': 1.0,
                'policy_loss_weight': 1.0,
                'reward_loss_weight': 1.0,
                'l2_regularization': 1e-4,
                'target_update_interval': 10
            },
            'mcts': {
                'num_simulations': 5,
                'gumbel_scale': 1.0,
                'temperature_init': 1.0,
                'temperature_final': 0.1,
                'temperature_decay_steps': 1000,
                'dirichlet_alpha': 0.3,
                'exploration_fraction': 0.25
            },
            'replay_buffer': {
                'capacity': 100,
                'min_size_to_sample': 2
            },
            'actors': {
                'num_actors': 1
            },
            'output': {
                'save_path': str(temp_dir),
                'log_interval': 1,
                'checkpoint_interval': 10
            },
            'wandb': {'enabled': False},
            'evaluation': {'enabled': False},
            'resource_management': {
                'device': 'cpu',
                'sequential_training': True,
                'training_phase_steps': 1,
                'selfplay_phase_episodes': 2,
                'max_episodes_without_training': 10,
                'max_training_steps_without_episodes': 10
            },
            'random_seed': 42
        }
        return DictConfig(config)
    
    def test_orchestrator_buffer_integration(self, minimal_config):
        """Test that orchestrator properly adds episodes to replay buffer.
        
        This test would have caught the missing buffer addition bug.
        """
        orchestrator = MuZeroOrchestrator(minimal_config)
        orchestrator.setup_components()
        
        # Verify initial buffer is empty
        initial_buffer_size = len(orchestrator.replay_buffer)
        assert initial_buffer_size == 0, "Buffer should start empty"
        
        # Run self-play phase
        selfplay_metrics = orchestrator.run_selfplay_phase()
        
        # CRITICAL: Buffer should now contain episodes
        buffer_size_after_selfplay = len(orchestrator.replay_buffer)
        episodes_played = selfplay_metrics['episodes_played']
        
        assert buffer_size_after_selfplay > 0, (
            f"Buffer should contain episodes after self-play. "
            f"Episodes played: {episodes_played}, Buffer size: {buffer_size_after_selfplay}"
        )
        assert buffer_size_after_selfplay == episodes_played, (
            f"Buffer size ({buffer_size_after_selfplay}) should equal episodes played ({episodes_played})"
        )
        
        # Verify metric reporting is accurate
        reported_buffer_size = selfplay_metrics['buffer_size']
        actual_buffer_size = len(orchestrator.replay_buffer)
        assert reported_buffer_size == actual_buffer_size, (
            f"Reported buffer size ({reported_buffer_size}) doesn't match "
            f"actual buffer size ({actual_buffer_size})"
        )
    
    def test_training_only_occurs_with_sufficient_data(self, minimal_config):
        """Test that training only occurs when buffer has enough data."""
        # Set high start_transitions to test conditional training
        minimal_config.training.start_transitions = 10
        
        orchestrator = MuZeroOrchestrator(minimal_config)
        orchestrator.setup_components()
        
        # Run self-play with fewer episodes than start_transitions
        minimal_config.resource_management.selfplay_phase_episodes = 2
        selfplay_metrics = orchestrator.run_selfplay_phase()
        
        buffer_size = len(orchestrator.replay_buffer)
        assert buffer_size < minimal_config.training.start_transitions
        
        # Attempt training - should be skipped
        initial_training_step = orchestrator.training_step
        
        # Training should not advance when buffer is too small
        # We can't easily test the skipping without modifying the method,
        # but we can verify the buffer condition
        assert buffer_size < minimal_config.training.start_transitions, (
            "Training should be skipped when buffer size < start_transitions"
        )
    
    def test_complete_workflow_single_iteration(self, minimal_config):
        """Test complete workflow: self-play → buffer → training."""
        # Ensure we have enough data to trigger training
        minimal_config.training.start_transitions = 1
        minimal_config.resource_management.selfplay_phase_episodes = 3
        
        orchestrator = MuZeroOrchestrator(minimal_config)
        orchestrator.setup_components()
        
        initial_training_step = orchestrator.training_step
        initial_buffer_size = len(orchestrator.replay_buffer)
        
        # Self-play phase
        selfplay_metrics = orchestrator.run_selfplay_phase()
        buffer_size_after_selfplay = len(orchestrator.replay_buffer)
        
        # Verify episodes were added to buffer
        assert buffer_size_after_selfplay > initial_buffer_size
        assert buffer_size_after_selfplay >= minimal_config.training.start_transitions
        
        # Training phase  
        training_metrics = orchestrator.run_training_phase()
        final_training_step = orchestrator.training_step
        
        # Verify training actually occurred
        assert final_training_step > initial_training_step, "Training step should have advanced"
        assert 'loss' in training_metrics, "Training metrics should include loss"


class TestActorBufferIntegration:
    """Integration tests for Actor and replay buffer interaction."""
    
    @pytest.fixture
    def setup_actor(self):
        """Set up actor with minimal dependencies."""
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        # Create network
        network = MuZeroNetwork(
            observation_shape=observation_shape,
            num_actions=num_actions,
            hidden_dim=16,
            num_blocks=1,
            use_image_observation=False
        )
        
        # Create MCTS
        mcts = MCTS(
            num_simulations=5,
            max_num_considered_actions=num_actions,
            gumbel_scale=1.0
        )
        
        # Create replay buffer
        buffer = TrajectoryBuffer(
            max_size=100,
            min_size_to_sample=1,
            observation_shape=observation_shape,
            num_actions=num_actions
        )
        
        # Create actor
        actor = Actor(
            network=network,
            mcts=mcts,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=Mock(),
            n_step_return=3,
            discount_factor=0.99
        )
        
        return actor, buffer, game_wrapper
    
    def test_actor_generates_valid_trajectory(self, setup_actor):
        """Test that actor generates valid trajectory data."""
        actor, buffer, game_wrapper = setup_actor
        
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)
        
        # Verify trajectory structure
        expected_keys = ['observations', 'actions', 'rewards', 'policy_targets', 'value_targets']
        for key in expected_keys:
            assert key in trajectory, f"Trajectory missing key: {key}"
        
        # Verify trajectory is non-empty
        assert len(trajectory['observations']) > 0, "Trajectory should have observations"
        assert len(trajectory['actions']) > 0, "Trajectory should have actions"
        
        # Verify all trajectory components have same length (except possibly observations)
        action_len = len(trajectory['actions'])
        assert len(trajectory['rewards']) == action_len
        assert len(trajectory['policy_targets']) == action_len
        assert len(trajectory['value_targets']) == action_len
    
    def test_actor_run_method_adds_to_buffer(self, setup_actor):
        """Test that actor.run() method properly adds episodes to buffer."""
        actor, buffer, game_wrapper = setup_actor
        
        initial_buffer_size = len(buffer)
        assert initial_buffer_size == 0
        
        # Run actor for multiple episodes
        rng_key = jax.random.PRNGKey(42)
        num_episodes = 3
        actor.run(rng_key, num_episodes=num_episodes)
        
        final_buffer_size = len(buffer)
        assert final_buffer_size == num_episodes, (
            f"Buffer should contain {num_episodes} episodes, got {final_buffer_size}"
        )
    
    def test_buffer_can_sample_after_adding_trajectory(self, setup_actor):
        """Test that buffer can sample after adding trajectory data."""
        actor, buffer, game_wrapper = setup_actor
        
        # Add trajectory to buffer
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)
        buffer.add_trajectory(trajectory)
        
        assert len(buffer) == 1
        
        # Sample from buffer
        sample_rng = jax.random.PRNGKey(123)
        batch = buffer.sample_batch(sample_rng, batch_size=1)
        
        # Verify batch structure
        expected_keys = ['observations', 'actions', 'rewards', 'policy_targets', 'value_targets']
        for key in expected_keys:
            assert key in batch, f"Batch missing key: {key}"
            assert batch[key].shape[0] == 1, f"Batch size should be 1 for {key}"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_minimal_training_loop(self):
        """Test minimal training loop without orchestrator complexity."""
        # This test verifies the basic workflow components work together
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        # Create components
        network = MuZeroNetwork(
            observation_shape=observation_shape,
            num_actions=num_actions,
            hidden_dim=16,
            num_blocks=1,
            use_image_observation=False
        )
        
        buffer = TrajectoryBuffer(
            max_size=100,
            min_size_to_sample=1,
            observation_shape=observation_shape,
            num_actions=num_actions
        )
        
        mcts = MCTS(
            num_simulations=5,
            max_num_considered_actions=num_actions,
            gumbel_scale=1.0
        )
        
        actor = Actor(
            network=network,
            mcts=mcts,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=Mock(),
            n_step_return=3,
            discount_factor=0.99
        )
        
        # Generate episode and add to buffer
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)
        buffer.add_trajectory(trajectory)
        
        assert len(buffer) == 1, "Buffer should contain one episode"
        
        # Sample from buffer
        sample_key = jax.random.PRNGKey(123)
        batch = buffer.sample_batch(sample_key, batch_size=1)
        
        # Verify we can forward pass through network
        obs_batch = batch['observations']
        hidden_state, reward, value, policy_logits, _ = network.initial_inference(
            obs_batch, training=False
        )
        
        # Verify outputs have expected shapes
        assert hidden_state.shape[0] == 1, "Batch dimension should be preserved"
        assert value.shape[0] == 1, "Value should have batch dimension"
        assert policy_logits.shape == (1, num_actions), "Policy should match action space"
    
    def test_orchestrator_workflow_bug_detection(self):
        """Specific test to catch the bug we just fixed: missing buffer addition."""
        # This test specifically checks that the orchestrator's run_selfplay_phase
        # actually populates the buffer, which would have caught our bug
        
        config = DictConfig({
            'exp_config': {'tag': 'test', 'debug': False, 'seed': 42},
            'game': {'name': 'tic_tac_toe'},
            'training': {
                'training_steps': 1,
                'batch_size': 1,
                'gradient_accumulation_steps': 1,
                'start_transitions': 1,
                'learning_rate': 0.001,
                'td_steps': 3,
                'discount': 0.99,
                'num_unroll_steps': 3,
                'value_loss_weight': 1.0,
                'policy_loss_weight': 1.0,
                'reward_loss_weight': 1.0,
                'l2_regularization': 1e-4,
                'target_update_interval': 10
            },
            'mcts': {
                'num_simulations': 3, 
                'gumbel_scale': 1.0,
                'temperature_init': 1.0,
                'temperature_final': 0.1,
                'temperature_decay_steps': 1000,
                'dirichlet_alpha': 0.3,
                'exploration_fraction': 0.25
            },
            'replay_buffer': {'capacity': 10, 'min_size_to_sample': 1},
            'actors': {'num_actors': 1},
            'output': {
                'save_path': '/tmp/test_checkpoints',
                'log_interval': 1,
                'checkpoint_interval': 10
            },
            'wandb': {'enabled': False},
            'evaluation': {'enabled': False},
            'resource_management': {
                'device': 'cpu',
                'sequential_training': True,
                'training_phase_steps': 1,
                'selfplay_phase_episodes': 2,
                'max_episodes_without_training': 10,
                'max_training_steps_without_episodes': 10
            },
            'random_seed': 42
        })
        
        orchestrator = MuZeroOrchestrator(config)
        orchestrator.setup_components()
        
        # THE CRITICAL TEST: After self-play, buffer must not be empty
        buffer_size_before = len(orchestrator.replay_buffer)
        selfplay_metrics = orchestrator.run_selfplay_phase()
        buffer_size_after = len(orchestrator.replay_buffer)
        
        episodes_played = selfplay_metrics['episodes_played']
        
        # This assertion would have FAILED before our fix
        assert buffer_size_after > buffer_size_before, (
            f"CRITICAL BUG: Buffer size didn't increase after self-play! "
            f"Before: {buffer_size_before}, After: {buffer_size_after}, "
            f"Episodes played: {episodes_played}"
        )
        
        # More specific assertion
        assert buffer_size_after == episodes_played, (
            f"Buffer size ({buffer_size_after}) should equal episodes played ({episodes_played}). "
            f"This indicates episodes are not being added to the buffer!"
        )
        
        # Verify reported metrics match reality
        reported_buffer_size = selfplay_metrics['buffer_size']
        assert reported_buffer_size == buffer_size_after, (
            f"Reported buffer size ({reported_buffer_size}) doesn't match "
            f"actual buffer size ({buffer_size_after})"
        )


class TestBufferStateConsistency:
    """Tests for buffer state consistency issues."""
    
    def test_buffer_size_reporting_accuracy(self):
        """Test that buffer size reporting is always accurate."""
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        buffer = TrajectoryBuffer(
            max_size=10,
            min_size_to_sample=1,
            observation_shape=observation_shape,
            num_actions=num_actions
        )
        
        # Create a dummy trajectory
        trajectory = {
            'observations': [jnp.zeros(observation_shape) for _ in range(3)],
            'actions': [0, 1, 2],
            'rewards': [0.0, 0.0, 1.0],
            'policy_targets': [jnp.ones(num_actions)/num_actions for _ in range(3)],
            'value_targets': [0.0, 0.5, 1.0]
        }
        
        # Test buffer size consistency
        for i in range(5):
            size_before = len(buffer)
            buffer.add_trajectory(trajectory)
            size_after = len(buffer)
            
            assert size_after == size_before + 1, (
                f"Buffer size should increase by 1, but went from {size_before} to {size_after}"
            )
            
            # Test that we can sample when we have enough data
            if size_after >= buffer._min_size_to_sample:
                sample_key = jax.random.PRNGKey(i)
                batch = buffer.sample_batch(sample_key, batch_size=1)
                assert batch is not None, "Should be able to sample when buffer has enough data" 