"""Integration tests for the complete MuZero JAX workflow.

These tests verify end-to-end behavior that unit tests might miss,
such as orchestrator properly adding episodes to replay buffer.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
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

# Import testing utilities like other tests do
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
    make_model, make_cfg, cfg_flat as common_cfg_flat
)


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
            'bootstrap': {
                'enabled': True,
                'min_episodes': 2,
                'c_puct': 1.25
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
        # Create a copy to avoid modifying the fixture
        import copy
        config = copy.deepcopy(minimal_config)
        
        # Force a scenario where batch_size > start_transitions so buffer
        # must reach batch_size before training occurs
        config.training.start_transitions = 1
        config.training.batch_size = 4
        config.resource_management.selfplay_phase_episodes = 1
        
        orchestrator = MuZeroOrchestrator(config)
        orchestrator.setup_components()
        
        selfplay_metrics = orchestrator.run_selfplay_phase()
        
        buffer_size = len(orchestrator.replay_buffer)
        assert buffer_size < config.training.batch_size
        
        initial_training_step = orchestrator.training_step
        metrics = orchestrator.run_training_phase()
        
        assert orchestrator.training_step == initial_training_step, "Training should be skipped"
        assert metrics['steps_trained'] == 0
        assert buffer_size < max(
            config.training.start_transitions, config.training.batch_size
        ), "Buffer should remain below the effective training threshold"

    def test_bootstrap_waits_for_batch_sized_buffer(self, minimal_config):
        """Bootstrap actor should keep running until buffer can satisfy training batch size."""
        import copy
        config = copy.deepcopy(minimal_config)
        config.training.start_transitions = 1
        config.training.batch_size = 3
        config.resource_management.selfplay_phase_episodes = 1
        config.bootstrap.min_episodes = 1
        
        orchestrator = MuZeroOrchestrator(config)
        orchestrator.setup_components()
        
        def dummy_episode():
            steps = 2
            obs = [np.zeros(orchestrator.game_wrapper.observation_shape, dtype=np.float32) for _ in range(steps)]
            actions = [0] * steps
            rewards = [0.0] * steps
            policy = [np.ones(orchestrator.muzero_config.num_actions, dtype=np.float32) / orchestrator.muzero_config.num_actions for _ in range(steps)]
            values = [0.0] * steps
            return {
                'observations': obs,
                'actions': actions,
                'rewards': rewards,
                'policy_targets': policy,
                'value_targets': values,
            }

        with patch.object(orchestrator.bootstrap_actor, 'play_episode', side_effect=[dummy_episode() for _ in range(6)]):
            # After first episode buffer < batch_size so bootstrap stays active
            orchestrator.run_selfplay_phase()
            assert orchestrator.use_bootstrap
            assert len(orchestrator.replay_buffer) == 1
            
            # Generate additional bootstrap episodes until buffer >= batch_size
            orchestrator.run_selfplay_phase()
            orchestrator.run_selfplay_phase()
            assert len(orchestrator.replay_buffer) >= config.training.batch_size
            
            # Next call should transition to MuZero actors automatically
            orchestrator.run_selfplay_phase()
            assert not orchestrator.use_bootstrap
    
    def test_complete_workflow_single_iteration(self, minimal_config):
        """Test complete workflow: self-play → buffer → training."""
        # Create a copy to avoid modifying the fixture
        import copy
        config = copy.deepcopy(minimal_config)
        
        # Ensure we have enough data to trigger training
        config.training.start_transitions = 1
        config.training.batch_size = 2  # Reduce batch size to allow training
        config.resource_management.selfplay_phase_episodes = 3
        
        orchestrator = MuZeroOrchestrator(config)
        orchestrator.setup_components()
        
        initial_training_step = orchestrator.training_step
        initial_buffer_size = len(orchestrator.replay_buffer)
        
        # Self-play phase
        selfplay_metrics = orchestrator.run_selfplay_phase()
        buffer_size_after_selfplay = len(orchestrator.replay_buffer)
        
        # Verify episodes were added to buffer
        assert buffer_size_after_selfplay > initial_buffer_size
        assert buffer_size_after_selfplay >= config.training.start_transitions
        
        # Training phase  
        training_metrics = orchestrator.run_training_phase()
        final_training_step = orchestrator.training_step
        
        # Verify training actually occurred
        assert final_training_step > initial_training_step, "Training step should have advanced"
        assert 'steps_trained' in training_metrics and training_metrics['steps_trained'] > 0, "Training should have occurred"


class TestActorBufferIntegration:
    """Integration tests for Actor and replay buffer interaction."""
    
    @pytest.fixture
    def setup_actor(self, common_cfg_flat):
        """Set up actor with minimal dependencies."""
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        # Create a config that matches tic_tac_toe dimensions
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import MockNetCfg
        tic_tac_toe_cfg = MockNetCfg(
            observation_shape=observation_shape,  # (27,)
            num_actions=num_actions,  # 9
            hidden_size=16,  # Reasonable hidden size
            value_support_size=0,  # Scalar values
            reward_support_size=0,  # Scalar rewards
            projection_output_size=8,
            use_projection=False,
            batch_size=1,
            noisy_net=False
        )
        
        # Create network using proper test utilities like other tests
        rng_key = jax.random.PRNGKey(42)
        network = make_model(rng_key, tic_tac_toe_cfg)
        
        # Create replay buffer
        buffer = TrajectoryBuffer(
            capacity=100,
            observation_shape=observation_shape,
            num_actions=num_actions
        )
        
        # Create actor config
        from types import SimpleNamespace
        actor_config = SimpleNamespace(num_actions=num_actions)
        
        # Create actor (will auto-create appropriate MCTS based on game type)
        actor = Actor(
            network=network,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=actor_config,
            num_simulations=3,  # Low for fast testing
            max_num_considered_actions=4,  # Low for fast testing
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
        batch_list = buffer.sample_batch(batch_size=1, rng_key=sample_rng)
        batch = batch_list[0]  # Get the first (and only) trajectory
        
        # Verify batch structure
        expected_keys = ['observations', 'actions', 'rewards', 'policy_targets', 'value_targets']
        for key in expected_keys:
            assert key in batch, f"Batch missing key: {key}"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_minimal_training_loop(self, common_cfg_flat):
        """Test minimal training loop without orchestrator complexity."""
        # This test verifies the basic workflow components work together
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        # Create a config that matches tic_tac_toe dimensions
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import MockNetCfg
        tic_tac_toe_cfg = MockNetCfg(
            observation_shape=observation_shape,  # (27,)
            num_actions=num_actions,  # 9
            hidden_size=16,  # Reasonable hidden size
            value_support_size=0,  # Scalar values
            reward_support_size=0,  # Scalar rewards
            projection_output_size=8,
            use_projection=False,
            batch_size=1,
            noisy_net=False
        )
        
        # Create components using proper test utilities
        rng_key = jax.random.PRNGKey(42)
        network = make_model(rng_key, tic_tac_toe_cfg)
        
        buffer = TrajectoryBuffer(
            capacity=100,
            observation_shape=observation_shape,
            num_actions=num_actions
        )
        
        # Create actor config
        from types import SimpleNamespace
        actor_config = SimpleNamespace(num_actions=num_actions)
        
        # Create actor (will auto-create appropriate MCTS based on game type)
        actor = Actor(
            network=network,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=actor_config,
            num_simulations=3,  # Low for fast testing
            max_num_considered_actions=4,  # Low for fast testing
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
        batch_list = buffer.sample_batch(batch_size=1, rng_key=sample_key)
        batch = batch_list[0]  # Get the first (and only) trajectory
        
        # Verify we can forward pass through network
        # Add batch dimension to observations from the single trajectory
        obs_batch = jnp.expand_dims(batch['observations'][0], axis=0)  # Take first obs and add batch dim
        hidden_state, reward, value, policy_logits, _, reward_hidden = network.initial_inference(
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
            'bootstrap': {
                'enabled': True,
                'min_episodes': 2,
                'c_puct': 1.25
            },
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
            'random_seed': 42,
            'self_play': {
                'episodes_per_iteration': 2,
                'parallel_episodes': 1,
                'use_gumbel': True,
                'exploration_fraction': 0.25,
                'dirichlet_alpha': 0.3,
                'n_step_return': 3,
                'discount_factor': 0.99
            },
            'network': {
                'hidden_dim': 16,
                'num_blocks': 1,
                'use_image_observation': False,
                'use_projection': False
            }
        })
        
        orchestrator = MuZeroOrchestrator(config)
        orchestrator.setup_components()
        
        # Verify initial state
        initial_buffer_size = len(orchestrator.replay_buffer)
        initial_training_step = orchestrator.training_step
        assert initial_buffer_size == 0, "Buffer should start empty"
        
        # Run one iteration of self-play
        selfplay_metrics = orchestrator.run_selfplay_phase()
        
        # CRITICAL: This should have caught our bug
        buffer_size_after_selfplay = len(orchestrator.replay_buffer)
        episodes_played = selfplay_metrics['episodes_played']
        
        # This assertion would have failed with the original bug
        assert buffer_size_after_selfplay > 0, (
            f"WORKFLOW BUG: Self-play generated {episodes_played} episodes "
            f"but buffer only contains {buffer_size_after_selfplay} episodes. "
            f"Episodes are not being added to buffer!"
        )
        
        assert buffer_size_after_selfplay == episodes_played, (
            f"Buffer size mismatch: {buffer_size_after_selfplay} vs {episodes_played}"
        )


class TestBufferStateConsistency:
    """Tests for buffer state consistency issues."""
    
    def test_buffer_size_reporting_accuracy(self, common_cfg_flat):
        """Test that buffer size is accurately reported in metrics."""
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        # Create a config that matches tic_tac_toe dimensions
        from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import MockNetCfg
        tic_tac_toe_cfg = MockNetCfg(
            observation_shape=observation_shape,  # (27,)
            num_actions=num_actions,  # 9
            hidden_size=16,  # Reasonable hidden size
            value_support_size=0,  # Scalar values
            reward_support_size=0,  # Scalar rewards
            projection_output_size=8,
            use_projection=False,
            batch_size=1,
            noisy_net=False
        )
        
        # Create components using proper test utilities
        rng_key = jax.random.PRNGKey(42)
        network = make_model(rng_key, tic_tac_toe_cfg)
        
        buffer = TrajectoryBuffer(
            capacity=100,
            observation_shape=observation_shape,
            num_actions=num_actions
        )
        
        # Create actor config
        from types import SimpleNamespace
        actor_config = SimpleNamespace(num_actions=num_actions)
        
        # Create actor (will auto-create appropriate MCTS based on game type)
        actor = Actor(
            network=network,
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=actor_config,
            num_simulations=3,  # Low for fast testing
            max_num_considered_actions=4,  # Low for fast testing
            n_step_return=3,
            discount_factor=0.99
        )
        
        # Test initial state
        assert len(buffer) == 0, "Buffer should start empty"
        
        # Add multiple episodes
        episodes_to_add = 3
        for i in range(episodes_to_add):
            rng_key = jax.random.fold_in(jax.random.PRNGKey(42), i)
            trajectory = actor.play_episode(rng_key)
            buffer.add_trajectory(trajectory)
            
            # Check size is accurate after each addition
            expected_size = i + 1
            actual_size = len(buffer)
            assert actual_size == expected_size, (
                f"After adding episode {i+1}, buffer size should be {expected_size}, got {actual_size}"
            )
        
        # Final check
        final_size = len(buffer)
        assert final_size == episodes_to_add, (
            f"Final buffer size should be {episodes_to_add}, got {final_size}"
        ) 
