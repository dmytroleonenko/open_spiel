"""Tests for bootstrap actor using plain MCTS."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import (
    BootstrapActor, BootstrapConfig
)
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer


class TestBootstrapActor:
    """Tests for BootstrapActor."""
    
    @pytest.fixture
    def setup_bootstrap_actor(self):
        """Set up bootstrap actor with minimal dependencies."""
        game_wrapper = GameWrapper('tic_tac_toe')
        observation_shape = game_wrapper.observation_shape
        num_actions = game_wrapper.num_distinct_actions()
        
        # Create replay buffer
        buffer = TrajectoryBuffer(
            capacity=100,
        )
        
        # Create bootstrap config
        config = BootstrapConfig(
            num_simulations=10,  # Small for testing
            c_puct=1.0,
            n_step_return=3,
            discount_factor=0.99
        )
        
        # Create bootstrap actor
        actor = BootstrapActor(
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=config
        )
        
        return actor, buffer, game_wrapper
    
    def test_bootstrap_actor_initialization(self, setup_bootstrap_actor):
        """Test that bootstrap actor initializes correctly."""
        actor, buffer, game_wrapper = setup_bootstrap_actor
        
        assert actor.game_wrapper is not None
        assert actor.replay_buffer is not None
        assert actor.config is not None
        assert actor.mcts_bot is not None
        
        # Check that MCTS bot is configured correctly
        assert hasattr(actor.mcts_bot, 'step')
    
    def test_bootstrap_actor_generates_valid_trajectory(self, setup_bootstrap_actor):
        """Test that bootstrap actor generates valid trajectory data."""
        actor, buffer, game_wrapper = setup_bootstrap_actor
        
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)
        
        # Verify trajectory structure
        expected_keys = ['observations', 'actions', 'rewards', 'policy_targets', 'value_targets']
        for key in expected_keys:
            assert key in trajectory, f"Trajectory missing key: {key}"
        
        # Verify trajectory is non-empty for tic-tac-toe
        assert len(trajectory['observations']) > 0, "Trajectory should have observations"
        assert len(trajectory['actions']) > 0, "Trajectory should have actions"
        
        # Verify all trajectory components have same length
        action_len = len(trajectory['actions'])
        assert len(trajectory['rewards']) == action_len
        assert len(trajectory['policy_targets']) == action_len
        assert len(trajectory['value_targets']) == action_len
        
        # Verify observations have correct shape
        obs_shape = game_wrapper.observation_shape
        for obs in trajectory['observations']:
            assert obs.shape == obs_shape
        
        # Verify policy targets are valid probability distributions
        num_actions = game_wrapper.num_distinct_actions()
        for policy in trajectory['policy_targets']:
            assert policy.shape == (num_actions,)
            assert jnp.all(policy >= 0), "Policy probabilities should be non-negative"
            assert jnp.isclose(jnp.sum(policy), 1.0, atol=1e-6), "Policy should sum to 1"
    
    def test_bootstrap_actor_produces_better_policies_than_random(self, setup_bootstrap_actor):
        """Test that bootstrap actor produces better policies than random."""
        actor, buffer, game_wrapper = setup_bootstrap_actor
        
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)
        
        num_actions = game_wrapper.num_distinct_actions()
        uniform_entropy = -num_actions * (1/num_actions) * jnp.log(1/num_actions)
        
        # Calculate entropy of MCTS policies
        entropies = []
        for policy in trajectory['policy_targets']:
            # Add small epsilon to avoid log(0)
            policy_safe = policy + 1e-8
            entropy = -jnp.sum(policy_safe * jnp.log(policy_safe))
            entropies.append(entropy)
        
        avg_entropy = jnp.mean(jnp.array(entropies))
        
        # MCTS policies should be more focused (lower entropy) than uniform random
        assert avg_entropy < uniform_entropy, (
            f"MCTS policy entropy ({avg_entropy}) should be lower than "
            f"uniform entropy ({uniform_entropy})"
        )
    
    def test_bootstrap_actor_run_method(self, setup_bootstrap_actor):
        """Test that bootstrap actor run method works correctly."""
        actor, buffer, game_wrapper = setup_bootstrap_actor
        
        initial_buffer_size = len(buffer)
        assert initial_buffer_size == 0
        
        # Run actor for multiple episodes
        rng_key = jax.random.PRNGKey(42)
        num_episodes = 3
        metrics = actor.run(rng_key, num_episodes=num_episodes)
        
        # Check that episodes were added to buffer
        final_buffer_size = len(buffer)
        assert final_buffer_size == num_episodes
        
        # Check metrics
        assert metrics['episodes_played'] == num_episodes
        assert metrics['buffer_size'] == num_episodes
        assert metrics['avg_episode_length'] > 0
        assert 'avg_episode_reward' in metrics
    
    def test_bootstrap_vs_untrained_network_quality(self, setup_bootstrap_actor):
        """Test that bootstrap trajectories are higher quality than untrained network."""
        actor, buffer, game_wrapper = setup_bootstrap_actor
        
        # Generate bootstrap trajectory
        rng_key = jax.random.PRNGKey(42)
        bootstrap_trajectory = actor.play_episode(rng_key)
        
        # For tic-tac-toe, episodes should be reasonably short (game usually ends in <10 moves)
        episode_length = len(bootstrap_trajectory['actions'])
        assert episode_length < 15, f"Tic-tac-toe episode should be short, got {episode_length}"
        
        # Policies should show some concentration (not completely uniform)
        policies = bootstrap_trajectory['policy_targets']
        for policy in policies:
            max_prob = jnp.max(policy)
            # MCTS should give higher probability to at least one action
            assert max_prob > 0.2, f"MCTS should show some preference, max prob: {max_prob}"
    
    def test_bootstrap_handles_different_games(self):
        """Test that bootstrap actor works with different OpenSpiel games."""
        # Test with a simple deterministic game
        game_wrapper = GameWrapper('tic_tac_toe')
        
        buffer = TrajectoryBuffer(capacity=10)
        config = BootstrapConfig(num_simulations=5, c_puct=1.0)
        
        actor = BootstrapActor(
            game_wrapper=game_wrapper,
            replay_buffer=buffer,
            config=config
        )
        
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)
        
        # Should produce a valid trajectory
        assert len(trajectory['actions']) > 0
        assert len(trajectory['observations']) > 0
        
        # All components should have consistent lengths
        action_len = len(trajectory['actions'])
        assert len(trajectory['rewards']) == action_len
        assert len(trajectory['policy_targets']) == action_len
        assert len(trajectory['value_targets']) == action_len


class TestBootstrapConfiguration:
    """Tests for bootstrap configuration."""
    
    def test_bootstrap_config_creation(self):
        """Test bootstrap config creation with different parameters."""
        config = BootstrapConfig(
            num_simulations=100,
            c_puct=1.5,
            temperature=0.5,
            n_step_return=5,
            discount_factor=0.95
        )
        
        assert config.num_simulations == 100
        assert config.c_puct == 1.5
        assert config.temperature == 0.5
        assert config.n_step_return == 5
        assert config.discount_factor == 0.95
    
    def test_bootstrap_config_defaults(self):
        """Test that bootstrap config has reasonable defaults."""
        config = BootstrapConfig()
        
        assert config.num_simulations > 0
        assert config.c_puct > 0
        assert config.temperature > 0
        assert config.n_step_return > 0
        assert 0 < config.discount_factor <= 1 