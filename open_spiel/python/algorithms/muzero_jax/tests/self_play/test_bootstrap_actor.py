"""
Tests for the Bootstrap Actor implementation.

This module tests the bootstrap actor functionality, including chance node handling,
MCTS policy generation, and episode termination conditions.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import BootstrapActor, BootstrapConfig
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer


class TestBootstrapActor:
    """Test suite for the BootstrapActor class."""

    @pytest.fixture
    def mock_game_wrapper(self):
        """Create a mock GameWrapper for testing."""
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.num_distinct_actions.return_value = 9
        mock_wrapper.observation_shape = (27,)
        return mock_wrapper

    @pytest.fixture
    def mock_mcts_bot(self):
        """Create a mock MCTS bot for testing."""
        mock_bot = Mock()
        mock_bot.step.return_value = 0
        return mock_bot

    @pytest.fixture
    def bootstrap_actor(self, mock_game_wrapper):
        """Create a bootstrap actor for testing."""
        from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import BootstrapConfig
        
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)
        mock_config = BootstrapConfig(
            c_puct=1.0,
            num_simulations=10
        )
        
        # Mock the _game attribute that BootstrapActor tries to access
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100  # Return a reasonable max game length
        mock_game_wrapper._game = mock_game
        
        # Mock pyspiel.MCTSBot creation
        with patch('pyspiel.MCTSBot') as mock_mcts_bot_class:
            mock_mcts_bot_instance = Mock()
            mock_mcts_bot_instance.step_with_policy.return_value = ([(0, 1.0)], 0)
            mock_mcts_bot_class.return_value = mock_mcts_bot_instance
            
            actor = BootstrapActor(
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config
            )
            
            # Store the mock for access in tests
            actor.mcts_bot = mock_mcts_bot_instance
            return actor

    def test_bootstrap_actor_initialization(self, mock_game_wrapper):
        """Test that BootstrapActor initializes correctly."""
        from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import BootstrapConfig
        
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)
        mock_config = BootstrapConfig(
            c_puct=1.0,
            num_simulations=10
        )
        
        # Mock the _game attribute that BootstrapActor tries to access
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100  # Return a reasonable max game length
        mock_game_wrapper._game = mock_game
        
        # Mock pyspiel.MCTSBot creation  
        with patch('pyspiel.MCTSBot') as mock_mcts_bot_class:
            mock_mcts_bot_instance = Mock()
            mock_mcts_bot_class.return_value = mock_mcts_bot_instance
            
            actor = BootstrapActor(
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config
            )
            
            assert actor.game_wrapper == mock_game_wrapper
            assert actor.replay_buffer == mock_replay_buffer
            assert actor.config == mock_config

    def test_chance_nodes_are_sampled_without_logging(self, bootstrap_actor, mock_game_wrapper):
        """Chance nodes should be sampled but not yield trajectory entries."""
        mock_state = Mock()
        mock_state.is_terminal.side_effect = [False, False, True]
        mock_state.is_chance_node.side_effect = [True, False, False]
        mock_state.chance_outcomes.return_value = [(1, 0.2), (3, 0.8)]
        mock_state.observation_tensor.return_value = list(range(27))
        mock_state.legal_actions.return_value = [0]
        mock_state.rewards.return_value = [0.5]
        mock_state.apply_action.side_effect = lambda action: None
        mock_state.returns.return_value = [1.0]
        mock_game_wrapper._game.new_initial_state.return_value = mock_state

        with patch.object(bootstrap_actor, "_sample_chance_action", return_value=3) as mock_sampler:
            bootstrap_actor.mcts_bot.step_with_policy.return_value = ([(0, 1.0)], 0)
            data = bootstrap_actor.play_episode(jax.random.PRNGKey(0))

        mock_sampler.assert_called_once()
        # Only the decision node should appear in the trajectory.
        assert data['actions'] == [0]
        assert len(data['observations']) == 1
        assert len(data['policy_targets']) == 1

    def test_observation_none_breaks_episode(self, bootstrap_actor, mock_game_wrapper):
        """If observation tensors are missing we abort the episode."""
        mock_state = Mock()
        mock_state.is_terminal.return_value = False
        mock_state.is_chance_node.return_value = False
        mock_state.observation_tensor.return_value = None
        mock_state.legal_actions.return_value = []
        mock_state.rewards.return_value = []
        mock_state.returns.return_value = []
        mock_game_wrapper._game.new_initial_state.return_value = mock_state

        data = bootstrap_actor.play_episode(jax.random.PRNGKey(0))
        assert data['observations'] == []
        assert data['actions'] == []
        assert data['policy_targets'] == []
        assert data['rewards'] == []

    def test_policy_from_entries_normalizes_probabilities(self, bootstrap_actor):
        """Visit-count entries are normalized into a dense vector."""
        policy = bootstrap_actor._policy_from_entries([(1, 2.0), (3, 1.0)], legal_actions=[1, 3])
        assert policy[1] == pytest.approx(2.0 / 3.0)
        assert policy[3] == pytest.approx(1.0 / 3.0)
        assert jnp.sum(policy) == pytest.approx(1.0)

    def test_policy_from_entries_uniform_fallback(self, bootstrap_actor):
        """Empty entries fall back to a uniform distribution over legal moves."""
        policy = bootstrap_actor._policy_from_entries([], legal_actions=[0, 2, 4])
        assert policy[0] == pytest.approx(1 / 3)
        assert policy[2] == pytest.approx(1 / 3)
        assert policy[4] == pytest.approx(1 / 3)

    def test_policy_from_entries_handles_no_actions(self, bootstrap_actor):
        """If there are no entries and no legal actions we simply return all zeros."""
        policy = bootstrap_actor._policy_from_entries([], legal_actions=[])
        assert jnp.all(policy == 0.0)

    def test_complete_episode_records_actions_and_policies(self, bootstrap_actor, mock_game_wrapper):
        """End-to-end episode with two decision nodes."""
        step_count = 0

        def mock_is_terminal():
            return step_count >= 2

        def mock_is_chance_node():
            return False

        observations = [
            list(range(27)),
            list(range(1, 28)),
        ]

        def mock_observation_tensor():
            idx = min(step_count, len(observations) - 1)
            return observations[idx]

        def mock_apply_action(action):
            nonlocal step_count
            step_count += 1

        def mock_legal_actions():
            return [0, 1, 2, 3]

        rewards_per_step = [[0.0], [0.3]]

        def mock_rewards():
            if step_count == 0:
                return [0.0]
            idx = min(step_count - 1, len(rewards_per_step) - 1)
            return rewards_per_step[idx]

        mock_state = Mock()
        mock_state.is_terminal.side_effect = mock_is_terminal
        mock_state.is_chance_node.side_effect = mock_is_chance_node
        mock_state.observation_tensor.side_effect = mock_observation_tensor
        mock_state.apply_action.side_effect = mock_apply_action
        mock_state.legal_actions.side_effect = mock_legal_actions
        mock_state.rewards.side_effect = mock_rewards
        mock_state.returns.return_value = [1.0]
        mock_game_wrapper._game.new_initial_state.return_value = mock_state

        bootstrap_actor.mcts_bot.step_with_policy.side_effect = [
            ([(0, 0.25), (1, 0.75)], 1),
            ([(2, 1.0)], 2),
        ]

        data = bootstrap_actor.play_episode(jax.random.PRNGKey(0))

        assert data['actions'] == [1, 2]
        assert len(data['policy_targets']) == 2
        assert data['policy_targets'][0][1] == pytest.approx(0.75)
        assert data['policy_targets'][1][2] == pytest.approx(1.0)
        assert data['rewards'] == [0.0, 0.3]
        assert len(data['value_targets']) == 2

    def test_value_targets_match_discounted_returns(self, bootstrap_actor):
        """_compute_value_targets implements n-step discounted sums."""
        rewards = [1.0, 2.0, 3.0]
        targets = bootstrap_actor._compute_value_targets(rewards, final_value=0.5)
        assert len(targets) == 3
        # With n_step_return=5 the first target sums the discounted rewards only.
        expected = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        assert targets[0] == pytest.approx(expected)

    def test_sample_chance_action_respects_probabilities(self, bootstrap_actor):
        """_sample_chance_action honors provided probability weights."""
        outcomes = [(0, 0.0), (1, 1.0)]
        sampled = bootstrap_actor._sample_chance_action(jax.random.PRNGKey(0), outcomes)
        assert sampled == 1

    def test_sample_chance_action_uniform_fallback(self, bootstrap_actor):
        """Zero total probability falls back to uniform sampling over outcomes."""
        outcomes = [(7, 0.0)]
        sampled = bootstrap_actor._sample_chance_action(jax.random.PRNGKey(0), outcomes)
        assert sampled == 7

    def test_run_reports_zero_episode_stats(self, bootstrap_actor):
        """run() should still return stats even if no episodes are executed."""
        bootstrap_actor.replay_buffer = []
        stats = bootstrap_actor.run(jax.random.PRNGKey(0), num_episodes=0)
        assert stats['episodes_played'] == 0
        assert stats['avg_episode_length'] == 0
        assert stats['avg_episode_reward'] == 0



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
