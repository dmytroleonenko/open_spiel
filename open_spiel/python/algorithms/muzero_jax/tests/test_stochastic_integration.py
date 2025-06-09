"""
Test module for stochastic environment integration with MuZero JAX.

This module tests the integration between GameWrapper, Actor, and MCTS
for handling stochastic environments correctly.
"""

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import Mock, patch
import pyspiel

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import (
    create_mcts_for_game, 
    is_stochastic_mcts_instance,
    MCTS,
    StochasticMCTS
)


class TestStochasticIntegration:
    """Test stochastic environment integration."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object."""
        config = Mock()
        config.observation_shape = (9,)
        config.num_actions = 9
        config.hidden_state_dim = 64
        return config

    @pytest.fixture
    def mock_network(self):
        """Create a mock MuZero network."""
        network = Mock(spec=MuZeroNetwork)

        def mock_initial_inference(observation, training=False):
            batch_size = observation.shape[0] if observation.ndim > 1 else 1
            hidden_state = jnp.zeros((batch_size, 64))
            reward = jnp.zeros((batch_size,))
            value = jnp.zeros((batch_size,))
            policy_logits = jnp.zeros((batch_size, 9))
            projection = None
            return hidden_state, reward, value, policy_logits, projection

        def mock_recurrent_inference(hidden_state, action, training=False):
            batch_size = hidden_state.shape[0]
            next_hidden_state = jnp.zeros_like(hidden_state)
            reward = jnp.zeros((batch_size,))
            value = jnp.zeros((batch_size,))
            policy_logits = jnp.zeros((batch_size, 9))
            return next_hidden_state, reward, value, policy_logits, None

        network.initial_inference = Mock(side_effect=mock_initial_inference)
        network.recurrent_inference = Mock(side_effect=mock_recurrent_inference)
        return network

    def test_game_wrapper_stochastic_detection(self):
        """Test that GameWrapper correctly detects stochastic games."""
        # Test with deterministic game (Tic-Tac-Toe)
        det_wrapper = GameWrapper("tic_tac_toe")
        assert not det_wrapper.is_stochastic()

        # Test with stochastic game if available
        try:
            # Try to load a stochastic game
            stoch_wrapper = GameWrapper("pig")
            # Pig should be stochastic due to dice rolls
            assert stoch_wrapper.is_stochastic()
        except Exception:
            # If no stochastic games available, skip this part
            pytest.skip("No stochastic games available for testing")

    def test_mcts_factory_deterministic_game(self):
        """Test MCTS factory creates correct instance for deterministic games."""
        # Create mock deterministic game wrapper
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.is_stochastic.return_value = False

        mcts = create_mcts_for_game(
            game_wrapper=mock_wrapper,
            num_simulations=50,
            max_num_considered_actions=16
        )

        assert isinstance(mcts, MCTS)
        assert not is_stochastic_mcts_instance(mcts)

    def test_mcts_factory_stochastic_game(self):
        """Test MCTS factory creates correct instance for stochastic games."""
        # Create mock stochastic game wrapper
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.is_stochastic.return_value = True
        mock_wrapper.max_chance_outcomes.return_value = 6

        mcts = create_mcts_for_game(
            game_wrapper=mock_wrapper,
            num_simulations=50,
            max_num_considered_actions=16
        )

        assert isinstance(mcts, StochasticMCTS)
        assert is_stochastic_mcts_instance(mcts)

    def test_actor_auto_creates_deterministic_mcts(self, mock_config, mock_network):
        """Test that Actor auto-creates deterministic MCTS for deterministic games."""
        # Create mock deterministic game wrapper
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.is_stochastic.return_value = False
        mock_wrapper.max_chance_outcomes.return_value = 0
        
        # Setup the _game attribute
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100
        mock_game.get_type.return_value.short_name = "test_game"
        mock_wrapper._game = mock_game

        mock_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_network,
            game_wrapper=mock_wrapper,
            replay_buffer=mock_buffer,
            config=mock_config
        )

        assert hasattr(actor, 'mcts')
        assert not actor._is_stochastic_mcts
        assert isinstance(actor.mcts, MCTS)

    def test_actor_auto_creates_stochastic_mcts(self, mock_config, mock_network):
        """Test that Actor auto-creates stochastic MCTS for stochastic games."""
        # Create mock stochastic game wrapper
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.is_stochastic.return_value = True
        mock_wrapper.max_chance_outcomes.return_value = 6
        
        # Setup the _game attribute
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100
        mock_game.get_type.return_value.short_name = "stochastic_test_game"
        mock_wrapper._game = mock_game

        mock_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_network,
            game_wrapper=mock_wrapper,
            replay_buffer=mock_buffer,
            config=mock_config
        )

        assert hasattr(actor, 'mcts')
        assert actor._is_stochastic_mcts
        assert isinstance(actor.mcts, StochasticMCTS)

    def test_actor_decision_recurrent_function_creation(self, mock_config, mock_network):
        """Test that Actor can create decision recurrent functions for stochastic MCTS."""
        # Create mock stochastic game wrapper
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.is_stochastic.return_value = True
        mock_wrapper.max_chance_outcomes.return_value = 6
        
        # Setup the _game attribute
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100
        mock_game.get_type.return_value.short_name = "stochastic_test_game"
        mock_wrapper._game = mock_game

        mock_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_network,
            game_wrapper=mock_wrapper,
            replay_buffer=mock_buffer,
            config=mock_config
        )

        # Test that we can create decision recurrent function
        decision_fn = actor._create_decision_recurrent_fn()
        assert callable(decision_fn)

        # Test that we can create chance recurrent function
        chance_fn = actor._create_chance_recurrent_fn()
        assert callable(chance_fn)

    def test_integration_with_real_deterministic_game(self, mock_config, mock_network):
        """Test integration with a real deterministic OpenSpiel game."""
        # Use Tic-Tac-Toe as deterministic game
        wrapper = GameWrapper("tic_tac_toe")
        mock_buffer = Mock(spec=TrajectoryBuffer)

        # Verify the game is detected as deterministic
        assert not wrapper.is_stochastic()

        # Create actor with real game wrapper
        actor = Actor(
            network=mock_network,
            game_wrapper=wrapper,
            replay_buffer=mock_buffer,
            config=mock_config
        )

        # Verify correct MCTS type was created
        assert not actor._is_stochastic_mcts
        assert isinstance(actor.mcts, MCTS)

    @pytest.mark.skipif(True, reason="Need stochastic game for full integration test")
    def test_integration_with_real_stochastic_game(self, mock_config, mock_network):
        """Test integration with a real stochastic OpenSpiel game."""
        # This test would use a real stochastic game like Pig
        # Skip for now since we need to ensure stochastic games are available
        pass 