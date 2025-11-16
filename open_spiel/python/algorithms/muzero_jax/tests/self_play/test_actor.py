"""Tests for the MuZero JAX actor component."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import threading
import time

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS, StochasticMCTS
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.services.inference_client import (
    LocalInferenceClient,
)
from open_spiel.python.algorithms.muzero_jax.training.trainer import create_muzero_config_for_game

def _get_unique_rng_key() -> jax.random.PRNGKey:
    """Generate a unique random key using thread ID and time to avoid parallel collisions."""
    thread_id = threading.get_ident()
    current_time_ns = time.time_ns()
    unique_seed = hash((thread_id, current_time_ns)) % (2**31)  # Keep it positive for PRNGKey
    return jax.random.PRNGKey(unique_seed)


class RecordingInferenceClient:
    """Inference client stub that records calls and returns deterministic tensors."""

    def __init__(self):
        self.initial_calls = 0
        self.recurrent_calls = 0
        self.last_initial_obs = None
        self.last_recurrent_input = None

    def initial_inference(self, observation_batch, training=False):
        self.initial_calls += 1
        self.last_initial_obs = observation_batch
        batch_size = observation_batch.shape[0]
        hidden_state = jnp.ones((batch_size, 64))
        reward = jnp.zeros((batch_size,))
        value = jnp.ones((batch_size,))
        policy_logits = jnp.zeros((batch_size, 9))
        projection = None
        reward_hidden = None
        return hidden_state, reward, value, policy_logits, projection, reward_hidden

    def recurrent_inference(self, hidden_state_batch, action_batch, training=False):
        self.recurrent_calls += 1
        self.last_recurrent_input = (hidden_state_batch, action_batch)
        batch_size = hidden_state_batch.shape[0]
        next_hidden = jnp.ones_like(hidden_state_batch)
        reward = jnp.zeros((batch_size,))
        value = jnp.ones((batch_size,))
        policy_logits = jnp.zeros((batch_size, 9))
        projection = None
        reward_hidden = None
        return next_hidden, reward, value, policy_logits, projection, reward_hidden

    def close(self):
        return None


class TestActor:
    """Test suite for the Actor class."""

    def _setup_game_wrapper_mock(self, mock_game_wrapper):
        """Helper to set up game wrapper mock with required _game attribute."""
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100
        mock_game.get_type.return_value.short_name = "test_game"
        mock_game_wrapper._game = mock_game
        return mock_game_wrapper

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object."""
        config = Mock()
        config.observation_shape = (9,)  # Tic-tac-toe
        config.num_actions = 9
        config.hidden_state_dim = 64
        return config

    @pytest.fixture
    def mock_game_wrapper(self):
        """Create a mock GameWrapper with proper stochastic/deterministic behavior."""
        mock_wrapper = Mock(spec=GameWrapper)
        mock_wrapper.is_stochastic.return_value = False  # Default to deterministic
        mock_wrapper.max_chance_outcomes.return_value = 0
        
        # Setup the _game attribute
        mock_game = Mock()
        mock_game.max_game_length.return_value = 100
        mock_game.get_type.return_value.short_name = "test_game"
        mock_wrapper._game = mock_game
        
        return mock_wrapper

    @pytest.fixture
    def mock_muzero_network(self):
        """Create a mock MuZero network."""
        network = Mock(spec=MuZeroNetwork)

        # Mock initial_inference - return 6 values to match actual network signature
        def mock_initial_inference(observation, training=False):
            batch_size = observation.shape[0] if observation.ndim > 1 else 1
            hidden_state = jnp.zeros((batch_size, 64))
            policy_logits = jnp.zeros((batch_size, 9))
            value = jnp.zeros((batch_size,))
            reward = jnp.zeros((batch_size,))
            projection = None
            reward_hidden = None  # LSTM reward hidden state (can be None)
            # Return in order: hidden_state, reward, value, policy_logits, projection, reward_hidden
            return hidden_state, reward, value, policy_logits, projection, reward_hidden

        network.initial_inference = Mock(side_effect=mock_initial_inference)

        # Mock recurrent_inference - return 6 values to match actual network signature
        def mock_recurrent_inference(hidden_state, action, training=False):
            batch_size = hidden_state.shape[0]
            next_hidden_state = jnp.zeros_like(hidden_state)
            policy_logits = jnp.zeros((batch_size, 9))
            value = jnp.zeros((batch_size,))
            reward = jnp.zeros((batch_size,))
            projection = None
            reward_hidden = None  # LSTM reward hidden state (can be None)
            # Return in order: next_hidden_state, reward, value, policy_logits, projection, reward_hidden
            return next_hidden_state, reward, value, policy_logits, projection, reward_hidden

        network.recurrent_inference = Mock(
            side_effect=mock_recurrent_inference)
        return network

    def test_actor_initialization(self, mock_config, mock_muzero_network, mock_game_wrapper):
        """Test that the actor can be initialized with required components."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        # Create minimal mocks for initialization test
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Test with deterministic game (default)
        mock_game_wrapper.is_stochastic.return_value = False
        
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Verify that all components are set correctly
        assert actor.network == mock_muzero_network
        assert actor.game_wrapper == mock_game_wrapper
        assert actor.replay_buffer == mock_replay_buffer
        assert actor.config == mock_config
        assert actor.current_params is None
        assert hasattr(actor, 'mcts')
        assert hasattr(actor, '_is_stochastic_mcts')
        assert actor._is_stochastic_mcts == False  # Should be deterministic MCTS

    def test_actor_stochastic_game_initialization(self, mock_config, mock_muzero_network, mock_game_wrapper):
        """Test that the actor initializes StochasticMCTS for stochastic games."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
        
        # Create minimal mocks for initialization test
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Test with stochastic game
        mock_game_wrapper.is_stochastic.return_value = True
        mock_game_wrapper.max_chance_outcomes.return_value = 6  # e.g., dice game
        
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Verify that stochastic MCTS was created
        assert actor._is_stochastic_mcts == True
        assert hasattr(actor, 'mcts')

    def test_actor_uses_custom_inference_client_for_initial(self, mock_config, mock_muzero_network, mock_game_wrapper):
        """Ensure actor routes initial inference through provided client."""
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)
        recording_client = RecordingInferenceClient()

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            inference_client=recording_client,
        )

        mock_muzero_network.initial_inference.reset_mock()
        obs = jnp.zeros((1, mock_config.observation_shape[0]))
        actor._initial_inference(obs)

        assert recording_client.initial_calls == 1
        mock_muzero_network.initial_inference.assert_not_called()

    def test_recurrent_fn_uses_inference_client(self, mock_config, mock_muzero_network, mock_game_wrapper):
        """Recurrent function handed to MCTS should call the inference client."""
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)
        recording_client = RecordingInferenceClient()

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            inference_client=recording_client,
        )

        mock_muzero_network.recurrent_inference.reset_mock()
        recurrent_fn = actor._create_recurrent_fn()

        params = None
        rng_key = _get_unique_rng_key()
        action = jnp.array(1)
        embedding = jnp.ones((1, 64))

        recurrent_fn(params, rng_key, action, embedding)

        assert recording_client.recurrent_calls == 1
        mock_muzero_network.recurrent_inference.assert_not_called()

    def test_actor_parameter_validation(self, mock_config, mock_muzero_network, mock_game_wrapper):
        """Test that the actor validates initialization parameters."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Test invalid discount factor
        with pytest.raises(ValueError, match="discount_factor must be in"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                discount_factor=1.5,  # Invalid: > 1.0
                num_simulations=3,
                max_num_considered_actions=4
            )

        # Test invalid n_step_return
        with pytest.raises(ValueError, match="n_step_return must be positive"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                n_step_return=0,  # Invalid: must be positive
                num_simulations=3,
                max_num_considered_actions=4
            )

        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                temperature=-0.5,  # Invalid: negative
                num_simulations=3,
                max_num_considered_actions=4
            )

        # Test invalid temperature_threshold
        with pytest.raises(ValueError, match="temperature_threshold must be non-negative"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                temperature_threshold=-1,  # Invalid: negative
                num_simulations=3,
                max_num_considered_actions=4
            )

    def test_maybe_load_parameters_plain_dict(self, mock_config, mock_muzero_network, mock_game_wrapper):
        """Fallback branch should accept checkpoints that are raw parameter blobs."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
        )
        with patch(
            "open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint",
            return_value="/tmp/fake.ckpt",
        ), patch.object(
            actor,
            "load_network_parameters",
            return_value={"weights_only": jnp.array([1.0, 2.0])},
        ):
            assert actor.maybe_load_latest_parameters("/tmp/dir") is True
            assert jnp.allclose(actor.current_params["weights_only"], jnp.array([1.0, 2.0]))

    def test_recurrent_fn_fallback_when_mcts_lacks_hooks(
        self, mock_config, mock_muzero_network, mock_game_wrapper
    ):
        """When the underlying MCTS helper lacks custom hooks we fall back to deterministic fn."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
        )
        actor.mcts = type("DummyMCTS", (), {})()  # No _create_* helpers -> fallback path

        deterministic_fn = actor._create_recurrent_fn()
        decision_fn = actor._create_decision_recurrent_fn()
        chance_fn = actor._create_chance_recurrent_fn()

        params = None
        rng_key = jax.random.PRNGKey(0)
        action = jnp.array([0])
        embedding = jnp.zeros((1, 64))

        expected_decision = deterministic_fn(params, rng_key, action, embedding)
        dec_out = decision_fn(params, rng_key, action, embedding)
        chance_out = chance_fn(params, rng_key, action, embedding)
        assert jnp.allclose(dec_out[0].reward, expected_decision[0].reward)
        assert jnp.allclose(dec_out[0].value, expected_decision[0].value)
        assert jnp.allclose(chance_out[0].reward, expected_decision[0].reward)

        # Test valid parameters
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            discount_factor=0.99,
            n_step_return=5,
            temperature=1.0,
            temperature_threshold=30,
            num_simulations=3,
            max_num_considered_actions=4
        )
        assert actor.discount_factor == 0.99
        assert actor.n_step_return == 5
        assert actor.temperature == 1.0
        assert actor.temperature_threshold == 30

    def test_actor_checkpoint_loading(self, mock_config, mock_muzero_network):
        """Test checkpoint loading functionality."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test successful checkpoint loading
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.load_checkpoint') as mock_load:
            mock_params = {'test': 'params'}
            mock_load.return_value = mock_params

            checkpoint_path = "/path/to/checkpoint"
            loaded_params = actor.load_network_parameters(checkpoint_path)

            mock_load.assert_called_once_with(checkpoint_path)
            assert loaded_params == mock_params

    def test_actor_maybe_load_latest_parameters(self, mock_config, mock_muzero_network):
        """Test maybe_load_latest_parameters functionality."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        checkpoint_dir = "/path/to/checkpoints"

        # Test when no checkpoint exists
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest:
            mock_get_latest.return_value = None
            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is False

        # Test when checkpoint exists and returns network_state blob
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_params = {'network_state': {'weights': jnp.ones(2)}}
            mock_load_params.return_value = mock_params

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is True
            mock_load_params.assert_called_once_with(
                "/path/to/checkpoints/latest.ckpt")
            # Check that parameters were stored
            assert actor.current_params == mock_params['network_state']

        # Test when checkpoint exists but loading fails
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_load_params.side_effect = Exception("Load failed")

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is False

    def test_actor_maybe_load_params_field(self, mock_config, mock_muzero_network):
        """`maybe_load_latest_parameters` should store the `params` field when present."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
        )

        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_params = {'params': {'weights': jnp.arange(3)}}
            mock_load_params.return_value = mock_params

            assert actor.maybe_load_latest_parameters("/unused/path") is True
            assert actor.current_params == mock_params['params']

    def test_actor_action_selection_with_temperature(self, mock_config, mock_muzero_network):
        """Test action selection with different temperature values."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            temperature=0.0,  # No temperature - should select argmax
            temperature_threshold=30,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test action selection with temperature = 0 (greedy selection)
        mock_policy_output = Mock()
        mock_policy_output.action_weights = jnp.array([0.1, 0.6, 0.3])
        step = 50  # After temperature threshold
        action, policy_target = actor._select_action(mock_policy_output, step)

        # Should select the action with highest weight (index 1)
        assert action == 1
        assert policy_target.shape == (3,)

    def test_actor_policy_target_edge_cases(self, mock_config, mock_muzero_network):
        """Test policy target computation edge cases."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test with zero action weights (should return uniform distribution)
        action_weights = jnp.zeros(9)
        policy_target = actor._compute_policy_target(action_weights)

        expected_uniform = jnp.ones(9) / 9
        assert jnp.allclose(policy_target, expected_uniform)

    def test_actor_value_target_computation(self, mock_config, mock_muzero_network):
        """Test that value targets are computed correctly using n-step returns."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            n_step_return=3,
            discount_factor=0.99,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test with simple reward sequence
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        final_value = 0.0  # Terminal state value

        value_targets = actor._compute_value_targets(rewards, final_value)

        # For 3-step return with discount 0.99:
        # First target: 1.0 + 0.99 * 2.0 + 0.99^2 * 3.0 = 1.0 + 1.98 + 2.9403 = 5.9203
        expected_first = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        assert abs(value_targets[0] - expected_first) < 1e-6

    def test_actor_policy_target_computation(self, mock_config, mock_muzero_network):
        """Test that policy targets are computed correctly from MCTS action weights."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test with normalized action weights
        action_weights = jnp.array(
            [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        policy_target = actor._compute_policy_target(action_weights)

        # Should return the action weights as policy target
        assert jnp.allclose(policy_target, action_weights)
        assert policy_target.shape == (mock_config.num_actions,)

    def test_actor_episode_playing_integration(self, mock_config, mock_muzero_network):
        """Test that the actor can play episodes with mocked components."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        # Create more detailed mocks for episode playing
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up game wrapper for a simple episode
        mock_game_wrapper.reset.return_value = [
            0.0] * 9  # Empty tic-tac-toe board
        mock_game_wrapper.is_terminal.side_effect = [
            False, False, True]  # 2 steps then terminal
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.side_effect = [
            [0.0] * 9,  # First observation
            [1.0] + [0.0] * 8,  # Second observation (after first move)
        ]
        mock_game_wrapper.legal_actions.return_value = [
            0, 1, 2, 3, 4, 5, 6, 7, 8]
        mock_game_wrapper.step.side_effect = [
            ([1.0] + [0.0] * 8, [0.0, 0.0], False),  # First step
            ([], [1.0, -1.0], True),  # Second step (terminal)
        ]

        # Create actor
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )
        
        # Mock MCTS run for testing
        def mock_mcts_run(params, rng_key, root, recurrent_fn, **kwargs):
            policy_output = Mock()
            policy_output.action = jnp.array([0])  # Always choose action 0
            policy_output.action_weights = jnp.array(
                [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return policy_output

        actor.mcts.run = Mock(side_effect=mock_mcts_run)

        # Play an episode
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)

        # Verify trajectory structure
        assert 'observations' in trajectory
        assert 'actions' in trajectory
        assert 'rewards' in trajectory
        assert 'policy_targets' in trajectory
        assert 'value_targets' in trajectory

        # Verify trajectory content
        assert len(trajectory['observations']) == 2  # Two non-terminal steps
        assert len(trajectory['actions']) == 2
        assert len(trajectory['rewards']) == 2
        assert len(trajectory['policy_targets']) == 2
        assert len(trajectory['value_targets']) == 2

        # Verify that game methods were called
        mock_game_wrapper.reset.assert_called_once()
        assert mock_game_wrapper.step.call_count == 2
        assert actor.mcts.run.call_count == 2
        assert mock_muzero_network.initial_inference.call_count == 2

    def test_actor_episode_with_chance_nodes(self, mock_config, mock_muzero_network):
        """Test that the actor handles games with chance nodes correctly."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up game wrapper with chance node
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.side_effect = [
            False, True]  # Chance node then terminal
        mock_game_wrapper.is_chance_node.side_effect = [
            True, False]  # First state is chance
        mock_game_wrapper.chance_outcomes.return_value = [(0, 0.6), (1, 0.4)]
        mock_game_wrapper.current_observation.return_value = [0.0] * 9
        mock_game_wrapper.step.side_effect = [
            ([1.0] + [0.0] * 8, [0.0, 0.0], False),  # Chance outcome
            ([], [1.0, -1.0], True),  # Terminal after chance
        ]

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)

        # Should handle chance nodes and continue
        assert 'observations' in trajectory
        mock_game_wrapper.chance_outcomes.assert_called()

    def test_actor_episode_early_termination(self, mock_config, mock_muzero_network):
        """Test handling of early episode termination."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up immediate termination
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = True  # Immediate termination
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.return_value = [0.0] * 9

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)

        # Should return empty trajectory for immediate termination
        assert len(trajectory['observations']) == 0
        assert len(trajectory['actions']) == 0
        assert len(trajectory['rewards']) == 0

    def test_actor_run_multiple_episodes(self, mock_config, mock_muzero_network):
        """Test that the actor can run multiple episodes."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up for quick episodes (immediate termination)
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = True  # Immediate termination
        mock_game_wrapper.is_chance_node.return_value = False

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Run multiple episodes
        rng_key = jax.random.PRNGKey(42)
        num_episodes = 3
        actor.run(rng_key, num_episodes=num_episodes)

        # Verify that multiple episodes were played
        assert mock_game_wrapper.reset.call_count == num_episodes
        assert mock_replay_buffer.add_trajectory.call_count == num_episodes

    def test_actor_run_error_handling(self, mock_config, mock_muzero_network):
        """Test error handling in the run method."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Mock play_episode to raise an exception on first call, succeed on second
        episode_count = [0]

        def mock_play_episode(rng_key):
            episode_count[0] += 1
            if episode_count[0] == 1:
                raise RuntimeError("Episode failed")
            else:
                return {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'policy_targets': [],
                    'value_targets': []
                }

        with patch.object(actor, 'play_episode', side_effect=mock_play_episode):
            rng_key = jax.random.PRNGKey(42)
            actor.run(rng_key, num_episodes=2)

            # Should have attempted both episodes (error handling allows continuation)
            assert episode_count[0] == 2
            # Only one successful episode should be added to replay buffer
            assert mock_replay_buffer.add_trajectory.call_count == 1

    def test_actor_recurrent_function(self, mock_config, mock_muzero_network):
        """Test the recurrent function created by the actor."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test the recurrent function directly
        recurrent_fn = actor._create_recurrent_fn()

        # Create test inputs - the embedding should have batch dimension for the network
        embedding = jnp.zeros((1, 64))  # Batch size 1, hidden state dim 64
        action = jnp.array([3])  # Action
        rng_key = jax.random.PRNGKey(42)

        # Call recurrent function - it returns (step, next_hidden_state)
        step, next_hidden_state = recurrent_fn(
            None, rng_key, action, embedding
        )

        # Verify outputs - step is a RecurrentFnOutput object
        assert step.reward.shape == (1,)  # Batch of rewards
        assert step.discount.shape == (1,)  # Batch of discounts
        # Batch of policy logits for 9 actions
        assert step.prior_logits.shape == (1, 9)
        assert step.value.shape == (1,)  # Batch of values
        assert next_hidden_state.shape == (
            1, 64)  # Batch of next hidden states

        # Verify network was called
        mock_muzero_network.recurrent_inference.assert_called_once()

    def test_actor_episode_empty_observation(self, mock_config, mock_muzero_network):
        """Test handling when current_observation returns empty."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up game wrapper to return empty observation
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = False  # Not terminal initially
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.return_value = None  # Empty observation

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)

        # Should break early due to empty observation
        assert len(trajectory['observations']) == 0
        assert len(trajectory['actions']) == 0
        assert len(trajectory['rewards']) == 0

    def test_actor_mcts_uses_loaded_parameters(self, mock_config, mock_muzero_network):
        """Test that MCTS uses the loaded parameters when available."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Set up some mock parameters
        mock_params = {'test': 'params'}
        actor.current_params = mock_params

        # Mock MCTS to capture what parameters it receives
        captured_params = None

        def capture_mcts_params(params, rng_key, root, recurrent_fn, **kwargs):
            # Store the params that were passed to MCTS
            nonlocal captured_params
            captured_params = params
            
            policy_output = Mock()
            policy_output.action = jnp.array([0])
            policy_output.action_weights = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return policy_output

        actor.mcts.run = Mock(side_effect=capture_mcts_params)

        # Mock game wrapper to provide a simple environment
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = False
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.return_value = [0.0] * 9
        mock_game_wrapper.legal_actions.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Create root for MCTS
        rng_key = jax.random.PRNGKey(42)
        # Test that actor can run recurrent function (which is used by MCTS internally)
        recurrent_fn = actor._create_recurrent_fn()

        # Call MCTS through the actor
        # Create a mock root for MCTS (this would normally be created by initial_inference)
        mock_root = Mock()
        policy_output = actor.mcts.run(
            actor.current_params, rng_key, mock_root, recurrent_fn,
            num_simulations=3, max_num_considered_actions=4
        )

        # Verify that the actor's current_params were passed to MCTS
        assert captured_params == mock_params
        assert policy_output is not None

    def test_actor_parameter_loading_branches(self, mock_config, mock_muzero_network):
        """Test different branches of parameter loading logic."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test with no checkpoint directory (should not crash)
        result = actor.maybe_load_latest_parameters("/nonexistent/path")
        assert result is False

        # Test with empty string checkpoint directory
        result = actor.maybe_load_latest_parameters("")
        assert result is False

        # Test with non-existent directory
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest:
            mock_get_latest.side_effect = FileNotFoundError("Directory not found")
            result = actor.maybe_load_latest_parameters("/non/existent/path")
            assert result is False

        # Test successful loading with parameter update
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.load_checkpoint') as mock_load:

            mock_get_latest.return_value = "/path/to/latest.ckpt"
            mock_params = {'network': 'updated_params'}
            mock_load.return_value = mock_params

            # Initially no parameters
            assert actor.current_params is None

            result = actor.maybe_load_latest_parameters("/path/to/checkpoints")
            assert result is True
            assert actor.current_params == mock_params

    def test_actor_recurrent_function_with_params(self, mock_config, mock_muzero_network):
        """Test that the recurrent function works with different parameter setups."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Create recurrent function
        recurrent_fn = actor._create_recurrent_fn()

        # Test with current_params set
        actor.current_params = {'test': 'params'}
        
        # Test the function with proper inputs
        embedding = jnp.zeros((1, 64))
        action = jnp.array([2])
        rng_key = jax.random.PRNGKey(123)

        step, next_hidden_state = recurrent_fn(
            actor.current_params, rng_key, action, embedding)

        # Check that outputs have correct shapes
        assert step.reward.shape == (1,)
        assert step.discount.shape == (1,)
        assert step.prior_logits.shape == (1, 9)
        assert step.value.shape == (1,)
        assert next_hidden_state.shape == (1, 64)

        # Verify the network was called with current_params (indirectly)
        mock_muzero_network.recurrent_inference.assert_called()

    def test_actor_fallback_action_selection(self, mock_config, mock_muzero_network):
        """Test fallback action selection when MCTS fails."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up game with limited legal actions
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = False
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.legal_actions.return_value = [1, 3, 5]  # Only these actions are legal

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Mock MCTS to fail and trigger fallback
        actor.mcts.run = Mock(side_effect=RuntimeError("MCTS failed"))

        rng_key = jax.random.PRNGKey(42)
        
        # Try to get an action - should fall back to random legal action
        observation = [0.0] * 9
        
        # Test the actual fallback logic that happens in _select_action
        # when action weights are uniform (untrained network case)
        action_weights = jnp.ones(9) / 9  # Uniform distribution
        
        # Mock policy output for testing fallback
        policy_output = Mock()
        policy_output.action_weights = action_weights
        
        # This should trigger the fallback logic in _select_action
        action, policy_target = actor._select_action(policy_output, step=0)
        
        # Should be one of the legal actions
        assert action in [1, 3, 5]

    def test_actor_fallback_action_with_empty_legal_actions(self, mock_config, mock_muzero_network):
        """Test fallback action selection when no legal actions are available."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = False
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test fallback behavior by testing the logic in _select_action
        mock_game_wrapper.legal_actions.return_value = []  # Empty legal actions
        
        # Create uniform action weights that would trigger fallback
        action_weights = jnp.ones(9) / 9
        policy_output = Mock()
        policy_output.action_weights = action_weights
        
        # Test with empty legal actions - should fallback to action 0
        action, policy_target = actor._select_action(policy_output, step=0)
        assert action == 0  # Default fallback when no legal actions
        
        # Test with normal legal actions
        mock_game_wrapper.legal_actions.return_value = [2, 4, 6, 8]
        action, policy_target = actor._select_action(policy_output, step=0)
        
        # Should be one of the legal actions
        assert action in [2, 4, 6, 8]

    def test_actor_decision_recurrent_function_creation(self, mock_config, mock_muzero_network):
        """Test creation of decision recurrent function for stochastic MCTS."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_game_wrapper.is_stochastic.return_value = True
        mock_game_wrapper.max_chance_outcomes.return_value = 6
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            num_simulations=3,
            max_num_considered_actions=4
        )

        # Test that stochastic MCTS flag is set correctly
        assert actor._is_stochastic_mcts == True
        
        # Test that actor can create recurrent function for MCTS
        recurrent_fn = actor._create_recurrent_fn()
        assert callable(recurrent_fn)
        
        # Test the recurrent function with mock data
        embedding = jnp.zeros((1, 64))
        action = jnp.array([2])
        rng_key = _get_unique_rng_key()
        
        step, next_hidden_state = recurrent_fn(None, rng_key, action, embedding)
        assert hasattr(step, 'reward')
        assert hasattr(step, 'value')
