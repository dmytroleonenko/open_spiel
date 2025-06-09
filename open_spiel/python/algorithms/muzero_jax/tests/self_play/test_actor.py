import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS, StochasticMCTS


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

        # Mock initial_inference
        def mock_initial_inference(observation, training=False):
            batch_size = observation.shape[0] if observation.ndim > 1 else 1
            hidden_state = jnp.zeros((batch_size, 64))
            policy_logits = jnp.zeros((batch_size, 9))
            value = jnp.zeros((batch_size,))
            reward = jnp.zeros((batch_size,))
            projection = None
            return hidden_state, policy_logits, value, reward, projection

        network.initial_inference = Mock(side_effect=mock_initial_inference)

        # Mock recurrent_inference
        def mock_recurrent_inference(hidden_state, action, training=False):
            batch_size = hidden_state.shape[0]
            next_hidden_state = jnp.zeros_like(hidden_state)
            policy_logits = jnp.zeros((batch_size, 9))
            value = jnp.zeros((batch_size,))
            reward = jnp.zeros((batch_size,))
            # Return in the order expected by actor.py: next_hidden_state, reward, value, policy_logits, _
            return next_hidden_state, reward, value, policy_logits, None

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
                discount_factor=1.5  # Invalid: > 1.0
            )

        # Test invalid n_step_return
        with pytest.raises(ValueError, match="n_step_return must be positive"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                n_step_return=0  # Invalid: must be positive
            )

        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                temperature=-0.5  # Invalid: negative
            )

        # Test invalid temperature_threshold
        with pytest.raises(ValueError, match="temperature_threshold must be non-negative"):
            Actor(
                network=mock_muzero_network,
                game_wrapper=mock_game_wrapper,
                replay_buffer=mock_replay_buffer,
                config=mock_config,
                temperature_threshold=-1  # Invalid: negative
            )

        # Test valid parameters
        actor = Actor(
            network=mock_muzero_network,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            discount_factor=0.99,
            n_step_return=5,
            temperature=1.0,
            temperature_threshold=30
        )
        assert actor.discount_factor == 0.99
        assert actor.n_step_return == 5
        assert actor.temperature == 1.0
        assert actor.temperature_threshold == 30

    def test_actor_checkpoint_loading(self, mock_config, mock_muzero_network):
        """Test checkpoint loading functionality."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        checkpoint_dir = "/path/to/checkpoints"

        # Test when no checkpoint exists
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest:
            mock_get_latest.return_value = None
            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is False

        # Test when checkpoint exists and loads successfully
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_params = {'test': 'params'}
            mock_load_params.return_value = mock_params

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is True
            mock_load_params.assert_called_once_with(
                "/path/to/checkpoints/latest.ckpt")
            # Check that parameters were stored
            assert actor.current_params == mock_params

        # Test when checkpoint exists but loading fails
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_load_params.side_effect = Exception("Load failed")

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is False

    def test_actor_action_selection_with_temperature(self, mock_config, mock_muzero_network):
        """Test action selection with different temperature values."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            temperature=0.0,  # No temperature - should select argmax
            temperature_threshold=30
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Test with zero action weights (should return uniform distribution)
        action_weights = jnp.zeros(9)
        policy_target = actor._compute_policy_target(action_weights)

        expected_uniform = jnp.ones(9) / 9
        assert jnp.allclose(policy_target, expected_uniform)

    def test_actor_value_target_computation(self, mock_config, mock_muzero_network):
        """Test that value targets are computed correctly using n-step returns."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config,
            n_step_return=3,
            discount_factor=0.99
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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
        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
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

        # Set up MCTS mock
        def mock_mcts_run(params, rng_key, root, recurrent_fn, **kwargs):
            policy_output = Mock()
            policy_output.action = jnp.array([0])  # Always choose action 0
            policy_output.action_weights = jnp.array(
                [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return policy_output

        mock_mcts.run = Mock(side_effect=mock_mcts_run)

        # Create actor
        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

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
        assert mock_mcts.run.call_count == 2
        assert mock_muzero_network.initial_inference.call_count == 2

    def test_actor_episode_with_chance_nodes(self, mock_config, mock_muzero_network):
        """Test that the actor handles games with chance nodes correctly."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
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
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)

        # Should handle chance nodes and continue
        assert 'observations' in trajectory
        mock_game_wrapper.chance_outcomes.assert_called()

    def test_actor_episode_early_termination(self, mock_config, mock_muzero_network):
        """Test handling of early episode termination."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up immediate termination
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = True  # Immediate termination
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.return_value = [0.0] * 9

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up for quick episodes (immediate termination)
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = True  # Immediate termination
        mock_game_wrapper.is_chance_node.return_value = False

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Set up game wrapper to return empty observation
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.return_value = False  # Not terminal initially
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.return_value = None  # Empty observation

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
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

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        self._setup_game_wrapper_mock(mock_game_wrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Set up loaded parameters
        test_params = {'test': 'loaded_params'}
        actor.current_params = test_params

        # Set up game wrapper for a simple episode
        mock_game_wrapper.reset.return_value = [0.0] * 9
        mock_game_wrapper.is_terminal.side_effect = [
            False, True]  # One step then terminal
        mock_game_wrapper.is_chance_node.return_value = False
        mock_game_wrapper.current_observation.return_value = [0.0] * 9
        mock_game_wrapper.legal_actions.return_value = [
            0, 1, 2]  # Add proper legal actions mock
        mock_game_wrapper.step.return_value = ([1.0] + [0.0] * 8, [0.0], False)

        # Mock MCTS to capture the parameters passed to it
        captured_params = None

        def capture_mcts_params(params, rng_key, root, recurrent_fn, **kwargs):
            # Store the params that were passed to MCTS
            nonlocal captured_params
            captured_params = params
            # Return mock policy output
            mock_output = Mock()
            mock_output.action_weights = jnp.array([1.0] + [0.0] * 8)
            mock_output.search_tree = Mock()
            mock_output.search_tree.node_values = jnp.array([0.5])
            return mock_output

        mock_mcts.run.side_effect = capture_mcts_params

        # Play an episode
        rng_key = jax.random.PRNGKey(42)
        trajectory = actor.play_episode(rng_key)

        # Verify that the loaded parameters were passed to MCTS
        assert captured_params == test_params

    def test_actor_parameter_loading_branches(self, mock_config, mock_muzero_network):
        """Test different branches in parameter loading logic."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        checkpoint_dir = "/path/to/checkpoints"

        # Test the 'network_state' key branch (line 122)
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_params = {'network_state': {'test': 'network_state_params'}}
            mock_load_params.return_value = mock_params

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is True
            assert actor.current_params == {'test': 'network_state_params'}

        # Test the 'params' key branch (line 124)
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            mock_params = {'params': {'test': 'params_key'}}
            mock_load_params.return_value = mock_params

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is True
            assert actor.current_params == {'test': 'params_key'}

        # Test the direct assignment branch (else clause)
        with patch('open_spiel.python.algorithms.muzero_jax.self_play.actor.get_latest_checkpoint') as mock_get_latest, \
                patch.object(actor, 'load_network_parameters') as mock_load_params:

            mock_get_latest.return_value = "/path/to/checkpoints/latest.ckpt"
            # No 'network_state' or 'params' key
            mock_params = {'direct': 'params'}
            mock_load_params.return_value = mock_params

            result = actor.maybe_load_latest_parameters(checkpoint_dir)
            assert result is True
            assert actor.current_params == {'direct': 'params'}

    def test_actor_recurrent_function_with_params(self, mock_config, mock_muzero_network):
        """Test recurrent function when params is not None (covers line 149)."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Create the recurrent function
        recurrent_fn = actor._create_recurrent_fn()

        # Test with non-None params (should hit the pass statement on line 149)
        test_params = {'test': 'params'}
        rng_key = jax.random.PRNGKey(42)
        action = 0
        embedding = jnp.zeros((1, 64))

        # Call the recurrent function with params
        step, next_embedding = recurrent_fn(
            test_params, rng_key, action, embedding)

        # Verify the function executed without error and returned expected structure
        assert hasattr(step, 'reward')
        assert hasattr(step, 'discount')
        assert hasattr(step, 'prior_logits')
        assert hasattr(step, 'value')
        assert next_embedding is not None

    def test_actor_episode_max_steps_exceeded(self, mock_config, mock_muzero_network):
        """Test that actor stops episodes when max steps are exceeded."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Configure game wrapper to never end naturally
        mock_game_wrapper.reset.return_value = jnp.zeros(9)
        mock_game_wrapper.legal_actions.return_value = [0, 1, 2]
        mock_game_wrapper.is_terminal.return_value = False  # Never terminal
        mock_game_wrapper.current_observation.return_value = jnp.zeros(9)
        mock_game_wrapper.is_chance_node.return_value = False

        # Mock the _game attribute that actor tries to access
        mock_game = Mock()
        # Set to match expected test behavior
        mock_game.max_game_length.return_value = 3
        mock_game.get_type.return_value.short_name = "test_game"
        mock_game_wrapper._game = mock_game

        # Mock step to always return the same state
        def mock_step(action):
            return jnp.zeros(9), [0.0], False
        mock_game_wrapper.step = Mock(side_effect=mock_step)

        # Configure MCTS to always return action 0
        def mock_mcts_run(params, rng_key, root, recurrent_fn, num_simulations=None, **kwargs):
            mock_result = Mock()
            mock_result.action_weights = jnp.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            mock_result.search_tree = Mock()
            mock_result.search_tree.qvalues = jnp.array(
                [0.0, 0.0, 0.0])  # One Q-value per legal action
            return mock_result

        mock_mcts.run = Mock(side_effect=mock_mcts_run)

        # Set a very low max_steps to trigger early termination
        mock_config.max_episode_steps = 3

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Play episode - should terminate due to max steps
        rng_key = jax.random.PRNGKey(42)
        episode_data = actor.play_episode(rng_key)

        # Should have exactly max_episode_steps + 1 observations (initial + max_steps)
        assert len(episode_data['observations']
                   ) == mock_config.max_episode_steps + 1

    def test_actor_fallback_action_selection(self, mock_config, mock_muzero_network):
        """Test fallback action selection when invalid action is chosen."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Configure game wrapper to return legal actions [1, 2, 3] (excluding 0)
        mock_game_wrapper.legal_actions.return_value = [1, 2, 3]

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Create a mock policy output that would select action 0 (invalid)
        mock_policy_output = Mock()
        mock_policy_output.action_weights = jnp.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Mock the logger to capture the fallback message
        with patch.object(actor, 'logger') as mock_logger:
            # Call _select_action directly to test the fallback logic
            action, policy_target = actor._select_action(
                mock_policy_output, step=0)

            # Should have used the first legal action (1) as fallback
            assert action == 1  # First legal action from [1, 2, 3]

            # Should have logged the fallback usage
            mock_logger.info.assert_called_with(
                "Step 0: Using fallback legal action: 1")

    def test_actor_fallback_action_with_empty_legal_actions(self, mock_config, mock_muzero_network):
        """Test fallback action selection when no legal actions are available."""
        from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor

        mock_mcts = Mock(spec=MCTS)
        mock_game_wrapper = Mock(spec=GameWrapper)
        mock_replay_buffer = Mock(spec=TrajectoryBuffer)

        # Configure game wrapper with empty legal actions (edge case)
        mock_game_wrapper.legal_actions.return_value = []  # No legal actions

        actor = Actor(
            network=mock_muzero_network,
            mcts=mock_mcts,
            game_wrapper=mock_game_wrapper,
            replay_buffer=mock_replay_buffer,
            config=mock_config
        )

        # Create a mock policy output that would select action 0
        mock_policy_output = Mock()
        mock_policy_output.action_weights = jnp.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Mock the logger to capture the fallback message
        with patch.object(actor, 'logger') as mock_logger:
            # Call _select_action directly to test the fallback logic
            action, policy_target = actor._select_action(
                mock_policy_output, step=0)

            # Should have used action 0 as fallback when no legal actions available
            assert action == 0

            # Should have logged the fallback usage
            mock_logger.info.assert_called_with(
                "Step 0: Using fallback legal action: 0")
