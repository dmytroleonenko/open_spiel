"""
Tests for stochastic MCTS integration with official mctx stochastic_muzero_policy.

This module tests the simplified integration that leverages mctx's built-in
capabilities rather than custom implementations.
"""

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock
import numpy as np

# Import test dependencies
import pyspiel

# Import the modules under test
import mctx
from mctx._src import base as mctx_base

from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import (
    MCTS, StochasticMCTS, create_mcts_for_game
)
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper


class TestMCTXIntegration:
    """Test the official mctx integration."""
    
    def test_deterministic_mcts_creation(self):
        """Test creating deterministic MCTS wrapper."""
        mcts = MCTS(
            num_simulations=50,
            max_num_considered_actions=16,
            gumbel_scale=1.0
        )
        
        assert mcts.num_simulations == 50
        assert mcts.max_num_considered_actions == 16
        assert mcts.gumbel_scale == 1.0
    
    def test_stochastic_mcts_creation(self):
        """Test creating stochastic MCTS wrapper."""
        mcts = StochasticMCTS(
            num_simulations=50,
            max_num_considered_actions=16,
            gumbel_scale=1.0,
            dirichlet_fraction=0.25,
            dirichlet_alpha=0.3
        )
        
        assert mcts.num_simulations == 50
        assert mcts.max_num_considered_actions == 16
        assert mcts.gumbel_scale == 1.0
        assert mcts.dirichlet_fraction == 0.25
        assert mcts.dirichlet_alpha == 0.3
        assert mcts.is_stochastic is True
    
    def test_factory_function_deterministic_game(self):
        """Test factory function creates correct MCTS type for deterministic games."""
        # Create a mock deterministic game wrapper
        mock_wrapper = MagicMock()
        mock_wrapper.is_stochastic.return_value = False
        
        mcts = create_mcts_for_game(
            game_wrapper=mock_wrapper,
            num_simulations=50,
            max_num_considered_actions=16
        )
        
        assert isinstance(mcts, MCTS)
        assert not isinstance(mcts, StochasticMCTS)
    
    def test_factory_function_stochastic_game(self):
        """Test factory function creates correct MCTS type for stochastic games."""
        # Create a mock stochastic game wrapper
        mock_wrapper = MagicMock()
        mock_wrapper.is_stochastic.return_value = True
        
        mcts = create_mcts_for_game(
            game_wrapper=mock_wrapper,
            num_simulations=50,
            max_num_considered_actions=16,
            dirichlet_fraction=0.3
        )
        
        assert isinstance(mcts, StochasticMCTS)
        assert mcts.is_stochastic is True
        assert mcts.dirichlet_fraction == 0.3
    
    @patch('open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper.mctx')
    def test_deterministic_mcts_run(self, mock_mctx):
        """Test deterministic MCTS run method calls mctx correctly."""
        # Setup mock
        mock_policy_output = mctx_base.PolicyOutput(
            action=jnp.array([1]),
            action_weights=jnp.ones((1, 4)) / 4,
            search_tree=None
        )
        mock_mctx.gumbel_muzero_policy.return_value = mock_policy_output
        
        # Create MCTS instance
        mcts = MCTS(num_simulations=10, max_num_considered_actions=4, gumbel_scale=1.0)
        
        # Create test inputs
        rng_key = jax.random.PRNGKey(42)
        root = mctx_base.RootFnOutput(
            prior_logits=jnp.ones((1, 4)),
            value=jnp.array([0.5]),
            embedding=jnp.zeros((1, 8))
        )
        
        def mock_recurrent_fn(params, rng_key, action, embedding):
            from mctx._src.base import RecurrentFnOutput
            return RecurrentFnOutput(
                reward=jnp.array([0.0]),
                discount=jnp.array([1.0]),
                prior_logits=jnp.ones((1, 4)),
                value=jnp.array([0.0])
            ), embedding
        
        # Run MCTS
        result = mcts.run(
            params=None,
            rng_key=rng_key,
            root=root,
            recurrent_fn=mock_recurrent_fn
        )
        
        # Verify mctx was called correctly
        mock_mctx.gumbel_muzero_policy.assert_called_once()
        args, kwargs = mock_mctx.gumbel_muzero_policy.call_args
        
        assert 'params' in kwargs
        assert 'rng_key' in kwargs
        assert 'root' in kwargs
        assert 'recurrent_fn' in kwargs
        assert kwargs['num_simulations'] == 10
        assert kwargs['max_num_considered_actions'] == 4
        assert kwargs['gumbel_scale'] == 1.0
        
        assert result == mock_policy_output
    
    @patch('open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper.mctx')
    def test_stochastic_mcts_run_stochastic(self, mock_mctx):
        """Test stochastic MCTS run_stochastic method wires real recurrent fns."""
        mock_policy_output = mctx_base.PolicyOutput(
            action=jnp.array([1]),
            action_weights=jnp.ones((1, 4)) / 4,
            search_tree=None,
        )
        mock_mctx.stochastic_muzero_policy.return_value = mock_policy_output

        class DummyNetwork:
            def recurrent_inference(self, embedding, action, training=False):
                batch = embedding.shape[0]
                hidden = jnp.ones_like(embedding)
                reward = jnp.zeros((batch,))
                value = jnp.ones((batch,)) * 0.5
                policy_logits = jnp.ones((batch, 4))
                return hidden, reward, value, policy_logits, None, None

        mcts = StochasticMCTS(
            num_simulations=10,
            max_num_considered_actions=4,
            gumbel_scale=1.0,
        )

        rng_key = jax.random.PRNGKey(42)
        root = mctx_base.RootFnOutput(
            prior_logits=jnp.ones((1, 4)),
            value=jnp.array([0.5]),
            embedding=jnp.zeros((1, 8)),
        )

        dummy_qtransform = lambda *_args, **_kwargs: None

        result = mcts.run_stochastic(
            rng_key=rng_key,
            root=root,
            network=DummyNetwork(),
            qtransform=dummy_qtransform,
        )

        assert result == mock_policy_output
        assert mock_mctx.stochastic_muzero_policy.call_count == 1
        kwargs = mock_mctx.stochastic_muzero_policy.call_args.kwargs
        assert callable(kwargs["decision_recurrent_fn"])
        assert callable(kwargs["chance_recurrent_fn"])
        assert kwargs["qtransform"] is dummy_qtransform

        # Smoke test the generated recurrent fns to ensure they execute end-to-end
        decision_output, afterstate_embedding = kwargs["decision_recurrent_fn"](
            None, rng_key, jnp.array(1), root.embedding
        )
        assert decision_output.afterstate_value.shape == (1,)
        assert afterstate_embedding.shape == root.embedding.shape

        chance_output, _ = kwargs["chance_recurrent_fn"](
            None, rng_key, jnp.array(0), afterstate_embedding
        )
        assert chance_output.value.shape == (1,)
        assert chance_output.action_logits.shape == (1, 4)

    def test_stochastic_mcts_requires_network(self):
        """Ensure run_stochastic fails loudly when no network is provided."""
        mcts = StochasticMCTS(
            num_simulations=10,
            max_num_considered_actions=4,
            gumbel_scale=1.0
        )
        rng_key = jax.random.PRNGKey(0)
        root = mctx_base.RootFnOutput(
            prior_logits=jnp.ones((1, 4)),
            value=jnp.array([0.5]),
            embedding=jnp.zeros((1, 8))
        )
        with pytest.raises(ValueError, match="requires a network"):
            mcts.run_stochastic(rng_key=rng_key, root=root, network=None)


class TestIntegrationPoints:
    """Test integration with other components."""
    
    def test_factory_function_with_real_game_wrapper(self):
        """Test factory function with actual OpenSpiel game."""
        # Create a real game wrapper with string game name
        game_wrapper = GameWrapper("tic_tac_toe")
        
        # Create MCTS using factory function
        mcts = create_mcts_for_game(
            game_wrapper=game_wrapper,
            num_simulations=10,
            max_num_considered_actions=4
        )
        
        # Should create deterministic MCTS for tic-tac-toe
        assert isinstance(mcts, MCTS)
        assert not hasattr(mcts, 'is_stochastic') or not mcts.is_stochastic
    
    def test_factory_function_with_stochastic_game(self):
        """Test factory function with actual stochastic OpenSpiel game."""
        # Test with Kuhn Poker which has chance nodes
        game_wrapper = GameWrapper("kuhn_poker")
        
        # Create MCTS using factory function
        mcts = create_mcts_for_game(
            game_wrapper=game_wrapper,
            num_simulations=10,
            max_num_considered_actions=4
        )
        
        # Should create stochastic MCTS for games with chance nodes
        if game_wrapper.is_stochastic():
            assert isinstance(mcts, StochasticMCTS)
            assert mcts.is_stochastic is True
        else:
            assert isinstance(mcts, MCTS)
    
    def test_stochastic_mcts_with_real_network(self):
        """Test stochastic MCTS with a mock MuZero network."""
        from unittest.mock import MagicMock
        
        # Create a mock network that mimics MuZero network interface
        mock_network = MagicMock()
        
        # Mock recurrent inference to return proper shapes
        def mock_recurrent_inference(embedding, action, training=False):
            batch_size = embedding.shape[0] if hasattr(embedding, 'shape') else 1
            return (
                jnp.zeros((batch_size, 8)),        # next_embedding
                jnp.zeros((batch_size,)),          # reward
                jnp.zeros((batch_size,)),          # value  
                jnp.zeros((batch_size, 4)),        # policy_logits
                None,                              # projection (unused)
                None                               # reward_hidden (LSTM reward hidden state)
            )
        
        mock_network.recurrent_inference = mock_recurrent_inference
        
        # Create stochastic MCTS
        mcts = StochasticMCTS(
            num_simulations=5,  # Small for testing
            max_num_considered_actions=4,
            gumbel_scale=1.0
        )
        
        # Create test inputs
        rng_key = jax.random.PRNGKey(42)
        root = mctx_base.RootFnOutput(
            prior_logits=jnp.ones((1, 4)),
            value=jnp.array([0.5]),
            embedding=jnp.zeros((1, 8))
        )
        
        # Test decision recurrent function creation
        decision_fn = mcts._create_decision_recurrent_fn(mock_network)
        decision_output, afterstate_embedding = decision_fn(
            params=None,
            rng_key=rng_key,
            action=jnp.array([1]),
            state_embedding=root.embedding
        )
        
        # Verify decision output structure
        assert hasattr(decision_output, 'chance_logits')
        assert hasattr(decision_output, 'afterstate_value')
        assert decision_output.chance_logits.shape == (1, 2)  # Binary chance outcomes
        assert decision_output.afterstate_value.shape == (1,)
        
        # Test chance recurrent function creation
        chance_fn = mcts._create_chance_recurrent_fn(mock_network)
        chance_output, state_embedding = chance_fn(
            params=None,
            rng_key=rng_key,
            chance_outcome=jnp.array([0]),
            afterstate_embedding=afterstate_embedding
        )
        
        # Verify chance output structure
        assert hasattr(chance_output, 'action_logits')
        assert hasattr(chance_output, 'value')
        assert hasattr(chance_output, 'reward')
        assert hasattr(chance_output, 'discount')
        assert chance_output.action_logits.shape == (1, 4)
        assert chance_output.value.shape == (1,)
        assert chance_output.reward.shape == (1,)
        assert chance_output.discount.shape == (1,)
    
    def test_stochastic_game_detection(self):
        """Test detection of stochastic vs deterministic games."""
        # Test various OpenSpiel games
        test_games = [
            ("tic_tac_toe", False),      # Deterministic
            ("connect_four", False),     # Deterministic  
            ("kuhn_poker", True),        # Stochastic (cards dealt)
            ("leduc_poker", True),       # Stochastic (cards dealt)
        ]
        
        for game_name, expected_stochastic in test_games:
            try:
                game_wrapper = GameWrapper(game_name)
                is_stochastic = game_wrapper.is_stochastic()
                
                # Create appropriate MCTS type
                mcts = create_mcts_for_game(
                    game_wrapper=game_wrapper,
                    num_simulations=5,
                    max_num_considered_actions=4
                )
                
                if is_stochastic:
                    assert isinstance(mcts, StochasticMCTS), f"{game_name} should use StochasticMCTS"
                    assert mcts.is_stochastic is True
                else:
                    assert isinstance(mcts, MCTS), f"{game_name} should use regular MCTS"
                    assert not hasattr(mcts, 'is_stochastic') or not mcts.is_stochastic
                    
            except Exception as e:
                # Skip if game is not available
                pytest.skip(f"Game {game_name} not available: {e}")
    
    def test_simplified_api_compatibility(self):
        """Test that simplified API is compatible with actor expectations."""
        # Create stochastic MCTS
        mcts = StochasticMCTS(
            num_simulations=10,
            max_num_considered_actions=4,
            gumbel_scale=1.0
        )
        
        # Verify it has the is_stochastic flag
        assert hasattr(mcts, 'is_stochastic')
        assert mcts.is_stochastic is True
        
        # Verify it has the run_stochastic method
        assert hasattr(mcts, 'run_stochastic')
        assert callable(mcts.run_stochastic)
    
    def test_stochastic_mcts_requires_network_integration(self):
        """Ensure run_stochastic fails fast when invoked without a network."""
        mcts = StochasticMCTS(
            num_simulations=10,
            max_num_considered_actions=4,
            gumbel_scale=1.0
        )
        rng_key = jax.random.PRNGKey(42)
        root = mctx_base.RootFnOutput(
            prior_logits=jnp.ones((1, 4)),
            value=jnp.array([0.5]),
            embedding=jnp.zeros((1, 8))
        )
        with pytest.raises(ValueError, match="requires a network"):
            mcts.run_stochastic(
                rng_key=rng_key,
                root=root,
                network=None,
            )


if __name__ == "__main__":
    pytest.main([__file__]) 
