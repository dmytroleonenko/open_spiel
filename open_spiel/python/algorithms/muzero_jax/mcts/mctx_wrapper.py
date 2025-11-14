import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, Any, TYPE_CHECKING

# Import MuZero components for type hints
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig
else:
    MuZeroConfig = Any

import mctx
from mctx._src import base as mctx_base


class MCTS:
    """
    A simple wrapper around DeepMind's mctx GumbelMuZero implementation.

    Attributes:
        num_simulations: number of MCTS simulations to run per root.
        max_num_considered_actions: number of actions to consider at root per simulation (for sequential halving).
        gumbel_scale: scale of the Gumbel noise.
    """
    def __init__(
        self,
        num_simulations: int,
        max_num_considered_actions: int,
        gumbel_scale: float = 1.0,
    ):
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions
        self.gumbel_scale = gumbel_scale

    def run(
        self,
        params,
        rng_key,
        root,
        recurrent_fn,
        invalid_actions=None,
        max_depth=None,
        loop_fn=jax.lax.fori_loop,
        qtransform=None,
    ):
        """
        Executes Gumbel MCTS using mctx.gumbel_muzero_policy.

        Args:
            params: model parameters passed to mctx policy.
            rng_key: JAX PRNGKey for stochastic operations.
            root: a RootFnOutput with fields (prior_logits, value, embedding).
            recurrent_fn: function to compute next-state predictions.
            invalid_actions: optional mask of invalid actions at root.
            max_depth: maximum tree depth for interior nodes (unused for root-only MCTS).
            loop_fn: loop function to drive simulations (defaults to jax.lax.fori_loop).
            qtransform: optional Q-transform function for mctx.

        Returns:
            A mctx PolicyOutput with fields (action, action_weights, search_tree).
        """
        # Build kwargs, only including qtransform if it's not None
        kwargs = {
            'params': params,
            'rng_key': rng_key,
            'root': root,
            'recurrent_fn': recurrent_fn,
            'num_simulations': self.num_simulations,
            'invalid_actions': invalid_actions,
            'max_depth': max_depth,
            'loop_fn': loop_fn,
            'max_num_considered_actions': self.max_num_considered_actions,
            'gumbel_scale': self.gumbel_scale,
        }
        if qtransform is not None:
            kwargs['qtransform'] = qtransform
            
        return mctx.gumbel_muzero_policy(**kwargs)


class StochasticMCTS:
    """
    A wrapper around mctx.stochastic_muzero_policy for stochastic environments.
    
    This class uses the official DeepMind mctx stochastic implementation.
    The stochastic_muzero_policy function expects a different API than our custom
    implementation - it uses a unified recurrent function and different root structure.
    
    Attributes:
        num_simulations: number of MCTS simulations to run per root.
        dirichlet_fraction: float from 0 to 1 for Dirichlet noise mixing.
        dirichlet_alpha: concentration parameter for Dirichlet distribution.
        pb_c_init: constant c_1 in the PUCT formula.
        pb_c_base: constant c_2 in the PUCT formula.
        temperature: temperature for action selection.
        is_stochastic: flag indicating this is a stochastic MCTS implementation
    """
    
    def __init__(
        self,
        num_simulations: int,
        max_num_considered_actions: int,  # Keep for compatibility
        gumbel_scale: float = 1.0,        # Keep for compatibility
        dirichlet_fraction: float = 0.25,
        dirichlet_alpha: float = 0.3,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652,
        temperature: float = 1.0,
    ):
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions  # For compatibility
        self.gumbel_scale = gumbel_scale  # For compatibility
        self.dirichlet_fraction = dirichlet_fraction
        self.dirichlet_alpha = dirichlet_alpha
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        self.temperature = temperature
        self.is_stochastic = True  # Flag for identification

    def run_stochastic(
        self,
        rng_key,
        root,
        network=None,  # Add network parameter for proper integration
        invalid_actions=None,
        max_depth=None,
        loop_fn=jax.lax.fori_loop,
        qtransform=None,
    ):
        """
        Executes Stochastic MCTS using mctx.stochastic_muzero_policy.
        
        This method uses the official mctx stochastic implementation with proper
        recurrent functions that handle both decision and chance nodes.

        Args:
            rng_key: JAX PRNGKey for stochastic operations.
            root: a RootFnOutput with fields (prior_logits, value, embedding).
            network: MuZero network for inference (required for stochastic environments).
            invalid_actions: optional mask of invalid actions at root.
            max_depth: maximum tree depth for interior nodes.
            loop_fn: loop function to drive simulations.
            qtransform: optional Q-transform function for mctx.

        Returns:
            A mctx PolicyOutput with fields (action, action_weights, search_tree).
        """
        if network is None:
            raise ValueError("StochasticMCTS requires a network instance for stochastic environments.")
        
        # Create decision and chance recurrent functions
        decision_recurrent_fn = self._create_decision_recurrent_fn(network)
        chance_recurrent_fn = self._create_chance_recurrent_fn(network)

        # Use the official stochastic_muzero_policy
        kwargs = {
            'params': None,  # No params needed for embedded NNX models
            'rng_key': rng_key,
            'root': root,
            'decision_recurrent_fn': decision_recurrent_fn,
            'chance_recurrent_fn': chance_recurrent_fn,
            'num_simulations': self.num_simulations,
            'invalid_actions': invalid_actions,
            'max_depth': max_depth,
            'loop_fn': loop_fn,
            'dirichlet_fraction': self.dirichlet_fraction,
            'dirichlet_alpha': self.dirichlet_alpha,
            'pb_c_init': self.pb_c_init,
            'pb_c_base': self.pb_c_base,
            'temperature': self.temperature,
        }
        if qtransform is not None:
            kwargs['qtransform'] = qtransform
            
        return mctx.stochastic_muzero_policy(**kwargs)
    
    def _create_decision_recurrent_fn(self, network):
        """
        Create decision recurrent function for expanding decision nodes.
        
        Args:
            network: MuZero network for inference
            
        Returns:
            Decision recurrent function compatible with mctx.stochastic_muzero_policy
        """
        def decision_recurrent_fn(params, rng_key, action, state_embedding):
            """
            Decision recurrent function for stochastic MCTS.
            
            This handles expanding decision nodes where the agent takes actions.
            """
            # Convert action to proper format
            action_array = jnp.array([action]) if jnp.isscalar(action) else action
            if action_array.ndim == 0:  # pragma: no cover - jnp.isscalar already guards scalars
                action_array = action_array[None]  # pragma: no cover
            
            # Use network recurrent inference for decision transitions
            next_hidden_state, reward, value, policy_logits, _, reward_hidden = network.recurrent_inference(
                state_embedding, action_array, training=False
            )
            
            # For decision nodes, we need to provide chance outcome probabilities
            # In many games, chance outcomes are uniform or game-specific
            batch_size = next_hidden_state.shape[0]
            
            # Create uniform chance logits (will be game-specific in real implementation)
            # For now, assume binary chance outcomes (common in many games)
            num_chance_outcomes = 2
            chance_logits = jnp.zeros((batch_size, num_chance_outcomes))  # Uniform distribution
            
            # Use predicted value as afterstate value
            afterstate_value = value
            
            from mctx._src.base import DecisionRecurrentFnOutput
            output = DecisionRecurrentFnOutput(
                chance_logits=chance_logits,
                afterstate_value=afterstate_value
            )
            
            return output, next_hidden_state
            
        return decision_recurrent_fn
    
    def _create_chance_recurrent_fn(self, network):
        """
        Create chance recurrent function for expanding chance nodes.
        
        Args:
            network: MuZero network for inference
            
        Returns:
            Chance recurrent function compatible with mctx.stochastic_muzero_policy
        """
        def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate_embedding):
            """
            Chance recurrent function for stochastic MCTS.
            
            This handles expanding chance nodes where the environment determines outcomes.
            """
            # For chance nodes, the environment determines the transition
            # The afterstate_embedding is already the result of the action
            # Now we need to apply the chance outcome
            
            batch_size = afterstate_embedding.shape[0]
            
            # In a real implementation, this would use game-specific chance mechanics
            # For now, assume chance doesn't significantly change the state
            state_embedding = afterstate_embedding
            
            # Get action logits for the new state using representation + prediction
            # Use a dummy zero action for the recurrent inference call
            dummy_action = jnp.zeros((batch_size,), dtype=jnp.int32)
            _, _, value, action_logits, _, reward_hidden = network.recurrent_inference(
                state_embedding, dummy_action, training=False
            )
            
            # Chance transitions typically don't provide immediate rewards
            reward = jnp.zeros((batch_size,))
            
            # Chance transitions don't end episodes (full discount)
            discount = jnp.ones((batch_size,))
            
            from mctx._src.base import ChanceRecurrentFnOutput
            output = ChanceRecurrentFnOutput(
                action_logits=action_logits,
                value=value,
                reward=reward,
                discount=discount
            )
            
            return output, state_embedding
            
        return chance_recurrent_fn


def create_mcts_for_game(
    game_wrapper,
    num_simulations: int,
    max_num_considered_actions: int, 
    gumbel_scale: float = 1.0,
    **kwargs  # Additional StochasticMCTS parameters
):
    """
    Factory function to create appropriate MCTS instance based on game type.
    
    Args:
        game_wrapper: The game wrapper instance to check for stochasticity
        num_simulations: number of MCTS simulations
        max_num_considered_actions: max actions to consider at root
        gumbel_scale: scale for Gumbel noise
        **kwargs: Additional parameters for StochasticMCTS (dirichlet_fraction, etc.)
    
    Returns:
        Either MCTS or StochasticMCTS instance based on game type
    """
    if hasattr(game_wrapper, 'is_stochastic') and game_wrapper.is_stochastic():
        return StochasticMCTS(
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
            **kwargs
        )
    else:
        return MCTS(
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
        )


def is_stochastic_mcts_instance(mcts_instance) -> bool:
    """
    Check if an MCTS instance is designed for stochastic environments.
    
    Args:
        mcts_instance: MCTS or StochasticMCTS instance
    
    Returns:
        True if instance is StochasticMCTS, False if regular MCTS
    """
    return isinstance(mcts_instance, StochasticMCTS)


"""
Stochastic MCTS implementation using mctx.

This module provides a wrapper around DeepMind's mctx library for handling
stochastic environments in MuZero. The implementation leverages mctx's official
stochastic_muzero_policy API for proper handling of chance nodes.

Dependencies:
    - mctx: DeepMind's JAX-based MCTS library
    - OpenSpiel: For game state management and stochastic environment detection

Key Features:
    - Automatic detection of stochastic vs deterministic games
    - Seamless integration with mctx.stochastic_muzero_policy
    - Fallback to deterministic MCTS when appropriate
    - Compatible with existing MuZero actor interface

Usage:
    ```python
    # Automatic selection based on game type
    mcts = create_mcts_for_game(game_wrapper, num_simulations=100)
    
    # Manual stochastic MCTS creation
    stochastic_mcts = StochasticMCTS(num_simulations=100)
    policy_output = stochastic_mcts.run_stochastic(
        rng_key=rng_key,
        root=root,
        network=muzero_network
    )
    ```

Architecture:
    - StochasticMCTS: Main class for stochastic environment handling
    - Decision/Chance recurrent functions: Interface with MuZero network
    - Factory functions: Automatic MCTS type selection
""" 
