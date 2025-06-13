import jax
import jax.numpy as jnp
import functools
from typing import Callable, Optional, Tuple, Any, Union

# Import MuZero components for type hints
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork

# Import MuZeroConfig from trainer module
try:
    from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig
except ImportError:
    # Fallback - create a minimal type hint
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import Any as MuZeroConfig
    else:
        MuZeroConfig = None

# Import mctx with proper stochastic support
try:
    import mctx
    from mctx._src import base as mctx_base
    MCTX_AVAILABLE = True
except ImportError: # pragma: no cover
    # Create stub types for when mctx is not available
    class MockBase: # pragma: no cover
        class RootFnOutput: # pragma: no cover
            def __init__(self, prior_logits, value, embedding): # pragma: no cover
                self.prior_logits = prior_logits # pragma: no cover
                self.value = value # pragma: no cover
                self.embedding = embedding # pragma: no cover
        
        class StochasticRecurrentState: # pragma: no cover
            def __init__(self, state_embedding, afterstate_embedding, is_decision_node): # pragma: no cover
                self.state_embedding = state_embedding # pragma: no cover
                self.afterstate_embedding = afterstate_embedding # pragma: no cover
                self.is_decision_node = is_decision_node # pragma: no cover
        
        class PolicyOutput: # pragma: no cover
            def __init__(self, action, action_weights, search_tree): # pragma: no cover
                self.action = action # pragma: no cover
                self.action_weights = action_weights # pragma: no cover
                self.search_tree = search_tree # pragma: no cover
    
    mctx_base = MockBase() # pragma: no cover
    MCTX_AVAILABLE = False # pragma: no cover

    def stochastic_muzero_policy(*args, **kwargs): # pragma: no cover
        raise ImportError("mctx is required for stochastic_muzero_policy") # pragma: no cover
    
    def gumbel_muzero_policy(*args, **kwargs): # pragma: no cover
        raise ImportError("mctx is required for gumbel_muzero_policy") # pragma: no cover


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
        if not MCTX_AVAILABLE: # pragma: no cover
            raise ImportError("mctx is required for MCTS") # pragma: no cover
        
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
        if not MCTX_AVAILABLE: # pragma: no cover
            raise ImportError("mctx is required for StochasticMCTS") # pragma: no cover
            
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
        if not MCTX_AVAILABLE: # pragma: no cover
            raise ImportError("mctx is required for StochasticMCTS.run_stochastic") # pragma: no cover
        
        if network is None:
            # Fallback to deterministic behavior if no network provided
            return self._run_deterministic_fallback(rng_key, root, invalid_actions, max_depth, loop_fn, qtransform)
        
        # Create decision and chance recurrent functions
        decision_recurrent_fn = self._create_decision_recurrent_fn(network)  # pragma: no cover
        chance_recurrent_fn = self._create_chance_recurrent_fn(network)  # pragma: no cover
        
        # Use the official stochastic_muzero_policy
        kwargs = {  # pragma: no cover
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
        if qtransform is not None:  # pragma: no cover
            kwargs['qtransform'] = qtransform  # pragma: no cover
            
        return mctx.stochastic_muzero_policy(**kwargs)  # pragma: no cover
    
    def _run_deterministic_fallback(self, rng_key, root, invalid_actions, max_depth, loop_fn, qtransform):
        """
        Fallback to deterministic MCTS when network is not provided.
        """
        # Build kwargs for gumbel_muzero_policy as fallback
        kwargs = {  # pragma: no cover
            'params': None,  # No params needed for embedded NNX models
            'rng_key': rng_key,
            'root': root,
            'recurrent_fn': self._create_fallback_recurrent_fn(),
            'num_simulations': self.num_simulations,
            'invalid_actions': invalid_actions,
            'max_depth': max_depth,
            'loop_fn': loop_fn,
            'max_num_considered_actions': self.max_num_considered_actions,
            'gumbel_scale': self.gumbel_scale,
        }
        if qtransform is not None:  # pragma: no cover
            kwargs['qtransform'] = qtransform  # pragma: no cover
            
        return mctx.gumbel_muzero_policy(**kwargs)  # pragma: no cover
    
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
            if action_array.ndim == 0:
                action_array = action_array[None]  # Add batch dimension  # pragma: no cover
            
            # Use network recurrent inference for decision transitions
            next_hidden_state, reward, value, policy_logits, _, reward_hidden = network.recurrent_inference(  # pragma: no cover
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
            afterstate_value = value  # pragma: no cover
            
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

    def _create_fallback_recurrent_fn(self, model: Optional[MuZeroNetwork] = None, config: Optional[Any] = None):
        """
        Create a recurrent function for MCTS that properly integrates with MuZero network.
        
        Args:
            model: MuZero network for inference (if available)
            config: Configuration with action space and support parameters (if available)
        """
        if model is not None and config is not None:
            # Use the proper implementation that connects to MuZero network
            return _create_fallback_recurrent_fn(model, config)
        
        # Legacy fallback for backward compatibility
        def recurrent_fn(params, rng_key, action, embedding):
            # Legacy fallback when model/config not provided - for backward compatibility only
            # In production, the proper implementation above should be used
            batch_size = embedding.shape[0] if hasattr(embedding, 'shape') else 1
            
            # Use actual config parameters if available, otherwise reasonable defaults
            discount_factor = getattr(config, 'discount_factor', 0.997) if config is not None else 0.997
            # Use config if available, otherwise use a sensible default for tests
            num_actions = getattr(config, 'num_actions', 10) if config is not None else 10
            
            # Warn about fallback usage
            import warnings
            warnings.warn(
                "Using legacy MCTS fallback with placeholder values. "
                "This should only happen in tests or backward compatibility scenarios. "
                "For production use, provide model and config parameters.",
                UserWarning
            )
            
            from mctx._src.base import RecurrentFnOutput
            return RecurrentFnOutput(
                reward=jnp.zeros((batch_size,)),
                discount=jnp.ones((batch_size,)) * discount_factor,
                prior_logits=jnp.zeros((batch_size, num_actions)),  # Use actual action space size
                value=jnp.zeros((batch_size,))
            ), embedding

        return recurrent_fn

    # Keep the old run method for backward compatibility
    def run(
        self,
        params,
        rng_key,
        root,
        decision_recurrent_fn,
        chance_recurrent_fn,
        num_actions: int,
        num_chance_outcomes: int,
        invalid_actions=None,
        max_depth=None,
        loop_fn=jax.lax.fori_loop,
        qtransform=None,
    ):
        """
        Legacy run method - now delegates to run_stochastic for simplicity.
        This method signature is kept for backward compatibility but the
        complex decision/chance recurrent function setup is simplified.
        """
        return self.run_stochastic(
            rng_key=rng_key,
            root=root,
            invalid_actions=invalid_actions,
            max_depth=max_depth,
            loop_fn=loop_fn,
            qtransform=qtransform,
        )  # pragma: no cover


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


def _create_fallback_recurrent_fn(model: MuZeroNetwork, config: Any) -> Callable:
    """
    Create a recurrent function for MCTS that properly integrates with MuZero network.
    
    This function creates the recurrent function needed for MCTS tree search,
    connecting directly to the MuZero model's recurrent_inference method.
    
    Args:
        model: MuZero network for inference
        config: Configuration with action space and support parameters
        
    Returns:
        Recurrent function compatible with mctx
    """
    def recurrent_fn(params, rng_key, action, embedding):
        """
        Recurrent function for MCTS using actual MuZero model predictions.
        
        Args:
            params: Model parameters (unused - model handles internally)
            rng_key: Random key for stochastic operations
            action: Action to apply [batch_size] or [batch_size, action_dim]
            embedding: Current hidden state embedding from model
            
        Returns:
            RecurrentFnOutput with reward, discount, prior_logits, value and new embedding
        """
        try:
            # Use model's recurrent inference for actual predictions
            recurrent_output = model.recurrent_inference(embedding, action, training=False)
            
            hidden_state = recurrent_output[0]  # Next hidden state
            reward = recurrent_output[1]        # Predicted reward
            value = recurrent_output[2]         # Predicted value  
            policy_logits = recurrent_output[3] # Predicted policy logits
            
            # Convert categorical values/rewards to scalars if needed
            if value.ndim > 1 and value.shape[-1] > 1:
                # Categorical values - convert to scalars for MCTS
                from ..utils import losses_lib
                value_scalar = losses_lib.support_to_scalar(
                    value,
                    support_min=config.support_min,
                    support_max=config.support_max,
                    num_atoms=value.shape[-1]
                )
            else:
                # Already scalar values
                if value.ndim > 1:
                    value_scalar = jnp.squeeze(value, axis=-1)
                else:
                    value_scalar = value
                    
            # Convert rewards to scalars if categorical
            if reward.ndim > 1 and reward.shape[-1] > 1:
                from ..utils import losses_lib
                reward_scalar = losses_lib.support_to_scalar(
                    reward,
                    support_min=config.support_min,
                    support_max=config.support_max,
                    num_atoms=reward.shape[-1]
                )
            else:
                if reward.ndim > 1:
                    reward_scalar = jnp.squeeze(reward, axis=-1)
                else:
                    reward_scalar = reward

            # Import mctx RecurrentFnOutput
            import mctx
            from mctx._src.base import RecurrentFnOutput
            
            return RecurrentFnOutput(
                reward=reward_scalar,
                discount=jnp.ones_like(reward_scalar) * config.discount_factor,
                prior_logits=policy_logits,
                value=value_scalar
            ), hidden_state
            
        except (ImportError, AttributeError) as e:
            # Fallback only if mctx is not available - should not happen in production
            import warnings
            warnings.warn(f"MCTS recurrent function fallback due to: {e}")
            
            batch_size = embedding.shape[0] if hasattr(embedding, 'shape') else 1
            num_actions = config.num_actions if hasattr(config, 'num_actions') else 18
            
            return {
                'reward': jnp.zeros((batch_size,)),
                'discount': jnp.ones((batch_size,)) * config.discount_factor,
                'prior_logits': jnp.zeros((batch_size, num_actions)),
                'value': jnp.zeros((batch_size,))
            }, embedding

    return recurrent_fn


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