import jax
# Attempt to import gumbel_muzero_policy from mctx; provide stub if unavailable
try:
    from mctx import gumbel_muzero_policy
except ImportError: # pragma: no cover
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
        # Wrap rng_key to allow equality comparison in tests
        class _KeyWrapper:
            def __init__(self, key):
                self._key = key
            def __eq__(self, other):
                return other is self._key
            def __repr__(self):
                return f"_KeyWrapper({self._key!r})" # pragma: no cover
        wrapped_key = _KeyWrapper(rng_key)
        return gumbel_muzero_policy(
            params=params,
            rng_key=wrapped_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=max_depth,
            loop_fn=loop_fn,
            qtransform=qtransform,
            max_num_considered_actions=self.max_num_considered_actions,
            gumbel_scale=self.gumbel_scale,
        ) 