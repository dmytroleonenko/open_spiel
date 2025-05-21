from typing import Protocol, Tuple, Any
import chex

class MuZeroModel(Protocol):
    """Protocol for a MuZero model usable by MCTS."""

    def initial_inference(
        self,
        observation: chex.ArrayTree,
        rng_key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, float, chex.Array, chex.ArrayTree]:
        """Generates initial hidden state, value, policy logits, and reward."""
        ...

    def recurrent_inference(
        self,
        hidden_state: chex.ArrayTree,
        action: chex.Array,
        rng_key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, float, chex.Array, chex.ArrayTree]:
        """Generates next hidden state, value, policy logits, and reward from current state and action."""
        ... 