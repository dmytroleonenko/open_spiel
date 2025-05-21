from typing import Protocol, Tuple, Any
import chex
import dataclasses

@dataclasses.dataclass
class ModelOutput:
    hidden_state: chex.ArrayTree
    reward: chex.Array
    policy_logits: chex.Array
    value: chex.Array

class MuZeroModel(Protocol):
    """Protocol for a MuZero model usable by MCTS."""

    def initial_inference(
        self,
        observation: chex.ArrayTree,
        rng_key: chex.PRNGKey,
        training: bool = False
    ) -> ModelOutput:
        """Generates initial hidden state, value, policy logits, and reward."""
        ...

    def recurrent_inference(
        self,
        hidden_state: chex.ArrayTree,
        action: chex.Array,
        rng_key: chex.PRNGKey,
        training: bool = False
    ) -> ModelOutput:
        """Generates next hidden state, value, policy logits, and reward from current state and action."""
        ... 