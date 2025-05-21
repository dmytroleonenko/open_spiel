from typing import Protocol, Tuple, Optional
import chex
import jax.numpy as jnp

class MuZeroModel(Protocol):
  """Protocol for a MuZero-like model usable by MCTS."""

  def initial_inference(
      self,
      observation: chex.Array,
      training: bool
  ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """h(observation) -> hidden_state, reward, policy_logits, value."""
    ...

  def recurrent_inference(
      self,
      hidden_state: chex.Array,
      action: chex.Array,
      training: bool
  ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """g(hidden_state, action) -> next_hidden_state, reward, policy_logits, value."""
    ...

  def prediction(
        self,
        hidden_state: chex.Array,
        training: bool
    ) -> Tuple[chex.Array, chex.Array]:
      """f(hidden_state) -> policy_logits, value"""
      ...

  # Optional, for EfficientZeroV2 style full model that combines all three
  # If not present, MCTS will call initial_inference and recurrent_inference
  # via the main network object which should then dispatch to its components.
  # For now, the MCTS expects prediction() and recurrent_inference() mainly.
  # The main MuZeroNetwork in network.py provides these. 