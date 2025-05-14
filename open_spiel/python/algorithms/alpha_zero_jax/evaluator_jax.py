import numpy as np
import jax
import jax.numpy as jnp # For jax.nn.softmax, though jax.nn.softmax is also fine
import flax.linen as nn # For type hinting model
from open_spiel.python.algorithms import mcts
import pyspiel

class AlphaZeroEvaluatorJAX(mcts.Evaluator):
  """A JAX-based AlphaZero MCTS Evaluator."""

  def __init__(self, game: pyspiel.Game, model: nn.Module, variables: dict):
    """Initializes the JAX AlphaZero evaluator.

    Args:
      game: The game to evaluate.
      model: The JAX/Flax model to use for inference.
      variables: The model parameters and other variables (like batch stats).
    """
    if game is None:
      raise ValueError("Game cannot be None.")
    if model is None:
      raise ValueError("Model cannot be None.")
    if variables is None:
      raise ValueError("Variables cannot be None.")

    super().__init__()
    self._game = game
    self._model = model
    self.variables = variables # Allow variables to be updated from outside
    # No explicit cache implemented in this JAX version yet.
    # If a cache is added later (e.g., LRU cache), it should be initialized here.

  def update_variables(self, new_variables: dict):
    """Updates the model variables for the evaluator."""
    self.variables = new_variables
    # If a cache dependent on model variables exists, it should be cleared here.
    self.clear_cache() 

  def clear_cache(self):
    """Clears any internal cache. Placeholder for now."""
    # Currently, no explicit Python-level caching is implemented in this evaluator.
    # JAX's JIT compilation might have its own caching, but that's not controlled here.
    # If an LRU cache or similar is added, this method would clear it.
    pass # No cache to clear yet

  def cache_info(self) -> str:
    """Returns information about the cache. Placeholder for now."""
    # Returns a string describing the cache state (e.g., size, hits, misses).
    return "Cache: Not implemented in JAX evaluator yet."

  def _inference(self, state: pyspiel.State) -> tuple[np.ndarray, np.ndarray]:
    """Performs a single model inference on the state."""
    # Ensure observation_tensor and legal_actions_mask are numpy arrays
    obs_tensor_list = state.observation_tensor()
    obs_tensor = np.array(obs_tensor_list, dtype=np.float32)
    obs_tensor = np.expand_dims(obs_tensor, 0) # Add batch dimension

    legal_actions_mask_list = state.legal_actions_mask()
    legal_actions_mask = np.array(legal_actions_mask_list, dtype=bool)
    legal_actions_mask = np.expand_dims(legal_actions_mask, 0) # Add batch dimension

    # apply expects JAX arrays if inputs are to be traced for JIT, etc.
    # However, for pure inference, numpy arrays are often fine and auto-converted.
    # For consistency and to avoid potential issues, explicit conversion is safer.
    # obs_tensor_jax = jnp.asarray(obs_tensor)
    # Note: self.model.apply can take JAX or NumPy arrays as input for `obs_tensor`.
    # Flax modules handle this conversion internally. Sending NumPy arrays is fine here.

    policy_logits_batch, value_output_batch = self._model.apply(
        self.variables, obs_tensor, training=False)

    # Convert JAX arrays to NumPy arrays for further processing with NumPy/OpenSpiel
    value_scalar = np.array(value_output_batch[0, 0])
    policy_logits = np.array(policy_logits_batch[0])

    # Softmax and masking
    policy_probs = jax.nn.softmax(policy_logits) # policy_logits is now a np array, jax.nn.softmax handles it
    policy_probs = np.array(policy_probs) # Ensure it's a numpy array after softmax

    # Ensure legal_actions_mask[0] is boolean for multiplication, though it should be
    masked_policy = policy_probs * legal_actions_mask[0].astype(np.float32)
    
    # Normalize the masked policy
    policy_sum = np.sum(masked_policy)
    if policy_sum > 0:
      masked_policy /= policy_sum
    else:
      # This case should ideally not happen if there's at least one legal action
      # and the policy gives non-zero probability to it after softmax.
      # If it does, it might indicate an issue (e.g. all legal actions have zero logits -> uniform small probs,
      # or numerical instability).
      # Fallback: use a uniform distribution over legal actions.
      # print("Warning: sum of masked policy is zero. Falling back to uniform.", flush=True)
      num_legal_actions = np.sum(legal_actions_mask[0])
      if num_legal_actions > 0:
        masked_policy = legal_actions_mask[0].astype(np.float32) / num_legal_actions
      # If num_legal_actions is 0 (shouldn't happen in a non-terminal state), masked_policy remains all zeros.

    return value_scalar, masked_policy

  def evaluate(self, state: pyspiel.State) -> np.ndarray:
    """Returns a value for the given state.

    The value is from the perspective of the current player in the state.
    For a two-player zero-sum game, this is usually [v, -v].
    """
    value, _ = self._inference(state)
    # Assuming a two-player zero-sum game
    return np.array([value, -value])

  def prior(self, state: pyspiel.State) -> list[tuple[pyspiel.Action, float]]:
    """Returns a probability distribution over legal actions.

    The distribution is represented as a list of (action, probability) pairs.
    For chance nodes, it returns the probability distribution defined by the game.
    """
    if state.is_chance_node():
      return state.chance_outcomes()
    
    if state.is_terminal():
        return [] # No legal actions from a terminal state

    legal_actions = state.legal_actions()
    if not legal_actions:
        # This can happen if a state is incorrectly considered non-terminal but has no legal actions.
        # Or if the MCTS calls prior on a terminal state (which it shouldn't for action selection).
        return []

    _, policy = self._inference(state)

    return [(action, policy[action]) for action in legal_actions] 