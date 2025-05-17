import numpy as np
import jax
import jax.numpy as jnp # For jax.nn.softmax, though jax.nn.softmax is also fine
import flax.linen as nn # For type hinting model
from open_spiel.python.algorithms import mcts
import pyspiel
from open_spiel.python.utils import lru_cache # Added for LRUCache

class AlphaZeroEvaluatorJAX(mcts.Evaluator):
  """A JAX-based AlphaZero MCTS Evaluator."""

  def __init__(self, game: pyspiel.Game, model: nn.Module, variables: dict, cache_size: int = 2**16, inference_batch_size: int = 1):
    """Initializes the JAX AlphaZero evaluator.

    Args:
      game: The game to evaluate.
      model: The JAX/Flax model to use for inference.
      variables: The model parameters and other variables (like batch stats).
      cache_size: The size of the LRU cache for inferences.
      inference_batch_size: The batch size for model inference.
    """
    if game is None:
      raise ValueError("Game cannot be None.")
    if model is None:
      raise ValueError("Model cannot be None.")
    if variables is None:
      raise ValueError("Variables cannot be None.")

    # Game-type validations
    if game.num_players() != 2:
      raise ValueError(
          f"Game must be a 2-player game, got {game.num_players()}")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError(
          f"Game must have terminal rewards, got {game_type.reward_model}")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError(
          f"Game must have sequential dynamics, got {game_type.dynamics}")

    super().__init__()
    self._game = game
    self._model = model
    self.variables = variables # Allow variables to be updated from outside
    self._cache = lru_cache.LRUCache(cache_size)
    self.inference_batch_size = inference_batch_size # Store inference_batch_size

    # Define and JIT-compile the model application function
    # We capture self._model.apply by passing it as an argument to the static function
    @jax.jit
    def _jit_apply_model(model_apply_fn, current_variables, current_obs_tensor, current_legals_mask):
        return model_apply_fn(current_variables, current_obs_tensor, current_legals_mask, training=False)
    
    self._jit_apply_model_fn = jax.jit(
        lambda variables, obs, legals: self._model.apply(variables, obs, training=False, legals_mask=legals)
    )

  def update_variables(self, new_variables: dict):
    """Updates the model variables for the evaluator."""
    self.variables = new_variables
    # If a cache dependent on model variables exists, it should be cleared here.
    self.clear_cache() 

  def clear_cache(self):
    """Clears any internal cache."""
    if hasattr(self, '_cache') and self._cache:
        self._cache.clear()

  def cache_info(self) -> str:
    """Returns information about the cache."""
    if hasattr(self, '_cache') and self._cache:
        return self._cache.info()
    return "Cache: Not initialized."

  def _perform_actual_inference(self, batch_obs_tensors: list[list[float]], batch_legal_actions_masks: list[list[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Performs model inference on a batch of states, without caching."""
    # Ensure observation_tensor and legal_actions_mask are numpy arrays
    obs_batch_np = np.array(batch_obs_tensors, dtype=np.float32)
    # obs_batch_np = np.expand_dims(obs_tensor_np, 0) # No longer needed, already a batch
    obs_batch_jax = jnp.asarray(obs_batch_np) # Explicit conversion to JAX array

    legals_batch_np = np.array(batch_legal_actions_masks, dtype=bool)
    # legals_batch_np = np.expand_dims(legals_mask_np, 0) # No longer needed
    legals_batch_jax = jnp.asarray(legals_batch_np) # Explicit conversion to JAX array

    # The model internally masks logits using legals_mask_jax
    policy_logits_batch_jax, value_output_batch_jax = self._jit_apply_model_fn(
        self.variables, obs_batch_jax, legals_batch_jax)

    # Convert JAX arrays to NumPy arrays for further processing
    value_outputs_np = np.array(value_output_batch_jax[:, 0]) # Shape: (batch_size,)
    policy_logits_batch_np = np.array(policy_logits_batch_jax) # Shape: (batch_size, num_actions)

    results = []
    for i in range(len(batch_obs_tensors)):
      value_scalar = value_outputs_np[i]
      policy_logits_np = policy_logits_batch_np[i]
      current_legal_mask = batch_legal_actions_masks[i] # This is list[int]

      # Apply softmax to the (already masked by model) logits
      policy_probs = jax.nn.softmax(policy_logits_np)
      policy_probs_np = np.array(policy_probs) # Ensure it's a numpy array

      policy_sum = np.sum(policy_probs_np)
      if not np.isclose(policy_sum, 1.0) and np.any(current_legal_mask):
          num_legal_actions = np.sum(current_legal_mask)
          if num_legal_actions > 0:
              policy_probs_np = np.array(current_legal_mask, dtype=np.float32) / num_legal_actions
          else:
              policy_probs_np = np.zeros_like(current_legal_mask, dtype=np.float32)
      elif not np.any(current_legal_mask):
          policy_probs_np = np.zeros_like(current_legal_mask, dtype=np.float32)
      
      results.append((value_scalar, policy_probs_np))

    return results

  def _inference(self, state: pyspiel.State) -> tuple[np.ndarray, np.ndarray]:
    """Performs a single model inference on the state, using the cache."""
    # Get observation and legal_actions_mask for cache key generation
    obs_tensor_list = state.observation_tensor()
    legal_actions_mask_list = state.legal_actions_mask()

    # Convert to NumPy arrays for .tobytes()
    obs_tensor_np_for_key = np.array(obs_tensor_list, dtype=np.float32)
    legal_actions_mask_np_for_key = np.array(legal_actions_mask_list, dtype=bool)

    cache_key = obs_tensor_np_for_key.tobytes() + legal_actions_mask_np_for_key.tobytes()
    
    # _perform_actual_inference now expects a batch and returns a list of results.
    # For a single state, we pass a batch of one and take the first result.
    value_policy_tuple = self._cache.make(
        cache_key, lambda: self._perform_actual_inference([obs_tensor_list], [legal_actions_mask_list])[0])
    return value_policy_tuple

  def evaluate(self, state: pyspiel.State) -> np.ndarray:
    """Returns a value for the given state.

    The value is from the perspective of the current player in the state.
    For a two-player zero-sum game, this is usually [v, -v].
    """
    value, _ = self._inference(state)
    # Assuming a two-player zero-sum game
    return np.array([value, -value])

  def prior(self, state: pyspiel.State) -> list[tuple[int, float]]:
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