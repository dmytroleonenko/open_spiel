from typing import Tuple
import jax
import jax.numpy as jnp
import chex

from open_spiel.python.algorithms.muzero_jax.mcts.mcts_state import MCTSState, INVALID_NODE_INDEX
from open_spiel.python.algorithms.muzero_jax.mcts.core import MCTSConfig # For config parameters

# Small constant to prevent division by zero or log of zero.
EPSILON = 1e-6

def select_child_jax(
    mcts_state: MCTSState,
    current_node_idx: chex.Numeric, # Index of the current (parent) node
    key: chex.PRNGKey, # PRNGKey for potential stochasticity (not used in standard UCB)
    config: MCTSConfig
) -> Tuple[chex.Numeric, chex.Numeric]: # (selected_child_idx, selected_action)
    """
    Selects a child of the current_node_idx based on the UCB formula.
    This function is designed to be JAX-compatible.

    Args:
        mcts_state: The current state of the MCTS tree.
        current_node_idx: The index of the node from which to select a child.
        key: JAX PRNGKey. Currently unused but good practice for future extensions.
        config: MCTS configuration.

    Returns:
        A tuple containing:
            - selected_child_idx: The index of the selected child node.
            - selected_action: The action taken to reach the selected child.
    """
    del key # Unused for deterministic UCB selection

    # Get properties of the current (parent) node
    parent_visit_count = mcts_state.visit_count[current_node_idx]
    parent_prior_probs = mcts_state.prior_probabilities[current_node_idx] # Shape (num_actions,)
    
    # Calculate the UCB constant pb_c
    # pb_c = log((parent_visit_count + config.c_base + 1) / config.c_base) + config.c_init
    # Ensure arguments to log are positive
    log_arg = (parent_visit_count + config.c_base + EPSILON) / (config.c_base + EPSILON) # Add EPSILON to parent_visit_count for safety at N=0
    pb_c = jnp.log(log_arg + EPSILON) + config.c_init
    
    sqrt_parent_visit_count = jnp.sqrt(parent_visit_count + EPSILON) # Add EPSILON for N=0 case

    best_ucb_score = -jnp.inf
    selected_child_idx = INVALID_NODE_INDEX
    selected_action = -1 # Invalid action

    # Iterate over all possible actions from the current node
    # This loop will be unrolled by JAX if the outer function is JIT-compiled.
    for action in range(config.num_actions):
        child_idx = mcts_state.children_node_indices[current_node_idx, action]
        
        # --- Calculate UCB score for the current action/child ---
        # Initialize UCB for illegal/unexpanded children to -inf
        ucb_score_for_action = -jnp.inf

        # Only consider valid children that have been expanded (i.e., child_idx is valid)
        # is_valid_child = (child_idx != INVALID_NODE_INDEX)

        # Use jax.lax.cond to avoid materializing both branches if not needed,
        # or rely on JAX to optimize the select if is_valid_child is a scalar boolean per loop iter.
        # For a simple loop unroll, direct computation and masking is often fine.

        # Value term Q(s,a) calculation
        # Q(s,a) = R(s,a) + gamma * V(s')
        # R(s,a) is reward_from_parent_action[child_idx]
        # V(s') is value_sum[child_idx] / visit_count[child_idx]
        
        # Default values if child is invalid or unvisited (for Q_sa part)
        child_reward = 0.0
        child_value_estimate = 0.0 # Default to 0 if child not visited or invalid
        
        # Get child properties if valid
        # These lines will execute for all actions, but results are masked later by is_valid_child
        _child_reward = mcts_state.reward_from_parent_action[child_idx]
        _child_value_sum = mcts_state.value_sum[child_idx]
        _child_visit_count = mcts_state.visit_count[child_idx]

        # Calculate child_value_estimate safely
        _child_value_estimate = jnp.where(
            _child_visit_count > 0,
            _child_value_sum / (_child_visit_count + EPSILON),
            0.0 # Default value for unvisited children (can also be network's value if stored before visit)
                # Or use mcts_state.value_from_network[child_idx] if it's more appropriate for unvisited explored nodes.
                # Standard MCTS uses empirical average or 0.
        )
        
        # If child_idx is INVALID_NODE_INDEX, these _child_reward and _child_value_estimate
        # will effectively be from mcts_state.arrays[-1], which should be handled by the mask.
        # This is less clean. Better to use a conditional approach or ensure masked arrays.

        # Let's compute Q_sa only if the child is valid
        # We compute Q_sa for ALL actions, and then use a mask.
        # This is more JAX-friendly for vectorization if this loop was vectorized,
        # but for unrolled loop, explicit cond might be clearer if JAX doesn't optimize well.

        q_sa = _child_reward + config.discount * _child_value_estimate
        
        # Normalize Q_sa using MinMaxStats
        # The MinMaxStats.normalize method is defined in core.py and expects min_val, max_val, delta.
        normalized_q_sa = jax.tree_util.Partial(
            _normalize_value,
            min_val=mcts_state.min_max_stats_minimum,
            max_val=mcts_state.min_max_stats_maximum,
            delta=config.value_minmax_delta # MCTSConfig should store this
        )(q_sa)

        # Exploration term
        # P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        prior_for_action = parent_prior_probs[action]
        exploration_term = pb_c * prior_for_action * (sqrt_parent_visit_count / (1 + _child_visit_count))

        # UCB Score = Normalized Q(s,a) + Exploration Term
        current_action_ucb = normalized_q_sa + exploration_term
        
        # Mask out UCB for invalid children (those not yet created/linked)
        # If child_idx is INVALID_NODE_INDEX, this action is not selectable.
        is_valid_child_for_ucb = (child_idx != INVALID_NODE_INDEX)
        ucb_score_for_action = jnp.where(is_valid_child_for_ucb, current_action_ucb, -jnp.inf)

        # --- Update best UCB score and selected child/action ---
        is_better_and_valid = (ucb_score_for_action > best_ucb_score)
        
        best_ucb_score = jnp.where(is_better_and_valid, ucb_score_for_action, best_ucb_score)
        selected_child_idx = jnp.where(is_better_and_valid, child_idx, selected_child_idx)
        selected_action = jnp.where(is_better_and_valid, action, selected_action)

    # Fallback if no valid child was found (e.g. terminal node or error)
    # This should ideally not happen if called on a non-terminal, expanded node.
    # If all UCBs are -inf, selected_action might remain -1.
    # Consider adding a check or default action if selected_action is still -1.
    # For example, select the action with the highest prior if all UCBs are -inf.
    # However, standard UCB expects at least one selectable child from an expanded node.
    # If selected_action == -1 here, it means no children were valid (child_idx != INVALID_NODE_INDEX).
    # This means current_node_idx might be a leaf that wasn't expanded, or terminal.
    
    # Ensure selected_child_idx is integer type for indexing
    selected_child_idx = jnp.asarray(selected_child_idx, dtype=jnp.int32)
    selected_action = jnp.asarray(selected_action, dtype=jnp.int32)
    
    return selected_child_idx, selected_action 

# Helper function to mirror MinMaxStats.normalize behavior for JAX
# This could also be part of MCTSState if it had methods, or a utility.
def _normalize_value(value: chex.Numeric, min_val: chex.Numeric, max_val: chex.Numeric, delta: float) -> chex.Numeric:
    """Normalizes value to [0, 1] given min_val and max_val from MinMaxStats like structure."""
    # delta = max_val - min_val
    # Using a pre-computed or configured delta from MCTSConfig as max_val and min_val might be jnp.inf
    # The original MinMaxStats in core.py initializes min/max to inf/-inf and has a delta too.
    # The MCTSState has min_max_stats_minimum and _maximum.
    # The original MinMaxStats has a value_minmax_delta in config.

    # Handle case where min and max are not yet updated (still inf)
    # or where they are too close (delta is small).
    # The original MinMaxStats.normalize checks if self.maximum > self.minimum
    
    # Effective delta for normalization:
    # If min/max are inf/-inf, actual_delta would be inf. Use configured delta.
    # If they are set, use their difference, but ensure it's at least the configured delta.
    current_range = max_val - min_val
    
    # If min_val is inf or max_val is -inf, current_range is problematic.
    # This happens if no values have been recorded yet.
    # In this case, the original normalize returns the value itself (or 0 if value is also inf).
    # Let's replicate: if min_val == jnp.inf or max_val == -jnp.inf, effectively, range is not defined. Default to value or 0.
    # A robust way: if current_range is not positive and finite, don't normalize (or return 0/midpoint).

    is_valid_range = (max_val > min_val) & jnp.isfinite(min_val) & jnp.isfinite(max_val)

    # Use the actual range if valid and larger than configured delta, else use configured delta.
    # This logic is tricky to exactly replicate from the original class method without having the class instance.
    # The original normalize: if self.maximum > self.minimum: return (value - self.minimum) / (self.maximum - self.minimum)
    # else: return value. This seems to imply no scaling if range is zero or inverted.

    # Simplified: if valid range, normalize, else return a default (e.g., 0 or 0.5 if trying to map to [0,1])
    # or the value itself if that's preferred when stats are not ready.
    # The original behavior for uninitialized MinMaxStats (max < min) is to return the value as is.
    norm_value = jnp.where(
        is_valid_range,
        (value - min_val) / (current_range + EPSILON), # Add EPSILON to avoid div by zero if range is tiny
        value # Or 0.0 if unnormalized values are not in a sensible range for UCB scores
              # Let's stick to 'value' as per original MinMaxStats for uninitialized case.
    )
    # Clip to [0, 1] as UCB expects normalized values in this range.
    return jnp.clip(norm_value, 0.0, 1.0) 