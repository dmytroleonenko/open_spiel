import dataclasses
import jax
import jax.numpy as jnp
import chex
from typing import Optional
from jax.tree_util import register_pytree_node_class

# Sentinel value for unvisited/invalid nodes or padding.
# Using a large negative number for indices where 0 might be a valid index.
INVALID_NODE_INDEX = -1
# Using NaN for float values where 0 might be a meaningful value.
# However, be cautious with NaNs as they propagate and can make debugging harder.
# Often, using a specific numerical sentinel (like -1 or a very large/small number)
# and a separate boolean mask array is safer for JIT compilation and avoids NaN issues.
# For this initial definition, we'll use common JAX types.
# Let's use 0 for counts and sums initially, and rely on visit_count for validity.

@register_pytree_node_class
@dataclasses.dataclass
class MCTSState:
    """
    A JAX-compatible PyTree to store the entire MCTS tree and related search statistics.
    Nodes are referenced by their integer index (0 to max_nodes-1).
    The root node is always at index 0.
    """
    # --- Tree Structure ---
    # parent_indices[node_idx] = parent_node_idx
    parent_indices: chex.ArrayDevice        # Shape: (max_nodes,), dtype: int32
    # action_from_parent[node_idx] = action_idx that led to this node
    action_from_parent: chex.ArrayDevice    # Shape: (max_nodes,), dtype: int32
    
    # children_actions[node_idx, child_k_idx] = action_idx for k-th child slot
    # children_node_indices[node_idx, child_k_idx] = node_idx of k-th child
    # We need a fixed number of potential children per node for JAX arrays.
    # This implies num_actions is the max number of children.
    # children_node_indices[parent_idx, action] = child_idx
    children_node_indices: chex.ArrayDevice # Shape: (max_nodes, num_actions), dtype: int32

    # --- Node Statistics ---
    # visit_count[node_idx] = N(s) (visit count of the state represented by node_idx)
    visit_count: chex.ArrayDevice           # Shape: (max_nodes,), dtype: int32
    # value_sum[node_idx] = W(s) (sum of backed-up values through node_idx)
    value_sum: chex.ArrayDevice             # Shape: (max_nodes,), dtype: float32
    # reward_from_parent_action[node_idx] = R(s_parent, a_to_node)
    # (immediate reward obtained by taking action_from_parent[node_idx] from parent_indices[node_idx])
    reward_from_parent_action: chex.ArrayDevice # Shape: (max_nodes,), dtype: float32
    
    # --- Network Predictions stored at nodes ---
    # hidden_state[node_idx] = h_s (hidden state for the state s represented by node_idx)
    # The shape of hidden_state_dim will depend on the model's representation network.
    # For example, (max_nodes, feature_dim) or (max_nodes, H, W, C).
    # We'll need to know hidden_state_dim at initialization.
    hidden_state: chex.ArrayDevice          # Shape: (max_nodes, *hidden_state_dim)
    
    # policy_logits[node_idx, action_idx] = P(s, a) (raw policy logits from network for state s at node_idx)
    policy_logits: chex.ArrayDevice         # Shape: (max_nodes, num_actions), dtype: float32
    # prior_probabilities[node_idx, action_idx] = pi(a|s) (prior probability after softmax, dirichlet noise)
    prior_probabilities: chex.ArrayDevice   # Shape: (max_nodes, num_actions), dtype: float32

    # value_from_network[node_idx] = V(s) (scalar value prediction from network for state s at node_idx)
    value_from_network: chex.ArrayDevice    # Shape: (max_nodes,), dtype: float32

    # --- Search Status ---
    # is_expanded[node_idx] = True if node_idx has been expanded
    is_expanded: chex.ArrayDevice           # Shape: (max_nodes,), dtype: bool
    # Represents the next available free index in the node arrays.
    # Starts at 1 because root is at 0.
    num_allocated_nodes: chex.Numeric # Scalar int, can be ArrayDevice of shape ()

    # --- MinMax Statistics for UCB normalization ---
    # Using a single MinMaxStats for the whole search, as in current MCTS.
    # These will be updated functionally.
    min_max_stats_minimum: chex.Numeric     # Scalar float
    min_max_stats_maximum: chex.Numeric     # Scalar float

    # --- Miscellaneous ---
    # Root observation, needed if we re-evaluate root or for some MCTS variants.
    # Not strictly part of the tree structure but useful for context.
    # This is the initial observation given to run_mcts.
    # Its shape will vary per game. For now, keep it optional or define its role clearly.
    # For a JIT-friendly structure, it's better if all inputs are explicit.
    # Let's assume root_hidden_state is already in hidden_state[0].

    # Key for JAX random operations within the MCTS search loop if needed
    # (e.g., if sampling during simulations for stochastic envs becomes part of JITted loop)
    # This might be better passed as an argument to the loop body.
    # key: chex.PRNGKey 

    # --- Config-like parameters that might be needed by JITted functions ---
    # These are static for a given search but need to be available.
    # Alternatively, pass MCTSConfig object if it's JAX-registrable or its relevant fields.
    discount: chex.Numeric                  # Scalar float
    num_actions: chex.Numeric               # Scalar int, for array shapes
    # c_base, c_init for UCB, etc. can be passed via config.

    @classmethod
    def new_state(
        cls,
        max_nodes: int,
        hidden_state_example: chex.ArrayDevice, # Used to get shape and dtype
        num_actions: int,
        discount: float,
        min_max_delta: float = 1e-6 # From MinMaxStats default in core.py
    ) -> "MCTSState":
        """Initializes a new, empty MCTSState tree."""
        
        hidden_state_dim = hidden_state_example.shape
        hidden_state_dtype = hidden_state_example.dtype

        return cls(
            parent_indices=jnp.full((max_nodes,), INVALID_NODE_INDEX, dtype=jnp.int32),
            action_from_parent=jnp.full((max_nodes,), INVALID_NODE_INDEX, dtype=jnp.int32),
            children_node_indices=jnp.full((max_nodes, num_actions), INVALID_NODE_INDEX, dtype=jnp.int32),
            visit_count=jnp.zeros((max_nodes,), dtype=jnp.int32),
            value_sum=jnp.zeros((max_nodes,), dtype=jnp.float32),
            reward_from_parent_action=jnp.zeros((max_nodes,), dtype=jnp.float32),
            hidden_state=jnp.zeros((max_nodes, *hidden_state_dim), dtype=hidden_state_dtype),
            policy_logits=jnp.zeros((max_nodes, num_actions), dtype=jnp.float32),
            prior_probabilities=jnp.zeros((max_nodes, num_actions), dtype=jnp.float32),
            value_from_network=jnp.zeros((max_nodes,), dtype=jnp.float32),
            is_expanded=jnp.zeros((max_nodes,), dtype=jnp.bool_),
            num_allocated_nodes=jnp.array(0, dtype=jnp.int32), # Initially 0, root will be 1st.
            min_max_stats_minimum=jnp.array(jnp.inf, dtype=jnp.float32),
            min_max_stats_maximum=jnp.array(-jnp.inf, dtype=jnp.float32),
            discount=jnp.array(discount, dtype=jnp.float32),
            num_actions=jnp.array(num_actions, dtype=jnp.int32)
        )

    def tree_flatten(self):
        children = (
            self.parent_indices, self.action_from_parent, self.children_node_indices,
            self.visit_count, self.value_sum, self.reward_from_parent_action,
            self.hidden_state, self.policy_logits, self.prior_probabilities, self.value_from_network,
            self.is_expanded, self.num_allocated_nodes,
            self.min_max_stats_minimum, self.min_max_stats_maximum,
            self.discount, self.num_actions
        )
        # No non-JAX array metadata needed for aux_data for MCTSState currently.
        aux_data = None 
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# For now, relying on JAX's automatic dataclass handling. If issues arise,
# explicit registration above can be uncommented and completed. 