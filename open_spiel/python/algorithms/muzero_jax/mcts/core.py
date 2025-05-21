import dataclasses
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
import chex

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.mcts.node import Node
from open_spiel.python.algorithms.muzero_jax.types import MuZeroModel


@dataclasses.dataclass
class MinMaxStats:
    """A class to store minimum and maximum statistics for normalization."""
    # These are initialized as Python floats (jnp.inf is a float).
    # They may become JAX DeviceArrays after the first update() if the input
    # `value` is a JAX array, due to JAX's type promotion rules.
    # This is generally fine but worth noting for type-sensitive operations.
    minimum: float = jnp.inf
    maximum: float = -jnp.inf
    minmax_delta: float = 1e-6  # As in EfficientZeroV2 MinMaxStats, to avoid division by zero

    def update(self, value: float):
        self.minimum = jnp.minimum(self.minimum, value)
        self.maximum = jnp.maximum(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            value_clipped = jnp.clip(value, self.minimum, self.maximum)
            normalized_value = (value_clipped - self.minimum) / jnp.maximum(
                (self.maximum - self.minimum), self.minmax_delta
            )
        else:
            return jnp.clip(value, 0.0, 1.0)

        return jnp.clip(normalized_value, 0.0, 1.0)

    def clear(self):
        self.minimum = jnp.inf
        self.maximum = -jnp.inf


@dataclasses.dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int
    num_actions: int  # To be set by the game/model
    discount: float
    q_init: float = 0.0  # Initial Q-value for new nodes
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_exploration_fraction: float = 0.25 # Renamed from dirichlet_epsilon for clarity if needed, or keep as is.
    c_base: float = 19652.0
    c_init: float = 1.25
    value_minmax_delta: float = 1e-3 # Used in MinMaxStats normalization
    mask_illegal_actions: bool = True
    max_depth: Optional[int] = None  # Maximum depth of the search tree


class MCTS:
    config: MCTSConfig
    model: Optional[MuZeroModel] = None

    def __init__(self, config: MCTSConfig, model: Optional[MuZeroModel] = None): # Model can be None for testing
        self.config = config
        self.model = model

    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """Calculates the PUCT score for a child node based on EfficientZeroV2's formula."""
        q_sa = child.reward_from_parent_action + self.config.discount * child.get_q_value()
        normalized_q_sa = min_max_stats.normalize(q_sa)
        pb_c = jnp.log((parent.visit_count + self.config.c_base + 1) / self.config.c_base) + self.config.c_init
        exploration_bonus = child.prior * pb_c * (jnp.sqrt(parent.visit_count) / (child.visit_count + 1))
        return normalized_q_sa + exploration_bonus

    def _select_child(self, node: Node, min_max_stats: MinMaxStats, key: Optional[chex.PRNGKey] = None) -> Node:
        """
        Selects a child of the given node.
        Uses UCB score for selection.
        """
        best_child = None
        best_ucb_score = -jnp.inf

        if not node.children:
            raise ValueError("Cannot select child from a node with no children (not expanded).")

        for action, child_node in node.children.items():
            ucb_score = self._ucb_score(node, child_node, min_max_stats)
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_child = child_node
        
        if best_child is None:
            raise RuntimeError(f"Could not select a child for node {node}. Scores: {[self._ucb_score(node, c, min_max_stats) for c in node.children.values()]}")

        return best_child
        
    def _expand_node(self, node: Node, hidden_state: jnp.ndarray, policy_logits: jnp.ndarray, value_from_network: float, legal_actions_mask: Optional[jnp.ndarray] = None):
        """
        Expands a leaf node by creating children nodes.
        Stores network predictions in the expanded node.
        Populates children based on policy_logits and legal_actions_mask.
        """
        if node.is_expanded():
            return

        node.hidden_state = hidden_state
        node.value_from_network = value_from_network
        policy_probs = jax.nn.softmax(policy_logits)

        for action in range(self.config.num_actions):
            if legal_actions_mask is not None and not legal_actions_mask[action]:
                continue

            child_prior = policy_probs[action].item()
            new_child_node = Node(
                prior=child_prior,
                action=action,
                parent=node,
            )
            node.children[action] = new_child_node
        node.policy_logits = policy_logits

    def _backup(self, search_path: list[Node], leaf_value: float, min_max_stats: MinMaxStats):
        """
        Propagates the leaf_value up the search_path.
        Updates visit_count, value_sum for each node.
        Updates min_max_stats with Q-values encountered.
        """
        current_discounted_value = leaf_value
        for node in reversed(search_path):
            node.value_sum += current_discounted_value
            node.visit_count += 1
            if node.parent is not None:
                 q_sa_for_stats = node.reward_from_parent_action + self.config.discount * node.get_q_value()
                 min_max_stats.update(q_sa_for_stats)
            current_discounted_value = node.reward_from_parent_action + self.config.discount * current_discounted_value

    def run_mcts(
        self,
        key: chex.PRNGKey,
        root_state: chex.Array, # Observation or initial hidden state
        legal_actions_mask: Optional[chex.Array] = None,
        # Optional existing root node, e.g., from a previous search if reusing tree
        root_node: Optional[Node] = None
    ) -> Tuple[chex.Array, Node]:
        """Runs the MCTS search from the given root state.

        NOTE: This implementation currently uses Python loops and direct Node
        object mutations, making it not JIT-compatible. A future refactoring
        (see TODO.md, MCTS JIT-compilation) will be needed to convert this to
        a functional style suitable for @jax.jit, aligning with EfficientZeroV2
        performance goals.

        Args:
        // ... existing code ...
        """
        # chex.assert_rank(initial_hidden_state, 2) # Rank depends on obs type
        if self.model is None:
            raise ValueError("MCTS model cannot be None when running simulations.")

        min_max_stats = MinMaxStats(minmax_delta=self.config.value_minmax_delta)
        if root_node is None:
            root_node = Node(prior=1.0, action=None, parent=None, reward_from_parent_action=0.0)
        
        # Squeeze batch dim if present, assume single instance for MCTS.
        # Model expects batched input, so will unsqueeze later or handle inside model.
        # For now, assume initial_hidden_state is for a single instance.
        if root_state.ndim > 1 and root_state.shape[0] == 1:
             root_state_unbatched = root_state.squeeze(axis=0)
        else:
             root_state_unbatched = root_state

        # Model call expects batch dimension. Add it back if it was squeezed or not present.
        if root_state_unbatched.ndim == 1: # e.g. (state_dim,)
            root_state_for_model = jnp.expand_dims(root_state_unbatched, axis=0)
        elif root_state_unbatched.ndim == 2 and not (root_state_unbatched.shape[0] ==1): # e.g. (H,W) for single channel image
             root_state_for_model = jnp.expand_dims(root_state_unbatched, axis=0)
        elif root_state_unbatched.ndim == 3: # e.g. (H,W,C)
             root_state_for_model = jnp.expand_dims(root_state_unbatched, axis=0)
        else:
            root_state_for_model = root_state # Assumed already batched e.g. (1, state_dim)

        # Use initial_inference for the root node expansion
        # initial_inference returns: hidden_state, value, policy_logits, reward
        # We primarily need policy_logits and value for the root here.
        # The hidden_state returned is the initial hidden_state based on the observation (root_state).
        # The reward is the reward obtained by transitioning to this first (root) state.
        # As MCTS starts from a state, this reward is typically 0 or not directly used for root prior.
        initial_key, expansion_key = jax.random.split(key)
        root_hidden_state_batched, root_value_batched, root_policy_logits_batched, root_reward_batched = \
            self.model.initial_inference(root_state_for_model, initial_key)

        # Ensure outputs are unbatched if model returned a batch of 1
        root_policy_logits = root_policy_logits_batched.squeeze(axis=0) if root_policy_logits_batched.ndim > 1 and root_policy_logits_batched.shape[0] == 1 else root_policy_logits_batched
        root_value = root_value_batched.squeeze(axis=0) if root_value_batched.ndim > 0 and root_value_batched.shape[0] == 1 else root_value_batched
        root_hidden_state = root_hidden_state_batched.squeeze(axis=0) if root_hidden_state_batched.ndim > 1 and root_hidden_state_batched.shape[0] == 1 else root_hidden_state_batched

        # The root_node.value_from_net should store the scalar value from the model.
        root_node.value_from_net = root_value.item() # Ensure it's a Python scalar

        # Store the hidden state in the root node as it's now known from initial_inference
        root_node.hidden_state = root_hidden_state
        # Also store the policy logits used for expansion (after potential noise)
        # This will be updated after noise application if any
        root_node.policy_logits = root_policy_logits

        # Expand the root node with its policy and value
        self._expand_node(
            root_node,
            root_node.hidden_state,
            root_node.policy_logits,
            root_node.value_from_net,
            legal_actions_mask
        )
        # Add Dirichlet noise to root policy priors if configured (after children are created)
        if self.config.dirichlet_alpha > 0:  # pragma: no cover
            key, noise_key = jax.random.split(key)
            num_valid_children = sum(
                1 for action_idx in root_node.children
                if (legal_actions_mask is None or legal_actions_mask[action_idx])
            )
            if num_valid_children > 0:  # pragma: no cover
                dirichlet_noise = jax.random.dirichlet(
                    noise_key,  # pragma: no cover
                    alpha=jnp.full(num_valid_children, self.config.dirichlet_alpha)
                )
                noise_idx = 0
                for action, child in root_node.children.items():
                    if legal_actions_mask is None or legal_actions_mask[action]:
                        child.prior = (
                            (1 - self.config.dirichlet_exploration_fraction) * child.prior
                            + self.config.dirichlet_exploration_fraction * dirichlet_noise[noise_idx]
                        )
                        noise_idx += 1

        # Main MCTS simulation loop
        for sim in range(self.config.num_simulations):
            key, loop_key = jax.random.split(key)
            current_node = root_node
            search_path = [current_node]

            while current_node.is_expanded():
                current_node = self._select_child(current_node, min_max_stats, key=loop_key)
                search_path.append(current_node)
            
            # If search_path has only one element (root), it means we couldn't select a child (e.g. root not expanded or no valid children)
            # In this case, the leaf is the root itself. Its value is already known from the initial prediction.
            if len(search_path) == 1: # current_node is still root_node
                leaf_value_scalar = root_node.value_from_network # Use root's network value
                # No recurrent inference needed, and backup is just for the root itself if it was chosen as leaf.
                # However, standard backup expects a path. If root is the leaf, path is [root].
                # The _backup logic needs to handle this. Or, we ensure path always has at least one step if possible.
                # For now, if path is just [root], the loop for backup will update root and min_max_stats if applicable.
                # The main issue is that parent_of_leaf = search_path[-2] will fail.
                # So, if len(search_path) == 1, we are at the root, and it's the leaf.
                self._backup(search_path, leaf_value_scalar, min_max_stats)
                continue # Skip recurrent inference for this simulation if root is the unexpandable leaf

            parent_of_leaf = search_path[-2]
            parent_hidden_state_unbatched = parent_of_leaf.hidden_state
            
            # Add batch dim for model input
            if parent_hidden_state_unbatched.ndim == 1:
                parent_hidden_state_for_model = jnp.expand_dims(parent_hidden_state_unbatched, axis=0)
            elif parent_hidden_state_unbatched.ndim == 2 and not (parent_hidden_state_unbatched.shape[0]==1): # (H,W)
                 parent_hidden_state_for_model = jnp.expand_dims(parent_hidden_state_unbatched, axis=0)
            elif parent_hidden_state_unbatched.ndim == 3: # (H,W,C)
                 parent_hidden_state_for_model = jnp.expand_dims(parent_hidden_state_unbatched, axis=0)
            else:
                parent_hidden_state_for_model = parent_hidden_state_unbatched # Assume already (1, ...)
            
            action_array = jnp.array([current_node.action])
            if action_array.ndim == 0: # Ensure action is at least 1D for model
                action_array = jnp.expand_dims(action_array, axis=0)

            # Get a new key for this recurrent inference step
            current_key, loop_key = jax.random.split(loop_key) # Update loop_key for next iteration/use

            leaf_hidden_state_batched, leaf_value_net_batched, leaf_policy_logits_batched, leaf_reward_batched = self.model.recurrent_inference(
                parent_hidden_state_for_model,
                action_array,
                current_key # Pass the PRNG key
            )

            # Unbatch hidden_state only if batch dimension is 1
            if leaf_hidden_state_batched.ndim > 1 and leaf_hidden_state_batched.shape[0] == 1:
                leaf_hidden_state_unbatched = leaf_hidden_state_batched.squeeze(axis=0)
            else:
                leaf_hidden_state_unbatched = leaf_hidden_state_batched
            current_node.reward_from_parent_action = leaf_reward_batched.item() if leaf_reward_batched.ndim == 0 else leaf_reward_batched[0].item()
            leaf_value_scalar = leaf_value_net_batched.item() if leaf_value_net_batched.ndim == 0 else leaf_value_net_batched[0].item()
            # Unbatch policy_logits only if batch dimension is 1
            if leaf_policy_logits_batched.ndim > 1 and leaf_policy_logits_batched.shape[0] == 1:
                leaf_policy_logits_unbatched = leaf_policy_logits_batched.squeeze(axis=0)
            else:
                leaf_policy_logits_unbatched = leaf_policy_logits_batched

            self._expand_node(
                current_node,
                leaf_hidden_state_unbatched,
                leaf_policy_logits_unbatched,
                leaf_value_scalar,
                legal_actions_mask
            )
            self._backup(search_path, leaf_value_scalar, min_max_stats)

        visit_counts = jnp.zeros(self.config.num_actions)
        if root_node.children:
            for action, child in root_node.children.items():
                if legal_actions_mask is None or legal_actions_mask[action]:
                     visit_counts = visit_counts.at[action].set(child.visit_count)
        
        if jnp.sum(visit_counts) > 0:
            final_policy = visit_counts / jnp.sum(visit_counts)
        else:
            if legal_actions_mask is not None:
                num_legal = jnp.sum(legal_actions_mask)
                final_policy = jnp.where(legal_actions_mask, 1.0/num_legal, 0.0) if num_legal > 0 else jnp.ones(self.config.num_actions) / self.config.num_actions
            else:
                final_policy = jnp.ones(self.config.num_actions) / self.config.num_actions
        return final_policy, root_node