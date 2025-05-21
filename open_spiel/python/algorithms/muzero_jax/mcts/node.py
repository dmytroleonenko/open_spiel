import dataclasses
from typing import Dict, Optional, TYPE_CHECKING

import jax.numpy as jnp

# To prevent circular imports, TYPE_CHECKING is used for type hints
if TYPE_CHECKING:
    # This is a forward reference to the Node class itself for the parent and children attributes
    pass  # pragma: no cover


@dataclasses.dataclass(eq=False)
class Node:
    """
    A node in the Monte Carlo Search Tree.
    Stores statistics for a state encountered during MCTS.
    """
    prior: float  # p_sa: prior probability of selecting the action leading to this node
    action: Optional[int] = None  # The action taken from the parent to reach this node. None for root.
    parent: Optional["Node"] = None  # Parent node in the MCTS tree. None for root.
    children: Dict[int, "Node"] = dataclasses.field(
        default_factory=dict
    )  # Children nodes, keyed by action.

    visit_count: int = 0  # N(s,a) or N(s) depending on context (here, for the state this node represents)
    value_sum: float = 0.0  # W(s,a) or W(s) - sum of backed-up values through this node
    
    # r_k: reward observed when transitioning from parent to this state s_k
    # This is the immediate reward obtained by taking 'action' from 'parent.state'
    reward_from_parent_action: float = 0.0

    # --- Network predictions stored at this node ---
    # These are populated when the node's state is evaluated by the network (e.g., during expansion).
    # The state s_k that this node represents.
    hidden_state: Optional[jnp.ndarray] = None
    
    # p_k: policy logits for actions taken from s_k, as predicted by the network.
    policy_logits: Optional[jnp.ndarray] = None
    
    # v_k: value of s_k, as predicted by the network.
    value_from_network: Optional[float] = None
    
    # to_play: int = -1 # Player whose turn it is at this node.
    # OpenSpiel's AlphaZero uses this. EfficientZero does not explicitly store it in the Node.
    # It's often implicitly handled by the game state or network input.
    # For now, we'll omit it unless it becomes necessary.

    def __post_init__(self):
        # Ensure hidden_state and policy_logits are None by default if not provided,
        # even if an empty JAX array or similar was passed somehow, though dataclass defaults handle most cases.
        if self.hidden_state is not None and not isinstance(self.hidden_state, jnp.ndarray):
            raise TypeError(f"hidden_state must be a jax.numpy.ndarray or None, got {type(self.hidden_state)}")
        if self.policy_logits is not None and not isinstance(self.policy_logits, jnp.ndarray):
            raise TypeError(f"policy_logits must be a jax.numpy.ndarray or None, got {type(self.policy_logits)}")

    def is_expanded(self) -> bool:
        """Checks if this node has been expanded (i.e., has children)."""
        return len(self.children) > 0

    def get_q_value(self) -> float:
        """
        Calculates the Q-value (average backed-up value) for the state represented by this node.
        Q(s) = W(s) / N(s)
        If this node represents state s_k resulting from action a_{k-1} from parent state s_{k-1},
        this can be interpreted as Q(s_{k-1}, a_{k-1}) after sufficient backups.
        """
        if self.visit_count == 0:
            # Return 0 or a neutral value. Some MCTS variants might use parent's value
            # or -infinity/+infinity depending on whose turn it is to play.
            # For MuZero's PUCT, unvisited children often get a Q-value of 0 for the first selection.
            return 0.0
        return self.value_sum / self.visit_count

    def __repr__(self):
        return (
            f"Node(action={self.action}, prior={self.prior:.3f}, visits={self.visit_count}, "
            f"Q={self.get_q_value():.3f}, R={self.reward_from_parent_action:.3f}, "
            f"V_net={self.value_from_network if self.value_from_network is not None else 'N/A'}, "
            f"expanded={self.is_expanded()}, children={len(self.children)})"
        )

    def __hash__(self) -> int:
        """Returns a hash for the node.

        Nodes are hashed based on their action, parent's action (to distinguish
        root nodes with same action but different histories if parent is None vs root),
        prior probability, and visit count.
        We explicitly cast prior to float because JAX 0-d arrays are not hashable.
        """
        # Ensure parent_action is hashable, even if parent is None
        parent_action = self.parent.action if self.parent else -1 # Or some other sentinel
        return hash((self.action, parent_action, float(self.prior), self.visit_count))

    def __eq__(self, other: object) -> bool:
        """Checks if two nodes are equal.

        Nodes are considered equal if they have the same action, parent's action,
        prior probability, and visit count.
        """
        if not isinstance(other, Node):
            return NotImplemented

        parent_action_self = self.parent.action if self.parent else -1
        parent_action_other = other.parent.action if other.parent else -1

        return (
            self.action == other.action and
            parent_action_self == parent_action_other and
            # Comparing floats for equality can be tricky, but for priors/visit counts
            # that are either set or incremented, direct comparison should be fine.
            # Using math.isclose might be needed if they result from complex calculations.
            float(self.prior) == float(other.prior) and # Cast to float for consistency
            self.visit_count == other.visit_count
        ) 