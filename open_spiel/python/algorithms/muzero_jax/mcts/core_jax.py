import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional
import dataclasses # Import standard dataclasses

from open_spiel.python.algorithms.muzero_jax.mcts.mcts_state import MCTSState, INVALID_NODE_INDEX
from open_spiel.python.algorithms.muzero_jax.mcts.core import MCTSConfig # For config parameters
from open_spiel.python.algorithms.muzero_jax.types import MuZeroModel, ModelOutput # For model type hint and ModelOutput

# Root node is always at index 0
ROOT_INDEX = 0

def prepare_initial_mcts_state(
    key: chex.PRNGKey,
    model: MuZeroModel,
    initial_observation: chex.Array,
    config: MCTSConfig,
    max_nodes: int, # Max nodes for MCTSState pre-allocation
    legal_actions_mask: Optional[chex.Array] = None,
) -> MCTSState:
    """
    Prepares the initial MCTSState for a search, including root node initialization via model.initial_inference.

    Args:
        key: JAX PRNGKey for any stochastic operations (e.g., model inference if it uses dropout).
        model: The MuZero model capable of initial_inference.
        initial_observation: The initial observation from the environment.
        config: MCTS configuration dataclass.
        max_nodes: Maximum number of nodes the MCTSState can hold.
        legal_actions_mask: Optional mask for legal actions at the root.

    Returns:
        An MCTSState initialized with the root node's properties set from initial_inference.
    """
    # Model call expects batch dimension.
    # This logic is similar to the start of the original run_mcts.
    # TODO(djl): Consolidate observation batching/unbatching logic if it becomes complex.
    if initial_observation.ndim == 1: # e.g. (state_dim,)
        initial_observation_for_model = jnp.expand_dims(initial_observation, axis=0)
    elif initial_observation.ndim == 2 and not (initial_observation.shape[0] ==1): # e.g. (H,W) for single channel image
        initial_observation_for_model = jnp.expand_dims(initial_observation, axis=0)
    elif initial_observation.ndim == 3: # e.g. (H,W,C)
        initial_observation_for_model = jnp.expand_dims(initial_observation, axis=0)
    elif initial_observation.ndim > 0 and initial_observation.shape[0] == 1: # Already batched
        initial_observation_for_model = initial_observation
    else: # Should ideally be batched (1, ...) or unbatched (features,) or (H,W,C)
        # This case might need more robust handling or clearer input spec.
        # For now, assume it's correctly shaped if not matching above.
        initial_observation_for_model = initial_observation 

    # Perform initial inference
    # Protocol: hidden_state, reward, policy_logits, value
    model_out: ModelOutput = model.initial_inference(initial_observation_for_model, key, training=False) # MCTS is for eval

    root_hidden_state_batched = model_out.hidden_state
    root_reward_batched = model_out.reward
    root_policy_logits_batched = model_out.policy_logits
    root_value_batched = model_out.value

    # Ensure outputs are unbatched (remove leading dim of 1 if present)
    # Simpler squeeze: if it has a leading dim of 1 and more than 1 dim total, squeeze it.
    root_hidden_state = root_hidden_state_batched.squeeze(axis=0) if root_hidden_state_batched.shape[0] == 1 and root_hidden_state_batched.ndim > 1 else root_hidden_state_batched
    root_policy_logits = root_policy_logits_batched.squeeze(axis=0) if root_policy_logits_batched.shape[0] == 1 and root_policy_logits_batched.ndim > 1 else root_policy_logits_batched
    # Value is scalar-like. Squeeze if it's a 1-element 1D array, otherwise take as is if 0D.
    root_value = root_value_batched.squeeze(axis=0) if root_value_batched.ndim == 1 and root_value_batched.shape[0] == 1 else root_value_batched
    # root_reward is also scalar-like, handle similarly. Used for reward_from_parent_action for root.
    # Ensure it becomes a Python scalar for storage or direct use if MCTSState expects that.
    if root_reward_batched.ndim == 1 and root_reward_batched.shape[0] == 1:
        root_reward_scalar = root_reward_batched.squeeze(axis=0).item()
    elif root_reward_batched.ndim == 0:
        root_reward_scalar = root_reward_batched.item()
    else:
        # This case implies reward is not a scalar or (1,) batched scalar, which might be unexpected.
        # For now, attempt to get item if it's somehow batched differently but still one element.
        # Or raise an error if shape is not (1,) or (). Robust handling depends on expected model output.
        # Assuming model provides either a 0-dim scalar or a 1-dim scalar with shape (1,)
        # If it's truly batched e.g. (B,), and B > 1, this logic isn't for that.
        # Given MCTS usually deals with single instances, this should be okay.
        raise ValueError(f"Unexpected shape for root_reward_batched: {root_reward_batched.shape}")

    # Initialize an empty MCTSState
    mcts_state = MCTSState.new_state(
        max_nodes=max_nodes,
        hidden_state_example=root_hidden_state, # Use the actual root hidden state for shape/dtype
        num_actions=config.num_actions,
        discount=config.discount,
        min_max_delta=config.value_minmax_delta 
    )

    # Populate root node (index ROOT_INDEX=0)
    # Parent of root is invalid, action from parent is invalid/None
    # Children are initially all invalid
    # Visit count for root starts at 0, will be incremented during backup of first simulation
    # Value sum for root starts at 0, will be updated during backup
    
    # Store network predictions for the root node
    # It is critical to use .at[ROOT_INDEX].set() for JAX functional updates
    new_hidden_state_array = mcts_state.hidden_state.at[ROOT_INDEX].set(root_hidden_state)
    new_policy_logits_array = mcts_state.policy_logits.at[ROOT_INDEX].set(root_policy_logits)
    new_value_from_network_array = mcts_state.value_from_network.at[ROOT_INDEX].set(root_value)
    # The reward for reaching the root state (from a conceptual pre-root state) is root_reward_scalar
    new_reward_from_parent_action_array = mcts_state.reward_from_parent_action.at[ROOT_INDEX].set(root_reward_scalar)

    # Calculate prior probabilities for root children (after potential dirichlet noise)
    # Raw probabilities before noise
    root_policy_probs = jax.nn.softmax(root_policy_logits)
    
    # Apply dirichlet noise if configured
    # This needs to be JAX-compatible. The original MCTS code calculates num_valid_children.
    # For JIT, if legal_actions_mask is dynamic, this becomes trickier.
    # Assuming legal_actions_mask is static for the root for now for simplicity, or handled carefully.
    final_root_priors = root_policy_probs
    if config.dirichlet_alpha > 0 and config.dirichlet_exploration_fraction > 0:
        key_noise, _ = jax.random.split(key) # Use a new key for noise
        
        if legal_actions_mask is not None:
            num_valid_actions = jnp.sum(legal_actions_mask)
            # Ensure num_valid_actions is appropriate for dirichlet call (e.g. > 0)
            # Create alpha vector only for valid actions
            # This part is tricky if num_valid_actions is not static for JIT.
            # For now, let's assume if mask is provided, it's used to select relevant probs.
            # A simpler JIT approach might be to apply noise to all, then mask, but that changes distribution.
            
            # A robust JIT way: generate noise for all actions, then select.
            dirichlet_noise_all = jax.random.dirichlet(
                key_noise,
                alpha=jnp.full(config.num_actions, config.dirichlet_alpha)
            )
            masked_noise = jnp.where(legal_actions_mask, dirichlet_noise_all, 0.0)
            # Renormalize masked_noise if needed, or use as is if subsequent selection handles it.
            # The original code applies noise only to valid children priors.
            # Let's stick to applying noise to corresponding elements in root_policy_probs.
            
            # Mix operation, element-wise, then rely on subsequent masking for invalid actions
            mixed_priors = (
                (1 - config.dirichlet_exploration_fraction) * root_policy_probs
                + config.dirichlet_exploration_fraction * masked_noise # Use masked_noise here
            )
            # Ensure it sums to 1 over legal actions if that's a requirement post-noise.
            # For now, this mixes noise into the probability distribution.
            # If an action is illegal, its root_policy_probs element should ideally be -inf or very small
            # before softmax, or its final_prior set to 0.
            final_root_priors = jnp.where(legal_actions_mask, mixed_priors, 0.0)
            # Normalize if needed: final_root_priors /= jnp.sum(final_root_priors)
        else:
            # No legal_actions_mask, apply noise to all actions' priors
            dirichlet_noise = jax.random.dirichlet(
                key_noise,
                alpha=jnp.full(config.num_actions, config.dirichlet_alpha)
            )
            final_root_priors = (
                (1 - config.dirichlet_exploration_fraction) * root_policy_probs
                + config.dirichlet_exploration_fraction * dirichlet_noise
            )

    new_prior_probabilities_array = mcts_state.prior_probabilities.at[ROOT_INDEX].set(final_root_priors)

    # Root node is considered allocated, but not yet expanded (children not created in MCTSState yet)
    # Expansion will happen in the first simulation step that selects the root.
    # Or, we can choose to fully expand the root here (create its children entries).
    # The original MCTS code expands the root (_expand_node) right after noise.
    # Let's defer actual child creation to the expand_node_jax function for consistency.
    # is_expanded for root remains False until its children are processed.
    # num_allocated_nodes becomes 1 as root is now initialized.
    new_num_allocated_nodes = jnp.array(1, dtype=jnp.int32)

    return dataclasses.replace( # Use standard dataclasses.replace
        mcts_state,
        hidden_state=new_hidden_state_array,
        policy_logits=new_policy_logits_array,
        value_from_network=new_value_from_network_array,
        reward_from_parent_action=new_reward_from_parent_action_array,
        prior_probabilities=new_prior_probabilities_array,
        num_allocated_nodes=new_num_allocated_nodes,
        # MinMaxStats are initialized in MCTSState.new_state, remain as is for now.
        # visit_count[ROOT_INDEX] and value_sum[ROOT_INDEX] remain 0.
        # parent_indices[ROOT_INDEX] etc. remain INVALID_NODE_INDEX.
    ) 

def _expand_node_jax(
    mcts_state: MCTSState,
    parent_node_idx: int,
    parent_hidden_state: chex.Array, # Hidden state of the parent node to be expanded
    network_policy_logits: chex.Array, # Raw policy logits from network for parent_node_idx
    network_value: chex.Numeric, # Scalar value prediction from network for parent_node_idx
    network_reward: chex.Numeric, # Scalar reward prediction from network for parent_node_idx (reward for *reaching* parent_node_idx)
    model: MuZeroModel, # Needed for recurrent_inference
    key: chex.PRNGKey, # For recurrent_inference
    config: MCTSConfig,
    legal_actions_mask: Optional[chex.Array] = None,
) -> MCTSState:
    """
    Expands a leaf node in the MCTS tree.

    This function is designed to be JIT-compatible. It updates the MCTSState
    by creating children for the given parent_node_idx, populating their
    initial statistics and network predictions obtained from a recurrent_inference call.

    Args:
        mcts_state: The current MCTS state.
        parent_node_idx: The index of the leaf node to expand.
        parent_hidden_state: The hidden state of the node being expanded.
        network_policy_logits: Policy logits for the parent node (from initial or recurrent inference).
        network_value: Value for the parent node (from initial or recurrent inference).
        network_reward: Reward for reaching the parent node (from initial or recurrent inference).
        model: The MuZero model for performing recurrent_inference.
        key: JAX PRNGKey for recurrent_inference.
        config: MCTS configuration.
        legal_actions_mask: Optional mask for legal actions from the parent_node_idx.
                           If None, all actions are assumed legal up to num_actions.

    Returns:
        The updated MCTSState with the parent_node_idx expanded and its children initialized.
    """
    
    # Idempotency: If node is already expanded, do nothing.
    # This must be a functional conditional for JIT.
    # However, a simple Python conditional is fine if _expand_node_jax is part of a larger JIT block
    # where this check happens before repeated calls *within that block* for the same node.
    # For true JIT safety for standalone calls, jax.lax.cond would be needed here.
    # Given the typical MCTS loop structure, this Python-level guard is often sufficient
    # as a given node is usually expanded only once per MCTS batch simulation pass.
    # Let's use a direct return, assuming higher-level logic prevents re-expansion in a hot loop if performance critical.
    # If _expand_node_jax itself is to be JITted and called multiple times with potentially same parent_idx,
    # this guard needs to be jax.lax.cond.
    # For now, making it a simple return, implying it's correctly handled by caller or not in a JIT hot path for repeated calls.
    # To be fully robust for `jax.jit(expand_node_jax)(...)`, this needs to be functional.
    # We can make it functional with jax.lax.cond around the entire expansion logic.

    def expansion_logic(mcts_state_arg: MCTSState) -> MCTSState:
        # Original expansion logic will go here
        # 1. Update parent node's network predictions and mark as expanded
        mcts_state_arg = dataclasses.replace(
            mcts_state_arg,
            is_expanded=mcts_state_arg.is_expanded.at[parent_node_idx].set(True),
            # (a) Store the parent_hidden_state passed to this function
            hidden_state=mcts_state_arg.hidden_state.at[parent_node_idx].set(parent_hidden_state),
            policy_logits=mcts_state_arg.policy_logits.at[parent_node_idx].set(network_policy_logits),
            value_from_network=mcts_state_arg.value_from_network.at[parent_node_idx].set(network_value),
            # (b) Store the network_reward for reaching this parent_node_idx
            reward_from_parent_action=mcts_state_arg.reward_from_parent_action.at[parent_node_idx].set(network_reward)
        )

        # Calculate prior probabilities for children
        child_priors = jax.nn.softmax(network_policy_logits)

        # Point (d) will modify this block
        # Apply legal_actions_mask to priors only if config.mask_illegal_actions is True
        if config.mask_illegal_actions and legal_actions_mask is not None:
            child_priors = child_priors * legal_actions_mask
            # Normalize only if there are legal actions, otherwise priors remain 0 for illegal actions
            sum_child_priors = jnp.sum(child_priors)
            child_priors = jax.lax.cond(
                sum_child_priors > 1e-9, # Check if sum is not zero (or very small)
                lambda x: x / sum_child_priors, # Normalize
                lambda x: x, # Otherwise, keep as is (e.g. all zeros if no legal actions)
                child_priors
            )

        mcts_state_arg = dataclasses.replace(
            mcts_state_arg,
            prior_probabilities=mcts_state_arg.prior_probabilities.at[parent_node_idx].set(child_priors)
        )

        # 2. Sequentially create child nodes
        current_mcts_state_children_loop = mcts_state_arg
        
        def loop_body_create_child(action_idx, loop_val):
            current_mcts_state_in_loop, current_key_in_loop = loop_val
            
            # Point (d) will modify this block for config.mask_illegal_actions
            # Determine if action is considered legal for child creation based on config and mask
            is_action_considered_legal_by_mask = True
            if legal_actions_mask is not None:
                is_action_considered_legal_by_mask = legal_actions_mask[action_idx] > 0
            
            # Action is only truly illegal for creation if masking is enabled and mask says so.
            # If masking is disabled, all actions are considered creatable (subject to space).
            is_action_creatable = jax.lax.cond(
                config.mask_illegal_actions,
                lambda: is_action_considered_legal_by_mask, # If masking enabled, respect mask
                lambda: True # If masking disabled, always consider creatable for loop logic
            )

            has_space = current_mcts_state_in_loop.num_allocated_nodes < current_mcts_state_in_loop.parent_indices.shape[0]

            def create_child_node(state_operand: MCTSState, key_for_child_inference: chex.PRNGKey) -> MCTSState:
                child_node_idx = state_operand.num_allocated_nodes
                action_tensor = jnp.array([action_idx], dtype=jnp.int32)
                
                recurrent_model_out: ModelOutput = model.recurrent_inference(
                    jnp.expand_dims(parent_hidden_state, axis=0),
                    jnp.expand_dims(action_tensor, axis=0),
                    key_for_child_inference,
                    training=False
                )
                child_hidden_state = recurrent_model_out.hidden_state.squeeze(axis=0)
                child_reward = recurrent_model_out.reward.squeeze(axis=0)
                child_policy_logits = recurrent_model_out.policy_logits.squeeze(axis=0)
                child_value = recurrent_model_out.value.squeeze(axis=0)

                new_parent_indices = state_operand.parent_indices.at[child_node_idx].set(parent_node_idx)
                new_action_from_parent = state_operand.action_from_parent.at[child_node_idx].set(action_idx)
                new_children_node_indices = state_operand.children_node_indices.at[parent_node_idx, action_idx].set(child_node_idx)
                new_hidden_state_arr = state_operand.hidden_state.at[child_node_idx].set(child_hidden_state)
                new_policy_logits_arr = state_operand.policy_logits.at[child_node_idx].set(child_policy_logits)
                new_value_from_network_arr = state_operand.value_from_network.at[child_node_idx].set(child_value)
                new_reward_from_parent_action_arr = state_operand.reward_from_parent_action.at[child_node_idx].set(child_reward)

                return dataclasses.replace(
                    state_operand,
                    parent_indices=new_parent_indices,
                    action_from_parent=new_action_from_parent,
                    children_node_indices=new_children_node_indices,
                    hidden_state=new_hidden_state_arr,
                    policy_logits=new_policy_logits_arr,
                    value_from_network=new_value_from_network_arr,
                    reward_from_parent_action=new_reward_from_parent_action_arr,
                    num_allocated_nodes=state_operand.num_allocated_nodes + 1
                )

            key_child, new_key_for_loop_body = jax.random.split(current_key_in_loop)
            
            updated_state = jax.lax.cond(
                is_action_creatable & has_space, # Use is_action_creatable here
                lambda s_param: create_child_node(s_param, key_child),
                lambda s_param: s_param, 
                current_mcts_state_in_loop
            )
            return updated_state, new_key_for_loop_body

        final_mcts_state_children_loop, _ = jax.lax.fori_loop(
            0, 
            mcts_state_arg.num_actions, 
            loop_body_create_child, 
            (current_mcts_state_children_loop, key) 
        )
        return final_mcts_state_children_loop

    # Actual logic with idempotency guard using jax.lax.cond
    return jax.lax.cond(
        mcts_state.is_expanded[parent_node_idx],
        lambda state_true: state_true,  # If already expanded, return state as is
        lambda state_false: expansion_logic(state_false),  # If not expanded, apply expansion logic
        mcts_state
    ) 