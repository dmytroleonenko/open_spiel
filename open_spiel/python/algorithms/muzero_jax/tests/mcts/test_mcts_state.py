import jax
import jax.numpy as jnp
import pytest
from open_spiel.python.algorithms.muzero_jax.mcts.mcts_state import MCTSState, INVALID_NODE_INDEX


def test_mcts_state_new_state_initialization():
    max_nodes = 100
    # Example hidden state: could be flat vector or multi-dimensional (e.g., image-like)
    hidden_state_example_flat = jnp.zeros((10,), dtype=jnp.float32)
    hidden_state_example_conv = jnp.zeros((4, 4, 16), dtype=jnp.float32)
    num_actions = 5
    discount = 0.99

    for hs_example in [hidden_state_example_flat, hidden_state_example_conv]:
        state = MCTSState.new_state(
            max_nodes=max_nodes,
            hidden_state_example=hs_example,
            num_actions=num_actions,
            discount=discount
        )

        assert state.parent_indices.shape == (max_nodes,)
        assert state.parent_indices.dtype == jnp.int32
        assert jnp.all(state.parent_indices == INVALID_NODE_INDEX)

        assert state.action_from_parent.shape == (max_nodes,)
        assert state.action_from_parent.dtype == jnp.int32
        assert jnp.all(state.action_from_parent == INVALID_NODE_INDEX)

        assert state.children_node_indices.shape == (max_nodes, num_actions)
        assert state.children_node_indices.dtype == jnp.int32
        assert jnp.all(state.children_node_indices == INVALID_NODE_INDEX)

        assert state.visit_count.shape == (max_nodes,)
        assert state.visit_count.dtype == jnp.int32
        assert jnp.all(state.visit_count == 0)

        assert state.value_sum.shape == (max_nodes,)
        assert state.value_sum.dtype == jnp.float32
        assert jnp.all(state.value_sum == 0.0)

        assert state.reward_from_parent_action.shape == (max_nodes,)
        assert state.reward_from_parent_action.dtype == jnp.float32
        assert jnp.all(state.reward_from_parent_action == 0.0)

        expected_hs_shape = (max_nodes, *hs_example.shape)
        assert state.hidden_state.shape == expected_hs_shape
        assert state.hidden_state.dtype == hs_example.dtype
        assert jnp.all(state.hidden_state == 0)

        assert state.policy_logits.shape == (max_nodes, num_actions)
        assert state.policy_logits.dtype == jnp.float32
        assert jnp.all(state.policy_logits == 0.0)

        assert state.prior_probabilities.shape == (max_nodes, num_actions)
        assert state.prior_probabilities.dtype == jnp.float32
        assert jnp.all(state.prior_probabilities == 0.0)

        assert state.value_from_network.shape == (max_nodes,)
        assert state.value_from_network.dtype == jnp.float32
        assert jnp.all(state.value_from_network == 0.0)

        assert state.is_expanded.shape == (max_nodes,)
        assert state.is_expanded.dtype == jnp.bool_
        assert jnp.all(state.is_expanded == False)

        assert state.num_allocated_nodes.shape == ()
        assert state.num_allocated_nodes.dtype == jnp.int32
        assert state.num_allocated_nodes.item() == 0

        assert state.min_max_stats_minimum.shape == ()
        assert state.min_max_stats_minimum.dtype == jnp.float32
        assert state.min_max_stats_minimum.item() == jnp.inf

        assert state.min_max_stats_maximum.shape == ()
        assert state.min_max_stats_maximum.dtype == jnp.float32
        assert state.min_max_stats_maximum.item() == -jnp.inf
        
        assert state.discount.shape == ()
        assert state.discount.dtype == jnp.float32
        assert jnp.isclose(state.discount.item(), discount)

        assert state.num_actions.shape == ()
        assert state.num_actions.dtype == jnp.int32
        assert state.num_actions.item() == num_actions 