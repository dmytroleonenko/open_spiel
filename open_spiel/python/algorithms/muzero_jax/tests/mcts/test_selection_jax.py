import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
import dataclasses

from open_spiel.python.algorithms.muzero_jax.mcts.mcts_state import MCTSState, INVALID_NODE_INDEX
from open_spiel.python.algorithms.muzero_jax.mcts.core import MCTSConfig
from open_spiel.python.algorithms.muzero_jax.mcts.selection_jax import select_child_jax, EPSILON

# Default config for tests
DEFAULT_NUM_ACTIONS = 3
DEFAULT_MAX_NODES = 10
ROOT_INDEX = 0

@pytest.fixture
def base_config() -> MCTSConfig:
    return MCTSConfig(
        num_simulations=10,
        num_actions=DEFAULT_NUM_ACTIONS,
        discount=0.99,
        c_base=19652,
        c_init=1.25,
        value_minmax_delta=0.01, # Small delta for MinMaxStats initialization
        dirichlet_alpha=0.3,
        dirichlet_exploration_fraction=0.25,
        temperature=1.0
    )

@pytest.fixture
def dummy_hidden_state() -> jnp.ndarray:
    return jnp.zeros((4,)) # Example hidden state shape

@pytest.fixture
def initial_mcts_state(base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray) -> MCTSState:
    return MCTSState.new_state(
        max_nodes=DEFAULT_MAX_NODES,
        hidden_state_example=dummy_hidden_state,
        num_actions=base_config.num_actions,
        discount=base_config.discount,
        min_max_delta=base_config.value_minmax_delta
    )

def test_select_child_no_children_expanded(initial_mcts_state: MCTSState, base_config: MCTSConfig):
    """Test selection when root has priors but no children are actually linked/expanded in MCTSState."""
    key = jax.random.PRNGKey(0)
    
    # Setup root node with some priors
    priors = jnp.array([0.5, 0.3, 0.2])
    state = dataclasses.replace(initial_mcts_state,
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(priors),
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(1) # Visited once to make pb_c calculations non-trivial
    )

    # Since children_node_indices are all INVALID_NODE_INDEX, no child should be selected
    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)
    
    assert selected_child_idx == INVALID_NODE_INDEX
    assert selected_action == -1

def test_select_child_single_valid_child(initial_mcts_state: MCTSState, base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray):
    """Test selection when only one child is valid and should be chosen."""
    key = jax.random.PRNGKey(1)
    
    # Child 1 (action 0)
    child1_idx = 1 
    
    state = dataclasses.replace(initial_mcts_state,
        num_allocated_nodes=jnp.array(2, dtype=jnp.int32), # Root + 1 child
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(jnp.array([0.8, 0.1, 0.1])),
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(10), # Parent visited 10 times
        
        # Child 1 (action 0) properties
        children_node_indices=initial_mcts_state.children_node_indices.at[ROOT_INDEX, 0].set(child1_idx),
        parent_indices=initial_mcts_state.parent_indices.at[child1_idx].set(ROOT_INDEX),
        action_from_parent=initial_mcts_state.action_from_parent.at[child1_idx].set(0),
        hidden_state=initial_mcts_state.hidden_state.at[child1_idx].set(dummy_hidden_state),
        reward_from_parent_action=initial_mcts_state.reward_from_parent_action.at[child1_idx].set(0.5), # Reward for action 0
        value_sum=initial_mcts_state.value_sum.at[child1_idx].set(0.7), # Value sum for child
        min_max_stats_minimum=jnp.array(0.0), # Set individual fields
        min_max_stats_maximum=jnp.array(1.0) # Set individual fields
    )
    state = dataclasses.replace(state,
        visit_count=initial_mcts_state.visit_count.at[child1_idx].set(1) # Child visited once
    )

    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)
    
    assert selected_child_idx == child1_idx
    assert selected_action == 0

def test_select_child_favors_higher_ucb(initial_mcts_state: MCTSState, base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray):
    """Test that the child with the highest UCB score is selected."""
    key = jax.random.PRNGKey(2)

    child1_idx, child2_idx = 1, 2

    # Setup state: parent (root) and two children
    state = dataclasses.replace(initial_mcts_state,
        num_allocated_nodes=jnp.array(3, dtype=jnp.int32),
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(jnp.array([0.4, 0.3, 0.3])), # Priors for actions 0, 1, 2
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(20), # Parent visited 20 times
        min_max_stats_minimum=jnp.array(0.0),
        min_max_stats_maximum=jnp.array(1.0) # Normalized values in [0,1]
    )

    # Child 1 (action 0) - Lower Q, lower visits (higher exploration bonus initially)
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 0].set(child1_idx),
        parent_indices=state.parent_indices.at[child1_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child1_idx].set(0),
        hidden_state=state.hidden_state.at[child1_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child1_idx].set(0.1),
        visit_count=state.visit_count.at[child1_idx].set(2),
        value_sum=state.value_sum.at[child1_idx].set(0.2) # Q_raw = 0.1 (reward) + 0.99 * (0.2/2) = 0.1 + 0.99 * 0.1 = 0.199
                                                       # Q_norm = (0.199 - 0) / (1-0) = 0.199
    )

    # Child 2 (action 1) - Higher Q, more visits (lower exploration bonus)
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 1].set(child2_idx),
        parent_indices=state.parent_indices.at[child2_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child2_idx].set(1),
        hidden_state=state.hidden_state.at[child2_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child2_idx].set(0.8),
        visit_count=state.visit_count.at[child2_idx].set(10),
        value_sum=state.value_sum.at[child2_idx].set(7.0) # Q_raw = 0.8 + 0.99 * (7.0/10) = 0.8 + 0.99 * 0.7 = 0.8 + 0.693 = 1.493
                                                        # Q_norm = (1.493 - 0) / (1-0) = 1.493 (capped at 1 by MinMax) -> 1.0
    )
    
    # Calculate UCB for child 1 (action 0)
    # Parent visits N(s) = 20. sqrt(N(s)) approx 4.47
    # Child 1 visits N(s,a0) = 2. Prior P(s,a0) = 0.4
    # pb_c = log((20 + c_base + EPSILON)/(c_base+EPSILON)) + c_init 
    #      = log(20 + base_config.c_base + EPSILON)/(base_config.c_base+EPSILON)) + base_config.c_init 
    #      = log(19672.000001 / 19652.000001) + 1.25
    #      = log(1.0010176) + 1.25 approx 0.001017 + 1.25 = 1.251017
    # Q_norm(s,a0) = 0.199
    # Exploration = pb_c * P(s,a0) * sqrt(N(s)) / (1 + N(s,a0))
    #             = 1.251017 * 0.4 * 4.47213 / (1 + 2)
    #             = 1.251017 * 0.4 * 4.47213 / 3 
    #             = 0.5004068 * 1.49071 = 0.7459
    # UCB(s,a0) = 0.199 + 0.7459 = 0.9449

    # Calculate UCB for child 2 (action 1)
    # Child 2 visits N(s,a1) = 10. Prior P(s,a1) = 0.3
    # Q_norm(s,a1) = 1.0 (normalized from 1.493, capped by MinMax(0,1))
    # Exploration = pb_c * P(s,a1) * sqrt(N(s)) / (1 + N(s,a1))
    #             = 1.251017 * 0.3 * 4.47213 / (1 + 10)
    #             = 1.251017 * 0.3 * 4.47213 / 11
    #             = 0.3753051 * 0.406557 = 0.1526
    # UCB(s,a1) = 1.0 + 0.1526 = 1.1526

    # Child 2 should have higher UCB
    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)
    
    assert selected_child_idx == child2_idx
    assert selected_action == 1

def test_select_child_all_children_equal_force_prior_tie_break(initial_mcts_state: MCTSState, base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray):
    """If all Q values and visit counts are equal, selection should break ties by prior."""
    key = jax.random.PRNGKey(3)
    child1_idx, child2_idx, child3_idx = 1, 2, 3

    state = dataclasses.replace(initial_mcts_state,
        num_allocated_nodes=jnp.array(4, dtype=jnp.int32),
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(jnp.array([0.2, 0.5, 0.3])), # Action 1 has highest prior
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(3), # N(s) = 3
        min_max_stats_minimum=jnp.array(0.0),
        min_max_stats_maximum=jnp.array(1.0)
    )

    # All children have same reward, value_sum, visit_count
    # Q_raw = 0.0 + 0.99 * (0.0/1) = 0.0. Q_norm = 0.0
    # Exploration term will be: pb_c * prior * sqrt(N(s))/(1+N(s,a))
    # pb_c = log((3+c_base+EPSILON)/(c_base+EPSILON)) + c_init approx 1.25
    # sqrt(N(s)) = sqrt(3) approx 1.732
    # N(s,a) = 1 for all children. (1+N(s,a)) = 2
    # Exploration term approx 1.25 * prior * 1.732 / 2 = 1.25 * prior * 0.866 = 1.0825 * prior
    # UCB = 0.0 + 1.0825 * prior. So highest prior wins.
    
    for i, child_idx in enumerate([child1_idx, child2_idx, child3_idx]):
        state = dataclasses.replace(state,
            children_node_indices=state.children_node_indices.at[ROOT_INDEX, i].set(child_idx),
            parent_indices=state.parent_indices.at[child_idx].set(ROOT_INDEX),
            action_from_parent=state.action_from_parent.at[child_idx].set(i),
            hidden_state=state.hidden_state.at[child_idx].set(dummy_hidden_state),
            reward_from_parent_action=state.reward_from_parent_action.at[child_idx].set(0.0),
            visit_count=state.visit_count.at[child_idx].set(1), # Visited once
            value_sum=state.value_sum.at[child_idx].set(0.0)   # Value sum zero
        )
    
    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)
    
    assert selected_child_idx == child2_idx # Corresponds to action 1 (prior 0.5)
    assert selected_action == 1

def test_ucb_with_zero_parent_visits(initial_mcts_state: MCTSState, base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray):
    """Test UCB calculation when parent visit count is zero. Expect exploration term to dominate via priors."""
    key = jax.random.PRNGKey(4)
    child1_idx, child2_idx = 1, 2

    state = dataclasses.replace(initial_mcts_state,
        num_allocated_nodes=jnp.array(3, dtype=jnp.int32),
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(jnp.array([0.2, 0.8, 0.0])), # Action 1 (child2) has higher prior
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(0), # Crucial: Parent not visited
        min_max_stats_minimum=jnp.array(0.0), # Default, won't affect much as Q is 0
        min_max_stats_maximum=jnp.array(1.0)  # Default
    )
    # When parent_visit_count is 0:
    # log_arg = (0 + c_base + EPSILON) / (c_base + EPSILON) = 1.0
    # pb_c = jnp.log(1.0 + EPSILON) + c_init approx c_init (1.25)
    # sqrt_parent_visit_count = sqrt(EPSILON) approx 0.0
    # Exploration term: pb_c * prior * (sqrt(0)) / (1 + child_visits) = 0 if child_visits > 0
    # This makes exploration term 0 if sqrt_parent_visit_count is exactly 0 due to EPSILON handling.
    # The formula is P(s,a) * sqrt(N(s)) / (1 + N(s,a)). If N(s)=0, sqrt(N(s))=0, so exploration = 0.
    # Q values will be 0 since children are unvisited.
    # So all UCBs will be 0. Tie-breaking should pick the first valid child in this case.
    # Let's re-check reference formula: EfficientZero uses log((N(s) + base + 1)/base) + c_init
    # if N(s)=0, log((base+1)/base) + c_init. This is a positive constant.
    # And sqrt(N(s)) / (1+N(s,a)). If N(s)=0, this is 0. Exploration = 0.
    # If N(s,a)=0, exploration is pb_c * P(s,a) * sqrt(N(s)). If N(s)=0 also, then 0.
    # The code has parent_visit_count + EPSILON for log_arg numerator, and parent_visit_count + EPSILON for sqrt.
    # If parent_visit_count = 0, log_arg is approx 1, pb_c is approx c_init.
    # sqrt_parent_visit_count is approx sqrt(EPSILON).
    # exploration_term = c_init * prior * sqrt(EPSILON) / (1 + child_visit_count)
    # This will be very small but positive, proportional to prior.

    # Child 1 (action 0)
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 0].set(child1_idx),
        parent_indices=state.parent_indices.at[child1_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child1_idx].set(0),
        hidden_state=state.hidden_state.at[child1_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child1_idx].set(0.0),
        visit_count=state.visit_count.at[child1_idx].set(0), # Unvisited
        value_sum=state.value_sum.at[child1_idx].set(0.0)
    )
    # Child 2 (action 1)
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 1].set(child2_idx),
        parent_indices=state.parent_indices.at[child2_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child2_idx].set(1),
        hidden_state=state.hidden_state.at[child2_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child2_idx].set(0.0),
        visit_count=state.visit_count.at[child2_idx].set(0), # Unvisited
        value_sum=state.value_sum.at[child2_idx].set(0.0)
    )

    # Q_sa for both is 0. Normalized Q_sa is 0.
    # UCB = exploration_term = c_init * prior * sqrt(EPSILON) / 1.0
    # Child with higher prior (action 1) should be selected.
    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)

    assert selected_child_idx == child2_idx
    assert selected_action == 1

def test_select_child_with_minmax_normalization(initial_mcts_state: MCTSState, base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray):
    """Test selection with non-trivial MinMaxStats affecting Q value normalization."""
    key = jax.random.PRNGKey(5)
    child1_idx, child2_idx = 1, 2

    # MinMax will be (min=10, max=20), so delta is 10.
    min_max_stats_minimum = jnp.array(10.0)
    min_max_stats_maximum = jnp.array(20.0)
    state = dataclasses.replace(initial_mcts_state,
        num_allocated_nodes=jnp.array(3, dtype=jnp.int32),
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(jnp.array([0.5, 0.5, 0.0])),
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(10),
        min_max_stats_minimum=min_max_stats_minimum,
        min_max_stats_maximum=min_max_stats_maximum
    )
    
    # Child 1 (action 0)
    # Raw Q = 0 (reward) + 0.99 * (12/1) = 11.88.
    # Normalized Q = (11.88 - 10) / (20 - 10) = 1.88 / 10 = 0.188
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 0].set(child1_idx),
        parent_indices=state.parent_indices.at[child1_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child1_idx].set(0),
        hidden_state=state.hidden_state.at[child1_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child1_idx].set(0.0),
        visit_count=state.visit_count.at[child1_idx].set(1),
        value_sum=state.value_sum.at[child1_idx].set(12.0) 
    )

    # Child 2 (action 1)
    # Raw Q = 0 (reward) + 0.99 * (18/1) = 17.82
    # Normalized Q = (17.82 - 10) / (20 - 10) = 7.82 / 10 = 0.782
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 1].set(child2_idx),
        parent_indices=state.parent_indices.at[child2_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child2_idx].set(1),
        hidden_state=state.hidden_state.at[child2_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child2_idx].set(0.0),
        visit_count=state.visit_count.at[child2_idx].set(1), # Same visits as child 1
        value_sum=state.value_sum.at[child2_idx].set(18.0)
    )

    # Both children have same prior (0.5) and same visit count (1).
    # Exploration term will be identical for both.
    # UCB difference will come purely from normalized Q values.
    # Child 2 has higher normalized Q (0.782 vs 0.188)
    
    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)

    assert selected_child_idx == child2_idx
    assert selected_action == 1

def test_select_child_one_child_heavily_visited(initial_mcts_state: MCTSState, base_config: MCTSConfig, dummy_hidden_state: jnp.ndarray):
    """Test that a child with many visits gets a lower exploration bonus, potentially losing to a less visited child with a good prior/Q."""
    key = jax.random.PRNGKey(6)
    child1_idx, child2_idx = 1, 2

    state = dataclasses.replace(initial_mcts_state,
        num_allocated_nodes=jnp.array(3, dtype=jnp.int32),
        prior_probabilities=initial_mcts_state.prior_probabilities.at[ROOT_INDEX].set(jnp.array([0.6, 0.4, 0.0])), # Child 1 (action 0) higher prior
        visit_count=initial_mcts_state.visit_count.at[ROOT_INDEX].set(100), # Parent heavily visited
        min_max_stats_minimum=jnp.array(0.0),
        min_max_stats_maximum=jnp.array(1.0)
    )

    # Child 1 (action 0): High prior, but also heavily visited. Decent Q.
    # Q_raw = 0.5 + 0.99 * (45/90) = 0.5 + 0.99 * 0.5 = 0.5 + 0.495 = 0.995. Q_norm = 0.995
    # N(s)=100, sqrt(N(s))=10. N(s,a0)=90. P(s,a0)=0.6
    # pb_c = log((100+c_base+EPSILON)/(c_base+EPSILON)) + c_init 
    #      = log((100+base_config.c_base+EPSILON)/(base_config.c_base+EPSILON)) + base_config.c_init 
    #      approx log(1.005) + 1.25 = 0.005 + 1.25 = 1.255
    # Expl(a0) = 1.255 * 0.6 * 10 / (1+90) = 1.255 * 0.6 * 10 / 91 = 0.753 * 0.10989 = 0.08275
    # UCB(a0) = 0.995 + 0.08275 = 1.07775
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 0].set(child1_idx),
        parent_indices=state.parent_indices.at[child1_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child1_idx].set(0),
        hidden_state=state.hidden_state.at[child1_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child1_idx].set(0.5),
        visit_count=state.visit_count.at[child1_idx].set(90), 
        value_sum=state.value_sum.at[child1_idx].set(45.0) 
    )

    # Child 2 (action 1): Lower prior, but very few visits. Slightly lower Q.
    # Q_raw = 0.4 + 0.99 * (0.3/1) = 0.4 + 0.297 = 0.697. Q_norm = 0.697
    # N(s,a1)=1. P(s,a1)=0.4
    # Expl(a1) = 1.255 * 0.4 * 10 / (1+1) = 1.255 * 0.4 * 10 / 2 = 0.502 * 5 = 2.51
    # UCB(a1) = 0.697 + 2.51 = 3.207
    state = dataclasses.replace(state,
        children_node_indices=state.children_node_indices.at[ROOT_INDEX, 1].set(child2_idx),
        parent_indices=state.parent_indices.at[child2_idx].set(ROOT_INDEX),
        action_from_parent=state.action_from_parent.at[child2_idx].set(1),
        hidden_state=state.hidden_state.at[child2_idx].set(dummy_hidden_state),
        reward_from_parent_action=state.reward_from_parent_action.at[child2_idx].set(0.4),
        visit_count=state.visit_count.at[child2_idx].set(1), 
        value_sum=state.value_sum.at[child2_idx].set(0.3)
    )
    
    # Child 2 should be selected due to much higher exploration bonus.
    selected_child_idx, selected_action = select_child_jax(state, ROOT_INDEX, key, base_config)

    assert selected_child_idx == child2_idx
    assert selected_action == 1 