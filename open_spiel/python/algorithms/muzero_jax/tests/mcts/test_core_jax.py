import jax
import jax.numpy as jnp
import pytest
import chex
from typing import Tuple, Optional
import dataclasses # Import standard dataclasses

from open_spiel.python.algorithms.muzero_jax.mcts.mcts_state import MCTSState, INVALID_NODE_INDEX
from open_spiel.python.algorithms.muzero_jax.mcts.core import MCTSConfig
from open_spiel.python.algorithms.muzero_jax.mcts.core_jax import _expand_node_jax, prepare_initial_mcts_state, ROOT_INDEX
from open_spiel.python.algorithms.muzero_jax.types import MuZeroModel, ModelOutput

# Minimal mock for MuZeroNetwork (similar to the one in test_mcts.py)
class MockJaxMuZeroNetwork(MuZeroModel):
    def __init__(self, num_actions, hidden_state_shape, policy_val=0.5, value_val=0.5, reward_val=0.1, return_unbatched_output=False, malformed_reward_shape=False):
        self.num_actions = num_actions
        self.hidden_state_shape = hidden_state_shape # This is the unbatched shape
        self.policy_val = policy_val
        self.value_val = value_val
        self.reward_val = reward_val
        self.return_unbatched_output = return_unbatched_output
        self.malformed_reward_shape = malformed_reward_shape

    def initial_inference(
        self,
        observation: chex.ArrayTree,
        key: chex.PRNGKey, 
        training: bool
    ) -> ModelOutput:
        if observation.ndim == 0: # Scalar observation
            batch_size = 1
            # Expand scalar to be (1,) or (1,1) if model expects at least 1D array per batch item
            # For simplicity, assume the model can handle it or it's expanded before internal processing.
            # Here, we'll ensure the output shapes are consistent with batch_size=1.
            obs_for_shape = jnp.expand_dims(observation, axis=0) # Make it (1,) for shape determination
        else:
            batch_size = observation.shape[0]
            obs_for_shape = observation

        hidden_state_b = jnp.full((batch_size, *self.hidden_state_shape), 0.1, dtype=jnp.float32)
        reward_b = jnp.full((batch_size,), self.reward_val, dtype=jnp.float32)
        policy_logits_b = jnp.full((batch_size, self.num_actions), self.policy_val, dtype=jnp.float32)
        value_b = jnp.full((batch_size,), self.value_val, dtype=jnp.float32)

        if self.malformed_reward_shape:
            # Return a reward that will trigger the ValueError in core_jax.py
            reward_b = jnp.array([[self.reward_val, self.reward_val + 1]], dtype=jnp.float32) # e.g. shape (1,2)
            # Ensure other outputs are consistent for batch_size=1 if this path is taken
            if batch_size == 1:
                 hidden_state_b = jnp.full((1, *self.hidden_state_shape), 0.1, dtype=jnp.float32)
                 policy_logits_b = jnp.full((1, self.num_actions), self.policy_val, dtype=jnp.float32)
                 value_b = jnp.full((1,), self.value_val, dtype=jnp.float32)
            # else: batch_size > 1 with malformed reward is not specifically handled here, assume test uses batch_size=1

        if self.return_unbatched_output:
            if batch_size == 1:
                return ModelOutput(
                    hidden_state=hidden_state_b.squeeze(axis=0),
                    reward=reward_b.squeeze(axis=0),
                    policy_logits=policy_logits_b.squeeze(axis=0),
                    value=value_b.squeeze(axis=0)
                )
            else:
                # This case is tricky for unbatching logic testing, ideally test with batch_size=1
                return ModelOutput(hidden_state_b, reward_b, policy_logits_b, value_b) # Or raise error
        return ModelOutput(hidden_state_b, reward_b, policy_logits_b, value_b)

    def recurrent_inference(
        self,
        hidden_state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
        training: bool
    ) -> ModelOutput:
        batch_size = hidden_state.shape[0]
        next_hidden_state = hidden_state * 0.9 # Dummy operation
        reward = jnp.full((batch_size,), self.reward_val + 0.1, dtype=jnp.float32)
        policy_logits = jnp.full((batch_size, self.num_actions), self.policy_val + 0.1, dtype=jnp.float32)
        value = jnp.full((batch_size,), self.value_val + 0.1, dtype=jnp.float32)
        return ModelOutput(next_hidden_state, reward, policy_logits, value)

@pytest.fixture
def mock_config_fixture():
    return MCTSConfig(
        num_simulations=10, # Not used directly by prepare_initial_mcts_state
        num_actions=3,
        discount=0.99,
        q_init=0.0,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_exploration_fraction=0.25,
        c_base=19652.0,
        c_init=1.25,
        value_minmax_delta=1e-3,
        mask_illegal_actions=True,
        max_depth=None
    )

@pytest.fixture
def mock_model_fixture(mock_config_fixture):
    return MockJaxMuZeroNetwork(
        num_actions=mock_config_fixture.num_actions, 
        hidden_state_shape=(4,) # Example 1D hidden state
    )

@pytest.fixture
def mock_model_fixture_unbatched_output(mock_config_fixture):
    return MockJaxMuZeroNetwork(
        num_actions=mock_config_fixture.num_actions, 
        hidden_state_shape=(4,), # Example 1D hidden state
        return_unbatched_output=True
    )

def test_prepare_initial_mcts_state_basic(mock_model_fixture, mock_config_fixture):
    key = jax.random.PRNGKey(0)
    initial_observation_flat = jnp.zeros((10,), dtype=jnp.float32) # Example obs
    initial_observation_2d_non_batched = jnp.zeros((8, 8), dtype=jnp.float32) # New: For ndim == 2 and shape[0] != 1
    initial_observation_conv = jnp.zeros((8, 8, 3), dtype=jnp.float32) # Example conv obs
    max_nodes = 50

    for obs_idx, initial_observation in enumerate([initial_observation_flat, initial_observation_2d_non_batched, initial_observation_conv]):
        mcts_state = prepare_initial_mcts_state(
            key=key,
            model=mock_model_fixture,
            initial_observation=initial_observation,
            config=mock_config_fixture,
            max_nodes=max_nodes,
            legal_actions_mask=None # Test without mask first
        )

        # Check MCTSState general properties (already tested in test_mcts_state.py but good for context)
        assert mcts_state.num_allocated_nodes.item() == 1
        assert jnp.isclose(mcts_state.discount.item(), mock_config_fixture.discount)
        assert mcts_state.num_actions.item() == mock_config_fixture.num_actions

        # Check root node (index ROOT_INDEX = 0)
        # Parent and action from parent should be invalid
        assert mcts_state.parent_indices[ROOT_INDEX].item() == -1 # INVALID_NODE_INDEX is -1
        assert mcts_state.action_from_parent[ROOT_INDEX].item() == -1
        
        # Visit count and value sum should be zero initially
        assert mcts_state.visit_count[ROOT_INDEX].item() == 0
        assert mcts_state.value_sum[ROOT_INDEX].item() == 0.0

        # Check stored network predictions for root
        expected_root_hidden_state = jnp.full(mock_model_fixture.hidden_state_shape, 0.1, dtype=jnp.float32)
        chex.assert_trees_all_close(mcts_state.hidden_state[ROOT_INDEX], expected_root_hidden_state, atol=1e-6)
        
        expected_policy_logits = jnp.full((mock_config_fixture.num_actions,), mock_model_fixture.policy_val, dtype=jnp.float32)
        chex.assert_trees_all_close(mcts_state.policy_logits[ROOT_INDEX], expected_policy_logits, atol=1e-6)

        expected_value = mock_model_fixture.value_val
        assert jnp.isclose(mcts_state.value_from_network[ROOT_INDEX].item(), expected_value)

        expected_reward = mock_model_fixture.reward_val
        assert jnp.isclose(mcts_state.reward_from_parent_action[ROOT_INDEX].item(), expected_reward)

        # Check prior probabilities (softmax of logits + dirichlet noise)
        # Without legal mask, dirichlet noise is applied to all actions
        # For this test, dirichlet_alpha=0.3, fraction=0.25
        # mock_model_fixture.policy_val is 0.5 for all actions.
        # policy_logits = [0.5, 0.5, 0.5]
        # policy_probs = softmax([0.5, 0.5, 0.5]) = [1/3, 1/3, 1/3]
        raw_probs = jax.nn.softmax(expected_policy_logits)
        key_noise, _ = jax.random.split(key)
        # Note: The key used here must match the one inside prepare_initial_mcts_state for noise generation
        # This is a bit tricky to test precisely without refactoring prepare_initial_mcts_state to output the key it used or take a subkey.
        # For now, let's check general properties or make noise predictable by fixing the key within the function for testing.
        # The function prepare_initial_mcts_state splits the input key, so we can try to replicate.
        
        # Replicating the noise logic from prepare_initial_mcts_state
        alpha_val = mock_config_fixture.dirichlet_alpha
        exploration_fraction = mock_config_fixture.dirichlet_exploration_fraction
        num_actions_val = mock_config_fixture.num_actions

        dirichlet_noise_expected = jax.random.dirichlet(
                key_noise, # This is the key that will be used for dirichlet noise inside the function
                alpha=jnp.full(num_actions_val, alpha_val)
            )
        expected_priors = (
            (1 - exploration_fraction) * raw_probs
            + exploration_fraction * dirichlet_noise_expected
        )
        chex.assert_trees_all_close(mcts_state.prior_probabilities[ROOT_INDEX], expected_priors, atol=1e-6)
        
        # Root node is allocated, but not expanded
        assert mcts_state.is_expanded[ROOT_INDEX].item() == False


def test_prepare_initial_mcts_state_with_legal_mask(mock_model_fixture, mock_config_fixture):
    key = jax.random.PRNGKey(1)
    initial_observation = jnp.zeros((10,), dtype=jnp.float32) 
    max_nodes = 50
    legal_actions_mask = jnp.array([True, False, True], dtype=jnp.bool_)

    mcts_state = prepare_initial_mcts_state(
        key=key,
        model=mock_model_fixture,
        initial_observation=initial_observation,
        config=mock_config_fixture,
        max_nodes=max_nodes,
        legal_actions_mask=legal_actions_mask
    )

    assert mcts_state.num_allocated_nodes.item() == 1
    
    # Check prior probabilities with legal mask
    expected_policy_logits = jnp.full((mock_config_fixture.num_actions,), mock_model_fixture.policy_val, dtype=jnp.float32)
    raw_probs = jax.nn.softmax(expected_policy_logits)
    key_noise, _ = jax.random.split(key) 

    alpha_val = mock_config_fixture.dirichlet_alpha
    exploration_fraction = mock_config_fixture.dirichlet_exploration_fraction
    num_actions_val = mock_config_fixture.num_actions

    dirichlet_noise_all_expected = jax.random.dirichlet(
        key_noise,
        alpha=jnp.full(num_actions_val, alpha_val)
    )
    masked_noise_expected = jnp.where(legal_actions_mask, dirichlet_noise_all_expected, 0.0)
    
    mixed_priors_expected = (
        (1 - exploration_fraction) * raw_probs
        + exploration_fraction * masked_noise_expected
    )
    final_priors_expected = jnp.where(legal_actions_mask, mixed_priors_expected, 0.0)
    # If normalization is added in prepare_initial_mcts_state, it should be added here too.
    # sum_final_priors = jnp.sum(final_priors_expected)
    # if sum_final_priors > 0:
    #    final_priors_expected /= sum_final_priors
        
    chex.assert_trees_all_close(mcts_state.prior_probabilities[ROOT_INDEX], final_priors_expected, atol=1e-6)


def test_prepare_initial_mcts_state_no_dirichlet(mock_model_fixture, mock_config_fixture):
    key = jax.random.PRNGKey(2)
    initial_observation = jnp.zeros((10,), dtype=jnp.float32)
    max_nodes = 50
    
    # Modify config for no dirichlet noise
    config_no_noise = dataclasses.replace(mock_config_fixture, dirichlet_alpha=0.0) # Use standard dataclasses.replace

    mcts_state = prepare_initial_mcts_state(
        key=key,
        model=mock_model_fixture,
        initial_observation=initial_observation,
        config=config_no_noise,
        max_nodes=max_nodes,
        legal_actions_mask=None
    )
    assert mcts_state.num_allocated_nodes.item() == 1
    expected_policy_logits = jnp.full((mock_config_fixture.num_actions,), mock_model_fixture.policy_val, dtype=jnp.float32)
    expected_priors = jax.nn.softmax(expected_policy_logits)
    chex.assert_trees_all_close(mcts_state.prior_probabilities[ROOT_INDEX], expected_priors, atol=1e-6)

    # Test with exploration fraction 0 as well
    config_no_noise_frac = dataclasses.replace(mock_config_fixture, dirichlet_exploration_fraction=0.0) # Use standard dataclasses.replace
    mcts_state_frac = prepare_initial_mcts_state(
        key=key,
        model=mock_model_fixture,
        initial_observation=initial_observation,
        config=config_no_noise_frac,
        max_nodes=max_nodes,
        legal_actions_mask=None
    )
    chex.assert_trees_all_close(mcts_state_frac.prior_probabilities[ROOT_INDEX], expected_priors, atol=1e-6) 

# Add new test case for 0-dim observation and unbatched model output
def test_prepare_initial_mcts_state_edge_cases(mock_model_fixture_unbatched_output, mock_config_fixture):
    key = jax.random.PRNGKey(3)
    max_nodes = 50

    # Test 1: 0-dimensional observation (should hit line 42 in core_jax.py)
    initial_observation_scalar = jnp.array(1.0, dtype=jnp.float32)
    # For this, use a model that returns batched output to test the interaction
    model_batched_output = MockJaxMuZeroNetwork(
        num_actions=mock_config_fixture.num_actions, 
        hidden_state_shape=(4,)
    )
    mcts_state_scalar_obs = prepare_initial_mcts_state(
        key=key,
        model=model_batched_output, # Use model that returns batched output
        initial_observation=initial_observation_scalar,
        config=mock_config_fixture,
        max_nodes=max_nodes,
        legal_actions_mask=None
    )
    assert mcts_state_scalar_obs.num_allocated_nodes.item() == 1 # Basic check
    # Check that hidden state was correctly processed (should be unbatched in state)
    expected_hs_shape_scalar = model_batched_output.hidden_state_shape
    assert mcts_state_scalar_obs.hidden_state[ROOT_INDEX].shape == expected_hs_shape_scalar


    # Test 2: Model returns unbatched output (should hit else clauses in unbatching logic)
    key_unbatched, _ = jax.random.split(key)
    initial_observation_for_unbatched_test = jnp.zeros((10,), dtype=jnp.float32) 
    mcts_state_unbatched_model = prepare_initial_mcts_state(
        key=key_unbatched,
        model=mock_model_fixture_unbatched_output, # Model returns unbatched data
        initial_observation=initial_observation_for_unbatched_test,
        config=mock_config_fixture,
        max_nodes=max_nodes,
        legal_actions_mask=None
    )
    assert mcts_state_unbatched_model.num_allocated_nodes.item() == 1 # Basic check
    # Check that network outputs stored are indeed the unbatched ones
    expected_hs_unbatched = jnp.full(mock_model_fixture_unbatched_output.hidden_state_shape, 0.1, dtype=jnp.float32)
    chex.assert_trees_all_close(mcts_state_unbatched_model.hidden_state[ROOT_INDEX], expected_hs_unbatched, atol=1e-6)
    
    expected_policy_logits_unbatched = jnp.full((mock_config_fixture.num_actions,), mock_model_fixture_unbatched_output.policy_val, dtype=jnp.float32)
    chex.assert_trees_all_close(mcts_state_unbatched_model.policy_logits[ROOT_INDEX], expected_policy_logits_unbatched, atol=1e-6)

    expected_value_unbatched = mock_model_fixture_unbatched_output.value_val
    assert jnp.isclose(mcts_state_unbatched_model.value_from_network[ROOT_INDEX].item(), expected_value_unbatched) 

def test_prepare_initial_mcts_state_malformed_reward(mock_config_fixture):
    key = jax.random.PRNGKey(4)
    max_nodes = 10
    initial_observation = jnp.zeros((5,), dtype=jnp.float32)

    model_malformed_reward = MockJaxMuZeroNetwork(
        num_actions=mock_config_fixture.num_actions,
        hidden_state_shape=(4,),
        malformed_reward_shape=True
    )

    with pytest.raises(ValueError, match="Unexpected shape for root_reward_batched"):
        prepare_initial_mcts_state(
            key=key,
            model=model_malformed_reward,
            initial_observation=initial_observation,
            config=mock_config_fixture,
            max_nodes=max_nodes,
            legal_actions_mask=None
        ) 

# Dummy MuZeroModel for testing
@dataclasses.dataclass
class DummyModel(MuZeroModel):
    num_actions: int
    hidden_state_dim: int = 4 # Example dimension

    def initial_inference(self, observation: chex.Array, key: chex.PRNGKey, training: bool) -> ModelOutput:
        batch_size = observation.shape[0]
        hidden_state = jnp.zeros((batch_size, self.hidden_state_dim))
        reward = jnp.zeros((batch_size,))
        policy_logits = jnp.zeros((batch_size, self.num_actions))
        value = jnp.zeros((batch_size,))
        return ModelOutput(hidden_state, reward, policy_logits, value)

    def recurrent_inference(self, hidden_state: chex.Array, action: chex.Array, key: chex.PRNGKey, training: bool) -> ModelOutput:
        batch_size = hidden_state.shape[0]
        next_hidden_state = jnp.zeros((batch_size, self.hidden_state_dim))
        # Simulate reward based on action for diversity in tests
        reward = action.astype(jnp.float32).squeeze(axis=-1) # Assuming action is (B, 1)
        # Simulate different policy & value for children
        policy_logits = jax.random.normal(key, (batch_size, self.num_actions))
        value = jax.random.normal(key, (batch_size,))
        return ModelOutput(next_hidden_state, reward, policy_logits, value)

@pytest.fixture
def dummy_config() -> MCTSConfig:
    return MCTSConfig(
        num_simulations=10,
        num_actions=3,
        discount=0.99,
        dirichlet_alpha=0.25,
        dirichlet_exploration_fraction=0.1,
        c_base=19652,
        c_init=1.25,
        value_minmax_delta=0.01
    )

@pytest.fixture
def dummy_model(dummy_config: MCTSConfig) -> DummyModel:
    return DummyModel(num_actions=dummy_config.num_actions)

@pytest.fixture
def initial_mcts_state_for_expansion(dummy_model: DummyModel, dummy_config: MCTSConfig) -> MCTSState:
    key = jax.random.PRNGKey(0)
    max_nodes = 20 # Sufficient for these tests
    # Use a dummy observation, e.g. a single feature or a small image representation
    # For DummyModel, the content doesn't matter, only batch dim and potentially overall shape if model used it.
    # Let's assume a simple 1D observation for simplicity.
    initial_observation = jnp.zeros((dummy_model.hidden_state_dim,)) 
    
    # Prepare an initial state, which includes initializing the root node (node 0)
    state = prepare_initial_mcts_state(
        key=key,
        model=dummy_model,
        initial_observation=initial_observation,
        config=dummy_config,
        max_nodes=max_nodes,
        legal_actions_mask=None # Assume all actions legal at root for this setup
    )
    # num_allocated_nodes will be 1 (root)
    return state


def test_expand_node_basic(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    key = jax.random.PRNGKey(1)
    state = initial_mcts_state_for_expansion
    parent_idx = ROOT_INDEX # Expand the root node

    # Get parent hidden state (already computed by prepare_initial_mcts_state for root)
    parent_hidden_state = state.hidden_state[parent_idx]
    
    # Network predictions for the parent (root) - these were set by prepare_initial_mcts_state
    # For _expand_node_jax, network_policy_logits, network_value, network_reward are those of the node being expanded.
    # For the root, these are from initial_inference.
    # Let's re-fetch them from the state to ensure consistency with what expand expects.
    network_policy_logits = state.policy_logits[parent_idx] # Logits for children of parent_idx
    network_value = state.value_from_network[parent_idx]    # Value of parent_idx
    network_reward = state.reward_from_parent_action[parent_idx] # Reward for reaching parent_idx (0 for root)

    expanded_state = _expand_node_jax(
        mcts_state=state,
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state,
        network_policy_logits=network_policy_logits, 
        network_value=network_value, 
        network_reward=network_reward,
        model=dummy_model,
        key=key,
        config=dummy_config,
        legal_actions_mask=None # Assume all actions are legal
    )

    # 1. Parent node should be marked as expanded
    assert expanded_state.is_expanded[parent_idx].item() is True

    # 2. Children should be created
    # num_actions = dummy_config.num_actions
    expected_num_children = dummy_config.num_actions
    # num_allocated_nodes = root (1) + num_children (3) = 4
    assert expanded_state.num_allocated_nodes.item() == 1 + expected_num_children

    child_base_idx = 1 # Children are allocated from index 1 onwards after root
    for i in range(expected_num_children):
        action = i
        child_idx = expanded_state.children_node_indices[parent_idx, action].item()
        
        assert child_idx == child_base_idx + i
        assert expanded_state.parent_indices[child_idx].item() == parent_idx
        assert expanded_state.action_from_parent[child_idx].item() == action
        
        # Check that child's network predictions were stored from recurrent_inference
        # These would be somewhat random due to DummyModel's recurrent_inference
        assert expanded_state.value_from_network[child_idx].item() != 0 # Default is 0, should be updated
        # Reward for child is reward for (parent, action_to_child)
        # DummyModel.recurrent_inference sets reward = action index
        assert expanded_state.reward_from_parent_action[child_idx].item() == float(action)
        
        assert expanded_state.visit_count[child_idx].item() == 0
        assert expanded_state.is_expanded[child_idx].item() is False
        # Policy logits for the child node should also be populated (for its future expansion)
        assert jnp.any(expanded_state.policy_logits[child_idx] != 0)

    # 3. Parent's prior probabilities should be set (softmax of network_policy_logits)
    expected_priors = jax.nn.softmax(network_policy_logits)
    chex.assert_trees_all_close(expanded_state.prior_probabilities[parent_idx], expected_priors, atol=1e-6)


def test_expand_node_with_legal_actions_mask(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    key = jax.random.PRNGKey(2)
    state = initial_mcts_state_for_expansion
    parent_idx = ROOT_INDEX
    parent_hidden_state = state.hidden_state[parent_idx]
    network_policy_logits = state.policy_logits[parent_idx]
    network_value = state.value_from_network[parent_idx]
    network_reward = state.reward_from_parent_action[parent_idx]

    # Only actions 0 and 2 are legal
    legal_actions_mask = jnp.array([1, 0, 1], dtype=jnp.bool_)
    num_legal_actions = jnp.sum(legal_actions_mask).item()

    expanded_state = _expand_node_jax(
        mcts_state=state,
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state,
        network_policy_logits=network_policy_logits,
        network_value=network_value,
        network_reward=network_reward,
        model=dummy_model,
        key=key,
        config=dummy_config,
        legal_actions_mask=legal_actions_mask
    )

    assert expanded_state.is_expanded[parent_idx].item() is True
    # num_allocated_nodes = root (1) + num_legal_children (2) = 3
    assert expanded_state.num_allocated_nodes.item() == 1 + num_legal_actions

    child_node_count = 0
    child_base_idx = 1 # Children start allocating from index 1

    for action_idx in range(dummy_config.num_actions):
        child_node_assigned_idx = expanded_state.children_node_indices[parent_idx, action_idx].item()
        if legal_actions_mask[action_idx]:
            # This should be the (child_node_count)-th allocated child after root
            expected_child_actual_idx = child_base_idx + child_node_count
            assert child_node_assigned_idx == expected_child_actual_idx
            assert expanded_state.parent_indices[expected_child_actual_idx].item() == parent_idx
            assert expanded_state.action_from_parent[expected_child_actual_idx].item() == action_idx
            # DummyModel.recurrent_inference sets reward = action index
            assert expanded_state.reward_from_parent_action[expected_child_actual_idx].item() == float(action_idx)
            child_node_count += 1
        else:
            # If action is illegal, no child should be created for it.
            # The children_node_indices for this action should remain INVALID_NODE_INDEX.
            assert child_node_assigned_idx == INVALID_NODE_INDEX
    
    assert child_node_count == num_legal_actions

    # Check parent's prior probabilities (should be masked and normalized)
    raw_priors = jax.nn.softmax(network_policy_logits)
    masked_priors = raw_priors * legal_actions_mask
    expected_priors = masked_priors / (jnp.sum(masked_priors) + 1e-9)
    chex.assert_trees_all_close(expanded_state.prior_probabilities[parent_idx], expected_priors, atol=1e-6)

def test_expand_node_no_space(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    key = jax.random.PRNGKey(3)
    state = initial_mcts_state_for_expansion
    parent_idx = ROOT_INDEX
    parent_hidden_state = state.hidden_state[parent_idx]
    network_policy_logits = state.policy_logits[parent_idx]
    network_value = state.value_from_network[parent_idx]
    network_reward = state.reward_from_parent_action[parent_idx]

    # Modify max_nodes in the state to simulate no space for new children
    # Root is at 0, num_allocated_nodes is 1.
    # If max_nodes is 1, no space for any children.
    state_no_space = dataclasses.replace(state, parent_indices=jnp.empty((1,), dtype=jnp.int32)) # Hacky way to set max_nodes for this test
    # More correctly, ensure num_allocated_nodes is already at max_nodes capacity of original state setup.
    # Let's make max_nodes = 1 for initial_state, so only root fits.
    # We can't change max_nodes of an existing MCTSState easily as arrays are fixed.
    # So, we test by setting num_allocated_nodes to max_nodes.
    max_nodes_original = state.parent_indices.shape[0]
    state_full = dataclasses.replace(state, num_allocated_nodes=jnp.array(max_nodes_original, dtype=jnp.int32))

    expanded_state = _expand_node_jax(
        mcts_state=state_full, # Use the state that's already full
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state,
        network_policy_logits=network_policy_logits,
        network_value=network_value,
        network_reward=network_reward,
        model=dummy_model,
        key=key,
        config=dummy_config,
        legal_actions_mask=None 
    )

    # Parent should still be marked as expanded (its own state updated)
    assert expanded_state.is_expanded[parent_idx].item() is True
    # Priors for parent should be set
    expected_priors = jax.nn.softmax(network_policy_logits)
    chex.assert_trees_all_close(expanded_state.prior_probabilities[parent_idx], expected_priors, atol=1e-6)

    # No new nodes should have been allocated
    assert expanded_state.num_allocated_nodes.item() == max_nodes_original 

    # children_node_indices for the parent should all remain INVALID_NODE_INDEX
    for action_idx in range(dummy_config.num_actions):
        assert expanded_state.children_node_indices[parent_idx, action_idx].item() == INVALID_NODE_INDEX 

def test_expand_node_idempotency(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    key = jax.random.PRNGKey(4) # New key for this test
    state = initial_mcts_state_for_expansion
    parent_idx = ROOT_INDEX

    parent_hidden_state = state.hidden_state[parent_idx]
    network_policy_logits = state.policy_logits[parent_idx]
    network_value = state.value_from_network[parent_idx]
    network_reward = state.reward_from_parent_action[parent_idx]

    # First expansion
    key1, key2 = jax.random.split(key)
    expanded_state_first_call = _expand_node_jax(
        mcts_state=state,
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state,
        network_policy_logits=network_policy_logits,
        network_value=network_value,
        network_reward=network_reward,
        model=dummy_model,
        key=key1,
        config=dummy_config,
        legal_actions_mask=None
    )

    num_allocated_after_first = expanded_state_first_call.num_allocated_nodes.item()
    assert expanded_state_first_call.is_expanded[parent_idx].item() is True
    assert num_allocated_after_first == 1 + dummy_config.num_actions # Root + children

    # Second expansion on the same node
    expanded_state_second_call = _expand_node_jax(
        mcts_state=expanded_state_first_call, # Use state from first call
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state, # Parent hs doesn't change for this test
        network_policy_logits=network_policy_logits,
        network_value=network_value,
        network_reward=network_reward,
        model=dummy_model,
        key=key2, # Use a different key to ensure no accidental collision if model used it differently
        config=dummy_config,
        legal_actions_mask=None
    )

    # Assert that the state is identical to after the first call
    # and no new nodes were allocated, children info remains same.
    chex.assert_trees_all_close(expanded_state_second_call, expanded_state_first_call)
    
    assert expanded_state_second_call.num_allocated_nodes.item() == num_allocated_after_first
    # Check a few child properties to be sure
    first_child_idx = expanded_state_first_call.children_node_indices[parent_idx, 0].item()
    assert expanded_state_second_call.children_node_indices[parent_idx, 0].item() == first_child_idx
    chex.assert_trees_all_close(expanded_state_second_call.hidden_state[first_child_idx],
                                 expanded_state_first_call.hidden_state[first_child_idx])


def test_expand_non_root_node(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    """Tests expansion of a non-root node, focusing on parent HS and reward updates."""
    key_setup, key_expand = jax.random.split(jax.random.PRNGKey(5))
    
    # 1. Initial setup: Expand root to get some children
    state_after_root_expansion = _expand_node_jax(
        mcts_state=initial_mcts_state_for_expansion,
        parent_node_idx=ROOT_INDEX,
        parent_hidden_state=initial_mcts_state_for_expansion.hidden_state[ROOT_INDEX],
        network_policy_logits=initial_mcts_state_for_expansion.policy_logits[ROOT_INDEX],
        network_value=initial_mcts_state_for_expansion.value_from_network[ROOT_INDEX],
        network_reward=initial_mcts_state_for_expansion.reward_from_parent_action[ROOT_INDEX],
        model=dummy_model,
        key=key_setup,
        config=dummy_config,
        legal_actions_mask=None
    )

    # 2. Select a child of the root to be the new parent_node_idx for expansion
    action_to_child = 1 # Let's pick the child resulting from action 1
    non_root_parent_idx = state_after_root_expansion.children_node_indices[ROOT_INDEX, action_to_child].item()
    assert non_root_parent_idx != INVALID_NODE_INDEX
    assert non_root_parent_idx != ROOT_INDEX

    # 3. Define hypothetical network outputs for this non_root_parent_idx
    # These would normally come from a recurrent_inference if we were simulating MCTS path selection.
    # For this test, we set them manually to distinct values.
    hypothetical_parent_hs = jnp.full_like(state_after_root_expansion.hidden_state[non_root_parent_idx], 0.77)
    # Note: The actual HS for non_root_parent_idx was set during root expansion by dummy_model.recurrent_inference.
    # We are now simulating that this node was *selected*, and we are about to *expand* it using 'hypothetical_parent_hs'.
    
    hypothetical_policy_logits = jnp.array([0.1, 0.8, 0.1], dtype=jnp.float32) * 2.0
    hypothetical_value = jnp.array(0.66, dtype=jnp.float32)
    # This is the reward for the action *leading to* non_root_parent_idx.
    # It was set when non_root_parent_idx was created as a child of root.
    # We need a *new* reward that _expand_node_jax is supposed to store for this node.
    hypothetical_reward_for_reaching_non_root_parent = jnp.array(0.88, dtype=jnp.float32) # Distinct fixed value
    # The original reward for this node (state_after_root_expansion.reward_from_parent_action[non_root_parent_idx])
    # was based on the action `action_to_child`. We are now providing a new value that
    # _expand_node_jax should store in reward_from_parent_action[non_root_parent_idx].

    # 4. Call _expand_node_jax for the non-root parent
    state_after_non_root_expansion = _expand_node_jax(
        mcts_state=state_after_root_expansion,
        parent_node_idx=non_root_parent_idx,
        parent_hidden_state=hypothetical_parent_hs, # This HS should be stored
        network_policy_logits=hypothetical_policy_logits,
        network_value=hypothetical_value,
        network_reward=hypothetical_reward_for_reaching_non_root_parent, # This reward should be stored
        model=dummy_model,
        key=key_expand,
        config=dummy_config,
        legal_actions_mask=None
    )

    # 5. Verifications
    # (a) Parent hidden_state update
    chex.assert_trees_all_close(
        state_after_non_root_expansion.hidden_state[non_root_parent_idx],
        hypothetical_parent_hs
    )

    # (b) Parent reward update
    assert jnp.isclose(
        state_after_non_root_expansion.reward_from_parent_action[non_root_parent_idx],
        hypothetical_reward_for_reaching_non_root_parent
    ).item(), "Parent reward_from_parent_action not updated correctly for non-root node."

    # Standard expansion checks for the non-root parent
    assert state_after_non_root_expansion.is_expanded[non_root_parent_idx].item() is True
    assert state_after_non_root_expansion.value_from_network[non_root_parent_idx].item() == hypothetical_value.item()
    expected_priors = jax.nn.softmax(hypothetical_policy_logits)
    chex.assert_trees_all_close(
        state_after_non_root_expansion.prior_probabilities[non_root_parent_idx],
        expected_priors, atol=1e-6
    )

    # Check children allocation for the non-root parent
    num_new_children = dummy_config.num_actions
    expected_total_nodes = state_after_root_expansion.num_allocated_nodes.item() + num_new_children
    assert state_after_non_root_expansion.num_allocated_nodes.item() == expected_total_nodes

    child_base_idx_non_root = state_after_root_expansion.num_allocated_nodes.item()
    for i in range(num_new_children):
        action = i
        child_idx = state_after_non_root_expansion.children_node_indices[non_root_parent_idx, action].item()
        assert child_idx == child_base_idx_non_root + i
        assert state_after_non_root_expansion.parent_indices[child_idx].item() == non_root_parent_idx
        assert state_after_non_root_expansion.action_from_parent[child_idx].item() == action


def test_expand_node_config_mask_false_with_mask_provided(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    key = jax.random.PRNGKey(6)
    state = initial_mcts_state_for_expansion
    parent_idx = ROOT_INDEX
    parent_hidden_state = state.hidden_state[parent_idx]
    network_policy_logits = state.policy_logits[parent_idx]
    network_value = state.value_from_network[parent_idx]
    network_reward = state.reward_from_parent_action[parent_idx]

    # Create a config where mask_illegal_actions is False
    config_masking_disabled = dataclasses.replace(dummy_config, mask_illegal_actions=False)

    # Provide a legal_actions_mask that would normally prune actions (e.g., only action 0 is legal)
    legal_actions_mask_to_ignore = jnp.array([True, False, False], dtype=jnp.bool_)

    expanded_state = _expand_node_jax(
        mcts_state=state,
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state,
        network_policy_logits=network_policy_logits,
        network_value=network_value,
        network_reward=network_reward,
        model=dummy_model,
        key=key,
        config=config_masking_disabled, # Use config with masking disabled
        legal_actions_mask=legal_actions_mask_to_ignore # Provide a mask that should be ignored
    )

    # Assert that all children are created because config.mask_illegal_actions is False
    expected_num_children = dummy_config.num_actions # All actions
    assert expanded_state.num_allocated_nodes.item() == 1 + expected_num_children

    for action_idx in range(dummy_config.num_actions):
        child_node_idx = expanded_state.children_node_indices[parent_idx, action_idx].item()
        assert child_node_idx != INVALID_NODE_INDEX # All children should be valid

    # Assert that parent's prior probabilities are NOT masked (simple softmax)
    expected_priors_unmasked = jax.nn.softmax(network_policy_logits)
    chex.assert_trees_all_close(
        expanded_state.prior_probabilities[parent_idx],
        expected_priors_unmasked, 
        atol=1e-6
    )


def test_prepare_initial_mcts_state_zero_dim_observation(mock_model_fixture, mock_config_fixture):
    """Specifically tests the else block for 0-dim observation in prepare_initial_mcts_state."""
    key = jax.random.PRNGKey(505)
    initial_observation_0d = jnp.array(42.0, dtype=jnp.float32) # 0-dimensional
    max_nodes = 5

    # Use the standard mock_model_fixture which returns batched output by default
    mcts_state = prepare_initial_mcts_state(
        key=key,
        model=mock_model_fixture, # This model expects batched input and returns batched output
        initial_observation=initial_observation_0d,
        config=mock_config_fixture,
        max_nodes=max_nodes,
        legal_actions_mask=None
    )
    # Basic assertion to ensure it runs through
    assert mcts_state.num_allocated_nodes.item() == 1
    # Check if hidden state matches expected from mock model (which internally handles batching)
    expected_root_hidden_state = jnp.full(mock_model_fixture.hidden_state_shape, 0.1, dtype=jnp.float32)
    chex.assert_trees_all_close(mcts_state.hidden_state[ROOT_INDEX], expected_root_hidden_state, atol=1e-6)

    assert mcts_state.is_expanded[ROOT_INDEX].item() == False

def test_expand_node_all_false_mask(initial_mcts_state_for_expansion: MCTSState, dummy_model: DummyModel, dummy_config: MCTSConfig):
    key = jax.random.PRNGKey(7)
    state = initial_mcts_state_for_expansion
    parent_idx = ROOT_INDEX
    parent_hidden_state = state.hidden_state[parent_idx]
    network_policy_logits = state.policy_logits[parent_idx]
    network_value = state.value_from_network[parent_idx]
    network_reward = state.reward_from_parent_action[parent_idx]

    # Config with masking enabled (default in dummy_config)
    config_masking_enabled = dummy_config 

    # Provide an all-False legal_actions_mask
    all_false_mask = jnp.array([False, False, False], dtype=jnp.bool_)
    assert dummy_config.num_actions == 3 # Ensure mask length matches

    expanded_state = _expand_node_jax(
        mcts_state=state,
        parent_node_idx=parent_idx,
        parent_hidden_state=parent_hidden_state,
        network_policy_logits=network_policy_logits,
        network_value=network_value,
        network_reward=network_reward,
        model=dummy_model,
        key=key,
        config=config_masking_enabled, 
        legal_actions_mask=all_false_mask
    )

    # Assert that NO children are created
    # num_allocated_nodes should remain 1 (only root)
    assert expanded_state.num_allocated_nodes.item() == 1 

    for action_idx in range(dummy_config.num_actions):
        child_node_idx = expanded_state.children_node_indices[parent_idx, action_idx].item()
        assert child_node_idx == INVALID_NODE_INDEX # All children should be invalid

    # Assert that parent's prior probabilities are all zero (or very close due to float precision)
    # because all actions were masked out before normalization. The current normalization logic
    # `x / sum_child_priors` would result in NaN if sum_child_priors is 0.
    # The updated code `jax.lax.cond(sum_child_priors > 1e-9, ..., lambda x: x)` should keep it as zeros.
    expected_priors_all_zero = jnp.zeros_like(network_policy_logits)
    chex.assert_trees_all_close(
        expanded_state.prior_probabilities[parent_idx],
        expected_priors_all_zero, 
        atol=1e-6
    )
    
    # Parent node itself is still marked expanded (its own properties are updated)
    assert expanded_state.is_expanded[parent_idx].item() is True

def test_prepare_initial_mcts_state_already_batched_observation(mock_model_fixture, mock_config_fixture):
    """Tests prepare_initial_mcts_state with an already batched observation to hit line 42."""
    key = jax.random.PRNGKey(8)
    # Observation that is already batched, e.g., (1, 10) for 1D features
    initial_observation_batched_1d = jnp.zeros((1, 10), dtype=jnp.float32)
    # Observation that is already batched, e.g., (1, 8, 8, 3) for 3D features (like an image)
    initial_observation_batched_3d = jnp.zeros((1, 8, 8, 3), dtype=jnp.float32)
    max_nodes = 5

    observations_to_test = [
        initial_observation_batched_1d,
        initial_observation_batched_3d
    ]

    for obs in observations_to_test:
        mcts_state = prepare_initial_mcts_state(
            key=key,
            model=mock_model_fixture,
            initial_observation=obs,
            config=mock_config_fixture,
            max_nodes=max_nodes,
            legal_actions_mask=None
        )
        # Basic assertion to ensure it runs through and processes the batched input correctly
        assert mcts_state.num_allocated_nodes.item() == 1
        # The model inside prepare_initial_mcts_state should receive the batched input as is.
        # The stored hidden state should correspond to the unbatched version from the model.
        expected_root_hidden_state = jnp.full(mock_model_fixture.hidden_state_shape, 0.1, dtype=jnp.float32)
        chex.assert_trees_all_close(mcts_state.hidden_state[ROOT_INDEX], expected_root_hidden_state, atol=1e-6)


def test_prepare_initial_mcts_state_zero_dim_observation(mock_model_fixture, mock_config_fixture):
    """Specifically tests the else block for 0-dim observation in prepare_initial_mcts_state."""
    key = jax.random.PRNGKey(505)
    initial_observation_0d = jnp.array(42.0, dtype=jnp.float32) # 0-dimensional
    max_nodes = 5

    # Use the standard mock_model_fixture which returns batched output by default
    mcts_state = prepare_initial_mcts_state(
        key=key,
        model=mock_model_fixture, # This model expects batched input and returns batched output
        initial_observation=initial_observation_0d,
        config=mock_config_fixture,
        max_nodes=max_nodes,
        legal_actions_mask=None
    )
    # Basic assertion to ensure it runs through
    assert mcts_state.num_allocated_nodes.item() == 1
    # Check if hidden state matches expected from mock model (which internally handles batching)
    expected_root_hidden_state = jnp.full(mock_model_fixture.hidden_state_shape, 0.1, dtype=jnp.float32)
    chex.assert_trees_all_close(mcts_state.hidden_state[ROOT_INDEX], expected_root_hidden_state, atol=1e-6)

    assert mcts_state.is_expanded[ROOT_INDEX].item() == False