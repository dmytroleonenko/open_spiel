import jax
import jax.numpy as jnp
import pytest
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    cross_entropy_loss, mse_loss, l2_regularization, Losses, Batch
)

# Test data
BATCH_SIZE = 2
SEQUENCE_LENGTH = 3
NUM_ACTIONS = 4

@pytest.fixture
def mock_logits() -> jnp.ndarray:
    return jnp.array([[[0.1, 0.2, 0.6, 0.1],
                       [0.3, 0.3, 0.2, 0.2],
                       [0.5, 0.1, 0.1, 0.3]],
                      [[0.8, 0.1, 0.05, 0.05],
                       [0.1, 0.7, 0.1, 0.1],
                       [0.2, 0.2, 0.3, 0.3]]], dtype=jnp.float32)

@pytest.fixture
def mock_policy_targets() -> jnp.ndarray:
    targets = jnp.zeros((BATCH_SIZE, SEQUENCE_LENGTH, NUM_ACTIONS), dtype=jnp.float32)
    targets = targets.at[0, 0, 2].set(1.0)
    targets = targets.at[0, 1, 0].set(1.0)
    targets = targets.at[0, 2, 0].set(1.0)
    targets = targets.at[1, 0, 0].set(1.0)
    targets = targets.at[1, 1, 1].set(1.0)
    targets = targets.at[1, 2, 3].set(1.0)
    return targets

@pytest.fixture
def mock_value_predictions() -> jnp.ndarray:
    return jnp.array([[0.5, -0.2, 1.2],
                       [0.8, 0.1, -0.5]], dtype=jnp.float32)

@pytest.fixture
def mock_value_targets() -> jnp.ndarray:
    return jnp.array([[0.6, -0.1, 1.0],
                       [0.7, 0.2, -0.6]], dtype=jnp.float32)

@pytest.fixture
def mock_reward_predictions() -> jnp.ndarray:
    # Assuming scalar rewards for simplicity in MSE
    return jnp.array([[0.9, 0.1, 1.1],
                       [0.7, 0.2, 0.4]], dtype=jnp.float32)

@pytest.fixture
def mock_reward_targets() -> jnp.ndarray:
    return jnp.array([[1.0, 0.0, 1.0],
                       [1.0, 0.0, 0.5]], dtype=jnp.float32)

@pytest.fixture
def mock_weights() -> jnp.ndarray:
    return jnp.array([[1.0, 0.5, 1.0],
                       [0.0, 1.0, 0.5]], dtype=jnp.float32)

@pytest.fixture
def mock_params() -> dict:
    # Simple mock parameters for L2 regularization test
    return {
        'conv1': {'kernel': jnp.ones((3, 3, 3, 16)), 'bias': jnp.zeros((16,))},
        'dense1': {'kernel': jnp.ones((32, 10)), 'bias': jnp.zeros((10,))},
        'embedding': jnp.ones((100, 8)) # Example of a param that might not be a weight matrix
    }

@pytest.fixture
def mock_empty_params() -> dict:
    return {}

# Tests for cross_entropy_loss
def test_cross_entropy_loss_unweighted(mock_logits, mock_policy_targets):
    loss = cross_entropy_loss(mock_logits, mock_policy_targets)
    # Expected value calculated manually for a small case or reference implementation
    # For this example, let's assume a placeholder value and focus on shape and type
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    # Add a specific value check if possible. Example:
    # expected_log_softmax_00 = jax.nn.log_softmax(mock_logits[0,0])
    # expected_loss_00 = -jnp.sum(mock_policy_targets[0,0] * expected_log_softmax_00)
    # ... and so on for all elements, then mean.
    # This can be tedious, consider a known small example or cross-check with tf/torch.

def test_cross_entropy_loss_weighted(mock_logits, mock_policy_targets, mock_weights):
    # Reshape weights to be (BATCH_SIZE, SEQUENCE_LENGTH) for policy loss
    # if policy targets are (B, T, A) and logits are (B, T, A), loss per item is (B,T)
    per_item_weights = mock_weights
    loss = cross_entropy_loss(mock_logits, mock_policy_targets, per_item_weights)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    # Add specific value check if possible

# Tests for mse_loss
def test_mse_loss_unweighted(mock_value_predictions, mock_value_targets):
    loss = mse_loss(mock_value_predictions, mock_value_targets)
    expected_loss = jnp.mean((mock_value_predictions - mock_value_targets)**2)
    assert jnp.allclose(loss, expected_loss)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

def test_mse_loss_weighted(mock_value_predictions, mock_value_targets, mock_weights):
    loss = mse_loss(mock_value_predictions, mock_value_targets, mock_weights)
    weighted_sq_error = ((mock_value_predictions - mock_value_targets)**2) * mock_weights
    expected_loss = jnp.sum(weighted_sq_error) / jnp.sum(mock_weights)
    assert jnp.allclose(loss, expected_loss)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

def test_mse_loss_distributional_predictions(mock_weights):
    # Test case where predictions might have an extra dimension (e.g. for distributional RL)
    # (B, T, 1) for predictions, (B, T) for targets
    dist_predictions = jnp.array([[[0.5], [-0.2], [1.2]],
                                   [[0.8], [0.1], [-0.5]]], dtype=jnp.float32)
    targets = jnp.array([[0.6, -0.1, 1.0],
                          [0.7, 0.2, -0.6]], dtype=jnp.float32)
    
    loss_unweighted = mse_loss(dist_predictions, targets)
    expected_loss_unweighted = jnp.mean((jnp.squeeze(dist_predictions, axis=-1) - targets)**2)
    assert jnp.allclose(loss_unweighted, expected_loss_unweighted)

    loss_weighted = mse_loss(dist_predictions, targets, mock_weights)
    weighted_sq_error = ((jnp.squeeze(dist_predictions, axis=-1) - targets)**2) * mock_weights
    expected_loss_weighted = jnp.sum(weighted_sq_error) / jnp.sum(mock_weights)
    assert jnp.allclose(loss_weighted, expected_loss_weighted)


# Tests for l2_regularization
def test_l2_regularization_non_zero_weight(mock_params):
    l2_reg_weight = 0.01
    loss = l2_regularization(mock_params, l2_reg_weight)
    
    expected_conv1_kernel_l2 = 0.5 * l2_reg_weight * jnp.sum(mock_params['conv1']['kernel']**2)
    expected_dense1_kernel_l2 = 0.5 * l2_reg_weight * jnp.sum(mock_params['dense1']['kernel']**2)
    # embedding is > 1D, so it's included by current heuristic
    expected_embedding_l2 = 0.5 * l2_reg_weight * jnp.sum(mock_params['embedding']**2)
    expected_total_l2 = expected_conv1_kernel_l2 + expected_dense1_kernel_l2 + expected_embedding_l2
    
    assert jnp.allclose(loss, expected_total_l2)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

def test_l2_regularization_zero_weight(mock_params):
    loss = l2_regularization(mock_params, 0.0)
    assert jnp.allclose(loss, jnp.array(0.0))
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

def test_l2_regularization_empty_params():
    loss = l2_regularization({}, 0.01)
    assert jnp.allclose(loss, jnp.array(0.0))
    loss_zero_weight = l2_regularization({}, 0.0)
    assert jnp.allclose(loss_zero_weight, jnp.array(0.0))


# Test Losses dataclass initialization
def test_losses_dataclass_initialization():
    losses = Losses()
    assert losses.policy_loss == 0.0
    assert losses.value_loss == 0.0
    assert losses.reward_loss == 0.0
    assert losses.l2_loss == 0.0
    assert losses.total_loss == 0.0

    losses_custom = Losses(policy_loss=jnp.array(1.0), total_loss=jnp.array(1.0))
    assert losses_custom.policy_loss == 1.0
    assert losses_custom.total_loss == 1.0 