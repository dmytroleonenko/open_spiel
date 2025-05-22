import pytest
import jax
import jax.numpy as jnp
import flax.experimental.nnx as nnx # For creating mock parameters for L2 reg
from open_spiel.python.algorithms.muzero_jax.training import losses

# --- Test scalar_mse_loss ---
def test_scalar_mse_loss_basic():
    pred = jnp.array([1., 2., 3.])
    target = jnp.array([1.5, 2.5, 2.5])
    # ((1-1.5)^2 + (2-2.5)^2 + (3-2.5)^2) / 3
    # = (0.25 + 0.25 + 0.25) / 3 = 0.75 / 3 = 0.25
    expected_loss = jnp.array(0.25)
    assert jnp.isclose(losses.scalar_mse_loss(pred, target), expected_loss)

def test_scalar_mse_loss_batch():
    pred = jnp.array([[1., 2.], [3., 4.]])
    target = jnp.array([[1.5, 2.5], [2.5, 3.5]])
    # ((0.25+0.25)/2 + (0.25+0.25)/2) / 2 = (0.25 + 0.25) / 2 = 0.25
    expected_loss = jnp.array(0.25)
    assert jnp.isclose(losses.scalar_mse_loss(pred, target), expected_loss)

# --- Test cross_entropy_loss_with_logits ---
def test_cross_entropy_loss_basic():
    # Batch size 1, 3 classes
    logits = jnp.array([[0., 1., 0.]]) # Softmax probs approx [0.21, 0.58, 0.21]
    targets = jnp.array([[0., 1., 0.]]) # True class is 1
    # For a target of [0,1,0], loss is -log(softmax(logits)[1])
    # softmax(logits)[1] = exp(1)/(exp(0)+exp(1)+exp(0)) = e / (2+e) approx 2.718 / (2+2.718) = 0.576
    # -log(0.576) approx 0.551
    expected_loss = -jnp.log(jnp.exp(1.) / (2 * jnp.exp(0.) + jnp.exp(1.)))
    assert jnp.isclose(losses.cross_entropy_loss_with_logits(logits, targets), expected_loss)

def test_cross_entropy_loss_batch():
    logits = jnp.array([[0., 1., 0.], [1., 0., 0.]])
    targets = jnp.array([[0., 1., 0.], [1., 0., 0.]])
    loss1 = -jnp.log(jnp.exp(1.) / (2 * jnp.exp(0.) + jnp.exp(1.)))
    loss2 = -jnp.log(jnp.exp(1.) / (2 * jnp.exp(0.) + jnp.exp(1.))) # Same due to symmetry
    expected_loss = (loss1 + loss2) / 2
    assert jnp.isclose(losses.cross_entropy_loss_with_logits(logits, targets), expected_loss)

# --- Test l2_regularization ---
def test_l2_regularization_basic():
    # Mock nnx.State or just a PyTree of params
    class MockModel(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(3, 2, rngs=rngs)
            self.linear2 = nnx.Linear(2, 1, rngs=rngs)
    
    key = jax.random.PRNGKey(0)
    model = MockModel(rngs=nnx.Rngs(params=key))
    # Manually set some param values for deterministic test
    model.linear1.kernel.value = jnp.array([[1.,2.,3.], [4.,5.,6.]]) # Shape (2,3)
    model.linear1.bias.value = jnp.array([0.1, 0.2])
    model.linear2.kernel.value = jnp.array([[10.], [20.]]) # Shape (2,1)
    model.linear2.bias.value = jnp.array([0.3])

    params = {'params': {'linear1': {'kernel': model.linear1.kernel.value, 'bias': model.linear1.bias.value }, 
                       'linear2': {'kernel': model.linear2.kernel.value, 'bias': model.linear2.bias.value }}}
    
    weight = 0.1
    # Sum of squares:
    # (1+4+9+16+25+36) + (0.01+0.04) + (100+400) + 0.09
    # = 91 + 0.05 + 500 + 0.09 = 591.14
    expected_sum_sq = 1**2+2**2+3**2+4**2+5**2+6**2 + 0.1**2+0.2**2 + 10**2+20**2 + 0.3**2
    expected_loss = weight * expected_sum_sq
    assert jnp.isclose(losses.l2_regularization(params, weight), expected_loss)

def test_l2_regularization_zero_weight():
    params = {'a': jnp.array([1.,2.])}
    assert losses.l2_regularization(params, 0.0) == 0.0
    assert losses.l2_regularization(params, -1.0) == 0.0 # Test negative weight too

# --- Test compute_policy_loss ---
def test_compute_policy_loss():
    logits = jnp.array([[0., 1., 0.]])
    targets = jnp.array([[0., 1., 0.]])
    expected_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    assert jnp.isclose(losses.compute_policy_loss(logits, targets), expected_loss)

# --- Test compute_scalar_value_loss ---
def test_compute_scalar_value_loss():
    pred = jnp.array([10., 20.])
    target = jnp.array([11., 19.])
    expected_loss = losses.scalar_mse_loss(pred, target)
    assert jnp.isclose(losses.compute_scalar_value_loss(pred, target), expected_loss)

def test_compute_scalar_value_loss_with_extra_dim():
    pred = jnp.array([[10.], [20.]])
    target = jnp.array([[11.], [19.]])
    expected_loss = losses.scalar_mse_loss(jnp.squeeze(pred), jnp.squeeze(target))
    assert jnp.isclose(losses.compute_scalar_value_loss(pred, target), expected_loss)

# --- Test compute_categorical_value_loss ---
def test_compute_categorical_value_loss():
    logits = jnp.array([[0., 0., 1.], [1., 0., 0.]]) # Batch 2, 3 classes
    targets = jnp.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    expected_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    assert jnp.isclose(losses.compute_categorical_value_loss(logits, targets), expected_loss)

# --- Test compute_scalar_reward_loss ---
def test_compute_scalar_reward_loss():
    pred = jnp.array([-1., 1.])
    target = jnp.array([-0.5, 0.5])
    expected_loss = losses.scalar_mse_loss(pred, target)
    assert jnp.isclose(losses.compute_scalar_reward_loss(pred, target), expected_loss)

def test_compute_scalar_reward_loss_with_extra_dim():
    pred = jnp.array([[-1.], [1.]])
    target = jnp.array([[-0.5], [0.5]])
    expected_loss = losses.scalar_mse_loss(jnp.squeeze(pred), jnp.squeeze(target))
    assert jnp.isclose(losses.compute_scalar_reward_loss(pred, target), expected_loss)

# --- Test compute_categorical_reward_loss ---
def test_compute_categorical_reward_loss():
    logits = jnp.array([[0.5, 0.5], [0.8, 0.2]]) # Batch 2, 2 classes (e.g. reward present/absent)
    targets = jnp.array([[0.4, 0.6], [0.7, 0.3]])
    expected_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    assert jnp.isclose(losses.compute_categorical_reward_loss(logits, targets), expected_loss)

# --- Test ValueError conditions ---
def test_scalar_mse_loss_shape_mismatch():
    with pytest.raises(ValueError, match="Prediction shape .* must match target shape"):
        losses.scalar_mse_loss(jnp.array([1,2]), jnp.array([1,2,3]))

def test_cross_entropy_shape_mismatch():
    with pytest.raises(ValueError, match="Logits shape .* must match targets shape"):
        losses.cross_entropy_loss_with_logits(jnp.array([[1,2]]), jnp.array([[1,2,3]]))

def test_cross_entropy_ndim_mismatch():
    with pytest.raises(ValueError, match="Logits and targets must be 2D"):
        losses.cross_entropy_loss_with_logits(jnp.array([1,2,3]), jnp.array([0,1,0])) 