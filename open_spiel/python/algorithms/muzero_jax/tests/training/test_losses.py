import pytest
import jax
import jax.numpy as jnp
import flax.experimental.nnx as nnx # For creating mock parameters for L2 reg
import optax
from open_spiel.python.algorithms.muzero_jax.training import losses

# --- Test scalar_mse_loss ---
def test_scalar_mse_loss_basic():
    pred = jnp.array([1., 2., 3.])
    target = jnp.array([1.5, 2.5, 2.5])
    # Per-item losses: (1-1.5)^2, (2-2.5)^2, (3-2.5)^2 = 0.25, 0.25, 0.25
    expected_loss = jnp.array([0.25, 0.25, 0.25])
    assert jnp.allclose(losses.scalar_mse_loss(pred, target), expected_loss)

def test_scalar_mse_loss_batch():
    pred = jnp.array([[1., 2.], [3., 4.]])
    target = jnp.array([[1.5, 2.5], [2.5, 3.5]])
    # Per-batch-item losses: [(1-1.5)^2+(2-2.5)^2], [(3-2.5)^2+(4-3.5)^2] = [0.5], [0.5]
    expected_loss = jnp.array([0.5, 0.5])
    assert jnp.allclose(losses.scalar_mse_loss(pred, target), expected_loss)

# --- Test cross_entropy_loss_with_logits ---
def test_cross_entropy_loss_basic():
    # Batch size 1, 3 classes
    logits = jnp.array([[0., 1., 0.]]) # Softmax probs approx [0.21, 0.58, 0.21]
    targets = jnp.array([[0., 1., 0.]]) # True class is 1
    # For a target of [0,1,0], loss is -log(softmax(logits)[1])
    # softmax(logits)[1] = exp(1)/(exp(0)+exp(1)+exp(0)) = e / (2+e) approx 2.718 / (2+2.718) = 0.576
    # -log(0.576) approx 0.551 (per-item loss for the single batch item)
    expected_loss = jnp.array([-jnp.log(jnp.exp(1.) / (2 * jnp.exp(0.) + jnp.exp(1.)))])
    assert jnp.allclose(losses.cross_entropy_loss_with_logits(logits, targets), expected_loss)

def test_cross_entropy_loss_batch():
    logits = jnp.array([[0., 1., 0.], [1., 0., 0.]])
    targets = jnp.array([[0., 1., 0.], [1., 0., 0.]])
    loss1 = -jnp.log(jnp.exp(1.) / (2 * jnp.exp(0.) + jnp.exp(1.)))
    loss2 = -jnp.log(jnp.exp(1.) / (2 * jnp.exp(0.) + jnp.exp(1.))) # Same due to symmetry
    expected_loss = jnp.array([loss1, loss2])  # Per-item losses
    assert jnp.allclose(losses.cross_entropy_loss_with_logits(logits, targets), expected_loss)

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
    assert jnp.allclose(losses.compute_policy_loss(logits, targets), expected_loss)

# --- Test compute_scalar_value_loss ---
def test_compute_scalar_value_loss():
    pred = jnp.array([10., 20.])
    target = jnp.array([11., 19.])
    # With default effective_iql_param=0.5, expect symmetric weighting
    base_loss = losses.scalar_mse_loss(pred, target)
    expected_loss = base_loss * 0.5  # IQL weighting with symmetric parameter
    assert jnp.allclose(losses.compute_scalar_value_loss(pred, target), expected_loss)

def test_compute_scalar_value_loss_with_extra_dim():
    pred = jnp.array([[10.], [20.]])
    target = jnp.array([[11.], [19.]])
    # With default effective_iql_param=0.5, expect symmetric weighting
    base_loss = losses.scalar_mse_loss(jnp.squeeze(pred), jnp.squeeze(target))
    expected_loss = base_loss * 0.5  # IQL weighting with symmetric parameter
    assert jnp.allclose(losses.compute_scalar_value_loss(pred, target), expected_loss)

# --- Test compute_categorical_value_loss ---
def test_compute_categorical_value_loss():
    logits = jnp.array([[0., 0., 1.], [1., 0., 0.]]) # Batch 2, 3 classes
    targets = jnp.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    # With default effective_iql_param=0.5, expect symmetric weighting
    base_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    expected_loss = base_loss * 0.5  # IQL weighting with symmetric parameter
    assert jnp.allclose(losses.compute_categorical_value_loss(logits, targets), expected_loss)

# --- Test compute_scalar_reward_loss ---
def test_compute_scalar_reward_loss():
    pred = jnp.array([-1., 1.])
    target = jnp.array([-0.5, 0.5])
    expected_loss = losses.scalar_mse_loss(pred, target)
    assert jnp.allclose(losses.compute_scalar_reward_loss(pred, target), expected_loss)

def test_compute_scalar_reward_loss_with_extra_dim():
    pred = jnp.array([[-1.], [1.]])
    target = jnp.array([[-0.5], [0.5]])
    expected_loss = losses.scalar_mse_loss(jnp.squeeze(pred), jnp.squeeze(target))
    assert jnp.allclose(losses.compute_scalar_reward_loss(pred, target), expected_loss)

# --- Test compute_categorical_reward_loss ---
def test_compute_categorical_reward_loss():
    """Test that categorical reward loss uses KL divergence."""
    logits = jnp.array([[0.5, 0.5], [0.8, 0.2]]) # Batch 2, 2 classes (e.g. reward present/absent)
    targets = jnp.array([[0.4, 0.6], [0.7, 0.3]])
    expected_loss = losses.compute_kl_loss(logits, targets)  # Now uses KL divergence
    assert jnp.allclose(losses.compute_categorical_reward_loss(logits, targets), expected_loss)

def test_categorical_reward_loss_vs_cross_entropy():
    """Verify that categorical reward loss differs from cross-entropy loss."""
    logits = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])
    targets = jnp.array([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]])
    
    kl_loss = losses.compute_categorical_reward_loss(logits, targets)
    cross_entropy_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    
    # KL loss and cross-entropy should generally differ
    assert not jnp.allclose(kl_loss, cross_entropy_loss)
    # Both should return per-batch losses
    assert kl_loss.shape == (2,)
    assert cross_entropy_loss.shape == (2,)

def test_categorical_reward_loss_equivalence_with_kl():
    """Test that categorical reward loss is equivalent to direct KL loss computation."""
    # Test with multiple cases
    test_cases = [
        # Uniform distributions
        (jnp.array([[0.0, 0.0, 0.0]]), jnp.array([[1/3, 1/3, 1/3]])),
        # Peaked distributions  
        (jnp.array([[5.0, 0.0, 0.0]]), jnp.array([[0.9, 0.05, 0.05]])),
        # Mixed batch
        (jnp.array([[1.0, 2.0], [0.5, 1.5]]), jnp.array([[0.3, 0.7], [0.6, 0.4]])),
    ]
    
    for logits, targets in test_cases:
        categorical_loss = losses.compute_categorical_reward_loss(logits, targets)
        direct_kl_loss = losses.compute_kl_loss(logits, targets)
        assert jnp.allclose(categorical_loss, direct_kl_loss), f"Failed for logits {logits}, targets {targets}"

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

# --- Test IQL weighting in categorical value loss ---
def test_compute_categorical_value_loss_with_iql():
    # Test IQL weighting with normalized support range
    logits = jnp.array([[0., 0., 1.], [1., 0., 0.]])  # Batch 2, 3 classes  
    targets = jnp.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    
    # Test with symmetric IQL (0.5)
    loss_symmetric = losses.compute_categorical_value_loss(logits, targets, 0.5)
    # Test with asymmetric IQL (1.0)
    loss_asymmetric = losses.compute_categorical_value_loss(logits, targets, 1.0)
    
    # IQL should modify the loss differently based on error sign
    assert not jnp.allclose(loss_symmetric, loss_asymmetric)
    assert loss_symmetric.shape == (2,)  # Per-batch losses

# --- Test symlog functions ---
def test_symlog():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = losses.symlog(x, base=2.0)
    
    # Symlog: sign(x) * log(|x| + 1) / log(base)
    expected = jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0) / jnp.log(2.0)
    assert jnp.allclose(result, expected)

def test_symexp():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = losses.symexp(x, base=2.0)
    
    # Symexp: sign(x) * (base^|x| - 1)
    expected = jnp.sign(x) * (jnp.power(2.0, jnp.abs(x)) - 1.0)
    assert jnp.allclose(result, expected)

def test_symlog_symexp_inverse():
    x = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    base = 2.0
    
    # Test that symexp(symlog(x)) ≈ x
    symlog_result = losses.symlog(x, base)
    reconstructed = losses.symexp(symlog_result, base)
    assert jnp.allclose(reconstructed, x, atol=1e-6)

# --- Test compute_symlog_loss ---
def test_compute_symlog_loss():
    pred = jnp.array([1.0, -2.0, 5.0])
    target = jnp.array([1.5, -1.5, 4.5])
    
    result = losses.compute_symlog_loss(pred, target, base=2.0)
    
    # Should transform both pred and target then compute MSE
    symlog_pred = losses.symlog(pred, 2.0)
    symlog_target = losses.symlog(target, 2.0)
    expected = losses.scalar_mse_loss(symlog_pred, symlog_target)
    
    assert jnp.allclose(result, expected)

# --- Test compute_kl_loss ---
def test_compute_kl_loss():
    logits = jnp.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]])
    target_probs = jnp.array([[0.2, 0.6, 0.2], [0.8, 0.1, 0.1]])
    
    result = losses.compute_kl_loss(logits, target_probs)
    
    # Manual KL computation for verification
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.log(jnp.clip(target_probs, 1e-8, 1.0))
    expected = jnp.sum(target_probs * (target_log_probs - log_probs), axis=-1)
    
    assert jnp.allclose(result, expected)
    assert result.shape == (2,)  # Per-batch losses

# --- Test discrete support transformations ---
def test_scalar_to_support():
    x = jnp.array([0.0, 1.0, -1.0, 10.0])
    num_atoms = 11
    
    result = losses.scalar_to_support(x, support_min=-5.0, support_max=5.0, num_atoms=num_atoms)
    
    assert result.shape == (4, 11)
    # Each distribution should sum to approximately 1
    assert jnp.allclose(jnp.sum(result, axis=-1), 1.0, atol=1e-5)

def test_support_to_scalar():
    # Create a peaked distribution around index 5 (middle of 11 atoms)
    logits = jnp.array([[-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
    
    result = losses.support_to_scalar(logits, support_min=-5.0, support_max=5.0, num_atoms=11)
    
    # Should be close to 0 (middle value)
    expected_value = 0.0
    assert jnp.allclose(result, expected_value, atol=1.0)  # Allow some tolerance for transformation
    assert result.shape == (1,)

def test_scalar_support_roundtrip():
    x = jnp.array([0.0, 2.0, -2.0])
    num_atoms = 21
    
    # Convert scalar to support and back
    support_dist = losses.scalar_to_support(x, support_min=-10.0, support_max=10.0, num_atoms=num_atoms)
    reconstructed = losses.support_to_scalar(support_dist, support_min=-10.0, support_max=10.0, num_atoms=num_atoms)
    
    # Should be reasonably close (transformations are lossy)
    assert jnp.allclose(reconstructed, x, atol=2.0)

def test_scalar_to_support_edge_cases():
    # Test with scalar input
    x_scalar = jnp.array(1.5)  # Convert to JAX array
    result_scalar = losses.scalar_to_support(x_scalar, num_atoms=5)
    assert result_scalar.shape == (5,)
    
    # Test with zero input
    x_zero = jnp.array(0.0)
    result_zero = losses.scalar_to_support(x_zero, num_atoms=3)
    assert result_zero.shape == (3,)
    
    # Test with negative input
    x_negative = jnp.array(-2.5)
    result_negative = losses.scalar_to_support(x_negative, num_atoms=7)
    assert result_negative.shape == (7,)

# --- Test compute_projection_consistency_loss ---
def test_compute_projection_consistency_loss():
    projection_current = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projection_initial = jnp.array([[0.8, 0.6, 0.0], [0.0, 0.8, 0.6]])
    
    result = losses.compute_projection_consistency_loss(projection_current, projection_initial)
    
    # Should compute symmetric cosine similarity loss
    # Note: No clipping needed since optax.cosine_similarity normalizes inputs
    sim1 = optax.cosine_similarity(projection_current, jax.lax.stop_gradient(projection_initial))
    sim2 = optax.cosine_similarity(jax.lax.stop_gradient(projection_current), projection_initial)
    expected = -sim1 + -sim2
    
    assert jnp.allclose(result, expected)
    assert result.shape == (2,)  # Per-batch losses

def test_optax_cosine_similarity_bounds():
    """Test that optax.cosine_similarity output is naturally bounded in [-1, 1]."""
    # Test with various input magnitudes and orientations
    vec1 = jnp.array([[1.0, 0.0], [100.0, 0.0], [-50.0, 25.0], [1e-6, 1e-6]])
    vec2 = jnp.array([[0.0, 1.0], [0.0, 100.0], [25.0, -50.0], [-1e-6, 1e-6]])
    
    similarities = optax.cosine_similarity(vec1, vec2)
    
    # Should be bounded in [-1, 1] without explicit clipping
    assert jnp.all(similarities >= -1.0), f"Found similarity < -1: {jnp.min(similarities)}"
    assert jnp.all(similarities <= 1.0), f"Found similarity > 1: {jnp.max(similarities)}"
    assert similarities.shape == (4,)
    
    # Test edge cases: identical and opposite vectors
    identical = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    opposite = jnp.array([[-1.0, -2.0], [-3.0, -4.0]])
    
    sim_identical = optax.cosine_similarity(identical, identical)
    sim_opposite = optax.cosine_similarity(identical, opposite)
    
    # Identical vectors should have cosine similarity of 1
    assert jnp.allclose(sim_identical, 1.0), f"Identical vectors sim: {sim_identical}"
    # Opposite vectors should have cosine similarity of -1
    assert jnp.allclose(sim_opposite, -1.0), f"Opposite vectors sim: {sim_opposite}"

# --- Test compute_policy_entropy ---
def test_compute_policy_entropy():
    # Test with uniform distribution (high entropy)
    uniform_logits = jnp.array([[0.0, 0.0, 0.0]])
    uniform_entropy = losses.compute_policy_entropy(uniform_logits)
    expected_max_entropy = jnp.log(3.0)  # log(num_actions)
    assert jnp.allclose(uniform_entropy, expected_max_entropy, atol=1e-5)
    
    # Test with peaked distribution (low entropy)
    peaked_logits = jnp.array([[10.0, 0.0, 0.0]])
    peaked_entropy = losses.compute_policy_entropy(peaked_logits)
    assert peaked_entropy[0] < uniform_entropy[0]
    
    # Test shape
    batch_logits = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    batch_entropy = losses.compute_policy_entropy(batch_logits)
    assert batch_entropy.shape == (2,)

# --- Test edge cases for squeeze operations in reward loss ---
def test_compute_scalar_reward_loss_squeeze_paths():
    # Test the squeeze branches that are currently missing coverage
    pred_2d = jnp.array([[1.0], [2.0]])  # Shape (2, 1) - will be squeezed
    target_2d = jnp.array([[1.1], [1.9]])  # Shape (2, 1) - will be squeezed
    
    result = losses.compute_scalar_reward_loss(pred_2d, target_2d)
    
    # Should squeeze and then compute MSE
    pred_squeezed = jnp.squeeze(pred_2d, axis=-1)
    target_squeezed = jnp.squeeze(target_2d, axis=-1) 
    expected = losses.scalar_mse_loss(pred_squeezed, target_squeezed)
    
    assert jnp.allclose(result, expected)

# --- Test additional edge cases for support transformations ---
def test_scalar_to_support_comprehensive_edge_cases():
    """Test all edge cases in scalar_to_support to improve coverage."""
    # Test 3D input to trigger the higher dimensional path
    x_3d = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])  # Shape (1, 2, 2)
    result_3d = losses.scalar_to_support(x_3d, support_min=-10.0, support_max=10.0, num_atoms=5)
    assert result_3d.shape == (1, 2, 2, 5)
    
    # Test with extreme values to trigger clamping
    x_extreme = jnp.array([-1000.0, 1000.0])
    result_extreme = losses.scalar_to_support(x_extreme, support_min=-10.0, support_max=10.0, num_atoms=21)
    assert result_extreme.shape == (2, 21)
    assert jnp.allclose(jnp.sum(result_extreme, axis=-1), 1.0, atol=1e-5)
    
    # Test with very small epsilon to trigger different paths
    x_small = jnp.array([0.1, -0.1])
    result_small = losses.scalar_to_support(x_small, epsilon=1e-8, num_atoms=11)
    assert result_small.shape == (2, 11)

def test_support_to_scalar_edge_cases():
    """Test edge cases in support_to_scalar for better coverage."""
    # Test with very peaked distribution (should handle numerical edge cases)
    num_atoms = 21
    logits_peaked = jnp.zeros((1, num_atoms))
    logits_peaked = logits_peaked.at[0, 10].set(100.0)  # Very peaked at center
    
    result_peaked = losses.support_to_scalar(logits_peaked, support_min=-10.0, support_max=10.0, num_atoms=num_atoms)
    assert result_peaked.shape == (1,)
    
    # Test with epsilon edge case - use consistent num_atoms parameter
    logits_small = jnp.ones((1, 11)) * 0.01  # Very small logits
    result_small = losses.support_to_scalar(logits_small, num_atoms=11, epsilon=1e-8)
    assert result_small.shape == (1,)
    
    # Test with uniform distribution
    num_atoms_uniform = 5
    logits_uniform = jnp.ones((2, num_atoms_uniform)) * 0.2  # Uniform
    result_uniform = losses.support_to_scalar(logits_uniform, num_atoms=num_atoms_uniform)
    assert result_uniform.shape == (2,)

def test_categorical_value_loss_iql_detailed():
    """More detailed test of IQL weighting to ensure all branches are covered."""
    # Create logits and targets that will have clear error signs  
    logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])  # Predict low vs high
    targets = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # Target high vs low
    
    # This should create positive error for first sample, negative for second
    loss_with_iql = losses.compute_categorical_value_loss(logits, targets, effective_iql_param=0.1)
    loss_without_iql = losses.compute_categorical_value_loss(logits, targets, effective_iql_param=1.0)
    
    # Should be different due to IQL weighting
    assert not jnp.allclose(loss_with_iql, loss_without_iql)
    assert loss_with_iql.shape == (2,)
    
    # Test with effective_iql_param = 1.0 (extreme asymmetric, but still applies weighting)
    loss_extreme_asym = losses.compute_categorical_value_loss(logits, targets, effective_iql_param=1.0)
    # The loss should still apply the weighting formula
    assert loss_extreme_asym.shape == (2,)

def test_compute_scalar_value_loss_iql_detailed():
    """Test IQL weighting in scalar value loss to ensure branch coverage."""
    # Create predictions and targets with clear error signs
    pred = jnp.array([10.0, 1.0])    # High prediction, low prediction  
    target = jnp.array([1.0, 10.0])  # Low target, high target
    # This creates: positive error (+9), negative error (-9)
    
    loss_with_iql = losses.compute_scalar_value_loss(pred, target, effective_iql_param=0.1)
    loss_without_iql = losses.compute_scalar_value_loss(pred, target, effective_iql_param=1.0)
    
    # Should be different due to IQL weighting
    assert not jnp.allclose(loss_with_iql, loss_without_iql)
    assert loss_with_iql.shape == (2,)
    
    # Test extreme asymmetric case (effective_iql_param=1.0) - still applies weighting formula
    loss_extreme_asym = losses.compute_scalar_value_loss(pred, target, effective_iql_param=1.0)
    # The loss should still apply the weighting formula
    assert loss_extreme_asym.shape == (2,)

def test_value_loss_squeeze_paths():
    """Test the squeeze paths in scalar value loss."""
    # Test with 2D inputs that need squeezing
    pred_2d = jnp.array([[10.0], [20.0]])  # Shape (2, 1)
    target_2d = jnp.array([[11.0], [19.0]]) # Shape (2, 1)
    
    result = losses.compute_scalar_value_loss(pred_2d, target_2d)
    
    # Should squeeze and compute MSE with IQL weighting
    pred_squeezed = jnp.squeeze(pred_2d, axis=-1)
    target_squeezed = jnp.squeeze(target_2d, axis=-1)
    base_loss = losses.scalar_mse_loss(pred_squeezed, target_squeezed)
    expected = base_loss * 0.5  # IQL weighting with default symmetric parameter
    
    assert jnp.allclose(result, expected)

def test_compute_projection_consistency_loss_numerical_equivalence():
    """Test that simplified SSL loss (without clipping) produces equivalent results to the clipped version."""
    # Test various edge cases to ensure clipping was truly redundant
    test_cases = [
        # Case 1: Normal vectors
        (jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 
         jnp.array([[0.8, 0.6, 0.0], [0.0, 0.8, 0.6]])),
        
        # Case 2: Very large magnitude vectors (should be normalized by optax.cosine_similarity)
        (jnp.array([[1000.0, 0.0], [0.0, 1000.0]]), 
         jnp.array([[1000.0, 0.0], [0.0, 1000.0]])),
        
        # Case 3: Very small magnitude vectors
        (jnp.array([[1e-6, 1e-6], [1e-8, 1e-8]]), 
         jnp.array([[1e-6, -1e-6], [-1e-8, 1e-8]])),
        
        # Case 4: Opposite vectors
        (jnp.array([[1.0, 2.0], [3.0, 4.0]]), 
         jnp.array([[-1.0, -2.0], [-3.0, -4.0]])),
        
        # Case 5: Orthogonal vectors 
        (jnp.array([[1.0, 0.0], [0.0, 1.0]]), 
         jnp.array([[0.0, 1.0], [1.0, 0.0]])),
    ]
    
    for i, (proj_current, proj_initial) in enumerate(test_cases):
        # Compute with our simplified version (current implementation)
        result_simplified = losses.compute_projection_consistency_loss(proj_current, proj_initial)
        
        # Manually compute what the old clipped version would have produced
        sim1 = optax.cosine_similarity(proj_current, jax.lax.stop_gradient(proj_initial))
        sim2 = optax.cosine_similarity(jax.lax.stop_gradient(proj_current), proj_initial)
        clipped_sim1 = jnp.clip(sim1, -1.0, 1.0)
        clipped_sim2 = jnp.clip(sim2, -1.0, 1.0)
        result_clipped = -clipped_sim1 + -clipped_sim2
        
        # Verify cosine similarities are naturally in bounds (no clipping needed)
        assert jnp.all(sim1 >= -1.0) and jnp.all(sim1 <= 1.0), f"Case {i}: sim1 out of bounds: {sim1}"
        assert jnp.all(sim2 >= -1.0) and jnp.all(sim2 <= 1.0), f"Case {i}: sim2 out of bounds: {sim2}"
        
        # Results should be identical (clipping was redundant)
        assert jnp.allclose(result_simplified, result_clipped, atol=1e-7), \
            f"Case {i}: simplified={result_simplified}, clipped={result_clipped}"
        
        # Verify that sim1 and clipped_sim1 are identical (and same for sim2)
        assert jnp.allclose(sim1, clipped_sim1, atol=1e-7), f"Case {i}: clipping changed sim1"
        assert jnp.allclose(sim2, clipped_sim2, atol=1e-7), f"Case {i}: clipping changed sim2"

# --- Test IQL effective parameter logic (Action Item 13) ---
def test_scalar_value_loss_effective_iql_symmetric():
    """Test that effective_iql_param=0.5 produces symmetric loss."""
    pred = jnp.array([10.0, 1.0])    # High prediction, low prediction  
    target = jnp.array([1.0, 10.0])  # Low target, high target
    # This creates: positive error (+9), negative error (-9)
    
    # With symmetric parameter (0.5), both errors should get same weight
    loss_symmetric = losses.compute_scalar_value_loss(pred, target, effective_iql_param=0.5)
    
    # Manually compute expected symmetric loss
    base_loss = losses.scalar_mse_loss(pred, target)  # [81.0, 81.0]
    error = pred - target  # [9.0, -9.0] 
    value_sign = (error > 0).astype(jnp.float32)  # [1.0, 0.0]
    # weights = (1.0 - value_sign) * 0.5 + value_sign * (1.0 - 0.5)
    # For positive error: weight = 0.0 * 0.5 + 1.0 * 0.5 = 0.5
    # For negative error: weight = 1.0 * 0.5 + 0.0 * 0.5 = 0.5  
    expected_weights = jnp.array([0.5, 0.5])
    expected_loss = base_loss * expected_weights
    
    assert jnp.allclose(loss_symmetric, expected_loss), f"Expected {expected_loss}, got {loss_symmetric}"
    assert jnp.allclose(loss_symmetric, jnp.array([40.5, 40.5])), "Symmetric loss should be equal for both samples"


def test_scalar_value_loss_effective_iql_asymmetric():
    """Test that effective_iql_param != 0.5 produces asymmetric loss."""
    pred = jnp.array([10.0, 1.0])    # High prediction, low prediction  
    target = jnp.array([1.0, 10.0])  # Low target, high target
    # This creates: positive error (+9), negative error (-9)
    
    # With asymmetric parameter (0.8), positive and negative errors get different weights
    loss_asymmetric = losses.compute_scalar_value_loss(pred, target, effective_iql_param=0.8)
    
    # Manually compute expected asymmetric loss
    base_loss = losses.scalar_mse_loss(pred, target)  # [81.0, 81.0]
    error = pred - target  # [9.0, -9.0] 
    value_sign = (error > 0).astype(jnp.float32)  # [1.0, 0.0]
    # weights = (1.0 - value_sign) * 0.8 + value_sign * (1.0 - 0.8) 
    # For positive error: weight = 0.0 * 0.8 + 1.0 * 0.2 = 0.2
    # For negative error: weight = 1.0 * 0.8 + 0.0 * 0.2 = 0.8
    expected_weights = jnp.array([0.2, 0.8])
    expected_loss = base_loss * expected_weights
    
    assert jnp.allclose(loss_asymmetric, expected_loss), f"Expected {expected_loss}, got {loss_asymmetric}"
    assert jnp.allclose(loss_asymmetric, jnp.array([16.2, 64.8])), "Asymmetric loss should differ for positive vs negative errors"


def test_categorical_value_loss_effective_iql_symmetric():
    """Test that effective_iql_param=0.5 produces symmetric loss for categorical values."""
    # Create logits and targets that will have clear error signs  
    logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])  # Predict low vs high
    targets = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # Target high vs low
    
    # With symmetric parameter (0.5), both types of errors should get same weight
    loss_symmetric = losses.compute_categorical_value_loss(logits, targets, effective_iql_param=0.5)
    
    # Manually compute expected symmetric loss
    base_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    
    # Compute expected values to determine error sign
    num_atoms = logits.shape[-1]
    support = jnp.linspace(-1.0, 1.0, num_atoms)  # [-1.0, 0.0, 1.0]
    
    pred_probs = jax.nn.softmax(logits)
    pred_value = jnp.sum(pred_probs * support, axis=-1)
    target_value = jnp.sum(targets * support, axis=-1)
    
    error = pred_value - target_value
    value_sign = (error > 0).astype(jnp.float32)
    # With 0.5: both positive and negative errors get weight 0.5
    expected_weights = jnp.array([0.5, 0.5])
    expected_loss = base_loss * expected_weights
    
    assert jnp.allclose(loss_symmetric, expected_loss), f"Expected {expected_loss}, got {loss_symmetric}"


def test_categorical_value_loss_effective_iql_asymmetric():
    """Test that effective_iql_param != 0.5 produces asymmetric loss for categorical values.""" 
    # Create logits and targets that will have clear error signs  
    logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])  # Predict low vs high
    targets = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # Target high vs low
    
    # With asymmetric parameter (0.2), positive and negative errors get different weights
    loss_asymmetric = losses.compute_categorical_value_loss(logits, targets, effective_iql_param=0.2)
    
    # Manually compute expected asymmetric loss
    base_loss = losses.cross_entropy_loss_with_logits(logits, targets)
    
    # Compute expected values to determine error sign
    num_atoms = logits.shape[-1]
    support = jnp.linspace(-1.0, 1.0, num_atoms)  # [-1.0, 0.0, 1.0]
    
    pred_probs = jax.nn.softmax(logits)
    pred_value = jnp.sum(pred_probs * support, axis=-1)
    target_value = jnp.sum(targets * support, axis=-1)
    
    error = pred_value - target_value
    value_sign = (error > 0).astype(jnp.float32) 
    # weights = (1.0 - value_sign) * 0.2 + value_sign * (1.0 - 0.2)
    # For positive error: weight = 0.0 * 0.2 + 1.0 * 0.8 = 0.8
    # For negative error: weight = 1.0 * 0.2 + 0.0 * 0.8 = 0.2
    expected_weights = (1.0 - value_sign) * 0.2 + value_sign * 0.8
    expected_loss = base_loss * expected_weights
    
    assert jnp.allclose(loss_asymmetric, expected_loss), f"Expected {expected_loss}, got {loss_asymmetric}"


def test_iql_effective_param_boundary_cases():
    """Test boundary cases for effective IQL parameter."""
    pred = jnp.array([5.0, 1.0])
    target = jnp.array([1.0, 5.0])  # positive error, negative error
    
    # Test effective_iql_param = 0.0 (extreme asymmetric)
    loss_zero = losses.compute_scalar_value_loss(pred, target, effective_iql_param=0.0)
    base_loss = losses.scalar_mse_loss(pred, target)
    # error = [4.0, -4.0], value_sign = [1.0, 0.0]
    # weights = (1.0 - value_sign) * 0.0 + value_sign * 1.0 = [1.0, 0.0]
    expected_zero = base_loss * jnp.array([1.0, 0.0])
    assert jnp.allclose(loss_zero, expected_zero)
    
    # Test effective_iql_param = 1.0 (extreme asymmetric - opposite direction)
    loss_one = losses.compute_scalar_value_loss(pred, target, effective_iql_param=1.0)
    # weights = (1.0 - value_sign) * 1.0 + value_sign * 0.0 = [0.0, 1.0]
    expected_one = base_loss * jnp.array([0.0, 1.0])
    assert jnp.allclose(loss_one, expected_one)


def test_iql_effective_param_always_applied():
    """Test that IQL weighting is always applied, regardless of the parameter value.""" 
    pred = jnp.array([3.0, 2.0])
    target = jnp.array([2.0, 3.0])  # positive error, negative error
    
    # Even with the "default" parameter 0.5, the weighting formula should be applied
    loss_default = losses.compute_scalar_value_loss(pred, target, effective_iql_param=0.5)
    
    # Manually compute with explicit weighting formula
    base_loss = losses.scalar_mse_loss(pred, target)
    error = pred - target
    value_sign = (error > 0).astype(jnp.float32)
    weights = (1.0 - value_sign) * 0.5 + value_sign * 0.5  # Should be [0.5, 0.5]
    expected_loss = base_loss * weights
    
    assert jnp.allclose(loss_default, expected_loss)
    
    # Test edge case: zero error (value_sign computation)
    pred_zero_error = jnp.array([2.0, 2.0])
    target_zero_error = jnp.array([2.0, 2.0])
    loss_zero_error = losses.compute_scalar_value_loss(pred_zero_error, target_zero_error, effective_iql_param=0.7)
    
    # Zero error should result in zero loss regardless of weights
    assert jnp.allclose(loss_zero_error, jnp.zeros(2))


def test_scalar_value_loss_parameter_name_change():
    """Test that the old iql_weight parameter was replaced with effective_iql_param."""
    # This test ensures the API change is complete
    pred = jnp.array([1.0, 2.0])
    target = jnp.array([1.5, 1.5])
    
    # The new parameter should work
    loss_new_param = losses.compute_scalar_value_loss(pred, target, effective_iql_param=0.3)
    assert loss_new_param.shape == (2,)
    
    # The function should have the new parameter name in its signature
    import inspect
    sig = inspect.signature(losses.compute_scalar_value_loss)
    param_names = list(sig.parameters.keys())
    assert 'effective_iql_param' in param_names, f"Expected 'effective_iql_param' in {param_names}"
    assert 'iql_weight' not in param_names, f"Old 'iql_weight' parameter should be removed from {param_names}"


def test_categorical_value_loss_parameter_name_change():
    """Test that the old iql_weight parameter was replaced with effective_iql_param in categorical loss."""
    logits = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    targets = jnp.array([[0.6, 0.4], [0.3, 0.7]])
    
    # The new parameter should work
    loss_new_param = losses.compute_categorical_value_loss(logits, targets, effective_iql_param=0.3)
    assert loss_new_param.shape == (2,)
    
    # The function should have the new parameter name in its signature
    import inspect
    sig = inspect.signature(losses.compute_categorical_value_loss)
    param_names = list(sig.parameters.keys())
    assert 'effective_iql_param' in param_names, f"Expected 'effective_iql_param' in {param_names}"
    assert 'iql_weight' not in param_names, f"Old 'iql_weight' parameter should be removed from {param_names}" 