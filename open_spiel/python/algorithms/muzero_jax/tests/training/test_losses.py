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
    
    # Test with explicit base=2.0
    result = losses.symlog(x, base=2.0)
    expected = jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0) / jnp.log(2.0)
    assert jnp.allclose(result, expected)
    
    # Test with default base (e) - EfficientZeroV2 pattern
    result_default = losses.symlog(x)
    expected_default = jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0) / jnp.log(jnp.e)
    assert jnp.allclose(result_default, expected_default)
    
    # Test that default is indeed base e
    result_e = losses.symlog(x, base=jnp.e) 
    assert jnp.allclose(result_default, result_e)

def test_symexp():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test with explicit base=2.0
    result = losses.symexp(x, base=2.0)
    expected = jnp.sign(x) * (jnp.power(2.0, jnp.abs(x)) - 1.0)
    assert jnp.allclose(result, expected)
    
    # Test with default base (e) - EfficientZeroV2 pattern
    result_default = losses.symexp(x)
    expected_default = jnp.sign(x) * (jnp.power(jnp.e, jnp.abs(x)) - 1.0)
    assert jnp.allclose(result_default, expected_default)
    
    # Test that default is indeed base e
    result_e = losses.symexp(x, base=jnp.e) 
    assert jnp.allclose(result_default, result_e)

def test_symlog_symexp_inverse():
    x = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    
    # Test with explicit base=2.0
    base = 2.0
    symlog_result = losses.symlog(x, base)
    reconstructed = losses.symexp(symlog_result, base)
    assert jnp.allclose(reconstructed, x, atol=1e-6)
    
    # Test with default base (e) - EfficientZeroV2 pattern 
    symlog_result_default = losses.symlog(x)
    reconstructed_default = losses.symexp(symlog_result_default)
    assert jnp.allclose(reconstructed_default, x, atol=1e-6)
    
    # Test that both default and explicit e give same results
    symlog_result_e = losses.symlog(x, base=jnp.e)
    reconstructed_e = losses.symexp(symlog_result_e, base=jnp.e)
    assert jnp.allclose(reconstructed_default, reconstructed_e)

# --- Test compute_symlog_loss ---
def test_compute_symlog_loss():
    # EfficientZeroV2 pattern: predictions are already in symlog space, only targets are transformed
    pred_symlog = jnp.array([1.0, -2.0, 5.0])  # Already in symlog space
    target_raw = jnp.array([1.5, -1.5, 4.5])   # Raw scalar targets
    
    result = losses.compute_symlog_loss(pred_symlog, target_raw, base=2.0)
    
    # EfficientZeroV2 pattern: prediction is already symlog, only transform target
    # loss = 0.5 * (prediction - symlog(target)) ** 2
    symlog_target = losses.symlog(target_raw, 2.0)
    expected = 0.5 * losses.scalar_mse_loss(pred_symlog, symlog_target)
    
    assert jnp.allclose(result, expected)
    
    # Test with default base (e)
    result_default = losses.compute_symlog_loss(pred_symlog, target_raw)
    symlog_target_e = losses.symlog(target_raw, jnp.e)
    expected_default = 0.5 * losses.scalar_mse_loss(pred_symlog, symlog_target_e)
    assert jnp.allclose(result_default, expected_default)

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

def test_comprehensive_missing_coverage():
    """Test functions and edge cases that are missing coverage."""
    import jax.random as jr
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    
    key = jr.key(42)
    
    # Test error conditions for cross_entropy_loss_with_logits
    logits_valid = jr.normal(key, (3, 5))
    targets_valid = jr.uniform(key, (3, 5))
    targets_valid = targets_valid / jnp.sum(targets_valid, axis=-1, keepdims=True)
    
    # Test valid case
    loss = losses_lib.cross_entropy_loss_with_logits(logits_valid, targets_valid)
    assert loss.shape == (3,)
    
    # Test with 1D inputs (should be per-item)
    scalar_pred = jnp.array([2.0])
    scalar_target = jnp.array([1.5])
    scalar_loss = losses_lib.scalar_mse_loss(scalar_pred, scalar_target)
    assert scalar_loss.shape == (1,)
    
    # Test l2_regularization with zero weight (should return 0.0)
    dummy_params = {'w': jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    l2_zero = losses_lib.l2_regularization(dummy_params, 0.0)
    assert l2_zero == 0.0
    
    # Test compute_scalar_value_loss with squeezing
    value_pred_unsqueezed = jr.normal(key, (3, 1))  # Shape (B, 1)
    target_val_unsqueezed = jr.normal(key, (3, 1))  # Shape (B, 1)
    value_loss = losses_lib.compute_scalar_value_loss(
        value_pred_unsqueezed, target_val_unsqueezed, effective_iql_param=0.7
    )
    assert value_loss.shape == (3,)
    
    # Test compute_scalar_reward_loss with squeezing
    reward_pred_unsqueezed = jr.normal(key, (3, 1))  # Shape (B, 1)
    target_rew_unsqueezed = jr.normal(key, (3, 1))  # Shape (B, 1)
    reward_loss = losses_lib.compute_scalar_reward_loss(
        reward_pred_unsqueezed, target_rew_unsqueezed
    )
    assert reward_loss.shape == (3,)


def test_symlog_symexp_functions():
    """Test symlog and symexp functions with various inputs and bases."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    
    # Test various inputs
    x_values = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    
    # Test with default base (e)
    symlog_default = losses_lib.symlog(x_values)
    symexp_default = losses_lib.symexp(symlog_default)
    
    # Test that symexp is inverse of symlog (approximately)
    assert jnp.allclose(x_values, symexp_default, atol=1e-5)
    
    # Test with base 2
    symlog_base2 = losses_lib.symlog(x_values, base=2.0)
    symexp_base2 = losses_lib.symexp(symlog_base2, base=2.0)
    assert jnp.allclose(x_values, symexp_base2, atol=1e-5)
    
    # Test that symlog preserves sign
    assert jnp.all(jnp.sign(symlog_default) == jnp.sign(x_values))
    
    # Test zero case specifically
    zero_input = jnp.array([0.0])
    assert jnp.abs(losses_lib.symlog(zero_input)[0]) < 1e-10
    assert jnp.abs(losses_lib.symexp(losses_lib.symlog(zero_input))[0]) < 1e-10


def test_scalar_to_support_function():
    """Test scalar_to_support function with various configurations."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    
    # Test with scalar input
    scalar_val = jnp.array(5.0)
    support_dist = losses_lib.scalar_to_support(
        scalar_val, support_min=-10.0, support_max=10.0, num_atoms=21
    )
    assert support_dist.shape == (21,)
    assert jnp.abs(jnp.sum(support_dist) - 1.0) < 1e-5  # Should sum to 1
    
    # Test with 1D batch input
    batch_vals = jnp.array([1.0, -2.0, 3.0])
    batch_support = losses_lib.scalar_to_support(
        batch_vals, support_min=-5.0, support_max=5.0, num_atoms=11
    )
    assert batch_support.shape == (3, 11)
    # Each row should sum to approximately 1
    row_sums = jnp.sum(batch_support, axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5)
    
    # Test with 2D input (should flatten and reshape)
    vals_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    support_2d = losses_lib.scalar_to_support(
        vals_2d, support_min=0.0, support_max=5.0, num_atoms=6
    )
    assert support_2d.shape == (2, 2, 6)


def test_support_to_scalar_function():
    """Test support_to_scalar function with various configurations."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test round-trip: scalar -> support -> scalar
    original_vals = jnp.array([1.5, -2.3, 0.0, 4.7])
    support_min, support_max, num_atoms = -5.0, 5.0, 11
    
    # Convert to support
    support_dist = losses_lib.scalar_to_support(
        original_vals, support_min=support_min, support_max=support_max, num_atoms=num_atoms
    )
    
    # Convert back to scalar using logits (add small noise to simulate real logits)
    logits = jnp.log(support_dist + 1e-8) + jr.normal(key, support_dist.shape) * 0.01
    recovered_vals = losses_lib.support_to_scalar(
        logits, support_min=support_min, support_max=support_max, num_atoms=num_atoms
    )
    
    # Should be approximately the same (some precision loss expected)
    assert jnp.allclose(original_vals, recovered_vals, atol=0.5)
    
    # Test with random logits
    random_logits = jr.normal(key, (3, 21))
    scalar_output = losses_lib.support_to_scalar(
        random_logits, support_min=-10.0, support_max=10.0, num_atoms=21
    )
    assert scalar_output.shape == (3,)
    assert jnp.all(jnp.isfinite(scalar_output))  # Should not have NaN/inf


def test_compute_policy_entropy():
    """Test policy entropy computation."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test uniform distribution (should have maximum entropy)
    uniform_logits = jnp.zeros((2, 4))  # Uniform over 4 actions
    uniform_entropy = losses_lib.compute_policy_entropy(uniform_logits)
    expected_uniform_entropy = jnp.log(4.0)  # log(num_actions) for uniform
    assert jnp.allclose(uniform_entropy, expected_uniform_entropy, rtol=1e-5)
    
    # Test deterministic distribution (should have zero entropy)
    deterministic_logits = jnp.array([[10.0, -10.0, -10.0, -10.0],
                                      [-10.0, 10.0, -10.0, -10.0]])
    det_entropy = losses_lib.compute_policy_entropy(deterministic_logits)
    assert jnp.allclose(det_entropy, 0.0, atol=1e-5)
    
    # Test with random logits
    random_logits = jr.normal(key, (3, 5))
    random_entropy = losses_lib.compute_policy_entropy(random_logits)
    assert random_entropy.shape == (3,)
    assert jnp.all(random_entropy >= 0.0)  # Entropy should be non-negative
    assert jnp.all(random_entropy <= jnp.log(5.0))  # Should not exceed max entropy


def test_compute_kl_loss():
    """Test KL divergence loss computation."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test with identical distributions (KL should be 0)
    logits = jr.normal(key, (2, 5))
    probs = jax.nn.softmax(logits)
    kl_identical = losses_lib.compute_kl_loss(logits, probs)
    assert jnp.allclose(kl_identical, 0.0, atol=1e-5)
    
    # Test with different distributions
    logits1 = jr.normal(key, (3, 4))
    logits2 = jr.normal(jr.split(key)[0], (3, 4))
    probs2 = jax.nn.softmax(logits2)
    kl_diff = losses_lib.compute_kl_loss(logits1, probs2)
    assert kl_diff.shape == (3,)
    assert jnp.all(kl_diff >= 0.0)  # KL divergence should be non-negative
    
    # Test numerical stability with very small probabilities
    small_probs = jnp.array([[1e-10, 1.0 - 1e-10], [0.5, 0.5]])
    test_logits = jr.normal(key, (2, 2))
    kl_stable = losses_lib.compute_kl_loss(test_logits, small_probs)
    assert jnp.all(jnp.isfinite(kl_stable))


def test_compute_symlog_loss():
    """Test symlog loss computation."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test with various predictions and targets
    predictions = jr.normal(key, (3,))  # Assumed to be in symlog space
    targets = jr.normal(jr.split(key)[0], (3,))  # Raw scalar values
    
    symlog_loss = losses_lib.compute_symlog_loss(predictions, targets)
    assert symlog_loss.shape == (3,)
    assert jnp.all(symlog_loss >= 0.0)
    
    # Test with different base
    symlog_loss_base2 = losses_lib.compute_symlog_loss(predictions, targets, base=2.0)
    assert symlog_loss_base2.shape == (3,)
    assert jnp.all(symlog_loss_base2 >= 0.0)
    
    # Test that loss is 0 when prediction equals symlog(target)
    target_val = jnp.array([2.0])
    symlog_target = losses_lib.symlog(target_val)
    zero_loss = losses_lib.compute_symlog_loss(symlog_target, target_val)
    assert jnp.allclose(zero_loss, 0.0, atol=1e-6)


def test_compute_projection_consistency_loss():
    """Test SSL projection consistency loss."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test with identical projections (loss should be close to minimum)
    projection_size = 64
    projection1 = jr.normal(key, (2, projection_size))
    projection1_normalized = projection1 / jnp.linalg.norm(projection1, axis=-1, keepdims=True)
    
    # Identical projections should have maximum cosine similarity (-1 * 2 = -2)
    loss_identical = losses_lib.compute_projection_consistency_loss(
        projection1_normalized, projection1_normalized
    )
    expected_loss = -2.0  # -cosine_sim - cosine_sim = -1 - 1 = -2
    assert jnp.allclose(loss_identical, expected_loss, atol=1e-5)
    
    # Test with orthogonal projections
    projection2 = jr.normal(jr.split(key)[0], (2, projection_size))
    projection2_normalized = projection2 / jnp.linalg.norm(projection2, axis=-1, keepdims=True)
    
    loss_different = losses_lib.compute_projection_consistency_loss(
        projection1_normalized, projection2_normalized
    )
    assert loss_different.shape == (2,)
    # Loss should be higher for different projections
    assert jnp.all(loss_different > loss_identical)


def test_compute_categorical_value_loss_with_iql():
    """Test categorical value loss with IQL weighting."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Create value logits and target distributions
    batch_size = 3
    num_atoms = 11
    value_logits = jr.normal(key, (batch_size, num_atoms))
    target_dist = jr.uniform(jr.split(key)[0], (batch_size, num_atoms))
    target_dist = target_dist / jnp.sum(target_dist, axis=-1, keepdims=True)
    
    # Test with different IQL parameters
    loss_symmetric = losses_lib.compute_categorical_value_loss(
        value_logits, target_dist, effective_iql_param=0.5
    )
    loss_asymmetric = losses_lib.compute_categorical_value_loss(
        value_logits, target_dist, effective_iql_param=0.8
    )
    
    assert loss_symmetric.shape == (batch_size,)
    assert loss_asymmetric.shape == (batch_size,)
    assert jnp.all(loss_symmetric >= 0.0)
    assert jnp.all(loss_asymmetric >= 0.0)


def test_compute_categorical_reward_loss():
    """Test categorical reward loss using KL divergence."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Create reward logits and target distributions
    batch_size = 3
    num_atoms = 11
    reward_logits = jr.normal(key, (batch_size, num_atoms))
    target_dist = jr.uniform(jr.split(key)[0], (batch_size, num_atoms))
    target_dist = target_dist / jnp.sum(target_dist, axis=-1, keepdims=True)
    
    # Test categorical reward loss (should use KL divergence)
    loss = losses_lib.compute_categorical_reward_loss(reward_logits, target_dist)
    assert loss.shape == (batch_size,)
    assert jnp.all(loss >= 0.0)
    
    # Compare with direct KL loss computation
    direct_kl_loss = losses_lib.compute_kl_loss(reward_logits, target_dist)
    assert jnp.allclose(loss, direct_kl_loss, rtol=1e-5)


# Test edge cases and error conditions that might have been missed
def test_edge_cases_and_error_conditions():
    """Test various edge cases and error conditions."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test MSE loss with multi-dimensional features
    pred_multi = jr.normal(key, (2, 3, 4))  # Batch of 2, features (3, 4)
    target_multi = jr.normal(jr.split(key)[0], (2, 3, 4))
    loss_multi = losses_lib.scalar_mse_loss(pred_multi, target_multi)
    assert loss_multi.shape == (2,)  # Should sum across feature dimensions
    
    # Test with very small values for numerical stability
    small_vals = jnp.array([1e-10, -1e-10, 0.0])
    small_symlog = losses_lib.symlog(small_vals)
    small_symexp = losses_lib.symexp(small_symlog)
    assert jnp.allclose(small_vals, small_symexp, atol=1e-8)
    
    # Test support conversion with edge values
    edge_vals = jnp.array([-300.0, 300.0, 0.0])  # At the boundaries
    edge_support = losses_lib.scalar_to_support(
        edge_vals, support_min=-300.0, support_max=300.0, num_atoms=601
    )
    assert edge_support.shape == (3, 601)
    assert jnp.all(jnp.sum(edge_support, axis=-1) > 0.9)  # Should be approximately normalized


def test_additional_squeeze_operations():
    """Test additional squeeze operations in value and reward loss functions."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test scalar value loss without extra dimensions (should not squeeze)
    pred_no_squeeze = jr.normal(key, (3,))
    target_no_squeeze = jr.normal(jr.split(key)[0], (3,))
    loss_no_squeeze = losses_lib.compute_scalar_value_loss(pred_no_squeeze, target_no_squeeze)
    assert loss_no_squeeze.shape == (3,)
    
    # Test scalar reward loss without extra dimensions (should not squeeze) 
    reward_loss_no_squeeze = losses_lib.compute_scalar_reward_loss(pred_no_squeeze, target_no_squeeze)
    assert reward_loss_no_squeeze.shape == (3,)
    
    # Test edge case where both prediction and target have shape (B, 1)
    pred_both_squeeze = jr.normal(key, (3, 1))
    target_both_squeeze = jr.normal(jr.split(key)[0], (3, 1))
    
    value_loss_both = losses_lib.compute_scalar_value_loss(pred_both_squeeze, target_both_squeeze)
    reward_loss_both = losses_lib.compute_scalar_reward_loss(pred_both_squeeze, target_both_squeeze)
    
    assert value_loss_both.shape == (3,)
    assert reward_loss_both.shape == (3,)


# --- Test Enhanced Entropy Functions ---
def test_compute_policy_entropy_enhanced():
    """Test enhanced discrete policy entropy computation with edge cases."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    # Standard case
    logits = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    entropy = losses_lib.compute_policy_entropy(logits)
    assert entropy.shape == (2,)
    assert jnp.all(entropy >= 0.0)  # Entropy should be non-negative
    
    # High entropy case (uniform distribution)
    uniform_logits = jnp.zeros((1, 4))  # Equal logits = uniform distribution
    uniform_entropy = losses_lib.compute_policy_entropy(uniform_logits)
    expected_uniform = jnp.log(4.0)  # log(num_actions) for uniform
    assert jnp.allclose(uniform_entropy, expected_uniform, rtol=1e-5)
    
    # Low entropy case (deterministic distribution)
    deterministic_logits = jnp.array([[10.0, 0.0, 0.0]])  # Very peaked
    deterministic_entropy = losses_lib.compute_policy_entropy(deterministic_logits)
    assert deterministic_entropy[0] < 0.1  # Should be very low
    
    # Test error handling
    try:
        invalid_logits = jnp.array([1.0, 2.0])  # 1D instead of 2D
        losses_lib.compute_policy_entropy(invalid_logits)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "2D" in str(e)


def test_compute_continuous_policy_entropy_normal():
    """Test continuous policy entropy for normal distribution."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test normal distribution entropy
    batch_size = 3
    action_dim = 2
    # Parameters: [means, log_stds]
    distribution_params = jr.normal(key, (batch_size, 2 * action_dim))
    
    entropy = losses_lib.compute_continuous_policy_entropy(
        distribution_params, distribution_type="normal"
    )
    assert entropy.shape == (batch_size,)
    assert jnp.all(entropy > 0.0)  # Should be positive for reasonable std
    
    # Test with higher variance (should have higher entropy)
    high_var_params = distribution_params.at[:, action_dim:].set(2.0)  # High log_stds
    high_entropy = losses_lib.compute_continuous_policy_entropy(
        high_var_params, distribution_type="normal"
    )
    assert jnp.all(high_entropy > entropy)
    
    # Test numerical stability with extreme log_stds
    extreme_params = distribution_params.at[:, action_dim:].set(10.0)  # Very high
    stable_entropy = losses_lib.compute_continuous_policy_entropy(
        extreme_params, distribution_type="normal"
    )
    assert jnp.all(jnp.isfinite(stable_entropy))


def test_compute_continuous_policy_entropy_squashed_normal():
    """Test continuous policy entropy for squashed normal distribution."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test squashed normal distribution entropy
    batch_size = 2
    action_dim = 3
    distribution_params = jr.normal(key, (batch_size, 2 * action_dim))
    
    entropy = losses_lib.compute_continuous_policy_entropy(
        distribution_params, distribution_type="squashed_normal"
    )
    assert entropy.shape == (batch_size,)
    
    # Should be lower than normal entropy due to squashing
    normal_entropy = losses_lib.compute_continuous_policy_entropy(
        distribution_params, distribution_type="normal"
    )
    assert jnp.all(entropy <= normal_entropy)


def test_compute_continuous_policy_entropy_error_cases():
    """Test error handling in continuous policy entropy."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test invalid shape
    try:
        invalid_params = jr.normal(key, (3,))  # 1D instead of 2D
        losses_lib.compute_continuous_policy_entropy(invalid_params, "normal")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "2D" in str(e)
    
    # Test odd parameter dimension
    try:
        odd_params = jr.normal(key, (2, 5))  # Odd number of params
        losses_lib.compute_continuous_policy_entropy(odd_params, "normal")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "even" in str(e)
    
    # Test unsupported distribution
    try:
        params = jr.normal(key, (2, 4))
        losses_lib.compute_continuous_policy_entropy(params, "unsupported")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "unsupported" in str(e)


def test_compute_policy_entropy_general_discrete():
    """Test general entropy function with discrete actions."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    logits = jr.normal(key, (3, 5))
    
    # Test discrete entropy
    entropy_discrete = losses_lib.compute_policy_entropy_general(
        logits, action_type="discrete", distribution_type="categorical"
    )
    
    # Should match direct discrete entropy computation
    entropy_direct = losses_lib.compute_policy_entropy(logits)
    assert jnp.allclose(entropy_discrete, entropy_direct)


def test_compute_policy_entropy_general_continuous():
    """Test general entropy function with continuous actions."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    distribution_params = jr.normal(key, (3, 4))  # 2 actions, means + log_stds
    
    # Test continuous entropy  
    entropy_continuous = losses_lib.compute_policy_entropy_general(
        distribution_params, action_type="continuous", distribution_type="normal"
    )
    
    # Should match direct continuous entropy computation
    entropy_direct = losses_lib.compute_continuous_policy_entropy(
        distribution_params, distribution_type="normal"
    )
    assert jnp.allclose(entropy_continuous, entropy_direct)


def test_compute_policy_entropy_general_error_handling():
    """Test error handling in general entropy function."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    policy_output = jr.normal(key, (2, 4))
    
    # Test unsupported action type
    try:
        losses_lib.compute_policy_entropy_general(
            policy_output, action_type="unsupported", distribution_type="normal"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "action_type" in str(e)


def test_entropy_mathematical_properties():
    """Test mathematical properties of entropy functions."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test that uniform discrete distribution has maximum entropy
    num_actions = 5
    uniform_logits = jnp.zeros((1, num_actions))
    uniform_entropy = losses_lib.compute_policy_entropy(uniform_logits)
    max_entropy = jnp.log(num_actions)
    assert jnp.allclose(uniform_entropy, max_entropy, rtol=1e-5)
    
    # Test that deterministic distribution has minimum entropy
    deterministic_logits = jnp.array([[-10.0, 10.0, -10.0]])  # Very peaked
    deterministic_entropy = losses_lib.compute_policy_entropy(deterministic_logits)
    assert deterministic_entropy[0] < 0.01  # Should be very close to 0
    
    # Test entropy monotonicity for continuous distributions
    action_dim = 2
    base_params = jnp.zeros((1, 2 * action_dim))
    
    # Low variance
    low_std_params = base_params.at[0, action_dim:].set(-1.0)  # log_std = -1
    low_entropy = losses_lib.compute_continuous_policy_entropy(
        low_std_params, "normal"
    )
    
    # High variance
    high_std_params = base_params.at[0, action_dim:].set(1.0)  # log_std = 1
    high_entropy = losses_lib.compute_continuous_policy_entropy(
        high_std_params, "normal"
    )
    
    assert high_entropy[0] > low_entropy[0]  # Higher variance should have higher entropy


def test_entropy_integration_with_trainer_config():
    """Test entropy functions work with trainer configuration patterns."""
    from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
    import jax.random as jr
    
    key = jr.key(42)
    
    # Test discrete action configuration
    discrete_logits = jr.normal(key, (4, 6))
    discrete_entropy = losses_lib.compute_policy_entropy_general(
        discrete_logits,
        action_type="discrete",
        distribution_type="categorical"
    )
    assert discrete_entropy.shape == (4,)
    assert jnp.all(discrete_entropy >= 0.0)
    
    # Test continuous action configuration
    continuous_params = jr.normal(key, (4, 8))  # 4 actions * 2 params each
    continuous_entropy = losses_lib.compute_policy_entropy_general(
        continuous_params,
        action_type="continuous", 
        distribution_type="normal"
    )
    assert continuous_entropy.shape == (4,)
    assert jnp.all(continuous_entropy > 0.0)
    
    # Test squashed normal configuration
    squashed_entropy = losses_lib.compute_policy_entropy_general(
        continuous_params,
        action_type="continuous",
        distribution_type="squashed_normal"
    )
    assert squashed_entropy.shape == (4,)
    assert jnp.all(squashed_entropy <= continuous_entropy)  # Should be lower due to squashing 