import jax
import jax.numpy as jnp
import optax # For softmax_cross_entropy
from typing import Any # For PyTrees

def scalar_mse_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Computes Mean Squared Error loss for scalar predictions (per-batch-item).
    
    Returns:
        Per-batch-item losses. Shape (batch_size,).
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} must match target shape {target.shape}") # pragma: no cover
    
    squared_errors = jnp.square(prediction - target)
    
    # Sum across all dimensions except the first (batch dimension)
    # This handles both scalar inputs (1D) and multi-dimensional features
    if squared_errors.ndim == 1:
        # Already per-batch-item for 1D inputs
        return squared_errors
    else:
        # Sum across feature dimensions for each batch item
        return jnp.sum(squared_errors, axis=tuple(range(1, squared_errors.ndim)))

def cross_entropy_loss_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Computes softmax cross-entropy loss (per-item).
    
    Args:
        logits: Network output before softmax. Shape (batch_size, num_classes).
        targets: Target probabilities (e.g., MCTS policy). Shape (batch_size, num_classes).
    
    Returns:
        Per-item cross-entropy losses. Shape (batch_size,).
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Logits shape {logits.shape} must match targets shape {targets.shape}") # pragma: no cover
    if logits.ndim != 2:
        raise ValueError(f"Logits and targets must be 2D (batch_size, num_classes), got {logits.ndim}D") # pragma: no cover
    return optax.softmax_cross_entropy(logits=logits, labels=targets)  # Return per-item losses, shape (batch_size,)

def l2_regularization(params: Any, weight: float) -> jax.Array:
    """Computes L2 regularization loss for a PyTree of parameters."""
    if weight <= 0:
        return jnp.array(0.0) # pragma: no cover
    
    sum_sq_params = jax.tree_util.tree_reduce(
        lambda acc, p: acc + jnp.sum(p**2), params, initializer=0.0
    )
    return weight * sum_sq_params

# --- Policy Loss ---
def compute_policy_loss(policy_logits: jax.Array, target_policy: jax.Array) -> jax.Array:
    """Computes policy loss using cross-entropy."""
    return cross_entropy_loss_with_logits(logits=policy_logits, targets=target_policy)

# --- Value Loss ---
def compute_scalar_value_loss(value_prediction: jax.Array, target_value: jax.Array, iql_weight: float = 1.0) -> jax.Array:
    """Computes value loss for scalar values using MSE with optional IQL weighting."""
    # Ensure predictions and targets are squeezed if they have an extra dim of 1
    if value_prediction.ndim > 1 and value_prediction.shape[-1] == 1:
        value_prediction = jnp.squeeze(value_prediction, axis=-1) # pragma: no cover
    if target_value.ndim > 1 and target_value.shape[-1] == 1:
        target_value = jnp.squeeze(target_value, axis=-1) # pragma: no cover
        
    base_loss = scalar_mse_loss(prediction=value_prediction, target=target_value)
    
    # Apply IQL-style weighting if specified (EfficientZeroV2 pattern)
    if iql_weight != 1.0:
        # IQL weighting: apply different weights based on sign of error
        # EfficientZeroV2: value_weight = (1 - value_sign) * iql_weight + value_sign * (1 - iql_weight)
        # where value_sign = (error > 0).float()
        error = value_prediction - target_value
        value_sign = (error > 0).astype(jnp.float32)
        weights = (1.0 - value_sign) * iql_weight + value_sign * (1.0 - iql_weight)
        return base_loss * weights
    
    return base_loss

def compute_categorical_value_loss(value_logits: jax.Array, target_value_distribution: jax.Array, iql_weight: float = 1.0) -> jax.Array:
    """Computes value loss for categorical distributions using cross-entropy with optional IQL weighting."""
    base_loss = cross_entropy_loss_with_logits(logits=value_logits, targets=target_value_distribution)
    
    # Apply IQL-style weighting if specified
    if iql_weight != 1.0:
        # For categorical case, compute expected values to determine error sign
        num_atoms = value_logits.shape[-1]
        support = jnp.linspace(-1.0, 1.0, num_atoms)  # Assume normalized support
        
        pred_probs = jax.nn.softmax(value_logits)
        pred_value = jnp.sum(pred_probs * support, axis=-1)
        target_value = jnp.sum(target_value_distribution * support, axis=-1)
        
        error = pred_value - target_value
        value_sign = (error > 0).astype(jnp.float32)
        weights = (1.0 - value_sign) * iql_weight + value_sign * (1.0 - iql_weight)
        return base_loss * weights
        
    return base_loss

# --- Reward Loss ---
def compute_scalar_reward_loss(reward_prediction: jax.Array, target_reward: jax.Array) -> jax.Array:
    """Computes reward loss for scalar rewards using MSE."""
    # Ensure predictions and targets are squeezed
    if reward_prediction.ndim > 1 and reward_prediction.shape[-1] == 1:
        reward_prediction = jnp.squeeze(reward_prediction, axis=-1) # pragma: no cover
    if target_reward.ndim > 1 and target_reward.shape[-1] == 1:
        target_reward = jnp.squeeze(target_reward, axis=-1) # pragma: no cover
    return scalar_mse_loss(prediction=reward_prediction, target=target_reward)

def compute_categorical_reward_loss(reward_logits: jax.Array, target_reward_distribution: jax.Array) -> jax.Array:
    """Computes reward loss for categorical distributions using KL divergence (EfficientZeroV2 pattern).
    
    This aligns with PyTorch's approach where categorical rewards use KL divergence for consistency
    with categorical value loss, instead of cross-entropy.
    """
    return compute_kl_loss(logits=reward_logits, target_probs=target_reward_distribution)

def compute_symlog_loss(prediction: jax.Array, target: jax.Array, base: float = 2.0) -> jax.Array:
    """Computes loss using symlog transformation (EfficientZeroV2 pattern)."""
    # Transform both prediction and target to symlog space
    symlog_pred = symlog(prediction, base)
    symlog_target = symlog(target, base)
    return scalar_mse_loss(symlog_pred, symlog_target)

def compute_kl_loss(logits: jax.Array, target_probs: jax.Array) -> jax.Array:
    """Computes KL divergence loss (EfficientZeroV2 pattern)."""
    # Convert logits to log probabilities
    log_probs = jax.nn.log_softmax(logits)
    # KL(target || prediction) = sum(target * log(target / prediction))
    # = sum(target * (log(target) - log(prediction)))
    # Handle numerical stability
    target_log_probs = jnp.log(jnp.clip(target_probs, 1e-8, 1.0))
    kl_per_atom = target_probs * (target_log_probs - log_probs)
    return jnp.sum(kl_per_atom, axis=-1)  # Sum over atoms, return per-batch


def compute_projection_consistency_loss(
    projection_current_step: jax.Array,  # Projection of h_k
    projection_initial_step: jax.Array   # Projection of h_0 (with stop_gradient applied)
) -> jax.Array:
    """Computes SSL consistency loss between two projected states (EfficientZeroV2 style).

    Args:
        projection_current_step: Projected hidden state from the current unroll step.
        projection_initial_step: Projected hidden state from the initial step (t=0).
                                 Should have stop_gradient applied externally.

    Returns:
        Per-item SSL consistency losses. Shape (batch_size,).
    """
    # EfficientZeroV2 style: bidirectional consistency loss 
    # loss = projection_loss(p_obs, p_pred.detach()) + projection_loss(p_pred, p_obs.detach())
    # where projection_loss(p, z) = -cosine_similarity(p, z) (per-item, not averaged)

    # Compute cosine similarity in both directions
    sim1 = optax.cosine_similarity(projection_current_step, jax.lax.stop_gradient(projection_initial_step))
    sim2 = optax.cosine_similarity(jax.lax.stop_gradient(projection_current_step), projection_initial_step)
    
    # Note: optax.cosine_similarity normalizes inputs, ensuring output is in [-1, 1]
    # No explicit clipping needed - this was redundant
    
    # Return negative cosine similarity as loss (per-item), bidirectional
    return -sim1 + -sim2  # Per-item losses, shape (batch_size,)

# --- Symlog functions for EfficientZeroV2 parity ---
def symlog(x: jax.Array, base: float = 2.0) -> jax.Array:
    """Symmetric logarithm transformation as used in EfficientZeroV2."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0) / jnp.log(base)

def symexp(x: jax.Array, base: float = 2.0) -> jax.Array:
    """Inverse of symlog transformation."""
    return jnp.sign(x) * (jnp.power(base, jnp.abs(x)) - 1.0)

# --- Discrete Support Transformations for EfficientZeroV2 parity ---
def scalar_to_support(x: jax.Array, support_min: float = -300.0, support_max: float = 300.0, 
                      num_atoms: int = 601, epsilon: float = 0.001) -> jax.Array:
    """Converts scalar values to categorical distribution over support.
    
    Based on EfficientZeroV2's DiscreteSupport.scalar_to_vector implementation.
    
    Args:
        x: Scalar values to convert. Shape (...,)
        support_min: Minimum value of support range
        support_max: Maximum value of support range  
        num_atoms: Number of atoms in the support
        epsilon: Small value for numerical stability
        
    Returns:
        Categorical distribution over support. Shape (..., num_atoms)
    """
    # Create support range
    scale = (support_max - support_min) / (num_atoms - 1)
    support_range = jnp.linspace(support_min, support_max, num_atoms)
    
    # Apply symlog-like transformation
    sign = jnp.sign(x)
    x_transformed = sign * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + epsilon * x
    
    # Normalize to support range
    x_normalized = x_transformed / scale
    
    # Clamp to valid range
    x_clamped = jnp.clip(x_normalized, support_min / scale, support_max / scale - 1e-5)
    x_shifted = x_clamped - support_min / scale
    
    # Get lower and upper indices for interpolation
    x_low_idx = jnp.floor(x_shifted)
    x_high_idx = jnp.ceil(x_shifted)
    
    # Compute interpolation weights
    p_high = x_shifted - x_low_idx
    p_low = 1.0 - p_high
    
    # Create target distribution
    target_shape = x.shape + (num_atoms,)
    target = jnp.zeros(target_shape)
    
    # Scatter weights to appropriate indices
    x_low_idx = jnp.clip(x_low_idx.astype(jnp.int32), 0, num_atoms - 1)
    x_high_idx = jnp.clip(x_high_idx.astype(jnp.int32), 0, num_atoms - 1)
    
    # Use advanced indexing to scatter values
    batch_indices = jnp.arange(x.shape[0])[:, None] if x.ndim > 0 else jnp.array([0])
    
    # Handle different input shapes
    if x.ndim == 0:  # Scalar input
        target = target.at[x_low_idx].add(p_low)
        target = target.at[x_high_idx].add(p_high)
    elif x.ndim == 1:  # 1D input
        target = target.at[batch_indices.squeeze(), x_low_idx].add(p_low)
        target = target.at[batch_indices.squeeze(), x_high_idx].add(p_high)
    else:  # Higher dimensional - flatten and reshape
        x_flat = x.reshape(-1)
        target_flat = target.reshape(-1, num_atoms)
        batch_flat = jnp.arange(x_flat.shape[0])
        x_low_flat = x_low_idx.reshape(-1)
        x_high_flat = x_high_idx.reshape(-1)
        p_low_flat = p_low.reshape(-1)
        p_high_flat = p_high.reshape(-1)
        
        target_flat = target_flat.at[batch_flat, x_low_flat].add(p_low_flat)
        target_flat = target_flat.at[batch_flat, x_high_flat].add(p_high_flat)
        target = target_flat.reshape(target_shape)
    
    return target

def support_to_scalar(logits: jax.Array, support_min: float = -300.0, support_max: float = 300.0,
                      num_atoms: int = 601, epsilon: float = 0.001) -> jax.Array:
    """Converts categorical distribution over support back to scalar values.
    
    Based on EfficientZeroV2's DiscreteSupport.vector_to_scalar implementation.
    
    Args:
        logits: Logits over support atoms. Shape (..., num_atoms)
        support_min: Minimum value of support range
        support_max: Maximum value of support range
        num_atoms: Number of atoms in the support  
        epsilon: Small value for numerical stability
        
    Returns:
        Scalar values. Shape (...,)
    """
    # Create support range
    scale = (support_max - support_min) / (num_atoms - 1)
    support_range = jnp.linspace(support_min, support_max, num_atoms)
    
    # Convert logits to probabilities
    value_probs = jax.nn.softmax(logits, axis=-1)
    
    # Compute expected value
    value = jnp.sum(value_probs * support_range, axis=-1) / scale
    
    # Apply inverse transformation
    sign = jnp.sign(value)
    abs_value = jnp.abs(value)
    
    # Inverse of symlog-like transformation: x = sign * ((sqrt(1 + 4*eps*(|v| + 1 + eps)) - 1) / (2*eps))^2 - 1)
    sqrt_term = jnp.sqrt(1.0 + 4.0 * epsilon * (abs_value * scale + 1.0 + epsilon))
    output = ((sqrt_term - 1.0) / (2.0 * epsilon)) ** 2 - 1.0
    output = sign * output
    
    # Handle numerical issues
    output = jnp.where(jnp.isnan(output), 0.0, output)
    output = jnp.where(jnp.abs(output) < epsilon, 0.0, output)
    
    return output

def compute_policy_entropy(policy_logits: jax.Array) -> jax.Array:
    """Computes entropy of policy distribution for regularization.
    
    Args:
        policy_logits: Policy logits. Shape (batch_size, num_actions)
        
    Returns:
        Per-batch entropy values. Shape (batch_size,)
    """
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    probs = jax.nn.softmax(policy_logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return entropy 