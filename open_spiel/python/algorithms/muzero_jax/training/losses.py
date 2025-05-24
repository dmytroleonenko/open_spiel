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
def compute_scalar_value_loss(value_prediction: jax.Array, target_value: jax.Array, effective_iql_param: float = 0.5) -> jax.Array:
    """Computes value loss for scalar values using MSE with IQL weighting.
    
    Args:
        value_prediction: Predicted values
        target_value: Target values  
        effective_iql_param: Effective IQL parameter (0.5 for symmetric loss, other values for asymmetric)
    """
    # Ensure predictions and targets are squeezed if they have an extra dim of 1
    if value_prediction.ndim > 1 and value_prediction.shape[-1] == 1:
        value_prediction = jnp.squeeze(value_prediction, axis=-1) # pragma: no cover
    if target_value.ndim > 1 and target_value.shape[-1] == 1:
        target_value = jnp.squeeze(target_value, axis=-1) # pragma: no cover
        
    base_loss = scalar_mse_loss(prediction=value_prediction, target=target_value)
    
    # Always apply IQL-style weighting (EfficientZeroV2 pattern)
    # IQL weighting: apply different weights based on sign of error
    # EfficientZeroV2: value_weight = (1 - value_sign) * effective_iql_param + value_sign * (1 - effective_iql_param)
    # where value_sign = (error > 0).float()
    error = value_prediction - target_value
    value_sign = (error > 0).astype(jnp.float32)
    weights = (1.0 - value_sign) * effective_iql_param + value_sign * (1.0 - effective_iql_param)
    return base_loss * weights

def compute_categorical_value_loss(value_logits: jax.Array, target_value_distribution: jax.Array, effective_iql_param: float = 0.5) -> jax.Array:
    """Computes value loss for categorical distributions using KL divergence with IQL weighting (EfficientZeroV2 pattern).
    
    This aligns with PyTorch's approach where categorical values use KL divergence for consistency
    with the EfficientZeroV2 implementation, instead of cross-entropy.
    
    Args:
        value_logits: Predicted value logits
        target_value_distribution: Target value distribution
        effective_iql_param: Effective IQL parameter (0.5 for symmetric loss, other values for asymmetric)
    """
    # EfficientZeroV2 pattern: use KL divergence for categorical value loss
    base_loss = compute_kl_loss(logits=value_logits, target_probs=target_value_distribution)
    
    # Always apply IQL-style weighting (EfficientZeroV2 pattern)
    # For categorical case, compute expected values to determine error sign
    num_atoms = value_logits.shape[-1]
    support = jnp.linspace(-1.0, 1.0, num_atoms)  # Assume normalized support
    
    pred_probs = jax.nn.softmax(value_logits)
    pred_value = jnp.sum(pred_probs * support, axis=-1)
    target_value = jnp.sum(target_value_distribution * support, axis=-1)
    
    error = pred_value - target_value
    value_sign = (error > 0).astype(jnp.float32)
    weights = (1.0 - value_sign) * effective_iql_param + value_sign * (1.0 - effective_iql_param)
    return base_loss * weights

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

def compute_symlog_loss(prediction: jax.Array, target: jax.Array, base: float = jnp.e) -> jax.Array:
    """Computes loss using symlog transformation (EfficientZeroV2 pattern).
    
    EfficientZeroV2 expects predictions to already be in symlog space and only 
    applies symlog transformation to targets.
    
    Args:
        prediction: Model predictions already in symlog space
        target: Target values in raw scalar space
        base: Base for symlog transformation (default: e for EfficientZeroV2 parity)
    """
    # PyTorch EfficientZeroV2 pattern: prediction is already in symlog space
    # loss = 0.5 * (prediction.squeeze() - symlog(target)) ** 2
    symlog_target = symlog(target, base)
    return 0.5 * scalar_mse_loss(prediction, symlog_target)

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
def symlog(x: jax.Array, base: float = jnp.e) -> jax.Array:
    """Symmetric logarithm transformation as used in EfficientZeroV2.
    
    EfficientZeroV2 uses natural log (base e) by default for consistency with PyTorch implementation.
    """
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0) / jnp.log(base)

def symexp(x: jax.Array, base: float = jnp.e) -> jax.Array:
    """Inverse of symlog transformation.
    
    EfficientZeroV2 uses natural log (base e) by default for consistency with PyTorch implementation.
    """
    return jnp.sign(x) * (jnp.power(base, jnp.abs(x)) - 1.0)

# --- Discrete Support Transformations for EfficientZeroV2 parity ---
def scalar_to_support(x: jax.Array, support_min: float = -300.0, support_max: float = 300.0, 
                      num_atoms: int = 601, epsilon: float = 0.001) -> jax.Array:
    """Converts scalar values to categorical distribution over support.
    
    Based on EfficientZeroV2's DiscreteSupport.scalar_to_vector implementation.
    For OpenSpiel environments, this uses the standard Atari-style transformation.
    
    Args:
        x: Scalar values to convert. Shape (...,)
        support_min: Minimum value of support range
        support_max: Maximum value of support range  
        num_atoms: Number of atoms in the support
        epsilon: Small value for numerical stability
        
    Returns:
        Categorical distribution over support. Shape (..., num_atoms)
    """
    # Apply symlog-like transformation (EfficientZeroV2 Atari style for OpenSpiel)
    sign = jnp.sign(x)
    x_transformed = sign * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + epsilon * x
    
    # Map to [0, num_atoms-1] index space  
    # Rescale transformed values to fit in the support range
    scale = (support_max - support_min) / (num_atoms - 1)
    
    # For better numerical behavior, we'll map the transformed space to index space more directly
    # The transformation typically maps small values to small values, so we can use this fact
    x_index_space = (x_transformed - support_min) / scale
    
    # Clamp to valid index range
    x_index_space = jnp.clip(x_index_space, 0.0, num_atoms - 1.0 - 1e-5)
    
    # Get lower and upper indices for interpolation
    x_low_idx = jnp.floor(x_index_space)
    x_high_idx = jnp.ceil(x_index_space)
    
    # Compute interpolation weights
    p_high = x_index_space - x_low_idx
    p_low = 1.0 - p_high
    
    # Create target distribution
    target_shape = x.shape + (num_atoms,)
    target = jnp.zeros(target_shape)
    
    # Scatter weights to appropriate indices
    x_low_idx = jnp.clip(x_low_idx.astype(jnp.int32), 0, num_atoms - 1)
    x_high_idx = jnp.clip(x_high_idx.astype(jnp.int32), 0, num_atoms - 1)
    
    # Handle different input shapes
    if x.ndim == 0:  # Scalar input
        target = target.at[x_low_idx].add(p_low)
        target = target.at[x_high_idx].add(p_high)
    elif x.ndim == 1:  # 1D input
        batch_indices = jnp.arange(x.shape[0])
        target = target.at[batch_indices, x_low_idx].add(p_low)
        target = target.at[batch_indices, x_high_idx].add(p_high)
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
    Uses Newton's method to precisely invert the forward transformation.
    
    Args:
        logits: Logits over support atoms. Shape (..., num_atoms)
        support_min: Minimum value of support range
        support_max: Maximum value of support range
        num_atoms: Number of atoms in the support  
        epsilon: Small value for numerical stability (should match scalar_to_support)
        
    Returns:
        Scalar values. Shape (...,)
    """
    # Create support range
    support_range = jnp.linspace(support_min, support_max, num_atoms)
    
    # Convert logits to probabilities and compute expected value
    value_probs = jax.nn.softmax(logits, axis=-1)
    y_target = jnp.sum(value_probs * support_range, axis=-1)  # This is the transformed value
    
    # Now we need to solve for x in: y_target = sign(x) * (sqrt(abs(x) + 1) - 1) + epsilon * x
    # Use Newton's method for precision
    
    def forward_transform(x):
        """The forward transformation from scalar_to_support"""
        sign = jnp.sign(x)
        return sign * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + epsilon * x
    
    def forward_derivative(x):
        """Derivative of the forward transformation"""
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        # d/dx [sign(x) * (sqrt(abs(x) + 1) - 1) + epsilon * x]
        # For x > 0: d/dx [sqrt(x + 1) - 1 + epsilon * x] = 1/(2*sqrt(x + 1)) + epsilon
        # For x < 0: d/dx [-sqrt(-x + 1) + 1 + epsilon * x] = 1/(2*sqrt(-x + 1)) + epsilon
        # For x = 0: derivative is 0.5 + epsilon (by continuity)
        sqrt_term = jnp.sqrt(abs_x + 1.0)
        derivative = 0.5 / jnp.maximum(sqrt_term, 1e-8) + epsilon
        return derivative
    
    # Newton's method to solve forward_transform(x) - y_target = 0
    # Initialize with a reasonable guess: for small values, x ≈ y_target / (0.5 + epsilon)
    x = y_target / (0.5 + epsilon)
    
    # Newton iterations (typically 3-5 iterations give excellent precision)
    for _ in range(5):
        fx = forward_transform(x) - y_target
        fpx = forward_derivative(x)
        # Avoid division by zero
        fpx = jnp.where(jnp.abs(fpx) < 1e-10, 1e-10, fpx)
        x_new = x - fx / fpx
        
        # Check for convergence (optional, but helps with numerical stability)
        converged = jnp.abs(x_new - x) < 1e-8
        x = jnp.where(converged, x, x_new)
        
        # Clamp to reasonable bounds to avoid numerical issues
        x = jnp.clip(x, -1000.0, 1000.0)
    
    # Final cleanup: handle edge cases
    x = jnp.where(jnp.isnan(x), 0.0, x)
    x = jnp.where(jnp.isinf(x), jnp.sign(x) * 1000.0, x)
    
    return x

def compute_policy_entropy(policy_logits: jax.Array) -> jax.Array:
    """Computes entropy of discrete policy distribution for regularization.
    
    For discrete actions, entropy is computed as H(π) = -∑ π(a) log π(a).
    This encourages exploration by discouraging overly deterministic policies.
    
    Args:
        policy_logits: Policy logits for discrete actions. Shape (batch_size, num_actions)
        
    Returns:
        Per-batch entropy values. Shape (batch_size,)
    """
    if policy_logits.ndim != 2:
        raise ValueError(f"Policy logits must be 2D (batch_size, num_actions), got {policy_logits.ndim}D") # pragma: no cover
    
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    probs = jax.nn.softmax(policy_logits, axis=-1)
    # Add small epsilon to prevent numerical issues with log(0)
    entropy = -jnp.sum(probs * jnp.clip(log_probs, min=-20.0, max=None), axis=-1)
    return entropy


def compute_continuous_policy_entropy(
    distribution_params: jax.Array, 
    distribution_type: str = "normal"
) -> jax.Array:
    """Computes entropy of continuous policy distribution for regularization.
    
    This function supports various continuous distributions that might be used
    in continuous action spaces, preparing for EfficientZeroV2's full feature set.
    
    Args:
        distribution_params: Parameters of the continuous distribution.
                           For 'normal': Shape (batch_size, 2 * action_dim) where first half is means,
                           second half is log_stds.
                           For 'categorical': Falls back to discrete entropy.
        distribution_type: Type of distribution ('normal', 'squashed_normal', 'truncated_normal', etc.)
        
    Returns:
        Per-batch entropy values. Shape (batch_size,)
        
    Raises:
        NotImplementedError: For distribution types not yet implemented
        ValueError: For invalid distribution parameters
    """
    if distribution_type == "normal":
        # For multivariate normal with diagonal covariance:
        # H(X) = 0.5 * log((2πe)^k * |Σ|) = 0.5 * k * log(2πe) + 0.5 * log(|Σ|)
        # For diagonal Σ: log(|Σ|) = sum(log(σ_i^2)) = 2 * sum(log(σ_i))
        
        if distribution_params.ndim != 2:
            raise ValueError(f"Distribution params must be 2D (batch_size, 2*action_dim), got {distribution_params.ndim}D") # pragma: no cover
        
        param_dim = distribution_params.shape[-1]
        if param_dim % 2 != 0:
            raise ValueError(f"Distribution params size must be even (means + log_stds), got {param_dim}") # pragma: no cover
        
        action_dim = param_dim // 2
        means = distribution_params[:, :action_dim]  # (batch_size, action_dim)
        log_stds = distribution_params[:, action_dim:]  # (batch_size, action_dim)
        
        # Clamp log_stds for numerical stability
        log_stds = jnp.clip(log_stds, min=-5.0, max=2.0)
        
        # Entropy = 0.5 * action_dim * log(2πe) + sum(log_stds)
        constant_term = 0.5 * action_dim * jnp.log(2 * jnp.pi * jnp.e)
        variable_term = jnp.sum(log_stds, axis=-1)
        entropy = constant_term + variable_term
        
        return entropy
        
    elif distribution_type == "squashed_normal":
        # For SquashedNormal (typically tanh-squashed), we need to account for the
        # log absolute determinant of the Jacobian of the transformation
        # This is a simplified version - full implementation would require the actual
        # sampled actions to compute the Jacobian term accurately
        
        if distribution_params.ndim != 2:
            raise ValueError(f"Distribution params must be 2D (batch_size, 2*action_dim), got {distribution_params.ndim}D") # pragma: no cover
        
        # Start with normal entropy
        normal_entropy = compute_continuous_policy_entropy(distribution_params, "normal")
        
        # Approximate Jacobian correction for tanh squashing
        # This is a rough approximation - exact computation requires sampled actions
        action_dim = distribution_params.shape[-1] // 2
        log_stds = distribution_params[:, action_dim:]
        stds = jnp.exp(jnp.clip(log_stds, min=-5.0, max=2.0))
        
        # Approximation: reduce entropy by expected squashing effect
        # This is conservative and encourages exploration
        squashing_correction = -0.5 * jnp.sum(jnp.log(1.0 + stds**2), axis=-1)
        
        return normal_entropy + squashing_correction
        
    elif distribution_type == "categorical":
        # Fallback to discrete entropy computation
        return compute_policy_entropy(distribution_params) # pragma: no cover
        
    else:
        raise NotImplementedError(f"Entropy computation for '{distribution_type}' distribution not implemented. "
                                f"Supported types: 'normal', 'squashed_normal', 'categorical'") # pragma: no cover


def compute_policy_entropy_general(
    policy_output: jax.Array,
    action_type: str = "discrete",
    distribution_type: str = "categorical"
) -> jax.Array:
    """General entropy computation function that handles both discrete and continuous actions.
    
    This is the main entry point for policy entropy computation in the trainer,
    designed to handle EfficientZeroV2's various action space configurations.
    
    Args:
        policy_output: Policy network output. Format depends on action_type:
                      - discrete: logits over actions, shape (batch_size, num_actions)
                      - continuous: distribution parameters, shape (batch_size, param_dim)
        action_type: Type of action space ('discrete' or 'continuous')
        distribution_type: Distribution type for continuous actions or 'categorical' for discrete
        
    Returns:
        Per-batch entropy values. Shape (batch_size,)
    """
    if action_type == "discrete":
        return compute_policy_entropy(policy_output)
    elif action_type == "continuous":
        return compute_continuous_policy_entropy(policy_output, distribution_type)
    else:
        raise ValueError(f"Unsupported action_type: {action_type}. Must be 'discrete' or 'continuous'") # pragma: no cover 