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
    # Ensure targets are proper distributions matching logits shape
    target_value_distribution = _ensure_distribution(target_value_distribution, value_logits.shape[-1])
    # Normalize target distributions if not proper probabilities (sum !=1 or negative values)
    sums = jnp.sum(target_value_distribution, axis=-1, keepdims=True)
    is_prob = (jnp.all(target_value_distribution >= 0, axis=-1, keepdims=True) &
               (jnp.abs(sums - 1.0) < 1e-4))
    target_value_distribution = jnp.where(
        is_prob,
        target_value_distribution,
        jax.nn.softmax(target_value_distribution, axis=-1),
    )

    # Compute KL divergence and ensure reduction to 1D array
    base_loss = compute_kl_loss(logits=value_logits, target_probs=target_value_distribution)
    if base_loss.ndim > 1:
        base_loss = jnp.sum(base_loss, axis=-1)
    
    # Always apply IQL-style weighting (EfficientZeroV2 pattern)
    # For categorical case, compute expected values to determine error sign
    num_atoms = value_logits.shape[-1]
    # Align support range with scalar_to_support default (-300..300) used everywhere else.
    # If callers need a different range they should pre-scale logits/targets beforehand.
    support = jnp.linspace(-300.0, 300.0, num_atoms)
    
    pred_probs = jax.nn.softmax(value_logits)
    pred_value = jnp.sum(pred_probs * support, axis=-1)
    target_value = jnp.sum(target_value_distribution * support, axis=-1)
    
    error = pred_value - target_value
    value_sign = (error > 0).astype(jnp.float32)
    weights = (1.0 - value_sign) * effective_iql_param + value_sign * (1.0 - effective_iql_param)

    # Ensure weights broadcast to base_loss shape
    if base_loss.shape != weights.shape:
        # If base_loss is (N, S) and weights is (N,), sum base_loss over S.
        # Current call sites always produce matching shapes, so this branch is
        # defensive for future mixed-head experiments.  Marked no-cover to avoid
        # writing contrived tests that fabricate impossible shapes.
        if base_loss.ndim > 1 and weights.ndim == 1 and base_loss.shape[0] == weights.shape[0]:  # pragma: no cover
            base_loss = jnp.sum(base_loss, axis=-1)  # pragma: no cover
        # If weights is (N, S) and base_loss is (N,), sum weights over S
        elif weights.ndim > 1 and base_loss.ndim == 1 and weights.shape[0] == base_loss.shape[0]:  # pragma: no cover
            weights = jnp.sum(weights, axis=-1)  # pragma: no cover
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
    # Ensure targets are proper distributions matching logits shape
    target_reward_distribution = _ensure_distribution(target_reward_distribution, reward_logits.shape[-1])
    # Normalize target distributions if not proper probabilities (sum !=1 or negative values)
    sums = jnp.sum(target_reward_distribution, axis=-1, keepdims=True)
    is_prob = (jnp.all(target_reward_distribution >= 0, axis=-1, keepdims=True) &
               (jnp.abs(sums - 1.0) < 1e-4))
    target_reward_distribution = jnp.where(
        is_prob,
        target_reward_distribution,
        jax.nn.softmax(target_reward_distribution, axis=-1),
    )

    # If target includes an extra trailing singleton dimension (e.g., shape [..., 1])
    # squeeze it so shapes align with logits during KL computation.
    if target_reward_distribution.ndim == reward_logits.ndim + 1 and target_reward_distribution.shape[-1] == 1:
        target_reward_distribution = jnp.squeeze(target_reward_distribution, axis=-1)

    # Compute KL divergence loss and ensure reduction to 1D array
    loss = compute_kl_loss(logits=reward_logits, target_probs=target_reward_distribution)
    if loss.ndim > 1:
        loss = jnp.sum(loss, axis=-1)
    
    # Ensure output is always at least 1D for consistent reshaping
    return jnp.atleast_1d(loss)

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
    # loss = 0.5 * mse_loss(prediction, symlog(target))
    symlog_target = symlog(target, base)
    return 0.5 * scalar_mse_loss(prediction, symlog_target)

def compute_symlog_value_loss(prediction: jax.Array, target: jax.Array, effective_iql_param: float = 1.0, base: float = jnp.e) -> jax.Array:
    """Computes symlog value loss with IQL weighting (Action Item 18).
    
    For IQL weighting with symlog values, error calculation is performed in scalar space
    by applying symexp to predictions, while the loss itself uses symlog space.
    
    Args:
        prediction: Model predictions in symlog space
        target: Target values in raw scalar space  
        effective_iql_param: IQL weighting parameter (1.0 = fully asymmetric, 0.5 = symmetric)
        base: Base for symlog transformation (default: e for EfficientZeroV2 parity)
    """
    # Compute base symlog loss
    base_loss = compute_symlog_loss(prediction, target, base)
    
    # For IQL weighting: calculate error in scalar space using symexp on predictions
    scalar_prediction = symexp(prediction, base)  # Convert symlog prediction to scalar
    error = scalar_prediction - target  # Error in scalar space
    value_sign = (error >= 0).astype(jnp.float32)  # 1 if overestimate, 0 if underestimate
    
    # Apply IQL weighting: higher weight for underestimates (value_sign=0)
    weights = (1.0 - value_sign) * effective_iql_param + value_sign * (1.0 - effective_iql_param)
    
    return base_loss * weights

def compute_kl_loss(logits: jax.Array, target_probs: jax.Array) -> jax.Array:
    """Computes KL divergence loss (EfficientZeroV2 pattern)."""
    # Convert logits to log probabilities
    log_probs = jax.nn.log_softmax(logits)
    # KL(target || prediction) = sum(target * log(target / prediction))
    # = sum(target * (log(target) - log(prediction)))
    # Handle numerical stability
    target_log_probs = jnp.log(jnp.clip(target_probs, 1e-8, 1.0))
    kl_per_atom = target_probs * (target_log_probs - log_probs)
    loss = jnp.sum(kl_per_atom, axis=-1)  # Sum over atoms, return per-batch
    # Ensure output is always at least 1D for consistent reshaping
    return jnp.atleast_1d(loss)


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
    This implements the canonical MuZero transformation from Appendix A.2.
    
    OpenSpiel environments are classified as DMC/Gym-style (discrete actions, MLP networks,
    abstract states) rather than Atari-style (pixel inputs, CNN networks).

    Args:
        x: Scalar values to convert. Shape (...,)
        support_min: Minimum value of support range
        support_max: Maximum value of support range  
        num_atoms: Number of atoms in the support
        epsilon: Small value for numerical stability (0.001 for EfficientZeroV2 parity)

    Returns:
        Categorical distribution over support. Shape (..., num_atoms)
    """
    # EfficientZeroV2 canonical transformation for DMC/Gym environments (OpenSpiel)
    # Reference: EfficientZeroV2/ez/utils/format.py DiscreteSupport.scalar_to_vector
    
    # First transform the support range bounds (EfficientZeroV2 does this)
    def transform_one(val):
        return jnp.sign(val) * (jnp.sqrt(jnp.abs(val) + 1.0) - 1.0) + epsilon * val
    
    x_min_transformed = transform_one(support_min)
    x_max_transformed = transform_one(support_max)
    scale = (x_max_transformed - x_min_transformed) / (num_atoms - 1)
    
    # Create transformed support range
    x_range = jnp.linspace(x_min_transformed, x_max_transformed, num_atoms)
    
    # Apply DMC/Gym transformation to input values
    x_transformed = transform_one(x)
    
    # Clamp to valid transformed range
    x_transformed = jnp.clip(x_transformed, x_min_transformed, x_max_transformed)
    
    # Map to index space
    x_index = (x_transformed - x_min_transformed) / scale
    x_index = jnp.clip(x_index, 0.0, num_atoms - 1.0 - 1e-5)
    
    # Get lower and upper indices for interpolation
    x_low_idx = jnp.floor(x_index)
    x_high_idx = jnp.ceil(x_index)
    
    # Compute interpolation weights
    p_high = x_index - x_low_idx
    p_low = 1.0 - p_high

    # Convert to integer indices
    x_low_idx = jnp.clip(x_low_idx.astype(jnp.int32), 0, num_atoms - 1)
    x_high_idx = jnp.clip(x_high_idx.astype(jnp.int32), 0, num_atoms - 1)

    # Create target distribution
    target_shape = x.shape + (num_atoms,)
    target = jnp.zeros(target_shape)

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

def support_to_scalar(support_dist: jax.Array, support_min: float = -300.0, support_max: float = 300.0,
                      num_atoms: int = 601, epsilon: float = 0.001) -> jax.Array:
    """Converts categorical distribution over support back to scalar values.

    Based on EfficientZeroV2's DiscreteSupport.vector_to_scalar implementation.
    This implements the canonical MuZero inverse transformation for DMC/Gym environments.

    Args:
        support_dist: Categorical distribution over support. Shape (..., num_atoms)
                     Should be probability distribution (not logits) when coming from scalar_to_support
        support_min: Minimum value of support range
        support_max: Maximum value of support range
        num_atoms: Number of atoms in the support
        epsilon: Small value for numerical stability (0.001 for EfficientZeroV2 parity)

    Returns:
        Scalar values. Shape (...,)
    """
    def transform_one(val):
        return jnp.sign(val) * (jnp.sqrt(jnp.abs(val) + 1.0) - 1.0) + epsilon * val

    # Compute transformation parameters
    x_min_transformed = transform_one(support_min)
    x_max_transformed = transform_one(support_max)
    scale = (x_max_transformed - x_min_transformed) / (num_atoms - 1)
    x_range = jnp.linspace(x_min_transformed, x_max_transformed, num_atoms)

    # Allow both probabilities (sum≈1) and logits as input.
    # Detect whether last-dim sums to ~1; if not, treat as logits and apply softmax.
    sums = jnp.sum(support_dist, axis=-1, keepdims=True)
    value_probs = jnp.where(
        jnp.abs(sums - 1.0) < 1e-4,
        support_dist,
        jax.nn.softmax(support_dist, axis=-1),  # convert logits to probabilities
    )

    batch_shape = support_dist.shape[:-1]
    value_support = jnp.broadcast_to(x_range, batch_shape + (num_atoms,))
    z = jnp.sum(value_support * value_probs, axis=-1) / scale

    # (Debug prints removed for cleanliness)

    sign = jnp.sign(z)
    abs_z = jnp.abs(z)
    sqrt_term = jnp.sqrt(1.0 + 4.0 * epsilon * (abs_z * scale + 1.0 + epsilon))
    numerator = sqrt_term - 1.0
    fraction = numerator / (2.0 * epsilon)
    output = sign * (fraction ** 2 - 1.0)

    # (Debug prints removed)

    # Numerical stability guards (match EfficientZeroV2 behaviour)
    output = jnp.where(jnp.isnan(output), 0.0, output)
    output = jnp.where(jnp.abs(output) < epsilon, 0.0, output)

    return output

def _ensure_distribution(target: jax.Array, num_atoms: int, support_min: float = -300.0, support_max: float = 300.0, epsilon: float = 0.001) -> jax.Array:
    """Utility: ensure *target* is a categorical distribution with *num_atoms* atoms.

    If *target* already has the correct last-dimension size it is returned unchanged.
    Otherwise, it is treated as scalar(s) and converted via `scalar_to_support`.
    """
    # If already distribution with correct support size, return as-is
    if target.ndim > 0 and target.shape[-1] == num_atoms:
        return target

    # Convert scalar targets to distribution
    target_flat = target.reshape(-1)
    target_support = jax.vmap(scalar_to_support, in_axes=(0, None, None, None))(
        target_flat, support_min, support_max, num_atoms
    )
    return target_support.reshape(target.shape + (num_atoms,))

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

    NOTE: OpenSpiel environments use discrete action spaces only.
    This function is provided for API completeness but should not be used
    in practice for OpenSpiel-based MuZero implementations.

    Args:
        distribution_params: Parameters of the continuous distribution.
        distribution_type: Type of distribution (not used for OpenSpiel)

    Returns:
        Per-batch entropy values. Shape (batch_size,)

    Raises:
        NotImplementedError: Always raised since OpenSpiel uses discrete actions only
    """
    # CRITICAL FIX (Action Item #7): OpenSpiel environments are discrete action only
    raise NotImplementedError(
        "Continuous action entropy is not supported for OpenSpiel environments. "
        "OpenSpiel games use discrete action spaces only. "
        "Use compute_policy_entropy() for discrete action entropy computation."
    )


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

# --- Temperature Scheduling for MCTS (Action Item 20) ---

def get_temperature(training_step: int, config: Any) -> float:
    """Computes temperature for MCTS based on current training step (EfficientZeroV2 pattern).
    
    This function replicates PyTorch's `agent.get_temperature(trained_steps)` functionality
    for temperature scheduling in MCTS policy target generation and data collection.
    
    Args:
        training_step: Current training step count
        config: MuZeroConfig containing temperature scheduling parameters
        
    Returns:
        Temperature value for MCTS at the current training step
        
    EfficientZeroV2 Pattern:
        - Uses linear decay from temperature_init to temperature_final
        - Decay occurs over temperature_decay_steps
        - Temperature is clamped to not go below temperature_final
        - Can be disabled by setting change_temperature=False
    """
    if not config.change_temperature:
        # No temperature scheduling, return initial temperature
        return config.temperature_init
    
    if training_step >= config.temperature_decay_steps:
        # Decay period finished, return final temperature
        return config.temperature_final
    
    # Linear decay from init to final over decay_steps
    decay_fraction = training_step / config.temperature_decay_steps
    temperature = config.temperature_init + decay_fraction * (config.temperature_final - config.temperature_init)
    
    # Ensure temperature doesn't go below final temperature
    return jnp.maximum(temperature, config.temperature_final)


def get_temperature_schedule(max_steps: int, config: Any) -> jax.Array:
    """Generates a complete temperature schedule array for analysis/debugging.
    
    Args:
        max_steps: Maximum number of training steps to generate schedule for
        config: MuZeroConfig containing temperature scheduling parameters
        
    Returns:
        Array of temperature values for each step from 0 to max_steps-1
        
    This function is useful for:
        - Visualizing the temperature schedule
        - Testing temperature scheduling behavior
        - Analysis of temperature decay patterns
    """
    steps = jnp.arange(max_steps)
    
    if not config.change_temperature:
        # No temperature scheduling, constant temperature
        return jnp.full(max_steps, config.temperature_init)
    
    # Compute decay fraction for all steps
    decay_fractions = steps / config.temperature_decay_steps
    
    # Linear interpolation between init and final temperatures
    temperatures = config.temperature_init + decay_fractions * (config.temperature_final - config.temperature_init)
    
    # Clamp to not go below final temperature (for steps beyond decay period)
    temperatures = jnp.maximum(temperatures, config.temperature_final)
    
    return temperatures


def validate_temperature_config(config: Any) -> bool:
    """Validates temperature configuration parameters.
    
    Args:
        config: MuZeroConfig containing temperature parameters
        
    Returns:
        True if configuration is valid, raises ValueError if invalid
        
    Validation checks:
        - temperature_init > 0 (positive temperature)
        - temperature_final > 0 (positive temperature) 
        - temperature_decay_steps > 0 (positive decay duration)
        - temperature_init >= temperature_final (decay should reduce temperature)
    """
    if config.temperature_init <= 0:
        raise ValueError(f"temperature_init must be positive, got {config.temperature_init}")
    
    if config.temperature_final <= 0:
        raise ValueError(f"temperature_final must be positive, got {config.temperature_final}")
    
    if config.temperature_decay_steps <= 0:
        raise ValueError(f"temperature_decay_steps must be positive, got {config.temperature_decay_steps}")
    
    if config.temperature_init < config.temperature_final:
        raise ValueError(f"temperature_init ({config.temperature_init}) should be >= temperature_final ({config.temperature_final}) for decay")
    
    return True 
