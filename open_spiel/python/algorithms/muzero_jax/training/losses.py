import jax
import jax.numpy as jnp
import optax # For softmax_cross_entropy
from typing import Any # For PyTrees

def scalar_mse_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Computes Mean Squared Error loss for scalar predictions."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} must match target shape {target.shape}") # pragma: no cover
    return jnp.mean(jnp.square(prediction - target))

def cross_entropy_loss_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Computes softmax cross-entropy loss.
    
    Args:
        logits: Network output before softmax. Shape (batch_size, num_classes).
        targets: Target probabilities (e.g., MCTS policy). Shape (batch_size, num_classes).
    
    Returns:
        Mean cross-entropy loss.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Logits shape {logits.shape} must match targets shape {targets.shape}") # pragma: no cover
    if logits.ndim != 2:
        raise ValueError(f"Logits and targets must be 2D (batch_size, num_classes), got {logits.ndim}D") # pragma: no cover
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))

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
def compute_scalar_value_loss(value_prediction: jax.Array, target_value: jax.Array) -> jax.Array:
    """Computes value loss for scalar values using MSE."""
    # Ensure predictions and targets are squeezed if they have an extra dim of 1
    if value_prediction.ndim > 1 and value_prediction.shape[-1] == 1:
        value_prediction = jnp.squeeze(value_prediction, axis=-1) # pragma: no cover
    if target_value.ndim > 1 and target_value.shape[-1] == 1:
        target_value = jnp.squeeze(target_value, axis=-1) # pragma: no cover
    return scalar_mse_loss(prediction=value_prediction, target=target_value)

def compute_categorical_value_loss(value_logits: jax.Array, target_value_distribution: jax.Array) -> jax.Array:
    """Computes value loss for categorical distributions using cross-entropy."""
    return cross_entropy_loss_with_logits(logits=value_logits, targets=target_value_distribution)

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
    """Computes reward loss for categorical distributions using cross-entropy."""
    return cross_entropy_loss_with_logits(logits=reward_logits, targets=target_reward_distribution)

# TODO: Implement support_to_scalar and scalar_to_support if needed for direct use,
# or ensure network outputs/targets are already in the correct format for these loss functions.
# EfficientZeroV2 uses these for converting between scalar and supported representations.
# For now, these losses assume inputs are already appropriately formatted.

def compute_projection_consistency_loss(
    projection_current_step: jax.Array,  # Projection of h_k
    projection_initial_step: jax.Array   # Projection of h_0
) -> jax.Array:
    """Computes SSL consistency loss between two projected states (SimSiam-style).

    Args:
        projection_current_step: Projected hidden state from the current unroll step.
        projection_initial_step: Projected hidden state from the initial step (t=0).
                                 One of the pair will have stop_gradient applied.

    Returns:
        The SSL consistency loss.
    """
    # EfficientZeroV2 style:
    # loss = projection_loss(p_obs, p_pred.detach()) + projection_loss(p_pred, p_obs.detach())
    # where projection_loss(p, z) = -cosine_similarity(p, z).mean()

    sim1 = optax.cosine_similarity(projection_current_step, jax.lax.stop_gradient(projection_initial_step))
    sim2 = optax.cosine_similarity(jax.lax.stop_gradient(projection_current_step), projection_initial_step)

    # Clip similarities to be within [-1, 1] before computing loss
    clipped_sim1 = jnp.clip(sim1, -1.0, 1.0)
    clipped_sim2 = jnp.clip(sim2, -1.0, 1.0)

    loss1 = -jnp.mean(clipped_sim1)
    loss2 = -jnp.mean(clipped_sim2)
    return loss1 + loss2 