import jax
import jax.numpy as jnp
import optax
import flax.linen as nn  # To be replaced with nnx if MuZeroNetwork uses it
from flax.training import train_state # May use custom state with nnx
from typing import Any, Callable, Dict, Tuple, NamedTuple
from dataclasses import dataclass, field

# Assuming MuZeroNetwork is available from:
# from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork, MuZeroNetworkParams
# For now, we might use a placeholder or define a simpler one if not fully available/compatible

class Batch(NamedTuple):
    observations: jnp.ndarray  # (B, T, ...) or (B, ...) for initial observation
    actions: jnp.ndarray  # (B, T) actions taken
    rewards: jnp.ndarray  # (B, T) rewards received
    target_policies: jnp.ndarray  # (B, T, num_actions) target policy from MCTS
    target_values: jnp.ndarray  # (B, T) target value from MCTS
    target_rewards: jnp.ndarray # (B, T) target rewards for reward prediction
    weights: jnp.ndarray # (B, T) for prioritized replay or loss weighting

@dataclass
class Losses:
    policy_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    value_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    reward_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    l2_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    total_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))

@dataclass
class TrainingMetrics:
    losses: Losses = field(default_factory=Losses)
    gradient_norm: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    # Add other metrics as needed, e.g., learning_rate

# Placeholder for model state. If using flax.training.train_state:
# class TrainingState(train_state.TrainState):
#   target_params: Any # For target network updates

# If using pure NNX, state management will be different.
# For now, let's assume a structure that can hold params and opt_state
# And potentially target_params for EMA.
# This will be refined once MuZeroNetwork and NNX integration is clearer.

class TrainingState(NamedTuple):
    # This assumes MuZeroNetwork is an nnx.Module and `model_state` holds its state.
    # If it's a regular Flax Module, params would be here.
    model_state: Any # This would be nnx.State or similar if using NNX directly
    tx: optax.GradientTransformation # Optimizer
    opt_state: optax.OptState
    target_model_state: Any # For target network
    step: int = 0

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """Computes weighted cross-entropy loss."""
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(targets * log_probs, axis=-1)
    if weights is not None:
        loss *= weights
        return jnp.sum(loss) / jnp.sum(weights)
    return jnp.mean(loss)

def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """Computes weighted mean squared error."""
    if predictions.ndim > targets.ndim and predictions.shape[-1] == 1:
        predictions = jnp.squeeze(predictions, axis=-1)
    loss = (predictions - targets)**2
    if weights is not None:
        loss *= weights
        return jnp.sum(loss) / jnp.sum(weights)
    return jnp.mean(loss)

def l2_regularization(params: Any, l2_reg_weight: float) -> jnp.ndarray:
    """Computes L2 regularization loss for model parameters."""
    if l2_reg_weight == 0:
        return jnp.array(0.0)
    
    # This needs to be adapted based on how params are structured with NNX
    # For flax.traverse_util.flatten_dict approach:
    # weight_sum_sq = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params) if p.ndim > 1)
    
    # For NNX, we'd iterate through model_state.params or similar.
    # This is a placeholder and will need adjustment.
    # For now, let's assume params is a pytree of arrays.
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves: # pragma: no cover
        return jnp.array(0.0)

    # A common practice is to only apply L2 to weights, not biases.
    # This depends on how parameters are named/structured.
    # For simplicity now, apply to all leaves that are arrays.
    weight_sum_sq = sum(jnp.sum(p**2) for p in leaves if isinstance(p, jnp.ndarray) and p.ndim > 1) # Heuristic for weights
    return 0.5 * l2_reg_weight * weight_sum_sq

# More to come: train_step, Trainer class/orchestration 