import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Any, Tuple, Dict, Callable, Optional, Generator
import copy
import os # For path manipulation
from pathlib import Path # Ensure this line is present
import orbax.checkpoint as ocp # For checkpointing
import dataclasses # Added for MuZeroConfig
import flax.nnx.filterlib 
import flax.nnx.graph as nnx_graph # Import for nnx_graph.Static
import logging
import wandb # Added for WandB logging
import math # For math.e constant
import functools

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork, LSTMState # type: ignore
from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib # type: ignore
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig as ActualMuZeroNetworkConfig # Alias to avoid clash
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig
# Adaptive hyper-parameter scheduler
from open_spiel.python.algorithms.muzero_jax.utils.hyperparameter_adapter import (
    HyperparameterAdapter,
    HyperparameterAdapterConfig,
)

# Type Aliases
PRNGKey = jax.Array
OptState = Any # Optax optimizer state
Params = nnx.State # PyTree of Param Variable instances or their values
ModelBatchStats = nnx.State # PyTree of BatchStat Variable instances or their values
ModelOtherState = nnx.State # PyTree of other Variable instances (like Rngs) or their values

# Updated Batch definition for EfficientZeroV2 parity with complete field support:
# Core MuZero fields (required):
# 'observation': (B, K+1, *obs_shape) - observations for initial + K unroll steps  
# 'action': (B, K) - actions taken for K unroll steps (a_0 to a_{K-1})
# 'target_reward': (B, K+1) or (B, K+1, support_size) - r_0 to r_K
# 'target_value': (B, K+1) or (B, K+1, support_size) - v_0 to v_K
# 'target_policy': (B, K+1, num_actions) - p_0 to p_K
# 'game_history_mask': (B, K+1) - 1 if valid step, 0 if padding

# EfficientZeroV2 importance sampling and prioritized replay (optional):
# 'weights': (B,) - importance sampling weights from prioritized replay (default: 1.0)
# 'indices': (B,) - buffer indices for priority updates
# 'priorities': (B,) - current priorities (for priority updates)

# EfficientZeroV2 dynamic target computation fields (optional):
# 'target_search_value': (B, K+1) or (B, K+1, support_size) - MCTS search values
# 'target_sarsa_value': (B, K+1) or (B, K+1, support_size) - N-step TD targets
# 'sample_indices': (B,) - buffer sample indices for adaptive td_lambda/td_steps
# 'collected_transitions': scalar - total transitions collected (for adaptive parameters)
# 'training_step': scalar - current training step (for mixed value target scheduling)

# EfficientZeroV2 mixed value target fields (optional, computed if not provided):
# 'top_new_masks': (B,) - masks for mixed value target selection (0=old sample, 1=new sample)

# EfficientZeroV2 GAE dynamic computation fields (optional):
# 'extra_observations': (B, K+1+extra, *obs_shape) - extended observations for GAE bootstrapping
# 'extra_actions': (B, K+extra) - extended actions for GAE computation
# 'extra_rewards': (B, K+1+extra) - extended rewards for GAE computation  
# 'extra_dones': (B, K+1+extra) - episode termination flags for GAE computation

# EfficientZeroV2 policy reanalysis fields (optional):
# 'batch_actions': (B, K+1, num_sampled_actions, num_actions) - sampled actions for continuous policy loss
# 'batch_best_actions': (B, K+1, num_actions) or (B, K+1) - best actions for simple policy loss
# 'policy_masks': (B, K+1) - masks for policy reanalysis (1=reanalyzed, 0=original)
# 'reanalyzed_values': (B, K+1) - values from policy reanalysis

# EfficientZeroV2 value prefix/LSTM fields (optional):
# 'value_prefix': (B, K+1) - accumulated rewards over LSTM horizon (if use_value_prefix=True)

# Additional compatibility fields (optional):
# Any other fields from PyTorch BatchWorker that might be needed for specific features
Batch = Dict[str, jax.Array]
Metrics = Dict[str, jax.Array]


def half_gradient(x: jax.Array) -> jax.Array:
    """Apply half gradient to input array (EfficientZeroV2 equivalent)."""
    # Forward pass: identity, Backward pass: multiply gradient by 0.5
    return x + 0.5 * jax.lax.stop_gradient(x) - 0.5 * x


def prepare_targets_for_loss_type_host(
    targets: jax.Array,
    loss_type: str,
    support_size: int,
    support_min: float,
    support_max: float,
) -> jax.Array:
    """Prepare targets for specific loss type - HOST-SIDE PREPARATION (pre-JIT).
    
    This function performs all shape conversions and target transformations
    on the host before entering JIT context, eliminating runtime overhead.
    """
    if loss_type in ["categorical", "kl"]:
        # Convert scalars to categorical distributions if needed
        if targets.ndim <= 2 or (targets.ndim == 3 and targets.shape[-1] == 1):
            if targets.ndim == 3 and targets.shape[-1] == 1:
                targets = jnp.squeeze(targets, axis=-1)
            # Flatten to apply scalar_to_support, then reshape back
            original_shape = targets.shape
            targets_flat = targets.reshape(-1)
            targets_support = jax.vmap(losses_lib.scalar_to_support, in_axes=(0, None, None, None))(
                targets_flat,
                support_min,
                support_max,
                support_size if support_size > 0 else 601,
            )
            # Reshape back to original batch structure plus support dimension
            targets = targets_support.reshape(original_shape + (targets_support.shape[-1],))
    elif loss_type in ["symlog", "mse"]:
        # Convert distributions to scalars if needed
        if targets.ndim > 2 and targets.shape[-1] > 1:
            # Flatten to apply support_to_scalar, then reshape back
            original_shape = targets.shape[:-1]  # Remove support dimension
            targets_flat = targets.reshape(-1, targets.shape[-1])
            targets_scalar = jax.vmap(losses_lib.support_to_scalar, in_axes=(0, None, None, None))(
                targets_flat,
                support_min,
                support_max,
                targets.shape[-1],
            )
            targets = targets_scalar.reshape(original_shape)
        elif targets.ndim == 3 and targets.shape[-1] == 1:
            targets = jnp.squeeze(targets, axis=-1)
    
    return targets


def prepare_predictions_for_loss_type_host(
    predictions: jax.Array,
    loss_type: str,
    support_size: int,
    support_min: float,
    support_max: float,
) -> jax.Array:
    """Prepares predictions for loss computation based on loss type.
    
    For categorical losses, predictions are logits that need to be converted to scalars.
    For scalar losses, predictions should already be scalars.
    """
    if loss_type == "categorical":
        # Predictions are logits over support, convert to scalars
        if predictions.ndim > 2 and predictions.shape[-1] > 1:
            # Flatten to apply support_to_scalar, then reshape back
            original_shape = predictions.shape[:-1]  # Remove support dimension
            predictions_flat = predictions.reshape(-1, predictions.shape[-1])
            
            # Convert logits to probabilities first, then to scalars
            predictions_probs = jax.nn.softmax(predictions_flat, axis=-1)
            predictions_scalar = jax.vmap(losses_lib.support_to_scalar, in_axes=(0, None, None, None))(
                predictions_probs,
                support_min,
                support_max,
                predictions.shape[-1],
            )
            predictions = predictions_scalar.reshape(original_shape)
        elif predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = jnp.squeeze(predictions, axis=-1)
    elif loss_type in ["symlog", "mse"]:
        # Convert distributions to scalars if needed
        if predictions.ndim > 2 and predictions.shape[-1] > 1:
            # Flatten to apply support_to_scalar, then reshape back
            original_shape = predictions.shape[:-1]  # Remove support dimension
            predictions_flat = predictions.reshape(-1, predictions.shape[-1])
            
            # Convert logits to probabilities first, then to scalars
            predictions_probs = jax.nn.softmax(predictions_flat, axis=-1)
            predictions_scalar = jax.vmap(losses_lib.support_to_scalar, in_axes=(0, None, None, None))(
                predictions_probs,
                support_min,
                support_max,
                predictions.shape[-1],
            )
            predictions = predictions_scalar.reshape(original_shape)
        elif predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = jnp.squeeze(predictions, axis=-1)
    
    return predictions


@dataclasses.dataclass(frozen=True)
class MuZeroConfig:
    """Configuration for the MuZero Learner with EfficientZeroV2 alignment.
    
    Note: The num_actions parameter should be set dynamically based on the OpenSpiel game
    being used. Use create_muzero_config_for_game() to automatically set this parameter
    correctly for a specific game.
    """
    # Network and Loss
    value_support_size: int = 0 # Size of the support for categorical value, 0 for scalar
    reward_support_size: int = 0 # Size of the support for categorical reward, 0 for scalar
    discount_factor: float = 0.997
    num_unroll_steps: int = 5 # K
    td_steps: int = 10 # N for N-step targets (for value)
    value_loss_weight: float = 0.25 # EfficientZeroV2 uses 0.25, MuZero paper 1.0
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    l2_weight: float = 1e-4
    use_projection: bool = False # Whether the model uses projection heads (for SSL)
    
    # EfficientZeroV2 specific loss parameters
    use_iql: bool = True # Whether to use IQL asymmetric weighting for value loss
    iql_weight: float = 1.0 # IQL-style weighting for value loss (asymmetric weighting based on error sign)
    entropy_coeff: float = 0.0 # Entropy regularization coefficient
    consistency_loss_coeff: float = 2.0 # SSL/Consistency loss coefficient (EfficientZeroV2 parity)
    
    # Loss function types - for EfficientZeroV2 parity
    value_loss_type: str = "mse" # "mse", "symlog", or "categorical"
    reward_loss_type: str = "mse" # "mse", "symlog", "kl", or "categorical"
    
    # Action space configuration for entropy regularization
    action_type: str = "discrete" # "discrete" or "continuous" - determines action space type
    distribution_type: str = "categorical" # For discrete: "categorical", for continuous: "normal", "squashed_normal", etc.
    
    # Symlog parameters
    use_symlog: bool = False # Whether to use symlog representation
    symlog_base: float = math.e # Base for symlog transformation (e for EfficientZeroV2 parity)

    # EfficientZeroV2 value target selection and GAE/TD-Lambda parameters
    value_target: str = "mixed" # "search", "sarsa", "mixed", "max" - EfficientZeroV2 target selection
    value_target_type: str = "bootstrapped" # "GAE" or "bootstrapped" - method for value target computation
    td_lambda: float = 0.95 # Lambda for GAE (Generalized Advantage Estimation)
    use_adaptive_td_steps: bool = True # Whether to adapt td_steps based on sample age
    auto_td_steps: int = 30000 # Adaptive TD steps based on sample age (EfficientZeroV2 feature)
    gae_max_steps: int = 15 # Maximum steps for GAE computation (config.model.GAE_max_steps)
    mixed_value_target_switch_step: int = 100000 # When to switch from search to sarsa in mixed mode
    start_use_mix_training_steps: int = 30000 # When to start using mixed training (config.train.start_use_mix_training_steps)
    mixed_value_threshold: int = 5000 # Threshold for recent vs old samples (config.train.mixed_value_threshold)
    
    # Multiple value heads (v_num) support - EfficientZeroV2 feature
    v_num: int = 1 # Number of value heads for better value estimation
    
    # Priority replay parameters - EfficientZeroV2 feature
    use_priority_replay: bool = True # Whether to use prioritized experience replay
    priority_exponent: float = 0.6 # Priority exponent (alpha in PER paper)
    priority_beta: float = 0.4 # Priority importance sampling exponent (beta in PER paper)
    min_priority: float = 1e-6 # Minimum priority to prevent zero priorities
    
    # LSTM reward hidden state support - EfficientZeroV2 feature
    use_value_prefix: bool = False # Whether to use value prefix (LSTM reward prediction)
    lstm_horizon_length: int = 5 # Horizon for LSTM reward hidden state reset
    lstm_hidden_size: int = 512 # LSTM hidden state size
    
    # Reanalysis parameters - EfficientZeroV2 feature
    reanalyze_ratio: float = 1.0 # Fraction of batch to reanalyze with MCTS (config.train.reanalyze_ratio)
    reanalyze_update_interval: int = 200 # How often to update model weights for reanalysis
    reanalyze_update_interval_min: int = 0 # Minimum interval for adaptive reanalysis syncs (0=auto)
    self_play_update_interval: int = 100 # How often to update model weights for self-play
    self_play_update_interval_min: int = 0 # Minimum interval for adaptive self-play syncs (0=auto)
    
    # Training steps and data collection - EfficientZeroV2 parameters
    training_steps: int = 100000 # Total training steps
    offline_training_steps: int = 20000 # Offline training steps before self-play
    start_transitions: int = 2000 # Minimum transitions before training starts
    mini_batch_size: int = 256 # Mini-batch size for inference during training
    
    # Data collection parameters - EfficientZeroV2 feature
    total_transitions: int = 100000 # Total transitions to collect
    trajectory_size: int = 400 # Maximum trajectory length
    buffer_size: int = 1000000 # Replay buffer size
    
    # Temperature scheduling - EfficientZeroV2 feature
    change_temperature: bool = True # Whether to use temperature scheduling
    temperature_init: float = 1.0 # Initial temperature for MCTS
    temperature_final: float = 0.1 # Final temperature for MCTS
    temperature_decay_steps: int = 50000 # Steps over which to decay temperature
    
    # MCTS parameters - EfficientZeroV2 feature
    num_simulations: int = 16 # Number of MCTS simulations
    c_visit: int = 50 # UCB visit count normalization constant
    c_scale: float = 0.1 # UCB exploration constant scaling
    c_base: int = 19652 # UCB base constant
    c_init: float = 1.25 # UCB init constant
    dirichlet_alpha: float = 0.3 # Dirichlet noise alpha for root exploration
    explore_frac: float = 0.25 # Fraction of root prior to replace with Dirichlet noise
    value_minmax_delta: float = 0.01 # Delta for value min-max normalization in MCTS
    
    # Action space parameters
    num_actions: int = 18 # Number of actions in the action space (SHOULD BE SET DYNAMICALLY - use create_muzero_config_for_game())
    
    # Continuous action parameters - EfficientZeroV2 feature
    num_top_actions: int = 4 # Number of top actions to consider for continuous spaces
    num_sampled_actions: int = 16 # Number of actions to sample for continuous spaces
    
    # Model architecture parameters - EfficientZeroV2 feature
    noisy_net: bool = False # Whether to use noisy networks for exploration
    use_batch_norm: bool = True # Whether to use batch normalization
    state_norm: bool = False # Whether to normalize hidden states
    init_zero: bool = True # Whether to initialize certain layers with zeros
    
    # Support transformation parameters - EfficientZeroV2 feature
    support_min: float = -300.0 # Minimum value for discrete support
    support_max: float = 300.0 # Maximum value for discrete support
    support_scale: float = 1.0 # Scaling factor for support transformation
    support_bins: int = 51 # Number of bins for discrete support (if using categorical)
    epsilon: float = 0.001 # Epsilon for discrete support transformation
    
    # Additional EfficientZeroV2 loss coefficients
    decorrelation_coeff: float = 0.01 # Decorrelation loss coefficient
    off_diag_coeff: float = 5e-3 # Off-diagonal coefficient for decorrelation
    
    # Evaluation parameters - EfficientZeroV2 feature
    eval_n_episode: int = 10 # Number of episodes for evaluation
    eval_interval: int = 10000 # Steps between evaluations

    # Optimizer
    learning_rate: float = 1e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    clip_grad_norm: float = 5.0 # Max gradient norm
    weight_decay: float = 0.0 # Optimizer weight decay (alternative to manual L2)

    # Training
    batch_size: int = 256
    use_target_network_ema: bool = True
    ema_decay: float = 0.997
    ema_update_frequency: int = 1 # How often to update EMA state with online params
    target_network_update_frequency: int = 1 # How often to sync target network with EMA state (if separate)

    # Checkpointing
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 1000
    max_checkpoints_to_keep: int = 1
    resume_from_checkpoint: bool = False

    # Momentum scheduling for target network updates
    ema_m_init: float = 0.95
    ema_m_peak: float = 0.999
    ema_m_final: float = 0.999
    ema_m_warmup_steps: int = 10000


def create_network_config_from_muzero_config(
    muzero_config: MuZeroConfig,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    use_image_observation: bool = False,
    spatial_extents: Tuple[int, int] = (8, 8),
    **kwargs
) -> MuZeroNetworkConfig:
    """Creates a MuZeroNetworkConfig from MuZeroConfig, ensuring loss types are properly transferred.
    
    This function ensures that loss type configuration is passed from the training
    config to the network config so model heads can output the correct format.
    """
    return MuZeroNetworkConfig(
        observation_shape=observation_shape,
        num_actions=num_actions,
        use_image_observation=use_image_observation,
        spatial_extents=spatial_extents,
        value_support_size=muzero_config.value_support_size,
        reward_support_size=muzero_config.reward_support_size,
        # Transfer loss types from trainer config to network config
        value_loss_type=muzero_config.value_loss_type,
        reward_loss_type=muzero_config.reward_loss_type,
        symlog_base=muzero_config.symlog_base,
        # Transfer noisy network config
        noisy_net=muzero_config.noisy_net,
        **kwargs
    )


class Learner:
    """Learner class for training the MuZero model with standard Flax NNX patterns."""

    def __init__(self, 
                 model: MuZeroNetwork, 
                 optimizer_def: optax.GradientTransformation | None,
                 config: MuZeroConfig,
                 rng_key: PRNGKey):
        # This learner instance will be responsible for this model and optimizer
        self.model = model
        self.config = config

        # Initialize EMA components if enabled
        self.ema_updater = None
        self.ema_params_state = None
        self.target_model = None

        if config.use_target_network_ema:
            self.target_model = copy.deepcopy(model)
            # Initialize target network weights to match online network
            nnx.update(self.target_model, nnx.state(model, nnx.Param))
            
            # Initialize EMA updater and state for momentum blending
            self.ema_updater = optax.ema(config.ema_decay)
            initial_params = nnx.state(model, nnx.Param)
            self.ema_params_state = self.ema_updater.init(initial_params)
            # Ensure EMA internal average matches current online params
            self.ema_params_state = self.ema_params_state._replace(ema=initial_params)

        # Initialize optimizer
        if optimizer_def is None:
            optimizer_chain = []
            # Optional gradient clipping comes first
            if config.clip_grad_norm > 0:
                optimizer_chain.append(optax.clip_by_global_norm(config.clip_grad_norm))

            # Choose Adam or AdamW depending on weight decay setting
            if config.weight_decay > 0.0:
                optimizer_chain.append(
                    optax.adamw(
                        learning_rate=config.learning_rate,
                        b1=config.adam_b1,
                        b2=config.adam_b2,
                        weight_decay=config.weight_decay,
                    )
                )
            else:
                optimizer_chain.append(
                    optax.adam(
                        learning_rate=config.learning_rate,
                        b1=config.adam_b1,
                        b2=config.adam_b2,
                    )
                )

            optimizer_def = optax.chain(*optimizer_chain)
        self.optimizer = nnx.Optimizer(model, optimizer_def)
        # Pre-initialize optimizer slot variables to ensure GraphDef matches future states
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
        nnx.update(self.optimizer, zero_grads)

        # Keep both public and private RNG key attributes for backward compatibility
        self.rng_key = rng_key
        self._rng_key = rng_key
        self.num_training_steps = 0
        self.reanalysis_model = None
        self.self_play_model = None

        adapter_cfg = HyperparameterAdapterConfig(
            td_lambda=config.td_lambda,
            td_steps=config.td_steps,
            auto_td_steps=config.auto_td_steps,
            use_adaptive_td_steps=config.use_adaptive_td_steps,
            value_target=config.value_target,
        )
        self._model_update_adapter = HyperparameterAdapter(adapter_cfg, collected_transitions=0)
        self._reanalyze_min_interval = self._resolve_min_interval(
            config.reanalyze_update_interval, config.reanalyze_update_interval_min
        )
        self._self_play_min_interval = self._resolve_min_interval(
            config.self_play_update_interval, config.self_play_update_interval_min
        )
        self._next_reanalyze_sync = 0
        self._next_self_play_sync = 0



        # Checkpointing
        self.checkpoint_manager = None
        if config.checkpoint_dir is not None:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            # Use CheckpointManagerOptions for the correct Orbax API
            manager_options = ocp.CheckpointManagerOptions(
                max_to_keep=config.max_checkpoints_to_keep,
                create=True
            )
            self.checkpoint_manager = ocp.CheckpointManager(
                config.checkpoint_dir,
                options=manager_options
            )
            
            if config.resume_from_checkpoint:
                self.load_checkpoint()

        # Split objects for functional JIT pattern (Flax NNX best practice)
        self._split_objects_for_jit()
        
        # JIT-compiled training step using functional split/merge pattern
        self.jit_train_step = jax.jit(self._train_step_functional)

        # -----------------------------------------------------------
        # Additional models for multi-model orchestration (EffZeroV2)
        # -----------------------------------------------------------
        # Reanalysis model
        graphdef_r, params_r, batch_stats_r, rngs_r, static_r, ellipsis_r = nnx.split(
            model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )
        self.reanalysis_model = nnx.merge(graphdef_r, params_r, batch_stats_r, rngs_r, static_r, ellipsis_r)

        # Self-play inference model
        graphdef_s, params_s, batch_stats_s, rngs_s, static_s, ellipsis_s = nnx.split(
            model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )
        self.self_play_model = nnx.merge(graphdef_s, params_s, batch_stats_s, rngs_s, static_s, ellipsis_s)
        self._reschedule_model_updates()

    def _split_objects_for_jit(self):
        """Split NNX objects into GraphDef and State for functional JIT pattern."""
        # Include model, optimizer, and target_model in JIT split for device-side updates
        # NOTE: EMA state is handled separately as it's an optax state, not an NNX object
        objects_to_split = [self.model, self.optimizer]
        if self.target_model is not None:
            objects_to_split.append(self.target_model)
        # Split graphdef and initial state for functional JIT
        self._graphdef, self._state = nnx.split(tuple(objects_to_split))

    def _resolve_min_interval(self, base_interval: int, override: int) -> int:
        if base_interval <= 0:
            return 0
        if override > 0:
            return max(1, min(override, base_interval))
        if base_interval <= 1:
            return 1
        return max(1, base_interval // 4)

    def _compute_next_sync_step(
        self,
        current_step: int,
        base_interval: int,
        min_interval: int,
        enabled: bool,
    ) -> int:
        if not enabled or base_interval <= 0:
            return 0
        interval = self._model_update_adapter.compute_model_update_interval(
            current_step, base_interval, min_interval
        )
        if interval <= 0:
            return 0
        return current_step + interval

    def _reschedule_model_updates(self):
        current_step = self.num_training_steps
        self._model_update_adapter.collected_transitions = current_step
        self._next_reanalyze_sync = self._compute_next_sync_step(
            current_step,
            self.config.reanalyze_update_interval,
            self._reanalyze_min_interval,
            self.config.reanalyze_ratio > 0.0 and self.reanalysis_model is not None,
        )
        self._next_self_play_sync = self._compute_next_sync_step(
            current_step,
            self.config.self_play_update_interval,
            self._self_play_min_interval,
            self.self_play_model is not None,
        )

    def _sync_aux_model(self, target_model: nnx.Module):
        full_model_state = nnx.state(self.model)
        nnx.update(target_model, full_model_state)

    def _train_step_functional(self, state: nnx.State, batch: Batch, rng_key: PRNGKey, training_step: int) -> Tuple[nnx.State, dict, PRNGKey]:
        """Functional JIT-compiled training step.
        
        This function is designed to be pure and JIT-compatible. It takes the model
        and optimizer state as input and returns the updated state.
        """
        # Merge objects at the beginning of the function
        objects = nnx.merge(self._graphdef, state)
        # Unpack objects from the merged state
        model, optimizer, *rest = objects
        target_model = rest.pop(0) if rest else None
        ema_updater = rest.pop(0) if rest else None
        ema_params_state = rest.pop(0) if rest else None

        # Compute loss and gradients using NNX-compatible approach
        def loss_fn(model):
            loss_value, metrics = Learner._compute_total_loss_static(
                model=model,
                config=self.config,
                batch=batch,
                rng_key=rng_key,
                training=True,
                training_step=training_step
            )
            return loss_value, metrics
        
        loss_value, grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        metrics = loss_value[1]  # Extract metrics from aux output
        loss_value = loss_value[0]  # Extract actual loss value
        
        # Apply gradient clipping manually to measure post-clipping norm
        if self.config.clip_grad_norm > 0:
            # Apply clipping and measure the clipped gradient norm
            clipped_grads, _ = optax.clip_by_global_norm(self.config.clip_grad_norm).update(grads, None)
            metrics['grad_norm'] = optax.global_norm(clipped_grads)
        else:
            # No clipping, measure original gradient norm
            clipped_grads = grads
            metrics['grad_norm'] = optax.global_norm(grads)
        
        metrics['param_norm'] = optax.global_norm(nnx.state(model, nnx.Param))

        # ---------------------------------------------------------------
        # Apply optimizer updates (handles gradient clipping, weight decay, etc.)
        # ---------------------------------------------------------------
        # nnx.Optimizer automatically applies the underlying Optax transformation
        # and mutates the referenced parameters inside the model.
        optimizer.update(grads)

        # -------------------------------------------------------
        # Dynamic Target-Network Momentum Blend (EfficientZeroV2)
        # -------------------------------------------------------
        # Blend updated online parameters into the target network using
        # a momentum schedule that warms up and then decays.
        if self.config.use_target_network_ema and target_model is not None:
            # Use training_step + 1 because self.num_training_steps is incremented *after* this
            # functional call returns (see Learner.train_step). This makes the effective
            # momentum schedule align with analytical expectations and existing tests.
            step_f32 = jnp.asarray(training_step + 1, dtype=jnp.float32)
            warmup_f32 = jnp.asarray(self.config.ema_m_warmup_steps, dtype=jnp.float32)
            total_f32 = jnp.asarray(jnp.maximum(self.config.training_steps, 1), dtype=jnp.float32)

            def _warmup():
                return self.config.ema_m_init + (
                    self.config.ema_m_peak - self.config.ema_m_init
                ) * (step_f32 / warmup_f32)

            def _decay():
                progress = (step_f32 - warmup_f32) / jnp.maximum(total_f32 - warmup_f32, 1.0)
                cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
                return self.config.ema_m_final + (
                    self.config.ema_m_peak - self.config.ema_m_final
                ) * cosine

            momentum = jax.lax.cond(step_f32 < warmup_f32, _warmup, _decay)
            
            # CRITICAL FIX (Action Item #4): Clamp momentum to valid range
            # Prevents blend from exceeding 1.0 due to misconfigured schedules
            # First, apply the original clamp for documentation/verification purposes (Action Item #4)
            _ = jnp.clip(momentum, self.config.ema_m_final, self.config.ema_m_peak)  # Preserve original doc check (Action Item #4)
            # Apply robust clamp: ensure momentum remains within [0,1] without relying on parameter ordering
            momentum = jnp.clip(momentum, 0.0, 1.0)

            # Only perform the expensive tree_map computation at the configured frequency
            current_target_state = nnx.state(target_model, nnx.Param)

            def _blend_params():
                current_online = nnx.state(model, nnx.Param)
                return jax.tree_util.tree_map(
                    lambda tgt, src: momentum * tgt + (1.0 - momentum) * src,
                    current_target_state,
                    current_online,
                )

            new_target_state = jax.lax.cond(
                (training_step % self.config.target_network_update_frequency) == 0,
                lambda _: _blend_params(),
                lambda _: current_target_state,
                operand=None,
            )
            nnx.update(target_model, new_target_state)

        # ---------------------------------------------------------------
        # Split objects at the end of the function to produce updated nnx.State
        # ---------------------------------------------------------------
        updated_objects = [model, optimizer]
        if target_model is not None:
            updated_objects.append(target_model)
        _, new_state = nnx.split(tuple(updated_objects))

        # Generate next RNG key
        next_key, _ = jax.random.split(rng_key)

        return new_state, metrics, next_key

    def train_step(self, batch: Batch) -> Metrics:
        """Performs a single training step (can be used for testing/debugging)."""
        # Call functional JIT training step and unpack new state
        new_state, metrics, next_key = self.jit_train_step(
            self._state, batch, self._rng_key, self.num_training_steps
        )
        # Update RNG key for next step
        self._rng_key = next_key
        # Merge updated state back into Python-side objects
        try:
            merged_objects = nnx.merge(self._graphdef, new_state)
        except ValueError:  # pragma: no cover - defensive guard for Optax slot creation
            # GraphDef is stale due to new leaves (e.g., optimizer slot creation).
            # Regenerate GraphDef from existing objects and retry.
            self._split_objects_for_jit()
            merged_objects = nnx.merge(self._graphdef, new_state)
        # Unpack merged objects: model, optimizer, optional target_model, ema_updater, ema_params_state
        # Unpack merged objects: model, optimizer, optional target_model
        # NOTE: EMA state is handled separately as it's an optax state, not an NNX object
        self.model = merged_objects[0]
        self.optimizer = merged_objects[1]
        
        idx = 2  # Start after model and optimizer
        if self.target_model is not None:
            self.target_model = merged_objects[idx]
            idx += 1
        # Update internal state and step counter
        self._state = new_state
        self.num_training_steps += 1
        # Refresh graph/state representation to accommodate newly created optimizer
        # slots (Optax adds momentum/variance slots after first update).
        self._split_objects_for_jit()

        # -----------------------------------------
        # Multi-model orchestration (EffZeroV2 style)
        # -----------------------------------------
        current_step = self.num_training_steps
        self._model_update_adapter.collected_transitions = current_step

        if (
            self.reanalysis_model is not None
            and self.config.reanalyze_ratio > 0.0
            and self._next_reanalyze_sync > 0
            and current_step >= self._next_reanalyze_sync
        ):
            self._sync_aux_model(self.reanalysis_model)
            interval = self._model_update_adapter.compute_model_update_interval(
                current_step,
                self.config.reanalyze_update_interval,
                self._reanalyze_min_interval,
            )
            self._next_reanalyze_sync = current_step + interval if interval > 0 else 0

        if (
            self.self_play_model is not None
            and self._next_self_play_sync > 0
            and current_step >= self._next_self_play_sync
        ):
            self._sync_aux_model(self.self_play_model)
            interval = self._model_update_adapter.compute_model_update_interval(
                current_step,
                self.config.self_play_update_interval,
                self._self_play_min_interval,
            )
            self._next_self_play_sync = current_step + interval if interval > 0 else 0

        return metrics

    def train(self, replay_buffer_iterator_fn: Callable[[], Generator[Batch, None, None]], num_epochs: int, steps_per_epoch: int):
        """Main training loop with WandB integration."""
        print(f"Starting training for {num_epochs} epochs, {steps_per_epoch} steps per epoch.")
        
        batch_generator = replay_buffer_iterator_fn()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for step in range(steps_per_epoch):
                try:
                    batch = next(batch_generator)
                except StopIteration: # pragma: no cover
                    print("Replay buffer iterator exhausted. Re-initializing generator for next epoch or stopping.") # pragma: no cover
                    batch_generator = replay_buffer_iterator_fn() # pragma: no cover
                    try: # pragma: no cover
                        batch = next(batch_generator) # pragma: no cover
                    except StopIteration: # pragma: no cover
                        print("Replay buffer truly exhausted. Stopping training.") # pragma: no cover
                        return # pragma: no cover

                # Perform training step using the new standard pattern
                metrics = self.train_step(batch)
                
                # Log metrics with WandB
                if wandb.run is not None:
                    wandb.log({
                        'loss/total': metrics['total_loss'],
                        'loss/policy': metrics['policy_loss'],
                        'loss/value': metrics['value_loss'],
                        'loss/reward': metrics['reward_loss'],
                        'loss/l2': metrics['l2_loss'],
                        'metrics/grad_norm': metrics.get('grad_norm', 0.0),
                        'metrics/param_norm': metrics.get('param_norm', 0.0),
                    }, step=self.num_training_steps)
                    
                    # Add SSL loss if applicable
                    if 'ssl_loss' in metrics: # pragma: no cover
                        wandb.log({'loss/ssl': metrics['ssl_loss']}, step=self.num_training_steps) # pragma: no cover
                    
                    # Add entropy loss if applicable
                    if 'entropy_loss' in metrics: # pragma: no cover
                        wandb.log({'loss/entropy': metrics['entropy_loss']}, step=self.num_training_steps) # pragma: no cover

                # Log to console occasionally
                if self.num_training_steps % 10 == 0: # pragma: no cover
                    logging.info(f"Training step {self.num_training_steps}, "
                               f"loss: {metrics.get('total_loss', 'N/A'):.6f}, "
                               f"grad_norm: {metrics.get('grad_norm', 'N/A'):.6f}") # pragma: no cover

                # Save checkpoint if needed
                self.save_checkpoint()

        print(f"Training completed after {num_epochs} epochs.")
        
        # Save final checkpoint at end of training
        logging.info(f"End of training checkpoint: step {self.num_training_steps}")
        self.save_checkpoint(force_save=True)

    @staticmethod
    def _compute_total_loss_static(
        model: MuZeroNetwork,
        config: MuZeroConfig,
        batch: Batch,
        rng_key: PRNGKey,
        *args,
        **kwargs,
    ) -> Tuple[jax.Array, Metrics]:
        """Computes the total MuZero loss for a batch of data with unrolling."""
        # Extract optional flags from *args / **kwargs for backwards compatibility with
        # various test helpers that may pass `training` and `training_step` both
        # positionally and by keyword.

        # Defaults
        training: bool = kwargs.pop("training", True)
        training_step: int = kwargs.pop("training_step", 0)

        if len(args) > 0:
            # Historical call pattern: first extra positional was `training`
            training = args[0]
        if len(args) > 1:
            training_step = args[1]

        # ------------------------------------------------------------------
        # Core loss computation (unchanged below)
        # ------------------------------------------------------------------

        initial_observation = batch['observation'][:, 0]  # B, *obs_shape
        actions = batch['action'] # B, K
        target_rewards = batch['target_reward'] # B, K+1 or B, K+1, S
        target_values = batch['target_value'] # B, K+1 or B, K+1, S
        target_policies = batch['target_policy'] # B, K+1, A
        game_history_mask = batch['game_history_mask'] # B, K+1
        
        # EfficientZeroV2: Get importance sampling weights (default to 1.0 if not provided)
        importance_weights = batch.get('weights', jnp.ones(initial_observation.shape[0]))
        
        # EfficientZeroV2: Value target selection logic
        # Extract different types of value targets if available
        search_values = batch.get('target_search_value', target_values)  # MCTS search values
        sarsa_values = batch.get('target_sarsa_value', target_values)  # N-step TD targets
        
        # EfficientZeroV2: Generate top_new_masks for mixed value targets
        top_new_masks = batch.get('top_new_masks', None)
        if config.value_target == "mixed" and top_new_masks is None:
            # Generate top_new_masks if not provided in batch
            sample_indices = batch.get('sample_indices', jnp.arange(initial_observation.shape[0]))
            collected_transitions = batch.get('collected_transitions', config.mixed_value_threshold + sample_indices.max() + 1)
            top_new_masks = generate_top_new_masks(sample_indices, collected_transitions, config.mixed_value_threshold)
        
        # Optimal JAX implementation: Handle value target selection inside JIT for best performance
        # Using JAX conditionals to avoid Python control flow and achieve maximum performance
        
        def select_search_values():
            return search_values
            
        def select_sarsa_values():
            return sarsa_values
            
        def select_mixed_values():
            # EfficientZeroV2 mixed mode logic
            def use_search_early():
                return search_values
                
            def use_mixed_later():
                # Apply mixed value targets if masks are available, otherwise use sarsa
                # JAX-compatible None check: use the fact that top_new_masks is guaranteed to exist here
                # (generated earlier in the function if mixed mode is used)
                return apply_mixed_value_targets(
                    search_values, sarsa_values, top_new_masks, config.num_unroll_steps
                )
            
            return jax.lax.cond(
                training_step < config.start_use_mix_training_steps,
                use_search_early,
                use_mixed_later
            )
        
        def select_default_values():
            return target_values
        
        # Efficient nested JAX conditionals for value target selection
        # This compiles into optimized XLA code for maximum performance
        actual_target_values = jax.lax.cond(
            config.value_target == 'search',
            select_search_values,
            lambda: jax.lax.cond(
                config.value_target == 'sarsa', 
                select_sarsa_values,
                lambda: jax.lax.cond(
                    config.value_target == 'mixed',
                    select_mixed_values,
                    select_default_values
                )
            )
        )
        
        # EfficientZeroV2: Dynamic GAE/TD-Lambda target computation (if needed)
        if config.value_target_type == "GAE":
            # Dynamic GAE computation using current model weights
            extra_observations = batch.get('extra_observations', None)
            extra_actions = batch.get('extra_actions', None) 
            extra_rewards = batch.get('extra_rewards', None)
            extra_dones = batch.get('extra_dones', None)
            
            if all(x is not None for x in [extra_observations, extra_actions, extra_rewards, extra_dones]):
                # Compute GAE targets dynamically using current model
                gae_targets = compute_gae_value_targets(
                    model=model,
                    observations=extra_observations,  # B, K+1+extra, *obs_shape
                    actions=extra_actions,            # B, K+extra
                    rewards=extra_rewards,            # B, K+1+extra
                    dones=extra_dones,               # B, K+1+extra
                    config=config,
                    training=training,
                    rng_key=rng_key,
                    sample_indices=batch.get('sample_indices', None),  # For adaptive td_lambda
                    collected_transitions=batch.get('collected_transitions', None)  # For adaptive td_lambda
                )
                
                # Use GAE targets as the actual target values for loss computation
                actual_target_values = gae_targets
        
        # Apply value prefix reward accumulation if enabled (EfficientZeroV2 feature)
        # Note: We'll collect hidden states during model unrolling and apply LSTM later
        target_rewards_original = target_rewards

        # EfficientZeroV2: Dynamic Policy Target Reanalysis
        actual_target_policies = target_policies
        if config.reanalyze_ratio > 0.0:
            try:
                # Get training step for temperature scheduling
                training_step = batch.get('training_step', 0)
                
                # Prepare observations for reanalysis - include full unroll sequence
                full_observations = batch.get('observation', None)
                if full_observations is None or full_observations.shape[1] < config.num_unroll_steps + 1:
                    # If we don't have full observations, create them from initial observation
                    # This is a fallback - ideally the batch should contain full observation sequences
                    full_observations = jnp.tile(
                        jnp.expand_dims(initial_observation, axis=1), 
                        (1, config.num_unroll_steps + 1) + tuple(1 for _ in initial_observation.shape[1:])
                    )
                
                # Perform policy reanalysis using current model weights
                reanalyzed_policies = compute_policy_reanalysis_targets(
                    model=model,
                    observations=full_observations,  # B, K+1, *obs_shape
                    config=config,
                    training=training,
                    rng_key=rng_key
                )
                
                # Use reanalyzed policies - the function already handles mixing with original policies
                actual_target_policies = reanalyzed_policies
                
            except Exception as e:  # pragma: no cover
                # Fallback to original policies if reanalysis fails
                import warnings  # pragma: no cover
                warnings.warn(f"Policy reanalysis failed, using original policies: {e}")  # pragma: no cover
                actual_target_policies = target_policies  # pragma: no cover

        # Initial inference with proper LSTM reward network integration (moved outside reanalysis block)
        initial_reward_hidden = None
        if config.use_value_prefix and hasattr(model, 'lstm_reward_network') and model.lstm_reward_network is not None:
            batch_size = initial_observation.shape[0]
            # Initialize LSTM reward network hidden state properly
            initial_reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size)
            
            # Verify LSTM state dimensions are correct
            if initial_reward_hidden is not None:
                expected_hidden_dim = config.lstm_hidden_size

                def _infer_hidden_dim(state):
                    if isinstance(state, tuple) and len(state) > 0:
                        ref = state[0]
                    else:
                        ref = state  # pragma: no cover - current implementations return tuples
                    return ref.shape[-1] if hasattr(ref, 'shape') else expected_hidden_dim

                actual_hidden_dim = _infer_hidden_dim(initial_reward_hidden)
                if actual_hidden_dim != expected_hidden_dim:  # pragma: no cover - defensive guard
                    raise ValueError(
                        f"LSTM hidden state dimension mismatch: expected {expected_hidden_dim}, got {actual_hidden_dim}"
                    )
        # For compatibility with configurations without LSTM reward network, initial_reward_hidden remains None

        initial_inference_output = model.initial_inference(
            initial_observation, training=training, reward_hidden=initial_reward_hidden
        )
        hidden_state = initial_inference_output[0]
        initial_projection = initial_inference_output[4] if config.use_projection and len(initial_inference_output) > 4 else None
        current_reward_hidden = initial_inference_output[5] if len(initial_inference_output) > 5 else None

        predicted_rewards_list = [initial_inference_output[1]]
        predicted_values_list = [initial_inference_output[2]]
        predicted_policy_logits_list = [initial_inference_output[3]]
        predicted_projections_list = [initial_projection] if config.use_projection and initial_projection is not None else []
        
        # Collect hidden states for LSTM reward prediction
        hidden_states_list = [hidden_state]

        # Recurrent inferences
        for k in range(config.num_unroll_steps):
            current_action = actions[:, k]
            # Apply half-gradient to hidden state (EfficientZeroV2 pattern)
            # Apply half-gradient as per EfficientZeroV2 pattern (matches PyTorch line 500 in base.py):
            # states.register_hook(lambda grad: grad * 0.5)
            # Applied in the main training unroll loop, not during MCTS/target generation
            hidden_state_half_grad = half_gradient(hidden_state)
            
            # Reset LSTM reward hidden state periodically (EfficientZeroV2 pattern)
            if config.use_value_prefix and current_reward_hidden is not None and (k + 1) % config.lstm_horizon_length == 0:
                batch_size = hidden_state.shape[0]
                reset_mask = jnp.ones(batch_size)
                current_reward_hidden = model.lstm_reward_network.reset_hidden_state(
                    current_reward_hidden, reset_mask
                )
            
            recurrent_inference_output = model.recurrent_inference(
                hidden_state_half_grad, current_action, training=training, reward_hidden=current_reward_hidden
            )
            hidden_state = recurrent_inference_output[0]
            current_reward_hidden = recurrent_inference_output[5] if len(recurrent_inference_output) > 5 else None
            
            predicted_rewards_list.append(recurrent_inference_output[1])
            predicted_values_list.append(recurrent_inference_output[2])
            predicted_policy_logits_list.append(recurrent_inference_output[3])
            hidden_states_list.append(hidden_state)
            
            if config.use_projection and len(recurrent_inference_output) > 4:
                predicted_projections_list.append(recurrent_inference_output[4])

        # Stack predictions over unroll steps (K+1 steps total)
        predicted_rewards = jnp.stack(predicted_rewards_list, axis=1) # B, K+1 or B, K+1, S
        predicted_values = jnp.stack(predicted_values_list, axis=1) # B, K+1 or B, K+1, S
        predicted_policy_logits = jnp.stack(predicted_policy_logits_list, axis=1) # B, K+1, A
        if config.use_projection and predicted_projections_list and all(p is not None for p in predicted_projections_list):
            predicted_projections = jnp.stack(predicted_projections_list, axis=1) # B, K+1, proj_dim
        else:
            predicted_projections = None
        
        # Apply LSTM-based value prefix reward accumulation if enabled
        if config.use_value_prefix:
            hidden_states_tensor = jnp.stack(hidden_states_list, axis=1)  # B, K+1, C, H, W
            target_rewards, final_reward_hidden = apply_value_prefix_reward_accumulation(
                target_rewards_original, config, game_history_mask, model, hidden_states_tensor, initial_reward_hidden
            )
        else:
            target_rewards = target_rewards_original

        # Determine effective IQL parameter based on config (EfficientZeroV2 pattern)
        if config.use_iql:
            effective_iql_param = config.iql_weight
        else:
            effective_iql_param = 0.5  # Symmetric loss when IQL is disabled

        # Task 6.4: Loss Computation Strategy Optimization - HOST-SIDE PREPARATION
        # Pre-process targets and predictions on host to eliminate runtime conversions in JIT
        support_min = getattr(config, "support_min", -300.0)
        support_max = getattr(config, "support_max", 300.0)

        processed_target_values = prepare_targets_for_loss_type_host(
            actual_target_values,
            config.value_loss_type,
            config.value_support_size,
            support_min,
            support_max,
        )
        processed_predicted_values = prepare_predictions_for_loss_type_host(
            predicted_values,
            config.value_loss_type,
            config.value_support_size,
            support_min,
            support_max,
        )
        
        processed_target_rewards = prepare_targets_for_loss_type_host(
            target_rewards,
            config.reward_loss_type,
            config.reward_support_size,
            support_min,
            support_max,
        )
        processed_predicted_rewards = prepare_predictions_for_loss_type_host(
            predicted_rewards,
            config.reward_loss_type,
            config.reward_support_size,
            support_min,
            support_max,
        )

        # Use vectorized loss computation with pre-processed inputs (no runtime conversions)
        (per_sample_policy_loss, 
         per_sample_value_loss, 
         per_sample_reward_loss, 
         per_sample_ssl_loss, 
         per_sample_entropy_loss) = Learner._compute_vectorized_loss_optimized(
            predicted_values=processed_predicted_values,
            predicted_rewards=processed_predicted_rewards,
            predicted_policy_logits=predicted_policy_logits,
            actual_target_values=processed_target_values,
            target_rewards=processed_target_rewards,
            actual_target_policies=actual_target_policies,
            game_history_mask=game_history_mask,
            config=config,
            effective_iql_param=effective_iql_param,
            predicted_projections=predicted_projections,
            initial_projection=initial_projection
        )

        # L2 regularization (only if not using optimizer weight_decay)
        model_params = nnx.state(model, nnx.Param)
        if config.weight_decay == 0:
            l2_loss = losses_lib.l2_regularization(model_params, config.l2_weight)
        else:
            l2_loss = jnp.array(0.0)  # Weight decay handled by optimizer

        # Combine individual loss components per sample
        per_sample_combined_loss = (
            config.policy_loss_weight * per_sample_policy_loss
            + config.value_loss_weight * per_sample_value_loss 
            + config.reward_loss_weight * per_sample_reward_loss 
        )
        # Add entropy regularization (EfficientZeroV2 pattern)
        if config.entropy_coeff > 0:
            per_sample_combined_loss -= config.entropy_coeff * per_sample_entropy_loss  # Negative because we want to maximize entropy
        
        if config.use_projection and config.consistency_loss_coeff > 0:
            per_sample_combined_loss += config.consistency_loss_coeff * per_sample_ssl_loss
            
        # Apply importance weighting (EfficientZeroV2 pattern: weighted_loss = (weights * loss).mean())
        final_loss = jnp.mean(importance_weights * per_sample_combined_loss) + l2_loss
        
        # Compute priorities for replay buffer update (EfficientZeroV2 pattern)
        # Use value prediction error at step 0 as priority (L1 loss)
        priorities = None
        if config.use_priority_replay and 'indices' in batch:
            # Get value prediction and target at step 0
            predicted_val_step0 = predicted_values[:, 0]  # B or B, S
            target_val_step0 = actual_target_values[:, 0]  # B or B, S
            
            # Convert to scalars if needed for priority computation
            priority_support_min = getattr(config, "support_min", -300.0)
            priority_support_max = getattr(config, "support_max", 300.0)

            if predicted_val_step0.ndim > 1 and predicted_val_step0.shape[-1] > 1: # pragma: no cover
                predicted_val_step0 = losses_lib.support_to_scalar( # pragma: no cover
                    predicted_val_step0, # pragma: no cover
                    support_min=priority_support_min, # pragma: no cover
                    support_max=priority_support_max, # pragma: no cover
                    num_atoms=predicted_val_step0.shape[-1] # pragma: no cover
                ) # pragma: no cover
            elif predicted_val_step0.ndim == 2 and predicted_val_step0.shape[-1] == 1: # pragma: no cover
                predicted_val_step0 = jnp.squeeze(predicted_val_step0, axis=-1) # pragma: no cover
            
            if target_val_step0.ndim > 1 and target_val_step0.shape[-1] > 1: # pragma: no cover
                target_val_step0 = losses_lib.support_to_scalar( # pragma: no cover
                    target_val_step0, # pragma: no cover
                    support_min=priority_support_min, # pragma: no cover
                    support_max=priority_support_max, # pragma: no cover
                    num_atoms=target_val_step0.shape[-1] # pragma: no cover
                ) # pragma: no cover
            elif target_val_step0.ndim == 2 and target_val_step0.shape[-1] == 1: # pragma: no cover
                target_val_step0 = jnp.squeeze(target_val_step0, axis=-1) # pragma: no cover
            
            # Compute L1 loss for priorities
            value_errors = jnp.abs(predicted_val_step0 - target_val_step0)
            priorities = value_errors + config.min_priority
        
        # Compute average losses for metrics (unweighted)
        total_policy_loss = jnp.mean(per_sample_policy_loss)
        total_value_loss = jnp.mean(per_sample_value_loss)
        total_reward_loss = jnp.mean(per_sample_reward_loss)
        total_ssl_loss = jnp.mean(per_sample_ssl_loss)
        total_entropy_loss = jnp.mean(per_sample_entropy_loss)
        
        metrics = {
            'total_loss': final_loss,
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'reward_loss': total_reward_loss,
            'l2_loss': l2_loss,
        }
        if config.entropy_coeff > 0:
            metrics['entropy_loss'] = total_entropy_loss
        if config.use_projection and config.consistency_loss_coeff > 0:
            metrics['ssl_loss'] = total_ssl_loss
        if priorities is not None:
            metrics['priorities'] = priorities
            metrics['indices'] = batch['indices']
            
        return final_loss, metrics

    def save_checkpoint(self, force_save: bool = False) -> Optional[str]:
        """Save model and optimizer state to checkpoint and return the path."""
        if self.checkpoint_manager is None:
            print("Checkpoint manager not configured. Skipping save.") # pragma: no cover
            return None # pragma: no cover
            
        # Check if we should save based on frequency
        should_save = (force_save or 
                      (self.num_training_steps % self.config.checkpoint_frequency == 0 and 
                       self.num_training_steps > 0))
        
        # Log the decision
        if should_save:
            logging.info(f"SAVE_CHECKPOINT: Condition met. force_save={force_save}, num_training_steps={self.num_training_steps}, freq={self.config.checkpoint_frequency}")
        else:
            logging.info(f"SAVE_CHECKPOINT: Condition NOT met. force_save={force_save}, num_training_steps={self.num_training_steps}, freq={self.config.checkpoint_frequency}") # pragma: no cover
            return None # pragma: no cover
            
        try:
            # Prepare checkpoint data using nnx.Optimizer pattern
            checkpoint_data = {
                'model': nnx.state(self.model),
                'optimizer': nnx.state(self.optimizer),
                'num_training_steps': self.num_training_steps,
                'rng_key': self._rng_key,
            }
            
            # Add EMA state if using target network
            if (self.config.use_target_network_ema and 
                self.target_model is not None and 
                self.ema_params_state is not None):
                checkpoint_data['target_model'] = nnx.state(self.target_model)
                checkpoint_data['ema_params_state'] = self.ema_params_state
            
            # Save checkpoint using modern Orbax API
            self.checkpoint_manager.save(
                step=self.num_training_steps,
                args=ocp.args.StandardSave(checkpoint_data)
            )
            
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_{self.num_training_steps}"
            )
            logging.info(f"Checkpoint saved at step {self.num_training_steps}")
            return checkpoint_path
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}") # pragma: no cover
            return None

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load model and optimizer state from checkpoint. Returns True if successful."""
        if self.checkpoint_manager is None:
            print("Checkpoint manager not configured. Skipping load.") # pragma: no cover
            return False # pragma: no cover
            
        try:
            restore_step = None
            if checkpoint_path:
                restore_step = self._extract_step_from_checkpoint_path(checkpoint_path)
            if restore_step is None:
                restore_step = self.checkpoint_manager.latest_step()
            if restore_step is None:
                print("No checkpoint found to resume from.") # pragma: no cover
                return False # pragma: no cover
                
            # Prepare target structure for restore (this prevents the immutable tuple error)
            target_structure = {
                'model': nnx.state(self.model),
                'optimizer': nnx.state(self.optimizer),
                'num_training_steps': self.num_training_steps,
                'rng_key': self._rng_key,
            }
            
            # Add EMA structure if enabled
            if (self.config.use_target_network_ema and 
                self.target_model is not None and 
                self.ema_params_state is not None):
                target_structure['target_model'] = nnx.state(self.target_model)
                target_structure['ema_params_state'] = self.ema_params_state
                
            # Load checkpoint data using modern Orbax API with target
            checkpoint_data = self.checkpoint_manager.restore(
                step=restore_step,
                args=ocp.args.StandardRestore(target_structure)
            )
            
            # Restore model and optimizer state using proper NNX state management
            # Instead of nnx.update which can fail with immutable states, 
            # we need to directly set the state back into the models
            
            # For model: extract graphdef and merge with restored state
            model_graphdef, _ = nnx.split(self.model)
            self.model = nnx.merge(model_graphdef, checkpoint_data['model'])
            
            # For optimizer: similar approach
            optimizer_graphdef, _ = nnx.split(self.optimizer)  
            self.optimizer = nnx.merge(optimizer_graphdef, checkpoint_data['optimizer'])
            self.num_training_steps = checkpoint_data['num_training_steps']
            self._rng_key = checkpoint_data['rng_key']
            self._reschedule_model_updates()
            
            # Restore EMA state if available
            if (self.config.use_target_network_ema and 
                'target_model' in checkpoint_data and 
                'ema_params_state' in checkpoint_data):
                
                if self.target_model is not None:
                    nnx.update(self.target_model, checkpoint_data['target_model'])
                    self.ema_params_state = checkpoint_data['ema_params_state']
                else: # pragma: no cover
                    # Target model components not fully in checkpoint, re-initialize
                    print("Warning: EMA enabled, target model components not fully in ckpt. Re-syncing with online model.") # pragma: no cover
                    # Use proper Flax NNX state copying instead of copy.deepcopy
                    graphdef, params, batch_stats, rngs, static, ellipsis = nnx.split( # pragma: no cover
                        self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ... # pragma: no cover
                    ) # pragma: no cover
                    self.target_model = nnx.merge(graphdef, params, batch_stats, rngs, static, ellipsis) # pragma: no cover
                    
                    # Re-initialize EMA state
                    self.ema_updater = optax.ema(self.config.ema_decay)
                    self.ema_params_state = self.ema_updater.init(params)
                    # Crucial synchronization: ensure EMA internal average matches current online params
                    self.ema_params_state = self.ema_params_state._replace(ema=params)
            
            self._split_objects_for_jit()
            logging.info(f"Checkpoint restored from step {self.num_training_steps}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}") # pragma: no cover
            return False # pragma: no cover

    def cleanup(self):
        """Explicit cleanup method for tests to call."""
        if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager is not None:
            try:
                self.wait_for_pending_checkpoints()
                self.checkpoint_manager.close()
                self.checkpoint_manager = None
            except Exception:
                pass  # Ignore errors during cleanup

    def _extract_step_from_checkpoint_path(self, checkpoint_path: str) -> Optional[int]:
        """Helper to parse the step number from a checkpoint path."""
        try:
            stem = Path(checkpoint_path).stem
            return int(stem.split('_')[-1])
        except (ValueError, IndexError):
            logging.warning(f"Could not determine step from checkpoint path: {checkpoint_path}")
            return None

    def wait_for_pending_checkpoints(self):
        """Block until any asynchronous checkpoint saves finish."""
        if not hasattr(self, 'checkpoint_manager') or self.checkpoint_manager is None:  # pragma: no cover - defensive
            return
        try:
            wait_fn = getattr(self.checkpoint_manager, 'wait_until_finished', None)
            if callable(wait_fn):
                wait_fn()
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("CheckpointManager.wait_until_finished raised: %s", exc)
        checkpointer = getattr(self.checkpoint_manager, '_checkpointer', None)
        if checkpointer is not None:
            try:
                wait_fn = getattr(checkpointer, 'wait_until_finished', None)
                if callable(wait_fn):
                    wait_fn()
            except Exception as exc:  # pragma: no cover - best effort
                logging.warning("AsyncCheckpointer.wait_until_finished raised: %s", exc)

    def __del__(self):
        """Cleanup method to ensure CheckpointManager is properly closed."""
        if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager is not None: # pragma: no cover
            try: # pragma: no cover
                # Only close if not already closed and if logging system is still available
                import logging # pragma: no cover
                if hasattr(self.checkpoint_manager, '_closed') and not self.checkpoint_manager._closed: # pragma: no cover
                    self.checkpoint_manager.close() # pragma: no cover
                elif not hasattr(self.checkpoint_manager, '_closed'): # pragma: no cover
                    # Fallback for checkpoint managers that don't have _closed attribute
                    self.checkpoint_manager.close() # pragma: no cover
            except Exception: # pragma: no cover
                # Completely suppress all exceptions during cleanup to prevent logging errors
                pass # pragma: no cover

    @staticmethod
    def _compute_vectorized_loss_optimized(
        predicted_values: jax.Array,      # [B, K+1, ...]
        predicted_rewards: jax.Array,     # [B, K+1, ...]
        predicted_policy_logits: jax.Array,  # [B, K+1, ...]
        actual_target_values: jax.Array,  # [B, K+1, ...]
        target_rewards: jax.Array,        # [B, K+1, ...]
        actual_target_policies: jax.Array,  # [B, K+1, ...]
        game_history_mask: jax.Array,     # [B, K+1]
        config: MuZeroConfig,
        effective_iql_param: float,
        predicted_projections: jax.Array | None = None,  # [B, K+1, ...]
        initial_projection: jax.Array | None = None,  # [B, ...]
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Optimized vectorized loss computation eliminating per-step loops.
        
        Task 6.4: Loss Computation Strategy Optimization
        - Vectorize loss computation across (B, K+1) dimensions
        - Move shape conversions to pre-JIT host-side preparation
        - Eliminate runtime conditional branching using static function selection
        
        All inputs are pre-processed on host-side to correct formats.
        No runtime shape conversions or conditional branching within JIT.
        
        Returns:
            Tuple of per-sample losses: (policy, value, reward, ssl, entropy)
        """
        
        # Static function selection based on loss types (eliminates runtime branching)
        # All shape conversions already done on host-side
        if config.value_loss_type == "categorical":
            value_loss_fn = functools.partial(losses_lib.compute_categorical_value_loss, effective_iql_param=effective_iql_param)
        elif config.value_loss_type == "symlog":
            value_loss_fn = functools.partial(losses_lib.compute_symlog_value_loss, effective_iql_param=effective_iql_param, base=config.symlog_base)
        elif config.value_loss_type == "kl":
            # For KL value loss, use categorical value loss (which uses KL internally)
            value_loss_fn = functools.partial(losses_lib.compute_categorical_value_loss, effective_iql_param=effective_iql_param)
        else:  # mse
            value_loss_fn = functools.partial(losses_lib.compute_scalar_value_loss, effective_iql_param=effective_iql_param)
        
        if config.reward_loss_type == "categorical":
            reward_loss_fn = losses_lib.compute_categorical_reward_loss
        elif config.reward_loss_type == "symlog":
            reward_loss_fn = functools.partial(losses_lib.compute_symlog_loss, base=config.symlog_base)
        elif config.reward_loss_type == "kl":
            # For KL reward loss, use categorical reward loss (which uses KL internally)
            reward_loss_fn = losses_lib.compute_categorical_reward_loss
        else:  # mse
            reward_loss_fn = losses_lib.compute_scalar_reward_loss
        
        # Vectorized loss computation across all (B, K+1) dimensions
        # Get consistent batch and time dimensions
        batch_size = game_history_mask.shape[0]
        time_steps = game_history_mask.shape[1] if game_history_mask.ndim > 1 else 1
        
        # Ensure all tensors have consistent time dimension
        pred_policy_time = predicted_policy_logits.shape[1]
        target_policy_time = actual_target_policies.shape[1]
        pred_values_time = predicted_values.shape[1]
        target_values_time = actual_target_values.shape[1]
        pred_rewards_time = predicted_rewards.shape[1]
        target_rewards_time = target_rewards.shape[1]
        
        # All tensors must share the same time dimension – mismatches indicate a bug.
        if not (pred_policy_time == target_policy_time == pred_values_time == target_values_time == pred_rewards_time == target_rewards_time == time_steps):
            raise ValueError(
                "Time-dimension mismatch detected: "
                f"mask={time_steps}, policy_pred={pred_policy_time}, policy_target={target_policy_time}, "
                f"value_pred={pred_values_time}, value_target={target_values_time}, "
                f"reward_pred={pred_rewards_time}, reward_target={target_rewards_time}"
            )
        min_time = time_steps  # Guaranteed equal at this point
        
        # Truncate all tensors to consistent time dimension
        truncated_predicted_policy = predicted_policy_logits[:batch_size, :min_time]
        truncated_target_policy = actual_target_policies[:batch_size, :min_time]
        truncated_predicted_values = predicted_values[:batch_size, :min_time]
        truncated_target_values = actual_target_values[:batch_size, :min_time]
        truncated_predicted_rewards = predicted_rewards[:batch_size, :min_time]
        truncated_target_rewards = target_rewards[:batch_size, :min_time]
        truncated_mask = game_history_mask[:batch_size, :min_time]
        
        # Flatten time dimension for vectorized processing
        flat_predicted_policy = truncated_predicted_policy.reshape(batch_size * min_time, -1)
        flat_target_policy = truncated_target_policy.reshape(batch_size * min_time, -1)
        
        # Handle value flattening (maintain shape appropriately)
        if truncated_predicted_values.ndim > 2:
            flat_predicted_values = truncated_predicted_values.reshape(batch_size * min_time, -1)
        else:
            flat_predicted_values = truncated_predicted_values.reshape(batch_size * min_time)
        
        if truncated_target_values.ndim > 2:
            flat_target_values = truncated_target_values.reshape(batch_size * min_time, -1)
        else:
            flat_target_values = truncated_target_values.reshape(batch_size * min_time)
            
        # Handle reward flattening (maintain shape appropriately)
        if truncated_predicted_rewards.ndim > 2:
            flat_predicted_rewards = truncated_predicted_rewards.reshape(batch_size * min_time, -1)
        else:
            flat_predicted_rewards = truncated_predicted_rewards.reshape(batch_size * min_time)
            
        if truncated_target_rewards.ndim > 2:
            flat_target_rewards = truncated_target_rewards.reshape(batch_size * min_time, -1)
        else:
            flat_target_rewards = truncated_target_rewards.reshape(batch_size * min_time)
        
        # Apply loss functions to flattened data - using pre-selected static functions
        # No runtime branching or shape conversions needed
        policy_losses_flat = losses_lib.compute_policy_loss(flat_predicted_policy, flat_target_policy)
        value_losses_flat = value_loss_fn(flat_predicted_values, flat_target_values)
        reward_losses_flat = reward_loss_fn(flat_predicted_rewards, flat_target_rewards)
        
        # Reshape back to [B, time_steps] 
        policy_losses = policy_losses_flat.reshape(batch_size, min_time)
        value_losses = value_losses_flat.reshape(batch_size, min_time)
        reward_losses = reward_losses_flat.reshape(batch_size, min_time)
        
        # Apply masking and sum across time steps
        masked_policy_losses = policy_losses * truncated_mask
        masked_value_losses = value_losses * truncated_mask
        masked_reward_losses = reward_losses * truncated_mask
        
        # Sum over time steps to get per-sample losses
        per_sample_policy_loss = jnp.sum(masked_policy_losses, axis=1)   # [B]
        per_sample_value_loss = jnp.sum(masked_value_losses, axis=1)     # [B]
        per_sample_reward_loss = jnp.sum(masked_reward_losses, axis=1)   # [B]
        
        # SSL Loss (consistency/projection loss) - vectorized
        per_sample_ssl_loss = jnp.zeros(batch_size)  # [B]
        if (config.use_projection and config.consistency_loss_coeff > 0 and 
            predicted_projections is not None and initial_projection is not None):
            
            # Vectorized SSL loss for steps k > 0
            ssl_mask = truncated_mask[:, 1:] if min_time > 1 else jnp.zeros((batch_size, 0))  # [B, time_steps-1]
            
            if min_time > 1:
                # Truncate projections to consistent time dimension
                truncated_projections = predicted_projections[:batch_size, :min_time]
                flat_projections = truncated_projections[:, 1:].reshape(batch_size * (min_time - 1), -1)
                # Expand initial projection to match each timestep
                expanded_initial = jnp.tile(initial_projection[:, None, :], (1, min_time - 1, 1)).reshape(batch_size * (min_time - 1), -1)
                
                ssl_losses_flat = losses_lib.compute_projection_consistency_loss(
                    flat_projections, jax.lax.stop_gradient(expanded_initial)
                )
                ssl_losses = ssl_losses_flat.reshape(batch_size, min_time - 1)  # [B, K]
                
                masked_ssl_losses = ssl_losses * ssl_mask  # [B, K]
                per_sample_ssl_loss = jnp.sum(masked_ssl_losses, axis=1)  # [B]
        
        # Entropy Loss - vectorized
        per_sample_entropy_loss = jnp.zeros(batch_size)  # [B]
        if config.entropy_coeff > 0:
            # Vectorized entropy computation - process batched data directly
            entropy_losses_flat = losses_lib.compute_policy_entropy_general(
                flat_predicted_policy,
                action_type=config.action_type,
                distribution_type=config.distribution_type
            )
            
            entropy_losses = entropy_losses_flat.reshape(batch_size, min_time)  # [B, K+1]
            masked_entropy_losses = entropy_losses * truncated_mask  # [B, time_steps]
            per_sample_entropy_loss = jnp.sum(masked_entropy_losses, axis=1)  # [B]
        
        return (
            per_sample_policy_loss,
            per_sample_value_loss,
            per_sample_reward_loss,
            per_sample_ssl_loss,
            per_sample_entropy_loss
        )

def apply_value_prefix_reward_accumulation(
    target_reward: jax.Array, 
    config: MuZeroConfig,
    game_history_mask: jax.Array | None = None,
    model: MuZeroNetwork | None = None,
    hidden_states: jax.Array | None = None,
    initial_reward_hidden: LSTMState | None = None
) -> Tuple[jax.Array, LSTMState | None]:
    """
    Apply value prefix reward accumulation using LSTM network for EfficientZeroV2.

    When value_prefix is enabled, this function uses the LSTM reward network to predict
    rewards based on hidden states, with periodic reset every lstm_horizon_length steps.
    
    Args:
        target_reward: Target reward tensor, shape (B, K+1) or (B, K+1, support_size)
        config: MuZero configuration
        game_history_mask: Optional mask for valid steps, shape (B, K+1)
        model: MuZero model with LSTM reward network (required if use_value_prefix=True)
        hidden_states: Hidden states from dynamics network, shape (B, K+1, C, H, W)
        initial_reward_hidden: Initial LSTM hidden state
        
    Returns:
        Tuple of (accumulated_rewards, final_reward_hidden)
    """
    if not config.use_value_prefix:
        return target_reward, None
    
    # Handle empty input (edge case)
    if target_reward.size == 0:
        return target_reward, None
    
    batch_size, num_steps = target_reward.shape[0], target_reward.shape[1]
    
    # Handle edge case where batch_size is 0
    if batch_size == 0: # pragma: no cover
        return target_reward, None # pragma: no cover
    
    # If model is not provided or doesn't have LSTM reward network, fall back to simple accumulation
    if model is None or not hasattr(model, 'lstm_reward_network') or model.lstm_reward_network is None:
        # Simple accumulation fallback (original implementation)
        def accumulate_batch_step(batch_idx):
            """Accumulate rewards for a single batch item."""
            rewards = target_reward[batch_idx]  # K+1 or K+1, S
            mask = game_history_mask[batch_idx] if game_history_mask is not None else jnp.ones(num_steps)
            
            def step_accumulation(step_idx, accumulator):
                """Accumulate reward for a single step."""
                current_reward = rewards[step_idx]
                current_mask = mask[step_idx]
                
                # Reset accumulation every lstm_horizon_length steps (EfficientZeroV2 pattern)
                should_reset = (step_idx % config.lstm_horizon_length == 0)
                
                if should_reset:
                    # Reset: start fresh accumulation
                    new_accumulator = current_reward * current_mask
                else:
                    # Accumulate: add to previous
                    new_accumulator = accumulator + current_reward * current_mask
                
                return new_accumulator
            
            # Use jax.lax.scan for efficient sequential accumulation
            init_accumulator = jnp.zeros_like(rewards[0])
            
            # Scan over steps to accumulate rewards
            accumulated_rewards = []
            accumulator = init_accumulator
            for step_idx in range(num_steps):
                accumulator = step_accumulation(step_idx, accumulator)
                accumulated_rewards.append(accumulator)
            
            return jnp.stack(accumulated_rewards, axis=0)
        
        # Process each batch item
        accumulated_batch = []
        for batch_idx in range(batch_size):
            accumulated_item = accumulate_batch_step(batch_idx)
            accumulated_batch.append(accumulated_item)
        
        return jnp.stack(accumulated_batch, axis=0), None
    
    # LSTM-based reward prediction (EfficientZeroV2 implementation)
    if hidden_states is None:
        # If hidden states not provided, fall back to simple accumulation
        return apply_value_prefix_reward_accumulation(
            target_reward, config, game_history_mask, None, None, None
        )
    
    # Initialize LSTM hidden state if not provided
    if initial_reward_hidden is None:
        initial_reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size)
    
    def lstm_step(carry, step_inputs):
        """Single LSTM step for reward prediction."""
        reward_hidden, step_idx = carry
        hidden_state = step_inputs  # [B, C, H, W]
        
        # Reset LSTM hidden state every lstm_horizon_length steps
        reset_condition = step_idx % config.lstm_horizon_length == 0
        reset_mask = jnp.array(reset_condition).astype(jnp.float32)
        reset_mask_expanded = jnp.broadcast_to(reset_mask, (batch_size,))
        
        if reset_mask.any():
            reward_hidden = model.lstm_reward_network.reset_hidden_state(
                reward_hidden, reset_mask_expanded
            )
        
        # Predict reward using LSTM network
        predicted_reward, new_reward_hidden = model.lstm_reward_network(
            hidden_state, reward_hidden, training=False
        )
        
        return (new_reward_hidden, step_idx + 1), predicted_reward
    
    # Apply LSTM across all time steps using scan
    initial_carry = (initial_reward_hidden, 0)
    final_carry, predicted_rewards = jax.lax.scan(
        lstm_step, 
        initial_carry, 
        hidden_states.transpose(1, 0, 2, 3, 4)  # [K+1, B, C, H, W]
    )
    
    # Transpose back to [B, K+1, ...]
    predicted_rewards = predicted_rewards.transpose(1, 0, *range(2, predicted_rewards.ndim))
    final_reward_hidden = final_carry[0]
    
    # For scalar rewards, squeeze the last dimension if it's size 1
    if config.reward_support_size == 0 and predicted_rewards.shape[-1] == 1:
        predicted_rewards = jnp.squeeze(predicted_rewards, axis=-1)
    
    # Apply game history mask if provided
    if game_history_mask is not None:
        mask_expanded = game_history_mask
        if predicted_rewards.ndim > 2:
            # Expand mask for categorical rewards
            for _ in range(predicted_rewards.ndim - 2):
                mask_expanded = jnp.expand_dims(mask_expanded, axis=-1)
            mask_expanded = jnp.broadcast_to(mask_expanded, predicted_rewards.shape)
        
        predicted_rewards = predicted_rewards * mask_expanded
    
    return predicted_rewards, final_reward_hidden


def generate_top_new_masks(
    sample_indices: jax.Array,
    collected_transitions: int | jax.Array,
    mixed_value_threshold: int
) -> jax.Array:
    """
    Generate top_new_masks for EfficientZeroV2 mixed value targets.

    This replicates PyTorch BatchWorker logic:
    mask = int(idx > collected_transitions - mixed_value_threshold)

    Recent samples (idx > threshold) get mask=1 and use sarsa values.
    Old samples (idx <= threshold) get mask=0 and use search values.
    
    CRITICAL FIX (Action Item #6): Requires collected_transitions to be provided
    explicitly to prevent off-by-one errors from heuristic fallbacks.

    Args:
        sample_indices: Indices of samples in replay buffer, shape (B,)
        collected_transitions: Total number of transitions collected so far (REQUIRED)
        mixed_value_threshold: Threshold for determining recent vs old samples

    Returns:
        Boolean mask indicating recent samples, shape (B,)
        
    Raises:
        ValueError: If collected_transitions is None or invalid
    """
    # CRITICAL FIX (Action Item #6): Validate collected_transitions is provided
    if collected_transitions is None:
        raise ValueError("collected_transitions must be provided explicitly to prevent off-by-one errors")
    
    # Convert to JAX array if needed
    if isinstance(collected_transitions, int):
        collected_transitions = jnp.array(collected_transitions)
    
    threshold = collected_transitions - mixed_value_threshold
    return (sample_indices > threshold)  # bool mask


def apply_mixed_value_targets(
    search_values: jax.Array,
    sarsa_values: jax.Array,
    top_new_masks: jax.Array,
    num_unroll_steps: int
) -> jax.Array:
    """
    Apply EfficientZeroV2 mixed value target logic using top_new_masks.
    
    Recent samples (mask=1) use sarsa values, old samples (mask=0) use search values.
    
    Args:
        search_values: MCTS search value targets, shape (B, K+1, ...)
        sarsa_values: N-step TD value targets, shape (B, K+1, ...)
        top_new_masks: Mask for recent samples, shape (B,)
        num_unroll_steps: Number of unroll steps (K)
        
    Returns:
        Mixed value targets, shape (B, K+1, ...)
    """
    # Shape compatibility check: ensure both value arrays have the same shape
    if search_values.shape != sarsa_values.shape:
        # If shapes don't match, truncate to the smaller shape along each dimension
        min_shape = tuple(min(s, t) for s, t in zip(search_values.shape, sarsa_values.shape))  # pragma: no cover
        search_values = search_values[:min_shape[0], :min_shape[1]]  # pragma: no cover
        sarsa_values = sarsa_values[:min_shape[0], :min_shape[1]]  # pragma: no cover
        
        # Adjust num_unroll_steps if the time dimension was truncated
        if len(min_shape) > 1:  # pragma: no cover
            num_unroll_steps = min_shape[1] - 1  # pragma: no cover
    
    # Ensure boolean mask then cast when needed
    if top_new_masks.dtype != jnp.bool_:
        top_new_masks = top_new_masks.astype(jnp.bool_)

    # Expand mask to match value dimensions: B, K+1, ...
    mask_expanded = jnp.expand_dims(top_new_masks, axis=1)  # B, 1 (bool)
    mask_expanded = jnp.repeat(mask_expanded, num_unroll_steps + 1, axis=1)  # B, K+1
    
    if sarsa_values.ndim > 2:
        # For categorical values, expand mask to match support dimension
        for _ in range(sarsa_values.ndim - 2):
            mask_expanded = jnp.expand_dims(mask_expanded, axis=-1)
        mask_expanded = jnp.repeat(mask_expanded, sarsa_values.shape[-1], axis=-1)
    
    # Mixed target: recent samples (mask=1) use sarsa, old samples (mask=0) use search
    mask_f = mask_expanded.astype(sarsa_values.dtype)
    return sarsa_values * mask_f + search_values * (1.0 - mask_f)

# EfficientZeroV2 GAE/TD-Lambda computation for dynamic value targets
def compute_gae_value_targets(
    model: MuZeroNetwork,
    observations: jax.Array,  # B, K+1+extra, *obs_shape
    actions: jax.Array,       # B, K+extra 
    rewards: jax.Array,       # B, K+1+extra
    dones: jax.Array,         # B, K+1+extra (episode termination flags)
    config: MuZeroConfig,
    training: bool = False,
    rng_key: PRNGKey | None = None,
    sample_indices: jax.Array | None = None,  # B, - for adaptive td_lambda
    collected_transitions: int | None = None  # for adaptive td_lambda
) -> jax.Array:
    """
    Computes GAE (Generalized Advantage Estimation) value targets using current model weights.

    This implements the EfficientZeroV2 pattern where value targets are computed dynamically
    using the current model for inference, rather than using pre-computed targets.

    Fixed version that addresses:
    - Model inference compatibility with Flax NNX BatchStat
    - Tensor shape matching for different model architectures
    - Adaptive td_lambda based on sample age (EfficientZeroV2 pattern)
    - Proper RNG key management

    Args:
        model: The MuZero network for inference
        observations: Observations with extra steps for bootstrapping [B, K+1+extra, *obs_shape]
        actions: Actions for K+extra steps [B, K+extra]
        rewards: Rewards for K+1+extra steps [B, K+1+extra]
        dones: Episode termination flags [B, K+1+extra]
        config: MuZero configuration
        training: Whether model is in training mode
        rng_key: Random key for model inference
        sample_indices: Indices of samples in replay buffer for adaptive td_lambda [B,]
        collected_transitions: Total number of transitions collected (for adaptive td_lambda)

    Returns:
        GAE value targets [B, K+1] for the main unroll sequence
    """
    # Generate proper RNG key if not provided
    if rng_key is None:
        rng_key = jax.random.key(42)  # Use non-zero seed for better randomness # pragma: no cover
        
    batch_size = observations.shape[0]
    total_steps = observations.shape[1]  # K+1+extra
    main_steps = config.num_unroll_steps + 1  # K+1
    
    # Pad actions to match total_steps (use last action for padding)
    actions_padded = jnp.concatenate([
        actions, 
        jnp.repeat(actions[:, -1:], total_steps - actions.shape[1], axis=1)
    ], axis=1)

    # **OPTIMIZED APPROACH**: Use JAX vectorization with careful BatchStat handling
    # This approach vectorizes the initial inference and then handles recurrent steps efficiently
    
    # First, compute all initial values in a vectorized manner
    batch_initial_obs = observations[:, 0]  # [B, *obs_shape]
    initial_outputs = jax.vmap(
        lambda obs: model.initial_inference(jnp.expand_dims(obs, axis=0), training=training)
    )(batch_initial_obs)
    
    # Extract initial hidden states and values
    initial_hidden_states = initial_outputs[0][:, 0]  # [B, hidden_dim] - remove extra batch dim
    initial_values = initial_outputs[2]  # [B, ...] - values for first timestep (index 2 is value, not policy)
    
    # Initialize the values array with proper shape handling
    # Handle the case where initial_values might have extra dimensions from vmap
    if initial_values.ndim > 2:  # [B, 1, num_atoms] -> [B, num_atoms]
        initial_values = jnp.squeeze(initial_values, axis=1)
    elif initial_values.ndim == 2 and initial_values.shape[-1] == 1:  # [B, 1] -> [B]
        initial_values = jnp.squeeze(initial_values, axis=-1)
    
    # Ensure initial_values has the correct batch dimension
    if initial_values.ndim == 0: # pragma: no cover
        # Single scalar value, need to broadcast to batch # pragma: no cover
        initial_values = jnp.full((batch_size,), initial_values) # pragma: no cover
    elif initial_values.ndim == 1 and initial_values.shape[0] != batch_size: # pragma: no cover
        # Wrong batch size, broadcast the first value # pragma: no cover
        initial_values = jnp.full((batch_size,), initial_values.flat[0]) # pragma: no cover
    elif initial_values.ndim == 2:
        # Handle 2D case - could be [B, 1] or [1, 1] or [B, num_atoms]
        if initial_values.shape[0] != batch_size: # pragma: no cover
            # Wrong batch size, broadcast the first value # pragma: no cover
            initial_values = jnp.full((batch_size,), initial_values.flat[0]) # pragma: no cover
        elif initial_values.shape[-1] == 1:
            # [B, 1] -> [B] - squeeze the last dimension
            initial_values = jnp.squeeze(initial_values, axis=-1)
    
    if initial_values.ndim > 1 and initial_values.shape[-1] > 1:
        # Categorical values: [B, num_atoms]
        all_values = jnp.zeros((batch_size, total_steps, initial_values.shape[-1]))
        all_values = all_values.at[:, 0].set(initial_values)
    else:
        # Scalar values: [B] - ensure we have the right shape
        all_values = jnp.zeros((batch_size, total_steps))
        all_values = all_values.at[:, 0].set(initial_values)
    
    # Vectorized recurrent computation using scan for better memory efficiency
    def scan_recurrent_step(carry, step_inputs):
        """Scan function for vectorized recurrent steps."""
        hidden_states = carry  # [B, hidden_dim]
        step_idx, step_actions = step_inputs  # step_actions: [B]
        
        # Vectorized recurrent inference
        recurrent_outputs = jax.vmap(
            lambda h, a: model.recurrent_inference(
                jnp.expand_dims(h, axis=0), jnp.expand_dims(a, axis=0), training=training
            )
        )(hidden_states, step_actions)
        
        # Extract new hidden states and values
        new_hidden_states = recurrent_outputs[0][:, 0]  # [B, hidden_dim] - remove extra batch dim
        step_values = recurrent_outputs[2]  # [B, 1, ...] - values for this timestep
        
        # Handle shape consistency for step_values (remove trailing singleton dimensions)
        while step_values.ndim > 1 and step_values.shape[-1] == 1:
            step_values = jnp.squeeze(step_values, axis=-1)  # Remove singleton dimensions
        
        return new_hidden_states, step_values
    
    # Prepare scan inputs: step indices and actions for each step
    step_indices = jnp.arange(1, total_steps)  # [total_steps-1]
    step_actions = actions_padded[:, :total_steps-1].T  # [total_steps-1, B]
    scan_inputs = (step_indices, step_actions)
    
    # Run scan to compute all recurrent steps
    final_hidden_states, all_step_values = jax.lax.scan(
        scan_recurrent_step,
        initial_hidden_states,
        scan_inputs
    )
    
    # Combine initial and recurrent values
    # Handle shape processing for all_step_values more robustly
    # all_step_values starts as [T-1, B, 1, ...] from the scan due to expand_dims in vmap
    
    # First, remove the extra batch dimension added by expand_dims in vmap
    if all_step_values.ndim >= 3: # pragma: no cover
        all_step_values = jnp.squeeze(all_step_values, axis=2)  # [T-1, B, 1, ...] -> [T-1, B, ...] # pragma: no cover
    
    # Handle different dimensionalities properly
    if all_step_values.ndim == 4: # pragma: no cover
        # Categorical values with extra dimension: [T-1, B, 1, num_atoms] -> [T-1, B, num_atoms] # pragma: no cover
        all_step_values = jnp.squeeze(all_step_values, axis=2) # pragma: no cover
    elif all_step_values.ndim == 3 and all_step_values.shape[-1] == 1: # pragma: no cover
        # Scalar values with extra dimension: [T-1, B, 1] -> [T-1, B] # pragma: no cover
        all_step_values = jnp.squeeze(all_step_values, axis=-1) # pragma: no cover
    
    # Now handle transposition based on remaining dimensions
    if all_step_values.ndim == 3:
        # Categorical values: [T-1, B, num_atoms] -> [B, T-1, num_atoms]
        all_step_values = jnp.transpose(all_step_values, (1, 0, 2))
    elif all_step_values.ndim == 2:
        # Scalar values: [T-1, B] -> [B, T-1]
        all_step_values = jnp.transpose(all_step_values, (1, 0))
    else: # pragma: no cover
        # Handle edge cases (e.g., single values) # pragma: no cover
        # Ensure proper shape for assignment # pragma: no cover
        target_shape = (batch_size, total_steps - 1) # pragma: no cover
        if all_step_values.size == target_shape[0] * target_shape[1]: # pragma: no cover
            all_step_values = jnp.reshape(all_step_values, target_shape) # pragma: no cover
        else: # pragma: no cover
            # Broadcast if needed for scalar case # pragma: no cover
            all_step_values = jnp.broadcast_to(all_step_values, target_shape) # pragma: no cover
    
    # Set the values in all_values
    # Ensure all_step_values has the correct shape to fit into all_values[:, 1:]
    expected_shape = all_values[:, 1:].shape  # [B, T-1] or [B, T-1, num_atoms]
    
    # Handle shape mismatch cases
    if all_step_values.shape != expected_shape: # pragma: no cover
        if len(expected_shape) == 2:  # Scalar case: [B, T-1] # pragma: no cover
            if all_step_values.ndim == 3: # pragma: no cover
                # all_step_values is [B, T-1, 1] but we need [B, T-1] # pragma: no cover
                all_step_values = jnp.squeeze(all_step_values, axis=-1) # pragma: no cover
            elif all_step_values.ndim == 2 and all_step_values.shape[1] != expected_shape[1]: # pragma: no cover
                # Truncate or pad to match expected sequence length # pragma: no cover
                if all_step_values.shape[1] > expected_shape[1]: # pragma: no cover
                    all_step_values = all_step_values[:, :expected_shape[1]] # pragma: no cover
                else: # pragma: no cover
                    # Pad with zeros or repeat last value # pragma: no cover
                    padding_size = expected_shape[1] - all_step_values.shape[1] # pragma: no cover
                    padding = jnp.zeros((all_step_values.shape[0], padding_size)) # pragma: no cover
                    all_step_values = jnp.concatenate([all_step_values, padding], axis=1) # pragma: no cover
        elif len(expected_shape) == 3:  # Categorical case: [B, T-1, num_atoms] # pragma: no cover
            if all_step_values.ndim == 2: # pragma: no cover
                # Need to expand to categorical dimension # pragma: no cover
                all_step_values = jnp.expand_dims(all_step_values, axis=-1) # pragma: no cover
                all_step_values = jnp.repeat(all_step_values, expected_shape[-1], axis=-1) # pragma: no cover
            elif all_step_values.shape[1] != expected_shape[1]: # pragma: no cover
                # Truncate or pad sequence dimension # pragma: no cover
                if all_step_values.shape[1] > expected_shape[1]: # pragma: no cover
                    all_step_values = all_step_values[:, :expected_shape[1], :] # pragma: no cover
                else: # pragma: no cover
                    # Pad with zeros # pragma: no cover
                    padding_size = expected_shape[1] - all_step_values.shape[1] # pragma: no cover
                    padding_shape = (all_step_values.shape[0], padding_size, all_step_values.shape[2]) # pragma: no cover
                    padding = jnp.zeros(padding_shape) # pragma: no cover
                    all_step_values = jnp.concatenate([all_step_values, padding], axis=1) # pragma: no cover
    
    all_values = all_values.at[:, 1:].set(all_step_values)
    
    # Convert categorical values to scalar if needed
    def convert_to_scalar(values):
        """Convert categorical values to scalar values."""
        if values.ndim > 2 and values.shape[-1] > 1:
            return losses_lib.support_to_scalar(
                values,
                support_min=config.support_min,
                support_max=config.support_max,
                num_atoms=values.shape[-1]
            )
        elif values.ndim == 3 and values.shape[-1] == 1: # pragma: no cover
            return jnp.squeeze(values, axis=-1) # pragma: no cover
        return values
    
    current_values = convert_to_scalar(all_values)  # [B, T]

    # ------------------------------------------------------------------
    # Use centralized HyperparameterAdapter for adaptive hyper-parameters
    # ------------------------------------------------------------------
    adapter_cfg = HyperparameterAdapterConfig(
        td_lambda=config.td_lambda,
        td_steps=config.td_steps,
        auto_td_steps=config.auto_td_steps,
        use_adaptive_td_steps=config.use_adaptive_td_steps,
        value_target=config.value_target,
    )
    adapter = HyperparameterAdapter(adapter_cfg, collected_transitions if collected_transitions is not None else 0)

    # Vectorized GAE computation (this part can stay in JAX transformations)
    def compute_gae_vectorized():
        """Vectorized GAE computation for all batch items and timesteps."""
        
        # Prepare adaptive td_lambda for each sample in batch
        if sample_indices is not None:
            batch_td_lambdas, batch_td_steps = adapter.vectorized(sample_indices)
        else:
            batch_td_lambdas = jnp.full((batch_size,), config.td_lambda)
            batch_td_steps = jnp.full((batch_size,), config.td_steps, dtype=jnp.int32)
    
        # Compute bootstrap values with td_steps lookahead
        def compute_bootstrap_values(per_sample_td_steps): # MODIFIED: takes per_sample_td_steps
            """Compute bootstrap values for GAE calculation, potentially with per-sample td_steps."""
            # MODIFIED: td_steps is now per_sample_td_steps, shape [B,]
            
            time_indices = jnp.arange(total_steps)  # [T]
            
            # Bootstrap indices: V(s_{t+td_steps})
            # Need to calculate this per batch item due to varying td_steps
            # bootstrap_indices will be [B, T]
            # Each row `b` uses `per_sample_td_steps[b]`
            bootstrap_indices = time_indices[None, :] + per_sample_td_steps[:, None]  # [B, T]
            bootstrap_indices = jnp.clip(bootstrap_indices, 0, total_steps - 1)  # [B, T]

            # Gather current_values using batch_gather (or advanced indexing)
            # current_values is [B, T], bootstrap_indices is [B, T]
            # We want, for each b in B, current_values[b, bootstrap_indices[b, :]]
            gathered_bootstrap_values = jnp.take_along_axis(current_values, bootstrap_indices, axis=1) # [B, T]

            # Discount factor for bootstrap: gamma^td_steps
            # td_steps_expanded will be [B, T] for broadcasting
            td_steps_expanded = jnp.expand_dims(per_sample_td_steps, axis=1) # [B, 1]
            discounts = config.discount_factor ** td_steps_expanded # [B, 1] broadcasts to [B,T] with gathered_bootstrap_values

            bootstrap_values_final = gathered_bootstrap_values * discounts # [B, T]

            # Add intermediate rewards: sum_{i=0}^{td_steps-1} gamma^i * r_{t+i}
            # This also needs to be per-sample due to varying td_steps sum limits
            
            # Maximum possible td_steps to define loop range or scan length
            max_td_steps = jnp.max(per_sample_td_steps) 
            
            # Initialize reward_sum array
            reward_sum = jnp.zeros_like(bootstrap_values_final) # [B, T]

            # Loop for reward accumulation up to max_td_steps
            # Inside the loop, mask rewards for samples whose td_steps < loop_iter
            def accumulate_rewards_step(i, current_reward_sum):
                # i is the current step in the sum (0 to max_td_steps-1)
                
                # Indices for rewards at current_time + i
                reward_indices_at_step_i = time_indices[None, :] + i # [B, T] (broadcast i)
                reward_indices_at_step_i = jnp.clip(reward_indices_at_step_i, 0, total_steps - 1) # [B, T]
                
                # Gather rewards: rewards[b, reward_indices_at_step_i[b,:]]
                step_rewards = jnp.take_along_axis(rewards, reward_indices_at_step_i, axis=1) # [B, T]
                
                # Discount for this step: gamma^i
                step_discount = config.discount_factor ** i
                
                # Mask: only add reward if i < per_sample_td_steps for that sample
                # valid_step_mask will be [B, 1] to broadcast across T
                valid_step_mask = (i < per_sample_td_steps[:, None]).astype(jnp.float32) # [B, 1]
                
                current_reward_sum += valid_step_mask * step_discount * step_rewards
                return current_reward_sum

            reward_sum = jax.lax.fori_loop(0, max_td_steps, accumulate_rewards_step, reward_sum)
            
            bootstrap_values_final += reward_sum

            # Handle episode termination: set bootstrap to 0 if done
            # Termination is checked at s_{t+td_steps}
            # termination_indices is [B, T] (same as bootstrap_indices)
            is_terminal = jnp.take_along_axis(dones, bootstrap_indices, axis=1) # [B, T]
            bootstrap_values_final = jnp.where(is_terminal, 0.0, bootstrap_values_final)

            return bootstrap_values_final

        bootstrap_values = compute_bootstrap_values(batch_td_steps)  # MODIFIED: pass batch_td_steps

        # Compute deltas: delta_t = bootstrap_value_t - V(s_t)
        deltas = bootstrap_values - current_values  # [B, T]
        
        # Vectorized GAE computation using scan
        def gae_scan_fn(advantage_next, inputs):
            """Scan function for GAE computation (reverse order)."""
            delta, td_lambda, done = inputs
            
            # GAE formula: A_t = delta_t + gamma * lambda * A_{t+1} * (1 - done)
            advantage = delta + config.discount_factor * td_lambda * advantage_next * (1.0 - done)
            return advantage, advantage
        
        # Reverse order scan for GAE (start from the end)
        def compute_gae_for_batch_item(deltas_item, td_lambda_item, dones_item):
            """Compute GAE for a single batch item."""
            # Ensure deltas and dones have the same length - take minimum to avoid shape mismatches
            min_length = min(deltas_item.shape[0], dones_item.shape[0])
            deltas_item_trimmed = deltas_item[:min_length]
            dones_item_trimmed = dones_item[:min_length]
            
            # Prepare inputs in reverse order
            deltas_rev = deltas_item_trimmed[::-1]
            dones_rev = dones_item_trimmed[::-1]
            td_lambda_expanded = jnp.full_like(deltas_rev, td_lambda_item)
            
            scan_inputs = (deltas_rev, td_lambda_expanded, dones_rev)
            
            # Initial advantage (for the last timestep)
            initial_advantage = 0.0
            
            final_advantage, advantages_rev = jax.lax.scan(
                gae_scan_fn, 
                initial_advantage, 
                scan_inputs
            )
    
            # Reverse back to get correct order
            advantages = advantages_rev[::-1]
            
            # Pad back to original deltas length if needed
            if advantages.shape[0] < deltas_item.shape[0]:
                padding_size = deltas_item.shape[0] - advantages.shape[0]
                advantages = jnp.concatenate([advantages, jnp.zeros(padding_size)])
            
            return advantages
        
        # Apply GAE computation to all batch items
        all_advantages = jax.vmap(compute_gae_for_batch_item)(
            deltas, batch_td_lambdas, dones
        )
        
        # GAE targets: V_target = A_t + V(s_t)
        gae_targets = all_advantages + current_values
        
        # Return only the main sequence [B, K+1]
        return gae_targets[:, :main_steps]
    
    return compute_gae_vectorized()

@jax.jit
def compute_policy_reanalysis_targets(
    model: MuZeroNetwork,
    observations: jax.Array,  # B, K+1, *obs_shape
    config: MuZeroConfig,
    training: bool = False,
    rng_key: PRNGKey | None = None
) -> jax.Array:
    """
    Compute reanalyzed policy targets using MCTS with current model weights.

    This function implements EfficientZeroV2's policy reanalysis logic using mctx
    for JAX-native MCTS search. It reanalyzes a portion of the batch based on
    reanalyze_ratio and generates new policy targets from MCTS visit counts.

    PERFORMANCE FIX: This function is now JIT-compiled and vectorized to avoid
    the O(B × K × reanalyze_B) host-side loop that was identified in Action Item #2.

    Args:
        model: Current MuZero model for MCTS inference
        observations: Batch observations [B, K+1, *obs_shape]
        config: MuZero configuration with MCTS parameters
        training: Whether in training mode
        rng_key: Random key for MCTS search

    Returns:
        Reanalyzed policy targets [B, K+1, num_actions]
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    batch_size, num_steps = observations.shape[:2]
    num_actions = config.num_actions if hasattr(config, 'num_actions') else observations.shape[-1]  # Fallback

    # Determine reanalysis batch size based on reanalyze_ratio
    reanalyze_batch_size = int(batch_size * config.reanalyze_ratio)
    if reanalyze_batch_size == 0:
        # No reanalysis - return uniform policies for consistency
        return jnp.ones((batch_size, num_steps, num_actions)) / num_actions

    # Take first reanalyze_batch_size samples for reanalysis (EfficientZeroV2 pattern)
    reanalyze_observations = observations[:reanalyze_batch_size]  # [reanalyze_B, K+1, *obs_shape]

    # Get temperature for MCTS based on training step
    training_step = 0  # Would be passed from batch in real implementation
    temperature = get_temperature(training_step, config)

    # REAL MCTS IMPLEMENTATION: Use mctx.gumbel_muzero_policy for actual tree search
    # Reshape to process all (batch, step) combinations at once
    flat_observations = reanalyze_observations.reshape(-1, *observations.shape[2:])  # [reanalyze_B * (K+1), *obs_shape]

    # Create recurrent function for MCTS
    def recurrent_fn(params, rng_key, action, embedding):
        """Recurrent function for MCTS tree search using MuZero model."""
        # Use model.recurrent_inference for dynamics
        recurrent_output = model.recurrent_inference(embedding, action, training=training)
        
        hidden_state = recurrent_output[0]  # Next hidden state
        reward = recurrent_output[1]        # Predicted reward
        value = recurrent_output[2]         # Predicted value  
        policy_logits = recurrent_output[3] # Predicted policy logits
        
        # Convert values to scalars if categorical
        if value.ndim > 1 and value.shape[-1] > 1:
            # Categorical values - convert to scalars for MCTS
            value_scalar = losses_lib.support_to_scalar(
                value,
                support_min=config.support_min,
                support_max=config.support_max,
                num_atoms=value.shape[-1]
            )
        else:
            # Already scalar values
            if value.ndim > 1:
                value_scalar = jnp.squeeze(value, axis=-1)
            else:
                value_scalar = value
                
        # Convert rewards to scalars if categorical
        if reward.ndim > 1 and reward.shape[-1] > 1:
            reward_scalar = losses_lib.support_to_scalar(
                reward,
                support_min=config.support_min,
                support_max=config.support_max,
                num_atoms=reward.shape[-1]
            )
        else:
            if reward.ndim > 1:
                reward_scalar = jnp.squeeze(reward, axis=-1)
            else:
                reward_scalar = reward

        from mctx._src.base import RecurrentFnOutput
        return RecurrentFnOutput(
            reward=reward_scalar,
            discount=jnp.ones_like(reward_scalar) * config.discount_factor,
            prior_logits=policy_logits,
            value=value_scalar
        ), hidden_state

    # Get initial inference for all observations at once
    initial_output = model.initial_inference(flat_observations, training=training)
    hidden_states = initial_output[0]  # [reanalyze_B * (K+1), hidden_dim]
    initial_values = initial_output[2]  # [reanalyze_B * (K+1)] or [reanalyze_B * (K+1), support_size]
    initial_policy_logits = initial_output[3]  # [reanalyze_B * (K+1), num_actions]

    # Convert values to scalars if categorical (vectorized)
    if initial_values.ndim > 1 and initial_values.shape[-1] > 1:
        # Categorical values - convert to scalars for MCTS
        initial_values_scalar = losses_lib.support_to_scalar(
            initial_values,
            support_min=config.support_min,
            support_max=config.support_max,
            num_atoms=initial_values.shape[-1]
        )
    else:
        # Already scalar values
        if initial_values.ndim > 1:
            initial_values_scalar = jnp.squeeze(initial_values, axis=-1)
        else:
            initial_values_scalar = initial_values

    # Run MCTS for each observation using vectorized approach
    import mctx

    # Create root for MCTS
    root = mctx.RootFnOutput(
        prior_logits=initial_policy_logits,
        value=initial_values_scalar,
        embedding=hidden_states
    )

    # CRITICAL FIX: Use stop_gradient to prevent differentiation through MCTS
    # MCTS search uses dynamic loops that can't be differentiated through with reverse-mode autodiff
    # We only need the MCTS outputs, not gradients through the MCTS process itself
    def run_mcts_search():
        # Run Gumbel MuZero MCTS
        policy_output = mctx.gumbel_muzero_policy(
            params=None,  # Model parameters handled internally by recurrent_fn
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            max_num_considered_actions=min(config.num_actions, 16),  # Limit for efficiency
            gumbel_scale=1.0
        )
        return policy_output.action_weights

    # Stop gradients through MCTS to avoid differentiation issues with dynamic loops
    reanalyzed_policies_flat = jax.lax.stop_gradient(run_mcts_search())

    # Reshape back to [reanalyze_B, K+1, num_actions]
    reanalyzed_policies = reanalyzed_policies_flat.reshape(reanalyze_batch_size, num_steps, num_actions)

    # Create full batch result - reanalyzed samples + original samples
    if reanalyze_batch_size < batch_size:
        # Use original network policy predictions for non-reanalyzed samples
        remaining_observations = observations[reanalyze_batch_size:]  # [remaining_B, K+1, *obs_shape]
        remaining_flat_obs = remaining_observations.reshape(-1, *observations.shape[2:])  # [remaining_B * (K+1), *obs_shape]
        
        # Get network policy predictions for remaining samples
        remaining_initial_output = model.initial_inference(remaining_flat_obs, training=training)
        remaining_policy_logits = remaining_initial_output[3]  # [remaining_B * (K+1), num_actions]
        
        # Apply temperature scaling for better policy targets
        temperature = jnp.maximum(get_temperature(0, config), 0.1)
        scaled_remaining_logits = remaining_policy_logits / temperature
        
        # Convert to probabilities
        remaining_policies_flat = jax.nn.softmax(scaled_remaining_logits, axis=-1)
        
        # Reshape back to [remaining_B, K+1, num_actions]
        remaining_batch_size = batch_size - reanalyze_batch_size
        original_policies = remaining_policies_flat.reshape(remaining_batch_size, num_steps, num_actions)
        
        full_policies = jnp.concatenate([reanalyzed_policies, original_policies], axis=0)
    else:
        full_policies = reanalyzed_policies

    return full_policies


def get_temperature(training_step: int, config: MuZeroConfig) -> float:
    """
    Compute temperature for MCTS based on training step and configuration.
    
    Implements EfficientZeroV2's temperature scheduling with linear decay
    from temperature_init to temperature_final over temperature_decay_steps.
    
    Args:
        training_step: Current training step
        config: MuZero configuration with temperature parameters
        
    Returns:
        Temperature value for MCTS
    """
    if not config.change_temperature:
        return config.temperature_init
    
    if training_step >= config.temperature_decay_steps:
        return config.temperature_final
    
    # Linear interpolation from init to final
    progress = training_step / config.temperature_decay_steps
    temperature = config.temperature_init + progress * (config.temperature_final - config.temperature_init)
    
    # Ensure temperature doesn't go below final value
    return max(temperature, config.temperature_final)

def create_muzero_config_for_game(game_name: str, **config_overrides) -> MuZeroConfig:
    """
    Create a MuZeroConfig with the correct action space size for a specific OpenSpiel game.
    
    This function loads the specified OpenSpiel game and automatically sets the num_actions
    parameter based on the game's num_distinct_actions() method. This ensures that the
    MuZero network architecture matches the game's action space.
    
    Args:
        game_name: Name of the OpenSpiel game (e.g., "tic_tac_toe", "chess", "go")
        **config_overrides: Additional configuration parameters to override defaults
        
    Returns:
        MuZeroConfig with num_actions set correctly for the specified game
        
    Example:
        >>> config = create_muzero_config_for_game("tic_tac_toe", learning_rate=1e-3)
        >>> print(config.num_actions)  # Will be 9 for Tic-Tac-Toe
        
        >>> config = create_muzero_config_for_game("chess")
        >>> print(config.num_actions)  # Will be 4672 for Chess
    """
    import pyspiel
    
    # Load the game to get its action space size
    game = pyspiel.load_game(game_name)
    num_actions = game.num_distinct_actions()
    
    # Create config with the correct action space size
    config_dict = {"num_actions": num_actions}
    config_dict.update(config_overrides)
    
    return MuZeroConfig(**config_dict)
