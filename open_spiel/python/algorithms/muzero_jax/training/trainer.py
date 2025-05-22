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

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork # type: ignore
from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib # type: ignore
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig as ActualMuZeroNetworkConfig # Alias to avoid clash

# Type Aliases
PRNGKey = jax.Array
OptState = Any # Optax optimizer state
Params = nnx.State # PyTree of Param Variable instances or their values
ModelBatchStats = nnx.State # PyTree of BatchStat Variable instances or their values
ModelOtherState = nnx.State # PyTree of other Variable instances (like Rngs) or their values

# Updated Batch definition:
# 'observation': (B, *obs_shape) - initial observation at index 0
# 'action': (B, K) - actions taken for K unroll steps (a_0 to a_{K-1})
# 'target_reward': (B, K+1) or (B, K+1, support_size) - r_0 to r_K
# 'target_value': (B, K+1) or (B, K+1, support_size) - v_0 to v_K
# 'target_policy': (B, K+1, num_actions) - p_0 to p_K
# 'game_history_mask': (B, K+1) - 1 if valid step, 0 if padding
Batch = Dict[str, jax.Array]
Metrics = Dict[str, jax.Array]


@dataclasses.dataclass(frozen=True)
class MuZeroConfig:
    """Configuration for the MuZero Learner."""
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
    ssl_consistency_loss_weight: float = 0.0 # Weight for self-supervised consistency loss

    # Optimizer
    learning_rate: float = 1e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    clip_grad_norm: float = 5.0 # Max gradient norm

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

    # TODO: Add other necessary configs, e.g., from EfficientZeroV2/ez/config/default_config.py


class Learner:
    """MuZero Learner.
    
    Handles model initialization, training steps, and overall training orchestration.
    """
    def __init__(self, 
                 model: MuZeroNetwork, 
                 optimizer: optax.GradientTransformation,
                 config: MuZeroConfig,
                 rng_key: PRNGKey):
        self.model = model
        self.optimizer = optimizer # This is the optimizer config/transform
        self.config = config
        self._rng_key, init_key = jax.random.split(rng_key)

        # Initial split of the model
        graphdef, model_params, batch_stats_state, rngs_state, static_state, ellipsis_state = nnx.split(
            self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )
        self.opt_state = self.optimizer.init(model_params) 
        
        self.target_model: Optional[MuZeroNetwork] = None
        self.ema_updater: Optional[optax.Ema] = None 
        self.ema_params_state: Optional[optax.EmaState] = None

        if self.config.use_target_network_ema:
            self.target_model = copy.deepcopy(self.model)
            
            target_graphdef_ema_init, target_model_params_ema_init, t_batch_stats_ema_init, t_rngs_ema_init, t_static_attrs_ema_init, t_other_vars_ema_init = nnx.split(
                self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
            )
            self.ema_updater = optax.ema(decay=self.config.ema_decay, debias=False)
            self.ema_params_state = self.ema_updater.init(target_model_params_ema_init)

            # Sync target_model to initial EMA state (which holds the initial online params)
            # self.ema_params_state.ema holds the initial parameter *values*.
            # t_batch_stats_ema_init, t_rngs_ema_init etc. are the states from target_model itself.
            nnx.update(self.target_model, 
                       self.ema_params_state.ema,  # This updates Param variables in target_model with values from ema_state.ema
                       t_batch_stats_ema_init,    # This ensures BatchStat variables in target_model are set (redundant if deepcopy worked perfectly)
                       t_rngs_ema_init)           # This ensures Rngs variables in target_model are set (redundant)
                       # Static and Other states are not passed to update typically as they are part of graphdef or not variable.

        self.num_training_steps = 0
        self.checkpoint_manager: Optional[ocp.CheckpointManager] = None
        if self.config.checkpoint_dir:
            # Pass options to ocp.CheckpointManager for compatibility
            manager_options = ocp.CheckpointManagerOptions(
                max_to_keep=self.config.max_checkpoints_to_keep, create=True
            )
            self.checkpoint_manager = ocp.CheckpointManager(
                Path(self.config.checkpoint_dir),
                options=manager_options 
                # For the new API with StandardSave/StandardRestore, 
                # an explicit handler here is often not needed, or should be passed 
                # via item_handlers or handler_registry if customizing.
            )

            if self.config.resume_from_checkpoint:
                self.load_checkpoint()
        
        # JIT compile the static train step logic
        # static_argnums for graphdef (0) and optimizer (5), config (7)
        # Optimizer is an optax.GradientTransformation, which is a PyTreeNode and JIT-compatible.
        # MuZeroConfig is a dataclass, also JIT-compatible if its fields are.
        # graphdef is a nnx.GraphDef, which is static.
        self.jit_static_train_step = jax.jit(
            Learner._static_train_step_logic, 
            static_argnames=("graphdef", "optimizer_transform", "config")
        )

    @staticmethod
    def _compute_total_loss_static(
        model: MuZeroNetwork,
        config: MuZeroConfig,
        batch: Batch,
        rng_key: PRNGKey,  # Ensure rng_key is used if any stochastic ops in loss/model
        training: bool
    ) -> Tuple[jax.Array, Metrics]:
        """Computes the total MuZero loss for a batch of data with unrolling."""
        initial_observation = batch['observation'][:, 0] # B, *obs_shape
        actions = batch['action'] # B, K
        target_rewards = batch['target_reward'] # B, K+1 or B, K+1, S
        target_values = batch['target_value'] # B, K+1 or B, K+1, S
        target_policies = batch['target_policy'] # B, K+1, A
        game_history_mask = batch['game_history_mask'] # B, K+1

        # Initial inference
        # Output: hidden_state, reward, value, policy_logits, projected_hidden_state (optional)
        initial_inference_output = model.initial_inference(initial_observation, training=training)
        hidden_state = initial_inference_output[0]
        initial_projection = initial_inference_output[4] if config.use_projection and len(initial_inference_output) > 4 else None

        predicted_rewards_list = [initial_inference_output[1]]
        predicted_values_list = [initial_inference_output[2]]
        predicted_policy_logits_list = [initial_inference_output[3]]
        # Store projections if used
        predicted_projections_list = [initial_projection] if config.use_projection and initial_projection is not None else []

        # Recurrent inferences
        for k in range(config.num_unroll_steps):
            current_action = actions[:, k]
            # Output: next_hidden_state, reward, value, policy_logits, projected_hidden_state (optional)
            recurrent_inference_output = model.recurrent_inference(
                hidden_state, current_action, training=training
            )
            hidden_state = recurrent_inference_output[0]
            predicted_rewards_list.append(recurrent_inference_output[1])
            predicted_values_list.append(recurrent_inference_output[2])
            predicted_policy_logits_list.append(recurrent_inference_output[3])
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

        # Compute losses per step
        total_policy_loss = jnp.array(0.0)
        total_value_loss = jnp.array(0.0)
        total_reward_loss = jnp.array(0.0)
        total_ssl_loss = jnp.array(0.0)

        for k_idx in range(config.num_unroll_steps + 1):
            step_mask = game_history_mask[:, k_idx] # B
            
            # Policy Loss
            p_loss = losses_lib.compute_policy_loss(
                predicted_policy_logits[:, k_idx], target_policies[:, k_idx]
            ) # Scalar per batch item
            total_policy_loss += jnp.sum(p_loss * step_mask) / jnp.maximum(jnp.sum(step_mask), 1.0)

            # Value Loss
            if config.value_support_size > 0:
                v_loss = losses_lib.compute_categorical_value_loss(
                    predicted_values[:, k_idx], target_values[:, k_idx]
                )
            else:
                # Squeeze scalar predictions/targets if they have a trailing dim of 1
                sv = predicted_values[:, k_idx]
                if sv.ndim == 2 and sv.shape[-1] == 1: sv = jnp.squeeze(sv, axis=-1)
                tv = target_values[:, k_idx]
                if tv.ndim == 2 and tv.shape[-1] == 1: tv = jnp.squeeze(tv, axis=-1)
                v_loss = losses_lib.compute_scalar_value_loss(sv, tv)
            total_value_loss += jnp.sum(v_loss * step_mask) / jnp.maximum(jnp.sum(step_mask), 1.0)

            # Reward Loss
            if config.reward_support_size > 0:
                r_loss = losses_lib.compute_categorical_reward_loss(
                    predicted_rewards[:, k_idx], target_rewards[:, k_idx]
                )
            else:
                sr = predicted_rewards[:, k_idx]
                if sr.ndim == 2 and sr.shape[-1] == 1: sr = jnp.squeeze(sr, axis=-1)
                tr = target_rewards[:, k_idx]
                if tr.ndim == 2 and tr.shape[-1] == 1: tr = jnp.squeeze(tr, axis=-1)
                r_loss = losses_lib.compute_scalar_reward_loss(sr, tr)
            total_reward_loss += jnp.sum(r_loss * step_mask) / jnp.maximum(jnp.sum(step_mask), 1.0)
            
            # SSL Loss for this step (if applicable)
            # Compare projection at step k_idx with initial_projection (from step 0)
            if config.use_projection and config.ssl_consistency_loss_weight > 0 and \
               predicted_projections is not None and initial_projection is not None and k_idx > 0: 
                # Make sure predicted_projections has shape (B, K+1, proj_dim)
                ssl_loss_step = losses_lib.compute_projection_consistency_loss(
                    predicted_projections[:, k_idx], # Projection at current unroll step k_idx
                    initial_projection # Projection from initial_inference (step 0)
                )
                total_ssl_loss += jnp.sum(ssl_loss_step * step_mask) / jnp.maximum(jnp.sum(step_mask), 1.0)


        # L2 regularization needs only Param state.
        # We split the model here again to get the params. This assumes `model` is the full model.
        # In the context of _static_train_step_logic, this means it's the model *before* gradient update.
        _, model_params_for_l2, _, _, _, _ = nnx.split(model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...) 
        l2_loss = losses_lib.l2_regularization(model_params_for_l2, config.l2_weight)

        final_loss = (
            config.policy_loss_weight * total_policy_loss
            + config.value_loss_weight * total_value_loss 
            + config.reward_loss_weight * total_reward_loss 
            + l2_loss
        )
        if config.use_projection and config.ssl_consistency_loss_weight > 0:
            final_loss += config.ssl_consistency_loss_weight * total_ssl_loss
        
        metrics = {
            'total_loss': final_loss,
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'reward_loss': total_reward_loss,
            'l2_loss': l2_loss,
        }
        if config.use_projection and config.ssl_consistency_loss_weight > 0:
            metrics['ssl_loss'] = total_ssl_loss
            
        return final_loss, metrics

    @staticmethod
    def _static_train_step_logic(
        graphdef: nnx.GraphDef,
        current_params: Params,
        current_batch_stats: ModelBatchStats,
        static_state: nnx.State, # nnx_graph.Static parts
        current_rngs_state: nnx.State, # nnx.Rngs parts
        current_ellipsis_state: nnx.State, # ... parts
        optimizer_transform: optax.GradientTransformation, # The optimizer object itself
        current_opt_state: OptState,
        config: MuZeroConfig,
        batch: Batch,
        rng_key: PRNGKey
    ) -> Tuple[Params, ModelBatchStats, nnx.State, nnx.State, OptState, Metrics]:
        """Static logic for a single training step, suitable for JIT."""

        # 1. Reconstruct the model for this step using provided state components
        model_for_step = nnx.merge(
            graphdef, 
            current_params, 
            current_batch_stats, 
            static_state, 
            current_rngs_state,
            current_ellipsis_state
        )

        # 2. Define loss_fn_for_grad
        def loss_fn_for_grad(model_to_grad: MuZeroNetwork) -> Tuple[jax.Array, Metrics]:
            # model_to_grad here is model_for_step with its state correctly handled by nnx.value_and_grad
            loss_value, metrics_from_loss = Learner._compute_total_loss_static(
                model_to_grad, config, batch, rng_key, training=True
            )
            return loss_value, metrics_from_loss

        # 3. Compute grads. model_for_step is updated with new batch_stats (and other mutable state like RNGs) here.
        (loss_value, metrics), grads = nnx.value_and_grad(
            loss_fn_for_grad, argnums=0, has_aux=True
        )(model_for_step)

        # 4. Extract the *new* state (batch_stats, other_state) from model_for_step
        #    Parameters (current_params) are not changed by the forward pass.
        (graphdef_after_fwd, # GraphDef should be static, but split returns it
         params_after_fwd, 
         new_batch_stats_from_fwd, 
         new_rngs_state_from_fwd, 
         static_attrs_after_fwd, 
         new_ellipsis_vars_from_fwd) = nnx.split(
            model_for_step, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )

        # 5. Update parameters using the optimizer
        #    current_params (the nnx.Param part of the model) is passed to optimizer.update
        updates, new_opt_state = optimizer_transform.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        
        metrics['grad_norm'] = optax.global_norm(grads)
        # new_params is the PyTree of updated parameter *values*.
        # To get their norm, we pass this PyTree directly.
        metrics['param_norm'] = optax.global_norm(new_params) 

        return (
            new_params, 
            new_batch_stats_from_fwd, 
            new_rngs_state_from_fwd, 
            new_ellipsis_vars_from_fwd, 
            new_opt_state, 
            metrics
        )


    def train_step(self, batch: Batch) -> Tuple[MuZeroNetwork, OptState, Metrics]:
        """Performs a single training step (non-JIT path, for tests/debugging)."""
        self._rng_key, step_rng = jax.random.split(self._rng_key)
        
        # Get current model state, including GraphDef
        model_graphdef, current_params, current_batch_stats, current_rngs, current_static_attrs, current_ellipsis_vars = nnx.split(
            self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )
        # Pass individual state components to _static_train_step_logic
        new_params, new_batch_stats, new_rngs_state, new_ellipsis_state, self.opt_state, metrics = Learner._static_train_step_logic(
            model_graphdef, 
            current_params, 
            current_batch_stats, 
            current_static_attrs, 
            current_rngs, # Pass current_rngs directly
            current_ellipsis_vars, # Pass current_ellipsis_vars directly
            self.optimizer, self.opt_state, self.config, batch, step_rng
        )
        
        # Update the main model instance with the new states
        nnx.update(self.model, new_params, new_batch_stats, new_rngs_state, new_ellipsis_state)

        # EMA Update logic (if enabled)
        if self.config.use_target_network_ema and self.target_model is not None and \
           self.ema_updater is not None and self.ema_params_state is not None:
            
            # EMA is updated with the *new parameters* of the online model
            # new_params are the nnx.State containing Param Variables with updated values.
            # self.ema_updater.init was called with params (State containing Variable objects).
            # optax.ema.update expects `updates` to be a PyTree of raw values, matching structure of `params` given to `init`.
            # `new_params` here is already a PyTree of raw values.
            
            # The `params` argument to ema.update should be the *current* EMA averaged parameters
            # new_params is the nnx.State containing Param Variables with updated values.
            # current_ema_values should be the current averaged *parameter values*.
            # self.ema_params_state.ema is an nnx.State of parameter *values*.
            current_ema_param_values_state = self.ema_params_state.ema

            updated_ema_values, self.ema_params_state = self.ema_updater.update(
                updates=new_params, # Pass the new parameter state (containing Variables)
                state=self.ema_params_state
                # The `params` argument is for when the `updates` are gradients, not the new params themselves.
                # When `updates` are the new parameters, `params` should not be passed or be None.
            )
            
            target_graphdef_train, _, target_bs_train, target_rngs_train, target_static_train, target_other_train = \
                nnx.split(self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
            # updated_ema_values is an nnx.State of parameter values
            nnx.update(self.target_model,
                       updated_ema_values, # This is a State of param values
                       target_bs_train,    # Keep existing batch stats
                       target_rngs_train)  # Keep existing rngs

        self.num_training_steps += 1
        return self.model, self.opt_state, metrics

    def train(self, replay_buffer_iterator_fn: Callable[[], Generator[Batch, None, None]], num_epochs: int, steps_per_epoch: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs, {steps_per_epoch} steps per epoch.")
        
        batch_generator = replay_buffer_iterator_fn() # Expect a generator

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for step in range(steps_per_epoch):
                try:
                    batch = next(batch_generator)
                except StopIteration: 
                    print("Replay buffer iterator exhausted. Re-initializing generator for next epoch or stopping.") 
                    batch_generator = replay_buffer_iterator_fn() # Re-initialize for next epoch or if it's a one-shot per epoch
                    try:
                        batch = next(batch_generator)
                    except StopIteration:
                        print("Replay buffer truly exhausted. Stopping training.")
                        return

                self._rng_key, step_rng = jax.random.split(self._rng_key)
                
                # Get current model state, including GraphDef
                model_graphdef, current_params, current_batch_stats, current_rngs, current_static_attrs, current_ellipsis_vars = nnx.split(
                    self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
                )
                # Pass individual state components to the JIT-compiled function
                new_params, new_batch_stats, new_rngs_state, new_ellipsis_state, self.opt_state, metrics = self.jit_static_train_step(
                    model_graphdef, 
                    current_params, 
                    current_batch_stats, 
                    current_static_attrs, 
                    current_rngs, # Pass current_rngs directly
                    current_ellipsis_vars, # Pass current_ellipsis_vars directly
                    self.optimizer, self.opt_state, self.config, batch, step_rng
                )
                
                # Update the main model instance with the new states
                nnx.update(self.model, new_params, new_batch_stats, new_rngs_state, new_ellipsis_state)

                # EMA Update logic
                if self.config.use_target_network_ema and self.target_model is not None and \
                   self.ema_updater is not None and self.ema_params_state is not None:

                    # new_params are the raw updated parameter values from the optimizer
                    # We need to ensure the structure matches what ema_updater expects.
                    # optax.ema.init was called with the PyTree of Param *Variables*.
                    # optax.ema.update expects `updates` to be a PyTree of raw values, matching structure of `params` given to `init`.
                    # `new_params` here is already a PyTree of raw values.
                    
                    # The `params` argument to ema.update should be the *current* EMA averaged parameters
                    # new_params is the nnx.State containing Param Variables with updated values.
                    # current_ema_values should be the current averaged *parameter values*.
                    # self.ema_params_state.ema is an nnx.State of parameter *values*.
                    current_ema_param_values_state = self.ema_params_state.ema

                    updated_ema_values, self.ema_params_state = self.ema_updater.update(
                        updates=new_params, # new_params are the latest online model's param values (as nnx.State)
                        state=self.ema_params_state
                        # The `params` argument is for when the `updates` are gradients, not the new params themselves.
                    )
                    
                    target_graphdef_train, _, target_bs_train, target_rngs_train, target_static_train, target_other_train = \
                        nnx.split(self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
                    # updated_ema_values is an nnx.State of parameter values
                    nnx.update(self.target_model,
                               updated_ema_values, # This is a State of param values
                               target_bs_train,    # Keep existing batch stats
                               target_rngs_train)  # Keep existing rngs


                self.num_training_steps += 1
                
                # WandB Logging
                if wandb.run is not None and metrics: # Ensure metrics exist and wandb is initialized
                    wandb.log(metrics, step=self.num_training_steps)

                if (self.checkpoint_manager and self.num_training_steps % self.config.checkpoint_frequency == 0 and self.num_training_steps > 0):
                    logging.info(f"Regular checkpoint: step {self.num_training_steps}, freq {self.config.checkpoint_frequency}")
                    self.save_checkpoint(force_save=False) # Regular periodic save
                elif (self.checkpoint_manager and step == steps_per_epoch -1 and epoch == num_epochs -1 ): 
                    logging.info(f"End of training checkpoint: step {self.num_training_steps}")
                    self.save_checkpoint(force_save=True) # Force save at the very end

                if step % 100 == 0: 
                    print(f"  Step {step + 1}/{steps_per_epoch}, Total Steps: {self.num_training_steps}, Loss: {metrics['total_loss']:.4f}")

        print("Training finished.")
        if self.checkpoint_manager:
            self.checkpoint_manager.wait_until_finished()


    def save_checkpoint(self, force_save: bool = False):
        """Saves the current learner state."""
        if not self.checkpoint_manager:
            print("Checkpoint manager not configured. Skipping save.")
            return

        save_step = self.num_training_steps
        
        # Split model states for saving
        model_graphdef, model_params, model_batch_stats, model_rngs, model_static, model_ellipsis = nnx.split(
            self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )

        items_to_save = {
            'model_graphdef': model_graphdef, # GraphDef is static but good to save for consistency
            'model_params': model_params,
            'model_batch_stats': model_batch_stats,
            'model_rngs': model_rngs,
            'model_static': model_static,
            'model_ellipsis': model_ellipsis,
            'opt_state': self.opt_state,
            'num_training_steps': self.num_training_steps,
            'rng_key': self._rng_key,
            'ema_params_state': self.ema_params_state
        }

        if self.target_model is not None:
            target_model_graphdef, target_model_params, target_model_batch_stats, target_model_rngs, target_model_static, target_model_ellipsis = nnx.split(
                self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
            )
            items_to_save['target_model_graphdef'] = target_model_graphdef
            items_to_save['target_model_params'] = target_model_params
            items_to_save['target_model_batch_stats'] = target_model_batch_stats
            items_to_save['target_model_rngs'] = target_model_rngs
            items_to_save['target_model_static'] = target_model_static
            items_to_save['target_model_ellipsis'] = target_model_ellipsis
        else:
            # Add placeholders if target_model is None but we want a consistent structure
            items_to_save['target_model_graphdef'] = None
            items_to_save['target_model_params'] = None
            items_to_save['target_model_batch_stats'] = None
            items_to_save['target_model_rngs'] = None
            items_to_save['target_model_static'] = None
            items_to_save['target_model_ellipsis'] = None
            
        save_args = ocp.args.StandardSave(items_to_save)

        if force_save or (self.num_training_steps > 0 and self.num_training_steps % self.config.checkpoint_frequency == 0) :
             logging.info(f"SAVE_CHECKPOINT: Condition met. force_save={force_save}, num_training_steps={self.num_training_steps}, freq={self.config.checkpoint_frequency}. Saving checkpoint for step {save_step}.")
             self.checkpoint_manager.save(save_step, args=save_args)
             print(f"Checkpoint saved at step {save_step}")
             self.checkpoint_manager.wait_until_finished() # Ensure save completes
        else:
             logging.info(f"SAVE_CHECKPOINT: Condition NOT met. force_save={force_save}, num_training_steps={self.num_training_steps}, freq={self.config.checkpoint_frequency}. Skipping save for step {save_step}.")

    def load_checkpoint(self) -> bool:
        """Loads the latest learner state from the checkpoint directory."""
        if not self.checkpoint_manager:
            print("Checkpoint manager not configured. Skipping load.")
            return False # pragma: no cover
            
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is None:
            print("No checkpoint found to resume from.")
            return False

        print(f"Attempting to load checkpoint from step {latest_step}...")
        
        # We provide an "abstract" version of the items to guide restoration,
        # especially for complex PyTrees or if types need to be exact.
        # For nnx.Modules, providing the instance is a good way to hint structure.
        
        # Split current model to provide abstract structure for its components
        model_graphdef_abs, model_params_abs, model_bs_abs, model_rngs_abs, model_static_abs, model_ellipsis_abs = nnx.split(
            self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
        )

        abstract_items_to_restore = {
            'model_graphdef': model_graphdef_abs, 
            'model_params': model_params_abs,
            'model_batch_stats': model_bs_abs,
            'model_rngs': model_rngs_abs,
            'model_static': model_static_abs,
            'model_ellipsis': model_ellipsis_abs,
            'opt_state': self.opt_state, # Optax states are PyTrees
            'num_training_steps': 0, # type hint
            'rng_key': self._rng_key, # type hint
            'ema_params_state': self.ema_params_state 
        }

        if self.target_model is not None:
            target_gdef_abs, target_params_abs, target_bs_abs, target_rngs_abs, target_static_abs, target_ellipsis_abs = nnx.split(
                self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
            abstract_items_to_restore['target_model_graphdef'] = target_gdef_abs
            abstract_items_to_restore['target_model_params'] = target_params_abs
            abstract_items_to_restore['target_model_batch_stats'] = target_bs_abs
            abstract_items_to_restore['target_model_rngs'] = target_rngs_abs
            abstract_items_to_restore['target_model_static'] = target_static_abs
            abstract_items_to_restore['target_model_ellipsis'] = target_ellipsis_abs
        else:
            abstract_items_to_restore['target_model_graphdef'] = None
            abstract_items_to_restore['target_model_params'] = None
            abstract_items_to_restore['target_model_batch_stats'] = None
            abstract_items_to_restore['target_model_rngs'] = None
            abstract_items_to_restore['target_model_static'] = None
            abstract_items_to_restore['target_model_ellipsis'] = None

        try:
            # Use ocp.args.StandardRestore
            restore_args = ocp.args.StandardRestore(abstract_items_to_restore)
            restored_state_dict = self.checkpoint_manager.restore(
                latest_step, 
                args=restore_args
            )
            
            # Update learner state from restored_state_dict.
            # Update self.model
            # GraphDef is static and part of the model structure, usually not updated unless model def changes.
            # However, if saved, we can merge it to ensure consistency if ever needed, though typically
            # the existing model's graphdef is correct.
            nnx.update(self.model, 
                       restored_state_dict['model_params'], 
                       restored_state_dict['model_batch_stats'], 
                       restored_state_dict['model_rngs'],
                       restored_state_dict['model_ellipsis']) 
            # model_static is part of graphdef, not updated here. model_graphdef also not directly updated into instance.

            self.opt_state = restored_state_dict['opt_state']
            self.num_training_steps = restored_state_dict['num_training_steps']
            self._rng_key = restored_state_dict['rng_key']

            if self.target_model is not None and restored_state_dict.get('target_model_params') is not None:
                nnx.update(self.target_model,
                           restored_state_dict['target_model_params'],
                           restored_state_dict['target_model_batch_stats'],
                           restored_state_dict['target_model_rngs'],
                           restored_state_dict['target_model_ellipsis'])
            elif self.config.use_target_network_ema and self.target_model is not None:
                print("Warning: EMA enabled, target model components not fully in ckpt. Re-syncing with online model.")
                # Re-initialize target model from current online model if its state wasn't fully saved/restored
                # This might happen if target_model was None during save but is now available.
                self.target_model = copy.deepcopy(self.model)
                # Fall through to ema_params_state handling which will use the new target_model params for EMA init if needed.

            if 'ema_params_state' in restored_state_dict and restored_state_dict['ema_params_state'] is not None:
                 self.ema_params_state = restored_state_dict['ema_params_state']
                 # If target_model was re-initialized or its params were just loaded,
                 # ensure it's synced with the loaded EMA state values.
                 if self.target_model and self.ema_updater and self.ema_params_state :
                    # Split the just-updated or deepcopied target_model to get its current non-Param state
                    _, _, target_bs_resume, target_rngs_resume, _, target_ellipsis_resume = nnx.split(
                        self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
                    )
                    nnx.update(self.target_model, 
                               self.ema_params_state.ema, # ema_params_state.ema contains the EMA *values*
                               target_bs_resume, 
                               target_rngs_resume,
                               target_ellipsis_resume)

            elif self.config.use_target_network_ema and self.ema_updater:
                print("Warning: EMA enabled, ema_params_state not in ckpt. Reinitializing EMA state from online model.")
                _, online_params_after_restore, _, _, _, _ = nnx.split(self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
                self.ema_params_state = self.ema_updater.init(online_params_after_restore) 
                if self.target_model: 
                    _, _, target_bs_reinit_ema, target_rngs_reinit_ema, _, target_ellipsis_reinit_ema = nnx.split(
                        self.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
                    )
                    nnx.update(self.target_model, 
                               self.ema_params_state.ema, 
                               target_bs_reinit_ema, 
                               target_rngs_reinit_ema,
                               target_ellipsis_reinit_ema)

            print(f"Successfully loaded checkpoint from step {latest_step}.")
            self.checkpoint_manager.wait_until_finished() 
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

# Example usage (for testing/illustration - will be in tests)
if __name__ == '__main__': # pragma: no cover
    from open_spiel.python.algorithms.muzero_jax.models.network import VisualRepresentationNetwork, MLPValuePolicyNetwork, DynamicsNetwork, RewardNetwork, MuZeroNetwork as TestMuZeroNetwork # type: ignore
    from open_spiel.python.algorithms.muzero_jax.models.layers import DownSample, ResidualBlock, FCResidualBlock, MLP # type: ignore

    key = jax.random.PRNGKey(42)
    key_model_init, key_learner_init, key_batch_gen = jax.random.split(key, 3)

    # Using ActualMuZeroNetworkConfig to define the structure for the main example
    net_config_main = ActualMuZeroNetworkConfig(
        observation_shape=(3, 96, 96),
        num_actions=18,
        num_channels=16,
        num_residual_blocks=1,
        num_fc_residual_blocks=1,
        num_hidden_units_fc=32,
        value_support_size=0,
        reward_support_size=0,
        downsample_channels=8,
        downsample_blocks=1,
        use_batch_norm=True,
        use_projection=False,
        spatial_extents=(96,96), # Should match observation if image
        use_image_observation=True,
        projection_hidden_dim=64, # Example value
        projection_head_output_dim=32, # Example value
        action_embedding_dim=16 # Example value for dummy dynamics compatibility
    )
    
    class MainVisualRepresentationNetwork(nnx.Module): # Renamed
        def __init__(self, config: ActualMuZeroNetworkConfig, *, rngs: nnx.Rngs):
            self.downsample = DownSample(config.observation_shape[0], config.downsample_channels, rngs=rngs) # Assuming obs_shape is (C,H,W)
            self.conv3x3 = nnx.Conv(in_features=config.downsample_channels, out_features=config.num_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs)
            self.bn_initial = nnx.BatchNorm(config.num_channels, use_running_average=not config.use_batch_norm, rngs=rngs) if config.use_batch_norm else nnx.Identity(rngs=rngs)
            self.residuals = [ResidualBlock(config.num_channels, config.num_channels, rngs=nnx.Rngs(params=jax.random.fold_in(rngs.params(), i), dropout=jax.random.fold_in(rngs.dropout(), i))) for i in range(config.num_residual_blocks)]
        def __call__(self, x: jax.Array, training: bool): 
            # Input x expected as (B, C, H, W)
            # Transpose to (B, H, W, C) for Flax NNX conv layers
            if x.shape[1] == net_config_main.observation_shape[0] and x.shape[2] == net_config_main.observation_shape[1] and x.shape[3] == net_config_main.observation_shape[2]: 
                x = jnp.transpose(x, (0, 2, 3, 1)) 
            x = self.downsample(x, training)
            x = self.conv3x3(x)
            x = self.bn_initial(x, use_running_average=not training)
            x = nnx.relu(x)
            for block in self.residuals:
                x = block(x, training)
            return x

    model_instance_main = TestMuZeroNetwork(
        representation_network_def=lambda config, *, rngs: MainVisualRepresentationNetwork(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: MLPValuePolicyNetwork(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: DynamicsNetwork(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: RewardNetwork(config, rngs=rngs),
        projection_network_def=None, 
        config=net_config_main, # Pass the ActualMuZeroNetworkConfig instance
        rngs=nnx.Rngs(params=key_model_init)
    )

    learner_config_main = MuZeroConfig(
        value_support_size=net_config_main.value_support_size,
        reward_support_size=net_config_main.reward_support_size,
        num_unroll_steps=2, 
        td_steps=2,
        batch_size=2, 
        l2_weight=1e-4,
        learning_rate=1e-3,
        use_projection=False,
        checkpoint_dir="/tmp/muzero_jax_test_checkpoints_main", 
        checkpoint_frequency=2, # Checkpoint more frequently for test
        use_target_network_ema=True, 
        ema_decay=0.95,
        resume_from_checkpoint=False # Start fresh for this example
    )
    
    optimizer_instance_main = optax.adam(learning_rate=learner_config_main.learning_rate)
    learner_main = Learner(model_instance_main, optimizer_instance_main, learner_config_main, key_learner_init)

    B_main = learner_config_main.batch_size
    K_main = learner_config_main.num_unroll_steps
    obs_shape_main = net_config_main.observation_shape 
    
    dummy_batches_main = []
    for i in range(5): # Generate a few batches
        k_batch = jax.random.fold_in(key_batch_gen, i)
        obs_batch = jax.random.uniform(k_batch, (B_main, obs_shape_main[0], obs_shape_main[1], obs_shape_main[2]))
        act_batch = jax.random.randint(k_batch, (B_main, K_main), 0, net_config_main.num_actions)
        
        val_target = jax.random.normal(k_batch, (B_main, K_main + 1))
        rew_target = jax.random.normal(k_batch, (B_main, K_main + 1))
        pol_target = jax.random.uniform(k_batch, (B_main, K_main + 1, net_config_main.num_actions))
        pol_target = pol_target / jnp.sum(pol_target, axis=-1, keepdims=True)
        mask = jnp.ones((B_main, K_main + 1), dtype=jnp.float32)

        dummy_batches_main.append({
            'observation': obs_batch, 'action': act_batch, 
            'target_reward': rew_target, 'target_value': val_target,
            'target_policy': pol_target, 'game_history_mask': mask,
        })

    def dummy_replay_buffer_iterator_fn_main() -> Generator[Batch, None, None]:
        for batch_item in dummy_batches_main:
            yield batch_item

    print("Starting dummy training loop with JIT...")
    learner_main.train(dummy_replay_buffer_iterator_fn_main, num_epochs=1, steps_per_epoch=len(dummy_batches_main))

    print("\nTrying to load from checkpoint...")
    learner_config_resume = dataclasses.replace(learner_config_main, resume_from_checkpoint=True)
    model_instance_resume = TestMuZeroNetwork( # Recreate model structure for new learner
        representation_network_def=lambda config, *, rngs: MainVisualRepresentationNetwork(config, rngs=rngs),
        prediction_network_def=lambda config, *, rngs: MLPValuePolicyNetwork(config, rngs=rngs),
        dynamics_network_def=lambda config, *, rngs: DynamicsNetwork(config, rngs=rngs),
        reward_network_def=lambda config, *, rngs: RewardNetwork(config, rngs=rngs),
        projection_network_def=None, 
        config=net_config_main,
        rngs=nnx.Rngs(params=jax.random.key(1)) # Can use a different key for init, loaded state will overwrite
    )
    optimizer_instance_resume = optax.adam(learning_rate=learner_config_resume.learning_rate)
    learner_resume = Learner(model_instance_resume, optimizer_instance_resume, learner_config_resume, jax.random.key(2))
    
    if learner_resume.num_training_steps > 0:
        print(f"Resumed successfully from step {learner_resume.num_training_steps}")
    else:
        print("Did not resume or resumed at step 0.")

    print("Done with dummy run.") 