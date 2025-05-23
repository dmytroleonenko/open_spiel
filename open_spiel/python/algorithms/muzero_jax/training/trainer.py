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
    """Learner class for training the MuZero model with standard Flax NNX patterns."""

    def __init__(self, 
                 model: MuZeroNetwork, 
                 optimizer_def: optax.GradientTransformation,
                 config: MuZeroConfig,
                 rng_key: PRNGKey):
        self.model = model
        self.config = config
        self._rng_key = rng_key
        self.num_training_steps = 0
        
        # Use nnx.Optimizer for standard Flax pattern
        self.optimizer = nnx.Optimizer(model, optimizer_def)
        
        # Target model and EMA for target network updates
        self.target_model = None
        self.ema_updater = None
        self.ema_params_state = None
        
        if config.use_target_network_ema:
            # Create target model as a copy of the main model
            graphdef, params, batch_stats, rngs, static, ellipsis = nnx.split(
                model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
            )
            self.target_model = nnx.merge(graphdef, params, batch_stats, rngs, static, ellipsis)
            
            # Set up EMA updater
            self.ema_updater = optax.ema(config.ema_decay)
            self.ema_params_state = self.ema_updater.init(params)

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

        # JIT-compiled training step using standard nnx pattern
        self.jit_train_step = nnx.jit(self._train_step_impl)

    @nnx.jit
    def _train_step_impl(self, batch: Batch) -> Tuple[Metrics]:
        """JIT-compiled training step implementation using standard nnx patterns."""
        self._rng_key, step_rng = jax.random.split(self._rng_key)
        
        def loss_fn(model: MuZeroNetwork) -> Tuple[jax.Array, Metrics]:
            """Loss function for gradient computation."""
            loss_value, metrics = self._compute_total_loss_static(
                model, self.config, batch, step_rng, training=True
            )
            return loss_value, metrics

        # Compute loss and gradients using standard nnx pattern
        (loss_value, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model)
        
        # Update model parameters using nnx.Optimizer
        self.optimizer.update(grads)
        
        # Add gradient and parameter norms to metrics
        metrics['grad_norm'] = optax.global_norm(grads)
        metrics['param_norm'] = optax.global_norm(nnx.state(self.model, nnx.Param))
        
        # Update EMA target network if enabled
        if self.config.use_target_network_ema and self.target_model is not None:
            self._update_target_network_ema()
            
        return metrics

    def _update_target_network_ema(self):
        """Update the target network using EMA of the online model parameters."""
        # Only update if EMA is enabled and components are initialized
        if (not self.config.use_target_network_ema or 
            self.target_model is None or 
            self.ema_updater is None or 
            self.ema_params_state is None):
            return
            
        # Get current model parameters
        current_params = nnx.state(self.model, nnx.Param)
        
        # Update EMA state
        updated_ema_params, self.ema_params_state = self.ema_updater.update(
            updates=current_params, 
            state=self.ema_params_state
        )
        
        # Update target model with EMA parameters
        nnx.update(self.target_model, updated_ema_params)

    def train_step(self, batch: Batch) -> Metrics:
        """Performs a single training step (can be used for testing/debugging)."""
        metrics = self.jit_train_step(batch)
        self.num_training_steps += 1
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
                except StopIteration: 
                    print("Replay buffer iterator exhausted. Re-initializing generator for next epoch or stopping.") 
                    batch_generator = replay_buffer_iterator_fn()
                    try:
                        batch = next(batch_generator)
                    except StopIteration:
                        print("Replay buffer truly exhausted. Stopping training.")
                        return

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
                    if 'ssl_loss' in metrics:
                        wandb.log({'loss/ssl': metrics['ssl_loss']}, step=self.num_training_steps)

                # Log to console occasionally
                if self.num_training_steps % 10 == 0:
                    logging.info(f"Training step {self.num_training_steps}, "
                               f"loss: {metrics.get('total_loss', 'N/A'):.6f}, "
                               f"grad_norm: {metrics.get('grad_norm', 'N/A'):.6f}")

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
        training: bool
    ) -> Tuple[jax.Array, Metrics]:
        """Computes the total MuZero loss for a batch of data with unrolling."""
        initial_observation = batch['observation'][:, 0] # B, *obs_shape
        actions = batch['action'] # B, K
        target_rewards = batch['target_reward'] # B, K+1 or B, K+1, S
        target_values = batch['target_value'] # B, K+1 or B, K+1, S
        target_policies = batch['target_policy'] # B, K+1, A
        game_history_mask = batch['game_history_mask'] # B, K+1

        # Apply gradient scaling per EfficientZeroV2 pattern
        gradient_scale = 1.0 / config.num_unroll_steps

        # Initial inference
        initial_inference_output = model.initial_inference(initial_observation, training=training)
        hidden_state = initial_inference_output[0]
        initial_projection = initial_inference_output[4] if config.use_projection and len(initial_inference_output) > 4 else None

        predicted_rewards_list = [initial_inference_output[1]]
        predicted_values_list = [initial_inference_output[2]]
        predicted_policy_logits_list = [initial_inference_output[3]]
        predicted_projections_list = [initial_projection] if config.use_projection and initial_projection is not None else []

        # Recurrent inferences
        for k in range(config.num_unroll_steps):
            current_action = actions[:, k]
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

        # Compute losses per step with gradient scaling
        total_policy_loss = jnp.array(0.0)
        total_value_loss = jnp.array(0.0)
        total_reward_loss = jnp.array(0.0)
        total_ssl_loss = jnp.array(0.0)

        for k_idx in range(config.num_unroll_steps + 1):
            step_mask = game_history_mask[:, k_idx] # B
            
            # Policy Loss with gradient scaling
            p_loss = losses_lib.compute_policy_loss(
                predicted_policy_logits[:, k_idx], target_policies[:, k_idx]
            ) * gradient_scale
            masked_p_loss = p_loss * step_mask
            total_policy_loss += jnp.sum(masked_p_loss) / jnp.maximum(jnp.sum(step_mask), 1.0)

            # Value Loss with gradient scaling
            if config.value_support_size > 0:
                v_loss = losses_lib.compute_categorical_value_loss(
                    predicted_values[:, k_idx], target_values[:, k_idx]
                ) * gradient_scale
            else:
                sv = predicted_values[:, k_idx]
                if sv.ndim == 2 and sv.shape[-1] == 1: sv = jnp.squeeze(sv, axis=-1)
                tv = target_values[:, k_idx]
                if tv.ndim == 2 and tv.shape[-1] == 1: tv = jnp.squeeze(tv, axis=-1)
                v_loss = losses_lib.compute_scalar_value_loss(sv, tv) * gradient_scale
            masked_v_loss = v_loss * step_mask
            total_value_loss += jnp.sum(masked_v_loss) / jnp.maximum(jnp.sum(step_mask), 1.0)

            # Reward Loss with gradient scaling
            if config.reward_support_size > 0:
                r_loss = losses_lib.compute_categorical_reward_loss(
                    predicted_rewards[:, k_idx], target_rewards[:, k_idx]
                ) * gradient_scale
            else:
                sr = predicted_rewards[:, k_idx]
                if sr.ndim == 2 and sr.shape[-1] == 1: sr = jnp.squeeze(sr, axis=-1)
                tr = target_rewards[:, k_idx]
                if tr.ndim == 2 and tr.shape[-1] == 1: tr = jnp.squeeze(tr, axis=-1)
                r_loss = losses_lib.compute_scalar_reward_loss(sr, tr) * gradient_scale
            masked_r_loss = r_loss * step_mask
            total_reward_loss += jnp.sum(masked_r_loss) / jnp.maximum(jnp.sum(step_mask), 1.0)
            
            # SSL Loss with stop_gradient and gradient scaling (EfficientZeroV2 pattern)
            if config.use_projection and config.ssl_consistency_loss_weight > 0 and \
               predicted_projections is not None and initial_projection is not None and k_idx > 0: 
                ssl_loss_step = losses_lib.compute_projection_consistency_loss(
                    predicted_projections[:, k_idx], 
                    jax.lax.stop_gradient(initial_projection)  # Stop gradient as in EfficientZeroV2
                ) * gradient_scale
                masked_ssl_loss = ssl_loss_step * step_mask
                total_ssl_loss += jnp.sum(masked_ssl_loss) / jnp.maximum(jnp.sum(step_mask), 1.0)

        # L2 regularization
        model_params = nnx.state(model, nnx.Param)
        l2_loss = losses_lib.l2_regularization(model_params, config.l2_weight)

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

    def save_checkpoint(self, force_save: bool = False):
        """Save model and optimizer state to checkpoint."""
        if self.checkpoint_manager is None:
            print("Checkpoint manager not configured. Skipping save.")
            return
            
        # Check if we should save based on frequency
        should_save = (force_save or 
                      (self.num_training_steps % self.config.checkpoint_frequency == 0 and 
                       self.num_training_steps > 0))
        
        # Log the decision
        if should_save:
            logging.info(f"SAVE_CHECKPOINT: Condition met. force_save={force_save}, num_training_steps={self.num_training_steps}, freq={self.config.checkpoint_frequency}")
        else:
            logging.info(f"SAVE_CHECKPOINT: Condition NOT met. force_save={force_save}, num_training_steps={self.num_training_steps}, freq={self.config.checkpoint_frequency}")
            return
            
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
            
            logging.info(f"Checkpoint saved at step {self.num_training_steps}")
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> bool:
        """Load model and optimizer state from checkpoint. Returns True if successful."""
        if self.checkpoint_manager is None:
            print("Checkpoint manager not configured. Skipping load.")
            return False
            
        try:
            latest_step = self.checkpoint_manager.latest_step()
            if latest_step is None:
                print("No checkpoint found to resume from.")
                return False
                
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
                step=latest_step,
                args=ocp.args.StandardRestore(target_structure)
            )
            
            # Restore model and optimizer state
            nnx.update(self.model, checkpoint_data['model'])
            nnx.update(self.optimizer, checkpoint_data['optimizer'])
            self.num_training_steps = checkpoint_data['num_training_steps']
            self._rng_key = checkpoint_data['rng_key']
            
            # Restore EMA state if available
            if (self.config.use_target_network_ema and 
                'target_model' in checkpoint_data and 
                'ema_params_state' in checkpoint_data):
                
                if self.target_model is not None:
                    nnx.update(self.target_model, checkpoint_data['target_model'])
                    self.ema_params_state = checkpoint_data['ema_params_state']
                else:
                    # Target model components not fully in checkpoint, re-initialize
                    print("Warning: EMA enabled, target model components not fully in ckpt. Re-syncing with online model.")
                    # Use proper Flax NNX state copying instead of copy.deepcopy
                    graphdef, params, batch_stats, rngs, static, ellipsis = nnx.split(
                        self.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
                    )
                    self.target_model = nnx.merge(graphdef, params, batch_stats, rngs, static, ellipsis)
                    
                    # Re-initialize EMA state
                    self.ema_updater = optax.ema(self.config.ema_decay)
                    self.ema_params_state = self.ema_updater.init(params)
            
            print(f"Checkpoint restored from step {latest_step}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return False

    def __del__(self):
        """Cleanup method to ensure CheckpointManager is properly closed."""
        if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager is not None:
            try:
                self.checkpoint_manager.close()
            except:
                pass  # Ignore errors during cleanup

# Example usage (for testing/illustration - will be in tests)
if __name__ == '__main__': # pragma: no cover
    from open_spiel.python.algorithms.muzero_jax.models.network import VisualRepresentationNetwork, MLPValuePolicyNetwork, DynamicsNetwork, RewardNetwork, MuZeroNetwork as TestMuZeroNetwork # type: ignore # pragma: no cover
    from open_spiel.python.algorithms.muzero_jax.models.layers import DownSample, ResidualBlock, FCResidualBlock, MLP # type: ignore # pragma: no cover

    key = jax.random.PRNGKey(42) # pragma: no cover
    key_model_init, key_learner_init, key_batch_gen = jax.random.split(key, 3) # pragma: no cover

    # Using ActualMuZeroNetworkConfig to define the structure for the main example
    net_config_main = ActualMuZeroNetworkConfig( # pragma: no cover
        observation_shape=(3, 96, 96), # pragma: no cover
        num_actions=18, # pragma: no cover
        num_channels=16, # pragma: no cover
        num_residual_blocks=1, # pragma: no cover
        num_fc_residual_blocks=1, # pragma: no cover
        num_hidden_units_fc=32, # pragma: no cover
        value_support_size=0, # pragma: no cover
        reward_support_size=0, # pragma: no cover
        downsample_channels=8, # pragma: no cover
        downsample_blocks=1, # pragma: no cover
        use_batch_norm=True, # pragma: no cover
        use_projection=False, # pragma: no cover
        spatial_extents=(96,96), # Should match observation if image # pragma: no cover
        use_image_observation=True, # pragma: no cover
        projection_hidden_dim=64, # Example value # pragma: no cover
        projection_head_output_dim=32, # Example value # pragma: no cover
        action_embedding_dim=16 # Example value for dummy dynamics compatibility # pragma: no cover
    )
    
    class MainVisualRepresentationNetwork(nnx.Module): # Renamed # pragma: no cover
        def __init__(self, config: ActualMuZeroNetworkConfig, *, rngs: nnx.Rngs): # pragma: no cover
            self.downsample = DownSample(config.observation_shape[0], config.downsample_channels, rngs=rngs) # Assuming obs_shape is (C,H,W) # pragma: no cover
            self.conv3x3 = nnx.Conv(in_features=config.downsample_channels, out_features=config.num_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, rngs=rngs) # pragma: no cover
            self.bn_initial = nnx.BatchNorm(config.num_channels, use_running_average=not config.use_batch_norm, rngs=rngs) if config.use_batch_norm else nnx.Identity(rngs=rngs) # pragma: no cover
            self.residuals = [ResidualBlock(config.num_channels, config.num_channels, rngs=nnx.Rngs(params=jax.random.fold_in(rngs.params(), i), dropout=jax.random.fold_in(rngs.dropout(), i))) for i in range(config.num_residual_blocks)] # pragma: no cover
        def __call__(self, x: jax.Array, training: bool): # pragma: no cover
            # Input x expected as (B, C, H, W)
            # Transpose to (B, H, W, C) for Flax NNX conv layers
            if x.shape[1] == net_config_main.observation_shape[0] and x.shape[2] == net_config_main.observation_shape[1] and x.shape[3] == net_config_main.observation_shape[2]: # pragma: no cover
                x = jnp.transpose(x, (0, 2, 3, 1)) # pragma: no cover
            x = self.downsample(x, training) # pragma: no cover
            x = self.conv3x3(x) # pragma: no cover
            x = self.bn_initial(x, use_running_average=not training) # pragma: no cover
            x = nnx.relu(x) # pragma: no cover
            for block in self.residuals: # pragma: no cover
                x = block(x, training) # pragma: no cover
            return x # pragma: no cover

    model_instance_main = TestMuZeroNetwork( # pragma: no cover
        representation_network_def=lambda config, *, rngs: MainVisualRepresentationNetwork(config, rngs=rngs), # pragma: no cover
        prediction_network_def=lambda config, *, rngs: MLPValuePolicyNetwork(config, rngs=rngs), # pragma: no cover
        dynamics_network_def=lambda config, *, rngs: DynamicsNetwork(config, rngs=rngs), # pragma: no cover
        reward_network_def=lambda config, *, rngs: RewardNetwork(config, rngs=rngs), # pragma: no cover
        projection_network_def=None, # pragma: no cover
        config=net_config_main, # Pass the ActualMuZeroNetworkConfig instance # pragma: no cover
        rngs=nnx.Rngs(params=key_model_init) # pragma: no cover
    )

    learner_config_main = MuZeroConfig( # pragma: no cover
        value_support_size=net_config_main.value_support_size, # pragma: no cover
        reward_support_size=net_config_main.reward_support_size, # pragma: no cover
        num_unroll_steps=2, # pragma: no cover
        td_steps=2, # pragma: no cover
        batch_size=2, # pragma: no cover
        l2_weight=1e-4, # pragma: no cover
        learning_rate=1e-3, # pragma: no cover
        use_projection=False, # pragma: no cover
        checkpoint_dir="/tmp/muzero_jax_test_checkpoints_main", # pragma: no cover
        checkpoint_frequency=2, # Checkpoint more frequently for test # pragma: no cover
        use_target_network_ema=True, # pragma: no cover
        ema_decay=0.95, # pragma: no cover
        resume_from_checkpoint=False # Start fresh for this example # pragma: no cover
    )
    
    optimizer_instance_main = optax.adam(learning_rate=learner_config_main.learning_rate) # pragma: no cover
    learner_main = Learner(model_instance_main, optimizer_instance_main, learner_config_main, key_learner_init) # pragma: no cover

    B_main = learner_config_main.batch_size # pragma: no cover
    K_main = learner_config_main.num_unroll_steps # pragma: no cover
    obs_shape_main = net_config_main.observation_shape # pragma: no cover
    
    dummy_batches_main = [] # pragma: no cover
    for i in range(5): # Generate a few batches # pragma: no cover
        k_batch = jax.random.fold_in(key_batch_gen, i) # pragma: no cover
        obs_batch = jax.random.uniform(k_batch, (B_main, obs_shape_main[0], obs_shape_main[1], obs_shape_main[2])) # pragma: no cover
        act_batch = jax.random.randint(k_batch, (B_main, K_main), 0, net_config_main.num_actions) # pragma: no cover
        
        val_target = jax.random.normal(k_batch, (B_main, K_main + 1)) # pragma: no cover
        rew_target = jax.random.normal(k_batch, (B_main, K_main + 1)) # pragma: no cover
        pol_target = jax.random.uniform(k_batch, (B_main, K_main + 1, net_config_main.num_actions)) # pragma: no cover
        pol_target = pol_target / jnp.sum(pol_target, axis=-1, keepdims=True) # pragma: no cover
        mask = jnp.ones((B_main, K_main + 1), dtype=jnp.float32) # pragma: no cover

        dummy_batches_main.append({ # pragma: no cover
            'observation': obs_batch, 'action': act_batch, # pragma: no cover
            'target_reward': rew_target, 'target_value': val_target, # pragma: no cover
            'target_policy': pol_target, 'game_history_mask': mask, # pragma: no cover
        })

    def dummy_replay_buffer_iterator_fn_main() -> Generator[Batch, None, None]: # pragma: no cover
        for batch_item in dummy_batches_main: # pragma: no cover
            yield batch_item # pragma: no cover

    print("Starting dummy training loop with JIT...") # pragma: no cover
    learner_main.train(dummy_replay_buffer_iterator_fn_main, num_epochs=1, steps_per_epoch=len(dummy_batches_main)) # pragma: no cover

    print("\nTrying to load from checkpoint...") # pragma: no cover
    learner_config_resume = dataclasses.replace(learner_config_main, resume_from_checkpoint=True) # pragma: no cover
    model_instance_resume = TestMuZeroNetwork( # Recreate model structure for new learner # pragma: no cover
        representation_network_def=lambda config, *, rngs: MainVisualRepresentationNetwork(config, rngs=rngs), # pragma: no cover
        prediction_network_def=lambda config, *, rngs: MLPValuePolicyNetwork(config, rngs=rngs), # pragma: no cover
        dynamics_network_def=lambda config, *, rngs: DynamicsNetwork(config, rngs=rngs), # pragma: no cover
        reward_network_def=lambda config, *, rngs: RewardNetwork(config, rngs=rngs), # pragma: no cover
        projection_network_def=None, # pragma: no cover
        config=net_config_main, # pragma: no cover
        rngs=nnx.Rngs(params=jax.random.key(1)) # Can use a different key for init, loaded state will overwrite # pragma: no cover
    )
    optimizer_instance_resume = optax.adam(learning_rate=learner_config_resume.learning_rate) # pragma: no cover
    learner_resume = Learner(model_instance_resume, optimizer_instance_resume, learner_config_resume, jax.random.key(2)) # pragma: no cover
    
    if learner_resume.num_training_steps > 0: # pragma: no cover
        print(f"Resumed successfully from step {learner_resume.num_training_steps}") # pragma: no cover
    else: # pragma: no cover
        print("Did not resume or resumed at step 0.") # pragma: no cover

    print("Done with dummy run.") # pragma: no cover 