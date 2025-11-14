#!/usr/bin/env python3
"""
Main orchestration script for MuZero JAX implementation.

This script coordinates the training process between actors (self-play) and learners,
manages configuration through Hydra, and provides logging through Weights & Biases.

Usage:
    python run_muzero_jax.py                     # Use default config
    python run_muzero_jax.py game=chess         # Override game
    python run_muzero_jax.py training.learning_rate=0.01  # Override specific params
"""

import os
import sys
import time
import logging
import tempfile
import threading
import dataclasses
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# MuZero JAX imports
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig, 
    Learner, 
    create_muzero_config_for_game,
    create_network_config_from_muzero_config
)
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    TrajectoryBuffer, 
    PrioritizedTrajectoryBuffer
)
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.utils.checkpointing import (
    create_checkpoint_manager,
    get_latest_checkpoint
)
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration-specific settings."""
    sequential_training: bool = True
    training_phase_steps: int = 10
    selfplay_phase_episodes: int = 5
    max_episodes_without_training: int = 100
    max_training_steps_without_episodes: int = 50


class MuZeroOrchestrator:
    """
    Main orchestrator for MuZero JAX training.
    
    Coordinates between actors (self-play) and learners, manages TPU/GPU resource
    allocation, and handles checkpointing and logging.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the orchestrator with Hydra configuration."""
        self.config = config
        self.orchestration_config = OrchestrationConfig(
            sequential_training=config.resource_management.sequential_training,
            training_phase_steps=config.resource_management.training_phase_steps,
            selfplay_phase_episodes=config.resource_management.selfplay_phase_episodes,
        )
        
        # Set up directories
        self.setup_directories()
        
        # Set random seed
        if hasattr(config.exp_config, 'seed'):
            self.set_random_seed(config.exp_config.seed)
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize training state (before setup_components so checkpoint loading can override)
        self.training_step = 0
        self.total_episodes = 0
        self.start_time = time.time()
        
        # Initialize components (checkpoint loading may update training_step)
        self.setup_components()
        
    def setup_directories(self):
        """Set up output directories."""
        self.save_path = Path(self.config.output.save_path)
        self.checkpoint_dir = self.save_path / "checkpoints"
        self.logs_dir = self.save_path / "logs"
        
        # Create directories
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Save path: {self.save_path}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        
    def set_random_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        # JAX uses a different random system, we'll set keys as needed
        self.rng_key = jax.random.PRNGKey(seed)
        logger.info(f"Set random seed to {seed}")
        
    def setup_logging(self):
        """Set up Weights & Biases logging."""
        if self.config.wandb.enabled:
            # Create tags
            tags = list(self.config.wandb.tags) if self.config.wandb.tags else []
            tags.append(self.config.game.name)
            tags.append(self.config.exp_config.tag)
            
            # Initialize wandb
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=f"{self.config.game.name}-{self.config.exp_config.tag}",
                tags=tags,
                notes=self.config.wandb.notes,
                config=OmegaConf.to_container(self.config, resolve=True),
                dir=str(self.logs_dir)
            )
            logger.info("Initialized Weights & Biases logging")
        else:
            logger.info("Weights & Biases logging disabled")
            
    def setup_components(self):
        """Initialize all MuZero components."""
        logger.info("Setting up MuZero components...")
        
        # Create MuZero configuration from Hydra config
        self.muzero_config = self.create_muzero_config_from_hydra_config()
        
        # Initialize game wrapper to get observation shape
        self.game_wrapper = GameWrapper(self.config.game.name)
        observation_shape = self.game_wrapper.observation_shape
        num_actions = self.game_wrapper.num_distinct_actions()
        
        logger.info(f"Game: {self.config.game.name}")
        logger.info(f"Observation shape: {observation_shape}")
        logger.info(f"Number of actions: {num_actions}")
        
        # Initialize network
        self.setup_network(observation_shape, num_actions)
        
        # Initialize replay buffer
        self.setup_replay_buffer(observation_shape, num_actions)
        
        # Initialize learner
        self.setup_learner()
        
        # Initialize actors
        self.setup_actors()
        
        logger.info("All components initialized successfully")
        
    def create_muzero_config_from_hydra_config(self) -> MuZeroConfig:
        """Convert Hydra config to MuZeroConfig."""
        # Start with game-specific defaults
        muzero_config = create_muzero_config_for_game(self.config.game.name)
        
        # Override with Hydra config values
        config_overrides = {
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'training_steps': self.config.training.training_steps,
            'start_transitions': self.config.training.start_transitions,  # Add start_transitions
            'discount_factor': self.config.training.discount,  # Note: field name is discount_factor
            'num_unroll_steps': self.config.training.num_unroll_steps,
            'td_steps': self.config.training.td_steps,
            'value_loss_weight': self.config.training.value_loss_weight,
            'policy_loss_weight': self.config.training.policy_loss_weight,
            'reward_loss_weight': self.config.training.reward_loss_weight,
            'l2_weight': self.config.training.l2_regularization,  # Note: field name is l2_weight
            'num_simulations': self.config.mcts.num_simulations,  # Note: field name is num_simulations
            'temperature_init': self.config.mcts.temperature_init,
            'temperature_final': self.config.mcts.temperature_final,
            'temperature_decay_steps': self.config.mcts.temperature_decay_steps,
            'dirichlet_alpha': self.config.mcts.dirichlet_alpha,
            'explore_frac': self.config.mcts.exploration_fraction,  # Note: field name is explore_frac
        }
        
        # Filter overrides to only include fields that exist in MuZeroConfig
        valid_overrides = {}
        for key, value in config_overrides.items():
            if hasattr(muzero_config, key):
                valid_overrides[key] = value
                
        # Use dataclasses.replace to create new config with overrides
        return dataclasses.replace(muzero_config, **valid_overrides)
        
    def setup_network(self, observation_shape, num_actions):
        """Initialize the MuZero network."""
        # Create network configuration
        # OpenSpiel games are always state-based, never image-based (following EfficientZeroV2 approach)
        self.network_config = create_network_config_from_muzero_config(
            self.muzero_config, 
            observation_shape, 
            num_actions,
            use_image_observation=False,  # OpenSpiel games are state-based, not image-based
        )
        
        # Import the network definitions
        from open_spiel.python.algorithms.muzero_jax.models.network import (
            RepresentationNetwork, DynamicsNetwork, PredictionNetwork, 
            RewardNetwork, ProjectionNetwork
        )
        
        # Initialize network with random parameters
        self.rng_key, network_key = jax.random.split(self.rng_key)
        rngs = nnx.Rngs(params=network_key)
        
        self.network = MuZeroNetwork(
            representation_network_def=RepresentationNetwork,
            dynamics_network_def=DynamicsNetwork,
            prediction_network_def=PredictionNetwork,
            reward_network_def=RewardNetwork,
            projection_network_def=ProjectionNetwork if self.muzero_config.use_projection else None,
            config=self.network_config,
            rngs=rngs
        )
        
        logger.info(f"Network initialized with config: {self.network_config}")
        
    def setup_replay_buffer(self, observation_shape, num_actions):
        """Initialize the replay buffer."""
        # Determine max trajectory length from config or use default
        max_trajectory_length = getattr(self.config.replay_buffer, 'max_trajectory_length', 200)
        
        if hasattr(self.config.replay_buffer, 'priority_alpha') and self.config.replay_buffer.priority_alpha > 0:
            # Use prioritized replay buffer
            self.replay_buffer = PrioritizedTrajectoryBuffer(
                capacity=self.config.replay_buffer.capacity,
                observation_shape=observation_shape,
                num_actions=num_actions,
                alpha=self.config.replay_buffer.priority_alpha,
                max_trajectory_length=max_trajectory_length
            )
            logger.info(f"Using prioritized replay buffer with alpha={self.config.replay_buffer.priority_alpha}")
        else:
            # Use standard replay buffer
            self.replay_buffer = TrajectoryBuffer(
                capacity=self.config.replay_buffer.capacity,
                observation_shape=observation_shape,
                num_actions=num_actions,
                max_trajectory_length=max_trajectory_length
            )
            logger.info("Using standard replay buffer")
        
        logger.info(f"Replay buffer initialized: capacity={self.config.replay_buffer.capacity}, "
                   f"obs_shape={observation_shape}, num_actions={num_actions}, "
                   f"max_traj_length={max_trajectory_length}")
            
    def setup_learner(self):
        """Initialize the learner."""
        # Create learner with correct constructor arguments
        self.rng_key, learner_key = jax.random.split(self.rng_key)
        
        # Set checkpoint_dir in config for learner (MuZeroConfig is frozen, so use replace)
        learner_config = dataclasses.replace(
            self.muzero_config,
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        self.learner = Learner(
            model=self.network,
            optimizer_def=None,  # Will use default from config
            config=learner_config,
            rng_key=learner_key
        )
        
        # Try to load existing checkpoint via the learner's manager first
        checkpoint_loaded = False
        if self.learner.checkpoint_manager is not None:
            latest_step = self.learner.checkpoint_manager.latest_step()
            if latest_step is not None:
                logger.info(f"Checkpoint manager reports latest step {latest_step}, attempting restore.")
                checkpoint_loaded = self.learner.load_checkpoint()
        
        # Fallback to filesystem scan (legacy checkpoints saved without manager metadata)
        if not checkpoint_loaded:
            latest_checkpoint = get_latest_checkpoint(str(self.checkpoint_dir))
            if latest_checkpoint:
                logger.info(f"Loading checkpoint from path: {latest_checkpoint}")
                checkpoint_loaded = self.learner.load_checkpoint(latest_checkpoint)
        
        if checkpoint_loaded:
            self.training_step = self.learner.num_training_steps
            logger.info(f"Resumed from training step: {self.training_step}")
        else:
            logger.info("No existing checkpoint found, starting from scratch")
            
    def setup_actors(self):
        """Initialize the actors for self-play."""
        self.actors = []
        num_actors = self.config.actors.num_actors
        
        # Create bootstrap actor for initial trajectory generation
        from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import (
            BootstrapActor, BootstrapConfig
        )
        
        bootstrap_config = BootstrapConfig(
            num_simulations=self.muzero_config.num_simulations,
            c_puct=getattr(self.config.bootstrap, 'c_puct', 1.25),
            n_step_return=self.muzero_config.td_steps,
            discount_factor=self.muzero_config.discount_factor,
        )
        
        self.bootstrap_actor = BootstrapActor(
            game_wrapper=GameWrapper(self.config.game.name),
            replay_buffer=self.replay_buffer,
            config=bootstrap_config,
        )
        
        # Create MuZero actors for later use (when network is trained)
        for i in range(num_actors):
            actor = Actor(
                network=self.network,
                game_wrapper=GameWrapper(self.config.game.name),
                replay_buffer=self.replay_buffer,
                config=self.muzero_config,
                num_simulations=self.muzero_config.num_simulations,
                max_num_considered_actions=self.game_wrapper.num_distinct_actions(),
                gumbel_scale=1.0,
                n_step_return=self.muzero_config.td_steps,
                discount_factor=self.muzero_config.discount_factor,
            )
            self.actors.append(actor)
            
        # Track whether we're in bootstrap phase
        self.use_bootstrap = getattr(self.config.bootstrap, 'enabled', True)
        self.bootstrap_episodes_generated = 0
        self.min_bootstrap_episodes = getattr(self.config.bootstrap, 'min_episodes', 50)
        
        logger.info(f"Initialized bootstrap actor and {num_actors} MuZero actors")
        logger.info(f"Will use bootstrap actor for first {self.min_bootstrap_episodes} episodes")
        
    def run_training_phase(self) -> Dict[str, Any]:
        """Run a training phase and return metrics."""
        metrics_list = []
        steps_trained = 0
        
        for _ in range(self.orchestration_config.training_phase_steps):
            # Check if we have enough data in the buffer
            if len(self.replay_buffer) < self.muzero_config.start_transitions:
                logger.info(f"Buffer size ({len(self.replay_buffer)}) below minimum ({self.muzero_config.start_transitions}), skipping training")  # pragma: no cover
                break
                
            # Generate random key for sampling
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            
            # Sample batch from replay buffer
            if isinstance(self.replay_buffer, PrioritizedTrajectoryBuffer):
                # For prioritized replay, get trajectories, indices, and importance weights
                trajectory_list, sampled_indices, importance_weights = self.replay_buffer.sample_batch(
                    self.config.training.batch_size, 
                    rng_key=sample_key
                )
            else:
                # For standard replay, only get trajectories
                trajectory_list = self.replay_buffer.sample_batch(
                    self.config.training.batch_size,
                    rng_key=sample_key
                )
                sampled_indices = None
                importance_weights = None
            
            # Convert list of trajectories to proper batch format
            batch = self._convert_trajectories_to_batch(trajectory_list, sampled_indices, importance_weights)
            
            # Perform training step
            metrics = self.learner.train_step(batch)
            metrics_list.append(metrics)
            steps_trained += 1
            self.training_step += 1
            
            # Update priorities if using prioritized replay
            if isinstance(self.replay_buffer, PrioritizedTrajectoryBuffer) and 'priorities' in metrics and sampled_indices is not None:
                try:
                    # Extract priorities from metrics (should be computed from TD errors)
                    new_priorities = jnp.array(metrics['priorities'])
                    
                    # Ensure priorities are positive and valid
                    new_priorities = jnp.maximum(new_priorities, self.muzero_config.min_priority)
                    
                    # Update priorities in the replay buffer using JAX arrays
                    self.replay_buffer.update_priorities(sampled_indices, new_priorities)
                    
                    logger.debug(f"Updated {len(sampled_indices)} priorities, mean priority: {jnp.mean(new_priorities):.6f}")
                except Exception as e:
                    logger.warning(f"Failed to update priorities: {e}")
            
            # Log metrics
            if self.training_step % self.config.output.log_interval == 0:
                self.log_training_metrics(metrics)  # pragma: no cover
                
            # Save checkpoint
            if self.training_step % self.config.output.checkpoint_interval == 0:
                checkpoint_path = self.learner.save_checkpoint(force_save=True)
                if checkpoint_path:
                    logger.info(f"Saved checkpoint: {checkpoint_path}")  # pragma: no cover
                
            # Check if training is complete
            if self.training_step >= self.muzero_config.training_steps:
                break
                
        # Aggregate metrics
        if metrics_list:
            aggregated_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    aggregated_metrics[key] = np.mean(values)
            aggregated_metrics['steps_trained'] = steps_trained
            return aggregated_metrics
        else:
            return {'steps_trained': 0}
            
    def _convert_trajectories_to_batch(self, trajectories: List[Dict], indices: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> Dict:
        """Convert list of trajectories to batched format expected by trainer.
        
        Args:
            trajectories: List of trajectory dictionaries
            indices: Buffer indices for priority replay (optional)
            weights: Importance sampling weights for priority replay (optional)
            
        Returns:
            Batch dictionary with all necessary fields for training
        """
        if not trajectories:
            raise ValueError("Cannot create batch from empty trajectory list")
        
        # Get dimensions
        batch_size = len(trajectories)
        # Use num_unroll_steps + 1 as the expected time dimension for training
        expected_length = self.muzero_config.num_unroll_steps + 1
        observation_shape = trajectories[0]['observations'][0].shape
        num_actions = self.game_wrapper.num_distinct_actions()
        
        # Use expected_length instead of max trajectory length to match model unrolling
        max_length = expected_length
        
        # Initialize batch arrays with padding
        batch_observations = np.zeros((batch_size, max_length, *observation_shape))
        batch_actions = np.zeros((batch_size, max_length), dtype=np.int32)
        batch_target_rewards = np.zeros((batch_size, max_length))  
        batch_target_values = np.zeros((batch_size, max_length))
        batch_target_policies = np.zeros((batch_size, max_length, num_actions))
        batch_masks = np.zeros((batch_size, max_length), dtype=np.float32)
        
        # Fill batch arrays
        for i, trajectory in enumerate(trajectories):
            traj_length = len(trajectory['actions'])
            # Limit to the expected length to match model unrolling
            effective_length = min(traj_length, max_length)
            
            # Fill observations (pad with last observation if needed)
            for j in range(max_length):
                if j < len(trajectory['observations']) and j < max_length:
                    batch_observations[i, j] = trajectory['observations'][j]
                elif len(trajectory['observations']) > 0:
                    # Pad with last observation
                    batch_observations[i, j] = trajectory['observations'][min(j, len(trajectory['observations']) - 1)]
            
            # Fill actions, rewards, values, policies (only up to effective_length)
            batch_actions[i, :effective_length] = trajectory['actions'][:effective_length]
            batch_target_rewards[i, :effective_length] = trajectory['rewards'][:effective_length]
            batch_target_values[i, :effective_length] = trajectory['value_targets'][:effective_length]
            
            for j in range(effective_length):
                batch_target_policies[i, j] = trajectory['policy_targets'][j]
            
            # Set mask (1.0 for valid steps, 0.0 for padding)
            batch_masks[i, :effective_length] = 1.0
        
        # Create base batch dictionary
        batch = {
            'observation': jnp.array(batch_observations),
            'action': jnp.array(batch_actions),
            'target_reward': jnp.array(batch_target_rewards),
            'target_value': jnp.array(batch_target_values),
            'target_policy': jnp.array(batch_target_policies),
            'game_history_mask': jnp.array(batch_masks),
        }
        
        # Add priority replay fields if provided
        if indices is not None:
            batch['indices'] = jnp.array(indices)
        if weights is not None:
            batch['weights'] = jnp.array(weights)
            
        return batch

    def run_selfplay_phase(self) -> Dict[str, Any]:
        """Run a self-play phase and return metrics."""
        episodes_played = 0
        total_episode_length = 0
        
        # Determine which actor to use
        if self.use_bootstrap:
            # Check if we should transition: need both min episodes AND enough transitions for training
            if (self.bootstrap_episodes_generated >= self.min_bootstrap_episodes and 
                len(self.replay_buffer) >= self.muzero_config.start_transitions):
                logger.info(f"Transitioning from bootstrap to MuZero actors after "
                           f"{self.bootstrap_episodes_generated} bootstrap episodes and "
                           f"{len(self.replay_buffer)} transitions in buffer")
                self.use_bootstrap = False
                
        if self.use_bootstrap:
            # Use bootstrap actor for initial trajectory generation
            logger.info(f"Using bootstrap actor (plain MCTS) for episode generation "
                       f"({self.bootstrap_episodes_generated}/{self.min_bootstrap_episodes})")
            
            for _ in range(self.orchestration_config.selfplay_phase_episodes):
                # Continue bootstrap until we have enough transitions for training
                # (The transition check above will handle the switch to MuZero actors)
                    
                # Play episode with bootstrap actor
                self.rng_key, episode_key = jax.random.split(self.rng_key)
                episode_data = self.bootstrap_actor.play_episode(episode_key)
                
                # Add episode data to replay buffer
                self.replay_buffer.add_trajectory(episode_data)
                
                # Extract episode length from episode data
                episode_length = len(episode_data.get('observations', []))
                episodes_played += 1
                total_episode_length += episode_length
                self.total_episodes += 1
                self.bootstrap_episodes_generated += 1
                
                # Log episode metrics
                if self.total_episodes % self.config.output.log_interval == 0:
                    self.log_episode_metrics(episode_length)
                    
        else:
            # Use MuZero actors with trained network
            logger.info("Using MuZero actors with neural network guidance")
            
            # Update actor parameters from latest checkpoint
            for actor in self.actors:
                actor.maybe_load_latest_parameters(str(self.checkpoint_dir))
                
            for _ in range(self.orchestration_config.selfplay_phase_episodes):
                # Round-robin through actors
                actor = self.actors[episodes_played % len(self.actors)]
                
                # Play episode
                self.rng_key, episode_key = jax.random.split(self.rng_key)
                episode_data = actor.play_episode(episode_key)
                
                # Add episode data to replay buffer
                self.replay_buffer.add_trajectory(episode_data)
                
                # Extract episode length from episode data
                episode_length = len(episode_data.get('observations', []))
                episodes_played += 1
                total_episode_length += episode_length
                self.total_episodes += 1
                
                # Log episode metrics
                if self.total_episodes % self.config.output.log_interval == 0:
                    self.log_episode_metrics(episode_length)
                
        return {
            'episodes_played': episodes_played,
            'avg_episode_length': total_episode_length / max(episodes_played, 1),
            'buffer_size': len(self.replay_buffer),
            'using_bootstrap': self.use_bootstrap,
            'bootstrap_episodes_generated': self.bootstrap_episodes_generated
        }
        
    def log_training_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics."""
        log_data = {
            'training_step': self.training_step,
            'total_episodes': self.total_episodes,
            'buffer_size': len(self.replay_buffer),
            'runtime_hours': (time.time() - self.start_time) / 3600,
            **metrics
        }
        
        if self.config.wandb.enabled:
            wandb.log(log_data, step=self.training_step)
            
        logger.info(f"Step {self.training_step}: {log_data}")
        
    def log_episode_metrics(self, episode_length: int):
        """Log episode metrics."""
        log_data = {
            'episode': self.total_episodes,
            'episode_length': episode_length,
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step,
        }
        
        if self.config.wandb.enabled:
            wandb.log(log_data, step=self.training_step)
            
        logger.info(f"Episode {self.total_episodes}: length={episode_length}, buffer_size={len(self.replay_buffer)}")
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation episodes."""
        if not self.config.evaluation.enabled:
            return {}
            
        logger.info("Running evaluation...")
        
        # Create evaluation actor with deterministic policy
        eval_game_wrapper = GameWrapper(self.config.game.name)
        eval_mcts = MCTS(
            num_simulations=self.muzero_config.num_simulations,
            max_num_considered_actions=eval_game_wrapper.num_distinct_actions(),
            gumbel_scale=1.0
        )
        eval_actor = Actor(
            network=self.network,
            mcts=eval_mcts,
            game_wrapper=eval_game_wrapper,
            replay_buffer=None,  # Don't add to replay buffer
            config=self.muzero_config,
            n_step_return=self.muzero_config.td_steps,
            discount_factor=self.muzero_config.discount_factor,
        )
        
        # Load latest parameters
        eval_actor.maybe_load_latest_parameters(str(self.checkpoint_dir))
        
        # Run evaluation episodes
        episode_lengths = []
        episode_rewards = []
        
        for _ in range(self.config.evaluation.num_episodes):
            self.rng_key, eval_key = jax.random.split(self.rng_key)
            episode_data = eval_actor.play_episode(eval_key)
            episode_length = len(episode_data.get('observations', []))
            episode_lengths.append(episode_length)
            # Note: episode rewards would need to be tracked separately
            
        # Compute evaluation metrics
        eval_metrics = {
            'eval_avg_episode_length': np.mean(episode_lengths),
            'eval_std_episode_length': np.std(episode_lengths),
            'eval_min_episode_length': np.min(episode_lengths),
            'eval_max_episode_length': np.max(episode_lengths),
        }
        
        logger.info(f"Evaluation results: {eval_metrics}")  # pragma: no cover
        
        if self.config.wandb.enabled:
            wandb.log(eval_metrics, step=self.training_step)  # pragma: no cover
            
        return eval_metrics
        
    def should_continue_training(self) -> bool:
        """Check if training should continue."""
        return self.training_step < self.muzero_config.training_steps
        
    def run(self):
        """Main training loop."""
        logger.info("Starting MuZero JAX training...")
        logger.info(f"Configuration: {OmegaConf.to_yaml(self.config)}")
        
        try:
            while self.should_continue_training():
                if self.orchestration_config.sequential_training:
                    # Sequential mode: alternate between self-play and training
                    
                    # Self-play phase
                    selfplay_metrics = self.run_selfplay_phase()
                    logger.info(f"Self-play phase completed: {selfplay_metrics}")
                    
                    # Training phase (only if we have enough data)
                    if len(self.replay_buffer) >= self.muzero_config.start_transitions:
                        training_metrics = self.run_training_phase()
                        logger.info(f"Training phase completed: {training_metrics}")
                    else:
                        logger.info(f"Skipping training phase, buffer size: {len(self.replay_buffer)}")
                        
                else:
                    # Concurrent mode: run both simultaneously
                    # This would require threading or multiprocessing
                    # For now, fall back to sequential mode
                    logger.warning("Concurrent mode not implemented, falling back to sequential")
                    self.orchestration_config.sequential_training = True
                    continue
                    
                # Run evaluation periodically
                if (self.training_step % self.config.evaluation.interval == 0 and 
                    self.training_step > 0):
                    self.run_evaluation()
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.cleanup()
            
        logger.info("Training completed!")
        
    def cleanup(self):
        """Clean up resources."""
        # Save final checkpoint
        if hasattr(self, 'learner'):
            final_checkpoint = self.learner.save_checkpoint(force_save=True)
            if final_checkpoint:
                logger.info(f"Saved final checkpoint: {final_checkpoint}")
            
        # Close wandb
        if self.config.wandb.enabled:
            wandb.finish()
            
        logger.info("Cleanup completed")


def setup_jax_environment(config: DictConfig):
    """Set up JAX environment and device configuration."""
    # Configure JAX based on config
    if config.resource_management.device == "cpu":
        jax.config.update('jax_platform_name', 'cpu')
        logger.info("Configured JAX to use CPU")
    elif config.resource_management.device == "gpu":
        # JAX will automatically use GPU if available
        logger.info("JAX will use GPU if available")
    elif config.resource_management.device == "tpu":
        # JAX will automatically use TPU if available
        logger.info("JAX will use TPU if available")
    else:  # auto
        logger.info("JAX will automatically select best available device")
        
    # Log available devices
    devices = jax.devices()
    logger.info(f"Available JAX devices: {devices}")
    logger.info(f"JAX default backend: {jax.default_backend()}")


@hydra.main(version_base="1.1", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point."""
    logger.info("Starting MuZero JAX orchestration")  # pragma: no cover
    logger.info(f"Working directory: {os.getcwd()}")  # pragma: no cover
    
    # Set up JAX environment
    setup_jax_environment(config)
    
    # Create and run orchestrator
    orchestrator = MuZeroOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":  # pragma: no cover
    main() 
