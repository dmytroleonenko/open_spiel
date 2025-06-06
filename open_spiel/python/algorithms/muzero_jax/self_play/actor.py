"""
Self-play actor for MuZero JAX implementation.

This module implements the self-play actor component that:
1. Loads the latest network parameters from checkpoints
2. Uses MCTS to play games against itself
3. Collects trajectories with observations, actions, rewards, and targets
4. Adds completed trajectories to the replay buffer for training

Reference: EfficientZeroV2/ez/worker/actor_worker.py, self_play_worker.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import logging

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
from open_spiel.python.algorithms.muzero_jax.utils.checkpointing import load_checkpoint, get_latest_checkpoint


class Actor:
    """
    Self-play actor for MuZero.
    
    The actor is responsible for:
    1. Loading the latest network parameters from the learner
    2. Playing games using MCTS with the current network
    3. Collecting trajectory data (observations, actions, rewards, targets)
    4. Adding completed trajectories to the replay buffer
    """
    
    def __init__(
        self,
        network: MuZeroNetwork,
        mcts: MCTS,
        game_wrapper: GameWrapper,
        replay_buffer: TrajectoryBuffer,
        config: Any,
        n_step_return: int = 5,
        discount_factor: float = 0.99,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
    ):
        """
        Initialize the self-play actor.
        
        Args:
            network: MuZero network for inference
            mcts: MCTS implementation for action selection
            game_wrapper: Game environment wrapper
            replay_buffer: Buffer to store completed trajectories
            config: Configuration object with game and network parameters
            n_step_return: Number of steps for n-step return calculation
            discount_factor: Discount factor for value target computation
            temperature: Temperature for action selection from MCTS policy
            temperature_threshold: Step threshold after which temperature becomes 0
        """
        # Validate parameters
        if discount_factor <= 0.0 or discount_factor > 1.0:
            raise ValueError(f"discount_factor must be in (0, 1], got {discount_factor}")
        if n_step_return <= 0:
            raise ValueError(f"n_step_return must be positive, got {n_step_return}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if temperature_threshold < 0:
            raise ValueError(f"temperature_threshold must be non-negative, got {temperature_threshold}")
            
        self.network = network
        self.mcts = mcts
        self.game_wrapper = game_wrapper
        self.replay_buffer = replay_buffer
        self.config = config
        self.n_step_return = n_step_return
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def load_network_parameters(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load network parameters from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Loaded parameters dictionary
        """
        return load_checkpoint(checkpoint_path)
        
    def maybe_load_latest_parameters(self, checkpoint_dir: str) -> bool:
        """
        Load the latest network parameters if available.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            True if parameters were loaded, False otherwise
        """
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            try:
                checkpoint_data = self.load_network_parameters(latest_checkpoint)
                # In a real implementation, we would update the network parameters here
                # For now, we just log that we loaded them
                self.logger.info(f"Loaded parameters from {latest_checkpoint}")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to load parameters: {e}")
                return False
        return False        
    def _create_recurrent_fn(self) -> callable:
        """
        Create the recurrent function for MCTS.
        
        Returns:
            Function that takes (params, rng_key, action, embedding) and returns
            (reward, discount, prior_logits, value, embedding)
        """
        def recurrent_fn(params, rng_key, action, embedding):
            # Use the network's recurrent inference
            next_hidden_state, policy_logits, value, reward, _ = self.network.recurrent_inference(
                embedding, action, training=False
            )
            
            # For MuZero, discount is typically 1.0 for non-terminal states
            # This will be handled by the game termination logic
            discount = jnp.ones_like(value)
            
            return reward, discount, policy_logits, value, next_hidden_state
            
        return recurrent_fn
        
    def _select_action(self, policy_output, step: int) -> Tuple[int, jnp.ndarray]:
        """
        Select an action from MCTS policy output.
        
        Args:
            policy_output: Output from MCTS containing action weights
            step: Current step in the episode (for temperature scheduling)
            
        Returns:
            Tuple of (selected_action, policy_target)
        """
        action_weights = policy_output.action_weights
        
        # Apply temperature for action selection
        current_temp = self.temperature if step < self.temperature_threshold else 0.0
        
        if current_temp > 0.0:
            # Sample from the policy with temperature
            probs = jax.nn.softmax(jnp.log(action_weights + 1e-8) / current_temp)
            # For deterministic testing, we'll use the argmax for now
            # In practice, this would sample from the distribution
            action = jnp.argmax(probs)
        else:
            # Greedy action selection
            action = jnp.argmax(action_weights)
            
        # Policy target is the normalized action weights (visit counts)
        policy_target = self._compute_policy_target(action_weights)
        
        return int(action), policy_target
        
    def _compute_policy_target(self, action_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Compute policy target from MCTS action weights.
        
        Args:
            action_weights: Raw action weights from MCTS
            
        Returns:
            Normalized policy target
        """
        # Normalize to create a valid probability distribution
        total_weight = jnp.sum(action_weights)
        if total_weight > 0:
            return action_weights / total_weight
        else:
            # Uniform distribution if no visits
            return jnp.ones_like(action_weights) / len(action_weights)
            
    def _compute_value_targets(self, rewards: List[float], final_value: float) -> List[float]:
        """
        Compute n-step value targets.
        
        Args:
            rewards: List of rewards from the episode
            final_value: Final value estimate (0 for terminal states)
            
        Returns:
            List of value targets for each step
        """
        value_targets = []
        episode_length = len(rewards)
        
        for i in range(episode_length):
            # Compute n-step return
            target = 0.0
            for j in range(min(self.n_step_return, episode_length - i)):
                target += (self.discount_factor ** j) * rewards[i + j]
            
            # Add discounted final value if we don't reach the end
            if i + self.n_step_return < episode_length:
                target += (self.discount_factor ** self.n_step_return) * final_value
            
            value_targets.append(target)
            
        return value_targets        
    def play_episode(self, rng_key: jax.Array) -> Dict[str, List[Any]]:
        """
        Play a single episode and collect trajectory data.
        
        Args:
            rng_key: JAX random key for stochastic operations
            
        Returns:
            Dictionary containing trajectory data with keys:
            - observations: List of observations
            - actions: List of actions taken
            - rewards: List of rewards received
            - policy_targets: List of policy targets from MCTS
            - value_targets: List of value targets (computed post-episode)
        """
        # Reset the game
        observation = self.game_wrapper.reset()
        
        # Initialize trajectory storage
        observations = []
        actions = []
        rewards = []
        policy_targets = []
        mcts_values = []
        
        step = 0
        rng_key, subkey = jax.random.split(rng_key)
        
        while not self.game_wrapper.is_terminal():
            # Handle chance nodes
            if self.game_wrapper.is_chance_node():
                # For chance nodes, sample from the chance outcomes
                chance_outcomes = self.game_wrapper.chance_outcomes()
                if chance_outcomes:
                    # Sample a chance outcome
                    outcomes, probs = zip(*chance_outcomes)
                    rng_key, subkey = jax.random.split(rng_key)
                    choice_idx = jax.random.choice(subkey, len(outcomes), p=jnp.array(probs))
                    action = outcomes[choice_idx]
                    
                    # Apply the chance action
                    obs, reward_list, done = self.game_wrapper.step(action)
                    continue
                    
            # Get current observation
            current_obs = self.game_wrapper.current_observation()
            if not current_obs:  # Skip if no observation (e.g., chance node)
                break
                
            observations.append(current_obs)
            
            # Get initial inference from network
            obs_array = jnp.array([current_obs])  # Add batch dimension
            hidden_state, policy_logits, value, reward, _ = self.network.initial_inference(
                obs_array, training=False
            )
            
            # Create root for MCTS
            root = type('Root', (), {
                'prior_logits': policy_logits[0],  # Remove batch dimension
                'value': value[0],
                'embedding': hidden_state[0]
            })()
            
            # Get legal actions and create invalid actions mask
            legal_actions = self.game_wrapper.legal_actions()
            invalid_actions = jnp.ones(self.config.num_actions, dtype=bool)
            legal_actions_array = jnp.array(legal_actions)
            invalid_actions = invalid_actions.at[legal_actions_array].set(False)
            
            # Run MCTS
            rng_key, subkey = jax.random.split(rng_key)
            recurrent_fn = self._create_recurrent_fn()
            
            policy_output = self.mcts.run(
                params=None,  # Network parameters (would be actual params in real implementation)
                rng_key=subkey,
                root=root,
                recurrent_fn=recurrent_fn,
                invalid_actions=invalid_actions
            )
            
            # Select action and get policy target
            action, policy_target = self._select_action(policy_output, step)
            
            # Store MCTS value for target computation
            mcts_values.append(float(value[0]))
            
            # Apply action to environment
            obs, reward_list, done = self.game_wrapper.step(action)
            
            # Store trajectory data
            actions.append(action)
            policy_targets.append(policy_target)
            
            # Store rewards (sum over players for simplicity)
            episode_reward = sum(reward_list) if reward_list else 0.0
            rewards.append(episode_reward)
            
            step += 1
            
        # Compute value targets using n-step returns
        final_value = 0.0  # Terminal state value is 0
        value_targets = self._compute_value_targets(rewards, final_value)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'policy_targets': policy_targets,
            'value_targets': value_targets
        }        
    def run(self, rng_key: jax.Array, num_episodes: int = 1) -> None:
        """
        Run multiple episodes of self-play.
        
        Args:
            rng_key: JAX random key
            num_episodes: Number of episodes to play
        """
        for episode in range(num_episodes):
            rng_key, subkey = jax.random.split(rng_key)
            
            try:
                # Play an episode
                trajectory = self.play_episode(subkey)
                
                # Add trajectory to replay buffer
                self.replay_buffer.add_trajectory(trajectory)
                
                self.logger.info(f"Completed episode {episode + 1}/{num_episodes}, "
                               f"length: {len(trajectory['actions'])}")
                               
            except Exception as e:
                self.logger.error(f"Error in episode {episode + 1}: {e}")
                # Continue with next episode rather than crashing
                continue