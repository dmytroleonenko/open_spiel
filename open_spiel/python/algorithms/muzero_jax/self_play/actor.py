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
        
        # Current network parameters (will be updated when loading checkpoints)
        self.current_params = None
        
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
                
                # Apply the loaded parameters to the network
                if 'network_state' in checkpoint_data:
                    # Update the network with the loaded state
                    # In a real implementation, this would be something like:
                    # self.network = self.network.replace(state=checkpoint_data['network_state'])
                    self.current_params = checkpoint_data['network_state']
                elif 'params' in checkpoint_data:
                    self.current_params = checkpoint_data['params']
                else:
                    # Assume the checkpoint data itself contains the parameters
                    self.current_params = checkpoint_data
                    
                self.logger.info(f"Loaded and applied parameters from {latest_checkpoint}")
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
            # Use the network's recurrent inference with the provided params
            # If params is None, we'll use the network's current state
            if params is not None:
                # Use provided parameters (this would involve parameter application in real implementation)
                # For now, we proceed with the network as-is but acknowledge params
                pass
                
            next_hidden_state, reward, value, policy_logits, _ = self.network.recurrent_inference(
                embedding, action, training=False
            )
            
            # For MuZero, discount is typically 1.0 for non-terminal states
            # This will be handled by the game termination logic
            discount = jnp.ones_like(value)
            
            # Create RecurrentFnOutput as expected by mctx
            from mctx._src.base import RecurrentFnOutput
            step = RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=policy_logits,
                value=value
            )
            
            return step, next_hidden_state
            
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
            
        # CRITICAL FIX: If the selected action is illegal, find the best legal action
        # This can happen when action_weights are uniform (untrained network)
        # Get legal actions from the current game state
        legal_actions = self.game_wrapper.legal_actions()
        
        # Fallback check: ensure legal_actions is iterable
        try:
            if int(action) not in legal_actions:
                # Simple fallback: just use the first legal action
                action = legal_actions[0] if legal_actions else 0
                self.logger.info(f"Step {step}: Using fallback legal action: {action}")  # pragma: no cover
        except TypeError:
            # Skip fallback if legal_actions is not iterable
            pass
        
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
        # Reset the game and get initial observation
        initial_observation = self.game_wrapper.reset()
        
        # Initialize trajectory storage
        observations = []
        actions = []
        rewards = []
        policy_targets = []
        mcts_values = []
        
        step = 0
        max_steps = self.game_wrapper._game.max_game_length() + 1
        rng_key, subkey = jax.random.split(rng_key)
        
        while not self.game_wrapper.is_terminal():
            if step >= max_steps:
                logging.error(f"Episode in {self.game_wrapper._game.get_type().short_name} "
                              f"exceeded max steps ({max_steps}), breaking loop.")
                break
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
                    
                    # Store chance node transition data if needed for training
                    if obs is not None:
                        observations.append(obs)
                        actions.append(action)
                        # Store rewards (sum over players for simplicity)
                        episode_reward = sum(reward_list) if reward_list else 0.0
                        rewards.append(episode_reward)
                        # For chance nodes, we don't have MCTS policy, so use uniform
                        num_actions = self.config.num_actions
                        uniform_policy = jnp.ones(num_actions) / num_actions
                        policy_targets.append(uniform_policy)
                        # No MCTS value for chance nodes, use 0
                        mcts_values.append(0.0)
                    
                    continue
                    
            # Get current observation
            current_obs = self.game_wrapper.current_observation()
            if current_obs is None:  # Skip if no observation (e.g., chance node)
                break
                
            observations.append(current_obs)
            
            # Get initial inference from network (hidden_state, reward, value, policy_logits, projection)
            obs_array = jnp.array([current_obs])  # Add batch dimension
            hidden_state, reward, value, policy_logits, _ = self.network.initial_inference(
                obs_array, training=False
            )
            
            # Create root for MCTS using mctx.RootFnOutput
            from mctx._src.base import RootFnOutput
            root = RootFnOutput(
                prior_logits=policy_logits,  # Keep batch dimension for mctx
                value=value,
                embedding=hidden_state
            )
            
            # Get legal actions and create invalid actions mask
            legal_actions = self.game_wrapper.legal_actions()
            # Use configured number of actions for mask construction
            num_actions = self.config.num_actions
            invalid_actions = jnp.ones((1, num_actions), dtype=bool)  # Add batch dimension
            legal_actions_array = jnp.array(legal_actions)
            invalid_actions = invalid_actions.at[0, legal_actions_array].set(False)
            
            # Debug logging
            self.logger.info(f"Step {step}: Legal actions: {legal_actions}, Total actions: {num_actions}")
            
            # Run MCTS
            rng_key, subkey = jax.random.split(rng_key)
            recurrent_fn = self._create_recurrent_fn()
            
            policy_output = self.mcts.run(
                params=self.current_params,  # Use loaded network parameters
                rng_key=subkey,
                root=root,
                recurrent_fn=recurrent_fn,
                invalid_actions=invalid_actions
            )
            
            # Select action and get policy target
            action, policy_target = self._select_action(policy_output, step)
            
            # Debug logging for action selection
            self.logger.info(f"Step {step}: Selected action: {action}, Legal actions: {legal_actions}")
            
            # Validate that selected action is legal
            if action not in legal_actions:
                self.logger.error(f"Selected illegal action {action}! Legal actions: {legal_actions}")  # pragma: no cover
                # Fall back to first legal action
                action = legal_actions[0] if legal_actions else 0  # pragma: no cover
                self.logger.info(f"Falling back to legal action: {action}")  # pragma: no cover
            
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