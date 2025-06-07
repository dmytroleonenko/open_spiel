"""Bootstrap Actor for generating initial trajectories using plain MCTS.

This actor uses traditional MCTS without neural network guidance to generate
high-quality initial trajectories before the MuZero network is trained.
"""

import logging
from typing import Dict, List, Any, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import TrajectoryBuffer


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap actor."""
    num_simulations: int = 100
    c_puct: float = 1.0
    temperature: float = 1.0
    temperature_threshold: int = 30
    n_step_return: int = 5
    discount_factor: float = 0.99


class BootstrapActor:
    """Bootstrap actor using plain MCTS for initial trajectory generation.
    
    This actor generates trajectories using traditional MCTS without neural network
    guidance. This produces higher quality initial trajectories compared to using
    an untrained MuZero network.
    """
    
    def __init__(
        self,
        game_wrapper: GameWrapper,
        replay_buffer: TrajectoryBuffer,
        config: BootstrapConfig,
    ):
        """Initialize bootstrap actor.
        
        Args:
            game_wrapper: OpenSpiel game wrapper
            replay_buffer: Replay buffer to store trajectories
            config: Bootstrap configuration
        """
        self.game_wrapper = game_wrapper
        self.replay_buffer = replay_buffer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Import OpenSpiel MCTS bot
        import pyspiel
        self.mcts_bot = pyspiel.MCTSBot(
            game=game_wrapper._game,
            evaluator=pyspiel.RandomRolloutEvaluator(1, 42),  # Single random rollout with seed
            uct_c=config.c_puct,
            max_simulations=config.num_simulations,
            max_memory_mb=1000,  # Memory limit in MB
            solve=False,
            seed=42,  # Provide a valid seed instead of None
            verbose=False
        )
    
    def play_episode(self, rng_key: jax.Array) -> Dict[str, List[Any]]:
        """Play a single episode using plain MCTS.
        
        Args:
            rng_key: JAX random key (for interface compatibility)
            
        Returns:
            Dictionary containing trajectory data
        """
        # Reset the game
        state = self.game_wrapper._game.new_initial_state()
        
        # Initialize trajectory storage
        observations = []
        actions = []
        rewards = []
        policy_targets = []
        mcts_values = []
        
        step = 0
        max_steps = self.game_wrapper._game.max_game_length() + 1
        
        while not state.is_terminal():
            if step >= max_steps:
                self.logger.error(f"Episode exceeded max steps ({max_steps}), breaking loop.")
                break
                
            # Handle chance nodes
            if state.is_chance_node():
                # Sample from chance outcomes
                chance_outcomes = state.chance_outcomes()
                if chance_outcomes:
                    outcomes, probs = zip(*chance_outcomes)
                    # Convert to numpy for sampling
                    outcomes_array = np.array(outcomes)
                    probs_array = np.array(probs)
                    choice_idx = np.random.choice(len(outcomes), p=probs_array)
                    action = outcomes_array[choice_idx]
                    
                    # Apply chance action
                    state.apply_action(action)
                    
                    # Store chance node data if needed
                    obs = jnp.array(state.observation_tensor())
                    if obs is not None:
                        observations.append(obs)
                        actions.append(action)
                        rewards.append(0.0)  # Chance nodes typically have no immediate reward
                        
                        # Uniform policy for chance nodes
                        num_actions = self.game_wrapper.num_distinct_actions()
                        uniform_policy = jnp.ones(num_actions) / num_actions
                        policy_targets.append(uniform_policy)
                        mcts_values.append(0.0)
                    
                    continue
            
            # Get current observation
            current_obs = jnp.array(state.observation_tensor())
            if current_obs is None:
                break
                
            observations.append(current_obs)
            
            # Use MCTS bot to select action
            action = self.mcts_bot.step(state)
            
            # Get MCTS policy (visit counts) for this state
            policy_target = self._get_mcts_policy(state, action)
            policy_targets.append(policy_target)
            
            # Get MCTS value estimate (simplified - could use bot's value if available)
            # For now, we'll compute this post-episode from actual returns
            mcts_values.append(0.0)  # Placeholder
            
            # Apply action
            state.apply_action(action)
            actions.append(action)
            
            # Get reward (sum over all players for simplicity)
            if state.is_terminal():
                returns = state.returns()
                episode_reward = sum(returns) if returns else 0.0
            else:
                episode_reward = 0.0
            rewards.append(episode_reward)
            
            step += 1
            
        # Compute value targets using n-step returns
        final_value = 0.0  # Terminal state value is 0
        value_targets = self._compute_value_targets(rewards, final_value)
        
        self.logger.info(f"Bootstrap episode completed: {len(actions)} steps")
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'policy_targets': policy_targets,
            'value_targets': value_targets
        }
    
    def _get_mcts_policy(self, state, selected_action: int) -> jnp.ndarray:
        """Get MCTS policy from visit counts.
        
        For plain MCTS, we approximate the policy by giving the selected action
        higher probability and distributing the rest among legal actions.
        
        Args:
            state: Current game state
            selected_action: Action selected by MCTS
            
        Returns:
            Policy target as probability distribution
        """
        num_actions = self.game_wrapper.num_distinct_actions()
        legal_actions = state.legal_actions()
        
        # Create policy target
        policy = jnp.zeros(num_actions)
        
        if legal_actions:
            if len(legal_actions) == 1:
                # Only one legal action - give it full probability
                policy = policy.at[legal_actions[0]].set(1.0)
            else:
                # Give most weight to selected action, distribute rest among legal actions
                selected_weight = 0.7
                other_weight = (1.0 - selected_weight) / (len(legal_actions) - 1)
                
                for action in legal_actions:
                    if action == selected_action:
                        policy = policy.at[action].set(selected_weight)
                    else:
                        policy = policy.at[action].set(other_weight)
        
        return policy
    
    def _compute_value_targets(self, rewards: List[float], final_value: float) -> List[float]:
        """Compute n-step value targets.
        
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
            for j in range(min(self.config.n_step_return, episode_length - i)):
                target += (self.config.discount_factor ** j) * rewards[i + j]
            
            # Add discounted final value if we don't reach the end
            if i + self.config.n_step_return < episode_length:
                target += (self.config.discount_factor ** self.config.n_step_return) * final_value
            
            value_targets.append(target)
            
        return value_targets
    
    def run(self, rng_key: jax.Array, num_episodes: int = 1) -> Dict[str, Any]:
        """Run multiple episodes of plain MCTS self-play.
        
        Args:
            rng_key: JAX random key
            num_episodes: Number of episodes to play
            
        Returns:
            Statistics about the episodes played
        """
        episode_lengths = []
        total_rewards = []
        
        for episode in range(num_episodes):
            rng_key, subkey = jax.random.split(rng_key)
            
            try:
                # Play an episode
                trajectory = self.play_episode(subkey)
                
                # Add trajectory to replay buffer
                self.replay_buffer.add_trajectory(trajectory)
                
                # Collect statistics
                episode_length = len(trajectory['actions'])
                total_reward = sum(trajectory['rewards'])
                
                episode_lengths.append(episode_length)
                total_rewards.append(total_reward)
                
                self.logger.info(f"Bootstrap episode {episode + 1}/{num_episodes}, "
                               f"length: {episode_length}, reward: {total_reward:.2f}")
                               
            except Exception as e:
                self.logger.error(f"Error in bootstrap episode {episode + 1}: {e}")
                continue
        
        return {
            'episodes_played': len(episode_lengths),
            'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'avg_episode_reward': np.mean(total_rewards) if total_rewards else 0,
            'buffer_size': len(self.replay_buffer)
        } 