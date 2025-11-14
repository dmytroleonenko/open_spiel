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
    support_min: float = -300.0
    support_max: float = 300.0


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
        step = 0
        max_steps = self.game_wrapper._game.max_game_length() + 1
        final_state = state

        while not state.is_terminal():
            if step >= max_steps:
                self.logger.error(f"Episode exceeded max steps ({max_steps}), breaking loop.")  # pragma: no cover
                break  # pragma: no cover
                
            # Handle chance nodes
            if state.is_chance_node():
                # Sample from chance outcomes
                chance_outcomes = state.chance_outcomes()
                if chance_outcomes:
                    rng_key, chance_key = jax.random.split(rng_key)
                    sampled_action = self._sample_chance_action(chance_key, chance_outcomes)
                    state.apply_action(sampled_action)
                    continue
            
            # Get current observation
            raw_observation = state.observation_tensor()
            if raw_observation is None:
                self.logger.warning("Encountered state with None observation tensor; aborting episode.")
                break
                
            current_obs = jnp.array(raw_observation)
            observations.append(current_obs)
            legal_actions = state.legal_actions()
            
            # Use MCTS bot to select action + visit-count policy
            policy_entries, action = self.mcts_bot.step_with_policy(state)
            
            # Build policy target from visit counts
            policy_target = self._policy_from_entries(policy_entries, legal_actions)
            policy_targets.append(policy_target)
            
            # Apply action
            state.apply_action(action)
            actions.append(action)
            
            # Get immediate reward for the reference player (default player 0)
            rewards_vector = state.rewards()
            reward_value = float(rewards_vector[0]) if rewards_vector else 0.0
            rewards.append(reward_value)
            
            step += 1
            final_state = state
            
        # Compute value targets using n-step returns
        returns = final_state.returns() if final_state is not None else []
        final_value = float(returns[0]) if returns else 0.0
        value_targets = self._compute_value_targets(rewards, final_value)
        
        self.logger.info(f"Bootstrap episode completed: {len(actions)} steps")
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'policy_targets': policy_targets,
            'value_targets': value_targets
        }
    
    def _policy_from_entries(
        self,
        policy_entries: List[Tuple[int, float]],
        legal_actions: List[int],
    ) -> jnp.ndarray:
        """Convert mctx visit-count entries into a dense probability vector."""
        num_actions = self.game_wrapper.num_distinct_actions()
        policy = jnp.zeros(num_actions)

        if policy_entries:
            total = sum(max(prob, 0.0) for _, prob in policy_entries)
            if total > 0:
                for action, prob in policy_entries:
                    if 0 <= action < num_actions:
                        policy = policy.at[action].set(float(prob) / total)
                return policy

        # Fallback to uniform distribution over legal actions if entries are empty/invalid
        if legal_actions:
            weight = 1.0 / len(legal_actions)
            for action in legal_actions:
                if 0 <= action < num_actions:
                    policy = policy.at[action].set(weight)
            return policy

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

    def _sample_chance_action(self, rng_key: jax.Array, chance_outcomes: List[Tuple[int, float]]) -> int:
        """Sample a chance outcome using the provided RNG key."""
        outcomes, probs = zip(*chance_outcomes)
        probs_array = jnp.array(probs, dtype=jnp.float32)
        total_prob = jnp.sum(probs_array)
        if total_prob <= 0:
            probs_array = jnp.ones_like(probs_array) / len(probs_array)
        else:
            probs_array = probs_array / total_prob

        idx = int(jax.random.choice(rng_key, len(outcomes), p=probs_array))
        return int(outcomes[idx])
    
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
        
        for episode in range(num_episodes):  # pragma: no cover
            rng_key, subkey = jax.random.split(rng_key)
            
            try:
                # Play an episode
                trajectory = self.play_episode(subkey)
                
                # Add trajectory to replay buffer
                self.replay_buffer.add_trajectory(trajectory)  # pragma: no cover
                
                # Collect statistics
                episode_length = len(trajectory['actions'])  # pragma: no cover
                total_reward = sum(trajectory['rewards'])  # pragma: no cover
                
                episode_lengths.append(episode_length)  # pragma: no cover
                total_rewards.append(total_reward)  # pragma: no cover
                
                self.logger.info(
                    f"Bootstrap episode {episode + 1}/{num_episodes}, "
                    f"length: {episode_length}, reward: {total_reward:.2f}"  # pragma: no cover
                )
                               
            except Exception as e:  # pragma: no cover
                self.logger.error(f"Error in bootstrap episode {episode + 1}: {e}")  # pragma: no cover
                continue  # pragma: no cover
        
        return {
            'episodes_played': len(episode_lengths),
            'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'avg_episode_reward': np.mean(total_rewards) if total_rewards else 0,
            'buffer_size': len(self.replay_buffer)
        }  # pragma: no cover 
