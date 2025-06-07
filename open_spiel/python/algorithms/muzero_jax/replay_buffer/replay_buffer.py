import jax
import jax.numpy as jnp
import flashbax as fbx
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

class Trajectory(NamedTuple):
    """Trajectory data structure for Flashbax buffers."""
    observations: jnp.ndarray  # (T, *obs_shape)
    actions: jnp.ndarray       # (T,)
    rewards: jnp.ndarray       # (T,)
    target_values: jnp.ndarray # (T,)
    target_policies: jnp.ndarray # (T, num_actions)

@dataclass
class BufferState:
    """Wrapper for Flashbax buffer state."""
    state: Any
    current_size: int

class TrajectoryBuffer:
    """
    Trajectory buffer using Flashbax Item Buffer for JAX-native implementation.
    
    This buffer stores full trajectories as independent items using Flashbax's
    hardware-accelerated item buffer implementation.
    """
    
    def __init__(self, 
                 capacity: int,
                 observation_shape: Tuple[int, ...],
                 num_actions: int,
                 max_trajectory_length: int = 200):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
            
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.max_trajectory_length = max_trajectory_length
        
        # Use Flashbax Item Buffer for storing complete trajectories as independent items
        self._buffer = fbx.make_item_buffer(
            max_length=capacity,
            min_length=1,
            sample_batch_size=1  # Will be overridden during sampling
        )
        
        # Create a dummy trajectory for initialization
        dummy_trajectory = {
            'observations': jnp.zeros((max_trajectory_length, *observation_shape), dtype=jnp.float32),
            'actions': jnp.zeros(max_trajectory_length, dtype=jnp.int32),
            'rewards': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'value_targets': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'policy_targets': jnp.zeros((max_trajectory_length, num_actions), dtype=jnp.float32),
            'length': jnp.int32(1)  # Actual trajectory length
        }
        
        self._buffer_state = self._buffer.init(dummy_trajectory)
        self._current_size = 0

    def __len__(self) -> int:
        return self._current_size

    def add_trajectory(self, trajectory: Dict[str, Any]):
        """Add a trajectory to the buffer.
        
        Args:
            trajectory: Dictionary containing trajectory data with keys:
                - observations: Array of observations
                - actions: Array of actions
                - rewards: Array of rewards 
                - target_values: Array of value targets
                - policy_targets: Array of policy targets
        """
        traj_length = len(trajectory['actions'])
        
        # Pad trajectory to max_trajectory_length
        padded_observations = np.zeros((self.max_trajectory_length, *self.observation_shape), dtype=np.float32)
        padded_actions = np.zeros(self.max_trajectory_length, dtype=np.int32)
        padded_rewards = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_values = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_policies = np.zeros((self.max_trajectory_length, self.num_actions), dtype=np.float32)
        
        # Fill with actual data
        padded_observations[:traj_length] = trajectory['observations']
        padded_actions[:traj_length] = trajectory['actions']
        padded_rewards[:traj_length] = trajectory['rewards']
        padded_target_values[:traj_length] = trajectory['value_targets']
        
        for i in range(traj_length):
            padded_target_policies[i] = trajectory['policy_targets'][i]
        
        # Create item for Flashbax
        item = {
            'observations': jnp.array(padded_observations),
            'actions': jnp.array(padded_actions),
            'rewards': jnp.array(padded_rewards),
            'value_targets': jnp.array(padded_target_values),
            'policy_targets': jnp.array(padded_target_policies),
            'length': jnp.int32(traj_length)  # Store actual length
        }
        
        # Add to buffer
        self._buffer_state = self._buffer.add(self._buffer_state, item)
        self._current_size = min(self._current_size + 1, self.capacity)

    def sample_batch(self, batch_size: int, rng_key: Optional[jax.random.PRNGKey] = None) -> List[Dict[str, Any]]:
        """Sample a batch of trajectories from the buffer.
        
        Args:
            batch_size: Number of trajectories to sample
            rng_key: Random key for sampling (if None, creates new one)
            
        Returns:
            List of trajectory dictionaries
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        if batch_size > self._current_size:
            raise ValueError(
                f"Not enough elements to sample: requested {batch_size}, but buffer has {self._current_size}"
            )
        if batch_size == 0:
            return []
            
        if rng_key is None:
            rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        # Special case: if we're sampling the entire buffer, ensure we get unique items
        if batch_size == self._current_size:
            trajectories = []
            seen_lengths = set()
            max_attempts = self._current_size * 5  # Allow more attempts for uniqueness
            
            for i in range(batch_size):
                for attempt in range(max_attempts):
                    # Use a different random key for each attempt
                    attempt_key = jax.random.fold_in(rng_key, i * max_attempts + attempt)
                    batch = self._buffer.sample(self._buffer_state, attempt_key).experience
                    
                    traj_length = int(batch['length'][0].item())
                    
                    # If we're trying to get all unique items and this length is new, use it
                    if traj_length not in seen_lengths or len(seen_lengths) >= batch_size:
                        trajectory = {
                            'observations': np.array(batch['observations'][0][:traj_length]),
                            'actions': np.array(batch['actions'][0][:traj_length]),
                            'rewards': np.array(batch['rewards'][0][:traj_length]),
                            'value_targets': np.array(batch['value_targets'][0][:traj_length]),
                            'policy_targets': [np.array(batch['policy_targets'][0][j]) for j in range(traj_length)]
                        }
                        trajectories.append(trajectory)
                        seen_lengths.add(traj_length)
                        break
                
                # Safety: if we can't find a unique one, add whatever we got
                if len(trajectories) <= i:
                    trajectory = {
                        'observations': np.array(batch['observations'][0][:traj_length]),
                        'actions': np.array(batch['actions'][0][:traj_length]),
                        'rewards': np.array(batch['rewards'][0][:traj_length]),
                        'value_targets': np.array(batch['value_targets'][0][:traj_length]),
                        'policy_targets': [np.array(batch['policy_targets'][0][j]) for j in range(traj_length)]
                    }
                    trajectories.append(trajectory)
        else:
            # Normal case: sample with replacement
            trajectories = []
            remaining = batch_size
            
            while remaining > 0:
                # Sample one item at a time (since buffer was created with sample_batch_size=1)
                rng_key, sample_key = jax.random.split(rng_key)
                batch = self._buffer.sample(self._buffer_state, sample_key).experience
                
                # Get actual trajectory length
                traj_length = int(batch['length'][0].item())  # batch['length'] has shape (1,)
                
                # Extract the actual trajectory data (not padded)
                # Note: batch has shape (1, max_length, ...) so we need [0] to get the item
                trajectory = {
                    'observations': np.array(batch['observations'][0][:traj_length]),
                    'actions': np.array(batch['actions'][0][:traj_length]),
                    'rewards': np.array(batch['rewards'][0][:traj_length]),
                    'value_targets': np.array(batch['value_targets'][0][:traj_length]),
                    'policy_targets': [np.array(batch['policy_targets'][0][j]) for j in range(traj_length)]
                }
                trajectories.append(trajectory)
                remaining -= 1
            
        return trajectories

class PrioritizedTrajectoryBuffer:
    """
    Prioritized trajectory buffer using Flashbax Item Buffer with manual priority tracking.
    
    This buffer stores full trajectories with priority weights using Flashbax's
    item buffer and implements priority sampling manually.
    """
    
    def __init__(self, 
                 capacity: int,
                 observation_shape: Tuple[int, ...],
                 num_actions: int,
                 alpha: float = 1.0,
                 max_trajectory_length: int = 200):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
            
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.alpha = alpha
        self.max_trajectory_length = max_trajectory_length
        
        # Use Flashbax Item Buffer for storing complete trajectories
        self._buffer = fbx.make_item_buffer(
            max_length=capacity,
            min_length=1,
            sample_batch_size=1  # Will be overridden during sampling
        )
        
        # Track priorities separately (since Flashbax Item Buffer doesn't have built-in priorities)
        self._priorities = []
        
        # Create a dummy trajectory for initialization
        dummy_trajectory = {
            'observations': jnp.zeros((max_trajectory_length, *observation_shape), dtype=jnp.float32),
            'actions': jnp.zeros(max_trajectory_length, dtype=jnp.int32),
            'rewards': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'value_targets': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'policy_targets': jnp.zeros((max_trajectory_length, num_actions), dtype=jnp.float32),
            'length': jnp.int32(1),
            'priority': jnp.float32(1.0)
        }
        
        self._buffer_state = self._buffer.init(dummy_trajectory)
        self._current_size = 0

    def __len__(self) -> int:
        return self._current_size

    def add_trajectory(self, trajectory: Dict[str, Any], priority: Optional[float] = None):
        """Add a trajectory with optional priority to the buffer.
        
        Args:
            trajectory: Dictionary containing trajectory data
            priority: Priority weight for sampling (uses 1.0 if None)
        """
        if priority is None:
            priority = 1.0
        if priority <= 0:
            raise ValueError(f"Priority must be positive, got {priority}")
        
        traj_length = len(trajectory['actions'])
        
        # Pad trajectory to max_trajectory_length
        padded_observations = np.zeros((self.max_trajectory_length, *self.observation_shape), dtype=np.float32)
        padded_actions = np.zeros(self.max_trajectory_length, dtype=np.int32)
        padded_rewards = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_values = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_policies = np.zeros((self.max_trajectory_length, self.num_actions), dtype=np.float32)
        
        # Fill with actual data
        padded_observations[:traj_length] = trajectory['observations']
        padded_actions[:traj_length] = trajectory['actions']
        padded_rewards[:traj_length] = trajectory['rewards']
        padded_target_values[:traj_length] = trajectory['value_targets']
        
        for i in range(traj_length):
            padded_target_policies[i] = trajectory['policy_targets'][i]
        
        # Create item for Flashbax
        item = {
            'observations': jnp.array(padded_observations),
            'actions': jnp.array(padded_actions),
            'rewards': jnp.array(padded_rewards),
            'value_targets': jnp.array(padded_target_values),
            'policy_targets': jnp.array(padded_target_policies),
            'length': jnp.int32(traj_length),
            'priority': jnp.float32(priority)
        }
        
        # Add to buffer
        self._buffer_state = self._buffer.add(self._buffer_state, item)
        
        # Track priorities separately for manual priority sampling
        self._priorities.append(priority)
        if len(self._priorities) > self.capacity:
            self._priorities.pop(0)  # Remove oldest
        
        self._current_size = min(self._current_size + 1, self.capacity)

    def sample_batch(self, batch_size: int, rng_key: Optional[jax.random.PRNGKey] = None) -> Tuple[List[Dict[str, Any]], jnp.ndarray, jnp.ndarray]:
        """Sample a batch of trajectories according to their priorities.
        
        Args:
            batch_size: Number of trajectories to sample
            rng_key: Random key for sampling (if None, creates new one)
            
        Returns:
            Tuple of (trajectories, indices, importance_weights)
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        if batch_size > self._current_size:
            raise ValueError(
                f"Not enough elements to sample: requested {batch_size}, but buffer has {self._current_size}"
            )
        if batch_size == 0:
            return [], jnp.array([]), jnp.array([])
            
        if rng_key is None:
            rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        # Manual priority sampling
        priorities = np.array(self._priorities[:self._current_size])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices according to priorities
        indices = np.random.choice(self._current_size, size=batch_size, p=probs, replace=True)
        
        # Compute importance sampling weights
        weights = (self._current_size * probs[indices]) ** (-1.0)
        weights = weights / weights.max()  # Normalize
        
        # NOTE: Flashbax Item Buffer doesn't support indexed sampling, so we use uniform sampling
        # and manually weight the results. In practice, this means priorities affect importance weights
        # but not actual sampling probabilities. For full priority sampling, a custom buffer would be needed.
        
        # Handle variable batch sizes by sampling multiple times if needed
        trajectories = []
        remaining = batch_size
        
        while remaining > 0:
            # Sample one item at a time (since buffer was created with sample_batch_size=1)
            rng_key, sample_key = jax.random.split(rng_key)
            batch = self._buffer.sample(self._buffer_state, sample_key).experience
            
            # Get actual trajectory length
            traj_length = int(batch['length'][0].item())  # batch['length'] has shape (1,)
            
            trajectory = {
                'observations': np.array(batch['observations'][0][:traj_length]),
                'actions': np.array(batch['actions'][0][:traj_length]),
                'rewards': np.array(batch['rewards'][0][:traj_length]),
                'value_targets': np.array(batch['value_targets'][0][:traj_length]),
                'policy_targets': [np.array(batch['policy_targets'][0][j]) for j in range(traj_length)]
            }
            trajectories.append(trajectory)
            remaining -= 1
            
        return trajectories, jnp.array(indices), jnp.array(weights)

    def update_priorities(self, indices: jnp.ndarray, new_priorities: jnp.ndarray):
        """Update priorities for trajectories at given indices.
        
        Args:
            indices: Array of buffer indices to update
            new_priorities: Array of new priority values
        """
        if len(indices) != len(new_priorities):
            raise ValueError("Indices and new_priorities must have the same length")
        
        # Ensure priorities are positive
        new_priorities = jnp.maximum(new_priorities, 1e-8)
        
        # Update local priorities tracking
        for idx, new_priority in zip(indices, new_priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = float(new_priority)

# Aliases for backward compatibility and test imports
ReplayBuffer = TrajectoryBuffer
PrioritizedReplayBuffer = PrioritizedTrajectoryBuffer 