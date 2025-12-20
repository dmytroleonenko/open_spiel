from collections import deque
import jax
import jax.numpy as jnp
import flashbax as fbx
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Sequence
from dataclasses import dataclass


def _make_numpy_rng(rng_key: Optional[jax.random.PRNGKey]) -> np.random.Generator:
    """Convert an optional JAX PRNG key into a NumPy Generator."""
    if rng_key is None:
        return np.random.default_rng()
    key_vals = np.asarray(rng_key, dtype=np.uint32).reshape(-1)
    base_seed = int(key_vals[0])
    if key_vals.size > 1:
        base_seed ^= int(key_vals[-1]) << 1
    seed = base_seed & 0xFFFFFFFF
    if seed == 0:
        seed = 1
    return np.random.default_rng(seed)


def _resolve_value_targets(
    trajectory: Dict[str, Any], traj_length: int
) -> np.ndarray:
    """Return a float32 array of length traj_length for value targets."""
    raw = None
    for key in ("target_values", "value_targets"):
        if key in trajectory:
            raw = trajectory[key]
            break
    if raw is None:
        return np.zeros(traj_length, dtype=np.float32)
    raw_arr = np.array(raw, dtype=np.float32).reshape(-1)
    resolved = np.zeros(traj_length, dtype=np.float32)
    if raw_arr.size:
        steps = min(traj_length, raw_arr.shape[0])
        resolved[:steps] = raw_arr[:steps]
    return resolved

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
                 observation_shape: Tuple[int, ...] | None = None,
                 num_actions: int | None = None,
                 max_trajectory_length: int = 200):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
            
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.max_trajectory_length = max_trajectory_length
        # Buffer state is lazy-initialized if shapes are not yet set
        self._current_size = 0
        self._buffer = None
        self._buffer_state = None
        self._trajectory_store: Dict[int, Dict[str, np.ndarray]] = {}
        self._trajectory_order: deque[int] = deque()
        self._next_traj_id = 0
        # Immediate setup if shapes provided
        if self.observation_shape is not None and self.num_actions is not None:
            self._buffer = fbx.make_item_buffer(
                max_length=capacity,
                min_length=1,
                sample_batch_size=1
            )
            dummy_trajectory = {
                'observations': jnp.zeros((self.max_trajectory_length, *self.observation_shape), dtype=jnp.float32),
                'actions': jnp.zeros(self.max_trajectory_length, dtype=jnp.int32),
                'rewards': jnp.zeros(self.max_trajectory_length, dtype=jnp.float32),
                'value_targets': jnp.zeros(self.max_trajectory_length, dtype=jnp.float32),
                'policy_targets': jnp.zeros((self.max_trajectory_length, self.num_actions), dtype=jnp.float32),
                'target_search_value': jnp.zeros(self.max_trajectory_length, dtype=jnp.float32),
                'target_sarsa_value': jnp.zeros(self.max_trajectory_length, dtype=jnp.float32),
                'length': jnp.int32(1)
            }
            self._buffer_state = self._buffer.init(dummy_trajectory)

    def _copy_for_storage(self, trajectory: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create a NumPy-based copy of a trajectory for internal storage."""
        policy_targets = trajectory['policy_targets']
        if isinstance(policy_targets, list):
            policy_targets_array = np.stack(
                [np.array(target, dtype=np.float32) for target in policy_targets],
                axis=0
            )
        else:
            policy_targets_array = np.array(policy_targets, dtype=np.float32)

        # Handle optional fields with fallbacks
        target_search_value = trajectory.get('target_search_value', trajectory['value_targets'])
        target_sarsa_value = trajectory.get('target_sarsa_value', trajectory['value_targets'])

        return {
            'observations': np.array(trajectory['observations'], dtype=np.float32),
            'actions': np.array(trajectory['actions'], dtype=np.int32),
            'rewards': np.array(trajectory['rewards'], dtype=np.float32),
            'value_targets': np.array(trajectory['value_targets'], dtype=np.float32),
            'policy_targets': policy_targets_array,
            'target_search_value': np.array(target_search_value, dtype=np.float32),
            'target_sarsa_value': np.array(target_sarsa_value, dtype=np.float32),
            'length': int(len(trajectory['actions']))
        }

    def _copy_for_return(self, stored: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Return a deepcopy-style version suitable for callers/tests."""
        return {
            'observations': np.array(stored['observations'], copy=True),
            'actions': np.array(stored['actions'], copy=True),
            'rewards': np.array(stored['rewards'], copy=True),
            'value_targets': np.array(stored['value_targets'], copy=True),
            'policy_targets': [
                np.array(step, copy=True) for step in stored['policy_targets']
            ],
            'target_search_value': np.array(stored.get('target_search_value', stored['value_targets']), copy=True),
            'target_sarsa_value': np.array(stored.get('target_sarsa_value', stored['value_targets']), copy=True),
        }

    def _store_trajectory(self, trajectory: Dict[str, Any]) -> int:
        """Insert the trajectory into the in-memory store and return its ID."""
        stored = self._copy_for_storage(trajectory)
        traj_id = self._next_traj_id
        self._next_traj_id += 1
        if len(self._trajectory_order) >= self.capacity:
            old_id = self._trajectory_order.popleft()
            self._trajectory_store.pop(old_id, None)
        self._trajectory_order.append(traj_id)
        self._trajectory_store[traj_id] = stored
        self._current_size = len(self._trajectory_store)
        return traj_id

    def _sample_ids(self, batch_size: int, rng_key: Optional[jax.random.PRNGKey]) -> np.ndarray:
        """Sample trajectory IDs uniformly."""
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        available = len(self._trajectory_order)
        if batch_size > available:
            raise ValueError(
                f"Not enough elements to sample: requested {batch_size}, but buffer has {available}"
            )
        if batch_size == 0:
            return np.array([], dtype=np.int64)
        ordered_ids = np.array(self._trajectory_order, dtype=np.int64)
        if batch_size == available:
            return ordered_ids.copy()
        rng = _make_numpy_rng(rng_key)
        # Sample without replacement to avoid duplicates when possible
        replace = batch_size > available
        return rng.choice(ordered_ids, size=batch_size, replace=replace)

    def __len__(self) -> int:
        return self._current_size

    def add_trajectory(self, trajectory: Dict[str, Any]):
        """Add a trajectory to the buffer.
        
        Args:
            trajectory: Dictionary containing trajectory data with keys:
                - observations: Array of observations
                - actions: Array of actions
                - rewards: Array of rewards 
                - target_values/value_targets: Array of value targets (optional; defaults to zeros)
                - policy_targets: Array of policy targets
        """
        if self._buffer is None:
            # Infer shapes if not provided, convert to numpy for list inputs
            if self.observation_shape is None:
                first_obs = trajectory['observations'][0]
                arr_obs = np.array(first_obs)
                self.observation_shape = tuple(arr_obs.shape)
            if self.num_actions is None:
                first_policy = trajectory['policy_targets'][0]
                arr_pol = np.array(first_policy)
                # Infer number of actions from last dimension
                self.num_actions = arr_pol.shape[-1]
            # Initialize item buffer now that shapes known
            self._buffer = fbx.make_item_buffer(
                max_length=self.capacity,
                min_length=1,
                sample_batch_size=1
            )
            dummy_trajectory = {
                'observations': jnp.zeros((self.max_trajectory_length, *self.observation_shape), dtype=jnp.float32),
                'actions': jnp.zeros(self.max_trajectory_length, dtype=jnp.int32),
                'rewards': jnp.zeros(self.max_trajectory_length, dtype=jnp.float32),
                'value_targets': jnp.zeros(self.max_trajectory_length, dtype=jnp.float32),
                'policy_targets': jnp.zeros((self.max_trajectory_length, self.num_actions), dtype=jnp.float32),
                'length': jnp.int32(1)
            }
            self._buffer_state = self._buffer.init(dummy_trajectory)
        traj_length = len(trajectory['actions'])
        resolved_target_values = _resolve_value_targets(trajectory, traj_length)

        # Handle optional fields with fallbacks
        target_search_value = trajectory.get('target_search_value', resolved_target_values)
        target_sarsa_value = trajectory.get('target_sarsa_value', resolved_target_values)

        self._store_trajectory({
            'observations': trajectory['observations'],
            'actions': trajectory['actions'],
            'rewards': trajectory['rewards'],
            'value_targets': resolved_target_values,
            'policy_targets': trajectory['policy_targets'],
            'target_search_value': target_search_value,
            'target_sarsa_value': target_sarsa_value
        })
        
        # Pad trajectory to max_trajectory_length
        padded_observations = np.zeros((self.max_trajectory_length, *self.observation_shape), dtype=np.float32)
        padded_actions = np.zeros(self.max_trajectory_length, dtype=np.int32)
        padded_rewards = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_values = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_policies = np.zeros((self.max_trajectory_length, self.num_actions), dtype=np.float32)
        padded_target_search_value = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_sarsa_value = np.zeros(self.max_trajectory_length, dtype=np.float32)
        
        # Fill with actual data
        padded_observations[:traj_length] = trajectory['observations']
        padded_actions[:traj_length] = trajectory['actions']
        padded_rewards[:traj_length] = trajectory['rewards']
        padded_target_values[:traj_length] = resolved_target_values
        padded_target_search_value[:traj_length] = target_search_value
        padded_target_sarsa_value[:traj_length] = target_sarsa_value
        
        for i in range(traj_length):
            padded_target_policies[i] = trajectory['policy_targets'][i]
        
        # Create item for Flashbax
        item = {
            'observations': jnp.array(padded_observations),
            'actions': jnp.array(padded_actions),
            'rewards': jnp.array(padded_rewards),
            'value_targets': jnp.array(padded_target_values),
            'policy_targets': jnp.array(padded_target_policies),
            'target_search_value': jnp.array(padded_target_search_value),
            'target_sarsa_value': jnp.array(padded_target_sarsa_value),
            'length': jnp.int32(traj_length)  # Store actual length
        }
        
        # Add to buffer
        self._buffer_state = self._buffer.add(self._buffer_state, item)
        self._current_size = len(self._trajectory_store)

    def sample_batch(self, batch_size: int, rng_key: Optional[jax.random.PRNGKey] = None) -> List[Dict[str, Any]]:
        """Sample a batch of trajectories from the buffer."""
        trajectories, _ = self.sample_batch_with_ids(batch_size, rng_key=rng_key)
        return trajectories

    def sample_batch_with_ids(
        self, batch_size: int, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Sample trajectories and return their internal IDs."""
        sampled_ids = self._sample_ids(batch_size, rng_key)
        batch = [self._copy_for_return(self._trajectory_store[int(traj_id)]) for traj_id in sampled_ids]
        return batch, sampled_ids

    def get_trajectory(self, trajectory_id: int) -> Dict[str, Any]:
        """Return a copy of a stored trajectory by ID."""
        if trajectory_id not in self._trajectory_store:
            raise KeyError(f"Trajectory id {trajectory_id} not found in buffer")
        return self._copy_for_return(self._trajectory_store[trajectory_id])

    def iter_trajectories(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Return all stored trajectories as (id, trajectory) pairs."""
        return [(traj_id, self._copy_for_return(self._trajectory_store[traj_id])) for traj_id in self._trajectory_order]

    def update_trajectory_targets(
        self,
        trajectory_id: int,
        *,
        policy_targets: Optional[np.ndarray] = None,
        value_targets: Optional[np.ndarray] = None,
    ):
        """Update policy/value targets for a stored trajectory."""
        if trajectory_id not in self._trajectory_store:
            raise KeyError(f"Trajectory id {trajectory_id} not found in buffer")
        stored = self._trajectory_store[trajectory_id]
        if policy_targets is not None:
            policy_arr = np.array(policy_targets, dtype=np.float32)
            steps = min(policy_arr.shape[0], stored['policy_targets'].shape[0])
            stored['policy_targets'][:steps] = policy_arr[:steps]
        if value_targets is not None:
            value_arr = np.array(value_targets, dtype=np.float32).reshape(-1)
            steps = min(value_arr.shape[0], stored['value_targets'].shape[0])
            stored['value_targets'][:steps] = value_arr[:steps]

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
        
        # Create a dummy trajectory for initialization
        dummy_trajectory = {
            'observations': jnp.zeros((max_trajectory_length, *observation_shape), dtype=jnp.float32),
            'actions': jnp.zeros(max_trajectory_length, dtype=jnp.int32),
            'rewards': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'value_targets': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'policy_targets': jnp.zeros((max_trajectory_length, num_actions), dtype=jnp.float32),
            'target_search_value': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'target_sarsa_value': jnp.zeros(max_trajectory_length, dtype=jnp.float32),
            'length': jnp.int32(1),
            'priority': jnp.float32(1.0)
        }
        
        self._buffer_state = self._buffer.init(dummy_trajectory)
        self._current_size = 0
        self._trajectory_store: Dict[int, Dict[str, np.ndarray]] = {}
        self._trajectory_order: deque[int] = deque()
        self._id_priorities: Dict[int, float] = {}
        self._next_traj_id = 0

    def __len__(self) -> int:
        return self._current_size

    def _copy_for_storage(self, trajectory: Dict[str, Any]) -> Dict[str, np.ndarray]:
        policy_targets = trajectory['policy_targets']
        if isinstance(policy_targets, list):
            policy_targets_array = np.stack(
                [np.array(target, dtype=np.float32) for target in policy_targets],
                axis=0
            )
        else:
            policy_targets_array = np.array(policy_targets, dtype=np.float32)

        # Handle optional fields with fallbacks
        target_search_value = trajectory.get('target_search_value', trajectory['value_targets'])
        target_sarsa_value = trajectory.get('target_sarsa_value', trajectory['value_targets'])

        return {
            'observations': np.array(trajectory['observations'], dtype=np.float32),
            'actions': np.array(trajectory['actions'], dtype=np.int32),
            'rewards': np.array(trajectory['rewards'], dtype=np.float32),
            'value_targets': np.array(trajectory['value_targets'], dtype=np.float32),
            'policy_targets': policy_targets_array,
            'target_search_value': np.array(target_search_value, dtype=np.float32),
            'target_sarsa_value': np.array(target_sarsa_value, dtype=np.float32),
            'length': int(len(trajectory['actions']))
        }

    def _copy_for_return(self, stored: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            'observations': np.array(stored['observations'], copy=True),
            'actions': np.array(stored['actions'], copy=True),
            'rewards': np.array(stored['rewards'], copy=True),
            'value_targets': np.array(stored['value_targets'], copy=True),
            'policy_targets': [
                np.array(step, copy=True) for step in stored['policy_targets']
            ],
            'target_search_value': np.array(stored.get('target_search_value', stored['value_targets']), copy=True),
            'target_sarsa_value': np.array(stored.get('target_sarsa_value', stored['value_targets']), copy=True),
        }

    def _store_trajectory(self, trajectory: Dict[str, Any], priority: float) -> int:
        stored = self._copy_for_storage(trajectory)
        traj_id = self._next_traj_id
        self._next_traj_id += 1
        if len(self._trajectory_order) >= self.capacity:
            old_id = self._trajectory_order.popleft()
            self._trajectory_store.pop(old_id, None)
            self._id_priorities.pop(old_id, None)
        self._trajectory_order.append(traj_id)
        self._trajectory_store[traj_id] = stored
        self._id_priorities[traj_id] = float(priority)
        self._current_size = len(self._trajectory_order)
        return traj_id

    def _sample_with_metadata(
        self, batch_size: int, rng_key: Optional[jax.random.PRNGKey]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        if batch_size > self._current_size:
            raise ValueError(
                f"Not enough elements to sample: requested {batch_size}, but buffer has {self._current_size}"
            )
        if batch_size == 0:
            empty = np.array([], dtype=np.int64)
            return [], empty, empty, jnp.array([])

        ordered_ids = np.array(self._trajectory_order, dtype=np.int64)
        priorities = np.array([self._id_priorities[idx] for idx in ordered_ids], dtype=np.float32)
        if batch_size == len(ordered_ids):
            sampled_ids = ordered_ids.copy()
            sampled_indices = np.arange(len(ordered_ids))
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            probs = priorities ** self.alpha
            probs = probs / probs.sum()
            rng = _make_numpy_rng(rng_key)
            sampled_indices = rng.choice(len(ordered_ids), size=batch_size, replace=True, p=probs)
            sampled_ids = ordered_ids[sampled_indices]
            weights = (len(ordered_ids) * probs[sampled_indices]) ** (-1.0)
            weights = weights / weights.max()

        batch = [self._copy_for_return(self._trajectory_store[int(traj_id)]) for traj_id in sampled_ids]
        return batch, sampled_ids, jnp.array(sampled_indices), jnp.array(weights)

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
        resolved_target_values = _resolve_value_targets(trajectory, traj_length)

        # Handle optional fields with fallbacks
        target_search_value = trajectory.get('target_search_value', resolved_target_values)
        target_sarsa_value = trajectory.get('target_sarsa_value', resolved_target_values)

        self._store_trajectory({
            'observations': trajectory['observations'],
            'actions': trajectory['actions'],
            'rewards': trajectory['rewards'],
            'value_targets': resolved_target_values,
            'policy_targets': trajectory['policy_targets'],
            'target_search_value': target_search_value,
            'target_sarsa_value': target_sarsa_value
        }, priority)
        
        # Pad trajectory to max_trajectory_length
        padded_observations = np.zeros((self.max_trajectory_length, *self.observation_shape), dtype=np.float32)
        padded_actions = np.zeros(self.max_trajectory_length, dtype=np.int32)
        padded_rewards = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_values = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_policies = np.zeros((self.max_trajectory_length, self.num_actions), dtype=np.float32)
        padded_target_search_value = np.zeros(self.max_trajectory_length, dtype=np.float32)
        padded_target_sarsa_value = np.zeros(self.max_trajectory_length, dtype=np.float32)
        
        # Fill with actual data
        padded_observations[:traj_length] = trajectory['observations']
        padded_actions[:traj_length] = trajectory['actions']
        padded_rewards[:traj_length] = trajectory['rewards']
        padded_target_values[:traj_length] = resolved_target_values
        padded_target_search_value[:traj_length] = target_search_value
        padded_target_sarsa_value[:traj_length] = target_sarsa_value
        
        for i in range(traj_length):
            padded_target_policies[i] = trajectory['policy_targets'][i]
        
        # Create item for Flashbax
        item = {
            'observations': jnp.array(padded_observations),
            'actions': jnp.array(padded_actions),
            'rewards': jnp.array(padded_rewards),
            'value_targets': jnp.array(padded_target_values),
            'policy_targets': jnp.array(padded_target_policies),
            'target_search_value': jnp.array(padded_target_search_value),
            'target_sarsa_value': jnp.array(padded_target_sarsa_value),
            'length': jnp.int32(traj_length),
            'priority': jnp.float32(priority)
        }
        
        # Add to buffer
        self._buffer_state = self._buffer.add(self._buffer_state, item)
        
        self._current_size = len(self._trajectory_order)

    def sample_batch(
        self, batch_size: int, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[List[Dict[str, Any]], jnp.ndarray, jnp.ndarray]:
        """Sample a batch of trajectories according to their priorities."""
        trajectories, _, indices, weights = self._sample_with_metadata(batch_size, rng_key)
        return trajectories, indices, weights

    def sample_batch_with_ids(
        self, batch_size: int, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample trajectories and expose their IDs alongside indices/weights."""
        return self._sample_with_metadata(batch_size, rng_key)

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
            idx = int(idx)
            if idx < 0 or idx >= len(self._trajectory_order):
                continue
            traj_id = self._trajectory_order[idx]
            self._id_priorities[traj_id] = float(new_priority)

    def update_priorities_by_ids(self, trajectory_ids: Sequence[int], new_priorities: Sequence[float]):
        """Update priorities referenced directly by trajectory id."""
        if len(trajectory_ids) != len(new_priorities):
            raise ValueError("trajectory_ids and new_priorities must have the same length")
        for traj_id, priority in zip(trajectory_ids, new_priorities):
            if traj_id in self._id_priorities:
                self._id_priorities[traj_id] = float(max(priority, 1e-8))

    def get_trajectory(self, trajectory_id: int) -> Dict[str, Any]:
        if trajectory_id not in self._trajectory_store:
            raise KeyError(f"Trajectory id {trajectory_id} not found in prioritized buffer")
        return self._copy_for_return(self._trajectory_store[trajectory_id])

    def priorities_snapshot(self) -> List[Tuple[int, float]]:
        """Return [(trajectory_id, priority)] in buffer order for inspection/testing."""
        return [
            (traj_id, float(self._id_priorities.get(traj_id, 0.0)))
            for traj_id in self._trajectory_order
        ]

    def priority_values(self) -> List[float]:
        """Return just the priority values in buffer order."""
        return [priority for _, priority in self.priorities_snapshot()]

    def iter_trajectories(self) -> List[Tuple[int, Dict[str, Any]]]:
        return [(traj_id, self._copy_for_return(self._trajectory_store[traj_id])) for traj_id in self._trajectory_order]

    def update_trajectory_targets(
        self,
        trajectory_id: int,
        *,
        policy_targets: Optional[np.ndarray] = None,
        value_targets: Optional[np.ndarray] = None,
    ):
        if trajectory_id not in self._trajectory_store:
            raise KeyError(f"Trajectory id {trajectory_id} not found in prioritized buffer")
        stored = self._trajectory_store[trajectory_id]
        if policy_targets is not None:
            policy_arr = np.array(policy_targets, dtype=np.float32)
            steps = min(policy_arr.shape[0], stored['policy_targets'].shape[0])
            stored['policy_targets'][:steps] = policy_arr[:steps]
        if value_targets is not None:
            value_arr = np.array(value_targets, dtype=np.float32).reshape(-1)
            steps = min(value_arr.shape[0], stored['value_targets'].shape[0])
            stored['value_targets'][:steps] = value_arr[:steps]

    def get_priority(self, trajectory_id: int) -> float:
        if trajectory_id not in self._id_priorities:
            raise KeyError(f"Trajectory id {trajectory_id} not found in prioritized buffer")
        return float(self._id_priorities[trajectory_id])

# Aliases for backward compatibility and test imports
ReplayBuffer = TrajectoryBuffer
PrioritizedReplayBuffer = PrioritizedTrajectoryBuffer 
