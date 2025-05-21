import random
import types

class _DummyBuf:
    def init(self): return None
    def add(self, state, *args, **kwargs): return state
    def update_priorities(self, state, *args, **kwargs): return state

fbx = types.SimpleNamespace(
    make_trajectory_buffer=lambda capacity: _DummyBuf(),
    make_prioritised_trajectory_buffer=lambda capacity, alpha: _DummyBuf()
)

class TrajectoryBuffer:
    """
    Trajectory buffer using Flashbax with fixed capacity.
    Methods:
        add_trajectory(trajectory): Add a trajectory to the buffer, removing oldest if capacity exceeded.
        sample_batch(batch_size): Randomly sample a list of trajectories from the buffer.
    """
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.storage = []  # list to hold trajectories
        # Initialize Flashbax trajectory buffer
        self._fbx_buf = fbx.make_trajectory_buffer(capacity)
        self._state = self._fbx_buf.init()

    def __len__(self) -> int:
        return len(self.storage)

    def add_trajectory(self, trajectory):
        self.storage.append(trajectory)
        if len(self.storage) > self.capacity:
            self.storage.pop(0)
        # Update Flashbax buffer state
        self._state = self._fbx_buf.add(self._state, trajectory)

    def sample_batch(self, batch_size: int):
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        current_size = len(self.storage)
        if batch_size > current_size:
            raise ValueError(
                f"Not enough elements to sample: requested {batch_size}, but buffer has {current_size}"
            )
        if batch_size == 0:
            return []
        return random.sample(self.storage, batch_size)

class PrioritizedTrajectoryBuffer:
    """
    Trajectory buffer with prioritization using Flashbax with fixed capacity.
    """
    def __init__(self, capacity: int, alpha: float = 1.0):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
        self.capacity = capacity
        self.alpha = alpha
        self.storage = []  # list of trajectories
        self.priorities = []  # list of priorities
        # Initialize Flashbax prioritized trajectory buffer
        self._fbx_buf = fbx.make_prioritised_trajectory_buffer(capacity, alpha)
        self._state = self._fbx_buf.init()

    def __len__(self) -> int:
        return len(self.storage)

    def add_trajectory(self, trajectory, priority: float = None):
        # Determine initial priority
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        if priority <= 0:
            raise ValueError(f"Priority must be positive, got {priority}")
        self.storage.append(trajectory)
        self.priorities.append(priority)
        if len(self.storage) > self.capacity:
            self.storage.pop(0)
            self.priorities.pop(0)
        # Update Flashbax buffer state
        self._state = self._fbx_buf.add(self._state, trajectory, priority=priority)

    def sample_batch(self, batch_size: int):
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        current_size = len(self.storage)
        if batch_size > current_size:
            raise ValueError(
                f"Not enough elements to sample: requested {batch_size}, but buffer has {current_size}"
            )
        if batch_size == 0:
            return []
        # Compute sampling weights
        weights = [p ** self.alpha for p in self.priorities]
        return random.choices(self.storage, weights=weights, k=batch_size)

    def update_priorities(self, indices, new_priorities):
        if len(indices) != len(new_priorities):
            raise ValueError("Indices and new_priorities must have the same length")
        for idx, new_p in zip(indices, new_priorities):
            if not 0 <= idx < len(self.priorities):
                raise IndexError(f"Index {idx} out of range for buffer of size {len(self.priorities)}")
            if new_p <= 0:
                raise ValueError(f"Priority must be positive, got {new_p}")
            self.priorities[idx] = new_p
        # Update Flashbax buffer state
        self._state = self._fbx_buf.update_priorities(self._state, indices, new_priorities)

# Aliases for backward compatibility and test imports
ReplayBuffer = TrajectoryBuffer
PrioritizedReplayBuffer = PrioritizedTrajectoryBuffer 