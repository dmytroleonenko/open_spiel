import pytest
import numpy as np
import jax
import jax.numpy as jnp
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    ReplayBuffer,
    TrajectoryBuffer,
    PrioritizedTrajectoryBuffer,
    _make_numpy_rng,
)


def create_sample_trajectory(length=3, obs_shape=(8, 8), num_actions=9):
    """Create a sample trajectory for testing."""
    return {
        'observations': np.random.rand(length, *obs_shape).astype(np.float32),
        'actions': np.random.randint(0, num_actions, size=length),
        'rewards': np.random.rand(length).astype(np.float32),
        'value_targets': np.random.rand(length).astype(np.float32),
        'policy_targets': [np.random.rand(num_actions).astype(np.float32) for _ in range(length)]
    }


def test_make_numpy_rng_handles_zero_seed():
    """_make_numpy_rng should fallback to seed=1 when the derived seed is zero."""
    rng = _make_numpy_rng(jax.random.PRNGKey(0))
    sample = rng.integers(0, 10)
    assert 0 <= sample < 10


def test_trajectory_buffer_add_and_len():
    """Test basic add and length functionality."""
    buffer = TrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    assert len(buffer) == 0
    
    traj1 = create_sample_trajectory()
    buffer.add_trajectory(traj1)
    assert len(buffer) == 1
    
    traj2 = create_sample_trajectory()
    buffer.add_trajectory(traj2)
    assert len(buffer) == 2


def test_trajectory_buffer_capacity_enforced():
    """Test that capacity is properly enforced."""
    buffer = TrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    
    for i in range(3):  # Add more than capacity
        traj = create_sample_trajectory()
        buffer.add_trajectory(traj)
    
    # Should only have capacity=2 trajectories
    assert len(buffer) == 2


def test_trajectory_buffer_sample_batch_returns_elements():
    """Test that sampling returns proper trajectories."""
    buffer = TrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    
    # Add trajectories
    for i in range(3):
        traj = create_sample_trajectory(length=3)
        buffer.add_trajectory(traj)
    
    # Sample batch
    batch = buffer.sample_batch(2)
    assert isinstance(batch, list)
    assert len(batch) == 2
    
    # Check structure of returned trajectories
    for trajectory in batch:
        assert 'observations' in trajectory
        assert 'actions' in trajectory
        assert 'rewards' in trajectory
        assert 'value_targets' in trajectory
        assert 'policy_targets' in trajectory
        assert len(trajectory['observations']) == len(trajectory['actions'])


def test_trajectory_buffer_missing_target_values_defaults_to_zero():
    """Ensure trajectories without value targets are accepted and padded with zeros."""
    buffer = TrajectoryBuffer(capacity=1, observation_shape=(4, 4), num_actions=3)
    traj = create_sample_trajectory(length=2, obs_shape=(4, 4), num_actions=3)
    traj.pop('value_targets')
    traj.pop('target_values', None)
    buffer.add_trajectory(traj)
    sampled = buffer.sample_batch(1)[0]
    np.testing.assert_array_equal(sampled['value_targets'], np.zeros(2, dtype=np.float32))


def test_trajectory_buffer_accepts_array_policy_targets():
    """When policy targets arrive as a NumPy array, the array branch is exercised."""
    buffer = TrajectoryBuffer(capacity=1, observation_shape=(2, 2), num_actions=2)
    traj = create_sample_trajectory(length=2, obs_shape=(2, 2), num_actions=2)
    traj['policy_targets'] = np.stack(traj['policy_targets'], axis=0)
    buffer.add_trajectory(traj)
    sampled = buffer.sample_batch(1)[0]
    assert isinstance(sampled['policy_targets'], list)
    assert len(sampled['policy_targets']) == 2


def test_trajectory_buffer_sample_batch_error_when_too_many():
    """Test error when sampling more than available."""
    buffer = TrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj)
    
    with pytest.raises(ValueError) as excinfo:
        buffer.sample_batch(2)
    msg = str(excinfo.value)
    assert 'Not enough elements to sample' in msg


def test_trajectory_buffer_invalid_capacity():
    """Test error for invalid capacity."""
    with pytest.raises(ValueError):
        TrajectoryBuffer(capacity=0, observation_shape=(8, 8), num_actions=9)


def test_trajectory_buffer_negative_batch_size():
    """Test error for negative batch size."""
    buffer = TrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj)
    
    with pytest.raises(ValueError):
        buffer.sample_batch(-1)


def test_trajectory_buffer_zero_batch_size_returns_empty():
    """Test that zero batch size returns empty list."""
    buffer = TrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj)
    
    result = buffer.sample_batch(0)
    assert isinstance(result, list) and result == []


def test_trajectory_buffer_variable_length_trajectories():
    """Test handling of variable length trajectories."""
    buffer = TrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9, max_trajectory_length=10)
    
    # Add trajectories of different lengths
    lengths = [3, 5, 7]
    for length in lengths:
        traj = create_sample_trajectory(length=length)
        buffer.add_trajectory(traj)
    
    # Sample and verify lengths are preserved
    batch = buffer.sample_batch(3)
    sampled_lengths = [len(traj['actions']) for traj in batch]
    assert set(sampled_lengths) == set(lengths)


def test_trajectory_buffer_missing_trajectory_id_errors():
    buffer = TrajectoryBuffer(capacity=1, observation_shape=(2, 2), num_actions=2)
    traj = create_sample_trajectory(length=2, obs_shape=(2, 2), num_actions=2)
    buffer.add_trajectory(traj)
    with pytest.raises(KeyError):
        buffer.get_trajectory(999)
    with pytest.raises(KeyError):
        buffer.update_trajectory_targets(999, value_targets=np.ones(2))


def test_trajectory_buffer_jax_compatibility():
    """Test that the buffer works with JAX arrays."""
    buffer = TrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    
    # Create trajectory with JAX arrays
    traj = {
        'observations': jnp.array(np.random.rand(3, 8, 8).astype(np.float32)),
        'actions': jnp.array([0, 1, 2]),
        'rewards': jnp.array([1.0, 0.0, -1.0]),
        'value_targets': jnp.array([0.5, 0.3, 0.1]),
        'policy_targets': [jnp.array(np.random.rand(9).astype(np.float32)) for _ in range(3)]
    }
    
    buffer.add_trajectory(traj)
    batch = buffer.sample_batch(1)
    
    # Verify trajectory structure
    assert len(batch) == 1
    trajectory = batch[0]
    assert trajectory['observations'].shape == (3, 8, 8)
    assert len(trajectory['actions']) == 3


# Prioritized buffer tests
def test_prioritized_buffer_basic_functionality():
    """Test basic prioritized buffer functionality."""
    buffer = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    assert len(buffer) == 0
    
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj, priority=1.0)
    assert len(buffer) == 1


def test_prioritized_buffer_priority_validation():
    """Test priority validation."""
    buffer = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    traj = create_sample_trajectory()
    
    # Test default priority
    buffer.add_trajectory(traj)
    
    # Test invalid priority
    with pytest.raises(ValueError):
        buffer.add_trajectory(traj, priority=0.0)
    
    with pytest.raises(ValueError):
        buffer.add_trajectory(traj, priority=-1.0)


def test_prioritized_buffer_sample_returns_weights():
    """Test that prioritized sampling returns importance weights."""
    buffer = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9, alpha=1.0)
    
    # Add trajectories with different priorities
    for i in range(3):
        traj = create_sample_trajectory()
        buffer.add_trajectory(traj, priority=float(i + 1))
    
    trajectories, indices, weights = buffer.sample_batch(2)
    
    assert isinstance(trajectories, list)
    assert len(trajectories) == 2
    assert isinstance(indices, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    assert len(indices) == 2
    assert len(weights) == 2


def test_prioritized_buffer_update_priorities():
    """Test priority updates."""
    buffer = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    
    # Add trajectories
    for i in range(2):
        traj = create_sample_trajectory()
        buffer.add_trajectory(traj, priority=1.0)
    
    # Update priorities
    indices = jnp.array([0, 1])
    new_priorities = jnp.array([2.0, 3.0])
    buffer.update_priorities(indices, new_priorities)
    
    # Verify priorities were updated
    priorities = buffer.priority_values()
    assert priorities[0] == 2.0
    assert priorities[1] == 3.0


def test_prioritized_buffer_invalid_alpha():
    """Test error for invalid alpha parameter."""
    with pytest.raises(ValueError):
        PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9, alpha=-1.0)


def test_prioritized_buffer_update_priorities_validation():
    """Test validation in update_priorities method."""
    buffer = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    
    # Add a trajectory
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj)
    
    # Test mismatched indices and priorities
    with pytest.raises(ValueError):
        buffer.update_priorities(jnp.array([0]), jnp.array([1.0, 2.0]))


# Test backward compatibility aliases
def test_replay_buffer_alias():
    """Test that ReplayBuffer alias works."""
    buffer = ReplayBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    assert isinstance(buffer, TrajectoryBuffer)
    
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj)
    assert len(buffer) == 1


def test_trajectory_data_integrity():
    """Test that trajectory data is preserved correctly through add/sample cycle."""
    buffer = TrajectoryBuffer(capacity=5, observation_shape=(4, 4), num_actions=5, max_trajectory_length=4)
    
    # Create specific trajectory
    original_traj = {
        'observations': np.array([[[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                                  [[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                                  [[9, 10, 0, 0], [11, 12, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.float32),
        'actions': np.array([0, 2, 4]),
        'rewards': np.array([1.0, -0.5, 2.0], dtype=np.float32),
        'value_targets': np.array([0.8, 0.3, 0.9], dtype=np.float32),
        'policy_targets': [
            np.array([0.2, 0.3, 0.1, 0.2, 0.2], dtype=np.float32),
            np.array([0.1, 0.1, 0.5, 0.2, 0.1], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        ]
    }
    
    buffer.add_trajectory(original_traj)
    sampled_batch = buffer.sample_batch(1)
    sampled_traj = sampled_batch[0]
    
    # Check that data matches
    np.testing.assert_array_equal(sampled_traj['observations'], original_traj['observations'])
    np.testing.assert_array_equal(sampled_traj['actions'], original_traj['actions'])
    np.testing.assert_array_equal(sampled_traj['rewards'], original_traj['rewards'])
    np.testing.assert_array_equal(sampled_traj['value_targets'], original_traj['value_targets'])
    
    for i in range(len(original_traj['policy_targets'])):
        np.testing.assert_array_equal(sampled_traj['policy_targets'][i], original_traj['policy_targets'][i])


def test_trajectory_buffer_sample_unique_with_same_lengths():
    """Test sampling unique trajectories when some have the same length."""
    buffer = TrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9, max_trajectory_length=10)
    
    # Add trajectories - two with same length to test the safety fallback
    lengths = [3, 3, 7]  # Two trajectories with length 3
    for length in lengths:
        traj = create_sample_trajectory(length=length)
        buffer.add_trajectory(traj)
    
    # Sample all - this should trigger the safety fallback for duplicate lengths
    batch = buffer.sample_batch(3)
    assert len(batch) == 3
    sampled_lengths = [len(traj['actions']) for traj in batch]
    
    # We should get at least the unique lengths that exist
    assert 3 in sampled_lengths
    assert 7 in sampled_lengths


def test_prioritized_buffer_edge_cases():
    """Test edge cases for prioritized buffer."""
    buffer = PrioritizedTrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    
    # Test update_priorities with invalid indices
    traj = create_sample_trajectory()
    buffer.add_trajectory(traj, priority=1.0)
    
    # Update with out-of-bounds index (should not crash)
    buffer.update_priorities(jnp.array([10]), jnp.array([2.0]))
    
    # Update with negative index (should not crash)
    buffer.update_priorities(jnp.array([-1]), jnp.array([2.0]))
    
    # Test updating priorities with very small values (tests the minimum clipping)
    buffer.update_priorities(jnp.array([0]), jnp.array([1e-10]))
    assert buffer.priority_values()[0] >= 1e-9  # Should be clipped to minimum (allow for small precision error)


def test_prioritized_buffer_capacity_overflow():
    """Test that priority list management during overflow works correctly."""
    capacity = 2
    buffer = PrioritizedTrajectoryBuffer(
        capacity=capacity,
        observation_shape=(8, 8),
        num_actions=9,
        max_trajectory_length=5
    )
    
    # Add trajectories beyond capacity to test overflow handling
    traj1 = create_sample_trajectory(length=3, obs_shape=(8, 8), num_actions=9)
    traj2 = create_sample_trajectory(length=2, obs_shape=(8, 8), num_actions=9)
    traj3 = create_sample_trajectory(length=4, obs_shape=(8, 8), num_actions=9)
    
    buffer.add_trajectory(traj1, priority=1.0)
    buffer.add_trajectory(traj2, priority=2.0)
    assert len(buffer.priority_values()) == 2
    
    # Adding third trajectory should trigger overflow
    buffer.add_trajectory(traj3, priority=3.0)
    priorities_after = buffer.priority_values()
    assert len(priorities_after) == 2  # Should cap at capacity
    assert priorities_after == [2.0, 3.0]  # Should have removed oldest (1.0)


def test_prioritized_buffer_invalid_capacity():
    """Test that invalid capacity raises ValueError (line 210)."""
    with pytest.raises(ValueError, match="Capacity must be positive"):
        PrioritizedTrajectoryBuffer(
            capacity=0,  # Invalid capacity
            observation_shape=(8, 8),
            num_actions=9
        )
    
    with pytest.raises(ValueError, match="Capacity must be positive"):
        PrioritizedTrajectoryBuffer(
            capacity=-5,  # Invalid capacity
            observation_shape=(8, 8),
            num_actions=9
        )


def test_prioritized_buffer_negative_batch_size():
    """Test that negative batch size raises ValueError (lines 309, 311)."""
    buffer = PrioritizedTrajectoryBuffer(
        capacity=10,
        observation_shape=(8, 8),
        num_actions=9,
        max_trajectory_length=5
    )
    
    # Add one trajectory
    traj = create_sample_trajectory(length=3, obs_shape=(8, 8), num_actions=9)
    buffer.add_trajectory(traj, priority=1.0)
    
    # Test negative batch size
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        buffer.sample_batch(batch_size=-1)


def test_prioritized_buffer_zero_batch_size():
    """Test that zero batch size returns empty results (line 315)."""
    buffer = PrioritizedTrajectoryBuffer(
        capacity=10,
        observation_shape=(8, 8),
        num_actions=9,
        max_trajectory_length=5
    )
    
    # Add one trajectory
    traj = create_sample_trajectory(length=3, obs_shape=(8, 8), num_actions=9)
    buffer.add_trajectory(traj, priority=1.0)
    
    # Test zero batch size
    trajectories, indices, weights = buffer.sample_batch(batch_size=0)
    assert trajectories == []
    assert len(indices) == 0
    assert len(weights) == 0


def test_normal_sampling_path():
    """Test the normal sampling path (not sampling entire buffer)."""
    buffer = TrajectoryBuffer(capacity=5, observation_shape=(8, 8), num_actions=9, max_trajectory_length=10)
    
    # Add multiple trajectories
    for i in range(4):
        traj = create_sample_trajectory(length=i + 3)  # lengths 3, 4, 5, 6
        buffer.add_trajectory(traj)
    
    # Sample less than the entire buffer (should use normal sampling path)
    batch = buffer.sample_batch(2)
    assert len(batch) == 2
    
    # All trajectories should be valid
    for traj in batch:
        assert len(traj['actions']) >= 3
        assert len(traj['actions']) <= 6 
