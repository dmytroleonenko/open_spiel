import pytest
import numpy as np
import jax
import jax.numpy as jnp
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    PrioritizedTrajectoryBuffer,
    TrajectoryBuffer,
)


def create_dummy_trajectory(length=5, num_actions=9, obs_shape=(8, 8)):
    """Create a dummy trajectory for testing."""
    return {
        'observations': np.random.rand(length, *obs_shape).astype(np.float32),
        'actions': np.random.randint(0, num_actions, size=length),
        'rewards': np.random.rand(length).astype(np.float32),
        'value_targets': np.random.rand(length).astype(np.float32),
        'policy_targets': [np.random.rand(num_actions).astype(np.float32) for _ in range(length)]
    }


def test_init_invalid_capacity():
    with pytest.raises(ValueError):
        PrioritizedTrajectoryBuffer(capacity=0, observation_shape=(8, 8), num_actions=9)
    with pytest.raises(ValueError):
        PrioritizedTrajectoryBuffer(capacity=-1, observation_shape=(8, 8), num_actions=9)


def test_init_invalid_alpha():
    with pytest.raises(ValueError):
        PrioritizedTrajectoryBuffer(capacity=10, observation_shape=(8, 8), num_actions=9, alpha=-0.5)


def test_add_and_len_and_priorities_default():
    buf = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9, alpha=1.0)
    assert len(buf) == 0
    
    # Add first trajectory
    traj1 = create_dummy_trajectory()
    buf.add_trajectory(traj1)
    assert len(buf) == 1
    
    # Add second trajectory with explicit priority
    traj2 = create_dummy_trajectory()
    buf.add_trajectory(traj2, priority=2.0)
    assert len(buf) == 2
    
    # Test invalid priority
    traj3 = create_dummy_trajectory()
    with pytest.raises(ValueError):
        buf.add_trajectory(traj3, priority=0)


def test_capacity_enforced():
    buf = PrioritizedTrajectoryBuffer(capacity=2, observation_shape=(8, 8), num_actions=9)
    
    traj1 = create_dummy_trajectory()
    traj2 = create_dummy_trajectory()
    traj3 = create_dummy_trajectory()
    
    buf.add_trajectory(traj1, priority=1.0)
    buf.add_trajectory(traj2, priority=2.0)
    buf.add_trajectory(traj3, priority=3.0)  # Should evict oldest
    
    assert len(buf) == 2


def test_sample_batch_with_rng():
    buf = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9, alpha=1.0)
    
    # Add some trajectories
    for i in range(3):
        traj = create_dummy_trajectory()
        buf.add_trajectory(traj, priority=float(i + 1))
    
    # Sample batch
    rng_key = jax.random.PRNGKey(42)
    trajectories, indices, weights = buf.sample_batch(2, rng_key=rng_key)
    
    assert len(trajectories) == 2
    assert len(indices) == 2
    assert len(weights) == 2
    assert isinstance(indices, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)


def test_sample_batch_error_when_too_many():
    buf = PrioritizedTrajectoryBuffer(capacity=1, observation_shape=(8, 8), num_actions=9)
    traj = create_dummy_trajectory()
    buf.add_trajectory(traj)
    
    with pytest.raises(ValueError):
        buf.sample_batch(2)


def test_update_priorities():
    buf = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    
    # Add trajectories
    for i in range(3):
        traj = create_dummy_trajectory()
        buf.add_trajectory(traj, priority=float(i + 1))
    
    # Sample to get indices
    rng_key = jax.random.PRNGKey(42)
    trajectories, indices, weights = buf.sample_batch(2, rng_key=rng_key)
    
    # Update priorities
    new_priorities = jnp.array([10.0, 20.0])
    buf.update_priorities(indices, new_priorities)
    
    # Test error conditions
    with pytest.raises(ValueError):
        buf.update_priorities(jnp.array([0]), jnp.array([1.0, 2.0]))  # Length mismatch


def test_sample_batch_zero_returns_empty():
    buf = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    traj = create_dummy_trajectory()
    buf.add_trajectory(traj, priority=1.0)
    
    trajectories, indices, weights = buf.sample_batch(0)
    assert len(trajectories) == 0
    assert len(indices) == 0
    assert len(weights) == 0


def test_sample_batch_negative_batch_size():
    buf = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    traj = create_dummy_trajectory()
    buf.add_trajectory(traj, priority=1.0)
    
    with pytest.raises(ValueError):
        buf.sample_batch(-1)


def test_add_trajectory_negative_priority():
    buf = PrioritizedTrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    traj = create_dummy_trajectory()
    
    with pytest.raises(ValueError):
        buf.add_trajectory(traj, priority=-5.0)


def test_prioritized_buffer_missing_target_values_and_array_policies():
    buf = PrioritizedTrajectoryBuffer(capacity=2, observation_shape=(4, 4), num_actions=3)
    traj = create_dummy_trajectory(length=2, num_actions=3, obs_shape=(4, 4))
    traj.pop('value_targets')
    traj.pop('target_values', None)
    traj['policy_targets'] = np.stack(traj['policy_targets'], axis=0)
    buf.add_trajectory(traj, priority=1.0)
    sampled, _, _ = buf.sample_batch(1)
    np.testing.assert_array_equal(sampled[0]['value_targets'], np.zeros(2, dtype=np.float32))
    assert isinstance(sampled[0]['policy_targets'], list)


def test_prioritized_buffer_missing_trajectory_errors_and_update_by_ids():
    buf = PrioritizedTrajectoryBuffer(capacity=1, observation_shape=(2, 2), num_actions=2)
    traj = create_dummy_trajectory(length=2, num_actions=2, obs_shape=(2, 2))
    buf.add_trajectory(traj, priority=1.0)
    with pytest.raises(KeyError):
        buf.get_trajectory(1234)
    with pytest.raises(KeyError):
        buf.update_trajectory_targets(1234, value_targets=np.ones(2))
    with pytest.raises(ValueError):
        buf.update_priorities_by_ids([0], [0.5, 0.7])
    with pytest.raises(KeyError):
        buf.get_priority(1234)


def test_standard_trajectory_buffer():
    """Test the standard (non-prioritized) trajectory buffer."""
    buf = TrajectoryBuffer(capacity=3, observation_shape=(8, 8), num_actions=9)
    assert len(buf) == 0
    
    # Add trajectory
    traj = create_dummy_trajectory()
    buf.add_trajectory(traj)
    assert len(buf) == 1
    
    # Sample trajectories
    rng_key = jax.random.PRNGKey(42)
    trajectories = buf.sample_batch(1, rng_key=rng_key)
    assert len(trajectories) == 1
    
    # Check trajectory structure
    sampled_traj = trajectories[0]
    assert 'observations' in sampled_traj
    assert 'actions' in sampled_traj
    assert 'rewards' in sampled_traj
    assert 'value_targets' in sampled_traj
    assert 'policy_targets' in sampled_traj


def test_trajectory_padding_and_extraction():
    """Test that trajectories are properly padded and extracted."""
    buf = TrajectoryBuffer(capacity=2, observation_shape=(4, 4), num_actions=5, max_trajectory_length=10)
    
    # Create a short trajectory
    short_traj = create_dummy_trajectory(length=3, num_actions=5, obs_shape=(4, 4))
    buf.add_trajectory(short_traj)
    
    # Sample and verify
    rng_key = jax.random.PRNGKey(42)
    trajectories = buf.sample_batch(1, rng_key=rng_key)
    sampled_traj = trajectories[0]
    
    # Should have same length as original
    assert len(sampled_traj['actions']) == 3
    assert len(sampled_traj['observations']) == 3
    assert len(sampled_traj['value_targets']) == 3
    assert len(sampled_traj['policy_targets']) == 3


def test_different_trajectory_lengths():
    """Test handling of trajectories with different lengths."""
    buf = TrajectoryBuffer(capacity=3, observation_shape=(2, 2), num_actions=3, max_trajectory_length=20)
    
    # Add trajectories of different lengths
    lengths = [5, 10, 3]
    for length in lengths:
        traj = create_dummy_trajectory(length=length, num_actions=3, obs_shape=(2, 2))
        buf.add_trajectory(traj)
    
    assert len(buf) == 3
    
    # Sample all trajectories
    rng_key = jax.random.PRNGKey(42)
    trajectories = buf.sample_batch(3, rng_key=rng_key)
    
    # Check that each trajectory maintains its original length
    sampled_lengths = [len(traj['actions']) for traj in trajectories]
    assert sorted(sampled_lengths) == sorted(lengths) 
