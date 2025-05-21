import pytest
from open_spiel.python.algorithms.muzero_jax.replay_buffer import ReplayBuffer


def test_add_and_len():
    buffer = ReplayBuffer(capacity=2)
    assert len(buffer) == 0
    buffer.add_trajectory('traj1')
    assert len(buffer) == 1
    buffer.add_trajectory('traj2')
    assert len(buffer) == 2


def test_capacity_enforced():
    buffer = ReplayBuffer(capacity=2)
    t1, t2, t3 = 't1', 't2', 't3'
    buffer.add_trajectory(t1)
    buffer.add_trajectory(t2)
    buffer.add_trajectory(t3)
    # After adding 3 trajectories with capacity 2, only the last two should remain
    assert len(buffer) == 2
    assert buffer.storage == [t2, t3]


def test_sample_batch_returns_elements():
    buffer = ReplayBuffer(capacity=3)
    items = ['a', 'b', 'c']
    for item in items:
        buffer.add_trajectory(item)
    batch = buffer.sample_batch(2)
    assert isinstance(batch, list)
    assert len(batch) == 2
    for element in batch:
        assert element in items


def test_sample_batch_not_mutate_original():
    buffer = ReplayBuffer(capacity=3)
    items = ['x', 'y', 'z']
    for item in items:
        buffer.add_trajectory(item)
    original = list(buffer.storage)
    _ = buffer.sample_batch(2)
    assert buffer.storage == original


def test_sample_batch_error_when_too_many():
    buffer = ReplayBuffer(capacity=2)
    buffer.add_trajectory('only')
    with pytest.raises(ValueError) as excinfo:
        buffer.sample_batch(2)
    msg = str(excinfo.value)
    assert 'Not enough elements to sample' in msg


def test_invalid_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_replay_buffer_negative_batch_size():
    buffer = ReplayBuffer(capacity=2)
    buffer.add_trajectory('a')
    with pytest.raises(ValueError):
        buffer.sample_batch(-1)


def test_replay_buffer_zero_batch_size_returns_empty():
    buffer = ReplayBuffer(capacity=2)
    buffer.add_trajectory('a')
    buffer.add_trajectory('b')
    result = buffer.sample_batch(0)
    assert isinstance(result, list) and result == [] 