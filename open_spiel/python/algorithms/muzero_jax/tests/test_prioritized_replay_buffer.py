import pytest
import random
from open_spiel.python.algorithms.muzero_jax.replay_buffer import PrioritizedReplayBuffer


def test_init_invalid_capacity():
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(capacity=0)
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(capacity=-1)


def test_init_invalid_alpha():
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(capacity=10, alpha=-0.5)


def test_add_and_len_and_priorities_default():
    buf = PrioritizedReplayBuffer(capacity=3, alpha=1.0)
    assert len(buf) == 0
    buf.add_trajectory('a')
    assert len(buf) == 1
    assert buf.priorities == [1.0]
    buf.add_trajectory('b', priority=2.0)
    assert len(buf) == 2
    assert buf.priorities == [1.0, 2.0]
    with pytest.raises(ValueError):
        buf.add_trajectory('c', priority=0)


def test_capacity_enforced_with_priorities():
    buf = PrioritizedReplayBuffer(capacity=2)
    buf.add_trajectory('a', priority=1.0)
    buf.add_trajectory('b', priority=2.0)
    buf.add_trajectory('c', priority=3.0)
    assert len(buf) == 2
    assert buf.storage == ['b', 'c']
    assert buf.priorities == [2.0, 3.0]


def test_sample_batch_weights_uniform_when_alpha_zero(monkeypatch):
    buf = PrioritizedReplayBuffer(capacity=3, alpha=0.0)
    items = ['x', 'y', 'z']
    for item in items:
        buf.add_trajectory(item, priority=random.uniform(1, 10))
    called = {}
    def fake_choices(population, weights, k):
        called['weights'] = weights
        return ['x'] * k
    monkeypatch.setattr(random, 'choices', fake_choices)
    batch = buf.sample_batch(2)
    assert batch == ['x', 'x']
    assert all(w == 1.0 for w in called['weights'])


def test_sample_batch_error_when_too_many():
    buf = PrioritizedReplayBuffer(capacity=1)
    buf.add_trajectory('only')
    with pytest.raises(ValueError):
        buf.sample_batch(2)


def test_update_priorities_and_errors():
    buf = PrioritizedReplayBuffer(capacity=3)
    for i in range(3):
        buf.add_trajectory(f't{i}', priority=i+1)
    buf.update_priorities([0, 2], [10.0, 20.0])
    assert buf.priorities == [10.0, 2.0, 20.0]
    with pytest.raises(ValueError):
        buf.update_priorities([0, 1], [1.0])
    with pytest.raises(IndexError):
        buf.update_priorities([5], [1.0])
    with pytest.raises(ValueError):
        buf.update_priorities([0], [0.0])


# Edge-case tests for PrioritizedReplayBuffer

def test_sample_batch_zero_returns_empty():
    buf = PrioritizedReplayBuffer(capacity=3)
    buf.add_trajectory('x', priority=1.0)
    buf.add_trajectory('y', priority=2.0)
    # sample zero should return empty list
    result = buf.sample_batch(0)
    assert isinstance(result, list)
    assert result == []


def test_sample_batch_negative_batch_size():
    buf = PrioritizedReplayBuffer(capacity=3)
    buf.add_trajectory('x', priority=1.0)
    with pytest.raises(ValueError):
        buf.sample_batch(-1)


def test_add_trajectory_negative_priority():
    buf = PrioritizedReplayBuffer(capacity=3)
    with pytest.raises(ValueError):
        buf.add_trajectory('a', priority=-5.0)


def test_update_priorities_length_mismatch():
    buf = PrioritizedReplayBuffer(capacity=3)
    buf.add_trajectory('a', priority=1.0)
    buf.add_trajectory('b', priority=2.0)
    # indices and new_priorities length mismatch
    with pytest.raises(ValueError):
        buf.update_priorities([0], [1.0, 2.0])


def test_update_priorities_out_of_range_and_zero():
    buf = PrioritizedReplayBuffer(capacity=3)
    for i in range(3):
        buf.add_trajectory(f't{i}', priority=i+1)
    # out of range index
    with pytest.raises(IndexError):
        buf.update_priorities([3], [1.0])
    # non-positive new priority
    with pytest.raises(ValueError):
        buf.update_priorities([0], [0.0])


def test_prioritized_negative_batch_size():
    buf = PrioritizedReplayBuffer(capacity=3)
    buf.add_trajectory('only', priority=1.0)
    with pytest.raises(ValueError):
        buf.sample_batch(-1)


def test_prioritized_zero_batch_size():
    buf = PrioritizedReplayBuffer(capacity=3)
    buf.add_trajectory('a', priority=1.0)
    buf.add_trajectory('b', priority=2.0)
    assert buf.sample_batch(0) == []


def test_prioritized_default_priority_assignment():
    buf = PrioritizedReplayBuffer(capacity=3)
    buf.add_trajectory('x', priority=4.2)
    buf.add_trajectory('y')  # default priority
    assert buf.priorities[-1] == 4.2 