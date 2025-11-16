import pytest

from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import LocalParameterPublisher


def test_latest_raises_before_publish():
    publisher = LocalParameterPublisher()
    with pytest.raises(RuntimeError):
        publisher.latest()


def test_wait_for_step_timeout():
    publisher = LocalParameterPublisher()
    with pytest.raises(TimeoutError):
        publisher.wait_for_step(min_step=1, timeout_s=0.01)


def test_publish_and_latest():
    publisher = LocalParameterPublisher()
    publisher.publish({"a": 1}, step=3)
    params, step = publisher.latest()
    assert step == 3
    assert params["a"] == 1
