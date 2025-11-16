import time

from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherClient,
    GrpcParameterPublisherServer,
    LocalParameterPublisher,
)


def test_subscribe_pushes_updates():
    backend = LocalParameterPublisher()
    server = GrpcParameterPublisherServer(backend)
    server.start()
    client = GrpcParameterPublisherClient(server.endpoint)

    client.publish({"version": 1}, step=1)
    stream = client.subscribe(min_step=1, timeout_s=2.0)
    params, step = next(stream)
    assert step == 1
    assert params["version"] == 1

    client.publish({"version": 2}, step=2)
    params2, step2 = client.wait_for_step(min_step=2, timeout_s=2.0)
    assert step2 == 2
    assert params2["version"] == 2

    client.close()
    server.stop()
