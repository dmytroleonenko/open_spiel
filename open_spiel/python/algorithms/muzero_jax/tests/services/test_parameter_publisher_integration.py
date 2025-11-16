import pytest

from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherClient,
    GrpcParameterPublisherServer,
    LocalParameterPublisher,
)


def test_grpc_publisher_waits_for_step():
    backend = LocalParameterPublisher()
    server = GrpcParameterPublisherServer(backend)
    server.start()
    client = GrpcParameterPublisherClient(server.endpoint)

    with pytest.raises(RuntimeError):
        backend.latest()

    client.publish({"step": 1}, step=1)
    params, step = client.wait_for_step(min_step=1, timeout_s=1.0)
    assert step == 1
    assert params["step"] == 1

    import grpc
    with pytest.raises(grpc.RpcError) as excinfo:
        client.wait_for_step(min_step=3, timeout_s=0.01)
    assert excinfo.value.code() == grpc.StatusCode.UNKNOWN

    # Subscribe should deliver immediately the latest params.
    stream = client.subscribe(min_step=1, timeout_s=2.0)
    params2, step2 = next(stream)
    assert step2 == 1
    assert params2["step"] == 1

    client.close()
    server.stop()
