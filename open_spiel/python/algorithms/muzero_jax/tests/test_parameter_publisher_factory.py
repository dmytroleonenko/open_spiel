from omegaconf import OmegaConf

from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import build_parameter_publisher_client
from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import GrpcParameterPublisherServer, LocalParameterPublisher


def test_builds_local_when_remote_disabled():
    cfg = OmegaConf.create({"publisher": {"remote_enabled": False, "rpc_endpoint": ""}})
    client, server = build_parameter_publisher_client(cfg)
    assert client is None
    assert server is None


def test_builds_grpc_when_enabled():
    backend = LocalParameterPublisher()
    server = GrpcParameterPublisherServer(backend)
    server.start()
    endpoint = server.endpoint

    cfg = OmegaConf.create({"publisher": {"remote_enabled": True, "rpc_endpoint": endpoint}})
    client, _ = build_parameter_publisher_client(cfg)
    assert client is not None
    params = {"a": 1}
    client.publish(params, step=3)
    got, step = backend.latest()
    assert step == 3
    assert got["a"] == 1
    client.close()
    server.stop()
