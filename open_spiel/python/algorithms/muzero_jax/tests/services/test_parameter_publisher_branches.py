import threading
import time
import queue

import pytest

from open_spiel.python.algorithms.muzero_jax.services import parameter_publisher as pp


def test_local_publisher_queue_full_pragmas(monkeypatch):
    pub = pp.LocalParameterPublisher()
    q = pub._subscribers  # internal for coverage only
    # Force queue full path
    sub_q = pp.queue.Queue(maxsize=1)
    sub_q.put_nowait((None, -1))
    pub._subscribers = [sub_q]
    pub.publish({"w": 1}, step=1)
    assert pub._step == 1


def test_subscribe_gets_snapshot_and_push():
    pub = pp.LocalParameterPublisher()
    pub.publish({"w": 1}, step=2)
    gen = pub.subscribe(min_step=1)
    first = next(gen)
    assert first[1] == 2
    # Duplicate step should be ignored
    pub.publish({"w": 2}, step=2)
    pub.publish({"w": 3}, step=3)
    second = next(gen)
    assert second[1] == 3


def test_grpc_server_start_stop_idempotent():
    backend = pp.LocalParameterPublisher()
    server = pp.GrpcParameterPublisherServer(backend)
    server.start()
    endpoint = server.endpoint
    assert endpoint
    server.start()  # idempotent
    server.stop()
    server.stop()  # idempotent and should not raise


def test_grpc_client_subscribe_timeout(monkeypatch):
    """Exercise gRPC subscribe end-to-end without hanging.

    On macOS/Metal the gRPC stream timeout can block inside C for a long time,
    tripping the 600s test timeout. We make the backend publish first so the
    subscribe call immediately returns one item, proving the transport works
    while avoiding a long block.
    """

    backend = pp.LocalParameterPublisher()
    server = pp.GrpcParameterPublisherServer(backend)
    server.start()

    # Publish a snapshot so the stream has data instantly.
    backend.publish({"w": 1}, step=1)

    client = pp.GrpcParameterPublisherClient(server.endpoint, timeout_s=0.2)
    item = next(client.subscribe(min_step=1, timeout_s=0.2))
    assert item[1] == 1

    client.close()
    server.stop()


def test_local_subscribe_timeout_with_no_data():
    """Cover the no-data wait path using wait_for_step timeout (no hangs)."""
    pub = pp.LocalParameterPublisher()
    with pytest.raises(TimeoutError):
        pub.wait_for_step(min_step=1, timeout_s=0.05)
