import threading
import time

import numpy as np

from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherClient,
    GrpcParameterPublisherServer,
    LocalParameterPublisher,
)


def test_local_publish_and_get_latest():
    publisher = LocalParameterPublisher()
    params_v1 = {"w": np.arange(3)}
    publisher.publish(params_v1, step=5)

    latest_params, latest_step = publisher.latest()
    assert latest_step == 5
    assert np.array_equal(latest_params["w"], params_v1["w"])


def test_wait_for_step_unblocks_after_publish():
    publisher = LocalParameterPublisher()
    result = {}

    def delayed_publish():
        time.sleep(0.05)
        publisher.publish({"key": "v2"}, step=7)

    thread = threading.Thread(target=delayed_publish)
    thread.start()
    params, step = publisher.wait_for_step(min_step=7, timeout_s=1.0)
    thread.join()

    assert step == 7
    assert params["key"] == "v2"


def test_grpc_round_trip():
    backend = LocalParameterPublisher()
    server = GrpcParameterPublisherServer(backend)
    server.start()
    client = GrpcParameterPublisherClient(server.endpoint)

    client.publish({"hello": "world"}, step=3)
    params, step = client.latest()

    assert step == 3
    assert params["hello"] == "world"

    client.close()
    server.stop()
