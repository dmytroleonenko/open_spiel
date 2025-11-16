"""Multi-process soak test for replay + parameter publisher RPC services.

This is a short end-to-end check (kept under a few seconds) that exercises the
gRPC servers with multiple processes appending/ sampling trajectories while a
separate process publishes parameters. It aims to mirror production usage
without mocks while still remaining deterministic and fast.
"""

import multiprocessing as mp
import time

import numpy as np
import pytest

from open_spiel.python.algorithms.muzero_jax.services import parameter_publisher as pp
from open_spiel.python.algorithms.muzero_jax.services import replay_service as rs


def _writer(endpoint: str, n: int, out: mp.Queue) -> None:
    client = rs.GrpcReplayClient(endpoint, timeout_s=2.0)
    start = time.perf_counter()
    for i in range(n):
        client.append({"obs": np.array([i], dtype=np.int32)}, priority=1.0)
    out.put(("writer_done", n, time.perf_counter() - start))


def _sampler(endpoint: str, batches: int, batch_size: int, out: mp.Queue) -> None:
    client = rs.GrpcReplayClient(endpoint, timeout_s=2.0)
    sampled = 0
    try:
        for _ in range(batches):
            trajs, ids, prios = client.sample(batch_size=batch_size)
            sampled += len(trajs)
            # sanity checks
            assert len(ids) == batch_size
            assert prios.shape[0] == batch_size
    except Exception as e:  # pragma: no cover - surfaced via Queue
        out.put(("sampler_error", str(e)))
        return
    out.put(("sampler_done", sampled))


def _publisher(endpoint: str, steps: int, out: mp.Queue) -> None:
    client = pp.GrpcParameterPublisherClient(endpoint, timeout_s=2.0)
    for step in range(1, steps + 1):
        client.publish({"w": step}, step=step)
    out.put(("published", steps))


@pytest.mark.timeout(20)
def test_replay_and_publisher_multiprocess_soak():
    # Start services.
    replay_backend = rs.InMemoryReplayService(capacity=64, alpha=0.5)
    replay_server = rs.GrpcReplayServer(replay_backend)
    replay_server.start()

    param_backend = pp.LocalParameterPublisher()
    param_server = pp.GrpcParameterPublisherServer(param_backend)
    param_server.start()

    writer_q, sampler_q, publisher_q = mp.Queue(), mp.Queue(), mp.Queue()
    writer = mp.Process(target=_writer, args=(replay_server.endpoint, 24, writer_q))
    sampler = mp.Process(target=_sampler, args=(replay_server.endpoint, 8, 3, sampler_q))
    publisher = mp.Process(target=_publisher, args=(param_server.endpoint, 3, publisher_q))

    for p in (writer, sampler, publisher):
        p.start()

    # Collect results with short joins to avoid hangs.
    for p in (writer, sampler, publisher):
        p.join(timeout=10)
        assert not p.is_alive(), f"{p.name} hung during soak test"

    # Validate replay throughput.
    writer_tag, n_written, elapsed = writer_q.get(timeout=1)
    assert writer_tag == "writer_done" and n_written == 24
    assert elapsed > 0

    sampler_tag, sampled_count = sampler_q.get(timeout=1)
    if sampler_tag == "sampler_error":
        pytest.fail(f"sampler process error: {sampled_count}")
    assert sampler_tag == "sampler_done" and sampled_count >= 8  # at least one per batch

    # Validate publisher cross-process visibility.
    pub_tag, steps = publisher_q.get(timeout=1)
    assert pub_tag == "published" and steps == 3
    latest_params, latest_step = pp.GrpcParameterPublisherClient(param_server.endpoint, timeout_s=2.0).latest()
    assert latest_step == 3 and latest_params["w"] == 3

    # Basic RPC metric: writer QPS should be reasonable (> 100 inserts/s in this tiny run).
    assert (n_written / elapsed) > 100

    replay_server.stop()
    param_server.stop()
