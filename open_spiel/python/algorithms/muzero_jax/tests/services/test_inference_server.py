import asyncio

import pytest


def test_batches_requests_up_to_batch_size():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        BatchingInferenceServer,
    )

    calls = []

    async def fake_infer(batch):
        calls.append(list(batch))
        return [value * 2 for value in batch]

    async def run():
        server = BatchingInferenceServer(
            infer_fn=fake_infer,
            batch_size=2,
            max_wait_ms=1000,
        )
        results = await asyncio.gather(server.submit(3), server.submit(7))
        await server.shutdown()
        return results

    results = asyncio.run(run())
    assert calls == [[3, 7]]
    assert results == [6, 14]


def test_timeout_flushes_partial_batch():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        BatchingInferenceServer,
    )

    calls = []

    async def fake_infer(batch):
        calls.append(list(batch))
        return [item.upper() for item in batch]

    async def run():
        server = BatchingInferenceServer(
            infer_fn=fake_infer,
            batch_size=5,
            max_wait_ms=5,
        )
        first = asyncio.create_task(server.submit("alpha"))
        await asyncio.sleep(0.001)
        second = asyncio.create_task(server.submit("beta"))
        results = await asyncio.gather(first, second)
        await server.shutdown()
        return results

    results = asyncio.run(run())
    assert calls == [["alpha", "beta"]]
    assert results == ["ALPHA", "BETA"]


def test_batching_server_validates_inputs():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        BatchingInferenceServer,
    )

    with pytest.raises(ValueError):
        BatchingInferenceServer(lambda _: _, batch_size=0)
    with pytest.raises(ValueError):
        BatchingInferenceServer(lambda _: _, max_wait_ms=-1)


def test_batching_server_rejects_after_shutdown():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        BatchingInferenceServer,
    )

    async def run():
        server = BatchingInferenceServer(lambda batch: batch)
        await server.shutdown()
        with pytest.raises(RuntimeError):
            await server.submit(1)

    asyncio.run(run())


def test_flush_pending_cancels_timer():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        BatchingInferenceServer,
    )

    async def run():
        server = BatchingInferenceServer(lambda batch: batch)
        server._timer_task = asyncio.create_task(asyncio.sleep(0))
        await server._flush_pending()

    asyncio.run(run())


def test_infer_fn_mismatch_raises_value_error():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        BatchingInferenceServer,
    )

    async def run():
        async def bad_fn(batch):
            return []

        server = BatchingInferenceServer(bad_fn, batch_size=1)
        with pytest.raises(ValueError):
            await server.submit(1)

    asyncio.run(run())


def test_maybe_await_returns_value_for_sync_input():
    from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
        _maybe_await,
    )

    async def run():
        result = await _maybe_await(42)
        assert result == 42

    asyncio.run(run())
