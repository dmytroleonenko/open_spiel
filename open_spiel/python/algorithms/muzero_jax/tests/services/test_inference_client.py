import threading

import jax.numpy as jnp
import pytest

from open_spiel.python.algorithms.muzero_jax.services.inference_client import (
    InferenceNetworkAdapter,
    LocalInferenceClient,
    LocalBatchingInferenceClient,
    build_inference_client,
    _slice_component,
    _split_batched_outputs,
)


class DummyNetwork:
    def __init__(self):
        self.initial_calls = 0
        self.recurrent_calls = 0

    def initial_inference(self, obs, training=False):
        self.initial_calls += 1
        batch = obs.shape[0]
        hidden = jnp.ones((batch, 4))
        reward = jnp.zeros((batch,))
        value = jnp.zeros((batch,))
        policy = jnp.zeros((batch, 3))
        projection = None
        reward_hidden = None
        return hidden, reward, value, policy, projection, reward_hidden

    def recurrent_inference(self, hidden_state, actions, training=False):
        self.recurrent_calls += 1
        batch = hidden_state.shape[0]
        next_hidden = jnp.ones_like(hidden_state)
        reward = jnp.zeros((batch,))
        value = jnp.zeros((batch,))
        policy = jnp.zeros((batch, 3))
        projection = None
        reward_hidden = None
        return next_hidden, reward, value, policy, projection, reward_hidden


def test_build_inference_client_defaults_to_local():
    cfg = None
    network = DummyNetwork()
    client = build_inference_client(network, cfg)
    assert isinstance(client, LocalInferenceClient)
    client.close()


def test_build_inference_client_local_batching(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        remote_enabled=False,
        enable_local_batching=True,
        batch_size=4,
        max_wait_ms=10,
    )
    network = DummyNetwork()
    client = build_inference_client(network, cfg)
    assert isinstance(client, LocalBatchingInferenceClient)
    client.close()


def test_local_batching_inference_client_batches_requests():
    network = DummyNetwork()
    client = LocalBatchingInferenceClient(network, batch_size=2, max_wait_ms=50)
    results = []

    def worker(seed):
        obs = jnp.ones((1, 3)) * seed
        results.append(client.initial_inference(obs))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert network.initial_calls == 1
    assert len(results) == 2
    client.close()


def test_local_batching_recurrent_inference():
    network = DummyNetwork()
    client = LocalBatchingInferenceClient(network, batch_size=2, max_wait_ms=50)
    hidden = jnp.ones((1, 4), dtype=jnp.float32)
    action = jnp.ones((1, 2), dtype=jnp.float32)
    client.recurrent_inference(hidden, action)
    client.close()
    client.close()  # second close exercises early return branch
    assert network.recurrent_calls == 1


def test_build_inference_client_remote_requires_endpoint():
    # Remote inference removed; any attempt should raise.
    from types import SimpleNamespace
    cfg = SimpleNamespace(remote_enabled=True)
    with pytest.raises(ValueError):
        build_inference_client(DummyNetwork(), cfg)


def test_inference_network_adapter_proxies_calls():
    client = LocalInferenceClient(DummyNetwork())
    adapter = InferenceNetworkAdapter(client)
    obs = jnp.ones((1, 3))
    adapter.initial_inference(obs)
    hidden = jnp.ones((1, 4))
    action = jnp.ones((1, 1))
    adapter.recurrent_inference(hidden, action)


def test_slice_component_handles_nested_lists():
    tensors = [jnp.ones((1, 2)), jnp.ones((1, 2)) * 2]
    result = _slice_component(tensors, 0)
    assert isinstance(result, list)
    assert len(result) == 2


def test_split_batched_outputs_wraps_non_tuple_output():
    tensor = jnp.ones((1, 4))
    result = _split_batched_outputs(tensor, 1)
    assert len(result) == 1
