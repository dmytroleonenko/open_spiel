import threading
from unittest.mock import patch

import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from open_spiel.python.algorithms.muzero_jax.services import inference_client as ic
from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator


class DummyNetwork:
    def __init__(self):
        self.initial_calls = 0

    def initial_inference(self, obs, training=False):
        self.initial_calls += 1
        batch = obs.shape[0]
        hidden = jnp.ones((batch, 4), dtype=jnp.float32)
        reward = jnp.zeros((batch,), dtype=jnp.float32)
        value = jnp.zeros((batch,), dtype=jnp.float32)
        policy = jnp.zeros((batch, 3), dtype=jnp.float32)
        projection = None
        reward_hidden = jnp.zeros((batch, 2), dtype=jnp.float32)
        return hidden, reward, value, policy, projection, reward_hidden

    def recurrent_inference(self, hidden, action, training=False):
        batch = hidden.shape[0]
        next_hidden = jnp.ones_like(hidden)
        reward = jnp.zeros((batch,), dtype=jnp.float32)
        value = jnp.zeros((batch,), dtype=jnp.float32)
        policy = jnp.zeros((batch, 3), dtype=jnp.float32)
        projection = None
        reward_hidden = jnp.zeros((batch, 2), dtype=jnp.float32)
        return next_hidden, reward, value, policy, projection, reward_hidden


@pytest.fixture
def grpc_server():
    network = DummyNetwork()
    server = ic.GrpcInferenceServer(
        network=network,
        host="127.0.0.1",
        port=0,
        batch_size=2,
        max_wait_ms=50,
    )
    server.start()
    server.start()
    yield server, network
    server.stop()
    server.stop()


def test_grpc_inference_batches_requests(grpc_server):
    server, network = grpc_server
    endpoint = server.endpoint
    client = ic.GrpcInferenceClient(endpoint, timeout_s=5.0)

    results = []

    def worker(seed):
        obs = jnp.ones((1, 3), dtype=jnp.float32) * seed
        results.append(client.initial_inference(obs))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert network.initial_calls == 1
    assert len(results) == 2
    hidden = jnp.ones((1, 4), dtype=jnp.float32)
    action = jnp.ones((1, 1), dtype=jnp.float32)
    client.recurrent_inference(hidden, action)
    client.close()


def test_orchestrator_uses_grpc_client_when_remote_enabled(grpc_server, mock_wandb, temp_save_path):
    server, _ = grpc_server
    cfg = OmegaConf.load("open_spiel/python/algorithms/muzero_jax/configs/config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.output.save_path = temp_save_path
    cfg.actors.num_actors = 1
    cfg.bootstrap.enabled = False
    cfg.training.training_steps = 1
    cfg.training.start_transitions = cfg.training.batch_size
    cfg.evaluation.enabled = False
    cfg.inference.remote_enabled = True
    cfg.inference.rpc_endpoint = server.endpoint
    cfg.inference.batch_size = 2
    cfg.inference.max_wait_ms = 50

    orchestrator = MuZeroOrchestrator(cfg)
    try:
        clients = orchestrator._inference_clients
        assert len(clients) == cfg.actors.num_actors
        assert all(client.__class__.__name__ == "GrpcInferenceClient" for client in clients)
    finally:
        orchestrator.cleanup()
@pytest.fixture
def mock_wandb():
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
        yield


@pytest.fixture
def temp_save_path(tmp_path):
    path = tmp_path / "grpc_save"
    path.mkdir()
    return str(path)
