"""
Inference client interfaces for MuZero actors.

`Actor` instances call into an `InferenceClient` rather than directly touching
the MuZero network. This allows us to swap in remote RPC implementations
without rewriting the actor logic.
"""

from __future__ import annotations

import asyncio
import pickle
import threading
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, Optional, Protocol, Sequence, Tuple

import grpc
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
    BatchingInferenceServer,
)


InferenceOutput = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, object, object]


class InferenceClient(Protocol):
    """Protocol describing the inference operations actors require."""

    def initial_inference(self, observation_batch, training: bool = False) -> InferenceOutput:
        ...

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False) -> InferenceOutput:
        ...

    def close(self) -> None:
        ...

    def refresh_params(self) -> None:
        """Optional hook to refresh parameters from a remote publisher."""
        ...


@dataclass
class LocalInferenceClient:
    """
    Synchronous inference client that simply calls into the MuZero network.

    This is the default path for local runs and unit tests. Remote actors can
    swap this out for a client that forwards requests over RPC.
    """

    network: MuZeroNetwork

    def initial_inference(self, observation_batch, training: bool = False) -> InferenceOutput:
        return self.network.initial_inference(observation_batch, training=training)

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False) -> InferenceOutput:
        return self.network.recurrent_inference(hidden_state_batch, action_batch, training=training)

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def refresh_params(self) -> None:  # pragma: no cover - local path is no-op
        return None


class LocalBatchingInferenceClient:
    """
    Uses :class:`BatchingInferenceServer` in a background asyncio loop so local
    actors can exercise the batching code-path without deploying a real RPC
    service. This is useful for development and tests, and mirrors the
    architecture planned for multi-process inference servers.
    """

    def __init__(
        self,
        network: MuZeroNetwork,
        *,
        batch_size: int = 32,
        max_wait_ms: int = 5,
    ):
        self._network = network
        self._batch_size = max(1, batch_size)
        self._max_wait_ms = max_wait_ms
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._initial_server = self._run_coro(self._create_server(self._batched_initial))
        self._recurrent_server = self._run_coro(self._create_server(self._batched_recurrent))
        self._closed = False

    def initial_inference(self, observation_batch, training: bool = False) -> InferenceOutput:
        del training  # Training flag unused in inference client wrapper
        return self._submit(self._initial_server, observation_batch)

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False) -> InferenceOutput:
        del training
        payload = (hidden_state_batch, action_batch)
        return self._submit(self._recurrent_server, payload)

    def close(self) -> None:
        if self._closed:
            return
        for server in (self._initial_server, self._recurrent_server):
            self._run_coro(server.shutdown())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._closed = True

    def refresh_params(self) -> None:
        # Local client holds live network; nothing to refresh.
        return None

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _create_server(self, infer_fn):
        return BatchingInferenceServer(
            infer_fn=infer_fn,
            batch_size=self._batch_size,
            max_wait_ms=self._max_wait_ms,
        )

    def _submit(self, server: BatchingInferenceServer, payload: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(server.submit(payload), self._loop)
        return future.result()

    async def _batched_initial(self, batch: Sequence[Any]) -> Sequence[InferenceOutput]:
        stacked = jnp.concatenate(batch, axis=0)
        outputs = self._network.initial_inference(stacked, training=False)
        return _split_batched_outputs(outputs, len(batch))

    async def _batched_recurrent(self, batch: Sequence[Tuple[Any, Any]]) -> Sequence[InferenceOutput]:
        hidden = jnp.concatenate([item[0] for item in batch], axis=0)
        actions = jnp.concatenate([item[1] for item in batch], axis=0)
        outputs = self._network.recurrent_inference(hidden, actions, training=False)
        return _split_batched_outputs(outputs, len(batch))


class GrpcInferenceClient(InferenceClient):
    """Inference client that issues RPCs to a remote inference service."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0):
        self._endpoint = endpoint
        self._timeout = timeout_s
        self._channel = grpc.insecure_channel(endpoint)
        self._initial_stub = self._channel.unary_unary(
            _RPC_METHOD_INITIAL,
            request_serializer=_serialize_message,
            response_deserializer=_deserialize_message,
        )
        self._recurrent_stub = self._channel.unary_unary(
            _RPC_METHOD_RECURRENT,
            request_serializer=_serialize_message,
            response_deserializer=_deserialize_message,
        )

    def initial_inference(self, observation_batch, training: bool = False) -> InferenceOutput:
        del training
        obs_np = np.asarray(observation_batch, dtype=np.float32)
        payload = {"observation": obs_np}
        result = self._initial_stub(payload, timeout=self._timeout)
        return _convert_rpc_result(result)

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False) -> InferenceOutput:
        del training
        hidden_np = np.asarray(hidden_state_batch, dtype=np.float32)
        action_np = np.asarray(action_batch, dtype=np.float32)
        payload = {"hidden_state": hidden_np, "action": action_np}
        result = self._recurrent_stub(payload, timeout=self._timeout)
        return _convert_rpc_result(result)

    def close(self) -> None:
        if self._channel:
            self._channel.close()
            self._channel = None

    def refresh_params(self) -> None:
        # Remote inference server is responsible for pulling parameters; client is stateless.
        return None


class GrpcInferenceServer:
    """Lightweight gRPC wrapper around :class:`BatchingInferenceServer`."""

    def __init__(
        self,
        network: MuZeroNetwork,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        batch_size: int = 32,
        max_wait_ms: int = 5,
    ):
        self._network = network
        self._host = host
        self._port = port
        self._batch_size = batch_size
        self._max_wait_ms = max_wait_ms
        self.endpoint: Optional[str] = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._initial_batcher = None
        self._recurrent_batcher = None
        self._grpc_server = None

    def start(self):
        if self._grpc_server is not None:
            return
        self._thread.start()
        self._initial_batcher = self._run_coro(self._create_batcher(self._run_initial_batch))
        self._recurrent_batcher = self._run_coro(self._create_batcher(self._run_recurrent_batch))
        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        handler = grpc.method_handlers_generic_handler(
            "muzero.InferenceService",
            {
                "InitialInference": grpc.unary_unary_rpc_method_handler(
                    self._handle_initial,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "RecurrentInference": grpc.unary_unary_rpc_method_handler(
                    self._handle_recurrent,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
            },
        )
        self._grpc_server.add_generic_rpc_handlers((handler,))
        bound_port = self._grpc_server.add_insecure_port(f"{self._host}:{self._port}")
        self._grpc_server.start()
        self.endpoint = f"{self._host}:{bound_port}"

    def stop(self):
        if self._grpc_server is None:
            return
        self._grpc_server.stop(grace=None)
        for batcher in (self._initial_batcher, self._recurrent_batcher):
            if batcher is not None:
                self._run_coro(batcher.shutdown())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._grpc_server = None
        self.endpoint = None

    async def _create_batcher(self, fn):
        return BatchingInferenceServer(
            infer_fn=fn,
            batch_size=self._batch_size,
            max_wait_ms=self._max_wait_ms,
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _submit(self, batcher: BatchingInferenceServer, payload: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(batcher.submit(payload), self._loop).result()

    def _handle_initial(self, request: Dict[str, Any], context):
        result = self._submit(self._initial_batcher, request["observation"])
        return result

    def _handle_recurrent(self, request: Dict[str, Any], context):
        payload = (request["hidden_state"], request["action"])
        result = self._submit(self._recurrent_batcher, payload)
        return result

    async def _run_initial_batch(self, batch: Sequence[np.ndarray]) -> Sequence[Dict[str, Any]]:
        tensor = jnp.concatenate([jnp.asarray(item) for item in batch], axis=0)
        outputs = self._network.initial_inference(tensor, training=False)
        per_sample = _split_batched_outputs(outputs, len(batch))
        return [_pack_rpc_result(sample) for sample in per_sample]

    async def _run_recurrent_batch(
        self, batch: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Sequence[Dict[str, Any]]:
        hidden = jnp.concatenate([jnp.asarray(item[0]) for item in batch], axis=0)
        actions = jnp.concatenate([jnp.asarray(item[1]) for item in batch], axis=0)
        outputs = self._network.recurrent_inference(hidden, actions, training=False)
        per_sample = _split_batched_outputs(outputs, len(batch))
        return [_pack_rpc_result(sample) for sample in per_sample]


class InferenceNetworkAdapter:
    """
    Adapter allowing components that expect a MuZeroNetwork-like object to call
    into an `InferenceClient`.

    mctx currently expects `.initial_inference` / `.recurrent_inference` methods,
    so we provide this shim to avoid duplicating logic when remote clients are
    supplied.
    """

    def __init__(self, client: InferenceClient):
        self._client = client

    def initial_inference(self, observation_batch, training: bool = False) -> InferenceOutput:
        return self._client.initial_inference(observation_batch, training=training)

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False) -> InferenceOutput:
        return self._client.recurrent_inference(hidden_state_batch, action_batch, training=training)


def build_inference_client(network: MuZeroNetwork, inference_cfg) -> InferenceClient:
    """
    Helper that constructs the appropriate inference client given the Hydra config.

    Args:
        network: MuZero network instance for local inference.
        inference_cfg: Config node (may be ``None``) with inference settings.
    """
    if inference_cfg is None:
        return LocalInferenceClient(network)

    remote_enabled = bool(getattr(inference_cfg, "remote_enabled", False))
    if remote_enabled:
        endpoint = getattr(inference_cfg, "rpc_endpoint", "")
        if not endpoint:
            raise ValueError("inference.rpc_endpoint must be set when remote_enabled=true.")
        timeout_s = float(getattr(inference_cfg, "timeout_s", 10.0))
        return GrpcInferenceClient(endpoint, timeout_s=timeout_s)

    if getattr(inference_cfg, "enable_local_batching", False):
        batch_size = getattr(inference_cfg, "batch_size", 32)
        max_wait_ms = getattr(inference_cfg, "max_wait_ms", 5)
        return LocalBatchingInferenceClient(
            network=network,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
        )

    return LocalInferenceClient(network)


def _split_batched_outputs(outputs: Any, batch_size: int) -> Sequence[InferenceOutput]:
    """Split batched network outputs into per-request tuples."""
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    per_sample = []
    for idx in range(batch_size):
        per_sample.append(tuple(_slice_component(component, idx) for component in outputs))
    return per_sample


def _slice_component(component: Any, idx: int) -> Any:
    if isinstance(component, (tuple, list)):
        return type(component)(_slice_component(sub, idx) for sub in component)
    if hasattr(component, "shape") and component.shape:
        return component[idx : idx + 1]
    return component


def _serialize_message(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_message(data: bytes) -> Any:
    return pickle.loads(data)


def _pack_rpc_result(result: InferenceOutput) -> Dict[str, np.ndarray]:
    hidden_state, reward, value, policy, _, reward_hidden = result
    return {
        "hidden_state": np.asarray(hidden_state),
        "reward": np.asarray(reward),
        "value": np.asarray(value),
        "policy_logits": np.asarray(policy),
        "reward_hidden": None if reward_hidden is None else np.asarray(reward_hidden),
    }


def _convert_rpc_result(result: Dict[str, Optional[np.ndarray]]) -> InferenceOutput:
    hidden = jnp.asarray(result["hidden_state"])
    reward = jnp.asarray(result["reward"])
    value = jnp.asarray(result["value"])
    policy = jnp.asarray(result["policy_logits"])
    reward_hidden = result["reward_hidden"]
    if reward_hidden is not None:
        reward_hidden = jnp.asarray(reward_hidden)
    return hidden, reward, value, policy, None, reward_hidden


_RPC_METHOD_INITIAL = "/muzero.InferenceService/InitialInference"
_RPC_METHOD_RECURRENT = "/muzero.InferenceService/RecurrentInference"
