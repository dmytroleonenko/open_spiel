"""
Inference client interfaces for MuZero actors.

`Actor` instances call into an `InferenceClient` rather than directly touching
the MuZero network. Only local paths are supported; remote RPC inference was
removed because mctx/JAX actors run under JIT and cannot block on RPC.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, Optional, Protocol, Sequence, Tuple
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.services.inference_server import (
    BatchingInferenceServer,
)
from open_spiel.python.algorithms.muzero_jax.training.losses import support_to_scalar, symexp


InferenceOutput = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, object, object]


def _convert_scalar_value(network, value_output, is_reward=False):
    """Converts network output (categorical logits or symlog) to scalar value for MCTS."""
    config = network.config
    loss_type = config.reward_loss_type if is_reward else config.value_loss_type

    if loss_type in ["categorical", "kl"]:
        support_size = config.reward_support_size if is_reward else config.value_support_size
        num_atoms = support_size if support_size > 0 else 601
        # Use configured support range (defaulting to [-300, 300] if not set)
        support_min = getattr(config, 'support_min', -300.0)
        support_max = getattr(config, 'support_max', 300.0)
        return support_to_scalar(
            value_output,
            support_min=support_min,
            support_max=support_max,
            num_atoms=num_atoms
        )
    elif loss_type == "symlog":
        base = getattr(config, 'symlog_base', jnp.e)
        return symexp(value_output, base=base)
    else:
        return value_output


def _process_inference_output(network, output: InferenceOutput) -> InferenceOutput:
    """Processes raw network output to ensure values/rewards are scalars for MCTS."""
    hidden_state, reward, value, policy_logits, projected_output, reward_hidden = output

    value = _convert_scalar_value(network, value, is_reward=False)
    reward = _convert_scalar_value(network, reward, is_reward=True)

    return (hidden_state, reward, value, policy_logits, projected_output, reward_hidden)


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
        output = self.network.initial_inference(observation_batch, training=training)
        return _process_inference_output(self.network, output)

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False) -> InferenceOutput:
        output = self.network.recurrent_inference(hidden_state_batch, action_batch, training=training)
        return _process_inference_output(self.network, output)

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
        outputs = _process_inference_output(self._network, outputs)
        return _split_batched_outputs(outputs, len(batch))

    async def _batched_recurrent(self, batch: Sequence[Tuple[Any, Any]]) -> Sequence[InferenceOutput]:
        hidden = jnp.concatenate([item[0] for item in batch], axis=0)
        actions = jnp.concatenate([item[1] for item in batch], axis=0)
        outputs = self._network.recurrent_inference(hidden, actions, training=False)
        outputs = _process_inference_output(self._network, outputs)
        return _split_batched_outputs(outputs, len(batch))


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
