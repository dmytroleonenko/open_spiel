"""
Replay service abstractions for distributed actors/learners.

This module keeps a transport-agnostic in-memory service plus lightweight gRPC
wrappers. The API mirrors the minimal needs of the MuZero learner/actors:

* append(trajectory, priority)
* sample(batch_size) -> trajectories, ids, priorities
* update_priorities(ids, priorities)
* __len__()

All payloads are plain Python/NumPy objects; we intentionally avoid protobuf
schemas for now and rely on pickle (same as the inference RPC path).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Sequence, Tuple

import grpc
import numpy as np

from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    _make_numpy_rng,
)
from open_spiel.python.algorithms.muzero_jax.services.inference_client import (
    _deserialize_message,
    _serialize_message,
)


class InMemoryReplayService:
    """Thread-safe replay store with optional priority sampling."""

    def __init__(self, capacity: int, *, alpha: float = 0.0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self._capacity = capacity
        self._alpha = alpha
        self._store: Dict[int, Dict[str, Any]] = {}
        self._priorities: Dict[int, float] = {}
        self._order: List[int] = []
        self._next_id = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def append(self, trajectory: Dict[str, Any], priority: float = 1.0) -> int:
        """Append a trajectory and return its ID."""
        with self._lock:
            traj_id = self._next_id
            self._next_id += 1
            copied = _deep_copy_traj(trajectory)
            if len(self._order) >= self._capacity:
                old = self._order.pop(0)
                self._store.pop(old, None)
                self._priorities.pop(old, None)
            self._order.append(traj_id)
            self._store[traj_id] = copied
            self._priorities[traj_id] = float(priority)
            self._not_empty.notify_all()
            return traj_id

    def sample(
        self, batch_size: int, rng_key: Any = None, timeout_s: float | None = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Sample trajectories, blocking until enough data exists."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._not_empty:
            ok = self._not_empty.wait_for(lambda: len(self._order) >= batch_size, timeout=timeout_s)
            if not ok:
                raise TimeoutError(f"Timed out waiting for {batch_size} items (have {len(self._order)})")
            ids = self._select_ids(batch_size, rng_key=rng_key)
            trajectories = [_deep_copy_traj(self._store[int(tid)]) for tid in ids]
            priorities = np.array([self._priorities[int(tid)] for tid in ids], dtype=np.float32)
            return trajectories, ids, priorities

    def update_priorities(self, ids: Sequence[int], priorities: Sequence[float]) -> None:
        if len(ids) != len(priorities):
            raise ValueError("ids and priorities must be same length")
        with self._lock:
            for tid, prio in zip(ids, priorities):
                if tid not in self._priorities:
                    continue
                self._priorities[int(tid)] = float(prio)
            self._not_empty.notify_all()

    def get_trajectory(self, traj_id: int) -> Dict[str, Any]:
        with self._lock:
            if traj_id not in self._store:
                raise KeyError(f"trajectory id {traj_id} not found")
            return _deep_copy_traj(self._store[traj_id])

    def update_targets(self, traj_id: int, *, policy_targets=None, value_targets=None) -> None:
        with self._lock:
            if traj_id not in self._store:
                raise KeyError(f"trajectory id {traj_id} not found")
            stored = self._store[traj_id]
            if policy_targets is not None:
                stored["policy_targets"] = _deep_copy_traj({"p": policy_targets})["p"]
            if value_targets is not None:
                stored["value_targets"] = _deep_copy_traj({"v": value_targets})["v"]

    # Internal helpers -----------------------------------------------------

    def _select_ids(self, batch_size: int, rng_key: Any) -> np.ndarray:
        available = np.array(self._order, dtype=np.int64)
        if batch_size > available.size:
            raise ValueError("Requested more samples than available")
        rng = _make_numpy_rng(rng_key)
        weights = np.array([self._priorities[i] for i in available], dtype=np.float64)
        if self._alpha == 0.0:
            probs = None
        else:
            powered = np.power(weights, self._alpha)
            total = powered.sum()
            probs = powered / total if total > 0 else None
        replace = batch_size > available.size
        return rng.choice(available, size=batch_size, replace=replace, p=probs)


def _deep_copy_traj(traj: Dict[str, Any]) -> Dict[str, Any]:
    copied = {}
    for key, value in traj.items():
        copied[key] = np.array(value, copy=True)
    return copied


# ----------------------- gRPC wrappers ------------------------------------

_RPC_APPEND = "/muzero.ReplayService/Append"
_RPC_SAMPLE = "/muzero.ReplayService/Sample"
_RPC_UPDATE = "/muzero.ReplayService/UpdatePriorities"
_RPC_GET = "/muzero.ReplayService/GetTrajectory"
_RPC_UPDATE_TARGETS = "/muzero.ReplayService/UpdateTargets"
_RPC_SIZE = "/muzero.ReplayService/Size"


class GrpcReplayServer:
    """Thin gRPC wrapper that exposes an InMemoryReplayService over RPC."""

    def __init__(self, backend: InMemoryReplayService, host: str = "127.0.0.1", port: int = 0):
        self._backend = backend
        self._host = host
        self._port = port
        self._server = None
        self.endpoint = None

    def start(self):
        if self._server is not None:  # pragma: no cover - idempotent
            return
        self._server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=8))
        handler = grpc.method_handlers_generic_handler(
            "muzero.ReplayService",
            {
                "Append": grpc.unary_unary_rpc_method_handler(
                    self._handle_append,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "Sample": grpc.unary_unary_rpc_method_handler(
                    self._handle_sample,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "UpdatePriorities": grpc.unary_unary_rpc_method_handler(
                    self._handle_update,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "GetTrajectory": grpc.unary_unary_rpc_method_handler(
                    self._handle_get,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "UpdateTargets": grpc.unary_unary_rpc_method_handler(
                    self._handle_update_targets,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "Size": grpc.unary_unary_rpc_method_handler(
                    self._handle_size,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
            },
        )
        self._server.add_generic_rpc_handlers((handler,))
        bound_port = self._server.add_insecure_port(f"{self._host}:{self._port}")
        self._server.start()
        self.endpoint = f"{self._host}:{bound_port}"

    def stop(self):
        if self._server is None:  # pragma: no cover - idempotent
            return
        self._server.stop(grace=None)
        self._server = None
        self.endpoint = None

    # Handlers ------------------------------------------------------------
    def _handle_append(self, request, context):  # pragma: no cover - exercised via client tests
        trajectory = request["trajectory"]
        priority = request.get("priority", 1.0)
        traj_id = self._backend.append(trajectory, priority=priority)
        return {"trajectory_id": traj_id}

    def _handle_sample(self, request, context):  # pragma: no cover - exercised via client tests
        batch_size = int(request["batch_size"])
        trajectories, ids, priorities = self._backend.sample(batch_size=batch_size, rng_key=None)
        return {
            "trajectories": trajectories,
            "ids": np.asarray(ids, dtype=np.int64),
            "priorities": np.asarray(priorities, dtype=np.float32),
        }

    def _handle_update(self, request, context):  # pragma: no cover - exercised via client tests
        ids = request["ids"]
        priorities = request["priorities"]
        self._backend.update_priorities(ids, priorities)
        return {"ok": True}

    def _handle_get(self, request, context):  # pragma: no cover - exercised via client tests
        trajectory = self._backend.get_trajectory(int(request["trajectory_id"]))
        return {"trajectory": trajectory}

    def _handle_update_targets(self, request, context):  # pragma: no cover - exercised via client tests
        traj_id = int(request["trajectory_id"])
        self._backend.update_targets(
            traj_id,
            policy_targets=request.get("policy_targets"),
            value_targets=request.get("value_targets"),
        )
        return {"ok": True}

    def _handle_size(self, request, context):  # pragma: no cover
        return {"size": len(self._backend)}


class GrpcReplayClient:
    """Client for the replay gRPC server."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0):
        self._endpoint = endpoint
        self._channel = grpc.insecure_channel(endpoint)
        self._timeout = timeout_s
        self._append_stub = self._channel.unary_unary(
            _RPC_APPEND, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._sample_stub = self._channel.unary_unary(
            _RPC_SAMPLE, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._update_stub = self._channel.unary_unary(
            _RPC_UPDATE, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._get_stub = self._channel.unary_unary(
            _RPC_GET, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._update_targets_stub = self._channel.unary_unary(
            _RPC_UPDATE_TARGETS, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._size_stub = self._channel.unary_unary(
            _RPC_SIZE, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )

    def append(self, trajectory: Dict[str, Any], priority: float = 1.0) -> int:
        result = self._append_stub(
            {"trajectory": trajectory, "priority": float(priority)},
            timeout=self._timeout,
        )
        return int(result["trajectory_id"])

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        result = self._sample_stub({"batch_size": int(batch_size)}, timeout=self._timeout)
        return result["trajectories"], np.asarray(result["ids"]), np.asarray(result["priorities"])

    def update_priorities(self, ids: Sequence[int], priorities: Sequence[float]) -> None:
        self._update_stub({"ids": list(ids), "priorities": list(priorities)}, timeout=self._timeout)

    def close(self):
        if self._channel:
            self._channel.close()
            self._channel = None

    def get_trajectory(self, traj_id: int) -> Dict[str, Any]:
        result = self._get_stub({"trajectory_id": int(traj_id)}, timeout=self._timeout)
        return result["trajectory"]

    def update_targets(self, traj_id: int, *, policy_targets=None, value_targets=None) -> None:
        self._update_targets_stub(
            {
                "trajectory_id": int(traj_id),
                "policy_targets": policy_targets,
                "value_targets": value_targets,
            },
            timeout=self._timeout,
        )

    def size(self) -> int:
        result = self._size_stub({}, timeout=self._timeout)
        return int(result["size"])


class RemoteReplayBufferAdapter:
    """
    Adapts a GrpcReplayClient to the replay buffer interface used by the
    orchestrator and reanalysis worker.
    """

    def __init__(self, client: GrpcReplayClient, *, prioritized: bool):
        self._client = client
        self._prioritized = prioritized
        self._approx_size = 0

    def __len__(self):
        try:
            return max(self._approx_size, self._client.size())
        except Exception:  # pragma: no cover - defensive for early server shutdown
            return self._approx_size

    def add_trajectory(self, trajectory):
        self._client.append(trajectory, priority=1.0)
        self._approx_size += 1

    def sample_batch(self, batch_size: int, rng_key=None):
        sample = self.sample_batch_with_ids(batch_size, rng_key=rng_key)
        if self._prioritized:
            trajectories, ids, weights = sample
            return trajectories, ids, weights
        trajectories, _ids = sample
        return trajectories

    def sample_batch_with_ids(self, batch_size: int, rng_key=None):
        trajectories, ids, priorities = self._client.sample(batch_size)
        if self._prioritized:
            importance = np.ones_like(priorities, dtype=np.float32)
            return trajectories, ids, importance
        return trajectories, ids

    def get_trajectory(self, traj_id: int):
        return self._client.get_trajectory(traj_id)

    def update_trajectory_targets(self, traj_id: int, *, policy_targets=None, value_targets=None):
        self._client.update_targets(traj_id, policy_targets=policy_targets, value_targets=value_targets)

    def update_priorities_by_ids(self, ids, priorities):
        self._client.update_priorities(ids, priorities)

    def update_priorities(self, ids, priorities):
        # Support prioritized path that calls update_priorities directly.
        self.update_priorities_by_ids(ids, priorities)


# Avoid circular dependency: futures from concurrent.futures
from concurrent import futures  # noqa: E402  (import after class definitions)
