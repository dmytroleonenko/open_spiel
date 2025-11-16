"""
Parameter publisher service used to distribute the latest MuZero weights.

The learner publishes `(params, step)` pairs; inference servers and actors can
poll or await a minimum step. Transport-agnostic core plus simple gRPC
wrappers, mirroring the inference/replay service style.
"""

from __future__ import annotations

import threading
import queue
import pickle
from typing import Any, Tuple

import grpc

_RPC_PUBLISH = "/muzero.ParameterService/Publish"
_RPC_LATEST = "/muzero.ParameterService/Latest"
_RPC_WAIT = "/muzero.ParameterService/WaitForStep"
_RPC_SUBSCRIBE = "/muzero.ParameterService/Subscribe"


class LocalParameterPublisher:
    """Thread-safe in-memory publisher with blocking wait semantics."""

    def __init__(self):
        self._params = None
        self._step = -1
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._subscribers: list[queue.Queue] = []

    def publish(self, params: Any, step: int) -> None:
        with self._cond:
            self._params = params
            self._step = int(step)
            self._cond.notify_all()
            for q in list(self._subscribers):
                try:
                    q.put_nowait((self._params, self._step))
                except queue.Full:  # pragma: no cover - best effort
                    continue

    def latest(self) -> Tuple[Any, int]:
        with self._lock:
            if self._params is None:
                raise RuntimeError("No parameters have been published yet")
            return self._params, self._step

    def wait_for_step(self, min_step: int, timeout_s: float | None = None) -> Tuple[Any, int]:
        def _ready():
            return self._params is not None and self._step >= min_step

        with self._cond:
            ok = self._cond.wait_for(_ready, timeout=timeout_s)
            if not ok:
                raise TimeoutError(f"Timed out waiting for step {min_step}")
            return self._params, self._step

    def subscribe(self, min_step: int = 0):
        """Generator yielding params/step as they arrive."""
        q: queue.Queue = queue.Queue(maxsize=8)
        with self._lock:
            self._subscribers.append(q)
            if self._params is not None and self._step >= min_step:
                try:
                    q.put_nowait((self._params, self._step))
                except queue.Full:  # pragma: no cover
                    pass
        last_step = -1
        try:
            while True:
                params, step = q.get()
                if step < min_step or step == last_step:
                    continue
                last_step = step
                yield params, step
        finally:  # pragma: no cover
            with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)


class GrpcParameterPublisherServer:
    """gRPC wrapper exposing LocalParameterPublisher."""

    def __init__(self, backend: LocalParameterPublisher, host: str = "127.0.0.1", port: int = 0):
        self._backend = backend
        self._host = host
        self._port = port
        self._server = None
        self.endpoint = None

    def start(self):
        if self._server is not None:
            return
        from concurrent import futures

        self._server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=4))
        handler = grpc.method_handlers_generic_handler(
            "muzero.ParameterService",
            {
                "Publish": grpc.unary_unary_rpc_method_handler(
                    self._handle_publish,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "Latest": grpc.unary_unary_rpc_method_handler(
                    self._handle_latest,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "WaitForStep": grpc.unary_unary_rpc_method_handler(
                    self._handle_wait,
                    request_deserializer=_deserialize_message,
                    response_serializer=_serialize_message,
                ),
                "Subscribe": grpc.unary_stream_rpc_method_handler(
                    self._handle_subscribe,
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
        if self._server is None:
            return
        self._server.stop(grace=None)
        self._server = None
        self.endpoint = None

    # Handlers ------------------------------------------------------------
    def _handle_publish(self, request, context):  # pragma: no cover - exercised via client tests
        self._backend.publish(request["params"], int(request["step"]))
        return {"ok": True}

    def _handle_latest(self, request, context):  # pragma: no cover - exercised via client tests
        params, step = self._backend.latest()
        return {"params": params, "step": step}

    def _handle_wait(self, request, context):  # pragma: no cover - exercised via client tests
        params, step = self._backend.wait_for_step(int(request["min_step"]), timeout_s=request.get("timeout_s"))
        return {"params": params, "step": step}

    def _handle_subscribe(self, request, context):  # pragma: no cover
        min_step = int(request.get("min_step", 0))
        # For now, stream only the next available params >= min_step, then end stream.
        params, step = self._backend.wait_for_step(min_step=min_step, timeout_s=None)
        yield {"params": params, "step": step}


class GrpcParameterPublisherClient:
    """Client for Parameter publisher gRPC service."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0):
        self._endpoint = endpoint
        self._timeout = timeout_s
        self._channel = grpc.insecure_channel(endpoint)
        self._publish_stub = self._channel.unary_unary(
            _RPC_PUBLISH, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._latest_stub = self._channel.unary_unary(
            _RPC_LATEST, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._wait_stub = self._channel.unary_unary(
            _RPC_WAIT, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )
        self._subscribe_stub = self._channel.unary_stream(
            _RPC_SUBSCRIBE, request_serializer=_serialize_message, response_deserializer=_deserialize_message
        )

    def publish(self, params: Any, step: int) -> None:
        self._publish_stub({"params": params, "step": int(step)}, timeout=self._timeout)

    def latest(self) -> Tuple[Any, int]:
        result = self._latest_stub({}, timeout=self._timeout)
        return result["params"], int(result["step"])

    def wait_for_step(self, min_step: int, timeout_s: float | None = None) -> Tuple[Any, int]:
        result = self._wait_stub({"min_step": int(min_step), "timeout_s": timeout_s}, timeout=self._timeout)
        return result["params"], int(result["step"])

    def close(self):
        if self._channel:
            self._channel.close()
            self._channel = None

    def subscribe(self, min_step: int = 0, timeout_s: float | None = None):
        call = self._subscribe_stub({"min_step": int(min_step)}, timeout=timeout_s)
        for item in call:
            yield item["params"], int(item["step"])
def _serialize_message(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_message(data: bytes) -> Any:
    return pickle.loads(data)
