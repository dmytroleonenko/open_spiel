"""
Adapter that lets inference clients pull latest parameters from a publisher.
"""

from __future__ import annotations

from typing import Optional

from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherClient,
)
from open_spiel.python.algorithms.muzero_jax.services.inference_client import InferenceClient


class ParameterRefreshingInferenceClient:
    """
    Wraps an existing InferenceClient and refreshes parameters from a publisher.
    For remote inference servers, this is a no-op; for local clients we just
    call refresh_params on the underlying inference client.
    """

    def __init__(self, base_client: InferenceClient, publisher_client: Optional[GrpcParameterPublisherClient]):
        self._base = base_client
        self._publisher = publisher_client

    def initial_inference(self, observation_batch, training: bool = False):
        return self._base.initial_inference(observation_batch, training=training)

    def recurrent_inference(self, hidden_state_batch, action_batch, training: bool = False):
        return self._base.recurrent_inference(hidden_state_batch, action_batch, training=training)

    def refresh_params(self):
        if hasattr(self._base, "refresh_params"):
            self._base.refresh_params()

    def close(self):
        if hasattr(self._base, "close"):
            self._base.close()
