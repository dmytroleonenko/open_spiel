"""Defines the message protocol for remote inference requests and responses."""
from dataclasses import dataclass
from typing import Any, Tuple
import queue as _std_queue
from uuid import uuid4
import numpy as _np
import multiprocessing as mp # Ensure this import is present
# import pyspiel # Not strictly needed here if only dealing with np arrays and basic types
from open_spiel.python.algorithms import mcts

# Message type identifiers
INFERENCE_REQ: str = "inference_req"
INFERENCE_RESP: str = "inference_resp"

@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    actor_id: int
    observation: Any
    legals_mask: Any

    def to_tuple(self) -> Tuple:
        """Serialize to a tuple message."""
        return (INFERENCE_REQ, self.request_id, self.actor_id, self.observation, self.legals_mask)

    @staticmethod
    def from_tuple(message: Tuple) -> "InferenceRequest":
        """Deserialize a tuple message into an InferenceRequest."""
        type_, request_id, actor_id, observation, legals_mask = message
        if type_ != INFERENCE_REQ:
            raise ValueError(f"Invalid message type: {type_}, expected {INFERENCE_REQ}")
        return InferenceRequest(request_id=request_id, actor_id=actor_id, observation=observation, legals_mask=legals_mask)

@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    value: Any
    policy_probs: Any

    def to_tuple(self) -> Tuple:
        """Serialize to a tuple message."""
        return (INFERENCE_RESP, self.request_id, self.value, self.policy_probs)

    @staticmethod
    def from_tuple(message: Tuple) -> "InferenceResponse":
        """Deserialize a tuple message into an InferenceResponse."""
        type_, request_id, value, policy_probs = message
        if type_ != INFERENCE_RESP:
            raise ValueError(f"Invalid message type: {type_}, expected {INFERENCE_RESP}")
        return InferenceResponse(request_id=request_id, value=value, policy_probs=policy_probs)

SHUTDOWN_SENTINEL = ("shutdown",)

class ShutdownException(Exception):
    """Custom exception to signal graceful shutdown."""
    pass

class RemoteEvaluator(mcts.Evaluator):
    """A proxy evaluator that sends inference requests to a central service."""

    def __init__(self, actor_id: int, 
                 request_queue: mp.Queue, 
                 response_queue: mp.Queue, 
                 timeout_ms: int = 10000, # Default to 10 seconds in ms
                 logger=None, 
                 log_level: int = 0):
        """Initializes a remote evaluator.

        Args:
            actor_id: Unique ID for this actor/client process.
            request_queue: The multiprocessing.Queue to send requests on.
            response_queue: The multiprocessing.Queue to receive responses from.
            timeout_ms: Timeout in milliseconds for waiting for a response.
            logger: Optional logger instance.
            log_level: Optional log level for internal messages.
        """
        super().__init__() # mcts.Evaluator has no __init__ args
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_sec = timeout_ms / 1000.0 if timeout_ms is not None else None # Convert ms to seconds
        self.logger = logger
        self.log_level = log_level
        self._request_counter = 0

    def _send_request(self, observation, legals_mask):
        request_id = uuid4().hex
        # Create InferenceRequest object and convert to tuple
        inference_request_obj = InferenceRequest(
            request_id=request_id,
            actor_id=self.actor_id,
            observation=observation,
            legals_mask=legals_mask
        )
        request_tuple = inference_request_obj.to_tuple()
        try:
            self.request_queue.put(request_tuple, timeout=self.timeout_sec) 
        except _std_queue.Full:
            if self.logger and self.log_level >= 1: # WARN level
                self.logger.print(f"RemoteEvaluator (Actor {self.actor_id}): Request queue full. Request {request_id} might be dropped or delayed.")
            raise
        return request_id # Return the generated request_id

    def _receive_response(self, request_id):
        response_data_tuple = None
        try:
            # Get the raw message first
            raw_message = self.response_queue.get(timeout=self.timeout_sec)

            # Check for SHUTDOWN_SENTINEL before unpacking
            if raw_message == SHUTDOWN_SENTINEL:
                if self.logger and self.log_level >= 2: # INFO level
                    self.logger.print(f"RemoteEvaluator (Actor {self.actor_id}): Received SHUTDOWN_SENTINEL on response queue.")
                raise ShutdownException("Received shutdown sentinel on response queue")
            
            # If not sentinel, proceed to unpack (assuming it's a valid response tuple)
            response_type, resp_request_id, value, policy_probs = raw_message
            
            if response_type != INFERENCE_RESP or resp_request_id != request_id:
                raise RuntimeError(f"Invalid response type or ID received: {raw_message}, expected for ID {request_id}")
            
            response_data_tuple = value, policy_probs

        except _std_queue.Empty: # mp.Queue.get() raises queue.Empty on timeout
            if self.logger and self.log_level >= 1: # WARN level
                self.logger.print(f"RemoteEvaluator (Actor {self.actor_id}): Timeout waiting for response for request {request_id}")
            raise TimeoutError(f"Timeout waiting for inference response for request {request_id}") from None
        except ShutdownException: # Re-raise if it was a ShutdownException from SHUTDOWN_SENTINEL check
            raise
        except ValueError as ve: # Handles unpacking errors if raw_message is not as expected
            if self.logger and self.log_level >= 0: # ERROR level
                self.logger.print(f"RemoteEvaluator (Actor {self.actor_id}): Error unpacking response for req {request_id}: {ve}. Message: {raw_message}")
            raise RuntimeError(f"Error unpacking response for request {request_id}") from ve
        except Exception as e: # pylint: disable=broad-except
            if self.logger and self.log_level >= 0: # ERROR level
                self.logger.print(f"RemoteEvaluator (Actor {self.actor_id}): Generic error receiving response for req {request_id}: {e}. Message: {raw_message if 'raw_message' in locals() else 'N/A'}")
            # Ensure any other exception is also wrapped or re-raised appropriately
            # If it's not a Timeout or Shutdown, it's likely a RuntimeError or similar critical issue.
            if not isinstance(e, (TimeoutError, ShutdownException, RuntimeError)):
                 raise RuntimeError(f"Unexpected error receiving response for request {request_id}") from e
            else:
                raise # Re-raise known critical exceptions
        
        return response_data_tuple
    
    def evaluate(self, state):
        """Evaluates a state using the central inference model."""
        observation = _np.asarray(state.observation_tensor(), dtype=_np.float32)
        legals_mask = _np.asarray(state.legal_actions_mask(), dtype=_np.bool_)
        
        req_id = self._send_request(observation, legals_mask)
        value, _ = self._receive_response(req_id) # Use the same req_id, ignore policy for evaluate
        
        return _np.array([value, -value]) # For a zero-sum game

    def prior(self, state):
        """Computes the-policy prior for a state, potentially using remote inference."""
        if state.is_chance_node():
            return state.chance_outcomes()

        observation = _np.asarray(state.observation_tensor(), dtype=_np.float32)
        legals_mask = _np.asarray(state.legal_actions_mask(), dtype=_np.bool_)
        
        req_id = self._send_request(observation, legals_mask)
        _, policy_probs = self._receive_response(req_id) # Use the same req_id, ignore value for prior (or use if needed)
        
        return [(action, policy_probs[action]) for action in state.legal_actions() if policy_probs[action] > 0]

# Add pyspiel import if state methods are directly called (they are)
from open_spiel.python import games # Needed for pyspiel.State type hint if used
import pyspiel # For pyspiel.State

# Removed duplicated and commented out imports that were here 