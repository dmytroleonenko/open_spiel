"""Defines the message protocol for remote inference requests and responses."""
from dataclasses import dataclass
from typing import Any, Tuple
import queue as _std_queue
from uuid import uuid4
import numpy as _np
import multiprocessing as mp # Ensure this import is present
# import pyspiel # Not strictly needed here if only dealing with np arrays and basic types
from open_spiel.python.algorithms import mcts
import time # Ensure time module is imported
import traceback # For traceback.format_exc() in _receive_response
import logging
import collections
import os # Added import

# Message type identifiers
INFERENCE_REQ: str = "inference_req"
INFERENCE_RESP: str = "inference_resp"

# --- EARLY DEBUG PRINT ---
print(f"[REMOTE_INFERENCE_DEBUG] Top of remote_inference.py. PID: {os.getpid()}", flush=True)

@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    actor_id: int
    observation: Any
    legals_mask: Any
    request_time: float # For monitoring staleness

    def to_tuple(self) -> Tuple:
        """Serialize to a tuple message."""
        return (INFERENCE_REQ, self.request_id, self.actor_id, self.observation, self.legals_mask, self.request_time)

    @staticmethod
    def from_tuple(message: Tuple) -> "InferenceRequest":
        """Deserialize a tuple message into an InferenceRequest."""
        type_, request_id, actor_id, observation, legals_mask, request_time = message
        if type_ != INFERENCE_REQ:
            raise ValueError(f"Invalid message type: {type_}, expected {INFERENCE_REQ}")
        return InferenceRequest(request_id=request_id, actor_id=actor_id, observation=observation, legals_mask=legals_mask, request_time=request_time)

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

SHUTDOWN_SENTINEL = object()

class ShutdownException(Exception):
    """Custom exception to signal graceful shutdown."""
    pass

class RemoteEvaluator:
    """An MCTS Evaluator that sends inference requests to a remote service."""

    def __init__(
        self,
        game: "pyspiel.Game",  # type: ignore # pylint: disable=undefined-variable
        actor_id: int,
        inference_request_queue: "_std_queue.Queue",  # type: ignore # pylint: disable=undefined-variable
        inference_response_queue: "_std_queue.Queue",  # type: ignore # pylint: disable=undefined-variable
        max_cache_size: int = 2**16,  # Aligns with AlphaZeroConfig default
        debug_mode: bool = False,
        log_path: str = "" # Added log_path parameter
    ):
        """Initializes a remote MCTS evaluator.

        Args:
          game: The game object.
          actor_id: A unique identifier for this actor/evaluator instance.
          inference_request_queue: A queue to send inference requests to.
          inference_response_queue: A queue to receive inference responses from.
          max_cache_size: Maximum size of the LRU cache for inference results.
          debug_mode: If True, enables more verbose logging for debugging.
          log_path: Directory to store log files. Defaults to current directory.
        """
        # --- EARLY INIT DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {actor_id} Init Start. PID: {os.getpid()}", flush=True)

        self._game = game
        self._actor_id = actor_id
        self._inference_request_queue = inference_request_queue
        self._inference_response_queue = inference_response_queue
        self._debug_mode = debug_mode

        # Setup dedicated logger for this RemoteEvaluator instance
        # --- PRE-LOGGER SETUP DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Pre-Logger Setup. debug_mode: {debug_mode}, log_path: '{log_path}'. PID: {os.getpid()}", flush=True)
        self.logger = logging.getLogger(f"RemoteEvaluator_Actor_{self._actor_id}")
        self.logger.setLevel(logging.DEBUG if self._debug_mode else logging.INFO) # Default to INFO, DEBUG if debug_mode
        
        # Create file handler
        filename = f"log-remote_evaluator_actor_{self._actor_id}.txt"
        # --- PRE-LOGFILE CREATION DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Pre-FileHandler. Filename: {filename}, log_path: '{log_path}'. PID: {os.getpid()}", flush=True)
        if log_path and not os.path.exists(log_path):
            # --- MKDIRS DEBUG PRINT ---
            print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Creating log_path: '{log_path}'. PID: {os.getpid()}", flush=True)
            os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, filename) if log_path else filename
        # Overwrite log file on each init for cleaner logs per run
        # --- FILEHANDLER INSTANTIATION DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Instantiating FileHandler with path: '{log_file_path}'. PID: {os.getpid()}", flush=True)
        file_handler = logging.FileHandler(log_file_path, mode='w') 
        file_handler.setLevel(logging.DEBUG) # Capture all levels in file
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        # --- POST-LOGGER SETUP DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Post-Logger Setup. PID: {os.getpid()}", flush=True)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.propagate = False # Avoid duplicate logs in parent/root logger if configured
        # else: # Handler already exists, useful if __init__ can be called multiple times
            # log_file_path = self.logger.handlers[0].baseFilename if self.logger.handlers else "N/A"


        # LRU Cache for inference results
        # The cache key will be a string representation of the observation tensor.
        # The value will be the (value, policy_probs) tuple.
        # --- PRE-CACHE INIT DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Pre-Cache Init. PID: {os.getpid()}", flush=True)
        self._cache = LRUCache(max_size=max_cache_size)
        self.logger.info(f"RemoteEvaluator for Actor {self._actor_id} initialized. Cache size: {max_cache_size}. Log file: {log_file_path}")

        # Metrics for inference rate
        # --- POST-INIT DEBUG PRINT ---
        print(f"[REMOTE_EVALUATOR_DEBUG] Actor {self._actor_id} Init End. PID: {os.getpid()}", flush=True)

        self._inference_count = 0
        self._first_inference_time = None
        self._last_log_time = time.time()
        self._log_interval_seconds = 10 # Log metrics every 10 seconds
        self._inferences_since_last_log = 0

        # Timeout for waiting for a specific response, in seconds
        self._response_wait_timeout_seconds = 1000.0 # e.g., 10 seconds
        self._response_get_interval_seconds = 0.1 # Timeout for individual queue.get() calls


    def _inference(self, state: "pyspiel.State") -> tuple[float, "_np.ndarray"]:  # type: ignore # pylint: disable=undefined-variable
        """Sends a state for inference and returns the value and policy."""
        # Check cache first
        obs_tensor = _np.asarray(state.observation_tensor()) # Ensure NumPy array
        obs_key = obs_tensor.tobytes() # Use .tobytes() for robust hashing

        cached_result = self._cache.get(obs_key)
        if cached_result is not None:
          # Old log removed: # if self._logger and self._log_level >= _TRACE_LEVEL:
          #   self._logger.print(f"Actor {self._actor_id}: Cache hit for state.")
          self.logger.debug(f"Actor {self._actor_id}: Cache hit for state.")
          return cached_result

        request_id = uuid4().hex
        # Old log removed: # if self._logger and self._log_level >= _DEBUG_LEVEL:
        #   self._logger.print(f"Actor {self._actor_id}: Sending inference request {request_id}")
        self.logger.debug(f"Actor {self._actor_id}: Sending inference request {request_id}")

        try:
          request = InferenceRequest(
              request_id=request_id,
              actor_id=self._actor_id,
              observation=obs_tensor, # Already NumPy array
              legals_mask=_np.asarray(state.legal_actions_mask()), # Ensure NumPy array
              request_time=time.time())
          self._inference_request_queue.put(request.to_tuple(), block=True)
        except Exception as e:
          self.logger.error(f"Actor {self._actor_id}: Error putting request {request_id} on queue: {e}")
          raise

        try:          
          wait_start_time = time.time()
          # --- [REMINF_DEBUG] Pre-response loop ---
          self.logger.debug(f"Actor {self._actor_id}: Entering wait loop for response to request {request_id}. Timeout: {self._response_wait_timeout_seconds}s, Interval: {self._response_get_interval_seconds}s")
          while True: # Loop to find the correct response
            if time.time() - wait_start_time > self._response_wait_timeout_seconds:
              self.logger.error(f"Actor {self._actor_id}: Timeout waiting for response for request {request_id} after {self._response_wait_timeout_seconds}s.")
              raise TimeoutError(f"Actor {self._actor_id}: Timeout waiting for response for request {request_id}")

            try:
              # Get with a short timeout to allow checking the overall wait_start_time
              # --- [REMINF_DEBUG] Pre-get from response queue ---
              self.logger.debug(f"Actor {self._actor_id}: Attempting .get() on response queue for request {request_id}. Loop time elapsed: {time.time() - wait_start_time:.2f}s")
              response_tuple = self._inference_response_queue.get(block=True, timeout=self._response_get_interval_seconds)
              # --- [REMINF_DEBUG] Post-get from response queue ---
              self.logger.debug(f"Actor {self._actor_id}: .get() returned for request {request_id}. Item type: {type(response_tuple)}, Item: {str(response_tuple)[:200]}")
            except _std_queue.Empty: # Timeout for this specific get() call
              # --- [REMINF_DEBUG] Response queue .get() timed out (Empty exception) ---
              self.logger.log(logging.DEBUG -1 if hasattr(logging, 'DEBUG') else 5, f"Actor {self._actor_id}: Response queue empty (timeout {self._response_get_interval_seconds}s) while waiting for {request_id}, retrying. Overall wait time: {time.time() - wait_start_time:.2f}s")
              continue # Continue to check overall timeout and retry get()

            if not isinstance(response_tuple, tuple):
              if response_tuple is SHUTDOWN_SENTINEL:
                  self.logger.info(f"Actor {self._actor_id}: Received SHUTDOWN_SENTINEL from response queue while waiting for {request_id}. Propagating shutdown.")
              else:
                  self.logger.error(
                      f"Actor {self._actor_id}: Received non-tuple, non-sentinel message from response queue while waiting for {request_id}. "
                      f"Type: {type(response_tuple)}, Value: {str(response_tuple)[:200]}. Treating as shutdown."
                  )
              try:
                  self._inference_response_queue.put(SHUTDOWN_SENTINEL, block=False)
              except Exception as e_put:
                   self.logger.warning(f"Actor {self._actor_id}: Error re-putting SHUTDOWN_SENTINEL on queue: {e_put}")
              raise ShutdownException("Invalid (non-tuple) message received while waiting for inference response.")

            # --- [REMINF_DEBUG] Response received, pre-deserialization ---
            self.logger.debug(f"Actor {self._actor_id}: Tuple message received from response queue: {response_tuple}. Attempting to deserialize for request {request_id}.")
            response = InferenceResponse.from_tuple(response_tuple)

            if response.request_id == request_id:
              # This is the response we were waiting for
              self.logger.debug(f"Actor {self._actor_id}: Received expected response for request {request_id}. Value: {response.value:.3f}")
              break # Exit the while True loop, proceed to process response
            else:
              # This is a response for a different request (e.g., stale)
              self.logger.warning(
                  f"Actor {self._actor_id}: Received response for {response.request_id} while waiting for {request_id}. Discarding stale/unexpected response."
              )
              # Loop again to get the next message
              continue
          
          # If loop broken, 'response' is the correct one.

          # Cache the successful result before returning
          self._cache.put(obs_key, (response.value, response.policy_probs))
          
          # Update inference metrics
          if self._first_inference_time is None:
              self._first_inference_time = time.time()
          self._inference_count += 1
          self._inferences_since_last_log += 1
          
          current_time = time.time()
          if current_time - self._last_log_time >= self._log_interval_seconds:
              elapsed_since_last_log = current_time - self._last_log_time
              if elapsed_since_last_log > 0:
                  inferences_per_sec_interval = self._inferences_since_last_log / elapsed_since_last_log
                  self.logger.info(f"Actor {self._actor_id}: Inferences in last {elapsed_since_last_log:.2f}s: {self._inferences_since_last_log}, Rate: {inferences_per_sec_interval:.2f} inf/s")
              
              if self._first_inference_time and (current_time - self._first_inference_time > 0):
                  overall_elapsed_time = current_time - self._first_inference_time
                  overall_inferences_per_sec = self._inference_count / overall_elapsed_time
                  self.logger.info(f"Actor {self._actor_id}: Total inferences: {self._inference_count}, Overall Rate: {overall_inferences_per_sec:.2f} inf/s (since first inference)")

              self._last_log_time = current_time
              self._inferences_since_last_log = 0
              if self.logger.handlers: # Ensure handler exists
                  self.logger.handlers[0].flush()
              
          return response.value, response.policy_probs

        except ShutdownException: # Re-raise if it's our specific shutdown signal
            raise
        except Exception as e:
          # Old log removed: # if self._logger and self._log_level >= _ERROR_LEVEL:
          #   self._logger.print(f"Actor {self._actor_id}: Error getting response for request {request_id} from queue: {e}", flush=True)
          self.logger.error(f"Actor {self._actor_id}: Error getting response for request {request_id} from queue: {e}")
          raise

    def evaluate(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        """Returns a value for the given state."""
        value, _ = self._inference(state)
        # Assuming a two-player zero-sum game.
        return _np.array([value, -value])

    def prior(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        """Returns a policy for the given state."""
        if state.is_chance_node():
            return state.chance_outcomes()
        elif state.is_terminal():
            return [] # Or handle as appropriate for MCTS, typically not called on terminal.

        _, policy_probs = self._inference(state)
        legal_actions = state.legal_actions()
        
        # Ensure policy_probs corresponds to legal_actions.
        # The policy_probs from inference should already be masked or aligned
        # with all possible actions. MCTS expects priors for legal actions only.
        
        priors = []
        for action in legal_actions:
            # policy_probs should be a vector over all actions.
            # We need to map the action to its index if policy_probs is not already
            # a dict or a structure that can be indexed by `action`.
            # Assuming policy_probs is a numpy array where indices match action numbers.
            if action < len(policy_probs):
                priors.append((action, policy_probs[action]))
            else:
                # This case should ideally not happen if policy_probs is for all actions
                # Old log removed: # if self._logger and self._log_level >= _WARN_LEVEL:
                #     self._logger.print(f"Actor {self._actor_id}: Action {action} out of bounds for policy_probs (len {len(policy_probs)}). Assigning 0 prior.", flush=True)
                self.logger.warning(f"Actor {self._actor_id}: Action {action} out of bounds for policy_probs (len {len(policy_probs)}). Assigning 0 prior.")
                priors.append((action, 0.0))
                
        # It's crucial that the sum of priors for legal actions is close to 1 (or normalized).
        # The inference should provide a valid probability distribution.
        # MCTS might normalize it anyway.
        return priors

    def cache_info(self):
        """Returns information about the cache."""
        # Old log removed: # if self._logger and self._log_level >= _INFO_LEVEL:
        #   self._logger.print(f"Actor {self._actor_id}: Cache Info: Size={self._cache.size()}, Max Size={self._cache.max_size}, Hits={self._cache.hits}, Misses={self._cache.misses}")
        # Get info from cache object correctly
        cache_stats = self._cache.info()
        ips_str = "N/A"
        if self._first_inference_time:
            elapsed = time.time() - self._first_inference_time
            if elapsed > 0:
                ips = self._inference_count / elapsed
                ips_str = f"{ips:.2f}"
        
        self.logger.info(f"Actor {self._actor_id}: Cache Info: Size={self._cache.size}, Max Size={self._cache.max_size}, Hits={cache_stats['hits']}, Misses={cache_stats['misses']}. Inferences: {self._inference_count}, Inf/Sec (overall): {ips_str}")
        if self.logger.handlers: # Ensure handler exists
            self.logger.handlers[0].flush()
        return cache_stats # Return the dict

    def clear_cache(self):
        """Clears the cache."""
        # Old log removed: # if self._logger and self._log_level >= _INFO_LEVEL:
        #   self._logger.print(f"Actor {self._actor_id}: Clearing cache.")
        self.logger.info(f"Actor {self._actor_id}: Clearing cache.")
        self._cache.clear()

# Helper for LRU Cache
# (A simple LRU cache implementation if not using an external library)
# For simplicity, using collections.OrderedDict as a basic LRU cache.
# A more robust LRU cache might be needed for high performance.
class LRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def info(self):
        return {"size": len(self.cache), "max_size": self.max_size, "hits": self.hits, "misses": self.misses}

    @property # Make size a property
    def size(self):
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

# Add pyspiel import if state methods are directly called (they are)
from open_spiel.python import games # Needed for pyspiel.State type hint if used
import pyspiel # For pyspiel.State

# Removed duplicated and commented out imports that were here 