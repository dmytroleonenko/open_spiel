"""Defines the message protocol for remote inference requests and responses."""
from dataclasses import dataclass
from typing import Any, Tuple
import queue as _std_queue
from uuid import uuid4
import numpy as _np
import multiprocessing as mp
import time
import traceback
import logging
import collections
import os
import sys
import math

# Message type identifiers
INFERENCE_REQ: str = "inference_req"
INFERENCE_RESP: str = "inference_resp"

# Define custom TRACE logging level (more verbose than DEBUG)
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

# Mapping from numeric log levels (0-4 from config) to Python's logging constants
REMOTE_LOGGING_LEVEL_MAP = {
    0: logging.ERROR,    # Corresponds to "ERROR" from config
    1: logging.WARNING,  # Corresponds to "WARN" from config
    2: logging.INFO,     # Corresponds to "INFO" from config
    3: logging.DEBUG,    # Corresponds to "DEBUG" from config
    4: TRACE_LEVEL_NUM,  # Corresponds to "TRACE" from config
}

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

def _format_duration_s(duration: float) -> str:
    """Format a duration in seconds to a human-readable string with appropriate units."""
    ns = duration * 1e9
    if ns < 1e3:
        return f"{ns:.0f}ns"
    elif ns < 1e6:
        return f"{ns/1e3:.2f}us"
    elif ns < 1e9:
        return f"{ns/1e6:.2f}ms"
    else:
        return f"{ns/1e9:.2f}s"

class RemoteEvaluator:
    """An MCTS Evaluator that sends inference requests to a remote service."""

    def __init__(
        self,
        game: "pyspiel.Game",  # type: ignore # pylint: disable=undefined-variable
        actor_id: int,
        inference_request_queue: "_std_queue.Queue",  # type: ignore # pylint: disable=undefined-variable
        inference_response_queue: "_std_queue.Queue",  # type: ignore # pylint: disable=undefined-variable
        numeric_log_level: int,
        max_cache_size: int = 2**17,
        log_path: str = ""
    ):
        """Initializes a remote MCTS evaluator.

        Args:
          game: The game object.
          actor_id: A unique identifier for this actor/evaluator instance.
          inference_request_queue: A queue to send inference requests to.
          inference_response_queue: A queue to receive inference responses from.
          numeric_log_level: Integer log level (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE).
          max_cache_size: Maximum size of the LRU cache for inference results.
          log_path: Directory to store log files. Defaults to current directory.
        """
        self._game = game
        self._actor_id = actor_id
        self._numeric_log_level = numeric_log_level
        self._inference_request_queue = inference_request_queue
        self._inference_response_queue = inference_response_queue

        actual_log_level = REMOTE_LOGGING_LEVEL_MAP.get(self._numeric_log_level, logging.INFO)
        
        self.logger = logging.getLogger(f"RemoteEvaluator_Actor_{self._actor_id}")
        self.logger.setLevel(actual_log_level)
        self.logger.propagate = False

        filename = f"log-remote_evaluator_actor_{self._actor_id}.txt"
        if log_path:
            if not os.path.exists(log_path):
                try:
                    os.makedirs(log_path, exist_ok=True)
                except OSError as e:
                    print(f"ERROR: Actor {self._actor_id}: Failed to create log directory {log_path}: {e}", file=sys.stderr)
        
        log_file_path = os.path.join(log_path, filename) if log_path else filename

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
            
        file_handler = logging.FileHandler(log_file_path, mode='w') 
        file_handler.setLevel(actual_log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

        self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} initializing with log_level {logging.getLevelName(actual_log_level)} (numeric: {self._numeric_log_level}). Log file: '{log_file_path}'. PID: {os.getpid()}")

        self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} Pre-Cache Init. PID: {os.getpid()}")
        self._cache = LRUCache(max_size=max_cache_size)
        self.logger.info(f"RemoteEvaluator for Actor {self._actor_id} initialized. Cache size: {max_cache_size}.")

        self._inference_count = 0
        self._first_inference_time = None
        self._last_log_time = time.time()
        self._log_interval_seconds = 10
        self._inferences_since_last_log = 0

        # Instrumentation: inference duration stats (30-second window)
        self._inference_durations = []
        self._stats_window_seconds = 30.0
        self._stats_last_time = time.time()

        self._response_wait_timeout_seconds = 1000.0
        self._response_get_interval_seconds = 0.1

        self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} initialization complete.")

    def _inference(self, state: "pyspiel.State") -> tuple[float, "_np.ndarray"]:  # type: ignore # pylint: disable=undefined-variable
        """Sends a state for inference and returns the value and policy."""
        obs_tensor = _np.asarray(state.observation_tensor())
        obs_key = obs_tensor.tobytes()

        cached_result = self._cache.get(obs_key)
        if cached_result is not None:
          self.logger.debug(f"Actor {self._actor_id}: Cache hit for state obs_key: {obs_key[:16].hex()}...")
          return cached_result

        request_id = uuid4().hex
        self.logger.debug(f"Actor {self._actor_id}: Cache miss. Sending inference request {request_id} for obs_key: {obs_key[:16].hex()}...")

        try:
          request = InferenceRequest(
              request_id=request_id,
              actor_id=self._actor_id,
              observation=obs_tensor,
              legals_mask=_np.asarray(state.legal_actions_mask()),
              request_time=time.time())
          self._inference_request_queue.put(request.to_tuple(), block=True)
        except Exception as e:
          self.logger.error(f"Actor {self._actor_id}: Error putting request {request_id} on queue: {e}")
          raise

        try:          
          wait_start_time = time.time()
          self.logger.debug(f"Actor {self._actor_id}: Entering wait loop for response to request {request_id}. Timeout: {self._response_wait_timeout_seconds}s, Interval: {self._response_get_interval_seconds}s")
          while True:
            if time.time() - wait_start_time > self._response_wait_timeout_seconds:
              self.logger.error(f"Actor {self._actor_id}: Timeout waiting for response for request {request_id} after {self._response_wait_timeout_seconds}s.")
              raise TimeoutError(f"Actor {self._actor_id}: Timeout waiting for response for request {request_id}")

            try:
              self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Attempting .get() on response queue for request {request_id}. Loop time elapsed: {time.time() - wait_start_time:.2f}s")
              response_tuple = self._inference_response_queue.get(block=True, timeout=self._response_get_interval_seconds)
              self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: .get() returned for request {request_id}. Item type: {type(response_tuple)}, Item: {str(response_tuple)[:200]}")
            except _std_queue.Empty:
              self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Response queue empty (timeout {self._response_get_interval_seconds}s) while waiting for {request_id}, retrying. Overall wait time: {time.time() - wait_start_time:.2f}s")
              continue

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

            self.logger.debug(f"Actor {self._actor_id}: Tuple message received from response queue: {response_tuple}. Attempting to deserialize for request {request_id}.")
            response = InferenceResponse.from_tuple(response_tuple)

            if response.request_id == request_id:
              self.logger.debug(f"Actor {self._actor_id}: Received expected response for request {request_id}. Value: {response.value:.3f}")
              break
            else:
              self.logger.warning(
                  f"Actor {self._actor_id}: Received response for {response.request_id} while waiting for {request_id}. Discarding stale/unexpected response."
              )
              continue
          
          self._cache.put(obs_key, (response.value, response.policy_probs))
          
          # Update inference counts and timing
          if self._first_inference_time is None:
              self._first_inference_time = time.time()
          self._inference_count += 1
          self._inferences_since_last_log += 1

          # Logging for inference rate
          current_time = time.time()
          if current_time - self._last_log_time >= self._log_interval_seconds:
              elapsed_since_last_log = current_time - self._last_log_time
              if elapsed_since_last_log > 0:
                  inferences_per_sec_interval = self._inferences_since_last_log / elapsed_since_last_log
                  self.logger.debug(f"Actor {self._actor_id}: Inferences in last {elapsed_since_last_log:.2f}s: {self._inferences_since_last_log}, Rate: {inferences_per_sec_interval:.2f} inf/s")
              if self._first_inference_time and (current_time - self._first_inference_time > 0):
                  overall_elapsed_time = current_time - self._first_inference_time
                  overall_inferences_per_sec = self._inference_count / overall_elapsed_time
                  self.logger.debug(f"Actor {self._actor_id}: Total inferences: {self._inference_count}, Overall Rate: {overall_inferences_per_sec:.2f} inf/s (since first inference)")
              self._last_log_time = current_time
              self._inferences_since_last_log = 0
              if self.logger.handlers:
                  self.logger.handlers[0].flush()

          # Record inference duration (wait time)
          end_time = time.time()
          duration = end_time - wait_start_time
          self._inference_durations.append(duration)

          # Log inference stats every stats window
          now = time.time()
          if now - self._stats_last_time >= self._stats_window_seconds:
              durations = self._inference_durations
              avg_dur = sum(durations) / len(durations)
              min_dur = min(durations)
              max_dur = max(durations)
              var = sum((d - avg_dur) ** 2 for d in durations) / len(durations)
              std_dur = math.sqrt(var)
              avg_str = _format_duration_s(avg_dur)
              min_str = _format_duration_s(min_dur)
              max_str = _format_duration_s(max_dur)
              std_str = _format_duration_s(std_dur)
              self.logger.info(f"Actor {self._actor_id} inference stats {self._stats_window_seconds:.0f}s: avg {avg_str}, min {min_str}, max {max_str}, std {std_str}")
              if self.logger.handlers:
                  self.logger.handlers[0].flush()
              # Reset stats window
              self._inference_durations = []
              self._stats_last_time = now

          return response.value, response.policy_probs

        except ShutdownException:
            raise
        except Exception as e:
          self.logger.error(f"Actor {self._actor_id}: Error getting response for request {request_id} from queue: {e}")
          raise

    def evaluate(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        """Returns a value for the given state."""
        value, _ = self._inference(state)
        return _np.array([value, -value])

    def prior(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        """Returns a policy for the given state."""
        if state.is_chance_node():
            return state.chance_outcomes()
        elif state.is_terminal():
            return []

        _, policy_probs = self._inference(state)
        legal_actions = state.legal_actions()
        
        priors = []
        for action in legal_actions:
            if action < len(policy_probs):
                priors.append((action, policy_probs[action]))
            else:
                self.logger.warning(f"Actor {self._actor_id}: Action {action} out of bounds for policy_probs (len {len(policy_probs)}). Assigning 0 prior.")
                priors.append((action, 0.0))
                
        return priors

    def cache_info(self):
        """Returns information about the cache."""
        cache_stats = self._cache.info()
        ips_str = "N/A"
        if self._first_inference_time:
            elapsed = time.time() - self._first_inference_time
            if elapsed > 0:
                ips = self._inference_count / elapsed
                ips_str = f"{ips:.2f}"
        
        self.logger.info(f"Actor {self._actor_id}: Cache Info: Size={self._cache.size}, Max Size={self._cache.max_size}, Hits={cache_stats['hits']}, Misses={cache_stats['misses']}. Inferences: {self._inference_count}, Inf/Sec (overall): {ips_str}")
        if self.logger.handlers:
            self.logger.handlers[0].flush()
        return cache_stats

    def clear_cache(self):
        """Clears the cache."""
        self.logger.info(f"Actor {self._actor_id}: Clearing cache.")
        self._cache.clear()

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

    @property
    def size(self):
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

from open_spiel.python import games
import pyspiel 