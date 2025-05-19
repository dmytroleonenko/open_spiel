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
import threading

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

        # For handling concurrent requests within the same actor (e.g. async MCTS)
        self._pending_requests_map = {}
        self._map_lock = threading.Lock()
        self._response_handler_thread = None
        self._shutdown_event = threading.Event()

        self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} initialization complete.")

    def start_response_handler(self):
        """Starts the dedicated response handler thread."""
        if self._response_handler_thread is None:
            self._shutdown_event.clear()
            self._response_handler_thread = threading.Thread(
                target=self._handle_responses_loop,
                name=f"RemoteEvaluator-ResponseHandler-{self._actor_id}",
                daemon=True)
            self._response_handler_thread.start()
            self.logger.info(f"Actor {self._actor_id}: Response handler thread started.")

    def stop_response_handler(self):
        """Stops the dedicated response handler thread."""
        self.logger.info(f"Actor {self._actor_id}: Stopping response handler thread...")
        self._shutdown_event.set()
        try:
            # Put a sentinel to unblock the queue.get() if it's waiting.
            self._inference_response_queue.put(SHUTDOWN_SENTINEL, block=False, timeout=1.0)
        except _std_queue.Full:
            self.logger.warning(f"Actor {self._actor_id}: Response queue full when trying to put SHUTDOWN_SENTINEL.")
        except Exception as e: # pylint: disable=broad-except
            self.logger.error(f"Actor {self._actor_id}: Error putting SHUTDOWN_SENTINEL on response queue: {e}")

        if self._response_handler_thread and self._response_handler_thread.is_alive():
            self._response_handler_thread.join(timeout=5.0)
            if self._response_handler_thread.is_alive():
                self.logger.warning(f"Actor {self._actor_id}: Response handler thread did not terminate in time.")
            else:
                self.logger.info(f"Actor {self._actor_id}: Response handler thread stopped.")
        self._response_handler_thread = None

    def _handle_responses_loop(self):
        """Loop to continuously get responses from the shared queue and dispatch them."""
        self.logger.info(f"Actor {self._actor_id}: Response handler loop started.")
        try:
            while not self._shutdown_event.is_set():
                try:
                    response_tuple = self._inference_response_queue.get(block=True, timeout=0.1) # Short timeout to check shutdown_event
                except _std_queue.Empty:
                    continue # Timeout, check shutdown_event and loop again

                if self._shutdown_event.is_set(): # Check again after get()
                    self.logger.info(f"Actor {self._actor_id}: Shutdown event set, exiting response loop (after get).")
                    # If we got a real message before shutdown, try to process it.
                    if response_tuple is SHUTDOWN_SENTINEL and response_tuple is not None: # check for None, response_tuple could be None
                         break 
                    # if response_tuple is not None and response_tuple is not SHUTDOWN_SENTINEL:
                    #     pass # process it below
                    elif response_tuple is SHUTDOWN_SENTINEL: # it is SHUTDOWN_SENTINEL
                        break # exit loop
                    # else: response_tuple is None, this case should not happen with block=True and timeout
                        # but if it does, it's safer to break
                        # break

                if response_tuple is SHUTDOWN_SENTINEL:
                    self.logger.info(f"Actor {self._actor_id}: Received SHUTDOWN_SENTINEL in response handler loop. Exiting.")
                    break

                if not isinstance(response_tuple, tuple):
                    self.logger.error(
                        f"Actor {self._actor_id}: Response handler received non-tuple, non-sentinel message. "
                        f"Type: {type(response_tuple)}, Value: {str(response_tuple)[:200]}. Discarding."
                    )
                    continue

                try:
                    response = InferenceResponse.from_tuple(response_tuple)
                    self.logger.debug(f"Actor {self._actor_id}: Response handler received response for request_id: {response.request_id}")
                except ValueError as e:
                    self.logger.error(f"Actor {self._actor_id}: Response handler failed to deserialize response tuple {response_tuple}: {e}. Discarding.")
                    continue

                with self._map_lock:
                    pending_item = self._pending_requests_map.pop(response.request_id, None)

                if pending_item:
                    event, response_data_holder = pending_item
                    response_data_holder.append(response)
                    event.set()
                    self.logger.debug(f"Actor {self._actor_id}: Response handler processed and signalled for request_id: {response.request_id}")
                else:
                    # This can happen if the request timed out in _inference() and was removed from the map.
                    self.logger.warning(
                        f"Actor {self._actor_id}: Response handler received response for unknown or timed-out request_id: {response.request_id}. Discarding."
                    )
        except Exception as e: # pylint: disable=broad-except
            self.logger.error(f"Actor {self._actor_id}: Exception in response handler loop: {e}\\n{traceback.format_exc()}")
        finally:
            self.logger.info(f"Actor {self._actor_id}: Response handler loop terminating. Cleaning up pending requests.")
            # Post-loop cleanup: Signal any remaining waiters that we are shutting down.
            with self._map_lock:
                for request_id, (event, response_data_holder) in self._pending_requests_map.items():
                    self.logger.warning(f"Actor {self._actor_id}: Forcing shutdown signal for pending request_id: {request_id} during cleanup.")
                    # Add a marker to indicate shutdown, so _inference knows this isn't a real response.
                    response_data_holder.append(SHUTDOWN_SENTINEL) 
                    event.set()
                self._pending_requests_map.clear()
            self.logger.info(f"Actor {self._actor_id}: Response handler loop finished cleanup.")

    def _inference(self, state: "pyspiel.State", mcts_search_timeout_sec: float) -> tuple[float, "_np.ndarray"]:  # type: ignore # pylint: disable=undefined-variable
        """Sends a state for inference and returns the value and policy."""
        obs_tensor = _np.asarray(state.observation_tensor())
        obs_key = obs_tensor.tobytes()

        cached_result = self._cache.get(obs_key)
        if cached_result is not None:
          self.logger.debug(f"Actor {self._actor_id}: Cache hit for state obs_key: {obs_key[:16].hex()}...")
          return cached_result

        request_id = uuid4().hex
        event = threading.Event()
        response_holder = []

        with self._map_lock:
            self._pending_requests_map[request_id] = (event, response_holder)

        self.logger.debug(f"Actor {self._actor_id}: Cache miss. Sending inference request {request_id} for obs_key: {obs_key[:16].hex()}... Timeout: {mcts_search_timeout_sec}s")

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

        # Wait for the response handler to signal completion
        timed_out = not event.wait(timeout=mcts_search_timeout_sec)

        if timed_out:
            with self._map_lock:
                # Clean up the map entry if it still exists
                self._pending_requests_map.pop(request_id, None)
            self.logger.error(f"Actor {self._actor_id}: Timeout waiting for response event for request {request_id} after {mcts_search_timeout_sec}s.")
            raise TimeoutError(f"Actor {self._actor_id}: Timeout waiting for response event for request {request_id}")

        # Event was set, response should be in response_holder
        if not response_holder:
            # This can happen if the response handler loop terminated and cleaned up, but the event was set.
            self.logger.error(f"Actor {self._actor_id}: Event set for {request_id} but response_holder is empty. Assuming shutdown.")
            raise ShutdownException(f"Event set for {request_id} but no response data, likely shutdown.")

        response_content = response_holder[0]

        if response_content is SHUTDOWN_SENTINEL:
            self.logger.warning(f"Actor {self._actor_id}: Received shutdown sentinel for request {request_id} via response_holder. Propagating shutdown.")
            raise ShutdownException(f"Shutdown sentinel received for request {request_id}.")

        if not isinstance(response_content, InferenceResponse):
             self.logger.error(f"Actor {self._actor_id}: Invalid content in response_holder for {request_id}. Expected InferenceResponse, got {type(response_content)}. Assuming shutdown.")
             raise ShutdownException(f"Invalid content in response_holder for {request_id}.")

        response: InferenceResponse = response_content
        self.logger.debug(f"Actor {self._actor_id}: Received expected response for request {request_id} via event. Value: {response.value:.3f}")

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
        duration = end_time - request.request_time
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

    def evaluate(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        self.logger.warning(
            f"Actor {self._actor_id}: evaluate() called directly. This method is deprecated for RemoteEvaluator. "
            f"Use prior_and_value() with an appropriate timeout instead."
        )
        # Provide a default timeout for legacy calls, though it might not be optimal.
        # This maintains basic functionality but should be updated in calling code.
        default_timeout = self._response_wait_timeout_seconds # Or some other reasonable default
        try:
            return self.prior_and_value(state, mcts_search_timeout_sec=default_timeout)
        except TimeoutError:
            self.logger.error(f"Actor {self._actor_id}: Timeout in legacy evaluate() call. Returning (None, None).")
            return (None, None)
        except ShutdownException:
            self.logger.warning(f"Actor {self._actor_id}: Shutdown in legacy evaluate() call. Returning (None, None).")
            return (None, None)

    def prior(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        self.logger.warning(
            f"Actor {self._actor_id}: prior() called directly. This method is deprecated for RemoteEvaluator. "
            f"Use prior_and_value() with an appropriate timeout instead."
        )
        default_timeout = self._response_wait_timeout_seconds # Or some other reasonable default
        try:
            value, policy = self.prior_and_value(state, mcts_search_timeout_sec=default_timeout)
            return policy
        except TimeoutError:
            self.logger.error(f"Actor {self._actor_id}: Timeout in legacy prior() call. Returning None for policy.")
            return None
        except ShutdownException:
            self.logger.warning(f"Actor {self._actor_id}: Shutdown in legacy prior() call. Returning None for policy.")
            return None

    def prior_and_value(self, state: "pyspiel.State", mcts_search_timeout_sec: float) -> Tuple[Any, Any]: # type: ignore # pylint: disable=undefined-variable
        """Returns value and policy for a state, using remote inference and caching."""
        # Note: mcts_search_timeout_sec is the timeout for the *overall* MCTS search step,
        # not just this single inference. We use a derived, shorter timeout for the queue wait.
        # The actual inference on the server side might also have its own timeout.

        current_player = state.current_player()
        observation = state.observation_tensor()
        legals_mask = _np.array(state.legal_actions_mask(current_player), dtype=_np.bool_)
        cache_key = (tuple(observation), tuple(legals_mask))

        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            if self.logger.isEnabledFor(TRACE_LEVEL_NUM): # TRACE
                self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Cache hit for state.")
            return cached_result
        
        if self.logger.isEnabledFor(TRACE_LEVEL_NUM): # TRACE
            self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Cache miss. Requesting inference.")

        request_id = uuid4().hex
        request_event = threading.Event() # Event to signal response received for this request
        response_data_container = {} # To store response or exception

        with self._map_lock:
            self._pending_requests_map[request_id] = (request_event, response_data_container)

        req = InferenceRequest(
            request_id=request_id,
            actor_id=self._actor_id, # Use the stored actor_id
            observation=_np.array(observation, dtype=_np.float32),
            legals_mask=legals_mask,
            request_time=time.time()
        )

        if self.logger.isEnabledFor(logging.DEBUG): # DEBUG for every request sent
            self.logger.debug(f"INVESTIGATE_ACTOR_ID: RemoteEvaluator (actor_id {self._actor_id}) sending InferenceRequest with request_id: {request_id}, actor_id_in_req: {req.actor_id}")

        try:
            self._inference_request_queue.put(req.to_tuple(), timeout=self._response_wait_timeout_seconds) # Use a timeout
        except _std_queue.Full:
            self.logger.error(f"Actor {self._actor_id}: Inference request queue full for req_id {request_id}. Cannot send.")
            with self._map_lock:
                self._pending_requests_map.pop(request_id, None) # Clean up
            return self._default_error_response(state) # Return default error policy/value

        # Wait for the response handler to signal completion
        timed_out = not request_event.wait(timeout=self._response_wait_timeout_seconds)

        if timed_out:
            with self._map_lock:
                # Clean up the map entry if it still exists
                self._pending_requests_map.pop(request_id, None)
            self.logger.error(f"Actor {self._actor_id}: Timeout waiting for response event for request {request_id} after {self._response_wait_timeout_seconds}s.")
            raise TimeoutError(f"Actor {self._actor_id}: Timeout waiting for response event for request {request_id}")

        # Event was set, response should be in response_data_container
        if not response_data_container:
            # This can happen if the response handler loop terminated and cleaned up, but the event was set.
            self.logger.error(f"Actor {self._actor_id}: Event set for {request_id} but response_data_container is empty. Assuming shutdown.")
            raise ShutdownException(f"Event set for {request_id} but no response data, likely shutdown.")

        response_content = response_data_container[0]

        if response_content is SHUTDOWN_SENTINEL:
            self.logger.warning(f"Actor {self._actor_id}: Received shutdown sentinel for request {request_id} via response_data_container. Propagating shutdown.")
            raise ShutdownException(f"Shutdown sentinel received for request {request_id}.")

        if not isinstance(response_content, InferenceResponse):
             self.logger.error(f"Actor {self._actor_id}: Invalid content in response_data_container for {request_id}. Expected InferenceResponse, got {type(response_content)}. Assuming shutdown.")
             raise ShutdownException(f"Invalid content in response_data_container for {request_id}.")

        response: InferenceResponse = response_content
        self.logger.debug(f"Actor {self._actor_id}: Received expected response for request {request_id} via event. Value: {response.value:.3f}")

        self._cache.put(cache_key, (response.value, response.policy_probs))

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
        duration = end_time - req.request_time
        self._inference_durations.append(duration)

        # Log inference stats every stats window
        now = time.time()
        if now - self._stats_last_time >= self._stats_window_seconds:
            if self._inference_durations:
                avg_duration = _format_duration_s(sum(self._inference_durations) / len(self._inference_durations))
                min_duration = _format_duration_s(min(self._inference_durations))
                max_duration = _format_duration_s(max(self._inference_durations))
                p95_duration = _format_duration_s(_np.percentile(self._inference_durations, 95))
                self.logger.debug(
                    f"Actor {self._actor_id}: Inference duration stats (last {self._stats_window_seconds:.0f}s, {len(self._inference_durations)} samples): "
                    f"Avg: {avg_duration}, Min: {min_duration}, Max: {max_duration}, p95: {p95_duration}"
                )
                self._inference_durations.clear()
            self._stats_last_time = current_time

        return response.value, response.policy_probs

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