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
        """Internal method to perform inference, returns (value, policy_probs)."""
        # Value here is the raw scalar value from the network for the current player.
        # Policy_probs is the raw policy array from the network.

        # Create a unique request ID
        request_id = str(uuid4())

        # Prepare the observation and legals_mask
        observation_tensor = _np.array(state.observation_tensor(), dtype=_np.float32)
        # Ensure observation_tensor is flat if the game returns a multi-dim observation
        # that the model expects flattened (unless model handles reshape).
        # For most AlphaZero models (MLP, Conv2D, some ResNets), flat or specific spatial is expected.
        # This was previously handled by game.state_to_feature_array in some actor logic.
        # For RemoteEvaluator, it should receive the canonical observation_tensor from the game.
        # Models like spatial_global_1dresnet_transformer expect a flat observation.
        # The model_jax.py's init_flax_model_and_variables correctly passes the game's
        # observation_tensor_shape to the model. If the model expects flat, it should handle it.
        # For robustness, ensure it's flat if that's the general expectation.
        # However, some models (like direct ResNeSt1D50_AZ) might expect non-flat.
        # For now, assume observation_tensor as is from game.observation_tensor() is correct.
        # If specific game requires flattening, it should be handled by a game-specific wrapper
        # or ensured that game.observation_tensor() provides the expected format.

        legals_mask_array = _np.array(state.legal_actions_mask(), dtype=_np.bool_)

        # Create the request object
        req = InferenceRequest(
            request_id=request_id,
            actor_id=self._actor_id,
            observation=observation_tensor,
            legals_mask=legals_mask_array,
            request_time=time.monotonic() # Use monotonic time for request timing
        )

        # Send the request
        if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
            self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Sending inference request {request_id} for state: {state.history_str() if hasattr(state, 'history_str') else 'N/A'}")
        
        response_future = None
        with self._map_lock:
            response_future = threading.Event() # Event to wait for the response
            self._pending_requests_map[request_id] = {"future": response_future, "data": None}

        try:
            self._inference_request_queue.put(req.to_tuple(), block=True, timeout=self._response_wait_timeout_seconds)
        except _std_queue.Full:
            self.logger.error(f"Actor {self._actor_id}: Inference request queue full. Request {request_id} dropped.")
            with self._map_lock:
                self._pending_requests_map.pop(request_id, None) # Clean up
            # Return a default "bad" value and uniform policy or raise an error
            # For MCTS, returning a very bad value (e.g., -infinity or -1 for normalized returns)
            # and a uniform policy might be a way to let MCTS explore other paths.
            # This matches the behavior if response times out.
            return self._get_default_eval_on_error(state)


        # Wait for the response (with timeout)
        if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
            self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Waiting for response for request {request_id}. Timeout: {mcts_search_timeout_sec:.2f}s")
        
        # Use the Event object to wait for the response
        if response_future.wait(timeout=mcts_search_timeout_sec):
            # Response received and processed by _handle_responses_loop
            with self._map_lock:
                processed_response_data = self._pending_requests_map.pop(request_id, {}).get("data")
            
            if processed_response_data:
                # processed_response_data should be a tuple (scalar_value, policy_array)
                scalar_nn_value, policy_probs_array = processed_response_data
                
                if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
                    self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id}: Received response for {request_id}. Value: {scalar_nn_value:.4f}, Policy sum: {_np.sum(policy_probs_array):.4f}")
                
                # IMPORTANT FIX: Convert scalar_nn_value to a Python float before returning,
                # to ensure it's handled correctly when constructing utility arrays later.
                # policy_probs_array should already be a 1D numpy array of floats.
                return float(scalar_nn_value), policy_probs_array
            else:
                # Should not happen if event was set and data was supposed to be there
                self.logger.error(f"Actor {self._actor_id}: Response event set for {request_id}, but no data found. This indicates a logic error in response handling.")
                return self._get_default_eval_on_error(state)

        else: # Timeout waiting for response
            self.logger.warning(f"Actor {self._actor_id}: Timeout waiting for inference response for request {request_id} (waited {mcts_search_timeout_sec:.2f}s).")
            with self._map_lock:
                self._pending_requests_map.pop(request_id, None) # Clean up on timeout
            return self._get_default_eval_on_error(state)


    def _get_default_eval_on_error(self, state: "pyspiel.State"): # type: ignore
        """Returns a default evaluation (e.g., 0 value, uniform policy) on error/timeout."""
        # Value: 0 for the current player (neutral)
        # Policy: Uniform over legal actions
        num_distinct_actions = self._game.num_distinct_actions()
        policy = _np.zeros(num_distinct_actions, dtype=_np.float32)
        
        if state.is_chance_node() or state.is_terminal():
            # For chance/terminal nodes, prior/evaluate might not be called or handled differently
            # by MCTS. If they are, a uniform policy over chance outcomes or empty policy for terminal
            # might be appropriate. Value should be actual returns if terminal.
            # This function is a fallback for network errors on *decision* nodes.
             pass # Policy remains zeros, value is 0.0

        else: # Decision node
            legal_actions = state.legal_actions(state.current_player())
            if legal_actions:
                prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    policy[action] = prob
            # else: no legal actions, policy remains zeros.

        # Return scalar 0.0 for value, and the policy array
        return 0.0, policy


    def evaluate(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        """Evaluate a state.

        This method is expected by the MCTSBot. It should return an array
        of utilities, one for each player.
        """
        # Use a default timeout for MCTS search leaf evaluations if not specified.
        # This timeout should be shorter than actor's game step timeout.
        # For now, let _inference handle its own timeout logic.
        # The mcts_search_timeout_sec is more relevant when prior_and_value is called
        # from an async MCTS that wants to limit wait time for a specific search path.
        # For a simple evaluate call, it can use a default internal timeout for the request.
        
        # _inference returns (scalar_value_for_current_player, policy_array)
        scalar_value_for_curr_player, _ = self._inference(state, mcts_search_timeout_sec=self._response_wait_timeout_seconds) # Use a longer default timeout for evaluation

        # Convert the scalar value (for the current player) to a utility array for all players.
        num_players = self._game.num_players()
        current_player = state.current_player()
        utility_array = _np.zeros(num_players, dtype=_np.float32)

        if state.is_terminal():
            utility_array = _np.array(state.returns(), dtype=_np.float32)
        elif state.is_chance_node():
            # MCTS usually doesn't evaluate chance nodes directly with the NN.
            # If it does, the concept of "value" is tricky. For now, treat as neutral.
             utility_array = _np.array(state.returns(), dtype=_np.float32) if hasattr(state, 'returns') else _np.zeros(num_players, dtype=_np.float32)
        else: # Decision node
            # scalar_value_for_curr_player is np.float32 as ensured by _inference
            py_float_val = float(scalar_value_for_curr_player) # Ensure it's a Python float

            if num_players == 1:
                utility_array[current_player] = py_float_val
            elif num_players == 2: # Assuming zero-sum for 2-player games
                utility_array[current_player] = py_float_val
                utility_array[1 - current_player] = -py_float_val
            else: # N-player general sum - assign to current, others 0 (simplification)
                self.logger.warning(
                    f"Actor {self._actor_id}: evaluate() for N-player ({num_players}p) game. "
                    f"Assigning NN value {py_float_val:.3f} to P{current_player}, others 0."
                )
                utility_array[current_player] = py_float_val
                # Other players' utilities remain 0.

        if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
            self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} evaluate() for P{current_player} returning utility_array: {utility_array}")
        return utility_array


    def prior(self, state: "pyspiel.State"):  # type: ignore # pylint: disable=undefined-variable
        """Return a policy for a state.

        This method is expected by MCTSBot. It returns a list of (action, prob)
        tuples.
        """
        if state.is_chance_node():
            return state.chance_outcomes()
        if state.is_terminal():
            return []

        # _inference returns (scalar_value_for_current_player, policy_array)
        _, policy_array = self._inference(state, mcts_search_timeout_sec=self._response_wait_timeout_seconds) # Use a longer default timeout for prior

        # Convert policy_array to list of (action, prob) for legal actions
        legal_actions = state.legal_actions(state.current_player())
        if not legal_actions:
            return []
            
        priors = []
        if policy_array is not None and len(policy_array) == self._game.num_distinct_actions():
            for action in legal_actions:
                priors.append((action, float(policy_array[action]))) # Ensure prob is float
        else: # Fallback to uniform if policy_array is bad
            self.logger.warning(f"Actor {self._actor_id}: Bad policy_array in prior(). Len: {len(policy_array) if policy_array is not None else 'None'}. Num distinct: {self._game.num_distinct_actions()}. Using uniform.")
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                priors.append((action, prob))
        
        if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
            self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} prior() for P{state.current_player()} returning: {priors[:5]}...")
        return priors

    # This method combines prior and value, primarily for internal use or by evaluators that can handle combined calls.
    # The main MCTSBot uses separate evaluate() and prior() calls.
    # However, our AlphaZeroJaxAsyncMCTSBot and RemoteMCTSBot (from actor_evaluator_logic) use this.
    def prior_and_value(self, state: "pyspiel.State", mcts_search_timeout_sec: float) -> Tuple[Any, Any]: # type: ignore # pylint: disable=undefined-variable
        """Returns (value_array_for_all_players, policy_array_for_all_distinct_actions)."""
        
        # Check cache first
        cache_key = state.observation_string() # Or another robust key
        cached_result = self._cache.get(cache_key)
        if cached_result:
            self._log_periodic_stats(cache_hit=True)
            if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
                 self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} prior_and_value CACHE HIT for state (history like: ...{state.history_str()[-50:] if hasattr(state, 'history_str') else 'N/A'}).")
            # Ensure cached result matches expected return format: (utility_array, policy_array)
            # The cache stores exactly what this function returns.
            return cached_result

        self._log_periodic_stats(cache_hit=False)
        if self._first_inference_time is None:
            self._first_inference_time = time.monotonic()

        # --- Actual Inference ---
        # _inference returns (scalar_value_for_current_player, raw_policy_array)
        scalar_nn_value, raw_policy_array = self._inference(state, mcts_search_timeout_sec=mcts_search_timeout_sec)
        # scalar_nn_value is already a Python float due to changes in _inference.
        # raw_policy_array is a 1D numpy array of floats.

        # --- Process value into utility_array for all players ---
        num_players = self._game.num_players()
        current_player = state.current_player() # Player at the current decision node
        
        utility_array = _np.zeros(num_players, dtype=_np.float32)

        if state.is_terminal():
            utility_array = _np.array(state.returns(), dtype=_np.float32)
        elif state.is_chance_node():
            # This case might be complex; MCTS typically doesn't ask for NN eval of chance nodes this way.
            # If it occurs, game returns or a neutral utility might be appropriate.
            # For now, let's assume returns() is valid for post-chance states if relevant, or zeros.
            utility_array = _np.array(state.returns(), dtype=_np.float32) if hasattr(state, 'returns') and callable(state.returns) else _np.zeros(num_players, dtype=_np.float32)
        else: # Decision node, use NN output (scalar_nn_value)
            # scalar_nn_value is already float
            if num_players == 1:
                utility_array[current_player] = scalar_nn_value
            elif num_players == 2: # Assuming zero-sum for 2-player games
                utility_array[current_player] = scalar_nn_value
                utility_array[1 - current_player] = -scalar_nn_value # Ensure opposite for opponent
            else: # N-player game
                self.logger.warning(
                    f"Actor {self._actor_id}: prior_and_value() for N-player ({num_players}p) game. "
                    f"Assigning NN value {scalar_nn_value:.3f} to P{current_player}, others 0."
                )
                utility_array[current_player] = scalar_nn_value
                # Other players' utilities remain 0. This is a simplification.
                # A more general approach might require the NN to output a utility vector,
                # or use game-specific logic to distribute the value.
        
        # --- Policy processing (ensure it's a flat numpy array for all distinct actions) ---
        # raw_policy_array from _inference is already in the desired format.
        # No further processing needed for policy_array itself here unless masking/normalization
        # specific to prior_and_value's contract is required (MCTSBot.prior expects list of tuples).
        # For this combined call, returning the raw policy array is usually fine if the consumer expects it.
        # The MCTSBot uses .prior() which formats it into list of (action, prob) for legal actions.
        # AlphaZeroJaxAsyncMCTSBot (in actor_evaluator_logic) consumes prior_and_value directly.
        # Its _AsyncRemoteEvaluatorAdapter converts policy_array to prior_tuples if needed.
        # The AlphaZeroBot (sync) uses .evaluate() and .prior() separately.
        
        final_policy_output = raw_policy_array # Should be a 1D numpy array of floats

        # Store in cache
        self._cache.put(cache_key, (utility_array, final_policy_output))

        if self.logger.level <= TRACE_LEVEL_NUM: # TRACE
            self.logger.log(TRACE_LEVEL_NUM, f"Actor {self._actor_id} prior_and_value CACHE MISS for state (history like: ...{state.history_str()[-50:] if hasattr(state, 'history_str') else 'N/A'}). Returning: util_arr={utility_array}, pol_sum={_np.sum(final_policy_output):.3f}")
        
        self._inference_count += 1
        self._inferences_since_last_log += 1
        # Record inference duration for stats
        # Note: _inference already includes wait time. If pure model execution time is needed, it's harder.
        # For now, the duration recorded by _inference's caller is more about total turnaround.

        return utility_array, final_policy_output # Return (1D utility array, 1D policy_probs array)


    def _log_periodic_stats(self, cache_hit: bool):
        # This method is not provided in the original file or the code block
        # It's assumed to exist as it's called in prior_and_value
        pass

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