# TODO: Long Narde Implementation Plan

## Overview
Implementation of Long Narde rules, based on a copy of "games/backgammon".

**Key Rules:**
1.  **Setup:** White: 15 checkers on point 24. Black: 15 checkers on point 12.
2.  **Movement:** Both move counter-clockwise into home (White: 1–6, Black: 13–18), then bear off.
3.  **Starting:** Each rolls 1 die; higher goes first (White). No doubles on first roll.
4.  **Turns:** Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves possible, skip turn. If only one die usable, use the higher value. Doubles grant 4 moves.
5.  **Head Rule:** Only 1 checker may leave the head per turn, except on the first turn if a double 6, 4, or 3 is rolled, allowing 2 checkers from the head.
6.  **Bearing Off:** Allowed once all checkers reach home. Use exact or higher rolls.
7.  **Ending/Scoring:** Mars (2 points) if loser bore off none; Oin (1 point) otherwise. `allow_last_roll_tie_` parameter enables optional tie rule.
8.  **Block (Bridge) Rule:** Forming a contiguous block of 6 checkers is illegal unless at least 1 opponent checker is ahead of the block.

## Current Status & Architecture

*   **Core Implementation:** All the rules are implemented.
*   **Code Structure:** Successfully modularized from monolithic `long_narde.cc` into `_state.cc`, `_moves.cc`, `_encoding.cc`, `_validation.cc`, `_legal_actions.cc`, `_api.cc`, `_utils.cc`, `_game.cc`.

## Plan: AlphaZero JAX Implementation

**Phase 1: Project Setup and Basic JAX Model Definition**

**Summary:** This phase established the foundational directory structure and core JAX components for the AlphaZero implementation. Key outcomes include the definition of `ConfigJAX` (configuration settings), `TrainInputJAX` (data structure for training examples), a basic `MLP_JAX` model using Flax, and a helper function (`init_flax_model_and_variables`) to initialize the Flax model and its variables.
Files created/modified: `open_spiel/python/examples/alpha_zero_jax.py`, `open_spiel/python/algorithms/alpha_zero_jax/model_jax.py`, `open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py`.

**Phase 2: MCTS Evaluator for JAX Model**

**Summary:** An MCTS evaluator compatible with the JAX model, `AlphaZeroEvaluatorJAX`, was implemented. This class, inheriting from `mcts.Evaluator`, uses the JAX model and its variables to provide policy priors and value estimates for MCTS rollouts by performing inference on game states.
File created/modified: `open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py`.

**Phase 3: Learner Implementation with JAX**

**Summary:** The AlphaZero learner component was adapted for JAX. This involved setting up the JAX model and optimizer (Optax), managing a replay buffer (`Buffer` class) for `TrainInputJAX` instances, and implementing a JIT-compiled `train_step_fn` for efficient model training. JAX-specific checkpointing using `flax.training.checkpoints` (later Orbax) was also integrated.
File created/modified: `open_spiel/python/examples/alpha_zero_jax.py` (later moved to `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py`).

**Phase 4: Actor and Evaluator Processes with JAX**

**Summary:** The actor and evaluator processes were adapted to use the JAX model and `AlphaZeroEvaluatorJAX`. This included initializing the JAX model and variables within each process (later refactored for centralized inference), and an `update_checkpoint` function to load new model weights. The core game playing logic (`_play_game`) was made compatible with the JAX-based evaluator.
File created/modified: `open_spiel/python/examples/alpha_zero_jax.py` (later moved to `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py`).

**Phase 5: Top-Level Script and Integration**

**Summary:** The main orchestrating script for the AlphaZero JAX implementation was developed. This script handles configuration (`ConfigJAX`), JAX PRNG key management, spawning actor, learner, and evaluator processes, and overall coordination. Initial testing was performed using "tic_tac_toe".
File created/modified: `open_spiel/python/examples/alpha_zero_jax.py` (later split into `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py` for the library and `open_spiel/python/examples/alpha_zero_jax.py` for the executable example).

**Phase 6: Implement ResNet and Conv2D Models (`model_jax.py`)**

**Summary:** More sophisticated neural network architectures, `ResNet_JAX` (leveraging `jax-resnet` components) and `Conv2D_JAX`, were implemented in Flax. This included various ResNet configurations (ResNet18-200, WideResNet, ResNeXt, ResNetD, ResNeSt) and a generic Conv2D model. The `init_flax_model_and_variables` function was updated to support these new model types, and `ConfigJAX` was extended with fields for detailed ResNet customization. Handling of `batch_stats` for Batch Normalization was ensured.
File created/modified: `open_spiel/python/algorithms/alpha_zero_jax/model_jax.py`.

**Phase 7: Refinement and Finalization**

**Summary:** This phase focused on general refinements, including adapting the `FileLogger`, documenting `ConfigJAX`, adding comments and docstrings, ensuring correct policy target handling in the training step, managing dependencies (`jax`, `flax`, `optax`), and making the main script executable with `absl.app` and `absl.flags`.
Files created/modified: Primarily `open_spiel/python/examples/alpha_zero_jax.py` and `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py`.

**Phase 8: Post-Review Fixes and Refinements**

**Summary:** A series of fixes and refinements were implemented based on code review and further testing. Key improvements included:
*   Corrected policy loss calculation and masking in the learner.
*   Integrated LRU caching into `AlphaZeroEvaluatorJAX`.
*   Added game-type validations to the JAX evaluator.
*   Aligned JAX model head structures (MLP, Conv2D, ResNet) more closely with TensorFlow reference implementations, including input flattening/reshaping for convolutional models.
*   Standardized RNG handling for determinism.
*   Improved logging discipline and `watcher` robustness.
*   Provided legacy-compatible defaults for ResNet configurations.
*   Ensured JIT compilation for inference in `AlphaZeroEvaluatorJAX`.
*   Resolved issues with `NaN` losses and checkpointing errors.
*   Standardized logging levels and suppressed verbose third-party logs.
*   Restructured the project by separating the core algorithm library from the example script.
Files created/modified: `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py`, `open_spiel/python/examples/alpha_zero_jax.py`, `open_spiel/python/algorithms/alpha_zero_jax/model_jax.py`, `open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py`.

## 9. Remote Inference Service Tasks

**Summary & Architecture Rationale:**
To address JAX/TPU resource conflicts and `XlaRuntimeError` when multiple processes attempt to initialize JAX or access the TPU (especially in environments like Colab), a remote inference service architecture was implemented.

**Key Features & Benefits:**
*   **Centralized TPU Interaction:** All JAX model initialization, inference, and TPU interactions are centralized in the main (learner) process. This ensures a single TPU context, eliminating cross-process conflicts.
*   **CPU-Bound Actors/Evaluators:** Actor and evaluator processes become pure-Python and CPU-bound, focusing on MCTS self-play and game simulation. They generate inference requests but do not directly interact with JAX or the TPU.
*   **Message-Based Communication:** A message protocol (`InferenceRequest`, `InferenceResponse`) and multiprocessing queues are used for actors/evaluators to send inference requests to the main process and receive results.
*   **Batched Inference:** The `InferenceServicer` in the main process batches requests from multiple actors/evaluators to maximize TPU throughput. It consists of a `BatchAssemblyThread` (collects requests) and an `InferenceExecutionThread` (executes batched model inference).
*   **`RemoteEvaluator` Stub:** Actors/evaluators use a `RemoteEvaluator` stub that handles the communication with the shared inference queues.
*   **Graceful Shutdown:** A `SHUTDOWN_SENTINEL` mechanism was implemented for clean termination of all processes.
*   **Dedicated Logging:** `RemoteEvaluator` instances now have their own dedicated log files (e.g., `log-remote_evaluator_actor_{actor_id}.txt`) for easier debugging of inference communication.

This architecture provides full CPU parallelism for game simulation and MCTS, efficient batched TPU inference, clear separation of concerns, and robustness in single-TPU environments.
Files created/modified: `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py`, `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`, `open_spiel/python/algorithms/alpha_zero_jax/actor_evaluator_logic.py`.

32. **Implement named log-level constants and suppress third-party verbose logs**
    *   Add in-code suppression of Orbax/Abseil INFO logs via `os.environ['GLOG_minloglevel']` and `absl.logging.set_verbosity`.
    *   Define `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE` constants at the top of `alpha_zero_jax.py`.
    *   Change example script's `--log_level` flag from `DEFINE_integer` to `DEFINE_enum("log_level", "INFO", ["ERROR","WARN","INFO","DEBUG","TRACE"], "Logging verbosity level")`.
    *   Replace all numeric `config.log_level >= N` checks with the new named constants.
    *   Citation: Chat discussion on 2025-05-16 about logging suppression and log-level constants.

## Phase 10: MCTS Enhancements for Chance Nodes and Async Search

**Summary of Completed Tasks:**
*   **Chance Node Evaluation Crash Fix:** Resolved an issue where the MCTS evaluator could be called on chance nodes. This was fixed by setting `dont_return_chance_node=True` in `AlphaZeroBot` (within `actor_evaluator_logic.py`) and `MCTSBot` constructors, ensuring evaluators only receive decision nodes.

**Further Work (Detailed in Phase 11):**
The initial integration of asynchronous MCTS (`async_mcts.py`) highlighted issues with concurrent inference requests within a single actor when `async_batch_size > 1`. The detailed plan to address this by refactoring `RemoteEvaluator` is covered in Phase 11.

## Phase 11: Enable `async_batch_size > 1` via `RemoteEvaluator` Refactoring

**Problem Statement**:
The current `RemoteEvaluator` in `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py` is not designed to safely handle multiple concurrent inference requests originating from within the *same* actor client (e.g., when `async_mcts.MCTSBot` uses a `ThreadPoolExecutor` with `async_batch_size > 1` to evaluate multiple MCTS leaves in parallel). Each parallel thread within the actor calls `RemoteEvaluator._inference()`, which sends a request and then polls the single shared `_inference_response_queue` for its specific `request_id`. This leads to a race condition where one thread can consume and discard a response meant for another thread within the same actor, causing timeouts, stalls, or incorrect behavior. The objective of this phase is to refactor `RemoteEvaluator` to correctly demultiplex responses to their originating internal calls, allowing `async_batch_size > 1` to function as intended.
Citation: Conversation history identifying response queue contention and successful workaround by setting `async_batch_size=1`.

**Tasks**:

-   [DONE] **Modify `RemoteEvaluator.__init__` (`remote_inference.py`)**:
    *   Add `self._pending_requests_map = {}` to store `(event, response_placeholder)` tuples keyed by `request_id`.
    *   Add `self._map_lock = threading.Lock()` to protect access to `_pending_requests_map`.
    *   Add `self._response_handler_thread = None` to hold the dedicated response handler thread.
    *   Add `self._shutdown_event = threading.Event()` to signal the handler thread to stop.
    *   Ensure `_actor_id`, `_logger` (the instance, not the FileLogger wrapper), and `_log_level` are available (they are already passed and used for dedicated logging).
    *   Citation: Design discussion for demultiplexing responses. `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`

-   [DONE] **Implement `RemoteEvaluator.start_response_handler()` (`remote_inference.py`)**:
    *   Creates and starts `self._response_handler_thread`.
    *   The thread target will be `self._handle_responses_loop`.
    *   Set `daemon=True` for the thread.
    *   Citation: Design discussion. `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`

-   [DONE] **Implement `RemoteEvaluator.stop_response_handler()` (`remote_inference.py`)**:
    *   Sets `self._shutdown_event.set()`.
    *   Puts `SHUTDOWN_SENTINEL` onto `self._inference_response_queue` to unblock the handler's `get()` call if it's waiting.
    *   Joins `self._response_handler_thread` with a timeout.
    *   Citation: Design discussion. `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`

-   [DONE] **Implement `RemoteEvaluator._handle_responses_loop()` (`remote_inference.py`)**:
    *   Loop while `not self._shutdown_event.is_set()`.
    *   Inside the loop, call `self._inference_response_queue.get(timeout=0.1)` (short timeout to allow checking `_shutdown_event`).
    *   Catch `queue.Empty` and continue if timeout.
    *   If `SHUTDOWN_SENTINEL` is received from the queue or `self._shutdown_event.is_set()`, break the loop.
    *   Deserialize `response_tuple` to `InferenceResponse`.
    *   With `self._map_lock`, `pop` the `(event, response_data_holder)` from `self._pending_requests_map` using `response.request_id`.
    *   If found: append `response` to `response_data_holder` and `event.set()`.
    *   Else (not found, e.g., timed out request), log a warning using `self.logger`.
    *   **Post-loop cleanup**: Iterate through any remaining items in `self._pending_requests_map` (acquire lock), set their events, and put a shutdown/error marker in their data holders to unblock any lingering `_inference` calls that might still be waiting.
    *   Citation: Design discussion. `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`

-   [DONE] **Modify `RemoteEvaluator._inference()` (`remote_inference.py`)**:
    *   Add `mcts_search_timeout_sec: float` parameter (this will be `AsyncMCTSBot.timeout`).
    *   Generate `request_id = uuid.uuid4().hex`.
    *   Create `event = threading.Event()` and `response_holder = []`.
    *   With `self._map_lock`, add `self._pending_requests_map[request_id] = (event, response_holder)`.
    *   Send `InferenceRequest` (using `self._actor_id`) to `self._inference_request_queue`.
    *   Wait for event: `timed_out = not event.wait(timeout=mcts_search_timeout_sec)`.
    *   If `timed_out`:
        *   With `self._map_lock`, attempt to `pop` `request_id` from `_pending_requests_map` to clean up.
        *   Log timeout using `self.logger` and raise `TimeoutError`.
    *   If not timed out:
        *   If `response_holder` is empty or contains a shutdown marker (from `_handle_responses_loop` cleanup), log and raise `ShutdownException`.
        *   Otherwise, the response is `response_holder[0]`. Return it.
    *   Citation: Design discussion. `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`

-   [DONE] **Modify `RemoteEvaluator.prior_and_value()` (`remote_inference.py`)**:
    *   Add `mcts_search_timeout_sec: float` parameter.
    *   Pass `mcts_search_timeout_sec` to the call to `self._inference()`.
    *   Wrap the `self._inference()` call in a `try...except TimeoutError:`. If `TimeoutError` occurs, log it via `self.logger` and return `(None, None)`.
    *   `ShutdownException` should also be caught and result in returning `(None, None)` or be re-raised if `MCTSBot` needs to handle it more directly. For now, assume `(None,None)` is fine.
    *   Citation: Design discussion. `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`

-   [DONE] **Update `MCTSBot.evaluate()` call in `async_mcts.py`**:
    *   In `async_mcts_search`, when `self.evaluator.prior_and_value(leaf_state)` is called (inside the `evaluate_batch` function submitted to the thread pool), pass `self.timeout` as the new `mcts_search_timeout_sec` argument.
    *   The `self.evaluator` here is the `_AsyncRemoteEvaluatorAdapter` instance, which wraps the `RemoteEvaluator`. The adapter will need to be updated to accept and pass this new timeout argument.
    *   Citation: Design discussion. `open_spiel/python/algorithms/async_mcts.py`

-   [DONE] **Update `_AsyncRemoteEvaluatorAdapter.prior_and_value()` (`actor_evaluator_logic.py`)**:
    *   Modify the `prior_and_value` method of `_AsyncRemoteEvaluatorAdapter` to accept the `mcts_search_timeout_sec` argument.
    *   Pass this argument through to the wrapped `self._remote_evaluator.prior_and_value()` call.
    *   Citation: Implicit from changes to `MCTSBot` and `RemoteEvaluator`. `open_spiel/python/algorithms/alpha_zero_jax/actor_evaluator_logic.py`

-   [DONE] **Update `actor` (and `evaluator` if applicable) in `actor_evaluator_logic.py`**:
    *   After creating the `RemoteEvaluator` instance (e.g., `remote_evaluator = RemoteEvaluator(...)`), call `remote_evaluator.start_response_handler()`.
    *   In the `finally` block of the actor/evaluator process, call `remote_evaluator.stop_response_handler()` before exiting.
    *   Citation: Design discussion. `open_spiel/python/algorithms/alpha_zero_jax/actor_evaluator_logic.py`

-   [DONE] **Testing**:
    *   Run with `async_mode=True` and `async_batch_size > 1` (e.g., 16).
    *   Verify that actors make progress and the learner trains without "Discarding stale/unexpected response" warnings in the `RemoteEvaluator` logs.
    *   Monitor CPU/TPU utilization to ensure the system is performing efficiently.