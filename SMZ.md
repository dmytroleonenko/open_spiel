# Stochastic MuZero for Long Narde - Implementation Plan

This plan tracks the remaining tasks for implementing Stochastic MuZero (SMZ) specifically for **Long Narde**, with a focus on correctly handling board rotation and experience propagation.

**Target Game:** Long Narde (not Backgammon). Key differences impacting implementation:
*   No bar, no hitting.
*   Both players move counter-clockwise.
*   Head rule restriction (one checker off head per turn, except initial special rolls).
*   Blocking rule (6-prime only allowed if opponent checkers are ahead).

## BLOCKER: Segmentation Fault in Core Game

*   **[ ] FIX: Segmentation Fault in `LongNardeState::ApplyAction`**
    *   **Issue:** Running `generate_new_playthrough.sh` crashes when applying action 0 (Bear Off from point 1) in a specific late-game state (details in conversation history).
    *   **Impact:** Prevents generating playthroughs for testing and potentially destabilizes training.
    *   **Next Steps:**
        1.  Create a minimal reproducible test case (C++ or Python) for the specific state and action.
        2.  Debug the C++ implementation (`long_narde_state.cc`, `long_narde_moves.cc`, etc.) to find the cause.
        3.  Fix the C++ code.
        4.  Verify the fix by rerunning the test case and the playthrough generation.
    *   **Status:** **BLOCKING** further SMZ development.

## Core Challenge: Adapting SMZ to Long Narde's Half-Move API

1.  **[X] Verify Game Observation/Action Structure:**
    *   **Task:** Determine the exact indices and format of the observation vector from `pyspiel.Game("long_narde").observation_tensor_spec()`. Verify point indices (0-23), borne-off indices, player index, and dice/move info.
    *   **Task:** Determine the action encoding scheme. How are integer actions mapped to `(from, to)` half-moves? Use `game.action_to_string()` and potentially inspect the game implementation.
    *   **File:** `stochastic_muzero.py` (for comments/verification), potentially requires running code to inspect game object.
    *   **Detail:** Accurate indices are vital for the `rotate_observation_long_narde` function.
    *   **Status:** Done. Layout is `(xy, player, point, borne)`. Shape `(28,)`. **Actions are simple indices 0-24**. `action_to_string` confirms this, outputting `Action(id=X, player=Y)`. The previous assumption of 1352 actions and `(from)-(to)` strings was incorrect.

2.  **[X] Implement Simplified Action Index Rotation:**
    *   **Task:** Remove the complex mapping functions (`_build_action_maps`, `map_action_index_to_tuple`, `map_tuple_to_action_index`).
    *   **Task:** Implement `rotate_action_long_narde(action_index, player_id)` to directly rotate the index. Assume indices 0-23 correspond to points and rotate using `(index + 12) % 24`. Assume index 24 (or potentially others) represent special actions (like Pass or Bear-off) that do *not* rotate relative to the board points.
    *   **File:** `stochastic_muzero.py`.
    *   **Detail:** Translates between the agent's canonical action index space and the environment's absolute action index space.
    *   **Status:** Done.

3.  **[X] Implement Legal Mask Rotation:**
    *   **Task:** Determine how the `legal_actions_mask` corresponds to actions.
    *   **Task:** Implement logic to rotate the `legal_actions_mask` in `store_step_data` when `player_id == 1`. This ensures the mask stored aligns with the canonical observation/action.
    *   **File:** `stochastic_muzero.py` (`store_step_data`).
    *   **Detail:** Needed if the mask is used directly in the learning target (less common for MCTS agents, but good practice).
    *   **Status:** Done.

4.  **[X] Integrate Action Rotation:**
    *   **Task:** Replace placeholder calls in `step` method: Rotate legal actions *before* passing to `_run_mcts`. Rotate selected canonical action *back* before returning `StepOutput`.
    *   **Task:** Replace placeholder call in `store_step_data`: Rotate the `action` before storing it as `canonical_action`.
    *   **File:** `stochastic_muzero.py` (`step`, `store_step_data`).
    *   **Status:** Done.

5.  **[~] Verify Reward/Value Propagation:**
    *   **Task:** Double-check `finalize_trajectory` ensures the final reward uses the *original* agent `player_id`.
    *   **Task:** Double-check `_backpropagate` correctly uses `node.to_play` vs `leaf_player` for negation, independent of observation rotation.
    *   **Task:** Confirm n-step target calculation in `_prepare_batch` correctly uses the stored (actual) rewards associated with the canonical states.
    *   **File:** `stochastic_muzero.py` (`finalize_trajectory`, `_backpropagate`, `_prepare_batch`).
    *   **Detail:** Ensures the canonical policy/value functions are trained towards the correct game outcomes.
    *   **Status:** Seems correct based on implementation, needs confirmation during testing.

6.  **[ ] Test Rotation Logic:** *(Blocked by segfault fix)*
    *   **Task:** Generate a sample game playthrough using `open_spiel/scripts/generate_new_playthrough.sh long_narde`.
    *   **Task:** Manually extract several Player 1 (Black) turns from the generated `long_narde.txt` playthrough file, noting the original observation tensor and legal actions.
    *   **Task:** For each extracted state, manually calculate the expected canonical (Player 0 perspective) observation and the expected set of canonical legal action indices based on the `rotate_observation_long_narde` and `rotate_action_long_narde` functions.
    *   **Task:** Create a test suite (e.g., `stochastic_muzero_test.py`) with test cases that call the rotation functions with the extracted original data and assert equality with the manually calculated expected canonical data (`np.testing.assert_array_equal` for observations, `self.assertEqual` for sets of actions).
    *   **Task:** (Optional) Add test cases for rotating canonical actions back to the original perspective.
    *   **Task:** (Optional) Add simpler unit tests with manually constructed edge-case board states (e.g., bearing off, head moves).
    *   **Task:** (Debugging) Add temporary debug logging in `step` and `store_step_data` to print original vs. canonical data during initial runs for manual verification.

## Refinement & Other Tasks
# ... (rest of plan remains the same) ...