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

*   **Core Implementation:** Most rules are implemented.
*   **Code Structure:** Successfully modularized from monolithic `long_narde.cc` into `_state.cc`, `_moves.cc`, `_encoding.cc`, `_validation.cc`, `_legal_actions.cc`, `_api.cc`, `_utils.cc`, `_game.cc`.
*   **Move Generation:** Uses iterative generation (`IterativeLegalMoves` -> `GenerateMoveSequences`). Currently clones state for each move branch exploration to ensure correctness, sacrificing apply/undo optimization benefits for now.
*   **Doubles Handling:** **Major Issue:** The current `dice_` state (2 elements) cannot correctly handle the 4 moves required on a double roll, causing test failures (`TestDoubleMove`) and instability. **State refactoring is the top priority.**
*   **Testing:** Includes test utilities (`long_narde_test_utils.h/cc`) and numerous tests. While many pass, some discrepancies with original `long_narde_test.cc` logic might exist, and doubles-related tests are unreliable due to the state issue.

## Current Priorities

1.  **[DONE] Refactor State for Doubles Handling:** Implement the proposed state refactoring to correctly handle 4 moves on doubles.
    *   **Goal:** Modify state representation (`dice_`) to robustly handle 4 moves, enabling future apply/undo optimization and fixing `TestDoubleMove`.
    *   **Files Affected:** `long_narde.h`, `_state.cc`, `_api.cc`, `_moves.cc` (modified in `_state.cc`), `_legal_actions.cc` (indirectly via `IsDieUsable`), `_validation.cc`.
    *   **Steps:**
        *   [X] **Modify State Definition (`.h`, `_state.cc`):
            *   Change `dice_` to `std::vector<int>(4)`, initialized with zeros.
            *   Remove `double_turn_`, `doubles_moves_made_`.
            *   Update `TurnHistoryInfo` struct (remove related fields).
        *   [X] **Update Dice Rolling (`_api.cc`):
            *   In `ProcessChanceRoll`: Populate `dice_` as `{d1, d2, 0, 0}` (non-double) or `{d, d, d, d}` (double `d-d`).
        *   [X] **Update Dice Accessors/Checkers (`_state.cc`, `_validation.cc`):
            *   Verify `DiceValue(i)` and `IsUsed(i)` logic (add 6 to mark used).
            *   Update `IsDieUsable(i)` (in `_validation.cc`) to use `DiceValue(i) > 0`.
            *   `UsableDiceOutcome(outcome)` (in `_validation.cc`) for die *values* remains correct.
        *   [X] **Update Move Application/Undo (`_state.cc` acting as `_moves.cc`, `_api.cc`):
            *   Modify `ApplyCheckerMove` (in `_state.cc`) to find *first* unused `dice_[i]` matching `move.die`, mark used (add `kNumDiceOutcomes`).
            *   Modify `UndoCheckerMove` (in `_state.cc`) to find *first* used `dice_[i]` matching `move.die`, mark unused (subtract `kNumDiceOutcomes`).
            *   Update `UndoAction` (in `_api.cc`) to remove logic for removed state fields and handle 4-dice history.
        *   [X] **Update Move Generation (`_legal_actions.cc`):
            *   Simplify `GenerateAllHalfMoves`: Loop `i` 0-3, uses `IsDieUsable(dice_[i])` which is now correct.
            *   Remove `max_moves_param` from `IterativeLegalMoves` / `GenerateMoveSequences` calls (already done).
        *   [X] **Build and Test:** (All tests passing after fixing `SetupDice` and `GetCanonicalPoint`)
            *   Incremental builds.
            *   Run all tests, focus on `TestDoubleMove`. Debug failures.

============
2.  **[HIGH] Implement New Action Encoding & Two-Phase Doubles Handling Plan**
    *   **Overall Goal:** Refactor Long Narde action encoding to use a 1250-ID space (2 * 25 * 25) representing pairs of half-moves. Doubles rolls allowing 4 moves will be handled as two distinct agent actions (phases), managed by a new state flag. Expensive recursive move generation will be cached per dice roll.
    *   **I. Core Action Encoding/Decoding (Target: 1250 ID Space)**
        *   [X] **Modify `LongNardeCheckerMovesToSpielMove` (in `long_narde_encoding.cc`):**
            *   Input: 1 or 2 `LongNardeCheckerMove`s for the current phase. (Enforced `size <= 2`, removed `EncodeDoubles` fallback).
            *   Pad input to exactly two `LongNardeCheckerMove`s (using Pass moves if necessary).
            *   Encode `pos`: `0-23` -> `0-23`; `kPassPos` -> `24`.
            *   Determine `first_move_used_actual_high_die` based on actual dice available for the current phase.
            *   `base_action_id = encoded_pos1 * 25 + encoded_pos0;`
            *   Apply offset: `if (!first_move_used_actual_high_die && !(/*both moves are passes*/)) { final_action_id += (25 * 25); }` (Ensure double pass does not get offset).
            *   Return `final_action_id`.
        *   [ ] **Modify `LongNardeSpielMoveToCheckerMoves` (in `long_narde_encoding.cc`):**
            *   Input: `Player player, Action spiel_move_id`.
            *   Determine `first_move_used_actual_high_die` from `spiel_move_id`.
            *   Extract `encoded_pos0`, `encoded_pos1`. Convert to `actual_pos0`, `actual_pos1`.
            *   Infer `die_for_move0`, `die_for_move1` by accessing `dice_` and considering `first_move_used_actual_high_die` and dice available for the *current phase*.
            *   Construct the two `LongNardeCheckerMove`s, populating `to_pos` using `GetToPos()`.
            *   Return vector of two `LongNardeCheckerMove`s.
        *   [ ] **Update `LongNardeGame::NumDistinctActions()` (in `long_narde.h`):**
            *   Return `1250`. (Update or remove `kNumDistinctActions` constant).
    *   **II. State Management for Doubles & Turn Progression**
        *   [ ] **Add/Update State Variables in `LongNardeState` (`long_narde.h`):**
            *   `bool is_handling_second_phase_of_doubles_ = false;`
            *   `std::vector<std::vector<LongNardeCheckerMove>> current_turn_full_legal_sequences_cache_;`
            *   `LongNardeCheckerMove first_phase_selected_move1_ = {kPassPos, kPassPos, 0};`
            *   `LongNardeCheckerMove first_phase_selected_move2_ = {kPassPos, kPassPos, 0};`
            *   `bool head_move_occurred_this_full_turn_ = false;`
            *   Ensure initialization, `Clone()`, and `TurnHistoryInfo` handling.
        *   [ ] **Modify `LongNardeApplyCheckerMove` (e.g., `long_narde_moves.cc`):**
            *   When marking a die used (`dice_[idx] += 6`), select the *first available slot index* in `dice_` whose `DiceValue()` matches `move.die`.
        *   [ ] **Modify `ProcessChanceRoll` (in `long_narde_api.cc`):**
            *   Clear `current_turn_full_legal_sequences_cache_`, `first_phase_selected_move1_`, `first_phase_selected_move2_`.
            *   Reset `is_handling_second_phase_of_doubles_ = false;`, `head_move_occurred_this_full_turn_ = false;`.
            *   Call `GenerateAndCacheFullLegalSequences()`:
                *   Invokes full sequence generator (e.g., `LongNardeGenerateMoveSequences`).
                *   Ensures canonical sequences (no strategically identical permutations).
                *   Applies filtering (`LongNardeFilterBestMoveSequences`, `LongNardeApplyHigherDieRuleIfNeeded`).
                *   Stores results in `current_turn_full_legal_sequences_cache_`.
        *   [ ] **Modify `DoApplyAction` (in `long_narde_api.cc`):**
            *   Chance Node: Call `ProcessChanceRoll()`.
            *   Player Turn:
                *   Decode `move_id` to `decoded_moves = {actual_m1, actual_m2}`.
                *   Apply moves: call `LongNardeApplyCheckerMove` for each; if head move, set `head_move_occurred_this_full_turn_ = true;`.
                *   Record `TurnHistoryInfo`.
                *   Determine `was_doubles_roll`.
                *   Count `usable_dice_remaining`.
                *   If `was_doubles_roll && !is_handling_second_phase_of_doubles_ && usable_dice_remaining > 0`:
                    *   `is_handling_second_phase_of_doubles_ = true;`
                    *   Store `actual_m1`, `actual_m2` in `first_phase_selected_move1_`, `first_phase_selected_move2_`.
                    *   Player continues turn.
                *   Else (not continuable double / second phase done):
                    *   `is_handling_second_phase_of_doubles_ = false;`
                    *   Fully clear `dice_`.
                    *   `LongNardeAdvanceToNextPlayer(...)`.
        *   [ ] **Modify `LegalActions` (in `long_narde_legal_actions.cc`):**
            *   If `is_handling_second_phase_of_doubles_`:
                *   Filter `current_turn_full_legal_sequences_cache_` by `first_phase_selected_move1_` & `first_phase_selected_move2_`.
                *   Extract suffixes `{s_m3, s_m4}`. Convert to Action IDs, return.
            *   Else (`!is_handling_second_phase_of_doubles_`):
                *   From `current_turn_full_legal_sequences_cache_`:
                    *   Non-double sequences (len 1-2) are direct actions.
                    *   Double sequences: extract prefixes `{m1, m2}`.
                *   Collect unique 1/2-move sequences/prefixes. Convert to Action IDs, return.
        *   [ ] **Update `is_on_first_turn_` Logic (`long_narde.h` / `long_narde_api.cc`):**
            *   Set if current player (post-`ProcessChanceRoll`) has all 15 checkers on head.
            *   Persists for both phases of a first-turn double.
        *   [ ] **Update Head Move Rule Logic (e.g., `IsLegalHeadMove` in `long_narde.h`):**
            *   Check against `head_move_occurred_this_full_turn_`.
            *   Consider `is_on_first_turn_` for special exceptions (2 from head on double 6,4,3).
    *   **III. Observation Tensor Updates**
        *   [ ] **Increment `kStateEncodingSize` (in `long_narde.h`) by 1.**
        *   [ ] **Modify `ObservationTensor` (in `long_narde_api.cc`):**
            *   Add `is_handling_second_phase_of_doubles_`.
            *   Dice observation: If `is_handling_second_phase_of_doubles_`, observe `DiceValue(2)` & `DiceValue(3)`; else `DiceValue(0)` & `DiceValue(1)`.
    *   **IV. Testing and Validation**
        *   [ ] Thoroughly test normal turns, all doubles scenarios (0-4 moves), first turn rules, head move rule across phases, undo, and encoding/decoding.

3.  **[MEDIUM] Verify/Fix Test Discrepancies:** After the doubles refactor, ensure tests accurately reflect original logic.
    *   [ ] **HeadRuleTest:** Verify `long_narde_test_movement.cc` covers original intent (Test Case #2).
    *   [ ] **FirstTurnDoublesExceptionTest:** Implement missing test (Test Case #3).
    *   [ ] **BlockingBridgeRuleTest:** Verify `long_narde_test_bridges.cc` covers all 4 original sub-cases (Test Case #4).
============