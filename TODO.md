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


## Plan

**Project: Migrate Long Narde to Two-Phase, Two-Slot Action Encoding**

**Goal:** Refactor the Long Narde game in OpenSpiel to use a two-phase action system for doubles, similar to `open_spiel/games/backgammon/backgammon.cc`. This involves changing action encoding, dice handling, state management for doubles, legal action generation, and updating test cases.

**I. Core Game Logic & State Modifications (`long_narde.h`, `long_narde_state.cc`, `long_narde_game.cc`)**

1. [DONE]  **State Flags for Doubles Handling (in `LongNardeState` class - `long_narde.h`):**
    *   Add `bool is_first_phase_of_doubles_ = false;`
        *   Initialize to `false` in constructor.
        *   Set to `true` in `ProcessChanceRoll` if a doubles roll occurs.
        *   Set to `false` in `DoApplyAction` after the first phase of a doubles turn is processed.
        *   Used by `LegalActions()` and `DoApplyAction()` to determine current phase.
    *   Ensure `UndoAction` correctly restores this flag from `TurnHistoryInfo`.
    *   [BUG FIXED] The two-phase turn mechanism for doubles is now implemented in `LongNardeAdvanceToNextPlayer` (`long_narde_api.cc`, lines 496-519): after phase 1, the player remains for phase 2, and only after both phases does the player advance. [DONE]

2. [DONE]  **Dice Representation and Handling:**
    *   **Keep `dice_` and `initial_dice_` as `std::vector<int>(4)`:** (No change to declaration)
        *   This simplifies managing the four pips of a doubles roll across two phases.
        *   Non-doubles: `dice_[0]`, `dice_[1]` used; `dice_[2]`, `dice_[3]` are 0.
        *   Doubles Phase 1: `dice_[0]`, `dice_[1]` used.
        *   Doubles Phase 2: `dice_[2]`, `dice_[3]` used.
    *   **Marking Dice Used:** Change from `-1` to `+6`.
        *   Modify `LongNardeState::DiceValue(int idx) const` (`long_narde_api.cc`):
            *   If `dice_[idx] > 6`, return `dice_[idx] - 6`.
            *   Else return `dice_[idx]`.
            *   Handle `idx` for all 4 dice slots.
        *   Modify `LongNardeState::IsDieUsable(int idx) const` (`long_narde_api.cc`):
            *   Return `dice_[idx] >= 1 && dice_[idx] <= 6`.
            *   Handle `idx` for all 4 dice slots.
        *   Update `LongNardeApplyCheckerMove` (`long_narde_state.cc`):
            *   When a die is used, find `d` in `dice_` (relevant to current phase) and change to `d + 6`.
            *   [BUG FIXED] All dice used (including for pass moves) are now marked with `+6` in `LongNardeApplyCheckerMove` (`long_narde_state.cc`, line 101+).
        *   Update `LongNardeUndoCheckerMove` (`long_narde_state.cc`):
            *   When undoing, find `d+6` in `dice_` and change back to `d`.
            *   [BUG FIXED] Undo logic for both regular and pass moves now correctly reverts `+6` marking in `LongNardeUndoCheckerMove` (`long_narde_state.cc`, line 153+).

3. [DONE]  **Turn History (`TurnHistoryInfo` struct and usage - `long_narde.h`, `long_narde_api.cc`):**
    *   Add `is_first_phase_of_doubles_` to `TurnHistoryInfo` to allow correct restoration during `UndoAction`.
    *   [NO BUG] Implementation matches plan.

4.  **Maximum Distinct Actions (`LongNardeGame::MaxGameLength` and related constants):**
    *   This is not `NumDistinctActions`. `MaxGameLength` is an estimate of total moves in a game. The change in how "moves" are counted (doubles are two player "moves") might slightly affect this estimate if it was tightly tuned, but likely no change needed unless it causes issues.
    *   `kNumDistinctActions` is handled in Encoding section.
    *   [DONE] Value and rationale reviewed/clarified in code; no change to value needed.

**II. Action Encoding/Decoding Modifications (`long_narde_encoding.cc`, `long_narde.h`)**

1. [DONE]  **Update Constants:**
    *   [DONE] All old encoding constants and helpers are removed from `long_narde_encoding.cc`. Only the new two-slot, two-phase encoding system remains documented and implemented.
2. [DONE]  **`LongNardeState::NumDistinctActions() const`:**
    *   Change to return `2 * kEncodingBase * kEncodingBase;` (e.g., 1250).
3. [DONE]  **`LongNardeState::LongNardeCheckerMovesToSpielMove(const std::vector<LongNardeCheckerMove>& moves) const`:**
    *   This function will now always encode **two** `LongNardeCheckerMove` objects (slots).
    *   The `moves` vector input should represent the 1 or 2 pips for the *current phase*.
    *   Pad `moves` with `kPassMove` (using the correct die for the pass) if only one actual move is made in the phase, to ensure two slots are always encoded.
    *   Determine `bool high_pip_first_for_phase`:
        *   For non-doubles: based on `dice_[0]` vs `dice_[1]`.
        *   For doubles phase 1: `dice_[0]` vs `dice_[1]` (pips are same, so this is conventional, e.g., always true).
        *   For doubles phase 2: `dice_[2]` vs `dice_[3]` (pips are same, so conventional).
    *   `slot0_pos = (moves[0].pos == kPassPos) ? (kEncodingBase - 1) : moves[0].pos;`
    *   `slot1_pos = (moves[1].pos == kPassPos) ? (kEncodingBase - 1) : moves[1].pos;`
    *   `action = slot0_pos + kEncodingBase * slot1_pos;`
    *   If `!high_pip_first_for_phase`, then `action += kEncodingBase * kEncodingBase;`
    *   Remove old encoding logic for >2 moves (doubles offset path).
4. [DONE]  **`LongNardeState::LongNardeSpielMoveToCheckerMoves(Player player, Action spiel_move) const`:**
    *   Decode `spiel_move` into `slot0_pos`, `slot1_pos`, and `high_pip_first_for_phase`.
    *   Determine active dice for the current phase:
        *   If `this->is_first_phase_of_doubles_` (or if not a doubles turn initially): use `this->dice_[0]` and `this->dice_[1]`.
        *   Else (second phase of doubles): use `this->dice_[2]` and `this->dice_[3]`.
    *   Use `DiceValue(idx)` to get original die values.
    *   Construct two `LongNardeCheckerMove` objects using the decoded positions and the determined dice for the phase, ordered by `high_pip_first_for_phase`.
    *   Calculate `to_pos` using `GetToPos()` for each reconstructed `LongNardeCheckerMove`.
    *   Return a vector of these (up to two) `LongNardeCheckerMove`s. Filter out pure pass moves if only one actual move was encoded, or return two passes if action was double pass.

**III. Core API Implementation Updates (`long_narde_api.cc`)**

- [x] `LongNardeState::DoApplyAction(Action move_id)`: (encoding/phase logic complete; legal move gen not yet)
    *   NOTE: Marked complete, but relies on phase logic (I.1) which has bugs (two-phase doubles not implemented).
- [x] `LongNardeState::UndoAction(Player player, Action action)`: (encoding/phase logic complete; legal move gen not yet)
    *   NOTE: Flag restoration is fine, but overall dice undo (I.2) has bugs.
- [x] `LongNardeState::ProcessChanceRoll(Action move_id)`: (encoding/phase logic complete)
    *   NOTE: Marked complete, but `is_first_phase_of_doubles_` flag (I.1) is not set.
- [x] `LongNardeState::DiceValue(int idx) const` and `IsDieUsable(int idx) const`: (complete)

**IV. [DONE] Legal Action Generation Modifications (`long_narde_legal_actions.cc`)**

*   This is the most complex change. The current file generates full sequences.
*   **New `LongNardeState::LegalActions() const` logic:**
  1. Determine current phase:
     - `is_doubles_turn_phase1 = (initial_dice_[0] == initial_dice_[1] && initial_dice_[0] > 0 && is_first_phase_of_doubles_)`.
     - `is_doubles_turn_phase2 = (initial_dice_[0] == initial_dice_[1] && initial_dice_[0] > 0 && !is_first_phase_of_doubles_)`.
     - `is_non_doubles_turn = !(initial_dice_[0] == initial_dice_[1] && initial_dice_[0] > 0)`.
  2. Identify active dice for this phase:
     - Phase 1 (doubles or non-doubles): Use `initial_dice_[0]` and `initial_dice_[1]`.
     - Phase 2 (doubles only): Use `initial_dice_[2]` and `initial_dice_[3]`.
  3. Adapt full sequence generation to only two pips:
     - Input: current board state, active dice for phase, `cur_player_`, `moved_from_head_`, `is_on_first_turn_`.
     - Output: valid 1–2 move sequences for this phase.
  4. Encode each sequence via `LongNardeCheckerMovesToSpielMove`, collect unique actions.
  5. Return unique encoded actions.
    *   [DONE] All phase/dice selection bugs in legal action generation are fixed (see RecLegalMoveSequences and pass dice selection logic).
    *   [DONE] The two-phase system and correct dice usage for pass moves are now enforced for both phases of doubles.

**V. Test Case Modifications**

*   **`long_narde_test_actions.cc`:**
    - [DONE] Restructure multi-pip doubles tests (`test-actionencodingtest-4`, `test-actionencodingtest-5`) into two-phase applications.
    - [DONE] Adapt encoding/decoding tests to new `NumDistinctActions()` and verify decoded moves match.
*   **`long_narde_test_movement.cc`:**
    - [DONE] Split 4-pip doubles tests (`TestBasicMovement`, `CheckerDistributionTest`) into two actions (phase 1 and phase 2).
    - [DONE] Update head-rule tests (`HeadRuleTestFirstTurn`, `HeadRuleTestNonFirstTurn`) to validate across both phases.
*   **`long_narde_test_endgame.cc`:**
    - [DONE] For any >2 pip action tests, split into two-phase applications.
    - Non-doubles bear-off tests remain unchanged.
*   **`long_narde_test_bridges.cc`:**
    - [DONE] Adjust double-roll bridge tests (e.g., `test-bridgetest-6`) to account for two-phase structure where appropriate.
*   **`long_narde_test_basic.cc` & `random_sim_test.cc`:** Minimal updates; confirm simulations and basic tests work with new phases.

**VI. Documentation & Cleanup**

1. Update code comments in `long_narde.h` and `.cc` files to describe two-phase encoding and state flags.
2. Remove old multi-pip encoding code and constants from `long_narde_encoding.cc`.
3. Update project README or game documentation to explain new action scheme and doubles phases.

**VII. Iterative Refinement and Debugging**

1. Implement changes incrementally: state flags & dice → encoding → API → legal actions → tests.
2. Run full test suite after each major change.
3. Use debug prints/logs to trace dice values, phase transitions, and action encode/decode.
4. Validate edge cases: pass moves, head-rule interactions, endgame scoring, bridge formation across phases.