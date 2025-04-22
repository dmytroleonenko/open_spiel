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

1.  **[HIGH] Refactor State for Doubles Handling:** Implement the proposed state refactoring to correctly handle 4 moves on doubles.
    *   **Goal:** Modify state representation (`dice_`) to robustly handle 4 moves, enabling future apply/undo optimization and fixing `TestDoubleMove`.
    *   **Files Affected:** `long_narde.h`, `_state.cc`, `_api.cc`, `_moves.cc`, `_legal_actions.cc`.
    *   **Steps:**
        *   [ ] **Modify State Definition (`.h`, `_state.cc`):**
            *   Change `dice_` to `std::vector<int>(4)`, initialized with zeros.
            *   Remove `double_turn_`, `doubles_moves_made_`.
            *   Update `TurnHistoryInfo` struct (remove related fields).
        *   [ ] **Update Dice Rolling (`_api.cc`):**
            *   In `ProcessChanceRoll`: Populate `dice_` as `{d1, d2, 0, 0}` (non-double) or `{d, d, d, d}` (double `d-d`).
        *   [ ] **Update Dice Accessors/Checkers (`_state.cc`):**
            *   Verify `DiceValue(i)` and `IsUsed(i)` logic (add 6).
            *   Update `UsableDiceOutcome(i)` to treat `0` as unusable.
        *   [ ] **Update Move Application/Undo (`_moves.cc`, `_api.cc`):**
            *   Modify `ApplyCheckerMove` to find *first* unused `dice_[i]` matching `move.die`, mark used (add 6).
            *   Modify `UndoCheckerMove` to find *first* used `dice_[i]` matching `move.die`, mark unused (subtract 6).
            *   Update `UndoAction` to remove logic for removed state fields.
        *   [ ] **Update Move Generation (`_legal_actions.cc`):**
            *   Simplify `GenerateAllHalfMoves`: Loop `i` 0-3, check `UsableDiceOutcome(dice_[i])`.
            *   Remove `max_moves_param` from `IterativeLegalMoves` / `GenerateMoveSequences` calls.
        *   [ ] **Build and Test:**
            *   Incremental builds.
            *   Run all tests, focus on `TestDoubleMove`. Debug failures.

2.  **[MEDIUM] Verify/Fix Test Discrepancies:** After the doubles refactor, ensure tests accurately reflect original logic.
    *   [ ] **HeadRuleTest:** Verify `long_narde_test_movement.cc` covers original intent (Test Case #2).
    *   [ ] **FirstTurnDoublesExceptionTest:** Implement missing test (Test Case #3).
    *   [ ] **BlockingBridgeRuleTest:** Verify `long_narde_test_bridges.cc` covers all 4 original sub-cases (Test Case #4).

3.  **[MEDIUM] Optimize Iterative Move Generation (Reduce Cloning):** Once correctness (especially doubles) is confirmed, implement apply/undo optimization in `IterativeLegalMoves`.
    *   [ ] Refactor `IterativeLegalMoves` loop: Use `ApplyCheckerMove`, manage context stack, recurse/iterate, then `UndoCheckerMove`. Avoid `Clone()`.
    *   [ ] Ensure `UndoCheckerMove` correctly restores all state (including sequence state like `moved_from_head_this_sequence`).
    *   [ ] Build, test correctness, and measure performance.

4.  **[LOW] Simplify `IsFirstTurn` Access:**
    *   [ ] Remove redundant `is_first_turn()` method, keep only `IsFirstTurn(Player player)`.
    *   [ ] Update call sites.
    *   [ ] Verify tests pass.

5.  **[FINAL] Commit Changes:**
    *   [ ] After addressing priorities and ensuring all tests pass, commit following TDD principles.
