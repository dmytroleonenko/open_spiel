# MOVE: Refactor Long Narde Action API to Half-Move Steps

This document describes the steps to change Long Narde's action API from a single "one-turn = one action" model to a sequential half-move model (2 calls for non-doubles, 4 calls for doubles), reducing the action space to 25.

## 1. State Extension: Tracking Partial Turns

- [x] **Add field `moves_remaining_`** to `LongNardeState` in [long_narde.h](mdc:open_spiel/games/long_narde/long_narde.h) (around line 200):
  - `int moves_remaining_;`
  - Purpose: track how many half-moves remain in the current turn.

- [x] **Initialize `moves_remaining_ = 0`** in the constructor in [long_narde_state.cc](mdc:open_spiel/games/long_narde/long_narde_state.cc) (around line 15):
  - Default value for new game states.

- [x] **Set `moves_remaining_` after a chance roll** in `DoApplyAction()` (see Step 3):
  - On non-doubles: `moves_remaining_ = 2`.
  - On doubles: `moves_remaining_ = 4`.

## 2. LegalActions Implementation

- [x] **Unified half-move lookahead**: `LegalActions()` calls `LongNardeGenerateMoveSequences(CurrentPlayer())` to enumerate all sequences of remaining half-moves (2 or 4), then filters them via `LongNardeFilterBestMoveSequences()` to keep only those with the maximum number of non-pass steps.
- [x] **First-step extraction**: From each best sequence, it takes the first `LongNardeCheckerMove` and maps its `pos` to an action ID (`pos` or `kNumPoints` for pass).
- [x] **Single branch**: When `moves_remaining_ > 0`, these extracted half-moves are returned; otherwise `{}` is returned for no moves.
- [x] **Rule enforcement**: Head-rule and bridge-rule violations are handled by the sequence generator and filter, so no illegal half-moves surface.
- [x] **Terminal & Chance**: If `IsTerminal()`, return `{}`; if `IsChanceNode()`, return `LegalChanceOutcomes()`.

This approach removes the old separate case for last half-move, ensuring the same lookahead applies uniformly to every step of a turn.

## 3. DoApplyAction Refactor

- [x] **Update `DoApplyAction()`** in [long_narde_api.cc](mdc:open_spiel/games/long_narde/long_narde_api.cc) (around line 36):
  - [x] **Chance branch**: Process dice roll and set `moves_remaining_ = (dice_[0] == dice_[1] ? 4 : 2);` (applied at line 38).
  - [x] **Half-move branch**: Apply single checker move (`LongNardeApplyCheckerMove`), decrement `moves_remaining_--`, and transition to chance when zero.

## 4. Action Encoding: 25 Half-Moves

- [x] **Replace `long_narde_encoding.cc`** encoding logic (around lines 1–100) to:
  1. Drop multi-digit (2-move) and doubles (>2) schemes.
  2. Only encode one half-move per action:
     - Source point ∈ [0…23] or pass → single integer in [0…24].
  3. Update `kNumDistinctActions` in [long_narde.h](mdc:open_spiel/games/long_narde/long_narde.h) to `25` (line 140).
  4. Simplify `NumDistinctActions()` to return `kNumDistinctActions` (now 25).

## 5. Test Updates

All tests have been refactored and are now fully compliant with the half-move API. Each test now:
- Queries `state->LegalActions()` and expects 2 or 4 legal half-moves per turn (except in forced/pass/bear-off scenarios).
- Applies each half-move via `ApplyAction(half_move_action)`.
- Verifies intermediate and/or final board state after each sequence.
- Only decodes and applies actions from `LegalActions()`.
- Never attempts to decode or apply an action unless it is present in the current `LegalActions()`.

### 5.1 Basic Tests (Low Complexity)
- **Files:** `long_narde_test_basic.cc`, `long_narde_test_common.h`  
- **Status:** [x] Refactored and compliant. After the chance roll, expect `moves_remaining_ == 2` and that `LegalActions()` returns 25 half-moves.

### 5.2 Endgame Tests (Medium Complexity)
- **File:** `long_narde_test_endgame.cc`  
- **Status:** [x] Refactored and compliant. Each full-turn action replaced with 2 or 4 sequential half-moves, then checks terminal and score.

### 5.3 Movement Logic Tests (Medium Complexity)
- **File:** `long_narde_test_movement.cc`  
- **Status:** [x] All tests reviewed and fully compliant with the half-move API. No further changes needed; all tests operate at the half-move level.

### 5.4 Bridge Tests (Medium Complexity)
- **File:** `long_narde_test_bridges.cc`  
- **Status:** [x] Refactored and compliant. Tests use half-move legality to ensure blocking moves do not appear, and sequential half-move applications still prevent bridges.

### 5.5 Action Encoding Tests (High Complexity)
- **File:** `long_narde_test_actions.cc`  
- **Status:** [x] All tests refactored for the half-move API. Each test:
    - Asserts that each half-move action ∈ [0..24], with pass=24.
    - Uses only actions from `LegalActions()`.
    - Decodes each action to a single move and checks its properties.
    - Applies and verifies moves in sequence.
    - Handles bear-off and pass logic robustly.
    - No legacy multi-move encoding logic remains.

### 5.6 Debugging & Lessons Learned
- All test failures due to action decoding were resolved by ensuring only legal actions are decoded and applied.
- Debug output is guarded by a `kDebugging` flag in the header; all debug output in the implementation is wrapped in `if (kDebugging)` blocks.
- When a test fails due to action decoding, the action, player, and all available half-moves are printed for diagnosis.
- Tests are robust to future rule changes: they do not assert the number of legal actions unless the rules guarantee it, and always select/apply moves by matching expected move properties, not by index or count.

### 8.5. Test Refactoring for Half-Move API: Common Pitfalls & Fixes

- **Problem:** Many legacy tests assert a fixed number of legal actions per turn (e.g., `SPIEL_CHECK_EQ(legal_actions.size(), 2)`), which is no longer valid under the half-move API. In forced-move or bear-off scenarios, only one legal half-move may be available at a time.
- **Solution:**
  - **Do not** assert a fixed number of legal actions per half-move unless the rules guarantee it for that specific board state.
  - **Instead,** for a sequence of moves (e.g., a non-double turn with two dice), apply the first half-move, then the second, checking the legality and correctness of each step.
  - **Pattern for test updates:**
    1. Query `LegalActions()` for the current half-move.
    2. Assert that the expected move is present and correct (by decoding and checking move properties).
    3. Apply the move.
    4. Repeat for the next half-move until all dice are used or the game is terminal.
    5. Only assert the number of legal actions if the rules guarantee it (e.g., in a scenario where multiple moves are always possible).
- **Example Fix:**
  ```cpp
  // Old (brittle):
  auto legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 2); // ❌ Not robust

  // New (robust):
  for (int i = 0; i < moves_expected; ++i) {
    auto legal_actions = state->LegalActions();
    // There may be 1 or more legal actions, depending on the board
    bool found = false;
    for (auto action : legal_actions) {
      auto moves = state->LongNardeSpielMoveToCheckerMoves(player, action);
      if (matches_expected(moves, i)) {
        state->ApplyAction(action);
        found = true;
        break;
      }
    }
    SPIEL_CHECK_TRUE(found); // Ensure the expected move is present
  }
  ```
- **Summary:**
  - Always check for the correct *sequence* of half-moves, not a fixed number of legal actions at each step.
  - This approach is robust to forced-move, pass, and bear-off scenarios, and should be used for all test updates related to the half-move API.

### 8.6. API Parity: Legal Move Selection Must Match Classic Rules

- **Requirement:** The new half-move API must preserve the legal move selection logic of the old API, especially for bearing off.
- **Canonical Rule (see Nardy Rules above, Rule 6):**
  - When bearing off, a checker may only be borne off with a die if no checker occupies a higher point. If a checker is on a higher point, the die must be used to move that checker, not to bear off a lower one.
- **Test Reference:**
  - The test `BearingOffLogicTestBlackNearEnd` is a canonical case. It must not be possible to bear off from point 13 with die 5 if point 14 is occupied.
- **Migration Guidance:**
  - When updating or reviewing the new API, always check that the set of legal moves for any state matches the old API and the rules above.
  - If a test fails because the new API allows a bear-off from a lower point with a higher die while a higher point is occupied, this is a bug in the move generation logic, not the test.
- **Action:**
  - All contributors must ensure that the new API does not relax or alter the move selection rules, especially for bearing off. Use the referenced test as a check.

## 6. Python Bindings & API Wrappers

- [x] Checked [long_narde_api.cc](mdc:open_spiel/games/long_narde/long_narde_api.cc) for any hard-coded action space sizes or encodings; all updated to 25.

## 7. Documentation & Constants

- [x] All references in README or comments that mention `kDigitBase`, `kEncodingBaseDouble`, or legacy action sizes have been updated or removed.
- [x] All unused constants (`kDigitBase`, `kPassOffset`, `kDoublesOffset`, and doubles encoding code) have been removed.

## 8. Lessons Learned & Best Practices (from Half-Move API Migration)

### 8.1. Action API & Encoding
- Always treat actions as half-moves: Each action is a single checker move (or pass), not a full turn. Never assume a single action encodes a full turn.
- Action IDs are positional: Action IDs 0–23 correspond to board points, 24 is always pass. Avoid legacy encoding logic.
- When decoding actions, always check if the action matches a valid half-move for the current player and board.

### 8.2. Test Refactoring
- Tests must expect multiple legal actions per half-move:
  - Never assert `las.size() == 1` unless the rules guarantee it.
  - For each half-move, iterate over all legal actions, decode, and select/apply the one matching the expected move.
  - Assert that the expected move is found and applied.
- When checking move legality or directionality:
  - Only decode actions that correspond to valid half-moves for the current player and board.
  - Skip pass actions (ID 24) when checking move properties.
- For regression/illegal move tests:
  - Always cross-check that no legal action decodes to an illegal move (e.g., landing on an occupied point).
- Preserve anchor comments:
  - Comments like `// EndTest: ...` are essential for test navigation and should never be removed.

### 8.3. Debugging & Diagnostics
- Guard debug output with a flag:
  - Use a `kDebugging` flag in the header to control debug output. Set it to `true` when diagnosing test failures.
  - All debug output in the implementation should be wrapped in `if (kDebugging)` blocks.
- When a test fails due to action decoding:
  - Print the action, player, and all available half-moves to quickly diagnose mismatches.
  - Most failures are due to tests attempting to decode actions that are not valid for the current board/player.

### 8.4. General Migration Advice
- Update all references to action space size:
  - Hard-coded action space sizes in C++ and Python must be set to 25.
- Remove all legacy multi-move encoding logic and constants.
- Tests should be robust to future rule changes:
  - Avoid brittle assertions about the number of legal actions.
  - Always select and apply moves by matching expected move properties, not by index or count.

### 8.5. Test Refactoring for Half-Move API: Common Pitfalls & Fixes

- **Problem:** Many legacy tests assert a fixed number of legal actions per turn (e.g., `SPIEL_CHECK_EQ(legal_actions.size(), 2)`), which is no longer valid under the half-move API. In forced-move or bear-off scenarios, only one legal half-move may be available at a time.
- **Solution:**
  - **Do not** assert a fixed number of legal actions per half-move unless the rules guarantee it for that specific board state.
  - **Instead,** for a sequence of moves (e.g., a non-double turn with two dice), apply the first half-move, then the second, checking the legality and correctness of each step.
  - **Pattern for test updates:**
    1. Query `LegalActions()` for the current half-move.
    2. Assert that the expected move is present and correct (by decoding and checking move properties).
    3. Apply the move.
    4. Repeat for the next half-move until all dice are used or the game is terminal.
    5. Only assert the number of legal actions if the rules guarantee it (e.g., in a scenario where multiple moves are always possible).
- **Example Fix:**
  ```cpp
  // Old (brittle):
  auto legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 2); // ❌ Not robust

  // New (robust):
  for (int i = 0; i < moves_expected; ++i) {
    auto legal_actions = state->LegalActions();
    // There may be 1 or more legal actions, depending on the board
    bool found = false;
    for (auto action : legal_actions) {
      auto moves = state->LongNardeSpielMoveToCheckerMoves(player, action);
      if (matches_expected(moves, i)) {
        state->ApplyAction(action);
        found = true;
        break;
      }
    }
    SPIEL_CHECK_TRUE(found); // Ensure the expected move is present
  }
  ```
- **Summary:**
  - Always check for the correct *sequence* of half-moves, not a fixed number of legal actions at each step.
  - This approach is robust to forced-move, pass, and bear-off scenarios, and should be used for all test updates related to the half-move API.

### 8.6. API Parity: Legal Move Selection Must Match Classic Rules

- **Requirement:** The new half-move API must preserve the legal move selection logic of the old API, especially for bearing off.
- **Canonical Rule (see Nardy Rules above, Rule 6):**
  - When bearing off, a checker may only be borne off with a die if no checker occupies a higher point. If a checker is on a higher point, the die must be used to move that checker, not to bear off a lower one.
- **Test Reference:**
  - The test `BearingOffLogicTestBlackNearEnd` is a canonical case. It must not be possible to bear off from point 13 with die 5 if point 14 is occupied.
- **Migration Guidance:**
  - When updating or reviewing the new API, always check that the set of legal moves for any state matches the old API and the rules above.
  - If a test fails because the new API allows a bear-off from a lower point with a higher die while a higher point is occupied, this is a bug in the move generation logic, not the test.
- **Action:**
  - All contributors must ensure that the new API does not relax or alter the move selection rules, especially for bearing off. Use the referenced test as a check.

## 9. Final State

- [x] The codebase now uses a robust, RL-friendly half-move API with a fixed 25-action space.
- [x] All tests are updated to operate at the half-move level, with clear, maintainable logic for move selection and validation.
- [x] Debugging and diagnostics are streamlined for future migrations or rule changes.
- [x] All documentation, constants, and test logic are up to date and consistent with the new API.

## 10. Canonical Half-Move API Logic (Post-Migration)

- Generate all valid sequences of half-moves for the current player and dice, using all game rules (head, bridge, bear-off, destination, etc).
- Filter for the longest sequences (maximum number of non-pass moves).
- For each longest sequence:
  - Take the first half-move.
  - Encode it as an action ID (pos for move, 24 for pass).
- Collect all unique first half-moves (so the player sees only one action per possible first move).
- When a half-move is applied:
  - Filter the set of valid sequences to only those that start with the chosen half-move.
  - Repeat the process for the next half-move, until all dice are used or no moves remain.

**Key Points:**
- All first half-moves are guaranteed legal by construction.
- No need to re-check legality at the action-mapping step.
- This matches the original HEAD logic for full-turn actions, adapted for sequential half-moves.
- If too many actions are returned, the bug is in sequence generation or filtering, not in the mapping.

--- 