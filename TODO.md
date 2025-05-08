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

### Phase 1: Modify `LongNardeState` Member Variables and Basic Accessors (`long_narde.h`, `long_narde_state.cc`)

1.  **Modify `board_` Member Variable (`long_narde.h`):** ✅
    *   Comment out or remove: `std::vector<std::vector<int>> board_;`
    *   Add new member: `uint8_t board_data_[kNumPlayers * kNumPoints];`
    *   Keep existing members `uint32_t player_occupancy_[kNumPlayers];` and `int checkers_on_board_count_[kNumPlayers];`. Their updates will be ensured.

2.  **Update Constructor (`LongNardeState::LongNardeState` in `long_narde_state.cc`):** ✅
    *   Initialize `board_data_` as a flat array (e.g., `std::fill_n(board_data_, kNumPlayers * kNumPoints, 0);` in the constructor body).
    *   The loop that initializes `player_occupancy_` and `checkers_on_board_count_` (currently lines 48-57) will now read from `board_data_` after `SetupInitialBoard` populates it.
        ```cpp
        // Example: After SetupInitialBoard() has populated board_data_
        // for (int p = 0; p < kNumPlayers; ++p) {
        //   uint32_t occ = 0;
        //   int cnt = 0;
        //   for (int pos = 0; pos < kNumPoints; ++pos) {
        //     uint8_t point_count = board_data_[p * kNumPoints + pos];
        //     if (point_count > 0) {
        //       occ |= (1u << pos);
        //       cnt += point_count;
        //     }
        //   }
        //   player_occupancy_[p] = occ;
        //   checkers_on_board_count_[p] = cnt;
        // }
        ```

3.  **Update `SetupInitialBoard()` (`long_narde_state.cc` lines 67-71):** ✅
    *   Change assignments to `board_` to use `board_data_` with 1D indexing:
        *   `board_data_[kXPlayerId * kNumPoints + kWhiteHeadPos] = kNumCheckersPerPlayer;`
        *   `board_data_[kOPlayerId * kNumPoints + kBlackHeadPos] = kNumCheckersPerPlayer;`
    *   **Important**: Ensure `player_occupancy_` and `checkers_on_board_count_` are correctly initialized after `board_data_` is set (either by calling the update logic here or by structuring constructor calls appropriately).

4.  **Refactor `board()` accessor to `GetCount()` (`long_narde.h`, `long_narde_state.cc` line 94):** ✅
    *   Rename `int board(int player, int pos) const` to `uint8_t GetCount(int player, int pos) const`.
    *   Make it `inline` in `long_narde.h`.
    *   Implement to access `board_data_` with 1D indexing and appropriate bounds checking (return 0 or error for invalid `pos`).
        ```cpp
        // // In long_narde.h
        // inline uint8_t GetCount(int player, int pos) const {
        //   if (pos < 0 || pos >= kNumPoints) { /* Handle error or return 0 */ }
        //   return board_data_[player * kNumPoints + pos];
        // }
        ```
    *   Remove old implementation from `long_narde_state.cc`.

5.  **Create `IsOccupied()` accessor (`long_narde.h`):** ✅
    *   Add new `inline` method using `player_occupancy_`:
        ```cpp
        // // In long_narde.h
        // inline bool IsOccupied(int player, int pos) const {
        //   if (pos < 0 || pos >= kNumPoints) { return false; }
        //   return (player_occupancy_[player] & (1u << pos)) != 0;
        // }
        ```

### Phase 2: Implement Board Modification Logic with Consistent Updates (`long_narde_state.cc`)

1.  **Create Private Helper Mutator Methods (Recommended) (`long_narde.h` private section, `long_narde_state.cc`):** ✅
    *   `void SetPointCount(int player, int pos, uint8_t count)`: Updates `board_data_`, `player_occupancy_`, and `checkers_on_board_count_`.
    *   `void IncrementPoint(int player, int pos)`: Updates `board_data_`, `player_occupancy_` (if count was 0), and `checkers_on_board_count_`.
    *   `void DecrementPoint(int player, int pos)`: Updates `board_data_`, `player_occupancy_` (if count becomes 0), and `checkers_on_board_count_`.

2.  **Refactor `LongNardeApplyCheckerMove` (`long_narde_state.cc` lines 104-196):** ✅
    *   Replace `board_[player][move.pos]--;` with `DecrementPoint(player, move.pos);` (or inlined logic).
    *   Replace `board_[player][move.to_pos]++;` with `IncrementPoint(player, move.to_pos);` (or inlined logic).
    *   For bearing off (`scores_[player]++;`): also decrement `checkers_on_board_count_[player]`.

3.  **Refactor `LongNardeUndoCheckerMove` (`long_narde_state.cc` line 198 onwards):** ✅
    *   Replace direct `board_` modifications with `IncrementPoint` / `DecrementPoint` (or inlined logic) to reverse the move.
    *   When undoing a bear-off (decrementing `scores_[player]`): also increment `checkers_on_board_count_[player]`.

### Phase 3: Update All Other Code to Use New Accessors/Mutators: ✅

1.  **Global Search and Replace:**
    *   Replace `board_[player][pos]` reads with `GetCount(player, pos)`. Analyze writes (should be in Apply/Undo).
    *   Replace `board(player, pos)` calls with `GetCount(player, pos)` or `IsOccupied(player, pos)` as appropriate.

2.  **Specific Function Refactoring (`long_narde_validation.cc` and others):**
    *   **`LongNardeState::IsFirstTurn`**: Use `GetCount()`.
    *   **`LongNardeState::WouldFormBlockingBridge`**: Adapt `temp_board` logic for 1D array or ensure reliance on `WouldFormBlockingBridgeOptim`.
    *   **`LongNardeState::LongNardeIsValidCheckerMove`**: Use `IsOccupied()` or `GetCount()` for checks.
    *   **`LongNardeState::AllInHome`**: Refactor to use `player_occupancy_` and a precomputed `non_home_mask_[player]`.
    *   **`LongNardeState::FurthestCheckerInHome`**: Refactor to use `player_occupancy_` and bit manipulation.
    *   **`LongNardeState::WouldFormBlockingBridgeOptim`**: Ensure it uses the dynamically updated `player_occupancy_`. Consider 5-point block pre-check.

3.  **Test Code (`long_narde_test_*.cc` files):**
    *   Update board setup and checks to use new accessors (`GetCount`, `IsOccupied`).

### Phase 4: Unit Tests for Bitboard Logic

1.  **Run All Unit Tests.** Add new tests for bitboard logic in `AllInHome`, `FurthestCheckerInHome`. ✅
2.  **Profile:** Re-run profiler on random simulation to measure impact. ✅

### Note on Redundant State: Remove checkers_on_board_count_

- The member `checkers_on_board_count_[player]` is redundant because:
  - `scores_[player]` tracks the number of checkers a player has borne off.
  - The total number of checkers per player is always 15 (`kNumCheckersPerPlayer`).
  - Therefore, `checkers_on_board_count_[player] + scores_[player] == kNumCheckersPerPlayer` always holds.
- **Action:** ✅
  - Remove `checkers_on_board_count_` from the state.
  - Wherever the count of checkers on the board is needed, use `kNumCheckersPerPlayer - scores_[player]`.
  - Update all logic and helper methods to reflect this simplification.
  - Only update `scores_[player]` on bear-off and undo; do not maintain a separate on-board count.


