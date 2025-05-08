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

- [x] 1. Define Bitboard State
   - Add two `uint32_t player_occupancy_[2]` fields to `LongNardeState`.
   - Add `int checkers_on_board_count_[2]` field to track on-board counts.

- [x] 2. Initialize Bitboards and Counts
   - In the constructor and `SetState`, populate `player_occupancy_` and `checkers_on_board_count_` from `board_`.
   - In `LongNardeApplyCheckerMove` and `LongNardeUndoCheckerMove`, update both bitboards and counts incrementally.

- [x] 3. Precompute Opponent-Ahead Masks
   - Create a static array `uint32_t opponent_ahead_mask_[2][kNumPoints]`.
   - Implement a one-time initializer to compute masks for each player and starting point.

- [x] 4. Implement Optimized Bridge Check Stub
   - Add early exits: bear-off moves, `checkers_on_board_count_[player] < 6`, or opponent has no checkers.
   - Compute `hypothetical_occ` via bitwise removal of `from_pos` and addition of `to_pos`.
   - Build `uint64_t doubled = hypothetical_occ | (uint64_t(hypothetical_occ) << kNumPoints)`.
   - For offsets 0–5: calculate `start = (to_pos - offset + kNumPoints) % kNumPoints`, test 6-bit window mask.
   - If a 6-block is found and `(player_occupancy_[opponent] & opponent_ahead_mask_[player][start]) == 0`, return `true`.
   - Return `false` if no illegal block detected.

- [x] 5. Wire Optimized Stub into Validations
   - Use both original and optimized checks in `LongNardeIsValidCheckerMove` and `HasIllegalBridge`.
   - Log mismatches with `ToString()` state and move parameters.

- [x] 6. Enable Profiling Instrumentation
   - Add `-pg` to CXXFLAGS or `CMAKE_CXX_FLAGS` for profiling.
   - Rebuild the project.

- [x] 7. Run Consistency and Performance Test
   - Execute `random_sim_test` under profiling; capture any warnings.
   - Use `gprof` with `gmon.out` to generate original profile.

- [x] 8. Swap Stub to Optimized Logic
   - Replace stub body of `WouldFormBlockingBridgeOptim` with actual optimized code.
   - Rebuild and rerun `random_sim_test` under the same profiling setup.

9. Compare Profiles
   - Generate profile reports for both original and optimized builds.
   - Compare CPU time spent in bridge checks and related loops.

10. Analyze and Iterate
    - Resolve any mismatches by inspecting logged states.
    - Measure speed-ups; explore further optimizations as needed.

