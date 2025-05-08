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
*   **Performance Optimization:** Profile with gprof shows `LongNardeGenerateAllHalfMoves` dominates runtime; optimize half-move generation by reducing dynamic allocations (std::set/std::vector growth), minimizing state cloning, and caching repeated computations.

## Plan

- [x] Locate and isolate the existing iterative DFS implementation
- [x] Declare a recursive helper method `RecLegalMoveSequences` in `long_narde.h`
- [x] **Add parallel move-generation harness:** in `LegalActions()`, call both the current iterative DFS and the new recursive helper (initially stubbed) and compare their result sets for equality under debug flag
- [x] Implement `RecLegalMoveSequences` in `long_narde_legal_actions.cc` with full apply/undo recursion
- [x] Remove `LongNardeIterativeLegalMoves`, `ExplorationState`, and related cloning code
- [x] Optimize `LongNardeGenerateAllHalfMoves` to use fixed-size containers and inline deduplication
- [x] Refactor `LongNardeFilterBestMoveSequences` to eliminate per-sequence `Clone()` calls (profiling shows Clone() accounts for ~18.75% of total runtime), using apply/undo and state caching instead of cloning
    - [x] Update `RecLegalMoveSequences` signature to use `std::set<std::pair<std::vector<LongNardeCheckerMove>, bool>>* movelist`
    - [x] In base-case insertion, compute `bool is_term = this->IsTerminal();` and insert `{moveseq, is_term}` into movelist
    - [x] In initial-pass insertion branch, compute `is_term` and insert `{pass_seq, is_term}` into movelist
    - [x] In forced-pass insertion branch, compute `is_term` and insert `{pass_seq, is_term}` into movelist
    - [x] Change callers (`LegalActions`, `LongNardeGenerateMoveSequences`) to declare and use `std::set<std::pair<std::vector<LongNardeCheckerMove>, bool>> movelist_set`
    - [x] After recursion, convert `movelist_set` into `std::vector<std::pair<std::vector<LongNardeCheckerMove>, bool>> movelist_with_flags`
    - [x] Update `LongNardeFilterBestMoveSequences` signature to accept `const std::vector<std::pair<std::vector<LongNardeCheckerMove>, bool>>& movelist_with_flags`
    - [x] Remove the clone-based terminal detection loop in `LongNardeFilterBestMoveSequences`
    - [x] Iterate over `movelist_with_flags` and collect sequences with `flag == true` into `terminal_sequences`
    - [x] Adjust filtering loops to unpack `(moveseq, is_term)` from each pair
    - [x] Preserve existing logic for computing `longest_sequence` and `max_non_pass` using only the `moveseq`
    - [x] Implement terminal-first override: if any sequence is terminal, set `filtered_movelist = terminal_sequences`
    - [x] Ensure the pass-only fallback block remains correct without cloning
    - [x] Update `LegalActions()` to call the updated filter and unpack its returned sequences
    - [x] In `LongNardeGenerateMoveSequences()`, extract only the `moveseq` component from flagged pairs for return
- [ ] Profile performance and compare clone overhead (random_sim_test + gperf)
- [x] Clean up debug logs and update this TODO with completed items