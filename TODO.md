# TODO: Long Narde Implementation Plan

## Overview
We are implementing the game rules of Long Narde, based on a copy of "games/backgammon". Key rules include:
1. Setup: White's 15 checkers on point 24; Black's 15 on point 12.
2. Movement: Both move checkers counter-clockwise into home (White 1–6, Black 13–18), then bear off.
3. Starting: Each rolls 1 die; higher is White and goes first (no doubles on first roll).
4. Turns: Roll 2 dice, move checkers exactly by each value. No landing on opponent; if no moves exist, skip; if only one is possible, use the higher die. Doubles grant 4 moves.
5. Head Rule: Only 1 checker may leave the head per turn, except on the first turn if a double 6, 4, or 3 is rolled, allowing 2 moves from the head.
6. Bearing Off: Once all checkers reach home, bear them off with exact or higher rolls.
7. Ending/Scoring: Mars (2 points) if loser bore off none, Oin (1 point) otherwise. Optional last roll tie if loser has >= 14 checkers off.
8. Block (Bridge) Rule: A contiguous block of 6 checkers cannot be formed unless at least 1 opponent checker is ahead.
9. Last Roll Tie: Implemented via `allow_last_roll_tie_` parameter.

## Current Status & Architecture Assessment

*   **Core Implementation:** Most rules (movement, setup, scoring, bearing off, bridge rule, head rule basics, last roll tie) are implemented.
*   **Code Structure:** The monolithic `long_narde.cc` has been successfully split into smaller, functionally-grouped files (`_state.cc`, `_moves.cc`, `_encoding.cc`, `_validation.cc`, `_legal_actions.cc`, `_api.cc`, `_utils.cc`, `_game.cc`). This significantly improves maintainability.
*   **Move Generation:** An iterative move generation approach (`IterativeLegalMoves` using `GenerateMoveSequences`) is in place. However, it currently relies on **cloning the game state** for each move exploration branch to avoid state consistency issues previously encountered with apply/undo. This works for correctness but misses the performance benefits of a true iterative apply/undo approach.
*   **Doubles Handling:** There's a fundamental issue identified: the current `dice_` state representation (2 elements) is insufficient and fragile for handling the 4 moves required on a double roll. This led to test failures (`TestDoubleMove`) and crashes during debugging attempts. A state refactor is needed.
*   **Testing:** Significant refactoring and addition of tests have occurred. Test utilities (`long_narde_test_utils.h/cc`) are in place. While many tests pass, some discrepancies with original test logic might remain, and the crucial `TestDoubleMove` is likely failing due to the doubles handling issue.
*   **Simplification/Docs:** Efforts to simplify `LegalActions`, consolidate constants, and improve documentation have been largely completed.

## Current Priorities

1.  **[HIGH] Refactor State for Doubles Handling:** Implement the proposed state refactoring to correctly handle 4 moves on doubles. **(IN PROGRESS)**
    *   **Goal:** Modify state representation (`dice_`, related flags/counters) to robustly handle 4 moves for doubles, enabling future apply/undo optimization.
    *   **Files Primarily Affected:** `long_narde.h`, `long_narde_state.cc`, `long_narde_api.cc`, `long_narde_moves.cc`, `long_narde_legal_actions.cc`.
    *   [ ] **Step 1: Modify State Definition (`long_narde.h`, `long_narde_state.cc`)**
        *   Change `dice_` from `std::vector<int>(2)` to `std::vector<int>(4)`. Initialize with zeros.
        *   Remove `double_turn_` member variable.
        *   Remove `doubles_moves_made_` member variable.
        *   Update `TurnHistoryInfo` struct to remove `double_turn_` and `doubles_moves_made_`.
    *   [ ] **Step 2: Update Dice Rolling (`long_narde_api.cc`)**
        *   In `ProcessChanceRoll` (or similar logic if moved): Populate `dice_` as `{d1, d2, 0, 0}` for non-doubles, and `{d, d, d, d}` for doubles `d-d`.
    *   [ ] **Step 3: Update Dice Accessors/Checkers (`long_narde_state.cc`)**
        *   Update `DiceValue(int i)` and `IsUsed(int i)` logic if necessary (marking used by adding 6 should still work, but verify access bounds).
        *   Update `UsableDiceOutcome(int i)` to treat `0` as unusable.
    *   [ ] **Step 4: Update Move Application/Undo (`long_narde_moves.cc`, `long_narde_api.cc`)**
        *   Modify `ApplyCheckerMove` to find the *first* unused `dice_[i]` matching `move.die`, mark it as used (add 6).
        *   Modify `UndoCheckerMove` to find the *first* used `dice_[i]` matching `move.die`, mark it as unused (subtract 6).
        *   Update `UndoAction` to remove logic related to saving/restoring `double_turn_` and `doubles_moves_made_` from `TurnHistoryInfo`.
    *   [ ] **Step 5: Update Move Generation (`long_narde_legal_actions.cc`)**
        *   Simplify `GenerateAllHalfMoves` to loop `i` from 0 to 3, check `UsableDiceOutcome(dice_[i])`.
        *   Remove `max_moves_param` from `IterativeLegalMoves` / `GenerateMoveSequences` calls if it exists (logic should naturally handle the 4 dice slots).
    *   [ ] **Step 6: Build and Test**
        *   Perform incremental builds.
        *   Run all tests, paying close attention to `TestDoubleMove` and any tests involving doubles logic.
        *   Debug and fix any failures until all tests pass.

2.  **[MEDIUM] Verify/Fix Test Discrepancies:** After the doubles refactor, re-evaluate and fix any remaining discrepancies between the refactored tests and the original logic/coverage from `long_narde_test.cc`.
    *   [ ] **HeadRuleTest:** Ensure `long_narde_test_movement.cc` accurately covers the original test's intent (Test Case #2).
    *   [ ] **FirstTurnDoublesExceptionTest:** Implement the missing test for the special first-turn doubles head rule exception (Test Case #3).
    *   [ ] **BlockingBridgeRuleTest:** Ensure `long_narde_test_bridges.cc` covers all 4 specific sub-cases from the original test (Test Case #4).

3.  **[MEDIUM] Optimize Iterative Move Generation (Reduce Cloning):** Once correctness is confirmed (especially after the doubles refactor), implement the apply/undo optimization in `IterativeLegalMoves`.
    *   [ ] Refactor the loop in `IterativeLegalMoves` to use `ApplyCheckerMove`, push minimal necessary context onto the stack, recurse/iterate, and then use `UndoCheckerMove`. Avoid `Clone()` within the main generation loop. (Task #10 from previous list).
    *   [ ] Ensure `UndoCheckerMove` correctly restores all necessary state. Pay attention to sequence-specific state like `moved_from_head_this_sequence`.
    *   [ ] Build and test thoroughly to confirm correctness and potentially measure performance improvement.

4.  **[LOW] Simplify `IsFirstTurn` Access:**
    *   [ ] Remove the redundant `is_first_turn()` method, keeping only `IsFirstTurn(Player player)`. (Task 3a from previous list).
    *   [ ] Update all call sites to use `IsFirstTurn(player)`.
    *   [ ] Verify tests still pass.

5.  **[FINAL] Commit Changes:**
    *   [ ] Once all tests pass and priorities are addressed, commit the changes following TDD principles (small, tested commits).

## Deferred / Completed Sections (For Reference - To Be Removed Later)

*   (Sections on Initial Action Items, Specific Code Changes, Build Issues, Test Failures, detailed Test Case Reviews, Optimization Task breakdown, Debugging notes were here - removed as they are completed or superseded by the new priorities).


# CURRENT PRIORITY: Correctness of Existing Move Generation

**Goal:** Ensure the existing recursive (cloning-based) move generation logic correctly implements all game rules before attempting major refactoring or optimization.

**Key Steps:**
*   [*] **Introduce `initial_dice_`:** Add `initial_dice_` member to `LongNardeState`, populate in `ProcessChanceRoll`, handle in `UndoAction`, and use in `IsValidCheckerMove` for the first-turn head rule exception. (Completed)
*   [*] **Refine `is_first_turn_` vs. `IsFirstTurn(player)`:** Ensure the member variable (`is_on_first_turn_`) reflects the start-of-turn status, while the method (`IsFirstTurn(player)`) checks current board state. Update logic to use the correct check where needed (especially around head rule validation during sequence generation). (Completed - `IsLegalHeadMove` now correctly uses `is_on_first_turn_` for the special case and the sequence flag otherwise).
*   [*] **Fix Validation Logic (`IsValidCheckerMove`)**: 
    *   Confirm `initial_dice_` is exclusively used for the special first-turn double rule application throughout the *entire* turn's move sequence generation. (Verified - `initial_dice_` is now set correctly and used in `IsLegalHeadMove`)
    *   Ensure head movement allowance during sequence generation correctly uses the `initial_dice_` check (for the special rule) or the sequence-local `moved_from_head_this_sequence` flag, not the current state's `IsFirstTurn(player)`. (Verified - `IsLegalHeadMove` now uses the sequence flag for the normal case).
*   [*] **Verify Bridge Rule (`WouldFormBlockingBridge`)**: Ensure it's correctly called and evaluated within the move generation/validation process. (Verified during recent debugging)
*   [*] **Comprehensive Testing**: Add/Update tests (e.g., `FirstTurnTest`, `HeadRuleTest`, `BridgeRuleTest`, `BearingOffLogicTest`, `ConsecutiveMovesTest`) to validate the corrected recursive `GenerateMoveSequences` against known-good scenarios and edge cases based on the rules. (Tests are passing after recent fixes for extra turn logic)

**Deferred Tasks (Until Correctness Confirmed):**
*   Task #2 (Comparison Test for `IterativeLegalMoves`) // Task numbers might be outdated
*   Task #10 (Refactor `IterativeLegalMoves` to use Apply/Undo) // Task numbers might be outdated

## Specific Code Changes Completed
- [*] Added constants for head positions: White head (point 24) and Black head (point 12)
- [*] Modified checker movement direction: Both players now move counter-clockwise
- [*] Modified `IsLegalFromTo` function to ensure players cannot land on opponent's checkers
- [*] Implemented `IsHeadPos` function to identify starting positions
- [*] Created `IsLegalHeadMove` function to enforce head movement restrictions
- [*] Added state tracking for first turn to enable the special head rule exception
- [*] Implemented `WouldFormBlockingBridge` function to check for illegal 6-point primes
- [*] Removed "hit" and "bar" mechanics from movement logic
- [*] Updated scoring function to implement mars (2 points) and oin (1 point) scoring
- [*] Updated home regions: White (points 1-6) and Black (points 13-18)
- [*] Modified `AllInHome` and related functions to check correct home regions
- [*] Updated visualization and string representation functions to properly display Long Narde positions
- [*] Added `allow_last_roll_tie_` tracking and modified `IsTerminal()` and `Returns()` to implement the last roll tie rule
- [*] Added test case to verify the last roll tie functionality
- [*] Added `data-testid` and `data-filename` attributes to verification checkboxes in `TESTS.html` for improved interactivity and debugging.

## Build Issues
- [*] Fixed build environment issues through a dedicated build script (build_long_narde.sh)
- [*] Implemented memory limits (1GB) and timeouts (10s) for the random simulation test
- [*] Redirected verbose random simulation test output to a log file for analysis

## Test Failures (Need to Fix)
- [*] **HeadRuleTest**: Fixed issue with the test logic for validating the head rule implementation. The test was incorrectly counting all potential head moves instead of checking the actual number of checkers moved from the head after applying the moves. Updated the test to simulate each action and count the actual checkers that moved from the head.
- [*] **RandomSimTest**: Fixed by disabling excessive debug output. The debug flags in the `LegalCheckerMoves` and `ApplyCheckerMove` functions were set to false to prevent generating millions of lines of debug information that was causing test timeouts.
- [*] **UndoRedoTest**: Had issues with `scores()` method instead of `score()` and needed fixes to use the proper checker move methods.

## Notes
- Follow TDD: Create or update tests that specify the desired behavior before modifying game logic.
- Ensure tests cover all edge cases of the Long Narde rules. 
- The implementation is nearly complete, but there are test failures that need to be resolved:
  1. ✓ Fix the head rule implementation test to properly validate only one checker can leave the head position after the first turn
  2. Investigate and fix the random simulation test failures, possibly related to move validation
  3. ✓ Complete any remaining fixes for proper board, dice, and score state handling

# Long Narde Test Cases Review

[*] Review `NoLandingOnOpponentTest` - Test looks good:
   - Tests basic landing prevention for both players
   - Tests landing prevention with doubles
   - Tests multiple opponent checkers
   - Tests edge cases near board boundaries
   - Includes random simulation test with 100 iterations
   - Comprehensive coverage of no-landing rule

[*] Review `BasicLongNardeTestsDoNotStartWithDoubles` - Test looks good:
   - Properly verifies that initial dice roll never results in doubles
   - Uses sufficient iterations (100)
   - Correctly checks both dice values are different
   - Aligns with Long Narde rules for initial roll

[*] Review `InitialBoardSetupTest` - Test looks good:
   - Correctly verifies White's 15 checkers start on point 24
   - Correctly verifies Black's 15 checkers start on point 12
   - Checks that no other points have any checkers
   - Uses proper constants for positions and number of checkers

[*] Review `HeadRuleTest` - Test has been fixed:
   - Modified test to properly count actual checkers moved from head position rather than just counting moves in action encodings
   - Tests regular turns allow only one checker from head
   - Tests first turn with non-doubles allows only one checker
   - Tests first turn with special doubles (6-6, 4-4, 3-3) allows two checkers
   - Tests first turn with non-special doubles (2-2, 1-1) allows only one checker
   - Tests both White and Black head movement rules
   - Correctly validates that no moves actually result in multiple checkers leaving the head after the first turn

[*] Review `BlockingBridgeRuleTest` - Test looks good:
   - Tests White can't create 6-point prime that traps Black
   - Tests White can create 6-point prime if Black has checkers ahead
   - Tests Black can't create 6-point prime that traps White
   - Tests Black can create 6-point prime if White has checkers ahead
   - Comprehensive coverage of bridge blocking rules

[*] Review `MovementDirectionTest` - Test is good but could be expanded:
   - Correctly verifies White moves counter-clockwise
   - Correctly verifies Black moves counter-clockwise
   - Could add tests with different dice combinations
   - Could add tests with checkers in different positions
   - Could verify bearing off moves follow direction rules

[*] Review `HomeRegionsTest` - Test looks good:
   - Correctly verifies White's home region (points 1-6)
   - Correctly verifies Black's home region (points 13-18)
   - Verifies all other points are not in home for either player
   - Uses proper constants and indices

[*] Review `BearingOffLogicTest` - Test looks good:
   - Tests White bearing off with exact and higher rolls
   - Tests Black bearing off with exact and higher rolls
   - Tests bearing off with doubles
   - Tests prevention of bearing off when checkers outside home
   - Tests score updates and undo functionality for both players
   - Comprehensive coverage of bearing off mechanics

[*] Review `ScoringSystemTest` - Test looks good:
   - Tests Mars scoring (2 points) when winner bears off all while opponent has none
   - Tests Oyn scoring (1 point) when winner bears off all while opponent has some
   - Tests both White and Black can score Mars
   - Tests last roll tie rule in winloss mode (no ties allowed)
   - Tests last roll tie rule in winlosstie mode (ties allowed)
   - Tests winlosstie mode with mars opportunity (verifies mars score takes precedence over tie)
   - Comprehensive coverage of scoring rules and game modes

[*] Review `ActionEncodingTest` - Test looks good:
   - Tests encoding/decoding of moves with high roll first
   - Tests encoding/decoding of moves with low roll first
   - Tests encoding/decoding of pass moves
   - Tests encoding/decoding of mixed regular and pass moves
   - Verifies all encoded actions are within valid range
   - Comprehensive coverage of move encoding functionality

[*] Review `IsPosInHomeTest` - Test looks good:
   - Correctly tests White's home region boundaries
   - Correctly tests Black's home region boundaries
   - Tests both inside and outside positions for each player
   - Uses proper indices and point numbers
   - Comprehensive coverage of home region checks

[*] Review `FurthestCheckerInHomeTest` - Test looks good:
   - Tests empty home board returns -1
   - Tests White's furthest checker in home
   - Tests Black's furthest checker in home
   - Tests both players having checkers in home
   - Tests edge cases at home region boundaries
   - Comprehensive coverage of furthest checker logic

[*] Review `BasicLongNardeTests` - Test looks good:
   - Calls all individual test functions
   - Includes LoadGameTest for basic game loading
   - Proper test organization and execution

[*] Add additional test cases if any important Long Narde rules are not covered by existing tests 
   - [*] Add SingleLegalMoveTest:
     - Created board positions where exactly one move is legal
     - Created board positions where no moves are possible
     - Verified pass action is correctly returned when no moves are available
     - Verified higher die is used when only one die can be used
     - Tested scenarios for both White and Black players

   - [*] Add ConsecutiveMovesTest:
     - Tested that doubles (e.g., 4-4) allow consecutive moves of the same checker
     - Tested proper handling of doubles on non-first turns (extra turn when both dice used)
     - Verified that extra turn is granted on any double roll (unless already extra turn)
     - Tested scenarios for both White and Black players

   - [*] Add UndoRedoTest:
     - Tested that undo/redo works correctly for head rule moves
     - Tested that undo/redo works for bridging attempts
     - Tested that undo/redo works for bearing off (including score updates)
     - Verified no state corruption after multiple undo/redo cycles
     - Fixed issues with method names (`score()` vs `scores()`)

   - [*] Fix RandomSimTest memory issues:
     - Diagnosed memory growth issue (history accumulation during long games)
     - Implemented a modified version with improved memory management via periodic state cloning
     - Added validation to detect any "invalid move from X to Y" errors
     - Added statistics tracking for better test reporting
     - Reduced verbosity by disabling debug output in `LegalCheckerMoves` and `ApplyCheckerMove`
     - Successfully fixed excessive debug output and timeout failures

   - [*] Add ComplexEndgameTest:
     - Tested near-endgame positions for mars/oin scoring (2 points vs 1 point)
     - Tested last roll tie scenarios extensively in winlosstie_scoring mode
     - Tested edge cases where one player has 14 checkers off
     - Verified tie behavior differences between scoring modes
     - Tested multiple edge cases of the Long Narde endgame scoring rules 

# Long Narde Test Cases TODO

This file contains a list of test cases from the original `long_narde_test.cc` that need to be properly implemented in separate files.

## Missing Test Cases

The following test cases from the original test file have not been properly implemented in separate files:

1. **HeadRuleTest** (Test Case #2)
   - Original Location: `long_narde_test.cc`
   - Expected Location: `long_narde_test_movement.cc`
   - Issue: A different test named `HeadRuleTestInternal` exists in `long_narde_test_movement.cc`, but it uses a different approach and doesn't exactly match the original test. The original test specifically checks for `actual_multi_head_moves` which isn't present in the new implementation.

2. **FirstTurnDoublesExceptionTest** (Test Case #3)
   - Original Location: `long_narde_test.cc`
   - Expected Location: `long_narde_test_movement.cc` or `long_narde_test_basic.cc`
   - Issue: This test is completely missing from the separate test files.

3. **BlockingBridgeRuleTest** (Test Case #4)
   - Original Location: `long_narde_test.cc`
   - Expected Location: `long_narde_test_bridges.cc`
   - Issue: While there is a `TestBridgeFormation` test in `long_narde_test_bridges.cc`, it doesn't exactly match the implementation of the original `BlockingBridgeRuleTest`. The original test had 4 specific test cases that aren't all covered.

## Correctly Implemented Tests

1. **InitialBoardSetupTest** (Test Case #1)
   - Original Location: `long_narde_test.cc`
   - Current Location: `long_narde_test_basic.cc`
   - Status: Correctly implemented with minor refactoring that preserves the test logic. 

# Test Case Implementation Issues

The following test cases from the original long_narde_test.cc are not properly implemented in separate files:

1. ~~`HomeRegionsTest` (test case 6) - Not implemented in any separate file~~ - **Fixed**: Implemented in long_narde_test_movement.cc
2. ~~`BearingOffLogicTest` (test case 7) - Not implemented in bearing_off_test.cc which handles a different bearing off test~~ - **Fixed**: Implemented in long_narde_test_endgame.cc
3. ~~`ScoringSystemTest` (test case 8) - Not implemented in any separate file, should probably be in long_narde_test_endgame.cc~~ - **Fixed**: Implemented in long_narde_test_endgame.cc

The following tests have differences between the original implementation and the separate file:

1. ~~`MovementDirectionTest` (test case 5) - Implementation in long_narde_test_movement.cc is different from the original~~ - **Fixed**: Updated in long_narde_test_movement.cc to match original
2. ~~`NoLandingOnOpponentTest` (test case 9) - Implementation in long_narde_test_movement.cc is different from the original~~ - **Fixed**: Updated in long_narde_test_movement.cc to match original 

# Long Narde Test Implementation TODOs

## Test Case Implementation Discrepancies

### Test Case #18: BearingOffFromPosition1Test
~~The implementation in `long_narde_test_endgame.cc` significantly differs from the original test in `long_narde_test.cc`:~~

~~- **Original Test**:~~
  ~~- Uses a board with White having 7 checkers at position 1~~
  ~~- Tests bearing off with dice values 1 and 3~~
  ~~- Checks if bearing off is allowed with both exact roll (1) and higher roll (3)~~
  ~~- Verifies pass action availability~~
  ~~- Tests a bug where bearing off should be allowed with any roll but is only working with exact roll~~

~~- **New Implementation**:~~
  ~~- Uses a different board setup (White has checkers at positions 1, 2, 3, 4, 5)~~
  ~~- Uses dice 6 and 2 instead of 1 and 3~~
  ~~- Only tests bearing off with higher roll (6 from position 1)~~
  ~~- Tests both White and Black bearing off~~
  ~~- Does not check for pass action~~

~~**Action Required**: Update `long_narde_test_endgame.cc` to match the original test's logic and edge cases.~~

**Fixed**: Updated BearingOffFromPosition1Test in long_narde_test_endgame.cc to match the original test's implementation.

# Long Narde Test Cases Review

// ... existing code ... 

# Optimization Tasks

Retrieval Hint: Search `Principle:` in knowledge graph for general coding guidelines.

**Note on Move Generation Comparison:**
*   Initial comparison tests (`long_narde_test_movegen_comparison.cc`) revealed that the first implementation of `RecursiveLegalMoves` was flawed. It incorrectly handled non-double dice, exploring only paths starting with the second die if the first die was completely unplayable, thus missing valid sequences. This was corrected by refactoring `FindRecursiveMoves` to explore paths starting with each distinct die independently. The iterative approach (`IterativeLegalMoves` via `GenerateMoveSequences`) handled this correctly from the start.

## Code Simplification
	1.	Refactor LegalActions Function
	•	What: Break down the complex LegalActions function into smaller, focused helper functions.
	•	Where: long_narde.cc (updated lines ~607-777)
	•	Why: Improve readability and maintainability by separating move generation, filtering, and higher-die rule logic.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorLegalActions`
	•	Tasks:
	•	[*] Create a helper for move sequence generation (e.g., `GenerateMoveSequences`).
	•	[*] Create a helper for move sequence filtering (e.g., `FilterBestMoveSequences`).
	•	[*] Create a helper for higher-die rule application (e.g., `ApplyHigherDieRuleIfNeeded`).
	•	[ ] Create a helper for doubles move handling. (Deferred - may fit better in RecLegalMoves/Encoding refactor)
	•	[ ] Use early returns to avoid deeply nested conditions. (Deferred - current structure is fairly linear)
	2.	Simplify RecLegalMoves Function (DEFERRED - See Current Priority)
	•	What: Convert the recursive RecLegalMoves function to an iterative approach using a stack or queue.
	•	Where: long_narde.cc (lines 1598–1681) -> Now in `long_narde_legal_actions.cc`
	•	Why: Reduce complexity and potential stack overflow risks. (**Deferring** until current logic is proven correct).
	•	Retrieval Hint: Use query `Task:LongNardeRefactorRecLegalMoves`
	•	Tasks:
	•	[x] Design an iterative algorithm structure (BFS/DFS style) to generate half-move sequences.
	•	[x] Implement stack-based (or queue-based) move generation that mimics the current recursive behavior.
	•	[x] Maintain existing pruning optimizations (e.g., bridging checks and head-rule validations) during iteration.
	•	[x] Add comprehensive tests to verify equivalence with the recursive approach.
    •   **Note:** The initial iterative approach using apply/undo on a single clone (`IterativeLegalMoves`) caused state consistency issues. The current implementation uses cloning per branch exploration, which resolved the test failures (e.g., `FirstTurnTest`, `TestBasicMovement`).
    •   **Update (Debugging FirstTurnTest):**
        *   Resolved `FirstTurnTest` failure related to `IterativeLegalMoves` head rule state management.
	3.	Clarify Action Encoding/Decoding
	•	What: Refactor encoding logic with helper functions and clear documentation.
	•	Where: long_narde.cc (lines 157–266 for encoding, 268–323 for decoding)
	•	Why: Make the complex encoding logic more maintainable.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorEncodingDecoding`
	•	Tasks:
	•	[x] Add detailed documentation of the encoding scheme (normal moves, doubles, pass moves, offsets).
	•	[x] Create helpers such as EncodeSingleMove, DecodeSingleMove, EncodeDoubles, and DecodeDoubles.
	•	[x] Create additional helpers for pass move handling. (Integrated into main helpers)
	•	[x] Add validation checks for encoding ranges to ensure no collisions. (Handled by design and SPIEL_CHECKs)
	3a. Simplify IsFirstTurn Access
	•   What: Remove the redundant `is_first_turn()` method, keeping only `IsFirstTurn(Player player)`.
	•   Where: `long_narde.h`, `long_narde.cc`, and any calling code (tests).
	•   Why: Improve API clarity and consistency. `IsFirstTurn(player)` is less ambiguous.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorIsFirstTurn`
	•   Tasks:
	    *   [ ] Remove the `is_first_turn()` declaration and definition.
	    *   [ ] Update all call sites (identified during refactoring in tests) to use `IsFirstTurn(player)` instead.
	    *   [ ] Verify tests still pass.

## Code Structure
	4.	Group Related Functions
	•	What: Organize related functions into logical sections.
	•	Where: long_narde.cc
	•	Why: Improve code navigation and maintainability.
	•	Tasks:
	•	[x] Group validation functions together.
	•	[x] Group move generation functions (including the new iterative version of move sequence generation).
	•	[x] Group encoding/decoding functions with their respective constants. (Done via file splitting)
	•	[x] Add clear section comments (e.g., // ===== Move Generation =====, // ===== Encoding/Decoding =====).
	5.	Consolidate Constants
	•	What: Reorganize constants between header and implementation.
	•	Where: long_narde.h and long_narde.cc
	•	Why: Better organize game rules versus implementation details.
	•	Tasks:
	•	[x] Move game rule constants (e.g., board size, home regions, head positions) to the header with clear documentation.
	•	[x] Move encoding constants (e.g., kDigitBase, kPassOffset, kDoublesOffset) to the implementation file. (Done for encoding constants)
	•	[x] Document the purpose of each constant.
	•	[x] Update any code references affected by the move.
	6.	Split long_narde.cc into Smaller Files
	•	What: Divide the large `long_narde.cc` file into smaller, more manageable units based on functionality.
	•	Where: `open_spiel/games/long_narde/` directory.
	•	Why: Improve build times, reduce cognitive load, enhance maintainability, and facilitate parallel development. Target ~300 lines per file.
	•	Retrieval Hint: Use query `Task:LongNardeSplitFile`
	•	Tasks:
	•	[ ] Identify logical functional groups (e.g., state_setup, movement, move_generation, encoding, validation, game_logic, string_utils).
	•	[*] Plan the new file structure (e.g., `long_narde_state.cc`, `long_narde_moves.cc`, `long_narde_encoding.cc`, etc.).
	•	[*] Create new `.cc` files for each group.
			*   [x] `long_narde_state.cc`: Contains `LongNardeState` constructor and `SetupInitialBoard`.
			*   [x] `long_narde_moves.cc`: Contains movement-related functions (`ApplyCheckerMove`, `UndoCheckerMove`, `GetToPos`).
			*   [x] `long_narde_encoding.cc`: Will contain encoding/decoding functions (`CheckerMovesToSpielMove`, `SpielMoveToCheckerMoves`, `NumDistinctActions`, etc.).
			*   [x] `long_narde_validation.cc`: Will contain validation functions (`IsValidCheckerMove`, `WouldFormBridge`, `IsHeadPos`, `IsLegalHeadMove`, `IsFirstTurn`, etc.).
			*   [x] `long_narde_legal_actions.cc`: Will contain `LegalActions` and its helpers (`GenerateMoveSequences`, `FilterBestMoveSequences`, `ApplyHigherDieRuleIfNeeded`, `RecLegalMoves` [iterative version later]).
			*   [x] `long_narde_api.cc`: Will contain core Spiel API implementations (`CurrentPlayer`, `DoApplyAction`, `UndoAction`, `IsTerminal`, `Returns`, `ObservationString`, `ObservationTensor`, `Clone`, `ChanceOutcomes`).
			*   [x] `long_narde_utils.cc`: Will contain general utility functions (`ToString`, `ActionToString`, `DiceToString`, `ParseScoringType`, etc.).
			*   [x] `long_narde_game.cc`: Will contain the `LongNardeGame` class definition and factory function.
	•	[*] Move corresponding function implementations from `long_narde.cc` to the new files.
			*   [x] Moved `LongNardeState::LongNardeState`, `LongNardeState::SetupInitialBoard` to `long_narde_state.cc`.
			*   [x] Moved movement functions to `long_narde_moves.cc`.
			*   [x] Moved encoding functions to `long_narde_encoding.cc`.
			*   [x] Moved validation functions to `long_narde_validation.cc`.
			*   [x] Moved legal action generation functions to `long_narde_legal_actions.cc`.
			*   [x] Move core API functions to `long_narde_api.cc`.
			*   [x] Move utility functions to `long_narde_utils.cc`.
			*   [x] Move `LongNardeGame` class and factory to `long_narde_game.cc`.
	•	[*] Ensure necessary includes (`long_narde.h`, etc.) are added to each new file.
			*   [x] Added includes to `long_narde_state.cc`.
			*   [x] Added includes to `long_narde_moves.cc`.
			*   [x] Added includes to `long_narde_encoding.cc`.
			*   [x] Added includes to `long_narde_validation.cc`.
			*   [x] Added includes to `long_narde_legal_actions.cc`.
			*   [x] Add includes to `long_narde_api.cc`.
			*   [x] Add includes to `long_narde_utils.cc`.
			*   [x] Add includes to `long_narde_game.cc`.
	•	[*] Update `CMakeLists.txt` to include the new source files in the `open_spiel_long_narde` target.
			*   [x] Added `long_narde_state.cc` to `open_spiel/games/CMakeLists.txt`.
			*   [x] Added `long_narde_moves.cc` to `CMakeLists.txt`.
			*   [x] Added `long_narde_encoding.cc` to `CMakeLists.txt`.
			*   [x] Added `long_narde_validation.cc` to `CMakeLists.txt`.
            *   [x] Added `long_narde_legal_actions.cc` to `CMakeLists.txt`.
			*   [x] Add `long_narde_api.cc` to `CMakeLists.txt`.
			*   [x] Add `long_narde_utils.cc` to `CMakeLists.txt`.
			*   [x] Add `long_narde_game.cc` to `CMakeLists.txt`.
	•	[*] Perform incremental builds and tests after moving each functional group to ensure correctness.
			*   [x] Build/test successful after moving state setup functions.
			*   [x] Build/test successful after moving movement functions.
			*   [x] Build/test successful after moving encoding functions.
			*   [x] Build/test successful after moving validation functions.
			*   [x] Build/test successful after moving legal action functions.
			*   [x] Build/test successful after moving API functions.
			*   [x] Build/test successful after moving utility functions.
			*   [x] Build/test successful after moving game class functions.
	•	[x] Refactor remaining code in `long_narde.cc` (likely containing `LongNardeGame` class and main state methods) for clarity.

## Comments and Documentation
	7.	Enhance Function Comments
	•	What: Add detailed comments for complex functions.
	•	Where: Key functions in long_narde.cc
	•	Why: Improve code understanding.
	•	Tasks:
	•	[x] Document the logic of the (now iterative) move sequence generation function.
	•	[x] Document the checks in IsValidCheckerMove (bounds, head rule, bearing off, opponent occupancy, bridging).
	•	[x] Document the bridge rule implementation and the rationale behind it.
	•	[x] Add parameter and return documentation using a consistent style (e.g., Doxygen).
	8.	Document Encoding Logic
	•	What: Add comprehensive documentation for the encoding scheme.
	•	Where: long_narde.cc (lines 157–266, 268–323)
	•	Why: Clarify the complex encoding schemes for normal moves, doubles, and pass moves.
	•	Tasks:
	•	[x] Document normal move encoding (e.g., pos * 6 + (die - 1)).
	•	[x] Document doubles move encoding (including use of an offset).
	•	[x] Document pass move handling (using kPassOffset + (die - 1)).
	•	[x] Add explanations for the encoding ranges and any potential edge cases.

## Performance
	9.	Optimize Move Sequence Storage
	•	What: Replace `std::set<std::vector<CheckerMove>>` with `std::vector` to potentially reduce overhead in move generation.
	•	Where: `long_narde_legal_actions.cc` (specifically `GenerateMoveSequences`, `IterativeLegalMoves`, `FilterBestMoveSequences`, `LegalActions`, `ApplyHigherDieRuleIfNeeded`), `long_narde.h` (declarations).
	•	Why: Reduce overhead from `std::set` operations (insertion, uniqueness checks).
	•	Retrieval Hint: Use query `Task:LongNardeOptimizeMoveStorage`
	•	Tasks:
		*   [x] Change `movelist` types from `set` to `vector` in relevant functions.
		*   [x] Replace `insert` with `push_back` in `IterativeLegalMoves`.
		*   [x] Add `std::sort` and `std::unique` in `GenerateMoveSequences` to maintain uniqueness after collection.
		*   [x] Update function declarations in `long_narde.h`.
		*   [x] Verify `CheckerMove::operator<` exists for sorting.
		*   [x] Build and test successfully.
	10.	Reduce Cloning in Move Generation (DEFERRED - See Current Priority)
	•	What: Modify `IterativeLegalMoves` to use apply/undo on the current state instead of cloning for most branches.
	•	Where: `long_narde_legal_actions.cc` (within `IterativeLegalMoves`).
	•	Why: Avoid expensive `Clone()` calls during the depth-first search. (**Deferring** until current logic is proven correct).
	•	Retrieval Hint: Use query `Task:LongNardeReduceCloning`
	•	Tasks:
		*   [ ] Refactor the loop in `IterativeLegalMoves` to apply a move, push state parameters (or a lighter context object), explore, and then undo the move. (**Note:** Current implementation uses cloning per branch, which works but is less efficient).
		*   [ ] Only clone when necessary (potentially never if using a purely recursive approach or if state needs to be preserved across stack unwinds).
		*   [x] Ensure `UndoCheckerMove` correctly restores all necessary state. Pay attention to sequence-specific state like `moved_from_head_this_sequence`.
		*   [x] Build and test successfully.

## Testing Refactoring
	11.	Consolidate Test Helper Functions
	•	What: Move common test setup functions (e.g., board setup, action application) into a shared test utility file.
	•	Where: Create `long_narde_test_utils.h/cc`.
	•	Why: Reduce code duplication in test files.
	•	Tasks:
		*   [x] Identify common setup patterns in existing tests.
		*   [x] Create helper functions in a new utility file.
		*   [x] Refactor existing tests to use the shared utilities.
		*   [x] Update `CMakeLists.txt` to include the new test utility files.

## Progress Tracking
	•	[x] Code Simplification (Tasks 1–3a)
	•	[x] Code Structure (Tasks 4–6)
	•	[x] Documentation (Tasks 7–8)
	•	[*] Performance (Tasks 9–10)
	•	[x] Testing Refactoring (Task 11)

## Summary of Key Recommendations
	•	Break large functions into smaller helpers for move generation, filtering, and encoding/decoding.
	•	Adopt an iterative approach for move sequence generation to simplify recursion and allow early pruning.
	•	Document thoroughly—especially the complex encoding schemes and bridging logic—to ease future maintenance.
	•	Group related logic (validation, move generation, encoding) with clear section comments to improve code navigation.
	•	Reduce cloning overhead and improve move sequence storage using vectors and post-processing instead of sets.
	•	Expand and automate tests for encoding, movement, and bridging rules to ensure robust functionality.

By addressing these tasks, the Long Narde codebase will become simpler, better organized, more performant, and easier to maintain or extend in the future. 


# Debugging Doubles Move Generation (Task #2 Comparison Test Failure)

**Problem:** The `TestDoubleMove` in `long_narde_test_movegen_comparison.cc` consistently failed, generating only 2 moves for a double roll instead of the expected 4. Attempts to fix this led to crashes (`Spiel Fatal Error: ... i < dice_.size()` with `i = 4, dice_.size() = 2`).

**Failed Approaches:**

1.  **Modify `GenerateAllHalfMoves` to Ignore `UsableDiceOutcome`:**
    *   Idea: If `double_turn_` is true, allow `GenerateAllHalfMoves` to generate moves for the double die value regardless of whether the two entries in `dice_` were marked as used.
    *   Result: Led to crashes, likely because `ApplyCheckerMove` couldn't find a corresponding usable die entry in `dice_` to mark after the 2nd move.

2.  **Refine `GenerateAllHalfMoves` Doubles Check:**
    *   Idea: If `double_turn_`, only generate a move for the double value if at least one corresponding entry in `dice_` was still marked usable.
    *   Result: Still led to the `dice_[4]` access crash, suggesting the state inconsistency occurred elsewhere or was more fundamental.

3.  **Modify `IterativeLegalMoves` to Manually Add Moves:**
    *   Idea: Keep `GenerateAllHalfMoves` simple. If it returned no usable moves during a doubles turn (because `dice_` entries were marked used) but fewer than 4 moves had been made, manually re-check and add valid moves for the double value within the `IterativeLegalMoves` loop.
    *   Result: Still led to the `dice_[4]` access crash.

4.  **Introduce `doubles_moves_made_` Counter:**
    *   Idea: Decouple doubles move tracking from the 2-element `dice_` array. Add a counter, increment/decrement it in `Apply/UndoCheckerMove`, and have `GenerateAllHalfMoves` check this counter (< 4) instead of `UsableDiceOutcome` for doubles.
    *   Initial Result: Still crashed (`dice_[4]` access).
    *   Correction: Realized the counter wasn't saved/restored in `TurnHistoryInfo` / `UndoAction`. Fixed that.
    *   Final Result: Still crashed with the exact same `dice_[4]` access error, indicating the root cause wasn't the undo history but likely a more fundamental mismatch between the 4 moves and the 2-slot `dice_` representation causing state corruption elsewhere.

**Conclusion from Failed Attempts:** The core issue appears to be the inherent difficulty and fragility of managing 4 potential moves using a state representation (`dice_`) designed for only 2 dice. Attempts to patch the logic lead to inconsistencies and crashes.

**Proposed New Approach: Refactor State Representation**

1.  **Change `dice_`:** Modify `std::vector<int> dice_` to always have **4 elements**.
2.  **RollDice:** Store non-doubles (e.g., 3-1) as `{3, 1, 0, 0}`. Store doubles (e.g., 4-4) as `{4, 4, 4, 4}`. ('0' indicates an unused/invalid slot).
3.  **Marking Used:** Continue marking used dice by adding 6 (becoming 7-12).
4.  **`UsableDiceOutcome`:** Update to treat '0' as unusable.
5.  **Remove `double_turn_` Flag:** Redundant. Determine if roll was doubles by checking `DiceValue(0) == DiceValue(1)`.
6.  **Remove `doubles_moves_made_` Counter:** Redundant. The state of the 4-element `dice_` array directly reflects moves made.
7.  **Simplify `GenerateAllHalfMoves`:** Loop `i` from 0 to 3. If `UsableDiceOutcome(dice_[i])` is true, generate moves for `DiceValue(i)`.
8.  **Simplify `ApplyCheckerMove` / `UndoCheckerMove`:** Find the *first available* `dice_` entry matching the `move.die` value and mark/unmark it.
9.  **Simplify `IterativeLegalMoves` / `GenerateMoveSequences`:** Remove `max_moves_param`. The logic naturally stops when `GenerateAllHalfMoves` finds no more usable dice in the 4-element array.
10. **Update Undo History:** Remove `double_turn_` and `doubles_moves_made_` from `TurnHistoryInfo` struct and associated save/restore logic.

This approach aligns the state directly with the maximum number of moves, simplifying logic and hopefully eliminating the source of state corruption. 

# TESTS.html Interactive Features Plan

## Goal
Enhance `TESTS.html` to allow users to view relevant C++ source code snippets for each test case, mark test cases as verified, and automatically manage verification status based on source code changes. This involves integrating source code viewing, `localStorage` persistence, checksum validation, and UI enhancements.

## I. Data Storage Structure (`localStorage`)

1.  **Source Code Files (`sourceFiles` key):** An object storing source code and checksums.
    *   Key: Filename (e.g., `"long_narde_test_actions.cc"`)
    *   Value: Object `{ code: "...", checksum: "sha256_hash_here" }`
2.  **Test Verification Statuses (`verificationStatuses` key):** An object storing verification state linked to source versions.
    *   Key: Unique test case ID (e.g., `"test-actionencodingtest-1"`)
    *   Value: Object `{ verified: true/false, verifiedChecksum: "sha256_hash_of_associated_file_at_verification_time_or_null" }`

## II. HTML Modifications (`TESTS.html`)

1.  **Dependencies:**
    *   [x] Include `highlight.js` core library.
    *   [x] Include `highlight.js` C++ language pack.
    *   [x] Include `highlight.js` CSS theme (e.g., `default.min.css`).
    *   [x] Include `highlightjs-line-numbers.js` plugin.
    *   [x] Include `highlightjs-line-numbers.js` CSS.
    *   [x] Include Bootstrap v3.4 CSS (for modals).
    *   [x] Include Bootstrap v3.4 JS (for modals).
    *   [x] Include jQuery (required by Bootstrap 3 JS).
2.  **Test Suites (`h2`):**
    *   [x] Add `id` attribute based on filename (e.g., `id="suite-long_narde_test_actions.cc"`).
    *   [x] Add `data-filename` attribute (e.g., `"long_narde_test_actions.cc"`).
    *   [x] Add CSS/JS to make the `h2` clickable for collapse/expand.
    *   [x] Add a collapse/expand icon (e.g., +/- or arrow).
    *   [x] Add a clickable "Load Source" link/button next to the filename within the `h2`.
3.  **Test Functions (`h3`):**
    *   [x] Make clickable (wrap in `<a>` or add JS listener).
    *   [x] Add `data-filename` attribute.
    *   [x] Add `data-lines` attribute (e.g., `"14-348"`).
4.  **Test Cases (`.test-case`):**
    *   [x] Add `<input type="checkbox" class="verification-checkbox">` near the `h4`.
    *   [x] Add `data-testid` attribute to checkbox (using test case `id`).
    *   [x] Add `data-filename` attribute to checkbox.
    *   [ ] Ensure the parent `.test-function` div has a unique ID or class identifiable by filename.
    *   [ ] Ensure the parent `.test-suite` div has a unique ID identifiable by filename.
5.  **Modals (Bootstrap 3 Style):**
    *   [ ] Create a modal (`#sourceInputModal`) for pasting source code:
        *   Include `<textarea id="sourceCodeInputArea">`.
        *   Include "Submit" and "Cancel" buttons.
        *   Include a hidden input or data attribute to store the target filename.
    *   [ ] Create a modal (`#codeViewModal`) for displaying code snippets:
        *   Include `<pre><code id="codeSnippetDisplay" class="language-cpp"></code></pre>`.
        *   Include a "Close" button.
        *   Include a title area to display filename and line range.

## III. JavaScript Logic

1.  **Helper Functions:**
    *   [x] `async calculateChecksum(string)`: Use `crypto.subtle.digest('SHA-256', ...)` to generate SHA-256 hash. Convert buffer to hex string.
    *   [x] `saveSourceCode(filename, code)`: Calculates checksum, saves `{code, checksum}` to `localStorage['sourceFiles'][filename]`. **Crucially, iterates `localStorage['verificationStatuses']` and sets `verified = false`, `verifiedChecksum = null` for any test linked to this `filename` if the new checksum doesn't match the previous one (if any was stored).** Updates the UI (checkboxes, suite highlighting) for the affected file. Handles potential `localStorage` errors (quota exceeded) with `alert()` and `console.error()`.
    *   [x] `getSourceCode(filename)`: Retrieves `{code, checksum}` from `localStorage['sourceFiles'][filename]`. Handles errors if not found.
    *   [x] `saveVerificationStatus(testId, filename, isVerified)`: Gets current checksum for `filename` via `getSourceCode`. Saves `{ verified: isVerified, verifiedChecksum: (isVerified ? currentChecksum : null) }` to `localStorage['verificationStatuses'][testId]`. Updates suite highlighting if needed. Handles errors.
    *   [x] `getVerificationStatus(testId, filename)`: Retrieves status object for `testId`. Gets current source info via `getSourceCode(filename)`. Returns `true` only if `status.verified === true` AND `status.verifiedChecksum === currentSource.checksum` AND `currentSource` exists. Returns `false` otherwise. Handles errors.
    *   [x] `displayCodeSnippet(filename, code, startLine, endLine)`:
        *   Shows `#codeViewModal`.
        *   Sets modal title.
        *   Sets the content of `#codeSnippetDisplay` to `code`.
        *   Calls `hljs.highlightElement(document.getElementById('codeSnippetDisplay'))`.
        *   Calls `hljs.lineNumbersBlock(document.getElementById('codeSnippetDisplay'), { startFrom: 1 })`.
        *   Scrolls the modal's `<pre>` block so `startLine` is near the top.
        *   Adds temporary visual highlighting (e.g., background color change) to lines from `startLine` to `endLine` within the `<pre>` block. (This might require manipulating the DOM generated by `highlightjs-line-numbers.js`).
    *   [x] `displaySourceInputModal(filename)`: Shows `#sourceInputModal`, sets its target filename, pre-populates `<textarea>` if code exists in `localStorage`.
    *   [x] `updateSuiteHighlight(filename)`: Checks if all checkboxes within the suite corresponding to `filename` are checked (using `getVerificationStatus`). Adds/removes a "verified-suite" class to the `