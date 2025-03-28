# TODO: Long Narde Implementation Plan

## Overview
We will create a copy of "games/backgammon" and modify it to implement the game rules of Long Narde, an ultra-short variant with the following rules:

1. Setup: White's 15 checkers on point 24; Black's 15 on point 12.
2. Movement: Both move checkers counter-clockwise into home (White 1–6, Black 13–18), then bear off.
3. Starting: Each rolls 1 die; higher is White and goes first.
4. Turns: Roll 2 dice, move checkers exactly by each value. No landing on opponent; if no moves exist, skip; if only one is possible, use the higher die.
5. Head Rule: Only 1 checker may leave the head per turn, except on the first turn if a double 6, 4, or 3 is rolled, allowing 2 moves from the head. Afterwards, no extra head moves are allowed.
6. Bearing Off: Once all checkers reach home, bear them off with exact or higher rolls.
7. Ending/Scoring: Game ends when a player bears off all checkers. If the loser has borne off none, the winner scores 2 (mars); otherwise, 1 (oin). Some events allow a last roll to tie.
8. Block (Bridge) Rule: A contiguous block of 6 checkers cannot be formed unless at least 1 opponent checker is ahead. Fully trapping all 15 opponent checkers is banned.
9. Last Roll Tie: If one player has borne off all checkers but the opponent has at least 14 checkers off, they get one last roll to potentially achieve a tie.

## Action Items
- [*] Create a new folder "games/long_narde" as a copy of "games/backgammon".
- [*] Adjust initial checker positions to reflect the new setup (White on point 24, Black on point 12).
- [*] Update movement logic: enforce counter-clockwise movement into designated home areas and bearing off.
- [*] Modify starting turn logic: each player rolls 1 die, with the higher roll making White go first.
- [*] Update turn logic: implement two-dice rolls, moving checkers exactly by each die value, enforcement of using the higher die if only one move is available, and skipping turn if no moves exist.
- [*] Implement the head rule: limit head moves to 1 per turn, with an exception on the first turn when a double 6, 4, or 3 permits moving 2 checkers from the head.
- [*] Adjust bearing off rules to require exact or higher die values once all checkers are in the home area.
- [*] Implement game-ending logic and scoring, including conditions for mars and oin, and a possible tie on a last roll.
- [*] Enforce the block rule: prevent forming a contiguous block of 6 checkers that fully traps the opponent.
- [*] Write comprehensive test cases for each rule modification before altering implementation (TDD approach).
- [*] Update documentation to reflect all changes.
- [*] Add the Long Narde game to the CMakeLists.txt
- [*] Implement last roll tie rule to allow a final chance for a player who has at least 14 checkers off
- [ ] Successfully build and run all tests to ensure the new rules are correctly implemented.
- [ ] Commit changes following TDD principles.

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
     - Verified that partial double usage (one die only) doesn't grant an extra turn
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

## Code Simplification
	1.	Refactor LegalActions Function
	•	What: Break down the complex LegalActions function into smaller, focused helper functions.
	•	Where: long_narde.cc (lines 1770–1892)
	•	Why: Improve readability and maintainability by separating move generation, filtering, and higher-die rule logic.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorLegalActions`
	•	Tasks:
	•	[ ] Create a helper for move sequence generation (e.g., GenerateMoveSequences).
	•	[ ] Create a helper for move sequence filtering (e.g., FilterLongestSequences).
	•	[ ] Create a helper for higher-die rule application (e.g., ApplyHigherDieRuleIfNeeded).
	•	[ ] Create a helper for doubles move handling.
	•	[ ] Use early returns to avoid deeply nested conditions.
	2.	Simplify RecLegalMoves Function
	•	What: Convert the recursive RecLegalMoves function to an iterative approach using a stack or queue.
	•	Where: long_narde.cc (lines 1598–1681)
	•	Why: Reduce complexity and potential stack overflow risks.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorRecLegalMoves`
	•	Tasks:
	•	[ ] Design an iterative algorithm structure (BFS/DFS style) to generate half-move sequences.
	•	[ ] Implement stack-based (or queue-based) move generation that mimics the current recursive behavior.
	•	[ ] Maintain existing pruning optimizations (e.g., bridging checks and head-rule validations) during iteration.
	•	[ ] Add comprehensive tests to verify equivalence with the recursive approach.
	3.	Clarify Action Encoding/Decoding
	•	What: Refactor encoding logic with helper functions and clear documentation.
	•	Where: long_narde.cc (lines 157–266 for encoding, 268–323 for decoding)
	•	Why: Make the complex encoding logic more maintainable.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorEncodingDecoding`
	•	Tasks:
	•	[ ] Add detailed documentation of the encoding scheme (normal moves, doubles, pass moves, offsets).
	•	[ ] Create helpers such as EncodeSingleMove, DecodeSingleMove, EncodeDoubles, and DecodeDoubles.
	•	[ ] Create additional helpers for pass move handling.
	•	[ ] Add validation checks for encoding ranges to ensure no collisions.
	3a. Simplify IsFirstTurn Access
	•   What: Remove the redundant `is_first_turn()` method, keeping only `IsFirstTurn(Player player)`.
	•   Where: `long_narde.h`, `long_narde.cc`, and any calling code (tests).
	•   Why: Improve API clarity and consistency. `IsFirstTurn(player)` is less ambiguous.
	•	Retrieval Hint: Use query `Task:LongNardeRefactorIsFirstTurn`
	•   Tasks:
	    *   [x] Remove the `is_first_turn()` declaration and definition.
	    *   [x] Update all call sites (identified during refactoring in tests) to use `IsFirstTurn(player)` instead.
	    *   [x] Verify tests still pass.

## Code Structure
	4.	Group Related Functions
	•	What: Organize related functions into logical sections.
	•	Where: long_narde.cc
	•	Why: Improve code navigation and maintainability.
	•	Tasks:
	•	[x] Group validation functions together.
	•	[x] Group move generation functions (including the new iterative version of move sequence generation).
	•	[x] Group encoding/decoding functions with their respective constants.
	•	[x] Add clear section comments (e.g., // ===== Move Generation =====, // ===== Encoding/Decoding =====).
	5.	Consolidate Constants
	•	What: Reorganize constants between header and implementation.
	•	Where: long_narde.h and long_narde.cc
	•	Why: Better organize game rules versus implementation details.
	•	Tasks:
	•	[x] Move game rule constants (e.g., board size, home regions, head positions) to the header with clear documentation.
	•	[x] Move encoding constants (e.g., kDigitBase, kPassOffset, kDoublesOffset) to the implementation file.
	•	[x] Document the purpose of each constant.
	•	[x] Update any code references affected by the move.

## Comments and Documentation
	6.	Enhance Function Comments
	•	What: Add detailed comments for complex functions.
	•	Where: Key functions in long_narde.cc
	•	Why: Improve code understanding.
	•	Tasks:
	•	[ ] Document the logic of the (now iterative) move sequence generation function.
	•	[ ] Document the checks in IsValidCheckerMove (bounds, head rule, bearing off, opponent occupancy, bridging).
	•	[ ] Document the bridge rule implementation and the rationale behind it.
	•	[ ] Add parameter and return documentation using a consistent style (e.g., Doxygen).
	7.	Document Encoding Logic
	•	What: Add comprehensive documentation for the encoding scheme.
	•	Where: long_narde.cc (lines 157–266, 268–323)
	•	Why: Clarify the complex encoding schemes for normal moves, doubles, and pass moves.
	•	Tasks:
	•	[*] Document normal move encoding (e.g., pos * 6 + (die - 1)).
	•	[*] Document doubles move encoding (including use of an offset).
	•	[*] Document pass move handling (using kPassOffset + (die - 1)).
	•	[*] Add explanations for the encoding ranges and any potential edge cases.

## Performance Optimization
	8.	Optimize LegalActions Cloning
	•	What: Reduce state cloning overhead in LegalActions.
	•	Where: long_narde.cc (lines 1773–1774)
	•	Why: Improve performance for frequent calls.
	•	Retrieval Hint: Use query `Task:LongNardePerfOptCloning`
	•	Tasks:
	•	[ ] Profile the current implementation of state cloning.
	•	[ ] Explore alternatives to full cloning (e.g., shallow copy of necessary fields or in-place apply/undo).
	•	[ ] Implement and test the improved approach.
	•	[ ] Measure and document performance impact.
	9.	Improve Move Sequence Storage
	•	What: Replace set-based storage with a vector and post-processing.
	•	Where: long_narde.cc (lines 1600, 1770–1892)
	•	Why: Reduce overhead from vector comparisons when storing move sequences.
	•	Retrieval Hint: Use query `Task:LongNardePerfOptMoveStorage`
	•	Tasks:
	•	[ ] Implement move sequence storage using a std::vector of sequences.
	•	[ ] Use sorting and std::unique for duplicate removal after sequence generation.
	•	[ ] Test performance impact and verify correctness.

## Algorithm Improvements
	10.	Validate Encoding Schemes
	•	What: Thoroughly test action encoding edge cases.
	•	Where: long_narde.cc (lines 157–266, 268–323)
	•	Why: Ensure robust handling of normal, doubles, and pass move encoding.
	•	Retrieval Hint: Use query `Task:LongNardeAlgoValidateEncoding`
	•	Tasks:
	•	[ ] Create tests for normal moves encoding.
	•	[ ] Create tests for doubles moves encoding.
	•	[ ] Create tests for pass moves encoding.
	•	[ ] Verify that encoding followed by decoding returns the original moves and that no collisions occur.
	11.	Enhance Bridge Rule Checks
	•	What: Improve and verify bridge rule testing.
	•	Where: long_narde.cc (lines 562–601, 1900–1968)
	•	Why: Ensure bridge rule correctness, especially for wrap-around cases and different player paths.
	•	Retrieval Hint: Use query `Task:LongNardeAlgoBridgeChecks`
	•	Tasks:
	•	[ ] Test wrap-around cases (e.g., bridging from point 23 to 0).
	•	[ ] Test bridging on different player paths (white vs. black).
	•	[ ] Test cases with partial bridge formation.
	•	[ ] Confirm that the bridge rule correctly prevents a full trap.

## Testing
	12.	Add Comprehensive Encoding Tests
	•	What: Create an encoding/decoding test suite.
	•	Where: New test file or long_narde_test.cc
	•	Why: Verify that all encoding cases (normal, doubles, pass) work correctly.
	•	Tasks:
	•	[*] Test encoding/decoding for normal moves.
	•	[*] Test encoding/decoding for doubles moves.
	•	[*] Test encoding/decoding for pass moves.
	•	[*] Test edge cases and validate that decode(encode(…)) equals the original.
	13.	Expand Movement Direction Tests
	•	What: Add test cases for movement directions.
	•	Where: long_narde_test_movement.cc
	•	Why: Improve coverage of movement logic, including bearing off.
	•	Tasks:
	•	[ ] Test various dice combinations.
	•	[ ] Test moves from different board positions.
	•	[ ] Test bearing off moves.
	•	[ ] Test constraints on movement direction (especially for black's wrap-around).
	14.	Implement Missing Test Cases
	•	What: Complete test coverage for all rules.
	•	Where: Test files.
	•	Why: Ensure full rule coverage and robustness.
	•	Tasks:
	•	[*] Implement missing `FirstTurnDoublesExceptionTest` (Covered by HeadRuleTest)
	•	[*] Ensure all original `BlockingBridgeRuleTest` cases are covered (Covered by TestBridgeFormation in long_narde_test_bridges.cc)
	•	[ ] Add tests for scoring edge cases (e.g., mars vs. oin scoring, tie scenarios).
	•	[*] Add comprehensive move validation tests. (Covered by various existing tests: NoLanding, HeadRule, Bridge, Movement, BearingOff, SingleLegal, Pass, HigherDie)

## Testing Refactoring
15. Refactor Tests to Reduce Reliance on Internal State/Constants
    *   What: Modify test files (starting with `long_narde_test_actions.cc`) to avoid direct state manipulation and usage of internal encoding constants.
    *   Where: `long_narde_test_*.cc`, `long_narde.h`, `long_narde.cc`, `long_narde_test_common.h`
    *   Why: Improve test robustness, maintainability, and encapsulation. Allows internal constants (`kDigitBase`, `kPassOffset`, etc.) to be properly kept internal to `long_narde.cc` (related to Task 5).
    *   Tasks:
        *   [x] Analyze all test files (`long_narde_test_*.cc`) for direct use of `SetState`, `MutableIsFirstTurn`, internal constants (`kDigitBase`, `kEncodingBaseDouble`, `kDoublesOffset`), or direct member access. 
            *   [x] `long_narde_test_actions.cc` (Refactored)
            *   [x] `long_narde_test_basic.cc` (Reviewed - OK)
            *   [x] `long_narde_test_movement.cc` (Refactored)
            *   [x] `long_narde_test_bridges.cc` (Reviewed - OK)
            *   [x] `long_narde_test_endgame.cc` (Refactored)
            *   [x] `long_narde_test_legacy.cc`
            *   [x] `bearing_off_test.cc` (Refactored)
            *   [x] `random_sim_test.cc` (Reviewed - OK)
        *   [x] Create public helper methods in `LongNardeState` or test utilities (`long_narde_test_common.h`) for setting up board states and dice in a controlled way (e.g., `SetupBoard(config)`, `SetDice(dice)`), replacing direct `SetState` calls where appropriate. (Done: `SetupBoardState`, `SetupDice` created and used)
        *   [x] Modify tests currently using `MutableIsFirstTurn` to use the new setup helpers or structure tests to naturally progress past the first turn via applying actions. (Done for `_movement.cc`, others TBD)
        *   [x] Refactor tests (`ActionEncodingTest` initially) that check specific action ID ranges based on internal constants (`kDigitBase`, `kDoublesOffset`). Instead, verify the *behavior* or *properties* of the decoded moves corresponding to those actions (e.g., number of moves, die values used, pass moves present). (`ActionEncodingTest` in `_actions.cc` done, check others)
        *   [x] Once tests are refactored, move `kDigitBase` back to `long_narde.cc`'s anonymous namespace. Ensure `kPassOffset`, `kEncodingBaseDouble`, `kDoublesOffset` remain internal.
        *   [x] Update Task 5 status in `TODO.md` to 'incomplete' or remove its completed checkmark until this refactoring allows its goal to be met.

⸻

## Progress Tracking
	•	[ ] Code Simplification (Tasks 1–3, 3a)
	•	[x] Code Structure (Tasks 4–5)
	•	[*] Documentation (Tasks 6–7)
	•	[ ] Performance (Tasks 8–9)
	•	[ ] Algorithms (Tasks 10–11)
	•	[*] Testing (Tasks 12–14)
	•	[x] Testing Refactoring (Task 15)

⸻

## Summary of Key Recommendations
	•	Break large functions into smaller helpers for move generation, filtering, and encoding/decoding.
	•	Adopt an iterative approach for move sequence generation to simplify recursion and allow early pruning.
	•	Document thoroughly—especially the complex encoding schemes and bridging logic—to ease future maintenance.
	•	Group related logic (validation, move generation, encoding) with clear section comments to improve code navigation.
	•	Reduce cloning overhead and improve move sequence storage using vectors and post-processing instead of sets.
	•	Expand and automate tests for encoding, movement, and bridging rules to ensure robust functionality.

By addressing these tasks, the Long Narde codebase will become simpler, better organized, more performant, and easier to maintain or extend in the future. 