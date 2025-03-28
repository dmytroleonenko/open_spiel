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





# Optimization Tasks

## Code Simplification
1. **Refactor LegalActions Function**
   - **What:** Break down the complex `LegalActions` function into smaller, focused helper functions.
   - **Where:** `long_narde.cc` (lines 1770-1892)
   - **Why:** Improve readability and maintainability by separating move generation, filtering, and higher-die rule logic.
   - **Tasks:**
     - Create helper for move sequence filtering
     - Create helper for higher-die rule application
     - Create helper for doubles move handling

2. **Simplify RecLegalMoves Function**
   - **What:** Convert recursive `RecLegalMoves` function to iterative approach using stack/queue
   - **Where:** `long_narde.cc` (lines 1598-1681)
   - **Why:** Reduce complexity and potential stack overflow risks
   - **Tasks:**
     - Design iterative algorithm structure
     - Implement stack-based move generation
     - Maintain existing pruning optimizations
     - Add comprehensive tests for equivalence

3. **Clarify Action Encoding/Decoding**
   - **What:** Refactor encoding logic with helper functions and clear documentation
   - **Where:** `long_narde.cc` (lines 157-266 for encoding, 268-323 for decoding)
   - **Why:** Make complex encoding logic more maintainable
   - **Tasks:**
     - Add detailed encoding scheme documentation
     - Create helpers for single move encoding/decoding
     - Create helpers for doubles move handling
     - Add validation checks for encoding ranges

## Code Structure
4. **Group Related Functions**
   - **What:** Organize related functions into logical sections
   - **Where:** `long_narde.cc`
   - **Why:** Improve code navigation and maintainability
   - **Tasks:**
     - Group validation functions
     - Group move generation functions
     - Group encoding/decoding functions
     - Add section comments

5. **Consolidate Constants**
   - **What:** Reorganize constants between header and implementation
   - **Where:** `long_narde.h` and `long_narde.cc`
   - **Why:** Better organize game rules vs implementation details
   - **Tasks:**
     - Move game rules to header
     - Move encoding constants to implementation
     - Document constant purposes
     - Update any affected code

## Comments and Documentation
6. **Enhance Function Comments**
   - **What:** Add detailed comments for complex functions
   - **Where:** Key functions in `long_narde.cc`
   - **Why:** Improve code understanding
   - **Tasks:**
     - Document `RecLegalMoves` logic
     - Document `IsValidCheckerMove` checks
     - Document bridge rule implementation
     - Add parameter/return documentation

7. **Document Encoding Logic**
   - **What:** Add comprehensive encoding documentation
   - **Where:** `long_narde.cc` (lines 157-266, 268-323)
   - **Why:** Clarify complex encoding schemes
   - **Tasks:**
     - Document normal move encoding
     - Document doubles move encoding
     - Document pass move handling
     - Add encoding range explanations

## Performance Optimization
8. **Optimize LegalActions Cloning**
   - **What:** Reduce state cloning overhead in `LegalActions`
   - **Where:** `long_narde.cc` (lines 1773-1774)
   - **Why:** Improve performance for frequent calls
   - **Tasks:**
     - Profile current implementation
     - Explore alternatives to full cloning
     - Implement and test improvements
     - Measure performance impact

9. **Improve Move Sequence Storage**
   - **What:** Replace set-based storage with vector and post-processing
   - **Where:** `long_narde.cc` (lines 1600, 1770-1892)
   - **Why:** Reduce overhead from vector comparisons
   - **Tasks:**
     - Implement vector-based storage
     - Add efficient duplicate removal
     - Test performance impact
     - Verify correctness

## Algorithm Improvements
10. **Validate Encoding Schemes**
    - **What:** Test action encoding edge cases
    - **Where:** `long_narde.cc` (lines 157-266, 268-323)
    - **Why:** Ensure robust encoding handling
    - **Tasks:**
      - Test normal moves encoding
      - Test doubles moves encoding
      - Test pass moves encoding
      - Verify no encoding collisions

11. **Enhance Bridge Rule Checks**
    - **What:** Improve bridge rule testing
    - **Where:** `long_narde.cc` (lines 562-601, 1900-1968)
    - **Why:** Verify bridge rule correctness
    - **Tasks:**
      - Test wrap-around cases
      - Test different player paths
      - Test partial bridge formation
      - Test bridge prevention

## Testing
12. **Add Comprehensive Encoding Tests**
    - **What:** Create encoding/decoding test suite
    - **Where:** New test file or `long_narde_test.cc`
    - **Why:** Verify encoding correctness
    - **Tasks:**
      - Test normal moves
      - Test doubles moves
      - Test pass moves
      - Test edge cases

13. **Expand Movement Direction Tests**
    - **What:** Add movement direction test cases
    - **Where:** `long_narde_test_movement.cc`
    - **Why:** Improve movement logic coverage
    - **Tasks:**
      - Test various dice combinations
      - Test different positions
      - Test bearing off moves
      - Test direction constraints

14. **Implement Missing Test Cases**
    - **What:** Complete test coverage
    - **Where:** Test files
    - **Why:** Ensure all rules are tested
    - **Tasks:**
      - Add first turn doubles tests
      - Add bridge rule edge cases
      - Add scoring edge cases
      - Add comprehensive move validation

## Progress Tracking
- [ ] Code Simplification (Tasks 1-3)
- [ ] Code Structure (Tasks 4-5)
- [ ] Documentation (Tasks 6-7)
- [ ] Performance (Tasks 8-9)
- [ ] Algorithms (Tasks 10-11)
- [ ] Testing (Tasks 12-14) 