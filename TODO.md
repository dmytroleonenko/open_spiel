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
- [ ] Fix build environment issues to successfully run tests
- [ ] Consider using a more compatible Python version for the build process

## Notes
- Follow TDD: Create or update tests that specify the desired behavior before modifying game logic.
- Ensure tests cover all edge cases of the Long Narde rules. 
- The implementation is complete, but we're facing build environment issues that need to be resolved to run the tests. 