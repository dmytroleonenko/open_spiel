# Long Narde Implementation

## Overview
This implementation creates a new game "Long Narde" based on Backgammon but with distinct rules. Long Narde is a tables game where both players move their checkers counter-clockwise with the goal of bearing them off first.

## Key Rule Differences
1. **Initial Setup**: All 15 white checkers start on point 24; all 15 black checkers start on point 12
2. **Movement Direction**: Both players move counter-clockwise (unlike Backgammon's opposing directions)
3. **No Hitting**: Players cannot land on opponent's checkers
4. **Head Rule**: Only 1 checker can leave the head per turn (with exceptions for specific doubles on first turn)
5. **Blocking Restriction**: Cannot form a continuous 6-point prime that would fully trap opponent
6. **Bearing Off**: Requires exact or higher rolls once all checkers are in home
7. **Scoring**: 2 points for mars (opponent has no checkers borne off), 1 point for oin (normal win)
8. **Last Roll Tie Rule**: If one player has borne off all checkers but the opponent has at least 14 checkers borne off, the opponent gets one last roll to try to achieve a tie

## Implementation Details
The implementation follows a test-driven development approach with comprehensive tests for each rule modification. Key changes include:

1. Created dedicated header and implementation files in `games/long_narde/`
2. Added necessary constants for head positions
3. Modified movement logic to enforce counter-clockwise movement
4. Implemented head rule restrictions
5. Added bridge/blocking rule enforcement
6. Updated home regions and scoring logic
7. Wrote comprehensive tests verifying all rules
8. Implemented the last roll tie mechanism that allows a near-complete opponent one final chance

## Scoring Options
The implementation supports multiple scoring types:
1. **winloss_scoring**: Simple +1/-1 for wins/losses
2. **enable_gammons**: +2/-2 for mars (when opponent has no checkers off), +1/-1 for regular wins
3. **full_scoring**: Same as enable_gammons but includes the last roll tie option

## Build Integration
The Long Narde game has been added to the CMakeLists.txt with appropriate build rules. Tests are available for verifying the implementation.

## Future Improvements
1. Fix build environment issues to successfully run all tests
2. Consider adding a specific Long Narde GUI or visualization
3. Implement AI strategies specific to Long Narde's unique rules 