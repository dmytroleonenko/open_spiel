# Long Narde Implementation

## Overview
This implementation creates a new game "Long Narde" based on Backgammon but with distinct rules. Long Narde is a tables game where both players move their checkers counter-clockwise with the goal of bearing them off first.

## Key Rule Differences
1. **Setup**: White's 15 checkers on point 24; Black's 15 checkers on point 12.
2. **Movement**: Both players move counter-clockwise (CCW) into their home boards (White: points 1–6, Black: points 13–18), and then bear off.
3. **Turns & Dice Usage**:
    * Players roll two dice. Moves must correspond exactly to the die values.
    * No landing on points occupied by an opponent's checker.
    * If no moves are possible, the turn is skipped.
    * If only one die's value can be played, the higher value must be used.
    * Doubles grant four moves of the die's value.
4. **Head Rule**: Only one checker may be moved from a player's head (starting point: White 24, Black 12) per turn.
    * First Turn Exception: If a player's first roll of the game is a double 3-3, 4-4, or 6-6, they may move two checkers from their head. For that first turn, after these two checkers are moved, no further checkers can be moved from the head.
5. **Bearing Off**: Once all 15 of a player's checkers are in their home board:
    * A die roll of 'n' allows a checker to be borne off from point 'n', even if higher points are occupied.
    * If point 'n' is empty, a checker must be moved from a higher-numbered point using the die value 'n', if possible.
    * If no such move is possible, a checker must be borne off from the highest-numbered point currently occupied by the player.
6. **Blocking (Bridge) Restriction**: A player cannot form a contiguous block of six checkers (a 6-prime) unless at least one of the opponent's checkers is ahead of (further along in their path than) the potential block. Fully trapping all 15 opponent checkers is disallowed.
7. **Ending & Scoring**: The game ends when one player successfully bears off all their checkers.
    * If the loser has borne off no checkers, the winner scores 2 points (a "mars").
    * Otherwise, the winner scores 1 point (an "oin").
8. **Last Roll Tie Rule**: Specific game situations (e.g., if the opponent has 14 checkers borne off when the winner finishes) may grant the opponent one last roll to attempt a tie.

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