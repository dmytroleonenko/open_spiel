# Long Narde Bugs and Investigation

## Issue: Infinite Games in Random Simulation

During random simulation testing, we've observed games that reach the maximum move limit (1000 moves) without terminating. This is concerning because:

1. Long Narde rules dictate that a player must make a move if one is available
2. Players should eventually bear off all their checkers, ending the game
3. The probability of both players being unable to move for extended periods is extremely low

## Debug Investigation (2023-03-24)

Using GDB to analyze a game that exceeded 900 moves, we found:

```
*** Long game detected at move 902 (simulation 5) ***
Current player: 0 (White)
Legal actions count: 1
Current board state:
Current player: x
Dice: 1 3
Scores - White: 8, Black: 0
Board:
White (X): 1:7
Black (O): 4:3 5:1 6:5 9:2 11:1 16:1 17:1 20:1
```

### Key Observations:

1. White player has 7 checkers all at position 1 (home board)
2. White has already borne off 8 checkers (score: 8)
3. White's dice roll is 1 and 3
4. White has only one legal action (624), which is the "pass" action code
5. This indicates the player cannot use either dice value, despite having checkers at position 1

### Suspected Issue:

The bearing off logic appears to be faulty. In Long Narde, a player can bear off when:
- All their remaining checkers are in their home board (positions 1-6 for White)
- The die value exactly matches the checker position OR is larger than the highest position containing checkers

In this case, White should be able to:
- Use the 1 die to bear off directly from position 1
- Use the 3 die to bear off from position 1 (since there are no checkers in positions 2+)

## Investigation Plan:

1. Locate the bearing off logic in the codebase
2. Understand how the rule is implemented
3. Create a test case that reproduces this exact scenario
4. Fix the bearing off logic to handle this case correctly
5. Verify the fix with extensive random simulation testing

This issue is critical because it can lead to games that never terminate naturally, which is contrary to the game rules. 