# Long Narde Test Suite Documentation

This document describes the test cases implemented in the Long Narde game test files located in `open_spiel/games/long_narde/`.

## Test Suite: `long_narde_test_movement.cc`

### Function: `TestBasicMovement`

**Test: Basic Movement**

*   **Objective:** Verify that basic checker movement works correctly for White player using SetupDice.
*   **Board Setup:** Initial state with White having 15 checkers at head (point 24, index 23).
*   **Dice Roll:** SetupDice({4, 4, 4, 4})
*   **Expectations:** All four half-moves are executed, and the turn switches to kChancePlayerId. White should have 13 checkers at the head and 2 at point 20.
*   **File/Function Ref:** Test: test-basicmovement-1
*   **Rules Involved:** Basic checker movement, head rule.

### Function: `InitialDiceTest`

**Test: Initial Dice Values**

*   **Objective:** Verify that the chance outcomes produce valid dice pairs in [1..6] with the highest die first (unless doubles) using SetupDice uniformly.
*   **Board Setup:** Initial state (chance node).
*   **Process:** Checks all 21 chance outcomes (dice combinations) to ensure they follow the rules, using SetupDice for dice setup.
*   **Expectations:** All dice values should be in range [1..6]. For non-doubles, the first die should be greater than or equal to the second die.
*   **File/Function Ref:** Test: test-initialdice-1
*   **Rules Involved:** Dice roll rules, chance node outcomes.

### Function: `CheckerDistributionTest`

**Test: Checker Distribution**

*   **Objective:** Verify that checkers are correctly distributed after moves.
*   **Board Setup:** Initial state with White having 15 checkers at head (point 24, index 23).
*   **Dice Roll:** {4, 2}
*   **Expectations:** After moving, White should have 13 checkers at the head, 1 at point 20 (index 19), and 1 at point 22 (index 21).
*   **File/Function Ref:** Test: test-checkerdistribution-1
*   **Rules Involved:** Checker movement, board state tracking.

### Function: `FirstTurnTest`

**Test: First Turn Logic**

*   **Objective:** Verify the special rules for the first turn, including the ability to move multiple checkers from the head on the first turn with specific dice rolls.
*   **Board Setup:** Initial state.
*   **Dice Roll:** Various, including {6, 6}, {4, 4}, and {3, 3}.
*   **Expectations:** On the first turn, if doubles of 6, 4, or 3 are rolled, the player should be able to move 2 checkers from the head. For other dice combinations, only 1 checker can leave the head.
*   **File/Function Ref:** Test: test-firstturn-1
*   **Rules Involved:** First turn special rules, head rule, doubles play.

### Function: `HeadRuleTest`

**Test: Head Rule for White**

*   **Objective:** Verify that the head rule is enforced correctly for White player (only one checker can leave the head per turn, except on first turn with specific doubles).
*   **Board Setup:** White has 15 checkers at head (point 24, index 23).
*   **Dice Roll:** {5, 3}
*   **Expectations:** White should only be able to move one checker from the head, even though both dice could be used to move from the head.
*   **File/Function Ref:** Test: test-headrule-1
*   **Rules Involved:** Head rule, checker movement restrictions.

### Function: `MovementDirectionTest`

**Test: Movement Direction**

*   **Objective:** Verify that White moves towards decreasing indices (clockwise) and Black effectively moves counter-clockwise with wrapping.
*   **Board Setup:** Various test positions for both White and Black.
*   **Dice Roll:** {3, 2}
*   **Expectations:** White's moves should always decrease the position index. Black's moves should follow the counter-clockwise direction, which may involve wrapping from index 0 to 23.
*   **File/Function Ref:** Test: test-movementdirection-1
*   **Rules Involved:** Movement direction rules, player-specific movement patterns.

### Function: `NoLandingOnOpponentTest`

**Test: No Landing on Opponent**

*   **Objective:** Ensure that if the opponent has a checker on some point, you cannot move onto it.
*   **Board Setup:** White has checkers at various points, Black has a checker at point 16 (index 15).
*   **Dice Roll:** {4, 2}
*   **Expectations:** White should not be able to land on point 16 where Black has a checker. No legal actions should include a move that lands on an opponent's checker.
*   **File/Function Ref:** Test: test-nolanding-1
*   **Rules Involved:** Landing restrictions, opponent blocking.

### Function: `TestIllegalLandingInLegalActions`

**Test: Illegal Landing in Legal Actions**

*   **Objective:** Verify that `LegalActions` does not generate moves landing on occupied points.
*   **Board Setup:** Custom board with specific checker placements to test landing restrictions.
*   **Dice Roll:** {4, 2}
*   **Expectations:** No legal actions should include moves that land on points occupied by opponent checkers.
*   **File/Function Ref:** Test: test-illegallanding-1
*   **Rules Involved:** Legal action generation, landing restrictions.

### Function: `TestHalfMoveGeneration`

**Test: Half-Move Generation for White**

*   **Objective:** Test that half-move generation produces correct moves for White player.
*   **Board Setup:** White has 1 checker at point 1 (index 0) and 1 at point 24 (index 23). Black has 1 checker at point 12 (index 11).
*   **Dice Roll:** {3, 5}
*   **Expectations:** White should be able to move from point 1 to off-board with die 3 and from point 24 to point 19 with die 5.
*   **File/Function Ref:** Test: test-halfmove-1
*   **Rules Involved:** Half-move generation, bearing off rules, basic movement.

### Function: `TestHalfMoveGenerationBlack`

**Test: Half-Move Generation for Black**

*   **Objective:** Test that half-move generation produces correct moves for Black player.
*   **Board Setup:** Black has 1 checker at point 13 (index 12) and 1 at point 12 (index 11). White has 1 checker at point 24 (index 23).
*   **Dice Roll:** {3, 5}
*   **Expectations:** Black should be able to move from point 13 to point 16 with die 3 and from point 12 to point 17 with die 5.
*   **File/Function Ref:** Test: test-halfmoveblack-1
*   **Rules Involved:** Half-move generation for Black, basic movement.

### Function: `HeadRuleTestBlack`

**Test: Head Rule for Black**

*   **Objective:** Verify that the head rule is enforced correctly for Black player.
*   **Board Setup:** Black has 15 checkers at head (point 12, index 11).
*   **Dice Roll:** {5, 3}
*   **Expectations:** Black should only be able to move one checker from the head, even though both dice could be used to move from the head.
*   **File/Function Ref:** Test: test-headruleblack-1
*   **Rules Involved:** Head rule for Black, checker movement restrictions.

### Function: `TestHalfMoveGenerationBlack`

**Test: Half-Move Generation for Black**

*   **Objective:** Test that half-move generation produces correct moves for Black player.
*   **Board Setup:** Black has 1 checker at point 13 (index 12) and 1 at point 12 (index 11). White has 1 checker at point 24 (index 23).
*   **Dice Roll:** {3, 5}
*   **Expectations:** Black should be able to move from point 13 to point 16 with die 3 and from point 12 to point 17 with die 5.
*   **File/Function Ref:** Test: test-halfmoveblack-1
*   **Rules Involved:** Half-move generation for Black, basic movement.

### Function: `TestMovementRules`

**Test: All Movement Rules**

*   **Objective:** Run all movement tests to verify all movement rules.
*   **Board Setup:** Various (calls all individual test functions)
*   **Dice Roll:** Various (depends on individual tests)
*   **Expectations:** All individual movement tests should pass, confirming that all movement rules are correctly implemented.
*   **File/Function Ref:** Test: test-movementrules-1
*   **Rules Involved:** All movement rules (comprehensive test).

## Test Suite: `long_narde_test_actions.cc`

### Function: `ActionEncodingTest`

**Test 1: Regular Move Encoding (High Roll First)**

*   **Objective:** Verify correct encoding and decoding of a regular two-checker move when the dice roll has the higher die first.
*   **Board Setup:** White: 13 checkers at head (point 24), 1 at point 15, 1 at point 20 (Total 15, Score 0). Black: 15 checkers at head (point 12) (Total 15, Score 0). White to move. (Setup on L58)
*   **Dice Roll:** {5, 3} (Setup on L59)
*   **Expectations:** The specific move sequence `{pos=14, to=19, die=5}, {pos=19, to=22, die=3}` is encoded to a valid Spiel `Action`. Decoding this `Action` recovers both original `CheckerMove`s (verified by checking `pos` and `die` for both moves).
*   **Rules Involved:** Action encoding/decoding, basic checker movement.

**Test 2: Pass Move Encoding**

*   **Objective:** Verify correct encoding and decoding of a "pass" move (passing both available dice).
*   **Board Setup:** Implicitly reuses state from Test 1 (Setup on L58-L59): White: 13 checkers at head (point 24), 1 at point 15, 1 at point 20 (Total 15, Score 0). Black: 15 checkers at head (point 12) (Total 15, Score 0). White to move. (*Note: No explicit `SetupBoardState` call for this test*).
*   **Dice Roll:** Implicitly {5, 3} (Setup on L59).
*   **Expectations:** A pass move sequence using `kPassPos` for both checkers `{kPassPos, kPassPos, 5}, {kPassPos, kPassPos, 3}` can be encoded to a Spiel `Action`. Decoding this `Action` recovers two pass `CheckerMove`s, identified by checking `pos == kPassPos` and `die` values 5 and 3.
*   **Rules Involved:** Action encoding/decoding, pass move representation (`kPassPos`).

**Test 3: Regular Move Encoding (Low Roll First)**

*   **Objective:** Verify that encoding the same two-checker move as in Test 1, but with the dice rolled low-first ({3, 5}), results in a *different* Action ID (due to encoding offset) but still decodes back to the correct moves.
*   **Board Setup:** Uses `SetupBoardState` (L133) with the `modified_board` from Test 1. White: 13 checkers at head (point 24), 1 at point 15, 1 at point 20 (Total 15, Score 0). Black: 15 checkers at head (point 12) (Total 15, Score 0). White to move. Score {0, 0} is correct.
*   **Dice Roll:** {3, 5} (Setup on L134).
*   **Expectations:** Encoding the same logical moves as Test 1 (`{pos=14, to=19, die=5}, {pos=19, to=22, die=3}`) results in a different `Action` ID (`action_low_roll`) compared to the ID from Test 1 (`action`). Decoding `action_low_roll` still recovers the original two `CheckerMove`s (verified by checking `pos` and `die`).
*   **Rules Involved:** Action encoding/decoding, encoding offset for low-roll-first scenarios, basic checker movement.

**Test 4: Doubles Encoding (4 moves)**

*   **Objective:** Verify the correct encoding and decoding of a 4-move sequence resulting from rolling doubles (non-first turn).
*   **Board Setup:** Uses `SetupBoardState` (L179). White: 1 checker each at point 21, point 22, point 23, point 24(head), 11 checkers at point 20 (Total 15, Score 0). Black: 15 checkers at head (point 12) (Total 15, Score 0). White to move. Score {0, 0} is correct.
*   **Dice Roll:** {2, 2, 2, 2} (Setup on L180).
*   **Expectations:** A 4-move sequence `{23->21(d2)}, {22->20(d2)}, {21->19(d2)}, {20->18(d2)}` is encoded to a valid Spiel `Action`. Decoding this `Action` recovers the 4 original `CheckerMove`s (verified by checking `pos` and `die == 2` for all 4 moves).
*   **Rules Involved:** Action encoding/decoding for doubles, doubles play (4 moves), head movement rule (non-first turn).

**Test 5: Doubles Encoding (3 moves + 1 pass)**

*   **Objective:** Verify the correct encoding and decoding of a sequence resulting from doubles, where only 3 moves are possible, forcing the 4th "move" to be a pass.
*   **Board Setup:** Implicitly reuses state from Test 4 (Setup on L179-L180): White: 1 checker each at point 21, point 22, point 23, point 24(head), 11 at point 20 (Total 15, Score 0). Black: 15 at head (point 12) (Total 15, Score 0). White to move. Score {0, 0} is correct. (*Note: No explicit `SetupBoardState` call*).
*   **Dice Roll:** Implicitly {2, 2, 2, 2} (Setup on L180).
*   **Expectations:** A sequence of 3 moves `{23->21(d2)}, {22->20(d2)}, {21->19(d2)}` plus one pass `{kPassPos, kPassPos, 2}` can be encoded. Decoding the `Action` should yield moves corresponding to the 3 unique non-pass starting positions (23, 22, 21, verified using `std::map`) and at least one pass move (`passes_found >= 1`). The action ID should differ from the 4-move action in Test 4.
*   **Rules Involved:** Action encoding/decoding for doubles, doubles play (partial usage), pass moves.

**Test 6: Standard Encoding: Single move + Pass**

*   **Objective:** Verify the correct encoding/decoding for a standard (non-double) roll where only one die can be played, forcing a pass with the other die.
*   **Board Setup:** Uses `SetupBoardState` (L280) with `modified_board` (from Test 1). White: 13 checkers at head (point 24), 1 at point 15, 1 at point 20 (Total 15, Score 0). Black: 15 checkers at head (point 12) (Total 15, Score 0). White to move. Score {0, 0} is correct.
*   **Dice Roll:** {4, 1} (Setup on L281).
*   **Expectations:** A sequence involving moving from point 15 with die 4 and passing with die 1 can be encoded. Decoding the `Action` should recover the move from point 15 (die 4) and the pass (die 1). No low-roll offset should be applied.
*   **Rules Involved:** Action encoding/decoding, basic checker movement, partial dice usage (forced pass).

**Test 7: Standard Encoding: Double Pass**

*   **Objective:** Re-verify the correct encoding/decoding of a double pass (passing both dice) with a standard roll, ensuring no low-roll offset is applied.
*   **Board Setup:** Uses `SetupBoardState` (L316) with `modified_board` (from Test 1). White: 13 checkers at head (point 24), 1 at point 15, 1 at point 20 (Total 15, Score 0). Black: 15 checkers at head (point 12) (Total 15, Score 0). White to move. Score {0, 0} is correct.
*   **Dice Roll:** {6, 5} (Setup on L317).
*   **Expectations:** A double pass move (`kPassPos` for both dice) can be encoded. Decoding the `Action` should recover the pass moves corresponding to dice 6 and 5. No low-roll offset should be applied.
*   **Rules Involved:** Action encoding/decoding, pass move representation (`kPassPos`).

## Function: `SingleLegalMoveTest`

**Test: Higher Die Rule Enforcement (White's Direction, Head Blocked)**

*   **Objective:** Verify that when White (moving towards lower points) has only one checker that can move, and that checker can use either die individually but not both sequentially, the game enforces playing the single move corresponding to the higher die.
*   **Board Setup:** Uses `SetupBoardState` (L358). White: 1 checker at Point 6 (index 5), 14 checkers at Point 24 (head, index 23). Black: 2 checkers at Point 1 (index 0) blocking the combined move from Point 6; 1 checker at Point 21 (index 20) and 1 checker at Point 22 (index 21) blocking moves from White's head; 11 checkers at Point 12 (head, index 11). White to move. Scores {0, 0} are correct.
*   **Dice Roll:** {2, 3} (Setup on L359).
*   **Expectations:** White's checkers at the head (Point 24) cannot move because Points 22 and 21 are blocked by Black. The only potentially movable White checker is at Point 6 (index 5). This checker can move to Point 4 (index 3) with die 2, or to Point 3 (index 2) with die 3. Playing both dice is impossible because the total move of 5 pips (2+3) from Point 6 lands exactly on Point 1 (index 0), which is blocked by Black checkers. Since only single-die moves are possible for the only movable checker, the higher die rule applies. The only legal action should be the move using die 3: Point 6 (index 5) -> Point 3 (index 2).
*   **File/Lines:** `open_spiel/games/long_narde/bearing_off_test.cc`, lines 1-65.
*   **Rules Involved:** Legal action generation, rule for playing maximum dice, rule for playing higher die when only single moves are possible, blocking, player direction, head movement rules.

## Function: `ConsecutiveMovesTest`

**Test: Normal Turn Sequence**

*   **Objective:** Verify correct state transitions when players take turns in sequence.
*   **Board Setup:** Starts with the default initial state. Both players have 15 checkers on their respective heads, scores are {0, 0}. No explicit `SetupBoardState` call.
*   **Dice Rolls:** W: {1, 1} -> B: {1, 2} (or {2, 1}). Dice are set via `ApplyAction` for chance outcomes.
*   **Expectations:**
    1.  After White's move, the current player is `kChancePlayerId` (for Black's turn).
    2.  After the second roll ({1, 2}), the current player is `kOPlayerId` (Black).
    3.  After Black's move, the current player is `kChancePlayerId` (for White's next turn).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 397-467.
*   **Rules Involved:** Game turn flow, chance nodes.

## Function: `UndoRedoTest`

**Test: Apply and Undo Action**

*   **Objective:** Verify that applying a legal move and then undoing it restores the game state (specifically the board) to its exact previous configuration.
*   **Board Setup:** Uses `SetupBoardState` (L494) for a mid-game scenario.
    *   White: 2@pt4, 3@pt6, 1@pt9, 2@pt11, 2@pt15, 3@pt18, 2@pt21 (Total 15 on board, Score 0).
    *   Black: 3@pt2, 2@pt7, 2@pt10, 1@pt13(head), 2@pt16, 2@pt19, 2@pt23 (Total 14 on board, Score 1).
    *   White to move. Scores {0, 1} reflect Black has borne off 1 checker.
*   **Dice Roll:** {4, 2} (Setup on L495).
*   **Expectations:** After applying a legal action, the board changes. After calling `UndoAction` with the same action, the board returns to the identical state it was in before `ApplyAction` was called.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 469-549.
*   **Rules Involved:** State modification (`ApplyAction`), state restoration (`UndoAction`).

## Function: `PassMoveBehaviorTest`

**Test Case 1: Forced Pass (No Valid Moves)**

*   **Objective:** Verify that if no checker can be legally moved with either die, the only legal action is the specific double pass action.
*   **Board Setup:** Uses `SetupBoardState` (L571). White: 1 checker @ point 5. Black: 1 checker each @ point 2, point 4, point 6 (Total W:1, Score 14; Total B:3, Score 12). Scores {14, 12} correct. White to move.
*   **Dice Roll:** {1, 3} (Setup on L572).
*   **Expectations:** White cannot move checker at point 5 (index 4) with either die. In Long Narde, White moves from higher indices to lower indices. With die 1, White would move from index 4 to index 3 (point 4), which is occupied by Black. With die 3, White would move from index 4 to index 1 (point 2), which is also occupied by Black. Since both potential landing spots are blocked, White must pass with both dice.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 529-561.
*   **Rules Involved:** Legal action generation, blocking rules, forced pass.

**Test Case 2: Valid Moves Available**

*   **Objective:** Verify that if at least one legal move is possible, the double pass action is *not* included in the legal actions list.
*   **Board Setup:** Uses `SetupBoardState` (L590). White: 1 checker @ point 2, 1 checker @ point 4. Black: board empty (Total W:2, Score 13; Total B:0, Score 15). Scores {13, 15} correct. White to move.
*   **Dice Roll:** {1, 3} (Setup on L591).
*   **Expectations:** White has valid moves (e.g., point 2->1 with d1, point 4->1 with d3). The list of legal actions should be non-empty and should *not* contain the specific action corresponding to passing both dice {1, 3}.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 562-586.
*   **Rules Involved:** Legal action generation, exclusion of pass action when moves exist.

**Test Case 3: Forced Pass (Doubles, No Valid Moves)**

*   **Objective:** Verify that the forced pass rule applies correctly when doubles are rolled, but no moves are possible.
*   **Board Setup:** Black: 1@pt3, 1@pt8, 13@pt12(head). White: 1@pt1, 1@pt6, 1@pt10, 12@pt24(head) (Total W:15, Score 0; Total B:15, Score 0). Scores {0, 0} correct. Black to move.
*   **Dice Roll:** {2, 2, 2, 2} (Setup on L625).
*   **Expectations:** Black cannot move from point 3 (blocked by W@pt1), point 8 (blocked by W@pt6), or head point 12 (blocked by W@pt10). The only legal action must be the encoded pass for {2, 2}.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 587-618.
*   **Rules Involved:** Legal action generation, blocking, forced pass with doubles.


## Function: `VerifyDicePlayBehavior`

**Scenario 1: Two-Move Sequence Possible (Overrides Higher Die Rule)**

*   **Objective:** Verify that when a full two-move sequence is possible, it is the only legal action, even if individual dice could be played differently, and the "play higher die first" rule for single moves does not apply.
*   **Board Setup:** Uses `SetupBoardState` (L650). White: 1@pt9 (index 8), 1@pt4 (index 3). Black: 1@pt1 (index 0). (Total W:2, Score 13; Total B:1, Score 14). White to move.
*   **Dice Roll:** {5, 3} (Setup on L652).
*   **Expectations:** White *can* play a two-move sequence: point 9->6 (index 8->5, using d3), then point 4->off (index 3->off, using d5). Because a two-move sequence exists, the rule requiring playing the higher die (5) first *if only single moves are possible* does not apply. The only legal action should be the one corresponding to the full two-move sequence {pt9->6 (d3), pt4->off (d5)}.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 628-687.
*   **Rules Involved:** Legal action generation, playing the maximum number of dice, interaction between movement and subsequent moves, bear off.

**Scenario 2: Only Lower Die Initially Playable**

*   **Objective:** Verify that when blocks prevent the higher die from being played initially, the legal actions correctly use only the lower die.
*   **Board Setup:** Uses `SetupBoardState` (L709). White: 1@pt6 (index 5), 1@pt9 (index 8). Black: 1@pt1 (index 0), 1@pt4 (index 3). (Total W:2, Score 13; Total B:2, Score 13). White to move.
*   **Dice Roll:** {5, 3} (Setup on L710).
*   **Expectations:** White cannot use die 5 from either point 6 (blocked by B@pt1) or point 9 (blocked by B@pt4). White *can* use die 3 from point 6 (to 3) or point 9 (to 6). Subsequent moves with die 5 are also blocked in both cases. Thus, only single-move sequences using die 3 are possible. The legal actions generated must correspond to these single moves using die 3 (e.g., {pt6->3(d3), Pass(d5)} and {pt9->6(d3), Pass(d5)}).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 689-745.
*   **Rules Involved:** Legal action generation, blocking, playing maximum number of dice.

**Scenario 3: Higher Die Rule Application (Forced Single Move)**

*   **Objective:** Verify that when both dice can make a valid *initial* move, but neither allows a *second* move, the "play higher die first" rule is enforced, resulting in a single-move action using the higher die.
*   **Board Setup:** Uses `SetupBoardState` (L770). White: 1@pt9 (index 8), 1@pt4 (index 3). Black: 1@pt1 (index 0), 1@pt6 (index 5). (Total W:2, Score 13; Total B:2, Score 13). White to move.
*   **Dice Roll:** {5, 3} (Setup on L772).
*   **Expectations:** White can move point 9->4(d5) or point 9->6(d3), but neither move allows a second move (bearing off from point 4 is invalid because not all checkers are in home, and other moves are blocked). Since only single-die moves are possible, the higher die (5) must be used. The only legal action should be the one corresponding to {pt9->4 (d5), Pass(d3)}.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 747-825.
*   **Rules Involved:** Legal action generation, playing maximum number of dice, higher die rule for single moves, blocking.

## Function: `SingleCheckerBearOffTest`

**Test: Single Checker Bear Off Rules**

*   **Objective:** Verify the rules for bearing off a single checker, including higher die rule, exact pip count requirements, and terminal state flexibility.
*   **Board Setup:** Various test scenarios including:
    - Checker at position 0 (White) or 12 (Black) with dice {1, 6}
    - Checker at position 1 (White) or 13 (Black) with dice {1, 3}
    - Checker at position 14 (Black) needing 3 pips with dice {1, 3} (terminal state)
*   **Expectations:**
    - Must use higher die when both dice can bear off
    - Must use exact die when only one die matches pip count
    - Cannot bear off with insufficient die value
    - For terminal states, both options should be valid: direct bear-off with higher die + pass, or move with lower die + bear-off with higher die
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 925-1053.
*   **Rules Involved:** Bear off rules, higher die rule, exact pip count requirements, terminal state flexibility.

## Function: `BearOffLastCheckerTest`

**Test: Last Checker Bear Off**

*   **Objective:** Verify the behavior when bearing off the last checker, ensuring proper handling of the unused die.
*   **Board Setup:**
    *   **White Test:** White: 1@pt2 (index 1, needs 2 pips). Black: 15@pt12(head). (Score {14, 0}).
    *   **Black Test:** Black: 1@pt14 (index 13, needs 2 pips). White: 15@pt24(head). (Score {0, 14}).
*   **Dice Roll:** {4, 5}
*   **Expectations:**
    *   White: Must bear off using die 5 (higher), pass die 4.
    *   Black: Must bear off using die 5 (higher), pass die 4.
    *   Only one legal action should be available in each case.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 1054-1143.
*   **Rules Involved:** Bear off rules, higher die rule, pass move rules, last checker special cases.

## Function: `DirectBearOffTest`

**Test: Direct Bear Off (White and Black)**

*   **Objective:** Verify that both White and Black can directly bear off two checkers when all checkers are in their respective home quadrants and the dice match exactly.
*   **Board Setup:**
    *   **White Test:** White: 1@pt1 (index 0), 1@pt2 (index 1). Black: 15@pt12(head). (Total W:2, Score 13; Total B:15, Score 0). White to move. (Setup L830)
    *   **Black Test:** Black: 1@pt14 (index 13, needs 2 pips), 1@pt15 (index 14, needs 3 pips). White: 15@pt24(head). (Total B:2, Score 13; Total W:15, Score 0). Black to move. (Setup L876)
*   **Dice Roll:** White Test: {1, 2}, Black Test: {2, 3}
*   **Expectations:**
    *   White: The legal actions should include the sequence {pt1->off (d1), pt2->off (d2)}.
    *   Black: The legal actions should include the sequence {pt14->off (d2), pt15->off (d3)}.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_actions.cc`, lines 829-922.
*   **Rules Involved:** Bear off rules, home quadrant rules, exact dice usage.

## Test Suite: `long_narde_test_bridges.cc`

### Function: `TestBridgeFormation`

**Test 1: Bridge Formation Legality (Opponent No Checkers)**

*   **Objective:** Verify that a move forming a 6-checker bridge is considered *legal* if the opponent (Black) has no checkers on the board at all, even though such a bridge would normally be illegal if Black had checkers trapped behind it.
*   **Board Setup:** White: Checkers in home board (pts 1-6 / indices 0-5) arranged as `{2, 1, 1, 0, 2, 1}` (Total 7 checkers, Score 8). Black: No checkers on the board (Score 15). White to move. (Setup on L48).
*   **Dice Roll:** {4, 1} (Setup on L49). The move tested uses die 1.
*   **Expectations:** The potential move from point 5 (index 4) to point 4 (index 3) using die 1, which would complete a 6-checker bridge `[2, 1, 1, 1, 1, 1]`, should *not* be flagged as forming an illegal bridge (`WouldFormBlockingBridge` should return `false`).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 19-56.
*   **Rules Involved:** Bridge formation rule (illegal 6-in-a-row), exception to bridge rule when opponent has no checkers on board, `WouldFormBlockingBridge` function logic.

**Test 2: Bridge Formation Legality (Opponent Relief)**

*   **Objective:** Verify that forming a 6-checker bridge *is* legal if the opponent has at least one checker positioned ahead of the bridge, providing "relief".
*   **Board Setup:** Same White setup as Test 1: Home board `{2, 1, 1, 0, 2, 1}` (Score 8). Black: 14 checkers at point 7 (index 6), 1 checker at point 19 (index 18). Critically, Black's checker at index 18 has path index 17, which is greater than the path index 6 of White's bridge start at index 5, making it "ahead" of the bridge. White to move. (Setup on L76).
*   **Dice Roll:** {4, 1} (Setup on L77). The move tested uses die 1.
*   **Expectations:** The potential move from point 5 (index 4) to point 4 (index 3) using die 1, which completes the 6-checker bridge, should *not* be flagged as illegal (`WouldFormBlockingBridge` should return `false`) because Black has a checker ahead of the bridge.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 58-86.
*   **Rules Involved:** Bridge formation rule (illegal 6-in-a-row), exception to bridge rule when opponent has checkers ahead ("relief"), `WouldFormBlockingBridge` function logic.

**Test 3: Illegal Bridge Direct Move Validation**

*   **Objective:** Verify that a direct move, which would form an illegal 6-checker bridge because the opponent has checkers trapped behind it, is correctly identified as invalid by the `IsValidCheckerMove` function.
*   **Board Setup:** White: Same home board setup as Test 1: `{2, 1, 1, 0, 2, 1}` (Score 8). Black: All 15 checkers are placed at Black's head (point 12 / index 11). This configuration ensures no Black checkers are "ahead" of White's home board (indices 0-5). White to move. (Setup on L103).
*   **Dice Roll:** {4, 1} (Setup on L104). The move tested uses die 1.
*   **Expectations:** The direct move from point 5 (index 4) to point 4 (index 3) using die 1 should be considered invalid. `IsValidCheckerMove` for this specific `CheckerMove` should return `false` because it forms a 6-in-a-row bridge, and all of Black's checkers are behind this bridge.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 88-120.
*   **Rules Involved:** Bridge formation rule (illegal 6-in-a-row), condition for illegality (opponent checkers must be trapped behind), `IsValidCheckerMove` logic.

**Test 4: Black Forming Illegal Bridge (White Behind)**

*   **Objective:** Verify that Black forming an illegal bridge (6-in-a-row in their home region, points 13-18 / indices 12-17) is correctly detected as illegal when White has checkers, but none are positioned ahead of the potential bridge.
*   **Board Setup:** Black: Checkers arranged near home region `{2@13, 1@14, 1@15, 1@17, 2@18}` (indices `{12, 13, 14, 16, 17}`) with a gap at point 16 (index 15). A checker is at point 21 (index 20) ready to move into the gap. (Total 8 checkers, Score 7). White: All checkers are behind Black's potential bridge start point (point 18 / index 17), e.g., 5@pt19, 5@pt20, 5@pt24(head). (Total 15, Score 0). Black to move. (Setup on L139).
*   **Dice Roll:** {5, 1} (Setup on L137). The move tested uses die 5.
*   **Expectations:** The potential move from point 21 (index 20) to point 16 (index 15) using die 5 should be flagged as illegal (`WouldFormBlockingBridge` returns `true`). Consequently, the direct checker move should also be invalid (`IsValidCheckerMove` returns `false`).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 116-150.
*   **Rules Involved:** Bridge formation rule (Black's home region), condition for illegality (opponent checkers trapped behind), `WouldFormBlockingBridge` logic, `IsValidCheckerMove` logic.

**Test 5: Black Forming Legal Bridge (White Ahead)**

*   **Objective:** Verify that Black forming a 6-checker bridge in their home region is legal if White has at least one checker positioned ahead of the bridge.
*   **Board Setup:** Same Black setup as Test 4: Checkers near home region `{2@13..2@18}` with a gap at point 16 (index 15), and a checker at point 21 (index 20) ready to move. (Total 8 checkers, Score 7). White: 1 checker at point 19 (index 18), which is behind the bridge from Black's perspective, and 14 checkers at point 1 (index 0), which are ahead of the bridge from Black's perspective. (Total 15, Score 0). Black to move. (Setup on L163).
*   **Dice Roll:** {5, 1} (Setup on L172). The move tested uses die 5.
*   **Expectations:** The potential move from point 21 (index 20) to point 16 (index 15) using die 5 should *not* be flagged as illegal (`WouldFormBlockingBridge` returns `false`). Consequently, the direct checker move should also be valid (`IsValidCheckerMove` returns `true`).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 152-193.
*   **Rules Involved:** Bridge formation rule (Black's home region), exception to bridge rule when opponent has at least one checker ahead of the bridge ("relief"), `WouldFormBlockingBridge` logic, `IsValidCheckerMove` logic.

**Test 6: White Wrap-Around Bridge Illegality (Black at Head)**

*   **Objective:** Verify that White forming a wrap-around bridge (e.g., points 24, 1, 2, 3, 4) is illegal if Black has checkers at its head position, which is not considered "ahead" of the bridge's starting point (in virtual coordinates).
*   **Board Setup:** White: Checkers forming a near wrap-around bridge `{1@pt24, 1@pt1, 1@pt2, 1@pt3, 1@pt4}` (indices `{23, 0, 1, 2, 3}`) with a gap at point 5 (index 4). A checker is at point 6 (index 5) ready to move into the gap. (Total 6 checkers, Score 9). Black: All 15 checkers at Black's head (point 12 / index 11). Crucially, Black's head (virtual coordinate 23) is at the same virtual coordinate as White's potential bridge start (point 24 / index 23, virtual coordinate 23), but not strictly ahead of it. White to move. (Setup on L209).
*   **Dice Roll:** {1, 2} (Setup on L210). The move tested uses die 1.
*   **Expectations:** The potential move from point 6 (index 5) to point 5 (index 4) using die 1 should be flagged as illegal (`WouldFormBlockingBridge` returns `true`), because Black is NOT considered ahead of the bridge start (equal virtual coordinates). Consequently, the direct checker move should be invalid (`IsValidCheckerMove` returns `false`).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 195-221.
*   **Rules Involved:** Bridge formation rule (wrap-around), definition of "ahead" using virtual coordinates (must be strictly ahead), `WouldFormBlockingBridge` logic, `IsValidCheckerMove` logic.

**Test 7: White Wrap-Around Bridge Illegality (Black Behind)**

*   **Objective:** Verify that White forming a wrap-around bridge (e.g., points 24, 1, 2, 3, 4) is illegal if Black's checkers are all positioned *behind* the bridge's starting point (in virtual coordinates).
*   **Board Setup:** Same White setup as Test 6: Near wrap-around bridge `{1@pt24..1@pt4}` with a gap at point 5 (index 4), checker at point 6 (index 5). (Total 6 checkers, Score 9). Black: All 15 checkers are moved to point 11 (index 10). Black's checker at index 10 (virtual coordinate 22) is now *behind* White's potential bridge start (point 4 / index 3, virtual coordinate 23). White to move. (Setup on L237).
*   **Dice Roll:** {1, 2} (Setup on L238). The move tested uses die 1.
*   **Expectations:** The potential move from point 6 (index 5) to point 5 (index 4) using die 1 should now be flagged as illegal (`WouldFormBlockingBridge` returns `true`), because Black is considered behind the bridge start. Consequently, the direct checker move should also be invalid (`IsValidCheckerMove` returns `false`).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 223-250.
*   **Rules Involved:** Bridge formation rule (wrap-around), definition of "ahead"/"behind" using virtual coordinates, illegality condition (opponent trapped behind), `WouldFormBlockingBridge` logic, `IsValidCheckerMove` logic.



**Check 3: White Move Towards Potential Bridge (Opponent Relief)**

*   **Objective:** Verify White (moving CCW, higher to lower indices) can legally move towards forming a bridge (target: indices 8 & 7) when the opponent (Black) has a checker positioned ahead, providing relief.
*   **Board Setup:** White: `{2@pt10(idx9), 2@pt11(idx10), 1@pt12(idx11), 1@pt13(idx12)}` (Score 9). Black: `{2@pt17(idx16), 2@pt18(idx17)}` plus a critical checker at point 1 (index 0), which is ahead of White's potential bridge target (index 8). (Score 10). White to move. (Setup on L291).
*   **Dice Roll:** {1, 3} (Setup on L302).
*   **Expectations:** The move for White from point 10 (index 9) to point 9 (index 8) using die 1 should be considered valid (`IsValidCheckerMove` returns `true`) because Black has a checker ahead (at index 0) providing relief.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 296-303.
*   **Rules Involved:** Bridge formation rule (White CCW), exception for opponent relief, `IsValidCheckerMove` logic.

**Check 4: White Move to Point Within Potential Bridge Range**

*   **Objective:** Verify that White can legally move a checker *to* a point that is part of a potential, but incomplete, bridge, if the move itself does not complete the bridge.
*   **Board Setup:** Same as Check 3. White to move. (Setup on L298).
*   **Dice Roll:** {1, 3} (Setup on L299).
*   **Expectations:** The move for White from point 10 (index 9) to point 14 (index 13) using die 4 should be considered valid (`IsValidCheckerMove` returns `true`). Landing on point 14 (index 13) is allowed because, although it's within the potential bridge range `{pt12, pt13, pt14}`, this specific move doesn't create the 6-in-a-row sequence.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_bridges.cc`, lines 310-314.
*   **Rules Involved:** `IsValidCheckerMove` logic, bridge formation rule (only applies when the bridge is completed), basic movement validity.
### Test 8: White forms legal bridge with opponent relief (CCW movement)\n\n* **Objective:** Verify White (moving CCW) can move towards forming a bridge (at indices 8 & 7) when Black has a checker ahead providing relief.\n* **Board Setup:** White: `{2@pt10(idx9), 2@pt11(idx10), 1@pt12(idx11), 1@pt13(idx12)}`. Black: `{2@pt17(idx16), 2@pt18(idx17)}` + 1@pt1(idx0). White to move.\n* **Dice Roll:** {1, 3}\n* **Expectations:** White can move 9->8 (die 1) and 10->7 (die 3) to form a legal bridge since Black has a checker ahead.\n* **File/Function Ref:** Anchor: `test-bridgetest-8` in `long_narde_test_bridges.cc`\n* **Rules Involved:** Bridge formation legality, opponent relief, counter-clockwise movement.\n
/* Test 9 removed: This test did not effectively evaluate valid or illegal bridge formation. See code for rationale. */

## Test Suite: `long_narde_test_endgame.cc`

### Function: `BearingOffBasicTest`

**Test 1: White Can Bear Off**

*   **Objective:** Verify that `AllInHome` returns true for White when all White checkers are within their home board (points 1-6 / indices 0-5), and the board is symmetric with Black's checkers in their home board (points 19-24 / indices 18-23).
*   **Board Setup:** White: Checkers distributed across points 1-6 `{3@1, 3@2, 3@3, 2@4, 2@5, 2@6}` (indices 0-5), rest empty. Black: Checkers distributed across points 19-24 `{3@19, 3@20, 3@21, 2@22, 2@23, 2@24}` (indices 18-23), rest empty. White to move. (Setup on L20).
*   **Dice Roll:** {6, 1} (Setup on L21).
*   **Expectations:** `AllInHome(kXPlayerId)` should return `true`.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 11-30.
*   **Rules Involved:** Home region definition, `AllInHome` logic.

**Test 2: White Cannot Bear Off (Checker Outside)**

*   **Objective:** Verify that `AllInHome` returns false for White if at least one White checker is outside the home board.
*   **Board Setup:** White: 1 checker at pt7 (idx6), 14 checkers at head (pt24/idx23). Black: Checkers distributed in their home (indices 12-17). White to move. (Corrected C++ setup, Black position adjusted for realism).
*   **Dice Roll:** {6, 1}
*   **Expectations:** `AllInHome(kXPlayerId)` should return `false` because one checker is outside home.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 39–51.
*   **Rules Involved:** Home region definition, `AllInHome` logic.

**Test 3: Black Can Bear Off**

*   **Objective:** Verify that `AllInHome` returns true for Black when all Black checkers are within their home board (points 13–18 / indices 12–17), and that the presence of all 15 white checkers legally arranged in the white home (indices 0–5) does not block Black (no illegal bridge).
*   **Board Setup:** Black: Checkers distributed across points 13–18 `{3@13, 3@14, 3@15, 2@16, 2@17, 2@18}` (indices 12–17). White: All 15 checkers legally arranged in home `{3@1, 3@2, 3@3, 2@4, 2@5, 2@6}` (indices 0–5). Black to move. (Setup on L58).
*   **Dice Roll:** {2, 2} (Setup on L54).
*   **Expectations:** `AllInHome(kOPlayerId)` should return `true`.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 47-61.
*   **Rules Involved:** Home region definition (Black), `AllInHome` logic.

### Function: `BearingOffLogicTest`

**Test 1: White Exact vs. Non-Exact Bear Off**

*   **Objective:** Verify that White can bear off with an exact die roll but not with a die roll that lands the checker on point 1 (index 0).
*   **Board Setup:** White: 1 checker at point 2 (index 1), 14 checkers at point 3 (index 2). Black: All checkers at head. White to move. (Setup on L73).
*   **Dice Roll:** {1, 3} (Setup on L74).
*   **Expectations:** A move using die 1 from point 2 (index 1) should land on point 1 (index 0) and *not* bear off. A move using die 3 from point 3 (index 2) should bear off.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 65-110.
*   **Rules Involved:** Bear off rules (exact pip count), basic movement.



### Function: `BearingOffFromPosition1Test`

*   **Objective:** Verify correct bearing-off logic from point 2 (index 1) when multiple checkers are present, ensuring both half-moves (using each die) are tested and the rule for bearing off with a higher die is enforced.
*   **Board Setup:** White: 13 checkers at point 1 (index 0), 2 checkers at point 2 (index 1). Points 3-6 (indices 2-5) are empty. Black: All checkers at head. White to move. (Setup on L164).
*   **Dice Roll:** {1, 3} (Setup on L166).
*   **Expectations:**
    *   Using die 1 from point 2 (index 1) should move a checker to point 1 (index 0), not bear off (since die 1 is not enough to bear off from index 1).
    *   Using die 3 from point 2 (index 1) *should* bear off, because the die roll (3) is greater than the exact pips needed (2), and all higher points (indices 2-5) are empty.
    *   Having two checkers at point 2 ensures both dice can be applied independently, verifying correct handling of multiple checkers and move sequencing.
    *   No pass move should be generated as valid moves exist.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 154-206.
*   **Rules Involved:** Bear off rules (exact pip count), bearing off with higher die when no checkers on higher points, correct sequencing with multiple checkers at the same point.

### Function: `BearingOffBlackTest`

*   **Objective:** Verify that Black is eligible to bear off only when all Black checkers are within the home region (indices 12-17). This test ensures that bearing off is not permitted if any Black checker is outside this range.
*   **Board Setup:** Black: All 15 checkers distributed within the home region (indices 12-17), none at indices 22 or 23. White: All 15 checkers at the head (index 24). Black to move.
*   **Dice Roll:** {2, 3}.
*   **Expectations:**
    *   `AllInHome(kOPlayerId)` should return `true` because all Black checkers are within the home region (12-17).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 218-281.
*   **Rules Involved:** Bear off rules (all checkers must be in home region to enable bearing off), `AllInHome` logic (using 12-17 region).

### Function: `EndgameScoreTest`

**Test 1: Mars Scoring (Default)**

*   **Objective:** Verify the default scoring for a "Mars" win (opponent has borne off zero checkers and has checkers remaining on the board, typically on the head).
*   **Board Setup:** White: All 15 checkers borne off (Score 15). Black: All 15 checkers still on the head (Score 0). (Setup on L274).
*   **Expectations:** The game state is terminal. White's return is 2.0, Black's return is -2.0.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 261-288.
*   **Rules Involved:** Terminal state detection, scoring rules (Mars).

**Test 2: Oin Scoring (Standard Win)**

*   **Objective:** Verify the scoring for a standard win ("Oin") where the opponent has borne off at least one checker.
*   **Board Setup:** White: All 15 checkers borne off (Score 15). Black: Has 10 checkers remaining on point 11 (index 10), meaning 5 checkers were borne off (Score 5). (Setup on L299).
*   **Expectations:** The game state is terminal. White's return is 1.0, Black's return is -1.0.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 290-312.
*   **Rules Involved:** Terminal state detection, scoring rules (Oin).

**Test 3: Tie Scoring (WinLossTie Mode)**

*   **Objective:** Verify that a tie occurs (0 points each) when both players have borne off all checkers and the game is loaded with `scoring_type=winlosstie_scoring`.
*   **Board Setup:** White: All 15 checkers borne off (Score 15). Black: All 15 checkers borne off (Score 15). (Setup on L327).
*   **Game Parameter:** `scoring_type=winlosstie_scoring` (L316).
*   **Expectations:** The game state is terminal. White's return is 0.0, Black's return is 0.0.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 314-340.
*   **Rules Involved:** Terminal state detection, scoring rules (Tie), game parameters (`scoring_type`).

### Function: `ScoringSystemTest`

*   **Objective:** Verify that the game correctly loads and recognizes the specified scoring system (`winloss_scoring` or `winlosstie_scoring`) via game parameters.
*   **Test Cases:**
    1.  Load game with default parameters: Expect `scoring_type` to be `winloss_scoring`.
    2.  Load game explicitly with `long_narde(scoring_type=winloss_scoring)`: Expect `scoring_type` parameter to be `winloss_scoring`.
    3.  Load game explicitly with `long_narde(scoring_type=winlosstie_scoring)`: Expect `scoring_type` parameter to be `winlosstie_scoring`.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 342-371.
*   **Rules Involved:** Game parameter parsing (`scoring_type`).

### Function: `SingleLegalMoveTestBlack`

**Test 1: Forced Sequence due to Block**

*   **Objective:** Verify that if a block prevents one die from being played initially, the game correctly generates the forced sequence using the other die first.
*   **Board Setup:** Black: 1 checker at point 23 (index 22), 14 checkers borne off. White: 1 checker at point 22 (index 21) blocking Black's use of die 1 from point 23. Remaining White checkers at head. Black to move. (Setup on L387).
*   **Dice Roll:** {1, 2} (Setup on L388).
*   **Expectations:** The only legal action should be the sequence: first move point 23 -> 21 (index 22 -> 20) using die 2, then second move point 21 -> 20 (index 20 -> 19) using die 1.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 375-414.
*   **Rules Involved:** Legal action generation, blocking, forced move sequences.

**Test 2: Single Die Max Play Rule (Black)**

*   **Objective:** Verify that when Black can only play one die (due to blocks), and both dice could potentially make a valid first move if the other was ignored, Black is forced to play the single available move (using the lower die in this case) according to the "play maximum number of dice" rule.
*   **Board Setup:** Black: 1 checker at point 19 (index 18), 1 checker at point 21 (index 20). White: Checkers blocking potential moves: point 14 (index 13) for die 5 from 19; point 16 (index 15) for die 5 from 21; point 17 (index 16) for die 2 from 19. Black to move. (Setup on L423).
*   **Dice Roll:** {5, 2} (Setup on L424).
*   **Expectations:** Die 5 is blocked from both possible starting positions (19, 21). Die 2 is blocked from point 19 (index 18) but is playable from point 21 (index 20) to point 19 (index 18). Since only a single die (2) can be played, that move must be made. The only legal action should correspond to the single move: point 21 -> 19 (index 20 -> 18) using die 2, with the other die (5) being passed.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 416-456.
*   **Rules Involved:** Legal action generation, blocking, playing maximum number of dice rule (forcing single available move).

### Function: `BearingOffLogicTestBlackNearEnd`

*   **Objective:** Verify Black's bear off logic for checkers close to being borne off, testing exact pips, higher die usage, and normal moves.
*   **Board Setup:** Black: 1 checker at point 14 (index 13, needs 2 pips), 1 checker at point 15 (index 14, needs 3 pips). 13 checkers already borne off (Score 13). White: Checkers at head. Black to move. (Setup on L476).
*   **Dice Roll:** {5, 2} (Setup on L477).
*   **Expectations:** The test checks if the following *individual* half-moves are possible within the generated legal full actions:
    *   Bear off from point 14 (index 13) with die 2 (exact pips): TRUE
    *   Bear off from point 14 (index 13) with die 5 (higher die, no checkers further back): TRUE
    *   Bear off from point 15 (index 14) with die 5 (higher die, no checkers further back): TRUE
    *   Bear off from point 15 (index 14) with die 2 (die too low): FALSE
    *   Move from point 15 (index 14) to point 13 (index 12) with die 2 (normal move): TRUE
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 462-524.
*   **Rules Involved:** Bear off rules (exact pips, higher die when furthest), basic movement validity.

### Function: `CannotBearOffIfNotAllInHomeTest`

*   **Objective:** Verify that neither player can make a bear-off move if they still have checkers outside their respective home boards (White: 0-5, Black: 12-17).

**Test 1: White Cannot Bear Off**

*   **Board Setup:** White: 14 checkers in home (indices 0-2), 1 checker at point 7 (index 6). Black: 14 in home (12-14), 1 at point 19 (index 18). White to move. (Setup on L556).
*   **Dice Roll:** {1, 6} (Setup on L557).
*   **Expectations:** `AllInHome(kXPlayerId)` is false. No legal action generated for White should start with a bear-off move.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 554-588.
*   **Rules Involved:** `AllInHome` logic, bear off prerequisite (all checkers must be home), legal action generation.

**Test 2: Black Cannot Bear Off**

*   **Board Setup:** Black: Same as above (14 in home 12-14, 1 at index 18). White: All checkers moved to head. Black to move. (Setup on L594).
*   **Dice Roll:** {1, 6} (Setup on L595).
*   **Expectations:** `AllInHome(kOPlayerId)` is false (assuming index 18 is outside the strict home definition). No legal action generated for Black should start with a bear-off move.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_endgame.cc`, lines 590-623.
*   **Rules Involved:** `AllInHome` logic, bear off prerequisite, legal action generation.

## Test Suite: `random_sim_test.cc`

### Function: `RunRandomSimTests` / `MemoryEfficientRandomSim`

*   **Objective:** Perform a series of random game simulations to stress-test the game implementation and check for basic consistency between legal move generation and validation.
*   **Setup:** Starts multiple games from the initial state.
*   **Process:** In each game, random legal actions (or chance outcomes) are chosen repeatedly until the game terminates or a maximum number of moves (1000) is reached.
*   **Expectations:**
    *   The simulations should complete without crashing.
    *   Games should terminate.
    *   An internal check verifies that every move selected from `LegalActions` is also considered valid by `IsValidCheckerMove`. Any discrepancies are reported.
*   **File/Lines:** `open_spiel/games/long_narde/random_sim_test.cc`, entire file (main logic in `MemoryEfficientRandomSim`, lines 18-128).
*   **Rules Involved:** General game flow, termination conditions, legal action generation (`LegalActions`), move validation (`IsValidCheckerMove`).

## Test Suite: `long_narde_test_basic.cc`

### Function: `InitialBoardSetupTest`

*   **Objective:** Verify the correct initial placement of checkers on the board.
*   **Board Setup:** Default initial state.
*   **Expectations:**
    *   White has 15 checkers on point 24 (index 23) and 0 on all other points.
    *   Black has 15 checkers on point 12 (index 11) and 0 on all other points.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_basic.cc`, lines 19-43.
*   **Rules Involved:** Initial board setup rules.

### Function: `BasicLongNardeTestsCheckNoHits`

*   **Objective:** Verify that no hits are possible in Long Narde. (Test reactivated after resolving memory leak issues).
*   **Setup:** Standard game setup.
*   **Expectations:** Test confirms no hits occur during gameplay simulation or state checks. (Note: Original `RandomSimTest` was commented; current implementation might use a different approach).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_basic.cc` (Check updated line numbers if necessary)
*   **Rules Involved:** No hitting rule.

<!-- Section for BasicLongNardeTestsDoNotStartWithDoubles removed as the function was deleted -->

### Function: `WhiteMovesFirstTest`

*   **Objective:** Verify that White (Player X) makes the first move after the initial dice roll. (Test logic simplified).
*   **Setup:** Starts a new game and applies the first chance outcome.
*   **Expectations:** After the chance node resolves, the `CurrentPlayer` must be `kXPlayerId` (White).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_basic.cc` (Check updated line numbers if necessary)
*   **Rules Involved:** Turn order rules (White moves first).

## Test Suite: `long_narde_test_legacy.cc`

### Function: `BasicLongNardeTests`

*   **Objective:** Runs a collection of legacy tests, including standard OpenSpiel random simulations and checks for core game properties.
*   **Process:**
    *   Loads the "long_narde" game.
    *   Runs `testing::RandomSimTest` (10 simulations).
    *   Runs `testing::RandomSimTestWithUndo` (10 simulations).
    *   Checks game type properties (ChanceMode, Dynamics, Information, Utility, RewardModel).
    *   Checks `NumPlayers()` and `MaxChanceOutcomes()`.
    *   Calls `TestMovementRules()` via an internal forwarder (`TestBasicMovementInternal`).
*   **Expectations:** All standard tests pass, game properties match expected values, and `TestMovementRules` passes.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_legacy.cc`, lines 21-49.
*   **Rules Involved:** General game properties, rules covered by `TestMovementRules`.

## Test Suite: `long_narde_test_movement.cc`

### Function: `TestBasicMovement`

*   **Objective:** Verify basic movement rules and initial game state progression.
*   **Setup:** Starts with the default initial state, applies a dice roll of {4, 4}.
*   **Expectations:**
    *   Initial state is a chance node with `CurrentPlayer() == kChancePlayerId`.
    *   After applying dice outcome, White (kXPlayerId) is the current player.
    *   Legal actions exist for White's first move.
    *   After White's move, the current player is Black (kOPlayerId).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 10-50.
*   **Rules Involved:** Basic game flow, turn order, legal action generation.

### Function: `FirstTurnTest`

*   **Objective:** Verify that on the first turn with special doubles (6,6), multiple checkers from the head are legal, and that after the first turn, `IsFirstTurn(player)` returns false.
*   **Setup:** Starts with the default initial state, applies a dice roll of {6, 6}.
*   **Expectations:**
    *   `IsFirstTurn(kXPlayerId)` returns true initially.
    *   Legal actions include moves with multiple checkers from the head.
    *   After White's move and the next dice roll, `IsFirstTurn(kXPlayerId)` returns false.
    *   For Black's first turn, `is_on_first_turn_` is true.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 52-100.
*   **Rules Involved:** First turn special rules, head movement rules.

### Function: `HeadRuleTest`

*   **Objective:** Verify the head rule: on the first turn with doubles, multiple checkers can move from the head; on subsequent turns, only one checker can move from the head per die.
*   **Setup:** Tests both first turn and non-first turn scenarios with doubles.
*   **Expectations:**
    *   On first turn with doubles, multiple checkers can move from the head.
    *   On non-first turn with doubles, only one checker can move from the head per die.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 102-150.
*   **Rules Involved:** Head rule, first turn vs. non-first turn rules.

### Function: `MovementDirectionTest`

*   **Objective:** Verify that White moves in decreasing index direction (clockwise) and Black effectively wraps around (counter-clockwise).
*   **Setup:** Sets up board states for both White and Black to test movement direction.
*   **Expectations:**
    *   White's moves decrease the position index.
    *   Black's moves effectively wrap around (counter-clockwise).
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 152-200.
*   **Rules Involved:** Player movement direction, board representation.

### Function: `NoLandingOnOpponentTest`

*   **Objective:** Verify that checkers cannot land on points occupied by opponent checkers.
*   **Setup:** Sets up a board with White and Black checkers at specific positions.
*   **Expectations:**
    *   No legal action allows landing on a point occupied by an opponent.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 202-250.
*   **Rules Involved:** Landing rules, blocking.

### Function: `HomeRegionsTest`

*   **Objective:** Verify the correct definition of home regions for both players.
*   **Setup:** Tests various positions on the board.
*   **Expectations:**
    *   White's home region is points 1-6 (indices 0-5).
    *   Black's home region is points 13-18 (indices 12-17).
    *   All other points are not in either player's home region.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 252-300.
*   **Rules Involved:** Home region definition, bearing off prerequisites.

### Function: `TestIllegalLandingInLegalActions`

*   **Objective:** Verify that `LegalActions()` does not generate moves landing on occupied points.
*   **Setup:** Sets up a board with specific checker positions to test landing rules.
*   **Expectations:**
    *   No legal action allows landing on a point occupied by any checker.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 302-350.
*   **Rules Involved:** Legal action generation, landing rules.

### Function: `TestHalfMoveGeneration`

*   **Objective:** Verify that half-move generation correctly handles individual dice.
*   **Setup:** Sets up board states to test half-move generation for White.
*   **Expectations:**
    *   Half-moves are correctly generated for individual dice.
    *   Illegal half-moves (landing on occupied points) are not generated.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 352-400.
*   **Rules Involved:** Half-move generation, legal action generation.

### Function: `HeadRuleTestBlack`

*   **Objective:** Verify the head rule for Black player.
*   **Setup:** Similar to `HeadRuleTest` but for Black player.
*   **Expectations:**
    *   On first turn with doubles, multiple Black checkers can move from the head.
    *   On non-first turn with doubles, only one Black checker can move from the head per die.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 402-450.
*   **Rules Involved:** Head rule for Black player, first turn vs. non-first turn rules.

### Function: `TestHalfMoveGenerationBlack`

*   **Objective:** Verify that half-move generation correctly handles individual dice for Black player.
*   **Setup:** Sets up board states to test half-move generation for Black.
*   **Expectations:**
    *   Half-moves are correctly generated for individual dice for Black.
    *   Illegal half-moves (landing on occupied points) are not generated.
*   **File/Lines:** `open_spiel/games/long_narde/long_narde_test_movement.cc`, lines 452-500.
*   **Rules Involved:** Half-move generation for Black, legal action generation.

## Impact of Removing the Legacy Test Suite

The legacy test suite (`long_narde_test_legacy.cc`) has been completely removed from the repository. This change simplifies the overall test structure by eliminating redundant code, reducing maintenance overhead, and ensuring that only modern, efficient tests are utilized. However, this removal may introduce risks if any unique edge cases from the legacy suite are not covered in the current tests. It is recommended to perform a thorough review of the remaining test coverage to maintain the integrity and reliability of the Long Narde game implementation.
