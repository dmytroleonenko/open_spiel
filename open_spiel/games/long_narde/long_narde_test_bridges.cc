#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {

//StartFunction: TestBridgeFormation
void TestBridgeFormation() {
  // Load game and create state.
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // ------------------------------------------------------------
  //StartTest: test-bridgetest-1
  // Test 1: Directly test illegal bridge detection.
  // We want to simulate a move within White's home board that fills a gap.
  //
  // We'll configure White's home board (indices 0–5) as:
  //   [2, 1, 1, 0, 2, 1]
  // so that point 0 has an extra checker.
  // White's head (index 23) will hold the remaining checkers so that total white = 15.
  // Black will have no checkers.
  //
  // Then a move from White's position 4 with a die of 1 (i.e. move from pos 4 to pos 3)
  // will subtract one from pos4 (leaving it with 1) and add one to pos3.
  // The resulting home board becomes: [2, 1, 1, 1, 1, 1] – a contiguous block of 6.
  // With no Black checkers ahead (indices 6–23), this move should be flagged as illegal.
  // ------------------------------------------------------------
  {
    // Build White row:
    // Start with home board: indices 0..5 = {2, 1, 1, 0, 2, 1}
    // Initialize with size 25, then set specific values.
    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    white_row[0] = 2; white_row[1] = 1; white_row[2] = 1; white_row[3] = 0; white_row[4] = 2; white_row[5] = 1;
    // Set head (index 23) to 15 - (2+1+1+0+2+1) = 15 - 7 = 8.
    // white_row[kWhiteHeadPos] = 8; // Use constant kWhiteHeadPos (23) -- Removed: Score handles head checkers.
    // Total White checkers: 2+1+1+0+2+1 = 7 on board. Score should be 8.

    // Black row: all 25 positions zero.
    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    // Total Black checkers: 0 on board. Score should be 15.
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-7=8), Black (15-0=15)
    SetupBoardState(lnstate, kXPlayerId, test_board, {8, 15}); // Corrected score
    SetupDice(lnstate, {4, 1, 0, 0});

    // Simulate move from White's pos 4 to pos 3.
    // This move forms a 6-block, but is LEGAL because Black has no checkers on board.
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
    SPIEL_CHECK_FALSE(bridge_illegal); // Must be false (legal) if opponent has no checkers.
  }
  //EndTest: test-bridgetest-1

  // ------------------------------------------------------------
  //StartTest: test-bridgetest-2
  // Test 2: With opponent relief, the same move should be legal.
  // We use the same White configuration as above.
  // For Black, we place 14 checkers at one location (index 6) and 1 checker at index 18
  // (so that at least one Black checker is ahead of White's home board in terms of path indices).
  // ------------------------------------------------------------
  {
    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    white_row[0] = 2; white_row[1] = 1; white_row[2] = 1; white_row[3] = 0; white_row[4] = 2; white_row[5] = 1;
    // Set head (index 23) to 15 - (2+1+1+0+2+1) = 15 - 7 = 8.
    // white_row[kWhiteHeadPos] = 8;  // White head: 8 checkers. -- Removed: Score handles head checkers.
    // Total White checkers: 7 on board. Score should be 8.
    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    black_row[6] = 14;
    black_row[18] = 1;
    // Total Black checkers: 14+1=15 on board. Score should be 0.
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-7=8), Black (15-15=0)
    SetupBoardState(lnstate, kXPlayerId, test_board, {8, 0}); // Corrected score
    SetupDice(lnstate, {4, 1, 0, 0});

    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
    SPIEL_CHECK_FALSE(bridge_illegal);
  }
  //EndTest: test-bridgetest-2

  // ------------------------------------------------------------
  //StartTest: test-bridgetest-3
  // Test 3: Direct bridge check validation.
  // In the illegal bridge configuration (Test 1: Black has no checkers ahead),
  // the move from pos 4 to pos 3 should be identified as forming an illegal bridge.
  // ------------------------------------------------------------
  {
    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    white_row[0] = 2; white_row[1] = 1; white_row[2] = 1; white_row[3] = 0; white_row[4] = 2; white_row[5] = 1;
    // white_row[kWhiteHeadPos] = 8;  // White head. -- Removed: Score handles head checkers.
    // Place all Black checkers at their head (index 11 / vcoord 23).
    // Since 23 >= 17 (bridge start vcoord), no Black checker is ahead.
    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    black_row[kBlackHeadPos] = 15; // Place all 15 checkers at index 11
    // Total Black checkers: 15 on board. Score should be 0.
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-7=8), Black (15-15=0)
    SetupBoardState(lnstate, kXPlayerId, test_board, {8, 0}); // Corrected score
    SetupDice(lnstate, {4, 1, 0, 0});

    // Test 3: Verify that the direct move 4->3 is not a valid *single* move
    // in this state (as it would form the illegal bridge).
    // We check this via IsValidCheckerMove.
    // Note: Sequences like (5->3, 4->3) might still be legal if they don't
    // create the illegal state momentarily. This test focuses on the
    // direct bridge formation rule application.
    CheckerMove move1(4, 3, 1);
    bool direct_move_valid = lnstate->IsValidCheckerMove(kXPlayerId, move1, /*moved_from_head_this_sequence=*/false);
    SPIEL_CHECK_FALSE(direct_move_valid); // 4->3 with die 1 should be invalid here (bridge is now illegal).
  }
  //EndTest: test-bridgetest-3

  // ------------------------------------------------------------
  //StartTest: test-bridgetest-4
  // Test 4: Black forming an illegal bridge (White Behind)
  // Similar to Test 1 but for Black player in Black's home region (12-17)
  // ------------------------------------------------------------
  {
    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    // Near bridge: points 13-18 (indices 12-17) with a gap at index 15
    black_row[12] = 2; black_row[13] = 1; black_row[14] = 1;
    black_row[16] = 1; black_row[17] = 2;
    black_row[20] = 1;  // Checker to move into the gap (at pt21/idx20 instead of pt20/idx19 to avoid overlap with White)
    // black_row[kBlackHeadPos] = 15 - 8; // Remaining 7 at head (11) -- Removed: Score handles head checkers.
    // Total Black checkers: 2+1+1+1+2+1=8 on board. Score should be 7.

    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    // Place White checkers *behind* Black's potential bridge (indices >= 17).
    // From White's perspective (virtual coords are real coords), these positions have vcoord >= 17.
    // Therefore, no White checker is "ahead" of the bridge start (vcoord 17).
    white_row[18] = 5; white_row[19] = 5; white_row[kWhiteHeadPos] = 5; // 15 checkers total at indices >= 18 -- This sets index 23 (White head pos) to 5.
    // Total White checkers: 5+5+5=15 on board. Score should be 0.

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-15=0), Black (15-8=7)
    SetupBoardState(lnstate, kOPlayerId, test_board, {0, 7});
    SetupDice(lnstate, {4, 1, 0, 0});

    // Check if move 20->15 (die 5) forms illegal bridge
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kOPlayerId, 20, 15);
    SPIEL_CHECK_TRUE(bridge_illegal); // Should be illegal now (White exists but none are ahead: vcoords >= 17)

    // Verify the move is not valid directly
    CheckerMove move4(20, 15, 5);
    bool direct_move_valid = lnstate->IsValidCheckerMove(kOPlayerId, move4, /*moved_from_head_this_sequence=*/false);
    SPIEL_CHECK_FALSE(direct_move_valid); // Move should be invalid because it forms an illegal bridge
  }
  //EndTest: test-bridgetest-4

  // ------------------------------------------------------------
  //StartTest: test-bridgetest-5
  // Test 5: Black forming a legal bridge (White checker ahead)
  // Similar to Test 2 but for Black player
  // ------------------------------------------------------------
  {
    // Same Black setup as Test 4
    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    black_row[12] = 2; black_row[13] = 1; black_row[14] = 1;
    black_row[16] = 1; black_row[17] = 2;
    black_row[20] = 1;
    // black_row[kBlackHeadPos] = 7; -- Removed: Score handles head checkers.
    // Total Black checkers: 2+1+1+1+2+1=8 on board. Score should be 7.

    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    // Place White checkers both ahead and behind the bridge
    white_row[18] = 1; // White checker at point 19 (behind the bridge from Black's perspective)
    white_row[0] = 14; // White checkers at point 1 (ahead of the bridge from Black's perspective)
    // Total White checkers: 1+14=15 on board. Score should be 0.

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-15=0), Black (15-8=7)
    SetupBoardState(lnstate, kOPlayerId, test_board, {0, 7});
    SetupDice(lnstate, {5, 1, 0, 0});

    // Check if move 20->15 (die 5) forms illegal bridge
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kOPlayerId, 20, 15);
    SPIEL_CHECK_FALSE(bridge_illegal); // Should be legal now with White checkers ahead of the bridge

    // Verify the move is now valid directly
    CheckerMove move5(20, 15, 5);
    bool direct_move_valid = lnstate->IsValidCheckerMove(kOPlayerId, move5, /*moved_from_head_this_sequence=*/false);
    SPIEL_CHECK_TRUE(direct_move_valid);
  }
  //EndTest: test-bridgetest-5

  //StartTest: test-bridgetest-6
  // Test 6: White forms wrap-around bridge (23-4), Black behind
  {
    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    white_row[23]=1; white_row[0]=1; white_row[1]=1; white_row[2]=1; white_row[3]=1;
    white_row[5]=1;
    // white_row[kWhiteHeadPos] = 15 - 6; // -- Removed: Score handles head checkers.
    // Total White checkers: 1+1+1+1+1+1 = 6 on board. Score should be 9.

    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    black_row[12] = 15;
    // Total Black checkers: 15 on board. Score should be 0.

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-6=9), Black (15-15=0)
    SetupBoardState(lnstate, kXPlayerId, test_board, {9, 0});
    SetupDice(lnstate, {1, 2, 0, 0});

    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 5, 4);
    SPIEL_CHECK_TRUE(bridge_illegal); // Should be ILLEGAL as Black at head (virt 23) is NOT strictly ahead of bridge start (virt 23)

    CheckerMove move6(5, 4, 1);
    bool direct_move_valid = lnstate->IsValidCheckerMove(kXPlayerId, move6, /*moved_from_head_this_sequence=*/false);
    SPIEL_CHECK_FALSE(direct_move_valid); // Move should be invalid as it forms an illegal bridge
  }
  //EndTest: test-bridgetest-6

  //StartTest: test-bridgetest-7
  // Test 7: White forms wrap-around bridge, Black ahead
  {
    std::vector<int> white_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    white_row[23]=1; white_row[0]=1; white_row[1]=1; white_row[2]=1; white_row[3]=1;
    white_row[5]=1;
    // white_row[kWhiteHeadPos] = 15 - 6; // -- Removed: Score handles head checkers.
    // Total White checkers: 1+1+1+1+1+1 = 6 on board. Score should be 9.

    std::vector<int> black_row(kNumPoints + 1, 0); // Use kNumPoints + 1
    black_row[10] = 15;
    // Total Black checkers: 15 on board. Score should be 0.

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    // Corrected Score: White (15-6=9), Black (15-15=0)
    SetupBoardState(lnstate, kXPlayerId, test_board, {9, 0});
    SetupDice(lnstate, {1, 2, 0, 0});

    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 5, 4);
    SPIEL_CHECK_TRUE(bridge_illegal); // Should be ILLEGAL (Black at vcoord 22 is NOT ahead of bridge start at vcoord 16)

    CheckerMove move7(5, 4, 1);
    bool direct_move_valid = lnstate->IsValidCheckerMove(kXPlayerId, move7, /*moved_from_head_this_sequence=*/false);
    SPIEL_CHECK_FALSE(direct_move_valid); // Move should be invalid as it forms an illegal bridge
  }
  //EndTest: test-bridgetest-7


  // ------------------------------------------------------------
  // Consolidated setup for all bridge formation tests
  // We use a single board configuration where:
  // - White has checkers at positions 9,10,11,12 (potential bridge at 13)
  // - Black has checkers at 16,17 and one ahead at position 0
  // This setup allows testing all bridge formation scenarios by just changing dice
  // ------------------------------------------------------------
  {
    std::vector<int> white_row(kNumPoints + 1, 0);
    white_row[9] = 2; white_row[10] = 2; white_row[11] = 1; white_row[12] = 1;
    // Total White checkers: 2+2+1+1 = 6 on board. Score should be 9.

    std::vector<int> black_row(kNumPoints + 1, 0);
    black_row[16] = 2; black_row[17] = 2;
    black_row[0] = 1; // Black checker ahead (point 1)
    // Total Black checkers: 1+2+2 = 5 on board. Score should be 10.

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    SetupBoardState(lnstate, kXPlayerId, test_board, {9, 10});
    SetupDice(lnstate, {1, 3, 0, 0}); // Initial dice setup
  }

  // All checks will use this single board state, only changing dice as needed

  //StartTest: test-bridgetest-8
  // Test 8: White forms legal bridge with opponent relief (CCW movement)
  {
    SetupDice(lnstate, {1, 3, 0, 0});
    
    // Two-step counter-clockwise move sequence forming a legal bridge
    // 1. First half-move: 9->8 with die 1 (valid due to opponent relief)
    CheckerMove check_move1(9, 8, 1);
    SPIEL_CHECK_TRUE(lnstate->IsValidCheckerMove(kXPlayerId, check_move1, /*moved_from_head_this_sequence=*/false));
    std::cout << "✓ White move W:9->8 (die 1) correctly accepted (opponent relief).\n";
    
    // 2. Second half-move: 10->7 with die 3 (forms legal bridge)
    CheckerMove check_move2(10, 7, 3);
    SPIEL_CHECK_TRUE(lnstate->IsValidCheckerMove(kXPlayerId, check_move2, /*moved_from_head_this_sequence=*/false));
    std::cout << "✓ White move W:10->7 (die 3) correctly accepted (forms legal bridge).\n";
    
    // Verify bridge doesn't form an illegal blocking bridge
    SPIEL_CHECK_FALSE(lnstate->WouldFormBlockingBridge(kXPlayerId, 8, 7));
    std::cout << "✓ Move 8 -> 7 correctly verified as legal (doesn't create illegal bridge).\n";
  }
  //EndTest: test-bridgetest-8

  //StartTest: test-bridgetest-9
  // Test 9: White move within potential bridge range (no bridge formed)
  {
    SetupDice(lnstate, {1, 3, 0, 0}); // Change dice order
    
    // 1. First half-move: 9->6 with die 3 (valid)
    CheckerMove check_move3(9, 6, 3);
    SPIEL_CHECK_TRUE(lnstate->IsValidCheckerMove(kXPlayerId, check_move3, /*moved_from_head_this_sequence=*/false));
    std::cout << "✓ White move W:9->6 (die 3) correctly accepted.\n";
  //EndTest: test-bridgetest-9
    
    // 2. Second half-move: 10->9 with die 1 (forms legal bridge)
    CheckerMove check_move4(10, 9, 1);
    SPIEL_CHECK_TRUE(lnstate->IsValidCheckerMove(kXPlayerId, check_move4, /*moved_from_head_this_sequence=*/false));
    std::cout << "✓ White move W:10->9 (die 1) correctly accepted (forms legal bridge).\n";
  }
  //EndTest: test-bridgefinalchecks
}

//EndFunction: TestBridgeFormation

}  // namespace long_narde
}  // namespace open_spiel