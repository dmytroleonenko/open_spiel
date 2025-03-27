#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {

void TestBridgeFormation() {
  // Load game and create state.
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // ------------------------------------------------------------
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
    std::vector<int> white_row = {2, 1, 1, 0, 2, 1};
    // Fill indices 6 through 22 with 0.
    white_row.resize(23, 0);
    // Set head (index 23) to 15 - (2+1+1+0+2+1) = 15 - 7 = 8.
    if (white_row.size() < 24) {
      white_row.push_back(8);
    } else {
      white_row[23] = 8;
    }
    // Total White checkers: 2+1+1+0+2+1+8 = 15.
    
    // Black row: all 24 positions zero.
    std::vector<int> black_row(24, 0);
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {1, 2};  // Die of 1 will be used.
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);

    // Simulate move from White's pos 4 to pos 3.
    // This move forms a 6-block, but is LEGAL because Black has no checkers on board.
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
    SPIEL_CHECK_FALSE(bridge_illegal); // Must be false (legal) if opponent has no checkers.
  }

  // ------------------------------------------------------------
  // Test 2: With opponent relief, the same move should be legal.
  // We use the same White configuration as above.
  // For Black, we place 14 checkers at one location (say index 2) and 1 checker at index 7
  // (so that at least one Black checker is ahead of White's home board).
  // ------------------------------------------------------------
  {
    std::vector<int> white_row = {2, 1, 1, 0, 2, 1};
    white_row.resize(23, 0);
    white_row.push_back(8);  // White head: 8 checkers.
    std::vector<int> black_row(24, 0);
    black_row[2] = 14;
    black_row[7] = 1;
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {1, 2};
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);

    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
    SPIEL_CHECK_FALSE(bridge_illegal);
  }

  // ------------------------------------------------------------
  // Test 3: Direct bridge check validation.
  // In the illegal bridge configuration (Test 1: Black has no checkers ahead),
  // the move from pos 4 to pos 3 should be identified as forming an illegal bridge.
  // ------------------------------------------------------------
  {
    std::vector<int> white_row = {2, 1, 1, 0, 2, 1};
    white_row.resize(23, 0);
    white_row.push_back(8);  // White head.
    // Place all Black checkers at their head (index 11 / vcoord 23).
    // Since 23 >= 17 (bridge start vcoord), no Black checker is ahead.
    std::vector<int> black_row(24, 0);
    black_row[kBlackHeadPos] = 15; // Place all 15 checkers at index 11
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {1, 2};
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);

    // Test 3: Verify that the direct move 4->3 is not a valid *single* move
    // in this state (as it would form the illegal bridge).
    // We check this via IsValidCheckerMove.
    // Note: Sequences like (5->3, 4->3) might still be legal if they don't
    // create the illegal state momentarily. This test focuses on the
    // direct bridge formation rule application.
    bool direct_move_valid = lnstate->IsValidCheckerMove(kXPlayerId, 4, 3, 1, true);
    SPIEL_CHECK_FALSE(direct_move_valid); // 4->3 with die 1 should be invalid here (bridge is now illegal).
  }

  // ------------------------------------------------------------
  // Test 4: Black forming an illegal bridge 
  // Similar to Test 1 but for Black player in Black's home region (12-17)
  // ------------------------------------------------------------
  {
    std::vector<int> black_row(24, 0);
    // Near bridge: points 13-18 (indices 12-17) with a gap at index 15
    black_row[12] = 2; black_row[13] = 1; black_row[14] = 1; 
    black_row[16] = 1; black_row[17] = 2;
    black_row[19] = 1;  // Checker to move into the gap
    black_row[kBlackHeadPos] = 15 - 8; // Remaining 7 at head (11)

    std::vector<int> white_row(24, 0);
    // Place White checkers *behind* Black's potential bridge (indices >= 17).
    // From White's perspective (virtual coords are real coords), these positions have vcoord >= 17.
    // Therefore, no White checker is "ahead" of the bridge start (vcoord 17).
    white_row[18] = 5; white_row[19] = 5; white_row[23] = 5; // 15 checkers total at indices >= 18

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {4, 1}; // Use die 4 to move 19->15
    lnstate->SetState(kOPlayerId, false, dice, {0, 0}, test_board);

    // Check if move 19->15 (die 4) forms illegal bridge
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kOPlayerId, 19, 15);
    SPIEL_CHECK_TRUE(bridge_illegal); // Should be illegal now (White exists but none are ahead: vcoords >= 17)

    // Verify the move is not valid directly
    bool direct_move_valid = lnstate->IsValidCheckerMove(kOPlayerId, 19, 15, 4, true);
    SPIEL_CHECK_FALSE(direct_move_valid); // Move should be invalid because it forms an illegal bridge
  }

  // ------------------------------------------------------------
  // Test 5: Black forming a legal bridge (White checker ahead)
  // Similar to Test 2 but for Black player
  // ------------------------------------------------------------
  {
    // Same Black setup as Test 4
    std::vector<int> black_row(24, 0);
    black_row[12] = 2; black_row[13] = 1; black_row[14] = 1; 
    black_row[16] = 1; black_row[17] = 2;
    black_row[19] = 1;
    black_row[kBlackHeadPos] = 7;

    std::vector<int> white_row(24, 0);
    // Place one White checker ahead of the bridge (>=18)
    white_row[18] = 1; // White checker at point 19 (ahead of Black's home region)
    white_row[0] = 14; // Rest of White's checkers behind

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {4, 1};
    lnstate->SetState(kOPlayerId, false, dice, {0, 0}, test_board);

    // Check if move 19->15 (die 4) forms illegal bridge
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kOPlayerId, 19, 15);
    SPIEL_CHECK_FALSE(bridge_illegal); // Should be legal now with White checker ahead

    // Verify the move is now valid directly
    bool direct_move_valid = lnstate->IsValidCheckerMove(kOPlayerId, 19, 15, 4, true);
    SPIEL_CHECK_TRUE(direct_move_valid);
  }

  // Test 6: White forms wrap-around bridge (23-4), Black behind
  {
    std::vector<int> white_row(24, 0);
    white_row[23]=1; white_row[0]=1; white_row[1]=1; white_row[2]=1; white_row[3]=1;
    white_row[5]=1;
    white_row[kWhiteHeadPos] = 15 - 6;

    std::vector<int> black_row(24, 0);
    black_row[12] = 15;

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {1, 2};
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);

    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 5, 4);
    SPIEL_CHECK_FALSE(bridge_illegal); // Should be legal as Black is ahead (virt 0 < virt 23)

    bool direct_move_valid = lnstate->IsValidCheckerMove(kXPlayerId, 5, 4, 1, true);
    SPIEL_CHECK_TRUE(direct_move_valid); // Move should be valid now
  }

  // Test 7: White forms wrap-around bridge, Black ahead
  {
    std::vector<int> white_row(24, 0);
    white_row[23]=1; white_row[0]=1; white_row[1]=1; white_row[2]=1; white_row[3]=1;
    white_row[5]=1;
    white_row[kWhiteHeadPos] = 15 - 6;

    std::vector<int> black_row(24, 0);
    black_row[10] = 15;

    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {1, 2};
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);

    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 5, 4);
    SPIEL_CHECK_TRUE(bridge_illegal); // Should be ILLEGAL (Black at vcoord 22 is NOT ahead of bridge start at vcoord 16)

    bool direct_move_valid = lnstate->IsValidCheckerMove(kXPlayerId, 5, 4, 1, true);
    SPIEL_CHECK_FALSE(direct_move_valid); // Move should be invalid as it forms an illegal bridge
  }
}

}  // namespace long_narde
}  // namespace open_spiel