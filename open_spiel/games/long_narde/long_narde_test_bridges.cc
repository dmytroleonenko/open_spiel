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
    bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
    SPIEL_CHECK_TRUE(bridge_illegal);
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
  // Test 3: Legal actions filtering.
  // In the illegal bridge configuration (Test 1: Black has no checkers ahead),
  // the move from pos 4 to pos 3 should not be among the legal actions.
  // ------------------------------------------------------------
  {
    std::vector<int> white_row = {2, 1, 1, 0, 2, 1};
    white_row.resize(23, 0);
    white_row.push_back(8);  // White head.
    // For Black, put all 15 checkers in a location within White's home board region,
    // e.g. index 2, so that no Black checker is ahead.
    std::vector<int> black_row(24, 0);
    black_row[2] = 15;
    std::vector<std::vector<int>> test_board = {white_row, black_row};
    std::vector<int> dice = {1, 2};
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);

    std::vector<Action> legal_actions = lnstate->LegalActions();
    bool found_illegal_bridge_move = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      for (const CheckerMove& move : moves) {
        // Look for a move from pos4 to pos3.
        if (move.pos == 4 && move.to_pos == 3) {
          found_illegal_bridge_move = true;
          break;
        }
      }
      if (found_illegal_bridge_move) break;
    }
    SPIEL_CHECK_FALSE(found_illegal_bridge_move);
  }
}

}  // namespace long_narde
}  // namespace open_spiel