#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel
{
  namespace long_narde
  {

    // StartFunction: TestBridgeFormation
    void TestBridgeFormation()
    {
      std::shared_ptr<const Game> game = LoadGame("long_narde");

      // StartTest: test-bridgetest-1
      //  Bridge formation is legal if opponent has no checkers on board.
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> white_row(kNumPoints, 0);
        white_row[0] = 2;
        white_row[1] = 1;
        white_row[2] = 1;
        white_row[3] = 0;
        white_row[4] = 2;
        white_row[5] = 1;
        std::vector<int> black_row(kNumPoints, 0);
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board);
        SetupDice(lnstate, {4, 1, 0, 0});
        bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
        SPIEL_CHECK_FALSE(bridge_illegal);
      }
      // EndTest: test-bridgetest-1

      // StartTest: test-bridgetest-2
      //  Bridge formation is legal if opponent has a checker ahead of the bridge.
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> white_row(kNumPoints, 0);
        white_row[0] = 2;
        white_row[1] = 1;
        white_row[2] = 1;
        white_row[3] = 0;
        white_row[4] = 2;
        white_row[5] = 1;
        std::vector<int> black_row(kNumPoints, 0);
        black_row[6] = 14;
        black_row[18] = 1;
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board);
        SetupDice(lnstate, {4, 1, 0, 0});
        bool bridge_illegal = lnstate->WouldFormBlockingBridge(kXPlayerId, 4, 3);
        SPIEL_CHECK_FALSE(bridge_illegal);
      }
      // EndTest: test-bridgetest-2

      // StartTest: test-bridgetest-3
      //  Bridge formation is illegal if opponent is trapped behind the bridge.
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> white_row(kNumPoints, 0);
        white_row[0] = 2;
        white_row[1] = 1;
        white_row[2] = 1;
        white_row[3] = 0;
        white_row[4] = 2;
        white_row[5] = 1;
        std::vector<int> black_row(kNumPoints, 0);
        black_row[kBlackHeadPos] = 15;
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board);
        SetupDice(lnstate, {4, 1, 0, 0});
        LongNardeCheckerMove move1(4, 3, 1);
        bool direct_move_valid = lnstate->LongNardeIsValidCheckerMove(kXPlayerId, move1, false);
        SPIEL_CHECK_FALSE(direct_move_valid);
      }
      // EndTest: test-bridgetest-3

      // StartTest: test-bridgetest-4
      //  Black cannot form a bridge if all White checkers are behind it.
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> black_row(kNumPoints, 0);
        black_row[12] = 2;
        black_row[13] = 1;
        black_row[14] = 1;
        black_row[16] = 1;
        black_row[17] = 2;
        black_row[20] = 8;
        std::vector<int> white_row(kNumPoints, 0);
        white_row[18] = 5;
        white_row[19] = 5;
        white_row[kWhiteHeadPos] = 5;
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kOPlayerId, test_board);
        SetupDice(lnstate, {5, 1, 0, 0});
        bool bridge_illegal = lnstate->WouldFormBlockingBridge(kOPlayerId, 20, 15);
        SPIEL_CHECK_TRUE(bridge_illegal);
        LongNardeCheckerMove move4(20, 15, 5);
        bool direct_move_valid = lnstate->LongNardeIsValidCheckerMove(kOPlayerId, move4, false);
        SPIEL_CHECK_FALSE(direct_move_valid);
      }
      // EndTest: test-bridgetest-4

      // StartTest: test-bridgetest-5
      //  Black can form a bridge if White has a checker ahead of the bridge.
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> black_row(kNumPoints, 0);
        black_row[12] = 2;
        black_row[13] = 1;
        black_row[14] = 1;
        black_row[16] = 1;
        black_row[17] = 2;
        black_row[20] = 8;
        std::vector<int> white_row(kNumPoints, 0);
        white_row[18] = 1;
        white_row[0] = 14;
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kOPlayerId, test_board);
        SetupDice(lnstate, {5, 1, 0, 0});
        bool bridge_illegal = lnstate->WouldFormBlockingBridge(kOPlayerId, 20, 15);
        SPIEL_CHECK_FALSE(bridge_illegal);
        LongNardeCheckerMove move5(20, 15, 5);
        bool direct_move_valid = lnstate->LongNardeIsValidCheckerMove(kOPlayerId, move5, false);
        SPIEL_CHECK_TRUE(direct_move_valid);
      }
      // EndTest: test-bridgetest-5

      // StartTest: test-bridgetest-6
      //  White cannot form a wrap-around bridge if Black is behind.
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> white_row(kNumPoints, 0);
        white_row[23] = 10;
        white_row[0] = 1;
        white_row[1] = 1;
        white_row[2] = 1;
        white_row[3] = 1;
        white_row[5] = 1;
        std::vector<int> black_row(kNumPoints, 0);
        black_row[12] = 15;
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board);
        SetupDice(lnstate, {1, 1, 0, 0});
        LongNardeCheckerMove move1(23, 22, 1);
        SPIEL_CHECK_FALSE(lnstate->LongNardeIsValidCheckerMove(kXPlayerId, move1, false));
        LongNardeCheckerMove move2(5, 4, 1);
        SPIEL_CHECK_FALSE(lnstate->LongNardeIsValidCheckerMove(kXPlayerId, move2, false));
        std::vector<int> white_row_seq = white_row;
        white_row_seq[3] -= 1;
        white_row_seq[2] += 1;
        white_row_seq[23] -= 1;
        white_row_seq[22] += 1;
        white_row_seq[5] -= 1;
        white_row_seq[4] += 1;
        std::vector<std::vector<int>> test_board_seq = {white_row_seq, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board_seq);
        SetupDice(lnstate, {1, 1, 0, 0});
        LongNardeCheckerMove move3(4, 3, 1);
        SPIEL_CHECK_FALSE(lnstate->LongNardeIsValidCheckerMove(kXPlayerId, move3, true));
        std::vector<int> white_row_seq2 = white_row_seq;
        white_row_seq2[4] -= 1;
        white_row_seq2[3] += 1;
        std::vector<std::vector<int>> test_board_seq2 = {white_row_seq2, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board_seq2);
        SetupDice(lnstate, {1, 1, 0, 0});
        LongNardeCheckerMove move4(23, 22, 1);
        SPIEL_CHECK_FALSE(lnstate->LongNardeIsValidCheckerMove(kXPlayerId, move4, true));
      }
      // EndTest: test-bridgetest-6

      // StartTest: test-bridgetest-8
      //  White forms a legal bridge with opponent relief (two-step sequence).
      {
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());
        std::vector<int> white_row(kNumPoints, 0);
        white_row[9] = 2;
        white_row[10] = 2;
        white_row[11] = 1;
        white_row[12] = 10;
        std::vector<int> black_row(kNumPoints, 0);
        black_row[0] = 11;
        black_row[16] = 2;
        black_row[17] = 2;
        std::vector<std::vector<int>> test_board = {white_row, black_row};
        SetupBoardState(lnstate, kXPlayerId, test_board);
        SetupDice(lnstate, {1, 3, 0, 0});
        LongNardeCheckerMove check_move1(9, 8, 1);
        SPIEL_CHECK_TRUE(lnstate->LongNardeIsValidCheckerMove(kXPlayerId, check_move1, false));
        LongNardeCheckerMove check_move2(10, 7, 3);
        SPIEL_CHECK_TRUE(lnstate->LongNardeIsValidCheckerMove(kXPlayerId, check_move2, false));
        SPIEL_CHECK_FALSE(lnstate->WouldFormBlockingBridge(kXPlayerId, 8, 7));
        std::cout << "✓ Move 8 -> 7 correctly verified as legal (doesn't create illegal bridge).\n";
      }
      // EndTest: test-bridgetest-8

      // EndTest: test-bridgefinalchecks
    }

    // EndFunction: TestBridgeFormation

  } // namespace long_narde
} // namespace open_spiel