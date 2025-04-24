#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

//StartFunction: BearingOffBasicTest
void BearingOffBasicTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  //StartTest: test-bobt-1
  // All checkers in home: bearing off should be allowed.
  std::vector<std::vector<int>> test_board = {
    {3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0}
  };
  SetupBoardState(lnstate, kXPlayerId, test_board);
  SetupDice(lnstate, {6, 1, 0, 0});
  bool all_checkers_in_home = lnstate->AllInHome(kXPlayerId);
  SPIEL_CHECK_TRUE(all_checkers_in_home);
  //EndTest: test-bobt-1

  //StartTest: test-bobt-2
  // Not all checkers in home: bearing off should not be allowed.
  test_board = {
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0}
  };
  SetupBoardState(lnstate, kXPlayerId, test_board);
  SetupDice(lnstate, {6, 1, 0, 0});
  all_checkers_in_home = lnstate->AllInHome(kXPlayerId);
  SPIEL_CHECK_FALSE(all_checkers_in_home);
  //EndTest: test-bobt-2

  //StartTest: test-bobt-3
  // Black: all checkers in home, should be able to bear off.
  test_board = {
    {3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0}
  };
  SetupBoardState(lnstate, kOPlayerId, test_board);
  SetupDice(lnstate, {2, 2, 0, 0});
  all_checkers_in_home = lnstate->AllInHome(kOPlayerId);
  SPIEL_CHECK_TRUE(all_checkers_in_home);
  //EndTest: test-bobt-3
}

//StartFunction: BearingOffLogicTest
void BearingOffLogicTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  //StartTest: test-bolt-1
  // Test bearing off: die 1 from pos 1 should not bear off, die 3 from pos 2 should.
  std::vector<std::vector<int>> test_board = {
    {0, 1, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}
  };
  SetupBoardState(lnstate, kXPlayerId, test_board);
  SetupDice(lnstate, {1, 3, 0, 0});

  std::vector<Action> legal_actions = lnstate->LegalActions();

  bool can_bear_off_with_1 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.pos == 1 && move.die == 1 && lnstate->IsOff(kXPlayerId, move.to_pos)) {
        can_bear_off_with_1 = true;
        break;
      }
    }
    if (can_bear_off_with_1) break;
  }

  bool can_bear_off_with_3 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.pos == 2 && move.die == 3 && lnstate->IsOff(kXPlayerId, move.to_pos)) {
        can_bear_off_with_3 = true;
        break;
      }
    }
    if (can_bear_off_with_3) break;
  }

  SPIEL_CHECK_FALSE(can_bear_off_with_1);
  SPIEL_CHECK_TRUE(can_bear_off_with_3);
  //EndTest: test-bolt-1
}

//StartFunction: BearingOffFromPosition1Test
void BearingOffFromPosition1Test() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

    //StartTest: test-bofp1-1
    // Test bearing off from point 2 (index 1) with multiple checkers present; only allowed with higher die if no checkers on higher points.
    std::vector<std::vector<int>> test_board = {
      {13, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}
    };
    SetupBoardState(lnstate, kXPlayerId, test_board);
    SetupDice(lnstate, {1, 3, 0, 0});
  
    std::vector<Action> legal_actions = lnstate->LegalActions();
  
    bool can_bear_off_with_1 = false;
    bool can_bear_off_with_3 = false;
    bool has_pass = false;
  
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      if (moves.size() >= 1 && moves[0].pos == 1 && moves[0].die == 1 && lnstate->IsOff(kXPlayerId, moves[0].to_pos)) {
        can_bear_off_with_1 = true;
      }
      if (moves.size() >= 1 && moves[0].pos == 1 && moves[0].die == 3 && lnstate->IsOff(kXPlayerId, moves[0].to_pos)) {
        can_bear_off_with_3 = true;
      }
      if (moves.size() >= 1 && moves[0].pos == kPassPos) {
        has_pass = true;
      }
    }
  
    SPIEL_CHECK_FALSE(can_bear_off_with_1);
    SPIEL_CHECK_TRUE(can_bear_off_with_3);
    SPIEL_CHECK_FALSE(has_pass);
    //EndTest: test-bofp1-1
}

//StartFunction: BearingOffBlackTest
void BearingOffBlackTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

   //StartTest: test-bobtBlack-1
   // Black can only bear off if all checkers are in home (indices 12-17).
   std::vector<std::vector<int>> test_board = {
     {15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0}
   };
   SetupBoardState(lnstate, kOPlayerId, test_board);
   SetupDice(lnstate, {2, 3, 0, 0});
   SPIEL_CHECK_TRUE(lnstate->AllInHome(kOPlayerId));
   //EndTest: test-bobtBlack-1
}

//StartFunction: EndgameScoreTest
void EndgameScoreTest() {
  // Test Mars, Oin, and tie scoring.

  //StartTest: test-est-1
  // Mars: White has borne off all checkers, Black has none off.
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<std::vector<int>> mars_board = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}
  };
  SetupBoardState(lnstate, kXPlayerId, mars_board);
  SetupDice(lnstate, {1, 1, 1, 1});
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  std::vector<double> returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 2.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -2.0);
  //EndTest: test-est-1

  //StartTest: test-est-2
  // Oin: White has borne off all checkers, Black has some off.
  state = game->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<std::vector<int>> oin_board = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0}
  };
  SetupBoardState(lnstate, kXPlayerId, oin_board);
  SetupDice(lnstate, {1, 1, 1, 1});
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 1.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -1.0);
  //EndTest: test-est-2

  //StartTest: test-est-3
  // Tie: both players have borne off all checkers (winlosstie mode).
  std::shared_ptr<const Game> game_tie = LoadGame("long_narde(scoring_type=winlosstie_scoring)");
  state = game_tie->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<std::vector<int>> tie_board = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  };
  SetupBoardState(lnstate, kXPlayerId, tie_board);
  SetupDice(lnstate, {1, 1, 1, 1});
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 0.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], 0.0);
  //EndTest: test-est-3
}

//StartFunction: ScoringSystemTest
void ScoringSystemTest() {
  // Test scoring_type parameter parsing.

  //StartTest: test-sst-1
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  auto params = game->GetParameters();
  auto scoring_str = params.count("scoring_type") > 0 ?
    params.at("scoring_type").string_value() : "winloss_scoring";
  SPIEL_CHECK_EQ(scoring_str, "winloss_scoring");

  game = LoadGame("long_narde(scoring_type=winloss_scoring)");
  params = game->GetParameters();
  scoring_str = params.at("scoring_type").string_value();
  SPIEL_CHECK_EQ(scoring_str, "winloss_scoring");

  game = LoadGame("long_narde(scoring_type=winlosstie_scoring)");
  params = game->GetParameters();
  scoring_str = params.at("scoring_type").string_value();
  SPIEL_CHECK_EQ(scoring_str, "winlosstie_scoring");
  //EndTest: test-sst-1
}
//EndFunction: ScoringSystemTest

//StartFunction: SingleLegalMoveTestBlack
void SingleLegalMoveTestBlack() {
  std::cout << "\n=== Running SingleLegalMoveTestBlack ===\n";

  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  //StartTest: test-slmtb-1
  // Black checker outside home: only one legal move sequence should exist.
  std::vector<std::vector<int>> test_board(2, std::vector<int>(kNumPoints, 0));
  test_board[kOPlayerId][22] = 1; // The one checker outside home
  test_board[kOPlayerId][12] = 14; // All other 14 checkers at pt13 (idx12)
  test_board[kXPlayerId][21] = 1; // White blocking checker
  test_board[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer - 1; // Remaining White checkers
  // No scores needed as no checkers are borne off
  SetupBoardState(lnstate, kOPlayerId, test_board);
  SetupDice(lnstate, {1, 2, 0, 0});
  std::vector<Action> legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actions[0]);
  SPIEL_CHECK_EQ(moves.size(), 2);
  bool found_22_to_20_with_die2 = false;
  bool found_20_to_19_with_die1 = false;
  for (const auto& move : moves) {
    if (move.pos == 22 && move.die == 2 && move.to_pos == 20) {
      found_22_to_20_with_die2 = true;
    }
    if (move.pos == 20 && move.die == 1 && move.to_pos == 19) {
      found_20_to_19_with_die1 = true;
    }
  }
  SPIEL_CHECK_TRUE(found_22_to_20_with_die2);
  SPIEL_CHECK_TRUE(found_20_to_19_with_die1);
  std::cout << "✓ SingleLegalMoveTestBlack passed\n";
  //EndTest: test-slmtb-1
}
//EndFunction: SingleLegalMoveTestBlack

//StartFunction: BearingOffLogicTestBlackNearEnd
void BearingOffLogicTestBlackNearEnd() {
  std::cout << "\n=== Running BearingOffLogicTestBlackNearEnd ===\n";
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  //StartTest: test-boltbne-1
  std::vector<std::vector<int>> test_board(2, std::vector<int>(kNumPoints, 0));
  test_board[kOPlayerId][13] = 1;
  test_board[kOPlayerId][14] = 1;
  test_board[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  SetupBoardState(lnstate, kOPlayerId, test_board);
  SetupDice(lnstate, {5, 2, 0, 0});
  SPIEL_CHECK_TRUE(lnstate->AllInHome(kOPlayerId));
  std::vector<Action> legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty());
  bool can_bear_off_13_with_2 = false;
  bool can_bear_off_13_with_5 = false;
  bool can_bear_off_14_with_5 = false;
  bool can_bear_off_14_with_2 = false;
  bool can_move_14_to_12_with_2 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.pos == 13 && move.die == 2 && lnstate->IsOff(kOPlayerId, move.to_pos)) can_bear_off_13_with_2 = true;
      if (move.pos == 13 && move.die == 5 && lnstate->IsOff(kOPlayerId, move.to_pos)) can_bear_off_13_with_5 = true;
      if (move.pos == 14 && move.die == 5 && lnstate->IsOff(kOPlayerId, move.to_pos)) can_bear_off_14_with_5 = true;
      if (move.pos == 14 && move.die == 2 && lnstate->IsOff(kOPlayerId, move.to_pos)) can_bear_off_14_with_2 = true;
      if (move.pos == 14 && move.die == 2 && move.to_pos == 12) can_move_14_to_12_with_2 = true;
    }
  }
  SPIEL_CHECK_TRUE(can_bear_off_13_with_2);
  SPIEL_CHECK_FALSE(can_bear_off_13_with_5);
  SPIEL_CHECK_TRUE(can_bear_off_14_with_5);
  SPIEL_CHECK_FALSE(can_bear_off_14_with_2);
  SPIEL_CHECK_FALSE(can_move_14_to_12_with_2);
  std::cout << "✓ BearingOffLogicTestBlackNearEnd passed\n";
  //EndTest: test-boltbne-1
}

//StartFunction: CannotBearOffIfNotAllInHomeTest
void CannotBearOffIfNotAllInHomeTest() {
  std::cout << "\n=== Running CannotBearOffIfNotAllInHomeTest ===\n";
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  //StartTest: test-cboiah-1
  // Not all checkers in home: verify no bear-off is possible for either player.
  std::vector<std::vector<int>> test_board(2, std::vector<int>(kNumPoints, 0));
  test_board[kXPlayerId][0] = 5;
  test_board[kXPlayerId][1] = 5;
  test_board[kXPlayerId][2] = 4;
  test_board[kXPlayerId][6] = 1;
  test_board[kOPlayerId][12] = 5;
  test_board[kOPlayerId][13] = 5;
  test_board[kOPlayerId][14] = 4;
  test_board[kOPlayerId][18] = 1;
  SetupBoardState(lnstate, kXPlayerId, test_board);
  SetupDice(lnstate, {1, 6, 0, 0});
  SPIEL_CHECK_FALSE(lnstate->AllInHome(kXPlayerId));
  std::vector<Action> legal_actions_white = lnstate->LegalActions();
  bool white_first_move_is_bear_off = false;
  for (Action action : legal_actions_white) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    CheckerMove first_move = kPassMove;
    for(const auto& m : moves) {
        if (m.pos != kPassPos) {
            first_move = m;
            break;
        }
    }
    if (first_move.pos != kPassPos && lnstate->IsOff(kXPlayerId, first_move.to_pos)) {
      white_first_move_is_bear_off = true;
      break;
    }
  }
  SPIEL_CHECK_FALSE(white_first_move_is_bear_off);
  std::cout << "  White cannot start turn with bear off (as expected).\n";
  //EndTest: test-cboiah-1

  //StartTest: test-cboiah-2
  // Black: not all checkers in home, verify no bear-off is possible.
  test_board[kXPlayerId].assign(kNumPoints, 0);
  test_board[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  SetupBoardState(lnstate, kOPlayerId, test_board);
  SetupDice(lnstate, {1, 6, 0, 0});
  bool black_all_in_home = lnstate->AllInHome(kOPlayerId);
  if (!black_all_in_home) {
      std::vector<Action> legal_actions_black = lnstate->LegalActions();
      bool black_first_move_is_bear_off = false;
      for (Action action : legal_actions_black) {
        std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
        CheckerMove first_move = kPassMove;
        for(const auto& m : moves) {
            if (m.pos != kPassPos) {
                first_move = m;
                break;
            }
        }
        if (first_move.pos != kPassPos && lnstate->IsOff(kOPlayerId, first_move.to_pos)) {
          black_first_move_is_bear_off = true;
          break;
        }
      }
      SPIEL_CHECK_FALSE(black_first_move_is_bear_off);
      std::cout << "  Black cannot start turn with bear off (as expected).\n";
  } else {
      std::cout << "  Skipping Black bear-off check as AllInHome was true.\n";
  }
  std::cout << "✓ CannotBearOffIfNotAllInHomeTest passed\n";
  //EndTest: test-cboiah-2
}

}  // namespace

//StartFunction: TestEndgame
void TestEndgame() {
  //StartTest: test-endgame
  std::cout << "\n=== Testing Endgame Rules ===" << std::endl;

  std::cout << "\n=== Running BearingOffBasicTest ===\n";
  BearingOffBasicTest();
  std::cout << "✓ BearingOffBasicTest passed\n";

  std::cout << "\n=== Running BearingOffLogicTest ===\n";
  BearingOffLogicTest();
  std::cout << "✓ BearingOffLogicTest passed\n";

  std::cout << "\n=== Running BearingOffFromPosition1Test ===\n";
  BearingOffFromPosition1Test();
  std::cout << "✓ BearingOffFromPosition1Test passed\n";

  std::cout << "\n=== Running BearingOffBlackTest ===\n";
  BearingOffBlackTest();
  std::cout << "✓ BearingOffBlackTest passed\n";

  std::cout << "\n=== Running EndgameScoreTest ===\n";
  EndgameScoreTest();
  std::cout << "✓ EndgameScoreTest passed\n";

  std::cout << "\n=== Running ScoringSystemTest ===\n";
  ScoringSystemTest();
  std::cout << "✓ ScoringSystemTest passed\n";

  CannotBearOffIfNotAllInHomeTest();

  std::cout << "\n=== Running SingleLegalMoveTestBlack ===\n";
  SingleLegalMoveTestBlack();
  std::cout << "✓ SingleLegalMoveTestBlack passed\n";

  std::cout << "\n=== Running BearingOffLogicTestBlackNearEnd ===\n";
  BearingOffLogicTestBlackNearEnd();
  std::cout << "✓ BearingOffLogicTestBlackNearEnd passed\n";

  std::cout << "✓ All endgame tests passed\n";
  //EndTest: test-endgame
}
//EndFunction: TestEndgame

}  // namespace long_narde
}  // namespace open_spiel