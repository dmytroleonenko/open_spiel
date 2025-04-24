#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/long_narde/long_narde.h"

namespace open_spiel {
namespace long_narde {
namespace {

//StartFunction: ActionEncodingTest
void ActionEncodingTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");

  // Get the kNumDistinctActions constant - using NumDistinctActions() method
  int kNumDistinctActions;
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    kNumDistinctActions = lnstate->NumDistinctActions();
  }

  // Prepare initial and modified boards for reuse
  std::vector<std::vector<int>> initial_board(2, std::vector<int>(kNumPoints, 0));
  initial_board[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  initial_board[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
  std::vector<std::vector<int>> modified_board(2, std::vector<int>(kNumPoints, 0));
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    SetupBoardState(lnstate, kXPlayerId, initial_board);
    SetupDice(lnstate, {5, 3, 0, 0});
    for (int pos = 0; pos < kNumPoints; ++pos) {
      modified_board[kXPlayerId][pos] = lnstate->board_[kXPlayerId][pos];
      modified_board[kOPlayerId][pos] = lnstate->board_[kOPlayerId][pos];
    }
    modified_board[kXPlayerId][kWhiteHeadPos] -= 2;
    modified_board[kXPlayerId][14] += 1;
    modified_board[kXPlayerId][19] += 1;
  }

  //StartTest: test-actionencodingtest-1
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    SetupBoardState(lnstate, kXPlayerId, modified_board);
    SetupDice(lnstate, {5, 3, 0, 0});
    std::vector<CheckerMove> test_moves = {
      {14, lnstate->GetToPos(kXPlayerId, 14, 5), 5},
      {19, lnstate->GetToPos(kXPlayerId, 19, 3), 3}
    };
    Action action = lnstate->CheckerMovesToSpielMove(test_moves);
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kNumDistinctActions);
    std::vector<CheckerMove> decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    SPIEL_CHECK_EQ(decoded_moves.size(), 2);
    bool first_move_found = false;
    bool second_move_found = false;
    for (const CheckerMove& move : decoded_moves) {
      if (move.pos == test_moves[0].pos && move.die == test_moves[0].die) first_move_found = true;
      if (move.pos == test_moves[1].pos && move.die == test_moves[1].die) second_move_found = true;
    }
    SPIEL_CHECK_TRUE(first_move_found);
    SPIEL_CHECK_TRUE(second_move_found);
  }
  //EndTest: test-actionencodingtest-1

  //StartTest: test-actionencodingtest-2
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    SetupBoardState(lnstate, kXPlayerId, modified_board);
    SetupDice(lnstate, {5, 3, 0, 0});
    std::vector<CheckerMove> pass_moves = {
      {kPassPos, kPassPos, 5},
      {kPassPos, kPassPos, 3}
    };
    Action action = lnstate->CheckerMovesToSpielMove(pass_moves);
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kNumDistinctActions);
    std::vector<CheckerMove> decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    SPIEL_CHECK_EQ(decoded_moves.size(), 2);
    bool first_pass_found = false;
    bool second_pass_found = false;
    for (const CheckerMove& move : decoded_moves) {
      if (move.pos == kPassPos && move.die == 5) first_pass_found = true;
      if (move.pos == kPassPos && move.die == 3) second_pass_found = true;
    }
    SPIEL_CHECK_TRUE(first_pass_found);
    SPIEL_CHECK_TRUE(second_pass_found);
  }
  //EndTest: test-actionencodingtest-2

  //StartTest: test-actionencodingtest-3
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    SetupBoardState(lnstate, kXPlayerId, modified_board);
    SetupDice(lnstate, {3, 5, 0, 0});
    std::vector<CheckerMove> test_moves = {
      {14, lnstate->GetToPos(kXPlayerId, 14, 5), 5},
      {19, lnstate->GetToPos(kXPlayerId, 19, 3), 3}
    };
    Action action_low_roll = lnstate->CheckerMovesToSpielMove(test_moves);
    SPIEL_CHECK_GE(action_low_roll, 0);
    SPIEL_CHECK_LT(action_low_roll, kNumDistinctActions);
    std::vector<CheckerMove> decoded_moves_low_roll = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action_low_roll);
    SPIEL_CHECK_EQ(decoded_moves_low_roll.size(), 2);
    bool first_move_found = false;
    bool second_move_found = false;
    for (const CheckerMove& move : decoded_moves_low_roll) {
      if (move.pos == test_moves[0].pos && move.die == test_moves[0].die) first_move_found = true;
      if (move.pos == test_moves[1].pos && move.die == test_moves[1].die) second_move_found = true;
    }
    SPIEL_CHECK_TRUE(first_move_found);
    SPIEL_CHECK_TRUE(second_move_found);
  }
  //EndTest: test-actionencodingtest-3

  //StartTest: test-actionencodingtest-4
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> doubles_board(2, std::vector<int>(kNumPoints, 0));
    doubles_board[kXPlayerId][23] = 1;
    doubles_board[kXPlayerId][22] = 1;
    doubles_board[kXPlayerId][21] = 1;
    doubles_board[kXPlayerId][20] = 1;
    doubles_board[kXPlayerId][19] = 11;
    doubles_board[kOPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
    SetupBoardState(lnstate, kXPlayerId, doubles_board);
    SetupDice(lnstate, {2, 2, 2, 2});
    std::vector<CheckerMove> doubles_moves = {
      {23, lnstate->GetToPos(kXPlayerId, 23, 2), 2},
      {22, lnstate->GetToPos(kXPlayerId, 22, 2), 2},
      {21, lnstate->GetToPos(kXPlayerId, 21, 2), 2},
      {20, lnstate->GetToPos(kXPlayerId, 20, 2), 2}
    };
    Action doubles_action = lnstate->CheckerMovesToSpielMove(doubles_moves);
    SPIEL_CHECK_GE(doubles_action, 0);
    SPIEL_CHECK_LT(doubles_action, kNumDistinctActions);
    std::vector<CheckerMove> decoded_doubles_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, doubles_action);
    SPIEL_CHECK_GE(decoded_doubles_moves.size(), 4);
    int moves_matched = 0;
    for (size_t i = 0; i < doubles_moves.size(); ++i) {
      bool match_found = false;
      for (size_t j = 0; j < decoded_doubles_moves.size(); ++j) {
        if (doubles_moves[i].pos == decoded_doubles_moves[j].pos && decoded_doubles_moves[j].die == 2) {
          match_found = true;
          break;
        }
      }
      if (match_found) moves_matched++;
    }
    SPIEL_CHECK_EQ(moves_matched, 4);
  }
  //EndTest: test-actionencodingtest-4

  //StartTest: test-actionencodingtest-5
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> doubles_board(2, std::vector<int>(kNumPoints, 0));
    doubles_board[kXPlayerId][23] = 1;
    doubles_board[kXPlayerId][22] = 1;
    doubles_board[kXPlayerId][21] = 1;
    doubles_board[kXPlayerId][20] = 1;
    doubles_board[kXPlayerId][19] = 11;
    doubles_board[kOPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
    SetupBoardState(lnstate, kXPlayerId, doubles_board);
    SetupDice(lnstate, {2, 2, 2, 2});
    std::vector<CheckerMove> doubles_moves_with_pass = {
      {23, lnstate->GetToPos(kXPlayerId, 23, 2), 2},
      {22, lnstate->GetToPos(kXPlayerId, 22, 2), 2},
      {21, lnstate->GetToPos(kXPlayerId, 21, 2), 2},
      kPassMove
    };
    Action doubles_pass_action = lnstate->CheckerMovesToSpielMove(doubles_moves_with_pass);
    SPIEL_CHECK_GE(doubles_pass_action, 0);
    SPIEL_CHECK_LT(doubles_pass_action, kNumDistinctActions);
    std::vector<CheckerMove> decoded_doubles_pass_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, doubles_pass_action);
    SPIEL_CHECK_GE(decoded_doubles_pass_moves.size(), 4);
    std::map<int, int> matched_pos_count;
    int passes_found = 0;
    for(const auto& decoded_move : decoded_doubles_pass_moves) {
      if (decoded_move.pos != kPassPos) {
        for (size_t i = 0; i < 3; ++i) {
          if (doubles_moves_with_pass[i].pos == decoded_move.pos && decoded_move.die == 2) {
            matched_pos_count[decoded_move.pos]++;
            break;
          }
        }
      } else {
        passes_found++;
      }
    }
    SPIEL_CHECK_EQ(matched_pos_count.size(), 3);
    SPIEL_CHECK_GE(passes_found, 1);
  }
  //EndTest: test-actionencodingtest-5

  //StartTest: test-actionencodingtest-6
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    SetupBoardState(lnstate, kXPlayerId, modified_board);
    SetupDice(lnstate, {4, 1, 0, 0});
    std::vector<CheckerMove> single_move_pass = {
      {14, lnstate->GetToPos(kXPlayerId, 14, 4), 4},
      {kPassPos, kPassPos, 1}
    };
    Action action_single_pass = lnstate->CheckerMovesToSpielMove(single_move_pass);
    SPIEL_CHECK_GE(action_single_pass, 0);
    std::vector<CheckerMove> decoded_single_pass = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action_single_pass);
    SPIEL_CHECK_EQ(decoded_single_pass.size(), 2);
    bool move_14_d4_found = false;
    bool pass_d1_found = false;
    for (const auto& move : decoded_single_pass) {
      if (move.pos == 14 && move.die == 4) move_14_d4_found = true;
      if (move.pos == kPassPos && move.die == 1) pass_d1_found = true;
    }
    SPIEL_CHECK_TRUE(move_14_d4_found);
    SPIEL_CHECK_TRUE(pass_d1_found);
  }
  //EndTest: test-actionencodingtest-6

  //StartTest: test-actionencodingtest-7
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    SetupBoardState(lnstate, kXPlayerId, modified_board);
    SetupDice(lnstate, {6, 5, 0, 0});
    std::vector<CheckerMove> double_pass_moves = {
      {kPassPos, kPassPos, 6},
      {kPassPos, kPassPos, 5}
    };
    Action action_double_pass = lnstate->CheckerMovesToSpielMove(double_pass_moves);
    SPIEL_CHECK_GE(action_double_pass, 0);
    std::vector<CheckerMove> decoded_double_pass = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action_double_pass);
    SPIEL_CHECK_EQ(decoded_double_pass.size(), 2);
    bool pass_d6_found = false;
    bool pass_d5_found = false;
    for (const auto& move : decoded_double_pass) {
      if (move.pos == kPassPos && move.die == 6) pass_d6_found = true;
      if (move.pos == kPassPos && move.die == 5) pass_d5_found = true;
    }
    SPIEL_CHECK_TRUE(pass_d6_found);
    SPIEL_CHECK_TRUE(pass_d5_found);
  }
  //EndTest: test-actionencodingtest-7
}
//EndFunction: ActionEncodingTest

//StartFunction: SingleLegalMoveTest
//StartTest: test-singlelegalmovetest-1
void SingleLegalMoveTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> test_board(2, std::vector<int>(kNumPoints, 0));
    test_board[kXPlayerId][5] = 1;
    test_board[kXPlayerId][23] = 14;
    test_board[kOPlayerId][0] = 2;
    test_board[kOPlayerId][20] = 1;
    test_board[kOPlayerId][21] = 1;
    test_board[kOPlayerId][11] = 11;
    SetupBoardState(lnstate, kXPlayerId, test_board);
    SetupDice(lnstate, {2, 3, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    bool found_correct_move = false;
    if (!legal_actions.empty()) {
      Action the_action = legal_actions[0];
      std::vector<CheckerMove> decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, the_action);
      for(const auto& move : decoded_moves) {
        if (move.pos != kPassPos) {
          if (move.pos == 5 && move.die == 3 && move.to_pos == 2) {
            found_correct_move = true;
            std::cout << "Found correct higher die move {Pt 6 -> Pt 3 (d3)}" << std::endl;
          }
          SPIEL_CHECK_FALSE(move.pos == 5 && move.die == 2 && move.to_pos == 3);
          break;
        }
      }
    }
    SPIEL_CHECK_TRUE(found_correct_move);
  }
}
//EndTest: test-singlelegalmovetest-1
//EndFunction: SingleLegalMoveTest

//StartFunction: ConsecutiveMovesTest
//StartTest: test-consecutivemovestest-1
void ConsecutiveMovesTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    lnstate->ApplyAction(15);
    auto first_turn_legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(first_turn_legal_actions.empty());
    Action first_action = first_turn_legal_actions[0];
    lnstate->ApplyAction(first_action);
    SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
    lnstate->ApplyAction(0);
    SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kOPlayerId);
    auto black_legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(black_legal_actions.empty());
    lnstate->ApplyAction(black_legal_actions[0]);
    SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
  }
}
//EndTest: test-consecutivemovestest-1
//EndFunction: ConsecutiveMovesTest

//StartFunction: UndoRedoTest
//StartTest: test-undoredotest-1
void UndoRedoTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> mid_game_board(2, std::vector<int>(kNumPoints, 0));
    mid_game_board[kXPlayerId][3] = 2;
    mid_game_board[kXPlayerId][5] = 3;
    mid_game_board[kXPlayerId][8] = 1;
    mid_game_board[kXPlayerId][10] = 2;
    mid_game_board[kXPlayerId][14] = 2;
    mid_game_board[kXPlayerId][17] = 3;
    mid_game_board[kXPlayerId][20] = 2;
    mid_game_board[kOPlayerId][1] = 3;
    mid_game_board[kOPlayerId][6] = 2;
    mid_game_board[kOPlayerId][9] = 2;
    mid_game_board[kOPlayerId][12] = 1;
    mid_game_board[kOPlayerId][15] = 2;
    mid_game_board[kOPlayerId][18] = 2;
    mid_game_board[kOPlayerId][22] = 2;
    SetupBoardState(lnstate, kXPlayerId, mid_game_board);
    SetupDice(lnstate, {4, 2, 0, 0});
    SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
    std::vector<std::vector<int>> board_before;
    for (int p = 0; p < 2; ++p) {
      std::vector<int> player_board;
      for (int i = 0; i < kNumPoints; ++i) {
        player_board.push_back(lnstate->board(p, i));
      }
      board_before.push_back(player_board);
    }
    std::vector<Action> legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    Action action_to_apply = legal_actions[0];
    lnstate->ApplyAction(action_to_apply);
    bool board_changed = false;
    for (int p = 0; p < 2; ++p) {
      for (int i = 0; i < kNumPoints; ++i) {
        if (lnstate->board(p, i) != board_before[p][i]) {
          board_changed = true;
          break;
        }
      }
      if (board_changed) break;
    }
    SPIEL_CHECK_TRUE(board_changed);
    lnstate->UndoAction(kXPlayerId, action_to_apply);
    bool board_restored = true;
    for (int p = 0; p < 2; ++p) {
      for (int i = 0; i < kNumPoints; ++i) {
        if (lnstate->board(p, i) != board_before[p][i]) {
          board_restored = false;
          break;
        }
      }
      if (!board_restored) break;
    }
    SPIEL_CHECK_TRUE(board_restored);
  }
}
//EndTest: test-undoredotest-1
//EndFunction: UndoRedoTest

//StartFunction: PassMoveBehaviorTest
void PassMoveBehaviorTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  //StartTest: test-passmovebehaviortest-1
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> no_moves_board(2, std::vector<int>(kNumPoints, 0));
    no_moves_board[kXPlayerId][4] = 1;
    no_moves_board[kOPlayerId][1] = 1;
    no_moves_board[kOPlayerId][3] = 1;
    no_moves_board[kOPlayerId][5] = 1;
    no_moves_board[kOPlayerId][kBlackHeadPos] += 12;
    SetupBoardState(lnstate, kXPlayerId, no_moves_board);
    SetupDice(lnstate, {1, 3, 0, 0});
    std::vector<CheckerMove> expected_pass_encoding_1_3 = {{kPassPos, kPassPos, 1}, {kPassPos, kPassPos, 3}};
    Action expected_pass_action_1_3 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_1_3);
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    SPIEL_CHECK_EQ(legal_actions[0], expected_pass_action_1_3);
  }
  //EndTest: test-passmovebehaviortest-1
  //StartTest: test-passmovebehaviortest-2
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> valid_moves_board(2, std::vector<int>(kNumPoints, 0));
    valid_moves_board[kXPlayerId][1] = 1;
    valid_moves_board[kXPlayerId][3] = 1;
    valid_moves_board[kOPlayerId][12] = 15;
    SetupBoardState(lnstate, kXPlayerId, valid_moves_board);
    SetupDice(lnstate, {1, 3, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    std::vector<CheckerMove> expected_pass_encoding_1_3 = {{kPassPos, kPassPos, 1}, {kPassPos, kPassPos, 3}};
    Action expected_pass_action_1_3 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_1_3);
    bool pass_found = false;
    for (Action action : legal_actions) {
      if (action == expected_pass_action_1_3) {
        pass_found = true;
        break;
      }
    }
    SPIEL_CHECK_FALSE(pass_found);
    SPIEL_CHECK_GT(legal_actions.size(), 0);
  }
  //EndTest: test-passmovebehaviortest-2
  //StartTest: test-passmovebehaviortest-3
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> no_moves_doubles_board(2, std::vector<int>(kNumPoints, 0));
    no_moves_doubles_board[kXPlayerId][0] = 1;
    no_moves_doubles_board[kXPlayerId][5] = 1;
    no_moves_doubles_board[kXPlayerId][9] = 1;
    no_moves_doubles_board[kXPlayerId][kWhiteHeadPos] = 12;
    no_moves_doubles_board[kOPlayerId][2] = 1;
    no_moves_doubles_board[kOPlayerId][7] = 1;
    no_moves_doubles_board[kOPlayerId][kBlackHeadPos] = 13;
    SetupBoardState(lnstate, kOPlayerId, no_moves_doubles_board);
    SetupDice(lnstate, {2, 2, 2, 2});
    std::vector<CheckerMove> expected_pass_encoding_2_2 = {{kPassPos, kPassPos, 2}, {kPassPos, kPassPos, 2}};
    Action expected_pass_action_2_2 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_2_2);
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    SPIEL_CHECK_EQ(legal_actions[0], expected_pass_action_2_2);
  }
  //EndTest: test-passmovebehaviortest-3
}
//EndFunction: PassMoveBehaviorTest

//StartFunction: VerifyDicePlayBehavior
void VerifyDicePlayBehavior() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  //StartTest: test-vsdpb-1
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][8] = 1;
    board[kXPlayerId][3] = 1;
    board[kOPlayerId][0] = 1;
    board[kXPlayerId][3] += 13;
    board[kOPlayerId][kBlackHeadPos] += 14;
    SetupBoardState(lnstate, kXPlayerId, board);
    SetupDice(lnstate, {5, 3, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    bool found_correct_sequence = false;
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      SPIEL_CHECK_EQ(moves.size(), 2);
      bool found_8_3_d5 = false;
      bool found_pass_d3 = false;
      int non_pass_count = 0;
      for(const auto& move : moves) {
        if (move.pos != kPassPos) non_pass_count++;
        if (move.pos == 8 && move.to_pos == 3 && move.die == 5) found_8_3_d5 = true;
        else if (move.pos == kPassPos && move.die == 3) found_pass_d3 = true;
      }
      SPIEL_CHECK_EQ(non_pass_count, 1);
      if (found_8_3_d5 && found_pass_d3) found_correct_sequence = true;
    }
    SPIEL_CHECK_TRUE(found_correct_sequence);
  }
  //EndTest: test-vsdpb-1
  //StartTest: test-vsdpb-2
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][5] = 1;
    board[kXPlayerId][8] = 1;
    board[kOPlayerId][0] = 1;
    board[kOPlayerId][3] = 1;
    board[kXPlayerId][5] += 7;
    board[kXPlayerId][8] += 6;
    board[kOPlayerId][0] += 7;
    board[kOPlayerId][3] += 6;
    SetupBoardState(lnstate, kXPlayerId, board);
    SetupDice(lnstate, {5, 3, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    bool all_used_lower_die = true;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      SPIEL_CHECK_GE(moves.size(), 2);
      int non_pass_count = 0;
      bool used_correct_die = false;
      for (const auto& move : moves) {
        if (move.pos != kPassPos) {
          non_pass_count++;
          if (move.die == 3) used_correct_die = true;
          else { used_correct_die = false; break; }
        }
      }
      SPIEL_CHECK_EQ(non_pass_count, 1);
      if (!used_correct_die) { all_used_lower_die = false; break; }
    }
    SPIEL_CHECK_TRUE(all_used_lower_die);
  }
  //EndTest: test-vsdpb-2
}
//EndFunction: VerifyDicePlayBehavior

//StartFunction: DirectBearOffTest
void DirectBearOffTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  // --- White Test ---
  //StartTest: test-dbo-1w
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][0] = 1;
    board[kXPlayerId][1] = 1;
    board[kOPlayerId][11] = 15;
    SetupBoardState(lnstate, kXPlayerId, board);
    SetupDice(lnstate, {1, 2, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    bool found_direct_bearoff = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      if (moves.size() == 2) {
        bool move1_ok = (moves[0].pos == 0 && moves[0].die == 1 && moves[0].to_pos == kBearOffPos) ||
                        (moves[1].pos == 0 && moves[1].die == 1 && moves[1].to_pos == kBearOffPos);
        bool move2_ok = (moves[0].pos == 1 && moves[0].die == 2 && moves[0].to_pos == kBearOffPos) ||
                        (moves[1].pos == 1 && moves[1].die == 2 && moves[1].to_pos == kBearOffPos);
        if (move1_ok && move2_ok) {
          found_direct_bearoff = true;
          break;
        }
      }
    }
    SPIEL_CHECK_TRUE(found_direct_bearoff);
  }
  //EndTest: test-dbo-1w
  // --- Black Test ---
  //StartTest: test-dbo-1b
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kOPlayerId][13] = 1;
    board[kOPlayerId][14] = 1;
    board[kXPlayerId][23] = 15;
    SetupBoardState(lnstate, kOPlayerId, board);
    SetupDice(lnstate, {2, 3, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    bool found_direct_bearoff = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
      if (moves.size() == 2) {
        bool move1_ok = (moves[0].pos == 13 && moves[0].die == 2 && moves[0].to_pos == kBearOffPos) ||
                        (moves[1].pos == 13 && moves[1].die == 2 && moves[1].to_pos == kBearOffPos);
        bool move2_ok = (moves[0].pos == 14 && moves[0].die == 3 && moves[0].to_pos == kBearOffPos) ||
                        (moves[1].pos == 14 && moves[1].die == 3 && moves[1].to_pos == kBearOffPos);
        if (move1_ok && move2_ok) {
          found_direct_bearoff = true;
          break;
        }
      }
    }
    SPIEL_CHECK_TRUE(found_direct_bearoff);
  }
  //EndTest: test-dbo-1b
}
//EndFunction: DirectBearOffTest

//StartFunction: SingleCheckerBearOffTest
void SingleCheckerBearOffTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  // --- Scenario 1: Higher Die Rule (Both playable) ---
  //StartTest: test-scbo-1w
  {
    // White Test (Checker at pos 0)
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> boardW(2, std::vector<int>(kNumPoints, 0));
    boardW[kXPlayerId][0] = 1;
    boardW[kOPlayerId][11] = 15;
    SetupBoardState(lnstate, kXPlayerId, boardW);
    SetupDice(lnstate, {1, 6, 0, 0});
    auto legal_actionsW = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsW.size(), 1);
    std::vector<CheckerMove> movesW = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actionsW[0]);
    bool found_die6W = false;
    for(const auto& m : movesW) { if(m.pos == 0 && m.die == 6 && m.to_pos == kBearOffPos) found_die6W = true; }
    SPIEL_CHECK_TRUE(found_die6W);
  }
  //EndTest: test-scbo-1w

  //StartTest: test-scbo-1b
  // Black Test (Checker at pos 12)
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints, 0));
    boardB[kOPlayerId][12] = 1;
    boardB[kXPlayerId][0] = 15;
    SetupBoardState(lnstate, kOPlayerId, boardB);
    SetupDice(lnstate, {1, 6, 0, 0});
    auto legal_actionsB = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsB.size(), 1);
    std::vector<CheckerMove> movesB = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
    bool found_die6B = false;
    for(const auto& m : movesB) { if(m.pos == 12 && m.die == 6 && m.to_pos == kBearOffPos) found_die6B = true; }
    SPIEL_CHECK_TRUE(found_die6B);
  }
  //EndTest: test-scbo-1b

  // --- Scenario 2: Only One Die Playable ---
  //StartTest: test-scbo-2b1
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints, 0));
    boardB[kOPlayerId][12] = 1;
    boardB[kXPlayerId][23] = 15;
    SetupBoardState(lnstate, kOPlayerId, boardB);
    SetupDice(lnstate, {1, 3, 0, 0});
    auto legal_actionsB = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsB.size(), 1);
    std::vector<CheckerMove> movesB = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
    bool found_die1B = false;
    bool found_die3B = false;
    for(const auto& m : movesB) {
      if(m.pos == 12 && m.die == 1 && m.to_pos == kBearOffPos) found_die1B = true;
      if(m.pos == 12 && m.die == 3 && m.to_pos == kBearOffPos) found_die3B = true;
    }
    SPIEL_CHECK_FALSE(found_die1B);
    SPIEL_CHECK_TRUE(found_die3B);
  }
  //EndTest: test-scbo-2b1
  //StartTest: test-scbo-2b2
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> boardB_pos14(2, std::vector<int>(kNumPoints, 0));
    boardB_pos14[kOPlayerId][14] = 1;
    boardB_pos14[kXPlayerId][23] = 15;
    SetupBoardState(lnstate, kOPlayerId, boardB_pos14);
    SetupDice(lnstate, {1, 3, 0, 0});
    auto legal_actionsB = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsB.size(), 2);
    bool found_two_step = false;
    bool found_direct_bearoff = false;
    for (const auto& action : legal_actionsB) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
      if (moves.size() == 2) {
        bool has_move_step = false;
        bool has_bearoff_step = false;
        for (const auto& m : moves) {
          if (m.pos == 14 && m.to_pos == 13 && m.die == 1) has_move_step = true;
          if (m.pos == 13 && m.to_pos == kBearOffPos && m.die == 3) has_bearoff_step = true;
        }
        if (has_move_step && has_bearoff_step) found_two_step = true;
        has_move_step = false;
        has_bearoff_step = false;
        for (const auto& m : moves) {
          if (m.pos == 14 && m.to_pos == kBearOffPos && m.die == 3) has_bearoff_step = true;
          if (m.pos == kPassPos && m.to_pos == kPassPos && m.die == 1) has_move_step = true;
        }
        if (has_move_step && has_bearoff_step) found_direct_bearoff = true;
      }
    }
    SPIEL_CHECK_TRUE(found_two_step);
    SPIEL_CHECK_TRUE(found_direct_bearoff);
  }
  //EndTest: test-scbo-2b2
}
//EndFunction: SingleCheckerBearOffTest

//StartFunction: BearOffLastCheckerTest
void BearOffLastCheckerTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  // --- White Test (Last checker at pos 1, needs 2 pips) ---
  //StartTest: test-bolc-1w
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][1] = 1;
    board[kOPlayerId][11] = 15;
    SetupBoardState(lnstate, kXPlayerId, board);
    SetupDice(lnstate, {4, 5, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actions[0]);
    SPIEL_CHECK_EQ(moves.size(), 2);
    bool found_bear_off = false;
    bool found_pass = false;
    int bear_off_die = -1;
    for (const auto& move : moves) {
      if (move.pos == 1 && move.to_pos == kBearOffPos && (move.die == 4 || move.die == 5)) {
        SPIEL_CHECK_EQ(move.die, 5);
        found_bear_off = true;
        bear_off_die = move.die;
      } else if (move.pos == kPassPos) {
        found_pass = true;
        SPIEL_CHECK_NE(move.die, bear_off_die);
        SPIEL_CHECK_TRUE(move.die == 4 || move.die == 5);
      }
    }
    SPIEL_CHECK_TRUE(found_bear_off);
    SPIEL_CHECK_TRUE(found_pass);
  }
  //EndTest: test-bolc-1w
  // --- Black Test (Last checker at pos 13, needs 2 pips) ---
  //StartTest: test-bolc-1b
  {
    std::unique_ptr<State> state = game->NewInitialState();
    auto lnstate = static_cast<LongNardeState*>(state.get());
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kOPlayerId][13] = 1;
    board[kXPlayerId][23] = 15;
    SetupBoardState(lnstate, kOPlayerId, board);
    SetupDice(lnstate, {4, 5, 0, 0});
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actions[0]);
    SPIEL_CHECK_EQ(moves.size(), 2);
    bool found_bear_off = false;
    bool found_pass = false;
    int bear_off_die = -1;
    for (const auto& move : moves) {
      if (move.pos == 13 && move.to_pos == kBearOffPos && (move.die == 4 || move.die == 5)) {
        SPIEL_CHECK_EQ(move.die, 5);
        found_bear_off = true;
        bear_off_die = move.die;
      } else if (move.pos == kPassPos) {
        found_pass = true;
        SPIEL_CHECK_NE(move.die, bear_off_die);
        SPIEL_CHECK_TRUE(move.die == 4 || move.die == 5);
      }
    }
    SPIEL_CHECK_TRUE(found_bear_off);
    SPIEL_CHECK_TRUE(found_pass);
  }
  //EndTest: test-bolc-1b
}
//EndFunction: BearOffLastCheckerTest

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

// Register the test function in the common interface
namespace open_spiel {
namespace long_narde {

// Implement the global test function that was declared in the header
//StartFunction: TestPassMoveBehavior
void TestPassMoveBehavior() {
  std::cout << "\n=== Testing Pass Move Behavior ===\n";
  PassMoveBehaviorTest();
  std::cout << "✓ Pass Move Behavior Test passed\n";
}
//EndFunction: TestPassMoveBehavior

//StartFunction: TestActionEncoding
void TestActionEncoding() {
  std::cout << "\n=== Testing Action Encoding ===\n";
  ActionEncodingTest();
  SingleLegalMoveTest();
  ConsecutiveMovesTest();
  UndoRedoTest();
  VerifyDicePlayBehavior();
  DirectBearOffTest();
  SingleCheckerBearOffTest();
  BearOffLastCheckerTest();

  std::cout << "✓ All action encoding tests passed\n";
}
//EndFunction: TestActionEncoding

}  // namespace long_narde
}  // namespace open_spiel

