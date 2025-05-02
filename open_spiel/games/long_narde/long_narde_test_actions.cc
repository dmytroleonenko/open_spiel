#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/long_narde/long_narde.h"

namespace open_spiel
{
  namespace long_narde
  {
    namespace
    {

      // StartFunction: ActionEncodingTest
      void ActionEncodingTest()
      {
        std::cout << "[TEST] Start: ActionEncodingTest" << std::endl;
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        LongNardeState *lnstate = static_cast<LongNardeState *>(state.get());

        // *** FIX: Explicitly set the player after initial state creation ***
        lnstate->cur_player_ = kXPlayerId; // Assume testing White's perspective after implicit first roll.
        // *** END FIX ***

        // Test Half-Move Encoding & Decoding for a nondouble roll
        SetupDice(lnstate, {5, 3, 0, 0});
        auto actions = state->LegalActions();
        SPIEL_CHECK_FALSE(actions.empty());
        for (Action a : actions) {
          auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, a);
          SPIEL_CHECK_EQ(moves.size(), 1);
          SPIEL_CHECK_EQ(a, (moves[0].pos == kPassPos ? kNumPoints : moves[0].pos));
          SPIEL_CHECK_GE(moves[0].die, 1);
          SPIEL_CHECK_LE(moves[0].die, 6);
        }

        // Test Pass Action Encoding & Decoding
        Action pass = kNumPoints;
        auto pmoves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, pass);
        SPIEL_CHECK_EQ(pmoves.size(), 1);
        SPIEL_CHECK_EQ(pmoves[0].pos, kPassPos);

        // Test applying a single decoded half-move
        std::shared_ptr<const Game> game2 = LoadGame("long_narde");
        std::unique_ptr<State> state2 = game2->NewInitialState();
        auto lnstate2 = static_cast<LongNardeState *>(state2.get());

        // *** FIX: Explicitly set the player after initial state creation (for second scenario) ***
        lnstate2->cur_player_ = kXPlayerId; // Assume testing White's perspective after implicit first roll.
        // *** END FIX ***

        // Use four dice for doubles setup
        SetupDice(lnstate2, {4, 4, 4, 4});
        auto legal_actions2 = state2->LegalActions();
        SPIEL_CHECK_FALSE(legal_actions2.empty());
        Action first_action = legal_actions2[0]; // Take the first legal half-move

        // Decode this specific action *before* applying
        auto moves_before_apply = lnstate2->LongNardeSpielMoveToCheckerMoves(kXPlayerId, first_action);
        SPIEL_CHECK_EQ(moves_before_apply.size(), 1); // Ensure it decodes to one half-move

        // Apply the valid half-move
        state2->ApplyAction(first_action);
        // We don't decode *after* applying, as per the new API rules.
        // The state is now advanced by one half-move. Further checks could go here if needed.

        std::cout << "[TEST] End: ActionEncodingTest" << std::endl;
      }

      // StartFunction: SingleLegalMoveTest
      // StartTest: test-singlelegalmovetest-1
      void SingleLegalMoveTest()
      {
        std::cout << "[TEST] Start: test-singlelegalmovetest-1" << std::endl;
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        {
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
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
          if (!legal_actions.empty())
          {
            Action the_action = legal_actions[0];
            std::vector<LongNardeCheckerMove> decoded_moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, the_action);
            for (const auto &move : decoded_moves)
            {
              if (move.pos != kPassPos)
              {
                if (move.pos == 5 && move.die == 3 && move.to_pos == 2)
                {
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
        std::cout << "[TEST] End: test-singlelegalmovetest-1" << std::endl;
      }
      // EndTest: test-singlelegalmovetest-1
      // EndFunction: SingleLegalMoveTest

      // StartFunction: ConsecutiveMovesTest
      // StartTest: test-consecutivemovestest-1
      void ConsecutiveMovesTest()
      {
        std::cout << "[TEST] Start: test-consecutivemovestest-1" << std::endl;
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        {
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          // Apply initial dice roll (simulate chance node)
          lnstate->ApplyAction(15); // arbitrary dice roll action
          // White's turn: apply all half-moves for this turn
          while (!lnstate->IsChanceNode()) {
            auto las = lnstate->LegalActions();
            SPIEL_CHECK_FALSE(las.empty());
            lnstate->ApplyAction(las[0]); // just pick the first legal half-move
          }
          SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
          // Apply next dice roll (simulate chance node)
          lnstate->ApplyAction(0); // arbitrary dice roll action
          SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kOPlayerId);
          // Black's turn: apply all half-moves for this turn
          while (!lnstate->IsChanceNode()) {
            auto las = lnstate->LegalActions();
            SPIEL_CHECK_FALSE(las.empty());
            lnstate->ApplyAction(las[0]);
          }
          SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
        }
        std::cout << "[TEST] End: test-consecutivemovestest-1" << std::endl;
      }
      // EndTest: test-consecutivemovestest-1
      // EndFunction: ConsecutiveMovesTest

      // StartFunction: UndoRedoTest
      // StartTest: test-undoredotest-1
      void UndoRedoTest()
      {
        std::cout << "[TEST] Start: test-undoredotest-1" << std::endl;
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        {
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
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
          for (int p = 0; p < 2; ++p)
          {
            std::vector<int> player_board;
            for (int i = 0; i < kNumPoints; ++i)
            {
              player_board.push_back(lnstate->board(p, i));
            }
            board_before.push_back(player_board);
          }
          std::vector<Action> legal_actions = lnstate->LegalActions();
          SPIEL_CHECK_FALSE(legal_actions.empty());
          Action action_to_apply = legal_actions[0];
          lnstate->ApplyAction(action_to_apply); // apply a single half-move
          bool board_changed = false;
          for (int p = 0; p < 2; ++p)
          {
            for (int i = 0; i < kNumPoints; ++i)
            {
              if (lnstate->board(p, i) != board_before[p][i])
              {
                board_changed = true;
                break;
              }
            }
            if (board_changed)
              break;
          }
          SPIEL_CHECK_TRUE(board_changed);
          lnstate->UndoAction(lnstate->CurrentPlayer(), action_to_apply); // undo the half-move
          bool board_restored = true;
          for (int p = 0; p < 2; ++p)
          {
            for (int i = 0; i < kNumPoints; ++i)
            {
              if (lnstate->board(p, i) != board_before[p][i])
              {
                board_restored = false;
                break;
              }
            }
            if (!board_restored)
              break;
          }
          SPIEL_CHECK_TRUE(board_restored);
        }
        std::cout << "[TEST] End: test-undoredotest-1" << std::endl;
      }
      // EndTest: test-undoredotest-1
      // EndFunction: UndoRedoTest

      // StartFunction: PassMoveBehaviorTest
      void PassMoveBehaviorTest()
      {
        std::cout << "\n=== Testing Pass Move Behavior ===\n";
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        // StartTest: test-passmovebehaviortest-1
        {
          std::cout << "[TEST] Start: test-passmovebehaviortest-1" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> no_moves_board(2, std::vector<int>(kNumPoints, 0));
          no_moves_board[kXPlayerId][4] = 1;
          no_moves_board[kOPlayerId][1] = 1;
          no_moves_board[kOPlayerId][3] = 1;
          no_moves_board[kOPlayerId][5] = 1;
          no_moves_board[kOPlayerId][kBlackHeadPos] += 12;
          SetupBoardState(lnstate, kXPlayerId, no_moves_board);
          SetupDice(lnstate, {1, 3, 0, 0});
          // Expect two sequential pass actions: first for die 3, then for die 1
          for (int die : {3, 1}) {
            auto legal_actions = lnstate->LegalActions();
            SPIEL_CHECK_EQ(legal_actions.size(), 1);
            Action pass_action = legal_actions[0];
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, pass_action);
            SPIEL_CHECK_EQ(moves.size(), 1);
            SPIEL_CHECK_EQ(moves[0].pos, kPassPos);
            SPIEL_CHECK_EQ(moves[0].die, die);
            lnstate->ApplyAction(pass_action);
          }
          std::cout << "[TEST] End: test-passmovebehaviortest-1" << std::endl;
        }
        // EndTest: test-passmovebehaviortest-1
        // StartTest: test-passmovebehaviortest-2
        {
          std::cout << "[TEST] Start: test-passmovebehaviortest-2" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> valid_moves_board(2, std::vector<int>(kNumPoints, 0));
          valid_moves_board[kXPlayerId][1] = 1;
          valid_moves_board[kXPlayerId][3] = 1;
          valid_moves_board[kOPlayerId][12] = 15;
          SetupBoardState(lnstate, kXPlayerId, valid_moves_board);
          SetupDice(lnstate, {1, 3, 0, 0});
          auto legal_actions = lnstate->LegalActions();
          // Under the half-move API, all legal actions must decode to a single half-move.
          for (Action action : legal_actions) {
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
            SPIEL_CHECK_EQ(moves.size(), 1);
            // No action should encode a multi-move pass.
            SPIEL_CHECK_FALSE(moves[0].pos == kPassPos && moves[0].die == 1 && moves.size() > 1);
          }
          SPIEL_CHECK_GT(legal_actions.size(), 0);
          std::cout << "[TEST] End: test-passmovebehaviortest-2" << std::endl;
        }
        // EndTest: test-passmovebehaviortest-2
        // StartTest: test-passmovebehaviortest-3
        {
          std::cout << "[TEST] Start: test-passmovebehaviortest-3" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
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
          auto legal_actions = lnstate->LegalActions();
          // Under the half-move API, must pass with die 2 four times in sequence.
          for (int i = 0; i < 4; ++i) {
            auto legal_actions = lnstate->LegalActions();
            SPIEL_CHECK_EQ(legal_actions.size(), 1);
            Action pass_action = legal_actions[0];
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, pass_action);
            SPIEL_CHECK_EQ(moves.size(), 1);
            SPIEL_CHECK_EQ(moves[0].pos, kPassPos);
            SPIEL_CHECK_EQ(moves[0].die, 2);
            lnstate->ApplyAction(pass_action);
          }
          std::cout << "[TEST] End: test-passmovebehaviortest-3" << std::endl;
        }
        // EndTest: test-passmovebehaviortest-3
      }
      // EndFunction: PassMoveBehaviorTest

      // StartFunction: VerifyDicePlayBehavior
      void VerifyDicePlayBehavior()
      {
        std::cout << "\n=== Testing Dice Play Behavior ===\n";
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        // StartTest: test-vsdpb-1
        {
          std::cout << "[TEST] Start: test-vsdpb-1" << std::endl;
          // Two sequential half-moves: first pos=8 die=5, then pass die=3
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
          board[kXPlayerId][8] = 1;
          board[kXPlayerId][3] = 1;
          board[kOPlayerId][0] = 1;
          board[kXPlayerId][3] += 13;
          board[kOPlayerId][kBlackHeadPos] += 14;
          SetupBoardState(lnstate, kXPlayerId, board);
          SetupDice(lnstate, {5, 3, 0, 0});
          // First half-move: move from pos 8 with die=5
          auto las1 = lnstate->LegalActions();
          SPIEL_CHECK_FALSE(las1.empty());
          Action first_move = -1;
          for (Action a : las1) {
            auto mv = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, a);
            SPIEL_CHECK_EQ(mv.size(), 1);
            if (mv[0].pos == 8 && mv[0].die == 5) {
              first_move = a;
              break;
            }
          }
          SPIEL_CHECK_GE(first_move, 0);
          lnstate->ApplyAction(first_move);
          // Second half-move: expect a pass with die=3
          auto las2 = lnstate->LegalActions();
          SPIEL_CHECK_EQ(las2.size(), 1);
          Action pass_move = las2[0];
          auto pmv = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, pass_move);
          SPIEL_CHECK_EQ(pmv.size(), 1);
          SPIEL_CHECK_EQ(pmv[0].pos, kPassPos);
          SPIEL_CHECK_EQ(pmv[0].die, 3);
          std::cout << "[TEST] End: test-vsdpb-1" << std::endl;
        }
        // EndTest: test-vsdpb-1
        // StartTest: test-vsdpb-2
        {
          std::cout << "[TEST] Start: test-vsdpb-2" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
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
          for (Action action : legal_actions)
          {
            // Decode before applying
            std::vector<LongNardeCheckerMove> moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
            SPIEL_CHECK_EQ(moves.size(), 1);
            int non_pass_count = 0;
            bool used_correct_die = false;
            for (const auto &move : moves)
            {
              if (move.pos != kPassPos)
              {
                non_pass_count++;
                if (move.die == 3)
                  used_correct_die = true;
                else
                {
                  used_correct_die = false;
                  break;
                }
              }
            }
            SPIEL_CHECK_EQ(non_pass_count, 1);
            if (!used_correct_die)
            {
              all_used_lower_die = false;
              break;
            }
            // If you want to check post-move state, clone before applying
            // std::unique_ptr<State> post_state = state->Clone();
            // static_cast<LongNardeState *>(post_state.get())->ApplyAction(action);
            // ... check post-move state if needed ...
          }
          SPIEL_CHECK_TRUE(all_used_lower_die);
          std::cout << "[TEST] End: test-vsdpb-2" << std::endl;
        }
        // EndTest: test-vsdpb-2
      }
      // EndFunction: VerifyDicePlayBehavior

      // StartFunction: DirectBearOffTest
      void DirectBearOffTest()
      {
        std::cout << "\n=== Testing Direct Bear Off ===\n";
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        // --- White Test ---
        // StartTest: test-dbo-1w
        {
          std::cout << "[TEST] Start: test-dbo-1w" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
          // Place two white checkers in bear-off zone
          board[kXPlayerId][0] = 1;
          board[kXPlayerId][1] = 1;
          board[kOPlayerId][11] = 15;
          // Roll dice: values 1 and 2
          SetupDice(lnstate, {1, 2, 0, 0});
          // Apply board configuration
          SetupBoardState(lnstate, kXPlayerId, board);

          // First half-move: bear off from point 1 using die=2
          auto actions1 = lnstate->LegalActions();
          SPIEL_CHECK_EQ(actions1.size(), 2);
          Action first = -1;
          for (Action a : actions1) {
            auto mv = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, a);
            SPIEL_CHECK_EQ(mv.size(), 1);
            if (mv[0].pos == 1 && mv[0].die == 2 && mv[0].to_pos == kBearOffPos) {
              first = a;
              break;
            }
          }
          SPIEL_CHECK_GE(first, 0);
          lnstate->ApplyAction(first);
          SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, 1), 0);

          // Second half-move: remaining bear-off from point 0 using die=1
          auto actions2 = lnstate->LegalActions();
          SPIEL_CHECK_EQ(actions2.size(), 1);
          Action second = actions2[0];
          auto mv2 = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, second);
          SPIEL_CHECK_EQ(mv2.size(), 1);
          SPIEL_CHECK_EQ(mv2[0].pos, 0);
          SPIEL_CHECK_EQ(mv2[0].die, 1);
          SPIEL_CHECK_EQ(mv2[0].to_pos, kBearOffPos);
          lnstate->ApplyAction(second);
          SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, 0), 0);

          // Game should be terminal after bearing off both checkers
          SPIEL_CHECK_TRUE(lnstate->IsTerminal());
          std::cout << "[TEST] End: test-dbo-1w" << std::endl;
        }
        // EndTest: test-dbo-1w
        //  --- Black Test ---
        // StartTest: test-dbo-1b
        {
          std::cout << "[TEST] Start: test-dbo-1b" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
          board[kOPlayerId][13] = 1;
          board[kOPlayerId][14] = 1;
          board[kXPlayerId][23] = 15;
          SetupDice(lnstate, {2, 3, 0, 0}); // Dice {2, 3}
          SetupBoardState(lnstate, kOPlayerId, board);

          // 1. First half-move: Bear off checker at 14 with die 3
          auto legal_actions1 = lnstate->LegalActions();
          bool found_14_d3 = false;
          Action action1 = -1;
          for (Action a : legal_actions1) {
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, a);
            if (moves.size() == 1 && moves[0].pos == 14 && moves[0].die == 3 && moves[0].to_pos == kBearOffPos) {
              found_14_d3 = true;
              action1 = a;
              break;
            }
          }
          SPIEL_CHECK_TRUE(found_14_d3);
          lnstate->ApplyAction(action1);
          SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, 14), 0);

          // 2. Second half-move: Bear off checker at 13 with die 2
          auto legal_actions2 = lnstate->LegalActions();
          bool found_13_d2 = false;
          Action action2 = -1;
          for (Action a : legal_actions2) {
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, a);
            if (moves.size() == 1 && moves[0].pos == 13 && moves[0].die == 2 && moves[0].to_pos == kBearOffPos) {
              found_13_d2 = true;
              action2 = a;
              break;
            }
          }
          SPIEL_CHECK_TRUE(found_13_d2);
          lnstate->ApplyAction(action2);
          SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, 13), 0);

          // 3. After both half-moves, the game should be terminal (no pass move)
          auto legal_actions3 = lnstate->LegalActions();
          SPIEL_CHECK_TRUE(lnstate->IsTerminal());
          SPIEL_CHECK_TRUE(legal_actions3.empty());
          std::cout << "[TEST] End: test-dbo-1b" << std::endl;
        }
        // EndTest: test-dbo-1b
      }
      // EndFunction: DirectBearOffTest

      // StartFunction: SingleCheckerBearOffTest
      void SingleCheckerBearOffTest()
      {
        std::cout << "\n=== Testing Single Checker Bear Off ===\n";
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        // --- Scenario 1: Higher Die Rule (Both playable) ---
        // StartTest: test-scbo-1w
        {
          std::cout << "[TEST] Start: test-scbo-1w" << std::endl;
          // White Test (Checker at pos 0)
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> boardW(2, std::vector<int>(kNumPoints, 0));
          boardW[kXPlayerId][0] = 1;
          boardW[kOPlayerId][11] = 15;
          SetupBoardState(lnstate, kXPlayerId, boardW);
          SetupDice(lnstate, {1, 6, 0, 0});
          auto legal_actionsW = lnstate->LegalActions();
          SPIEL_CHECK_EQ(legal_actionsW.size(), 1);
          std::vector<LongNardeCheckerMove> movesW = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, legal_actionsW[0]);
          bool found_die6W = false;
          for (const auto &m : movesW)
          {
            if (m.pos == 0 && m.die == 6 && m.to_pos == kBearOffPos)
              found_die6W = true;
          }
          SPIEL_CHECK_TRUE(found_die6W);
          std::cout << "[TEST] End: test-scbo-1w" << std::endl;
        }
        // EndTest: test-scbo-1w

        // StartTest: test-scbo-1b
        //  Black Test (Checker at pos 12)
        {
          std::cout << "[TEST] Start: test-scbo-1b" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints, 0));
          boardB[kOPlayerId][12] = 1;
          boardB[kXPlayerId][0] = 15;
          SetupBoardState(lnstate, kOPlayerId, boardB);
          SetupDice(lnstate, {1, 6, 0, 0});
          auto legal_actionsB = lnstate->LegalActions();
          SPIEL_CHECK_EQ(legal_actionsB.size(), 1);
          auto movesB = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
          SPIEL_CHECK_EQ(movesB.size(), 1);
          SPIEL_CHECK_EQ(movesB[0].pos, 12);
          SPIEL_CHECK_EQ(movesB[0].die, 6);
          SPIEL_CHECK_EQ(movesB[0].to_pos, kBearOffPos);
          std::cout << "[TEST] End: test-scbo-1b" << std::endl;
        }
        // EndTest: test-scbo-1b

        // --- Scenario 2: Only One Die Playable ---
        // StartTest: test-scbo-2b1
        {
          std::cout << "[TEST] Start: test-scbo-2b1" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints, 0));
          boardB[kOPlayerId][12] = 1;
          boardB[kXPlayerId][23] = 15;
          SetupBoardState(lnstate, kOPlayerId, boardB);
          SetupDice(lnstate, {1, 3, 0, 0});
          auto legal_actionsB = lnstate->LegalActions();
          SPIEL_CHECK_EQ(legal_actionsB.size(), 1);
          std::vector<LongNardeCheckerMove> movesB = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
          bool found_die1B = false;
          bool found_die3B = false;
          for (const auto &m : movesB)
          {
            if (m.pos == 12 && m.die == 1 && m.to_pos == kBearOffPos)
              found_die1B = true;
            if (m.pos == 12 && m.die == 3 && m.to_pos == kBearOffPos)
              found_die3B = true;
          }
          SPIEL_CHECK_FALSE(found_die1B);
          SPIEL_CHECK_TRUE(found_die3B);
          std::cout << "[TEST] End: test-scbo-2b1" << std::endl;
        }
        // EndTest: test-scbo-2b1
        // StartTest: test-scbo-2b2
        {
          std::cout << "[TEST] Start: test-scbo-2b2" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> boardB_pos14(2, std::vector<int>(kNumPoints, 0));
          boardB_pos14[kOPlayerId][14] = 1;
          boardB_pos14[kXPlayerId][23] = 15;
          SetupBoardState(lnstate, kOPlayerId, boardB_pos14);
          SetupDice(lnstate, {1, 3, 0, 0});
          auto legal_actionsB = lnstate->LegalActions();
          SPIEL_CHECK_EQ(legal_actionsB.size(), 1);
          bool found_bear_off = false;
          bool found_move_1 = false;
          if (!legal_actionsB.empty()) {
            Action the_action = legal_actionsB[0]; // Only one action expected
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, the_action);
            SPIEL_CHECK_EQ(moves.size(), 1);
            if (moves[0].pos == 14 && moves[0].to_pos == kBearOffPos && moves[0].die == 3) {
              found_bear_off = true;
            } else if (moves[0].pos == 14 && moves[0].to_pos == 13 && moves[0].die == 1) {
              // This branch should not be hit if the logic is correct
              found_move_1 = true; 
            }
          }
          SPIEL_CHECK_TRUE(found_bear_off);
          SPIEL_CHECK_FALSE(found_move_1); // Verify the lower die move is NOT legal
          std::cout << "[TEST] End: test-scbo-2b2" << std::endl;
        }
        // EndTest: test-scbo-2b2
      }
      // EndFunction: SingleCheckerBearOffTest

      // StartFunction: BearOffLastCheckerTest
      void BearOffLastCheckerTest()
      {
        std::cout << "\n=== Testing Bear Off Last Checker ===\n";
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        // --- White Test (Last checker at pos 1, needs 2 pips) ---
        // StartTest: test-bolc-1w
        {
          std::cout << "[TEST] Start: test-bolc-1w" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
          board[kXPlayerId][1] = 1; // Last checker at pos 1 (needs >= 2 pips)
          board[kOPlayerId][11] = 15; // Opponent checkers far away
          SetupBoardState(lnstate, kXPlayerId, board);
          SetupDice(lnstate, {4, 5, 0, 0}); // Dice {4, 5}

          // 1. Check first half-move: Must bear off last checker (pos 1) with highest possible die (5).
          auto legal_actions1 = lnstate->LegalActions();
          SPIEL_CHECK_EQ(legal_actions1.size(), 1); // Only one way to play the mandatory move
          Action action1 = legal_actions1[0];
          auto moves1 = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action1);
          SPIEL_CHECK_EQ(moves1.size(), 1);
          SPIEL_CHECK_EQ(moves1[0].pos, 1);
          SPIEL_CHECK_EQ(moves1[0].die, 5); // Must use die 5
          SPIEL_CHECK_EQ(moves1[0].to_pos, kBearOffPos);

          // Apply the first half-move (1 -> BearOff with die 5)
          lnstate->ApplyAction(action1);
          SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, 1), 0); // Checker left pos 1

          // 2. After bearing off the last checker, the game should be terminal (no pass move)
          auto legal_actions2 = lnstate->LegalActions();
          SPIEL_CHECK_TRUE(lnstate->IsTerminal());
          SPIEL_CHECK_TRUE(legal_actions2.empty());
          std::cout << "[TEST] End: test-bolc-1w" << std::endl;
        }
        // EndTest: test-bolc-1w
        //  --- Black Test (Last checker at pos 13, needs 2 pips) ---
        // StartTest: test-bolc-1b
        {
          std::cout << "[TEST] Start: test-bolc-1b" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());
          std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
          board[kOPlayerId][13] = 1;
          board[kXPlayerId][23] = 15;
          SetupDice(lnstate, {4, 5, 0, 0});
          SetupBoardState(lnstate, kOPlayerId, board);

          // 1. First half-move: Bear off checker at 13 with die 5
          auto legal_actions1 = lnstate->LegalActions();
          bool found_13_d5 = false;
          Action action1 = -1;
          for (Action a : legal_actions1) {
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, a);
            if (moves.size() == 1 && moves[0].pos == 13 && moves[0].die == 5 && moves[0].to_pos == kBearOffPos) {
              found_13_d5 = true;
              action1 = a;
              break;
            }
          }
          SPIEL_CHECK_TRUE(found_13_d5);
          lnstate->ApplyAction(action1);
          SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, 13), 0);

          // 2. After bearing off the last checker, the game should be terminal (no pass move)
          auto legal_actions2 = lnstate->LegalActions();
          SPIEL_CHECK_TRUE(lnstate->IsTerminal());
          SPIEL_CHECK_TRUE(legal_actions2.empty());
          std::cout << "[TEST] End: test-bolc-1b" << std::endl;
        }
        // EndTest: test-bolc-1b
      }
      // EndFunction: BearOffLastCheckerTest


      // StartFunction: HeadRuleAfterFirstHalfMoveTest
      // Test that after moving from head in the first half-move, subsequent
      // half-moves in the same turn cannot move from head again.
      void HeadRuleAfterFirstHalfMoveTest()
      {
        std::cout << "\n=== Testing Head Rule After First Half-Move ===\n";
        std::shared_ptr<const Game> game = LoadGame("long_narde");

        // StartTest: test-headrule-after-move-1
        {
          std::cout << "[TEST] Start: test-headrule-after-move-1" << std::endl;
          std::unique_ptr<State> state = game->NewInitialState();
          auto lnstate = static_cast<LongNardeState *>(state.get());

          // Board state BEFORE the first half-move (from debug log)
          std::vector<std::vector<int>> board = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 13}, // Player 0
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} // Player 1
          };
          SetupBoardState(lnstate, kOPlayerId, board); // Player 1's turn
          SetupDice(lnstate, {2, 1, 0, 0}); // Dice {2, 1}

          SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kOPlayerId);
          SPIEL_CHECK_FALSE(lnstate->moved_from_head()); // Before first move

          // 1. Find and apply the first half-move: Move from head (pos 11) with die 2.
          auto legal_actions1 = lnstate->LegalActions();
          Action head_move_action = -1;
          for (Action a : legal_actions1) {
            auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, a);
            // Assuming the head move is 11->9 with die 2
            if (moves.size() == 1 && moves[0].pos == 11 && moves[0].die == 2) {
              head_move_action = a;
              break;
            }
          }
          SPIEL_CHECK_GE(head_move_action, 0); // Ensure the head move was found
          lnstate->ApplyAction(head_move_action);

          // 2. Verify state after the first half-move.
          SPIEL_CHECK_TRUE(lnstate->moved_from_head()); // Should be true now
          SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, 11), 12); // One checker left head
          SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, 9), 2); // Checker arrived at pos 9
          // Check remaining die is 1
          int remaining_die = -1;
          for(int d : lnstate->dice_) if (d > 0) remaining_die = d;
          SPIEL_CHECK_EQ(remaining_die, 1);

          // 3. Get legal actions for the second half-move.
          auto legal_actions2 = lnstate->LegalActions();

          // 4. Assert the correct legal actions: only pos 2 and pos 9 should be available.
          std::set<Action> legal_actions_set(legal_actions2.begin(), legal_actions2.end());
          std::set<Action> expected_actions = {2, 9};

          SPIEL_CHECK_EQ(legal_actions_set.size(), 2);
          // Check that the sets contain the same elements
          SPIEL_CHECK_EQ(legal_actions_set.size(), expected_actions.size());
          for (Action expected_action : expected_actions) {
            SPIEL_CHECK_TRUE(legal_actions_set.count(expected_action) == 1); // Check presence
          }

          std::cout << "[TEST] End: test-headrule-after-move-1" << std::endl;
        }
        // EndTest: test-headrule-after-move-1
      }
      // EndFunction: HeadRuleAfterFirstHalfMoveTest

    } // namespace
  } // namespace long_narde
} // namespace open_spiel

// Register the test function in the common interface
namespace open_spiel
{
  namespace long_narde
  {

    // Implement the global test function that was declared in the header
    // StartFunction: TestPassMoveBehavior
    void TestPassMoveBehavior()
    {
      std::cout << "\n=== Testing Pass Move Behavior ===\n";
      PassMoveBehaviorTest();
      std::cout << "✓ Pass Move Behavior Test passed\n";
    }
    // EndFunction: TestPassMoveBehavior

    // StartFunction: TestActionEncoding
    void TestActionEncoding()
    {
      std::cout << "\n=== Testing Action Encoding ===\n";
      ActionEncodingTest();
      SingleLegalMoveTest();
      ConsecutiveMovesTest();
      UndoRedoTest();
      VerifyDicePlayBehavior();
      DirectBearOffTest();
      SingleCheckerBearOffTest();
      BearOffLastCheckerTest();
      HeadRuleAfterFirstHalfMoveTest();

      std::cout << "✓ All action encoding tests passed\n";
    }
    // EndFunction: TestActionEncoding

  } // namespace long_narde
} // namespace open_spiel
