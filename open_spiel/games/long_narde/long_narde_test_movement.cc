#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel
{
  namespace long_narde
  {
    namespace testing_internal
    { // Renamed from anonymous namespace

      //------------------------------------------------------------------------------
      // StartFunction: TestBasicMovement
      // StartTest: test-basicmovement-1
      void TestBasicMovement()
      {
        std::cout << "\n=== Running TestBasicMovement ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        LongNardeState *lnstate = static_cast<LongNardeState *>(state.get());

        SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
        SPIEL_CHECK_TRUE(lnstate->IsChanceNode());

        lnstate->ApplyAction(18); // Apply dice outcome 4,4 (special double)

        SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
        SPIEL_CHECK_FALSE(lnstate->IsChanceNode());
        SPIEL_CHECK_EQ(lnstate->dice(0), 4);
        SPIEL_CHECK_EQ(lnstate->dice(1), 4);

        // Special double: four moves of 4
        // Move two checkers from head (23 -> 19), then move those two again (19 -> 15)
        std::vector<Action> legal_actions = lnstate->LegalActions();
        SPIEL_CHECK_FALSE(legal_actions.empty());

        std::vector<LongNardeCheckerMove> checkers_moves = {
            {kWhiteHeadPos, kWhiteHeadPos - 4, 4},     // 23 -> 19
            {kWhiteHeadPos - 4, kWhiteHeadPos - 8, 4}, // 19 -> 15
            {kWhiteHeadPos, kWhiteHeadPos - 4, 4},     // 23 -> 19
            {kWhiteHeadPos - 4, kWhiteHeadPos - 8, 4}  // 19 -> 15
        };
        Action action = lnstate->LongNardeCheckerMovesToSpielMove(checkers_moves);

        lnstate->ApplyAction(action);

        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), 13);    // 15 - 2 = 13 left on head
        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 4), 0); // 0 left on point 19
        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 8), 2); // 2 ended up on point 15

        std::cout << "✓ Basic movement test passed\n";
      }
      // EndTest: test-basicmovement-1
      // EndFunction: TestBasicMovement

      //------------------------------------------------------------------------------
      // StartFunction: InitialDiceTest
      // StartTest: test-initialdice-1
      void InitialDiceTest()
      {
        std::cout << "\n=== Running InitialDiceTest ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        SPIEL_CHECK_TRUE(lnstate->IsChanceNode());

        std::vector<std::pair<Action, double>> outcomes = lnstate->ChanceOutcomes();
        SPIEL_CHECK_EQ(outcomes.size(), 21);

        for (const auto &outcome_pair : outcomes)
        {
          Action dice_action = outcome_pair.first;
          std::unique_ptr<State> clone = lnstate->Clone();
          auto clone_lnstate = static_cast<LongNardeState *>(clone.get());
          clone_lnstate->ApplyAction(dice_action);
          int die1 = clone_lnstate->dice(0);
          int die2 = clone_lnstate->dice(1);
          SPIEL_CHECK_GE(die1, 1);
          SPIEL_CHECK_LE(die1, 6);
          SPIEL_CHECK_GE(die2, 1);
          SPIEL_CHECK_LE(die2, 6);
          if (die1 != die2)
          {
            SPIEL_CHECK_GE(die1, die2);
          }
        }
        std::cout << "✓ Initial dice values verified\n";
      }
      // EndTest: test-initialdice-1
      // EndFunction: InitialDiceTest

      //------------------------------------------------------------------------------
      // StartFunction: CheckerDistributionTest
      // StartTest: test-checkerdistribution-1
      void CheckerDistributionTest()
      {
        std::cout << "\n=== Running CheckerDistributionTest ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        // Initial: White's 15 at pos 24, Black's 15 at pos 12
        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), kNumCheckersPerPlayer);
        SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, kBlackHeadPos), kNumCheckersPerPlayer);

        lnstate->ApplyAction(20); // Apply dice outcome 6,6

        std::vector<LongNardeCheckerMove> moves = {
            {kWhiteHeadPos, kWhiteHeadPos - 6, 6},
            {kWhiteHeadPos, kWhiteHeadPos - 6, 6}};
        Action action = lnstate->LongNardeCheckerMovesToSpielMove(moves);
        lnstate->ApplyAction(action);

        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), 13);
        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 6), 2);

        std::cout << "✓ Checker distribution verified\n";
      }
      // EndTest: test-checkerdistribution-1
      // EndFunction: CheckerDistributionTest

      //------------------------------------------------------------------------------
      // StartFunction: HeadRuleTestFirstTurn
      // Test: HeadRuleTestFirstTurn
      // Checks that on the first turn with special doubles (e.g., 4,4),
      // the player is allowed to move two checkers from their head point.
      //------------------------------------------------------------------------------
      // StartTest: test-headrule-first-turn-1
      void HeadRuleTestFirstTurn()
      {
        std::cout << "\n=== Running HeadRuleTestFirstTurn ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> stA = game->NewInitialState();
        auto lnA = static_cast<LongNardeState *>(stA.get());
        lnA->ApplyAction(18); // 4,4
        SPIEL_CHECK_TRUE(lnA->IsFirstTurn(kXPlayerId));
        std::vector<Action> first_turn_actions = lnA->LegalActions();
        bool can_move_2_from_head = false;
        for (Action a : first_turn_actions)
        {
          std::unique_ptr<State> c = lnA->Clone();
          auto cst = static_cast<LongNardeState *>(c.get());
          int init_head_count = cst->board(kXPlayerId, kWhiteHeadPos);
          cst->ApplyAction(a);
          int new_head_count = cst->board(kXPlayerId, kWhiteHeadPos);
          int diff = init_head_count - new_head_count;
          if (diff >= 2)
          {
            can_move_2_from_head = true;
            break;
          }
        }
        SPIEL_CHECK_TRUE(can_move_2_from_head);
        std::cout << "✓ Head rule first-turn test passed\n";
      }
      // EndTest: test-headrule-first-turn-1
      // EndFunction: HeadRuleTestFirstTurn

      //------------------------------------------------------------------------------
      // StartFunction: HeadRuleTestNonFirstTurn
      // Test: HeadRuleTestNonFirstTurn
      // Checks that on a non-first turn, even with doubles (e.g., 4,4,4,4),
      // the player is allowed to move only one checker from their head point per turn.
      // Subsequent moves must use checkers already off the head.
      //------------------------------------------------------------------------------
      // StartTest: test-headrule-non-first-turn-1
      void HeadRuleTestNonFirstTurn()
      {
        std::cout << "\n=== Running HeadRuleTestNonFirstTurn ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> stB = game->NewInitialState();
        auto lnB = static_cast<LongNardeState *>(stB.get());
        // Setup: White has 14 on head, 1 elsewhere (point 23)
        std::vector<std::vector<int>> board_non_first = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 14}, // White: 1@idx22, 14@idx23 (head)
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black: 15@idx11 (head)
        };
        SetupBoardState(lnB, kXPlayerId, board_non_first);
        SetupDice(lnB, {4, 4, 4, 4});
        SPIEL_CHECK_FALSE(lnB->IsFirstTurn(kXPlayerId)); // Ensure it's not the first turn

        std::vector<Action> la = lnB->LegalActions();
        SPIEL_CHECK_FALSE(la.empty());

        bool found_illegal_2_from_head = false;
        for (Action move : la)
        {
          std::unique_ptr<State> c = lnB->Clone();
          auto cst = static_cast<LongNardeState *>(c.get());
          int init_head_count = cst->board(kXPlayerId, kWhiteHeadPos);
          cst->ApplyAction(move);
          int new_head_count = cst->board(kXPlayerId, kWhiteHeadPos);
          int diff = init_head_count - new_head_count;
          if (diff > 1)
          { // Check if more than one checker moved from head
            found_illegal_2_from_head = true;
            std::cerr << "Illegal move found: " << lnB->ActionToString(kXPlayerId, move) << "\n";
            std::cerr << "Initial head count: " << init_head_count << ", New head count: " << new_head_count << "\n";
            break;
          }
        }
        SPIEL_CHECK_FALSE(found_illegal_2_from_head);
        std::cout << "✓ Head rule non-first-turn test passed\n";
      }
      // EndTest: test-headrule-non-first-turn-1
      // EndFunction: HeadRuleTestNonFirstTurn

      //------------------------------------------------------------------------------
      // StartFunction: MovementDirectionTest
      // Test: MovementDirectionTest
      // Verifies that White moves to lower indices, Black moves counterclockwise (wraps).
      //------------------------------------------------------------------------------
      // StartTest: test-movementdirection-1
      void MovementDirectionTest()
      {
        std::cout << "\n=== Running MovementDirectionTest ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        LongNardeState *lnstate = static_cast<LongNardeState *>(state.get());

        // White: check moves are strictly to lower indices
        std::vector<std::vector<int>> board_white_move = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        SetupBoardState(lnstate, kXPlayerId, board_white_move);
        SetupDice(lnstate, {3, 2, 0, 0});
        std::vector<Action> white_actions = lnstate->LegalActions();
        for (Action a : white_actions)
        {
          std::vector<LongNardeCheckerMove> moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, a);
          for (auto &m : moves)
          {
            if (m.pos == kPassPos)
              continue;
            SPIEL_CHECK_TRUE(m.to_pos <= m.pos || m.to_pos == kBearOffPos);
          }
        }

        // Black: check moves are not forward (engine ensures correct direction)
        std::vector<std::vector<int>> board_black_move = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        SetupBoardState(lnstate, kOPlayerId, board_black_move);
        SetupDice(lnstate, {3, 2, 0, 0});
        std::vector<Action> black_actions = lnstate->LegalActions();
        for (Action a : black_actions)
        {
          std::vector<LongNardeCheckerMove> moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, a);
          for (auto &m : moves)
          {
            if (m.pos == kPassPos)
              continue;
            // Ensure the destination point follows the CCW direction modulo board size,
            // handling wrap-around from kNumPoints-1 to 0, or it's a bear-off move.
            SPIEL_CHECK_TRUE(m.to_pos == (m.pos - m.die + kNumPoints) % kNumPoints || m.to_pos == kBearOffPos);
          }
        }
        std::cout << "✓ MovementDirectionTest passed\n";
      }
      // EndTest: test-movementdirection-1
      // EndFunction: MovementDirectionTest

      //------------------------------------------------------------------------------
      // StartFunction: NoLandingOnOpponentTestWhite
      // Test: NoLandingOnOpponentTestWhite
      // Ensures that a move cannot land on a point occupied by an opponent's checker (White's turn).
      //------------------------------------------------------------------------------
      // StartTest: test-nolanding-1
      void NoLandingOnOpponentTestWhite()
      {
        std::cout << "\n=== Running NoLandingOnOpponentTestWhite ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        // White: cannot land on Black's checker at point 16
        std::vector<std::vector<int>> board_white_no_land = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}};
        SetupBoardState(lnstate, kXPlayerId, board_white_no_land);
        SetupDice(lnstate, {4, 2, 0, 0});
        std::vector<Action> la = lnstate->LegalActions();
        bool found_move_landing_16 = false;
        for (Action a : la)
        {
          auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, a);
          for (auto &m : moves)
          {
            if (m.to_pos == 15)
            { // Point 16 has index 15
              found_move_landing_16 = true;
              break;
            }
          }
          if (found_move_landing_16)
            break;
        }
        SPIEL_CHECK_FALSE(found_move_landing_16);

        LongNardeCheckerMove white_move_attempt(19, 15, 4); // Attempt to move from point 20 (idx 19) to point 16 (idx 15)
        bool is_valid = lnstate->LongNardeIsValidCheckerMove(kXPlayerId, white_move_attempt, false);
        SPIEL_CHECK_FALSE(is_valid);

        std::cout << "✓ NoLandingOnOpponentTestWhite passed\n";
      }
      // EndTest: test-nolanding-1
      // EndFunction: NoLandingOnOpponentTestWhite

      //------------------------------------------------------------------------------
      // StartFunction: NoLandingOnOpponentTestBlack
      // Test: NoLandingOnOpponentTestBlack
      // Ensures that a move cannot land on a point occupied by an opponent's checker (Black's turn).
      //------------------------------------------------------------------------------
      // StartTest: test-nolanding-2
      void NoLandingOnOpponentTestBlack()
      {
        std::cout << "\n=== Running NoLandingOnOpponentTestBlack ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        // Black: cannot land on White's checker at point 13
        std::vector<std::vector<int>> board_black_no_land = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}};
        SetupBoardState(lnstate, kOPlayerId, board_black_no_land);
        SetupDice(lnstate, {3, 1, 0, 0});
        std::vector<Action> la = lnstate->LegalActions();
        bool found_black_landing_on_white = false;
        for (Action a : la)
        {
          auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, a);
          for (auto &m : moves)
          {
            // Attempt to move from point 16 (idx 15) to point 13 (idx 12) with die 3
            if (m.pos == 15 && m.die == 3 && m.to_pos == 12)
            {
              found_black_landing_on_white = true;
              break;
            }
          }
          if (found_black_landing_on_white)
            break;
        }
        SPIEL_CHECK_FALSE(found_black_landing_on_white);

        LongNardeCheckerMove black_move_attempt(15, 12, 3); // Attempt to move from point 16 (idx 15) to point 13 (idx 12)
        bool is_valid = lnstate->LongNardeIsValidCheckerMove(kOPlayerId, black_move_attempt, false);
        SPIEL_CHECK_FALSE(is_valid);

        std::cout << "✓ NoLandingOnOpponentTestBlack passed\n";
      }
      // EndTest: test-nolanding-2
      // EndFunction: NoLandingOnOpponentTestBlack

      //------------------------------------------------------------------------------
      // StartFunction: TestIllegalLandingInLegalActions
      // Test: IllegalLandingInLegalActions
      // Verifies that LegalActions does not generate moves landing on occupied points (regression test for random_sim_test bug).
      //------------------------------------------------------------------------------
      // StartTest: test-illegallanding-1
      void TestIllegalLandingInLegalActions()
      {
        std::cout << "\n=== Running TestIllegalLandingInLegalActions ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        // Regression test: ensure no move lands on an occupied point (see random_sim_test bug)
        std::vector<std::vector<int>> board_setup_with_scores = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        SetupBoardState(lnstate, kOPlayerId, board_setup_with_scores);
        SetupDice(lnstate, {1, 1, 1, 1});
        std::vector<Action> legal_actions = lnstate->LegalActions();
        SPIEL_CHECK_FALSE(legal_actions.empty());
        bool found_illegal_landing = false;
        int illegal_from_pos = 13;
        int illegal_to_pos = 12;
        int illegal_die = 1;
        for (Action action : legal_actions)
        {
          std::vector<LongNardeCheckerMove> moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, action);
          for (const auto &move : moves)
          {
            if (move.pos == illegal_from_pos && move.die == illegal_die && move.to_pos == illegal_to_pos)
            {
              found_illegal_landing = true;
              break;
            }
          }
          if (found_illegal_landing)
            break;
        }
        SPIEL_CHECK_FALSE(found_illegal_landing);
        std::cout << "✓ TestIllegalLandingInLegalActions passed (no illegal landings found)\n";
      }
      // EndTest: test-illegallanding-1
      // EndFunction: TestIllegalLandingInLegalActions

      // StartFunction: TestHalfMoveGeneration
      //  Test to verify that half-move generation produces correct moves
      // StartTest: test-halfmove-1
      void TestHalfMoveGeneration()
      {
        std::cout << "\n=== Running TestHalfMoveGeneration ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        // Setup: White has 1 at point 1 (idx 0) and 14 on head (idx 23). Black has 15 on head (idx 11).
        std::vector<std::vector<int>> test_board_with_scores = {
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
        SetupBoardState(lnstate, kXPlayerId, test_board_with_scores);
        SetupDice(lnstate, {3, 5, 0, 0});

        // Expecting only two half-moves: from point 24 with die 3 and die 5
        std::set<LongNardeCheckerMove> half_moves = lnstate->LongNardeGenerateAllHalfMoves(kXPlayerId, false);

        bool found_point24_die3 = false;
        bool found_point24_die5 = false;
        for (const auto &move : half_moves)
        {
          if (move.pos == 23 && move.die == 3)
            found_point24_die3 = true;
          if (move.pos == 23 && move.die == 5)
            found_point24_die5 = true;
        }
        SPIEL_CHECK_EQ(half_moves.size(), 2);
        SPIEL_CHECK_TRUE(found_point24_die3);
        SPIEL_CHECK_TRUE(found_point24_die5);

        // All legal actions should use at least one valid half-move
        std::vector<Action> legal_actions = lnstate->LegalActions();
        SPIEL_CHECK_GE(legal_actions.size(), 1);
        bool all_valid = true;
        for (Action action : legal_actions)
        {
          std::vector<LongNardeCheckerMove> moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
          bool action_valid = false;
          for (const auto &move : moves)
          {
            if (move.pos != kPassPos && half_moves.count(move) > 0)
            {
              action_valid = true;
              break;
            }
          }
          if (!action_valid)
          {
            all_valid = false;
            break;
          }
        }
        SPIEL_CHECK_TRUE(all_valid);
        std::cout << "✓ TestHalfMoveGeneration passed\n";
      }
      // EndTest: test-halfmove-1
      // EndFunction: TestHalfMoveGeneration

      //------------------------------------------------------------------------------
      // StartFunction: HeadRuleTestBlackFirstTurn
      // Test: HeadRuleTestBlack
      // Tests the head rule for Black player in both first-turn and non-first-turn scenarios
      //------------------------------------------------------------------------------
      // StartTest: test-headruleblack-1
      void HeadRuleTestBlackFirstTurn()
      {
        std::cout << "\n=== Running HeadRuleTestBlackFirstTurn ===\n";

        {
          // (A) Black first turn: should be able to move 2 from head with special doubles
          std::shared_ptr<const Game> game = LoadGame("long_narde");
          std::unique_ptr<State> stA = game->NewInitialState();
          auto lnA = static_cast<LongNardeState *>(stA.get());

          lnA->ApplyAction(0); // White roll
          auto white_actions = lnA->LegalActions();
          SPIEL_CHECK_FALSE(white_actions.empty());
          lnA->ApplyAction(white_actions[0]);

          SPIEL_CHECK_TRUE(lnA->IsChanceNode());
          lnA->ApplyAction(20); // Black rolls 6,6

          SPIEL_CHECK_EQ(lnA->CurrentPlayer(), kOPlayerId);
          SPIEL_CHECK_TRUE(lnA->IsFirstTurn(kOPlayerId));

          std::vector<Action> black_first_turn_actions = lnA->LegalActions();
          bool can_move_2_from_head = false;
          for (Action a : black_first_turn_actions)
          {
            std::unique_ptr<State> c = lnA->Clone();
            auto cst = static_cast<LongNardeState *>(c.get());
            int init_head_count = cst->board(kOPlayerId, kBlackHeadPos);
            cst->ApplyAction(a);
            int new_head_count = cst->board(kOPlayerId, kBlackHeadPos);
            int diff = init_head_count - new_head_count;
            if (diff >= 2)
            {
              can_move_2_from_head = true;
              break;
            }
          }
          SPIEL_CHECK_TRUE(can_move_2_from_head);
        }
        std::cout << "✓ Black head rule first-turn test passed\n";
      }
      // EndTest: test-headruleblack-1
      // EndFunction: HeadRuleTestBlackFirstTurn

      //------------------------------------------------------------------------------
      // StartFunction: HeadRuleTestBlackNonFirstTurn
      // Test: HeadRuleTestBlack
      // Tests the head rule for Black player in non-first-turn scenarios
      //------------------------------------------------------------------------------
      // StartTest: test-headruleblack-2
      void HeadRuleTestBlackNonFirstTurn()
      {
        std::cout << "\n=== Running HeadRuleTestBlackNonFirstTurn ===\n";

        {
          // (B) Black non-first turn: should not be able to move 2 from head
          std::shared_ptr<const Game> game = LoadGame("long_narde");
          std::unique_ptr<State> stB = game->NewInitialState();
          auto lnB = static_cast<LongNardeState *>(stB.get());
          std::vector<std::vector<int>> board_non_first = {
              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}, // White: 1@idx22, 14@idx23 (head)
              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}  // Black: 14@idx11 (head), 1@idx15 (pt 16)
          };
          SetupBoardState(lnB, kOPlayerId, board_non_first);
          SetupDice(lnB, {4, 4, 4, 4});
          SPIEL_CHECK_FALSE(lnB->IsFirstTurn(kOPlayerId));
          std::vector<Action> la = lnB->LegalActions();
          SPIEL_CHECK_FALSE(la.empty());

          // 1. Check: No move uses more than 1 checker from the head.
          bool found_illegal_2_from_head = false;
          for (Action move : la)
          {
            std::unique_ptr<State> c = lnB->Clone();
            auto cst = static_cast<LongNardeState *>(c.get());
            int init_head_count = cst->board(kOPlayerId, kBlackHeadPos);
            cst->ApplyAction(move);
            int new_head_count = cst->board(kOPlayerId, kBlackHeadPos);
            int diff = init_head_count - new_head_count;
            if (diff > 1)
            {
              found_illegal_2_from_head = true;
              std::cerr << "Illegal move found (moved >1 from head): "
                        << lnB->ActionToString(kOPlayerId, move) << "\n";
              break;
            }
          }
          SPIEL_CHECK_FALSE(found_illegal_2_from_head);

          // 2. Check: Exactly one legal move sequence exists.
          SPIEL_CHECK_EQ(la.size(), 1);

          // 3. Check: The single legal move sequence is exactly (11->7, 7->3).
          if (!la.empty())
          {
            Action the_action = la[0];
            std::vector<LongNardeCheckerMove> actual_moves_vec = lnB->LongNardeSpielMoveToCheckerMoves(kOPlayerId, the_action);
            // Convert to set to ignore order and filter out pass moves easily
            std::set<LongNardeCheckerMove> actual_moves;
            for (const auto &m : actual_moves_vec)
            {
              if (m.pos != kPassPos)
              {
                actual_moves.insert(m);
              }
            }

            std::set<LongNardeCheckerMove> expected_moves = {
                {11, 7, 4}, // Move from head
                {7, 3, 4}   // Move from intermediate point
            };

            SPIEL_CHECK_EQ(actual_moves.size(), 2);
            SPIEL_CHECK_EQ(actual_moves, expected_moves);
          }
        }
        std::cout << "✓ Black head rule non-first-turn test passed (incl. specific move check)\n";
      }
      // EndTest: test-headruleblack-2
      // EndFunction: HeadRuleTestBlackNonFirstTurn

      // StartFunction: TestHalfMoveGenerationBlack
      //  Test to verify that half-move generation produces correct moves for Black player
      // StartTest: test-halfmoveblack-1
      void TestHalfMoveGenerationBlack()
      {
        std::cout << "\n=== Running TestHalfMoveGenerationBlack ===\n";

        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        auto lnstate = static_cast<LongNardeState *>(state.get());

        // Setup: White 15 on head, Black 14 on head and 1 on point 17.
        std::vector<std::vector<int>> test_board_with_scores = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}, // White: 15@idx23 (head)
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}  // Black: 14@idx11 (head), 1@idx16 (pt 17)
        };
        SetupBoardState(lnstate, kOPlayerId, test_board_with_scores);
        SetupDice(lnstate, {4, 2, 0, 0});

        std::set<LongNardeCheckerMove> half_moves = lnstate->LongNardeGenerateAllHalfMoves(kOPlayerId, false);
        SPIEL_CHECK_EQ(half_moves.size(), 4); // Head(11) -> 7(d4), 9(d2); Point 17(16) -> 12(d4), 14(d2)

        std::vector<Action> legal_actions = lnstate->LegalActions();
        SPIEL_CHECK_FALSE(legal_actions.empty()); // Ensure at least one legal move exists.
        std::cout << "✓ TestHalfMoveGenerationBlack passed\n";
      }
      // EndTest: test-halfmoveblack-1
      // EndFunction: TestHalfMoveGenerationBlack

    } // namespace testing_internal

    //------------------------------------------------------------------------------
    // Master test function that runs all the above movement tests in one go.
    //------------------------------------------------------------------------------
    // StartFunction: TestMovementRules
    // StartTest: test-movementrules-1
    void TestMovementRules()
    {
      std::cout << "\n=== Running Movement Rules Tests ===\n";
      testing_internal::TestBasicMovement();
      testing_internal::InitialDiceTest();
      testing_internal::CheckerDistributionTest();
      testing_internal::HeadRuleTestFirstTurn();
      testing_internal::HeadRuleTestNonFirstTurn();
      testing_internal::MovementDirectionTest();
      testing_internal::NoLandingOnOpponentTestWhite();
      testing_internal::NoLandingOnOpponentTestBlack();
      testing_internal::TestIllegalLandingInLegalActions();
      testing_internal::TestHalfMoveGeneration();
      testing_internal::TestHalfMoveGenerationBlack();
      std::cout << "\n✓ All movement rules tests passed\n";
    }
    // EndTest: test-movementrules-1
    // EndFunction: TestMovementRules

  } // namespace long_narde
} // namespace open_spiel