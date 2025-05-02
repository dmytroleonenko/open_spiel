#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel
{
  namespace long_narde
  {
    namespace
    { // Start anonymous namespace

      // Utility: check if a specific action is in the legal actions list.
      bool ActionsContains(const std::vector<Action> &legal_actions, Action action)
      {
        return std::find(legal_actions.begin(), legal_actions.end(), action) !=
               legal_actions.end();
      }

      // StartFunction: InitialBoardSetupTest
      // StartTest: test-initial-board-setup
      void InitialBoardSetupTest()
      {
        std::shared_ptr<const Game> game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();
        LongNardeState *lnstate = static_cast<LongNardeState *>(state.get());

        // Verify initial board: White's 15 checkers on point 24, Black's 15 on point 12.
        SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), kNumCheckersPerPlayer);
        for (int i = 0; i < kNumPoints; ++i)
        {
          if (i != kWhiteHeadPos)
          {
            SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, i), 0);
          }
        }
        SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, kBlackHeadPos), kNumCheckersPerPlayer);
        for (int i = 0; i < kNumPoints; ++i)
        {
          if (i != kBlackHeadPos)
          {
            SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, i), 0);
          }
        }
      }
      // EndTest: test-initial-board-setup
      // EndFunction: InitialBoardSetupTest

      // StartFunction: WhiteMovesFirstTest
      // StartTest: test-white-moves-first
      void WhiteMovesFirstTest()
      {
        auto game = LoadGame("long_narde");
        std::unique_ptr<State> state = game->NewInitialState();

        // Force a nondouble dice roll (4,2) for deterministic two half-move turn.
        LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
        SetupDice(lnstate, {4, 2, 0, 0});

        // 1) Two half-moves should remain after our forced roll
        SPIEL_CHECK_EQ(lnstate->moves_remaining(), 2);
        // Current player should still be X
        SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);

        // 2) First half-move options (filtered by full-turn lookahead)
        auto legal1 = state->LegalActions();
        SPIEL_CHECK_EQ(legal1.size(), 1);
        SPIEL_CHECK_TRUE(ActionsContains(legal1, kWhiteHeadPos));

        // 3) Apply the selected half-move
        state->ApplyAction(legal1[0]);
        // Now only one half-move remains
        SPIEL_CHECK_EQ(lnstate->moves_remaining(), 1);

        // 4) Last half-move options: enumerate all remaining half-move sources
        auto legal2 = state->LegalActions();
        // Should present two distinct source positions (head and post-move head)
        SPIEL_CHECK_EQ(legal2.size(), 2);
        SPIEL_CHECK_TRUE(ActionsContains(legal2, legal1[0]));
        SPIEL_CHECK_NE(legal2[0], legal2[1]);
      }
      // EndTest: test-white-moves-first
      // EndFunction: WhiteMovesFirstTest
    } // End anonymous namespace

  } // namespace long_narde
} // namespace open_spiel