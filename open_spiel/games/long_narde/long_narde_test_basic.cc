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

        // Apply the first chance outcome (dice roll) to resolve the initial player.
        if (state->IsChanceNode())
        {
          auto outcomes = state->ChanceOutcomes();
          state->ApplyAction(outcomes[0].first);
        }

        auto lnstate = static_cast<const LongNardeState *>(state.get());
        SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
      }
      // EndTest: test-white-moves-first
      // EndFunction: WhiteMovesFirstTest
    } // End anonymous namespace

  } // namespace long_narde
} // namespace open_spiel