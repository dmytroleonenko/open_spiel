#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_

#include <vector>
#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
  namespace long_narde
  {

    // Common utility functions
    inline bool ActionsContains(const std::vector<Action> &legal_actions, Action action)
    {
      for (Action legal_action : legal_actions)
      {
        if (legal_action == action)
        {
          return true;
        }
      }
      return false;
    }

    // Exposed test functions - these are the main entry points for each test category
    void TestMovementRules();    
    void TestBridgeFormation();  
    void TestActionEncoding();   
    void TestEndgame();          
    void TestHeadRule();         
    void TestPassMoveBehavior(); 

    void TestBasicMovement();

    // Helper function to create a Long Narde game state
    std::unique_ptr<LongNardeState> CreateStateFromString(const std::string &board_string);

    // Helper functions for setting up test states
    // Sets the board configuration and scores for a given state.
    inline void SetupBoardState(LongNardeState *state, Player player,
                                const std::vector<std::vector<int>> &board_config)
    {
      SPIEL_CHECK_TRUE(state != nullptr);
      SPIEL_CHECK_EQ(board_config.size(), kNumPlayers);
      SPIEL_CHECK_EQ(board_config[0].size(), kNumPoints); 
      SPIEL_CHECK_EQ(board_config[1].size(), kNumPoints); 

      state->board_ = board_config;
      state->scores_[0] = kNumCheckersPerPlayer - std::accumulate(state->board_[0].begin(), state->board_[0].end(), 0);
      state->scores_[1] = kNumCheckersPerPlayer - std::accumulate(state->board_[1].begin(), state->board_[1].end(), 0);
      state->cur_player_ = player;
      state->moved_from_head_ = false; 
      state->turns_ = 0; 
    }

    // Sets the dice roll for a given state.
    inline void SetupDice(LongNardeState *state, const std::vector<int> &dice)
    {
      SPIEL_CHECK_TRUE(state != nullptr);
      SPIEL_CHECK_EQ(dice.size(), 4); 

      state->dice_ = dice; 

      state->moves_remaining_ = 0;
      for (int d : dice)
      {
        if (d > 0)
        {
          state->moves_remaining_++;
        }
      }

      state->initial_dice_.clear();
      if (dice.size() >= 1)
        state->initial_dice_.push_back(dice[0]);
      if (dice.size() >= 2)
        state->initial_dice_.push_back(dice[1]);
      while (state->initial_dice_.size() < 2)
      {
        state->initial_dice_.push_back(0);
      }
    }

    void TestPassMoveBehavior();
    void TestDoubleMove();
    void TestPartialMoveBlocked();

  } // namespace long_narde
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_