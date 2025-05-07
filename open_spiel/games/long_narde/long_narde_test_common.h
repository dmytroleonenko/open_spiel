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

    // Common constants for tests
    // constexpr int kNumPoints = 24; // Defined in long_narde.h
    // constexpr int kPassPos = -1;   // Defined in long_narde.h

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
    // void TestBasicSetup();     // Basic setup and initialization tests
    void TestMovementRules();    // Movement rules tests
    void TestBridgeFormation();  // Bridge formation tests
    void TestActionEncoding();   // Action encoding tests
    void TestEndgame();          // Endgame tests including bearing off
    void TestHeadRule();         // Head rule tests
    void TestPassMoveBehavior(); // Test pass move behavior

    // Add declaration for the new test function
    // void TestSimpleNonDoubleMove();

    // Original function for backward compatibility
    // void BasicLongNardeTests();

    // TestBasicMovement is implemented in long_narde_test_movement.cc
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

      // Verify that no point has checkers from both players
      for (int i = 0; i < kNumPoints; ++i) {
        SPIEL_CHECK_FALSE(state->board_[kXPlayerId][i] > 0 && state->board_[kOPlayerId][i] > 0);
      }
    }

    // Sets the dice roll for a given state.
    inline void SetupDice(LongNardeState *state, const std::vector<int> &dice)
    {
      SPIEL_CHECK_TRUE(state != nullptr);
      SPIEL_CHECK_EQ(dice.size(), 4); // Input vector MUST have 4 elements

      // Correctly set initial_dice_ as a 4-element vector
      state->initial_dice_.assign(4, 0); // Initialize with four 0s
      if (dice[0] > 0 && dice[0] == dice[1] && dice[0] == dice[2] && dice[0] == dice[3]) {
        // Doubles roll: copy all four dice values
        state->initial_dice_[0] = dice[0];
        state->initial_dice_[1] = dice[1];
        state->initial_dice_[2] = dice[2];
        state->initial_dice_[3] = dice[3];
      } else if (dice[0] > 0 && dice[1] > 0) {
        // Non-doubles roll (or incomplete doubles, treat as non-double for initial_dice purpose)
        // Store the actual first two dice values, others remain 0
        state->initial_dice_[0] = dice[0]; // Higher die by convention if non-double
        state->initial_dice_[1] = dice[1]; // Lower die by convention if non-double
        // initial_dice_[2] and initial_dice_[3] remain 0
      } else if (dice[0] > 0) {
        // Only one die value provided in the first two slots (should not happen for valid test setups)
        state->initial_dice_[0] = dice[0];
        // initial_dice_[1], initial_dice_[2], initial_dice_[3] remain 0
      }
      // If dice[0] is 0, initial_dice_ remains {0,0,0,0} (e.g. before any roll)

      state->dice_ = dice;

      // Reset phase-specific state before generating sequences, similar to ProcessChanceRoll
      state->is_handling_second_phase_of_doubles_ = false;
      state->head_move_occurred_this_full_turn_ = false;
      state->first_phase_selected_move1_ = {kPassPos, kPassPos, 0};
      state->first_phase_selected_move2_ = {kPassPos, kPassPos, 0};
      // current_turn_full_legal_sequences_cache_ will be cleared/rebuilt by GenerateAndCacheFullLegalSequences() below

      // Populate full legal sequences cache so LegalActions works correctly in tests
      state->GenerateAndCacheFullLegalSequences();
    }

    // Test functions from long_narde_test_pass.cc (or similar)
    void TestPassMoveBehavior();

    // Test functions from long_narde_test_movegen_comparison.cc
    // void TestSimpleNonDoubleMove();
    void TestDoubleMove();
    void TestPartialMoveBlocked();

  } // namespace long_narde
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_