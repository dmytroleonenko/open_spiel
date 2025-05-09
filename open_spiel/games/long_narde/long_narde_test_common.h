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
      // Ensure the input board config has the correct size (only indices 0-23)
      SPIEL_CHECK_EQ(board_config[0].size(), kNumPoints); // kNumPoints is 24
      SPIEL_CHECK_EQ(board_config[1].size(), kNumPoints); // kNumPoints is 24

      // Copy board_config into board_data_
      for (int p = 0; p < kNumPlayers; ++p) {
        for (int pos = 0; pos < kNumPoints; ++pos) {
          state->board_data_[p * kNumPoints + pos] = static_cast<uint8_t>(board_config[p][pos]);
        }
      }
      // Calculate scores internally from board_data_
      int sum0 = 0, sum1 = 0;
      for (int pos = 0; pos < kNumPoints; ++pos) {
        sum0 += state->board_data_[0 * kNumPoints + pos];
        sum1 += state->board_data_[1 * kNumPoints + pos];
      }
      state->scores_[0] = kNumCheckersPerPlayer - sum0;
      state->scores_[1] = kNumCheckersPerPlayer - sum1;
      state->cur_player_ = player;
      // Reset turn-specific flags that SetState would normally handle
      state->moved_from_head_ = false; // Default assumption
      // Reset outcome_ to indicate game is not over and turns_
      state->turns_ = 0; // Assuming 0 is a reasonable reset value for turns
    }

    // Sets the dice roll for a given state.
    inline void SetupDice(LongNardeState *state, const std::vector<int> &dice)
    {
      SPIEL_CHECK_TRUE(state != nullptr);
      SPIEL_CHECK_EQ(dice.size(), 4); // Input vector MUST have 4 elements

      // *** ADDED DEBUG LOG ***
      if (kDebugging)
      {
        std::cout << "[DEBUG SetupDice] Input dice: {"
                  << (dice.size() > 0 ? std::to_string(dice[0]) : "?") << ", "
                  << (dice.size() > 1 ? std::to_string(dice[1]) : "?") << ", "
                  << (dice.size() > 2 ? std::to_string(dice[2]) : "?") << ", "
                  << (dice.size() > 3 ? std::to_string(dice[3]) : "?") << "}\n";
      }
      // *** END ADDED DEBUG LOG ***

      // Directly set dice_ slots (allows two-phase encoding for doubles)
      state->dice_ = dice; // Assign the full 4-element input vector

      // Initialize initial_dice_ for two-phase handling
      state->initial_dice_ = dice; // Preserve all four dice for doubles
      // For non-doubles, initial_dice_[2] and [3] may be 0

      // Set doubles phase flag
      state->is_first_phase_of_doubles_ = (dice[2] != 0);

      // *** ADDED DEBUG LOG ***
      if (kDebugging)
      {
        std::cout << "[DEBUG SetupDice] Assigned state->dice_: {"
                  << (state->dice_.size() > 0 ? std::to_string(state->dice_[0]) : "?") << ", "
                  << (state->dice_.size() > 1 ? std::to_string(state->dice_[1]) : "?") << ", "
                  << (state->dice_.size() > 2 ? std::to_string(state->dice_[2]) : "?") << ", "
                  << (state->dice_.size() > 3 ? std::to_string(state->dice_[3]) : "?") << "}\n";
      }
      // *** END ADDED DEBUG LOG ***

      // Note: We no longer truncate to two elements; tests rely on full 4-element initial_dice_
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