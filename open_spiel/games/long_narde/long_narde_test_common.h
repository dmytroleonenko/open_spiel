#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_

#include <vector>
#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

// Common constants for tests
// constexpr int kNumPoints = 24; // Defined in long_narde.h
// constexpr int kPassPos = -1;   // Defined in long_narde.h

// Common utility functions
inline bool ActionsContains(const std::vector<Action>& legal_actions, Action action) {
  for (Action legal_action : legal_actions) {
    if (legal_action == action) {
      return true;
    }
  }
  return false;
}

// Exposed test functions - these are the main entry points for each test category
void TestBasicSetup();     // Basic setup and initialization tests
void TestMovementRules();  // Movement rules tests
void TestBridgeFormation(); // Bridge formation tests
void TestActionEncoding(); // Action encoding tests
void TestEndgame();        // Endgame tests including bearing off
void TestHeadRule();       // Head rule tests
void TestPassMoveBehavior(); // Test pass move behavior

// Original function for backward compatibility
void BasicLongNardeTests();

// TestBasicMovement is implemented in long_narde_test_movement.cc
void TestBasicMovement();

// Helper function to create a Long Narde game state
std::unique_ptr<LongNardeState> CreateStateFromString(const std::string& board_string);

// Helper functions for setting up test states
// Sets the board configuration and scores for a given state.
inline void SetupBoardState(LongNardeState* state, Player player,
                            const std::vector<std::vector<int>>& board_config,
                            const std::vector<int>& scores) {
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_EQ(board_config.size(), kNumPlayers);
  // Ensure the input board config has the correct size (including score position)
  SPIEL_CHECK_EQ(board_config[0].size(), kNumPoints + 1);
  SPIEL_CHECK_EQ(board_config[1].size(), kNumPoints + 1);
  SPIEL_CHECK_EQ(scores.size(), kNumPlayers);

  // Direct manipulation (allowed via friend declaration in LongNardeState)
  state->board_ = board_config;
  state->scores_ = scores;
  state->cur_player_ = player;
  // Reset turn-specific flags that SetState would normally handle
  // These might need adjustment based on specific test needs
  state->is_first_turn_ = false; // Default assumption for tests using this helper
  state->moved_from_head_ = false; // Default assumption
}

// Sets the dice roll and double turn status for a given state.
inline void SetupDice(LongNardeState* state, const std::vector<int>& dice,
                      bool double_turn) {
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_LE(dice.size(), 2); // Max 2 dice values

  // Direct manipulation (allowed via friend declaration)
  state->dice_.assign(dice.begin(), dice.end());
  // Ensure dice_ always has size 2, padding with 0 if necessary
  while (state->dice_.size() < 2) {
      state->dice_.push_back(0);
  }
  state->double_turn_ = double_turn;
}

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_