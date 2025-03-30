#include "open_spiel/games/long_narde/long_narde.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

/**
 * @brief Constructs a LongNardeState.
 *
 * Initializes the game state, including setting the scoring type based on game parameters,
 * and setting up the initial board configuration.
 *
 * @param game A shared pointer to the parent LongNardeGame object.
 */
LongNardeState::LongNardeState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1), // Initial turns count before first roll
      moved_from_head_(false),
      is_playing_extra_turn_(false),
      dice_({}),
      initial_dice_({}), // Initialize initial_dice_
      scores_({0, 0}),
      board_({std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
      turn_history_info_({}),
      allow_last_roll_tie_(false),
      // Initialize scoring_type_ based on game parameters
      scoring_type_(ParseScoringType(
          game->GetParameters().count("scoring_type") > 0 ?
          game->GetParameters().at("scoring_type").string_value() :
          kDefaultScoringType)) {
  SetupInitialBoard();
}

/**
 * @brief Sets up the initial checker positions on the board.
 *
 * Places 15 checkers for White (X) on point 24 (index 23) and
 * 15 checkers for Black (O) on point 12 (index 11).
 * All other points are initialized to 0 checkers.
 */
void LongNardeState::SetupInitialBoard() {
  board_[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  board_[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
}

// ===== Basic State Accessors =====

int LongNardeState::board(int player, int pos) const {
  // Bounds check for safety, returning 0 for invalid positions
  if (pos < 0 || pos >= kNumPoints) {
    return 0;
  }
  return board_[player][pos];
}

int LongNardeState::Opponent(int player) const { return 1 - player; }

} // namespace long_narde
} // namespace open_spiel 