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
      double_turn_(false),
      is_first_turn_(false),
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

/**
 * @brief Updates the internal dice_ member based on a chance outcome index.
 *
 * Long Narde uses predetermined chance outcomes (pairs of dice rolls).
 * This function looks up the dice pair corresponding to the outcome index
 * and stores them in the dice_ vector, ensuring the higher die is first.
 *
 * @param outcome The index into the kChanceOutcomeValues table (0-35).
 */
void LongNardeState::RollDice(int outcome) {
  SPIEL_CHECK_GE(outcome, 0);
  SPIEL_CHECK_LT(outcome, kChanceOutcomeValues.size());
  int die1 = kChanceOutcomeValues[outcome][0];
  int die2 = kChanceOutcomeValues[outcome][1];
  
  // Store dice values (convention: higher die first if different)
  dice_.clear(); // Ensure dice vector is empty before adding
  if (die1 != die2 && die1 < die2) {
    dice_.push_back(die2); // Higher die first
    dice_.push_back(die1);
  } else {
    dice_.push_back(die1); // If equal or die1 > die2
    dice_.push_back(die2);
  }
}

/**
 * @brief Gets the face value of a die from the internal dice_ vector.
 *
 * The internal dice_ vector may store values 7-12 to indicate a used die.
 * This function returns the actual face value (1-6) regardless of whether
 * the die has been marked as used.
 *
 * @param i The index of the die (0 or 1).
 * @return The face value of the die (1-6).
 */
int LongNardeState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());
  int raw_value = dice_[i];
  if (raw_value >= 1 && raw_value <= 6) {
    return raw_value; // Die is usable
  } else if (raw_value >= 7 && raw_value <= 12) {
    return raw_value - 6; // Die is marked used, return its face value
  } else {
    SpielFatalError(absl::StrCat("Bad dice value encountered in DiceValue(): ", raw_value));
    return 0; // Should be unreachable
  }
}

bool LongNardeState::UsableDiceOutcome(int outcome) const {
   // Checks if the *raw* value stored in dice_ represents a usable die (1-6)
  return outcome >= 1 && outcome <= 6;
}

} // namespace long_narde
} // namespace open_spiel 