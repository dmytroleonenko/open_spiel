#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel_utils.h"
#include <vector>
#include <memory>    // For std::unique_ptr, std::make_unique
#include <iostream>  // For std::cout (used in Clone #ifndef NDEBUG)
#include "open_spiel/abseil-cpp/absl/types/span.h"

namespace open_spiel {
namespace long_narde {

// ===== Core Spiel API Implementations =====

/**
 * @brief Returns the player whose turn it is.
 *
 * Returns kTerminalPlayerId if the game has ended.
 *
 * @return The player ID (0 or 1), or kTerminalPlayerId.
 */
Player LongNardeState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

/**
 * @brief Applies the given action (Spiel move) to the current state.
 *
 * Handles both chance node rolls and player moves.
 * For player moves, it decodes the Spiel action into checker moves, validates
 * them (including head rule checks), applies the valid moves, updates turn
 * history, and advances the game state (player turn, dice, etc.).
 * Grants an extra turn if doubles were rolled and it wasn't already an extra turn.
 *
 * @param move_id The encoded Spiel action to apply.
 */
void LongNardeState::DoApplyAction(Action move_id) {
  if (IsChanceNode()) {
    ProcessChanceRoll(move_id);
    return;
  }

  bool currently_extra = is_playing_extra_turn_;

  std::vector<CheckerMove> original_moves = SpielMoveToCheckerMoves(cur_player_, move_id);
  std::vector<CheckerMove> filtered_moves;
  int head_pos = (cur_player_ == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;
  bool used_head_move = false;
  
  for (const auto& m : original_moves) {
    if (m.pos == kPassPos) {
      filtered_moves.push_back(m);
      continue;
    }

    // Allow second head move only if:
    // (A) It is the first turn AND dice are double 6, 4, or 3, OR
    // (B) Not first turn => no second head move.
    // This check remains as a safeguard, although LegalActions should prevent invalid sequences.
    if (IsHeadPos(cur_player_, m.pos) && used_head_move) {
      // Use is_on_first_turn_ member variable and initial_dice_
      if (is_on_first_turn_) { 
        // Check initial_dice_ for special doubles
        bool is_special_double = false;
        if (initial_dice_.size() >= 2 && initial_dice_[0] > 0 && initial_dice_[0] == initial_dice_[1]) {
            int dieVal = initial_dice_[0];
            if (dieVal == 6 || dieVal == 4 || dieVal == 3) {
                is_special_double = true;
            }
        }
        if (!is_special_double) {
          // This move is invalid in the sequence, replace with Pass
          // Note: LegalActions should ideally not generate such sequences.
          filtered_moves.push_back(kPassMove);
          continue;
        }
      } else {
        // Normal turns: only one checker can leave the head.
        // Replace invalid second head move with Pass.
        filtered_moves.push_back(kPassMove);
        continue;
      }
    }
    if (IsHeadPos(cur_player_, m.pos)) {
      used_head_move = true;
      // moved_from_head_ is set within ApplyCheckerMove now
    }
    filtered_moves.push_back(m); // Add the original move if it passed checks
  }
  
  // Apply all valid moves from the filtered sequence
  for (const auto& m : filtered_moves) {
    if (m.pos != kPassPos) {
      // ApplyCheckerMove internally checks validity again (without head rule)
      // and sets moved_from_head_
      ApplyCheckerMove(cur_player_, m);
    }
  }

  // Store the state before applying the move for undo purposes.
  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, move_id,
                      moved_from_head_, currently_extra));

  // Clear dice *before* advancing to the next player/chance node
  dice_.clear();
  // Reset head move flag here as well, before AdvanceToNextPlayer might use it indirectly
  moved_from_head_ = false; 

  // Determine next player state (sets cur_player_ to kChancePlayerId and stores next actual player in prev_player_)
  AdvanceToNextPlayer(filtered_moves, move_id);

  // No need for further updates to cur_player_ or prev_player_ here, AdvanceToNextPlayer handles it.
  // No need to clear dice_ again, done above.
  // No need to reset moved_from_head_ again, done above.
}

/**
 * @brief Undoes the last action applied to the state.
 *
 * Restores the game state (player, dice, turn count, history flags) from the
 * most recent entry in the turn history.
 * If the action being undone was a player move, it also undoes the individual
 * checker moves in reverse order.
 *
 * @param player The player who made the action to undo.
 * @param action The Spiel action that was applied.
 */
void LongNardeState::UndoAction(Player player, Action action) {
  TurnHistoryInfo info = turn_history_info_.back();
  turn_history_info_.pop_back();

  // Restore state
  moved_from_head_ = info.moved_from_head;
  cur_player_ = info.player;
  prev_player_ = info.prev_player;
  dice_ = info.dice; // Should restore the 4-element dice correctly
  is_playing_extra_turn_ = info.is_playing_extra_turn;

  if (player == kChancePlayerId && info.dice.empty()) {
    cur_player_ = kChancePlayerId;
    prev_player_ = kChancePlayerId;
    turns_ = -1;
    return;
  }

  if (player != kChancePlayerId) {
    if (cur_player_ == kTerminalPlayerId) {
      cur_player_ = player;
    }
    std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(player, action);
    
    // Undo moves in reverse order
    for (int i = moves.size() - 1; i >= 0; --i) {
      UndoCheckerMove(player, moves[i]);
    }
    
    // Determine if the undone roll was doubles based on restored dice
    bool was_doubles = (info.dice.size() == 4 && info.dice[0] > 0 && 
                        info.dice[0] == info.dice[1] && 
                        info.dice[1] == info.dice[2] && 
                        info.dice[2] == info.dice[3]);

    // Only decrement turn counters if it wasn't a doubles turn
    // (since doubles don't increment the turn counter initially)
    if (!was_doubles) { 
      turns_--;
      if (player == kXPlayerId) {
        x_turns_--;
      } else if (player == kOPlayerId) {
        o_turns_--;
      }
    }
  }
}

/**
 * @brief Returns a string representation of the current game state from the perspective of the specified player.
 *
 * @param player The player for whom the observation is requested (0 or 1).
 * @return A string describing the current board state, dice, and scores.
 */
std::string LongNardeState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

/**
 * @brief Fills a provided tensor with the observation data for the specified player.
 *
 * The tensor encoding includes:
 * - Board representation from the player's perspective (24 values).
 * - Board representation from the opponent's perspective (24 values).
 * - Player's score (1 value).
 * - Opponent's score (1 value).
 * - Indicator for player's turn (1 value).
 * - Indicator for opponent's turn (1 value).
 * - Dice values (2 values, 0 if not rolled).
 * The total size is kStateEncodingSize.
 *
 * @param player The player for whom the observation tensor is requested (0 or 1).
 * @param values A span of floats to be filled with the observation data. Its size must equal kStateEncodingSize.
 */
void LongNardeState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int opponent = Opponent(player);
  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();

  // Board representation: Player's checkers perspective
  for (int i = 0; i < kNumPoints; ++i) {
      // Map board index i to the player's path index (0=farthest, 23=closest to home)
      int path_idx = GetPathIndex(player, i);
      *(values.begin() + path_idx) = board(player, i);
  }
  value_it += kNumPoints; // Move iterator past player's board section

  for (int i = 0; i < kNumPoints; ++i) {
       // Map board index i to the opponent's path index (0=farthest, 23=closest to home)
      int path_idx = GetPathIndex(opponent, i);
      *(values.begin() + kNumPoints + path_idx) = board(opponent, i);
  }
   value_it += kNumPoints; // Move iterator past opponent's board section

  // Scores and turn indicator
  *value_it++ = scores_[player];
  *value_it++ = scores_[opponent];
  *value_it++ = (cur_player_ == player) ? 1.0f : 0.0f; // Player's turn?
  *value_it++ = (cur_player_ == opponent) ? 1.0f : 0.0f; // Opponent's turn? (Should be redundant if not chance/terminal)

  // Dice (normalize? 0-6?) - Keep as raw values for now
  *value_it++ = (dice_.size() >= 1) ? DiceValue(0) : 0.0f;
  *value_it++ = (dice_.size() >= 2) ? DiceValue(1) : 0.0f;

  // Check if iterator reached the end
  SPIEL_CHECK_EQ(value_it, values.end());
}

/**
 * @brief Determines if the game has reached a terminal state.
 *
 * The game ends when either player has borne off all 15 checkers.
 * Special handling exists for the WinLossTie scoring rule, where the game
 * might not be terminal immediately if one player finishes, allowing the
 * opponent a potential last roll to tie (if they have 14 or 15 checkers off).
 *
 * @return True if the game is over, false otherwise.
 */
bool LongNardeState::IsTerminal() const {
  if (scores_[kXPlayerId] == kNumCheckersPerPlayer ||
      scores_[kOPlayerId] == kNumCheckersPerPlayer) {
    // Check for potential tie scenario if using that scoring rule
    if (scoring_type_ == ScoringType::kWinLossTieScoring) {
      // If White finished, but Black has 14 or 15 checkers borne off,
      // Black might get a last roll to tie. Game isn't terminal yet.
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer &&
          scores_[kOPlayerId] >= 14 && scores_[kOPlayerId] < kNumCheckersPerPlayer &&
          !allow_last_roll_tie_) { // Check if the tie roll has already been allowed/processed
            // This state might need refinement: how do we know if Black's *turn* is next?
            // If White just finished, cur_player_ should be Chance.
            // We need to ensure Black actually gets the chance roll.
             return false; // Potential tie possible, not terminal yet.
      }
       // Symmetrically for Black finishing
       if (scores_[kOPlayerId] == kNumCheckersPerPlayer &&
           scores_[kXPlayerId] >= 14 && scores_[kXPlayerId] < kNumCheckersPerPlayer &&
           !allow_last_roll_tie_) { // Check if the tie roll has already been allowed/processed
             return false; // Potential tie possible, not terminal yet.
       }
    }
    // If no tie is possible or the tie roll is handled, it's terminal.
    return true;
  }
  return false;
}

/**
 * @brief Returns the final scores for each player if the game is terminal.
 *
 * Returns {0, 0} if the game is not terminal.
 * Otherwise, returns the scores based on the game outcome:
 * - Tie (WinLossTieScoring only): {0, 0}
 * - White Wins (Mars): {2, -2}
 * - White Wins (Oin): {1, -1}
 * - Black Wins (Mars): {-2, 2}
 * - Black Wins (Oin): {-1, 1}
 *
 * @return A vector containing the final return for each player.
 */
std::vector<double> LongNardeState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  bool x_won = scores_[kXPlayerId] == kNumCheckersPerPlayer;
  bool o_won = scores_[kOPlayerId] == kNumCheckersPerPlayer;

  if (x_won && o_won) { // Tie occurred
    return {0.0, 0.0};
  } else if (x_won) {
    double score = (scores_[kOPlayerId] > 0) ? 1.0 : 2.0; // 1 for oin, 2 for mars
    return {score, -score};
  } else if (o_won) {
     double score = (scores_[kXPlayerId] > 0) ? 1.0 : 2.0; // 1 for oin, 2 for mars
     return {-score, score};
  } else {
      // Should not happen if IsTerminal() is true and it's not a tie
      SpielFatalError("Returns() called on non-terminal or inconsistent state.");
      return {0.0, 0.0};
  }
}

/**
 * @brief Returns the set of possible chance outcomes (dice rolls) and their probabilities.
 *
 * In Long Narde, the standard 30 non-double dice rolls (1-2, 1-3, ..., 5-6) are possible.
 * Doubles are excluded from the initial roll but handled during regular turns.
 * This function returns the fixed set of 30 non-double outcomes, each with equal probability.
 *
 * @return A vector of pairs, where each pair contains a chance action (encoded dice roll)
 *         and its probability (1.0 / 30.0).
 */
std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  // In Long Narde, the chance outcomes (dice rolls) are always the same,
  // regardless of whether it's the starting roll or a regular turn roll.
  return kChanceOutcomes;
}

/**
 * @brief Creates a deep copy of the current game state.
 *
 * Note: Currently implements a simple copy constructor. History management logic
 * previously present has been removed.
 *
 * @return A unique pointer to the newly created clone of the state.
 */
std::unique_ptr<State> LongNardeState::Clone() const {
#ifndef NDEBUG
    // std::cout << "Cloning state: " << ToString() << std::endl;
#endif
    return std::unique_ptr<State>(new LongNardeState(*this));
}

// ===== Chance Node Handling =====

/**
 * @brief Processes a chance outcome (dice roll) and updates the game state.
 *
 * This function is called when the current player is kChancePlayerId.
 * It takes the encoded chance action (dice roll), updates the internal dice_ state,
 * stores the roll history, determines the next player based on game rules (initial
 * turn, doubles extra turn, normal alternation), updates turn flags (is_first_turn_,
 * double_turn_, is_playing_extra_turn_), and handles the setup for a potential
 * last roll tie under the WinLossTieScoring rule.
 *
 * @param move_id The encoded chance action representing the dice roll.
 */
void LongNardeState::ProcessChanceRoll(Action move_id) {
  SPIEL_CHECK_GE(move_id, 0);
  SPIEL_CHECK_LT(move_id, game_->MaxChanceOutcomes());

  // Record the chance outcome in turn history.
  turn_history_info_.push_back(
      TurnHistoryInfo(kChancePlayerId, prev_player_, dice_, move_id,
                      moved_from_head_, is_playing_extra_turn_));

  // Ensure we have no dice set yet, then apply this new roll.
  SPIEL_CHECK_TRUE(dice_.empty());
  RollDice(move_id); // Sets dice_ based on outcome

  // *** Store the dice roll at the start of the turn ***
  initial_dice_ = dice_; 

  // Decide which player moves next.
  if (turns_ < 0) {
    // Initial state: White always starts in this implementation.
    turns_ = 0;
    cur_player_ = kXPlayerId; // White goes first
    is_playing_extra_turn_ = false; // Not an extra turn
    is_on_first_turn_ = true; // Set flag for X's first turn
  } else {
    // For subsequent turns, the player who should move was determined by 
    // AdvanceToNextPlayer and stored in prev_player_ by DoApplyAction.
    cur_player_ = prev_player_; 

    // is_playing_extra_turn_ was correctly set by AdvanceToNextPlayer
    // Set is_on_first_turn_ correctly for the upcoming player turn
    is_on_first_turn_ = (turns_ == 0 && cur_player_ == kXPlayerId) || (turns_ == 1 && cur_player_ == kOPlayerId);
  }

  prev_player_ = kChancePlayerId; // Mark that the previous phase *was* chance
  moved_from_head_ = false; // Reset for the start of the new player's turn

  // Check special condition for last-roll tie possibility
  // If player X just finished (score=15) and player O has 14 or 15 checkers off,
  // AND player O is the current player (meaning it's their turn to roll for the tie),
  // set the flag.
  if (scoring_type_ == ScoringType::kWinLossTieScoring) {
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer &&
          scores_[kOPlayerId] >= 14 && scores_[kOPlayerId] < kNumCheckersPerPlayer &&
          cur_player_ == kOPlayerId) { // Check if O is about to roll for the tie
        allow_last_roll_tie_ = true;
      }
      // Symmetrical check if O finished and X might tie
       else if (scores_[kOPlayerId] == kNumCheckersPerPlayer &&
                scores_[kXPlayerId] >= 14 && scores_[kXPlayerId] < kNumCheckersPerPlayer &&
                cur_player_ == kXPlayerId) { // Check if X is about to roll for the tie
         allow_last_roll_tie_ = true;
       } else {
         allow_last_roll_tie_ = false; // Reset if conditions not met
       }
  } else {
      allow_last_roll_tie_ = false; // Rule not active
  }
}

void LongNardeState::AdvanceToNextPlayer(const std::vector<CheckerMove>& applied_moves,
                                         Action spiel_action) {
  Player moving_player = cur_player_; // Player who just finished moving
  bool currently_extra = is_playing_extra_turn_;

  // Determine if the roll was doubles based on the initial roll for this turn
  // Use initial_dice_ as dice_ gets cleared before this is called.
  bool was_doubles_roll = (initial_dice_.size() == 4 && initial_dice_[0] > 0 && initial_dice_[0] == initial_dice_[1] && initial_dice_[1] == initial_dice_[2] && initial_dice_[2] == initial_dice_[3]);

  int num_actual_moves = 0;
  for(const auto& m : applied_moves) {
    if (m.pos != kPassPos) {
      num_actual_moves++;
    }
  }
  int potential_moves = was_doubles_roll ? 4 : 2;

  Player next_player_id; // Who will play AFTER the next dice roll?

  // Grant extra turn if doubles were rolled AND player wasn't already on an extra turn.
  // The number of actual moves made doesn't matter for granting the extra turn.
  if (was_doubles_roll && !currently_extra) {
    // Grant extra turn
    next_player_id = moving_player;
    is_playing_extra_turn_ = true;
    // is_on_first_turn_ remains unchanged
    // turns_ counter doesn't increment on doubles
  } else {
    // Normal alternation or end of extra turn
    next_player_id = NextPlayerRoundRobin(moving_player, num_players_);
    is_playing_extra_turn_ = false;

    // Only update flags if player actually changes
    if (next_player_id != moving_player) {
        is_on_first_turn_ = false;
        turns_++;
        if (next_player_id == kXPlayerId) x_turns_++; else o_turns_++;
    }
    // If player stays the same (end of extra turn), flags are already correct.
  }

  // Reset head move flag for the upcoming turn
  moved_from_head_ = false;

  // Store the player who will play *after* the chance node
  prev_player_ = next_player_id;

  // Set the current player to Chance to trigger dice roll on next ApplyAction
  cur_player_ = kChancePlayerId;
}

/**
 * @brief Checks if the game state represents an extra turn for the current player.
 * 
 * @return True if it is currently an extra turn, false otherwise.
 */
bool LongNardeState::IsExtraTurn() const {
  return is_playing_extra_turn_;
}

/**
 * @brief Returns the dice values as a string.
 *
 * Formats the current dice roll (e.g., "3 5", "6 6 (u)") indicating used dice.
 *
 * @return A string representation of the dice state.
 */
std::string LongNardeState::DiceToString() const {
  std::string dice_str = "";
  if (dice_.empty()) {
    dice_str = "(None)";
  } else {
    bool first_die = true;
    for (size_t i = 0; i < dice_.size(); ++i) {
      int die_val = DiceValue(i); // Get actual value (1-6)
      if (die_val == 0) continue; // Skip unused slots

      if (!first_die) dice_str += " ";
      dice_str += std::to_string(die_val);
      if (!IsDieUsable(i)) dice_str += "(u)";
      first_die = false;
    }
    if (first_die) { // Means all slots were 0 (shouldn't happen if dice_ not empty)
        dice_str = "(Invalid Dice State)";
    }
  }
  return dice_str;
}

void LongNardeState::RollDice(Action outcome) {
  SPIEL_CHECK_TRUE(dice_.empty());
  const std::vector<int>& roll = kChanceOutcomeValues[outcome];
  dice_.resize(4, 0); // Always resize to 4 elements, initialize with 0
  if (roll[0] == roll[1]) { // Doubles
    dice_[0] = roll[0];
    dice_[1] = roll[0]; // Keep index 1 for potential use in IsDieUsable logic if needed
    dice_[2] = roll[0];
    dice_[3] = roll[0];
  } else { // Not doubles
    // Ensure higher die is first
    if (roll[0] > roll[1]) {
        dice_[0] = roll[0];
        dice_[1] = roll[1];
    } else {
        dice_[0] = roll[1]; // Put higher die (roll[1]) first
        dice_[1] = roll[0]; // Put lower die (roll[0]) second
    }
    // dice_[2] and dice_[3] remain 0
  }
}

} // namespace long_narde
} // namespace open_spiel 