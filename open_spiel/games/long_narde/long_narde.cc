/**
 *   Long Narde Rules:
 *  1. Setup: White's 15 checkers on point 24; Black's 15 on point 12.
 *  2. Movement: Both move checkers CCW into home (White 1–6, Black 13–18), then bear off.
 *  3. Starting: Each rolls 1 die; higher is White and goes first. But in our open_spiel implementation white is always first without the dice roll
 *  4. Turns: Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves exist, skip; if only one is possible, use the higher die.
 *  5. Head Rule: Only 1 checker may leave the head (White 24, Black 12) per turn. Exception on the first turn: if you roll double 6, 4, or 3, you can move 2 checkers from the head; after that, no more head moves.
 *  6. Bearing Off: Once all your checkers reach home, bear them off with exact or higher rolls.
 *  7. Ending/Scoring: Game ends when someone bears off all. If the loser has none off, winner scores 2 (mars); otherwise 1 (oin). Some events allow a last roll to tie.
 *  8. Block (Bridge): You cannot form a contiguous block of 6 checkers unless at least 1 opponent checker is still ahead of it. Fully trapping all 15 opponent checkers is banned—even a momentary (going through in a sequence of moves) 6‑block that would leave no opponent checkers in front is disallowed.
 */
#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>
#include <queue>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace { // Anonymous namespace

// ===== Game Definition and Constants =====

// Facts about the game.
// Moved to long_narde_game.cc
// const GameType kGameType{...};

// Moved to long_narde_game.cc
// static std::shared_ptr<const Game> Factory(const GameParameters& params) { ... }

// Moved to long_narde_game.cc
// REGISTER_SPIEL_GAME(kGameType, Factory);

// Moved to long_narde_game.cc
// RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Chance outcome definitions (implementation detail) - Moved to long_narde.h
// Moved kChanceOutcomeValues definition to long_narde.h
// const std::vector<std::vector<int>> kChanceOutcomeValues = { ... };

// Other Implementation Constants
// Note: kBearOffPos and kNumNonDoubleOutcomes moved to long_narde.h as they relate more to game rules/structure.
// Moved to long_narde.h
// constexpr int kNumOffPosHumanReadable = -2; // Value used in string formatting for borne-off checkers.
constexpr int kNumBarPosHumanReadable = -3; // Value used in string formatting (legacy from backgammon, not used).

// ===== Encoding/Decoding Constants =====

// Long Narde uses a complex encoding scheme to represent a player's full turn
// (potentially involving multiple checker movements) as a single integer Action.
// There are two main schemes used:

// --- Scheme 1: Encoding for Non-Doubles Turns (or Doubles with <= 2 moves) ---
// Each individual half-move (moving one checker by one die's value) is encoded
// into a "digit".
// - Regular move: digit = `pos * 6 + (die - 1)`. `pos` is 0-23, `die` is 1-6.
//   Range: 0 * 6 + (1 - 1) = 0  to  23 * 6 + (6 - 1) = 138 + 5 = 143.
// - Pass move: digit = `kPassOffset + (die - 1)`. `die` is 1-6.
//   `kPassOffset` is chosen to be 144, so the range is 144 to 149.
// A full turn consists of up to two such half-moves. These two "digits" (d0, d1)
// are combined using a base, `kDigitBase`.
// The action is roughly `d1 * kDigitBase + d0`.
// An additional offset (`kDigitBase * kDigitBase`) can be added to indicate
// if the higher or lower die was used first, if necessary.

// constexpr int kDigitBase = 150;   // Base used to combine two half-move "digits". // MOVED TO .h
                                  // Must be >= 150 to accommodate the max digit value (149).
constexpr int kDigitBase = 150;   // Base used to combine two half-move "digits". // Moved from .h
constexpr int kPassOffset = 144;  // Offset for encoding pass half-moves.
                                  // Starts after the max regular move digit (143).

// --- Scheme 2: Encoding for Doubles Turns (with > 2 moves) ---
// When a player rolls doubles and can make more than two moves (up to four),
// a different encoding is used. This scheme encodes the *starting positions*
// of the checkers being moved.
// It uses base-25 (`kEncodingBaseDouble`) because there are 24 board points (0-23)
// plus a special value (24) to represent a pass or unused move slot.
// The four positions (p0, p1, p2, p3, with p0 being the least significant)
// are combined: `p3*B^3 + p2*B^2 + p1*B^1 + p0*B^0`, where B = `kEncodingBaseDouble`.
// An offset (`kDoublesOffset`) is added to distinguish these doubles actions
// from the non-doubles actions encoded using Scheme 1.

constexpr int kEncodingBaseDouble = 25;  // Base for encoding the *positions* in doubles moves (0-23 for points, 24 for pass).
constexpr int kDoublesOffset = 2 * kDigitBase * kDigitBase; // Offset added to doubles actions.
                                                           // Chosen to be larger than the maximum possible non-doubles action
                                                           // (which is approx. `1 * kDigitBase^2 + (kDigitBase-1)*kDigitBase + (kDigitBase-1)`).

} // End anonymous namespace

// ===== Utility Functions =====

// Moved to long_narde_utils.cc
// std::string PositionToString(int pos) { ... }

// Moved to long_narde_utils.cc
// std::string PositionToStringHumanReadable(int pos) { ... }

// Moved to long_narde_api.cc
// Player LongNardeState::CurrentPlayer() const { ... }

// Moved to long_narde_api.cc
// void LongNardeState::DoApplyAction(Action move_id) { ... }

// Moved to long_narde_api.cc
// void LongNardeState::UndoAction(Player player, Action action) { ... }

// Moved to long_narde_utils.cc
// std::string LongNardeState::ActionToString(Player player, Action move_id) const { ... }

// Moved to long_narde_api.cc
// std::string LongNardeState::ObservationString(Player player) const { ... }

// Moved to long_narde_api.cc
// void LongNardeState::ObservationTensor(Player player, absl::Span<float> values) const { ... }

// Moved to long_narde_api.cc
// bool LongNardeState::IsTerminal() const { ... }

// Moved to long_narde_api.cc
// std::vector<double> LongNardeState::Returns() const { ... }

// Moved to long_narde_api.cc
// std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const { ... }

// Moved to long_narde_api.cc
// std::unique_ptr<State> LongNardeState::Clone() const { ... }

// ===== Validation Functions =====

// Moved to long_narde_validation.cc
// bool LongNardeState::IsHeadPos(int player, int pos) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::IsFirstTurn(int player) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::IsLegalHeadMove(int player, int from_pos) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::HasIllegalBridge(int player) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::ValidateAction(Action action) const { ... }

// Moved to long_narde_validation.cc
// bool LongNardeState::IsOff(int player, int pos) const { ... }

// ===== Move Generation =====

// Moved to long_narde_legal_actions.cc
// std::set<std::vector<CheckerMove>> LongNardeState::GenerateMoveSequences(...) const { ... }

// Moved to long_narde_legal_actions.cc
// std::set<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player) const { ... }

// Moved to long_narde_legal_actions.cc
// int LongNardeState::RecLegalMoves(...) { ... }

// ===== State Manipulation Helpers =====

void LongNardeState::ProcessChanceRoll(Action move_id) {
  SPIEL_CHECK_GE(move_id, 0);
  SPIEL_CHECK_LT(move_id, game_->MaxChanceOutcomes());

  // Record the chance outcome in turn history.
  turn_history_info_.push_back(
      TurnHistoryInfo(kChancePlayerId, prev_player_, dice_,
                      move_id, double_turn_, is_first_turn_, moved_from_head_,
                      is_playing_extra_turn_));

  // Ensure we have no dice set yet, then apply this new roll.
  SPIEL_CHECK_TRUE(dice_.empty());
  RollDice(move_id); // Sets dice_ based on outcome
  initial_dice_ = dice_; // Store the roll in initial_dice_

  // Decide which player moves next.
  if (turns_ < 0) {
    // Initial state: White always starts in this implementation.
    turns_ = 0;
    cur_player_ = kXPlayerId; // White goes first
    prev_player_ = kChancePlayerId; // Previous was chance
    is_playing_extra_turn_ = false; // Not an extra turn
    is_first_turn_ = true; // It's the first turn for White
  } else if (double_turn_) {
    // Rolled doubles on the *previous* turn, granting an extra roll *now*.
    // Player remains the same.
    cur_player_ = prev_player_; // Player who rolled doubles continues
    is_playing_extra_turn_ = true;  // Mark that this roll is for an extra turn
    is_first_turn_ = false; // Cannot be the first turn if it's an extra turn
  } else {
    // Normal turn progression: pass to the opponent.
    cur_player_ = Opponent(prev_player_);
    is_playing_extra_turn_ = false;  // Reset for normal turn
    // Determine if it's the new player's first turn
    is_first_turn_ = IsFirstTurn(cur_player_); 
  }
  
  // Reset double_turn_ flag; it indicated the *previous* roll was doubles.
  // The current roll's nature (double or not) will determine the *next* state transition.
  double_turn_ = false; 
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

// Moved to long_narde_moves.cc
// void LongNardeState::ApplyCheckerMove(int player, const CheckerMove& move) { ... }

// Moved to long_narde_moves.cc
// void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) { ... }


// ===== Utility Functions (Continued) =====

int LongNardeState::board(int player, int pos) const {
  // Bounds check for safety, returning 0 for invalid positions
  if (pos < 0 || pos >= kNumPoints) {
    return 0;
  }
  return board_[player][pos];
}

int LongNardeState::Opponent(int player) const { return 1 - player; }

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

inline int CounterClockwisePos(int from, int pips, int num_points) {
  // DEPRECATED? Seems unused. GetToPos is used instead.
  int pos = from - pips;
  pos = (pos % num_points + num_points) % num_points; // Ensure positive result for modulo
  return pos;
}

// Moved to long_narde_moves.cc


int LongNardeState::GetVirtualCoords(int player, int real_pos) const {
   // DEPRECATED? Path Index seems more useful. Keep for now if needed elsewhere.
  if (real_pos < 0 || real_pos >= kNumPoints) {
    SpielFatalError(absl::StrCat("GetVirtualCoords called with invalid real_pos: ", real_pos));
    return -1;
  }

  if (player == kXPlayerId) {
    return real_pos; // White path matches indices 23->0
  } else { // kOPlayerId path 11->0->23->12 mapped to 0->23
    if (real_pos >= 0 && real_pos <= 11) { // Segment 1 (11..0) maps to 0..11
      return 11 - real_pos;
    } else { // Segment 2 (23..12) maps to 12..23
      return 12 + (23 - real_pos);
    }
  }
}

int LongNardeState::GetPathIndex(int player, int real_pos) const {
    // Returns the 0-based index along the player's path.
    // 0 = start (head), 23 = point just before bearoff.
    if (real_pos < 0 || real_pos >= kNumPoints) {
         SpielFatalError(absl::StrCat("GetPathIndex called with invalid real_pos: ", real_pos));
         return -1; // Should be unreachable
    }

    if (player == kXPlayerId) { // Path: 23 -> 0
        // Index 0 = pos 23, Index 1 = pos 22, ..., Index 23 = pos 0
        return 23 - real_pos;
    } else { // Path: 11 -> 0 -> 23 -> 12
        if (real_pos >= 0 && real_pos <= 11) { // First half: 11 down to 0
            // Index 0 = pos 11, Index 1 = pos 10, ..., Index 11 = pos 0
            return 11 - real_pos;
        } else { // Second half: 23 down to 12
            // Index 12 = pos 23, Index 13 = pos 22, ..., Index 23 = pos 12
            return 12 + (23 - real_pos);
        }
    }
}


bool LongNardeState::IsAhead(int player, int checker_pos_idx, int reference_pos_idx) const {
    // Checks if checker_pos_idx is *further along* the path than reference_pos_idx for 'player'.
    // Further along means closer to home / bear-off, which corresponds to a *higher* path index.
    SPIEL_CHECK_GE(checker_pos_idx, 0);
    SPIEL_CHECK_LT(checker_pos_idx, kNumPoints);
    SPIEL_CHECK_GE(reference_pos_idx, 0);
    SPIEL_CHECK_LT(reference_pos_idx, kNumPoints);

    return GetPathIndex(player, checker_pos_idx) > GetPathIndex(player, reference_pos_idx);
}

int LongNardeState::GetBlockPathStartRealPos(int player_for_path, int block_lowest_real_idx) const {
    // Given the lowest real index (0-23) of a 6-block, finds which of the 6 points
    // forming the block is *furthest back* along the path of 'player_for_path'.
    // Furthest back corresponds to the *lowest* path index.
    if (block_lowest_real_idx < 0 || block_lowest_real_idx >= kNumPoints) {
         SpielFatalError(absl::StrCat("GetBlockPathStartRealPos called with invalid block_lowest_real_idx: ", block_lowest_real_idx));
         return -1; // Should be unreachable
    }

    int furthest_back_real_pos = block_lowest_real_idx; // Start assumption
    int min_path_idx = GetPathIndex(player_for_path, block_lowest_real_idx);

    // Check the other 5 points in the block
    for (int i = 1; i < 6; ++i) {
        int current_real_pos = (block_lowest_real_idx + i) % kNumPoints; // Handle wrap-around
        
        int current_path_idx = GetPathIndex(player_for_path, current_real_pos);
        if (current_path_idx < min_path_idx) {
            min_path_idx = current_path_idx;
            furthest_back_real_pos = current_real_pos;
        }
    }
    return furthest_back_real_pos;
}

// Moved to long_narde_utils.cc
// std::string LongNardeState::ToString() const { ... }

// Moved to long_narde_utils.cc
// ScoringType ParseScoringType(const std::string& st_str) { ... }

// ===== Game Class Methods =====

// Moved to long_narde_game.cc
// LongNardeGame::LongNardeGame(const GameParameters& params) { ... }

// Moved to long_narde_game.cc
// double LongNardeGame::MaxUtility() const { ... }

// Moved to long_narde_legal_actions.cc
// LegalActions(...) etc. - We removed these higher up already.

// ===== Helper Functions =====

// Moved to long_narde_moves.cc
// int LongNardeState::GetToPos(int player, int from_pos, int die_value) const { ... } // Remove duplicate 'Moved to' comment if present

// --- Correctly placed functions ---
// Player LongNardeState::Opponent(int player) const { ... } // Remove duplicate if present

bool LongNardeState::IsPosInHome(int player, int pos) const {
  switch (player) {
    case kXPlayerId:
      // White's home: points 1-6 (indices 0-5)
      return (pos >= kWhiteHomeStart && pos <= kWhiteHomeEnd);
    case kOPlayerId:
      // Black's home: points 13-18 (indices 12-17)
      return (pos >= kBlackHomeStart && pos <= kBlackHomeEnd);
    default:
      SpielFatalError(absl::StrCat("Unknown player ID in IsPosInHome: ", player));
      return false; // Should be unreachable
  }
}

bool LongNardeState::AllInHome(Player player) const {
  int checkers_on_board = 0;
  if (player == kXPlayerId) {
    // White's home is points 1-6 (indices 0-5)
    // Check if any checkers are *outside* this range (points 7-24, indices 6-23)
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
      // Note: We don't need to sum checkers_on_board here as the return false above handles it.
    }
     // Also count checkers *inside* home
     for (int i = kWhiteHomeStart; i <= kWhiteHomeEnd; ++i) {
        checkers_on_board += board(player, i);
     }

  } else { // kOPlayerId
    // Black's home is points 13-18 (indices 12-17)
    // Check if any checkers are *outside* this range
    // Check indices 0-11 (points 1-12)
    for (int i = 0; i < kBlackHomeStart; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
       // Note: We don't need to sum checkers_on_board here.
    }
    // Check indices 18-23 (points 19-24)
    for (int i = kBlackHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
        // Note: We don't need to sum checkers_on_board here.
    }
    // Also count checkers *inside* home
    for (int i = kBlackHomeStart; i <= kBlackHomeEnd; ++i) {
        checkers_on_board += board(player, i);
    }
  }
  
  // Final check: Ensure the total count of checkers IN HOME + checkers BORNE OFF equals total checkers.
  return (checkers_on_board + scores_[player] == kNumCheckersPerPlayer);
}

// --- End of added functions ---

}  // namespace long_narde
}  // namespace open_spiel