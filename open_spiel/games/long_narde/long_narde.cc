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
const GameType kGameType{
    /*short_name=*/"long_narde",
    /*long_name=*/"Long Narde",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/2,
    /*max_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
        {"scoring_type", GameParameter(std::string(kDefaultScoringType))}
    }};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LongNardeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Chance outcome definitions (implementation detail)
const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    {0, 1.0 / 18}, {1, 1.0 / 18}, {2, 1.0 / 18}, {3, 1.0 / 18},
    {4, 1.0 / 18}, {5, 1.0 / 18}, {6, 1.0 / 18}, {7, 1.0 / 18},
    {8, 1.0 / 18}, {9, 1.0 / 18}, {10, 1.0 / 18}, {11, 1.0 / 18},
    {12, 1.0 / 18}, {13, 1.0 / 18}, {14, 1.0 / 18}, {15, 1.0 / 36},
    {16, 1.0 / 36}, {17, 1.0 / 36}, {18, 1.0 / 36}, {19, 1.0 / 36},
    {20, 1.0 / 36},
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

// Other Implementation Constants
// Note: kBearOffPos and kNumNonDoubleOutcomes moved to long_narde.h as they relate more to game rules/structure.
constexpr int kNumOffPosHumanReadable = -2; // Value used in string formatting for borne-off checkers.
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

// In Long Narde, pass moves are represented in two ways:
// 1. Internally as kPassPos (-1) in the game logic.
// 2. As position 24 when encoding/decoding actions (since valid board positions are 0-23).
std::string PositionToString(int pos) {
  if (pos == kPassPos) return "Pass";
  SPIEL_CHECK_GE(pos, 0);
  SPIEL_CHECK_LT(pos, kNumPoints);
  return absl::StrCat(pos + 1);
}

std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case kXPlayerId: return "x";
    case kOPlayerId: return "o";
    case kChancePlayerId: return "*";
    case kTerminalPlayerId: return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}

std::string PositionToStringHumanReadable(int pos) {
  if (pos == kNumOffPosHumanReadable) {
    return "Off";
  } else if (pos == kPassPos) {
    return "Pass";
  } else {
    // Convert human-readable point (1-24) to internal index (0-23)
    SPIEL_CHECK_GE(pos, 1);
    SPIEL_CHECK_LE(pos, kNumPoints);
    int internal_pos = pos - 1;
    return PositionToString(internal_pos);
  }
}

// ===== Encoding/Decoding Functions =====

Action LongNardeState::CheckerMovesToSpielMove(
    const std::vector<CheckerMove>& moves) const {
  SPIEL_CHECK_LE(moves.size(), 4);  // Allow up to 4 moves for doubles

  // Check if this is a doubles roll based on the current dice state.
  bool is_doubles = false;
  if (dice_.size() == 2 && DiceValue(0) == DiceValue(1)) {
    is_doubles = true;
  }

  // Use a separate, higher-range encoding for doubles when more than 2 moves are made (up to 4).
  // This is necessary because the standard encoding only supports two half-moves.
  if (is_doubles && moves.size() > 2) {
    // Encode up to 4 checker source positions using a base-25 system.
    // 0-23 represent board points, 24 represents a pass (kPassPos).
    // The die value is implicit (it's the doubles value).
    std::vector<int> positions(4, kEncodingBaseDouble - 1);  // Default to pass (encoded as 24)
    
    // Fill the positions array with actual positions from the provided moves.
    for (size_t i = 0; i < moves.size() && i < 4; ++i) {
      if (moves[i].pos == kPassPos) {
        positions[i] = kEncodingBaseDouble - 1;  // kPassPos encoded as 24
      } else {
        SPIEL_CHECK_GE(moves[i].pos, 0);
        SPIEL_CHECK_LT(moves[i].pos, kNumPoints);
        positions[i] = moves[i].pos; // Store the 'from' position
      }
    }
    
    // Encode the 4 positions into a single integer using base-25.
    // positions[0] is the least significant digit, positions[3] is the most significant.
    Action action_double = 0;
    for (int i = 3; i >= 0; --i) {
      action_double = action_double * kEncodingBaseDouble + positions[i];
    }
    
    // Add kDoublesOffset to distinguish this encoding from the non-doubles scheme.
    // The final action value will be >= kDoublesOffset.
    Action action = kDoublesOffset + action_double;
    
    SPIEL_CHECK_GE(action, kDoublesOffset); // Ensure it's in the doubles range
    SPIEL_CHECK_LT(action, NumDistinctActions()); // Ensure it's within the calculated total range
    return action;
  } else {
    // Standard encoding for non-doubles rolls or doubles rolls with 0, 1, or 2 moves.
    // This scheme encodes two "half-moves" (CheckerMove) into a single action.
    // The sequence 'moves' is guaranteed by LegalActions to be valid in this order.
    // We encode moves[0] as dig0 and moves[1] as dig1 directly.
    std::vector<CheckerMove> encoded_moves = moves; // Use a copy to add padding if needed

    // Ensure we always encode exactly two half-moves by adding Pass moves if necessary.
    while (encoded_moves.size() < 2) {
      // Add pass moves as padding.
      int die_val = 1;  // Default die value for padding.
      // Try to find an *unused* die value if possible for the pass padding.
      // This helps preserve information if decoding is done without full state context,
      // although full state context is generally assumed.
      int available_die = -1;
      if (dice_.size() >= 1 && UsableDiceOutcome(dice_[0])) available_die = DiceValue(0);
      else if (dice_.size() >= 2 && UsableDiceOutcome(dice_[1])) available_die = DiceValue(1);

      if (available_die != -1) {
          die_val = available_die;
      } else if (!encoded_moves.empty() && encoded_moves[0].die > 0) {
          // Fallback: use the die from the first move if no dice info available (should not happen in normal flow).
          die_val = encoded_moves[0].die;
      }
      // Ensure die_val is valid (1-6).
      die_val = std::max(1, std::min(6, die_val));
      // Add a pass move with the chosen die value.
      encoded_moves.push_back(CheckerMove(kPassPos, kPassPos, die_val));
    }

    // Helper function to encode a single half-move (CheckerMove) into an integer digit.
    // This digit represents either a normal move or a pass move.
    auto encode_move = [](const CheckerMove& move) -> int {
      if (move.pos == kPassPos) {
        // Encode a pass move. Uses a specific offset (kPassOffset).
        // The value is kPassOffset + (die - 1), ranging from 144 to 149.
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return kPassOffset + (move.die - 1); // 144 + 0..5
      } else {
        // Encode a normal move from a board position.
        // The value is pos * 6 + (die - 1).
        // pos is 0-23, die is 1-6.
        // Max value is 23 * 6 + 5 = 138 + 5 = 143.
        // This ensures no overlap with the pass encoding range (144-149).
        SPIEL_CHECK_GE(move.pos, 0);
        SPIEL_CHECK_LT(move.pos, kNumPoints);
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return move.pos * 6 + (move.die - 1); // 0..143
      }
    };

    // Encode the first two (potentially padded) moves.
    int dig0 = encode_move(encoded_moves[0]); // First half-move
    int dig1 = encode_move(encoded_moves[1]); // Second half-move

    // Combine the two digits into a single action using base kDigitBase (150).
    // dig0 is the least significant digit, dig1 is the most significant.
    // Max value is 149 * 150 + 149 = 22350 + 149 = 22499.
    Action action = dig1 * kDigitBase + dig0;

    // Determine if the *actual* dice roll (if available) had the lower die first.
    // This is needed because LegalActions might reorder moves (e.g., highest die first).
    // We need to distinguish action (5, 3) from roll (5, 3) vs action (5, 3) from roll (3, 5).
    bool actual_low_roll_first = false;
    if (dice_.size() >= 2) {
        // Use DiceValue to handle potential internal encoding (7-12) if dice were marked used.
        int die0_val = DiceValue(0);
        int die1_val = DiceValue(1);
        if (die0_val < die1_val) {
            actual_low_roll_first = true;
        }
    }
    // If dice_.size() < 2 (e.g., chance node), actual_low_roll_first remains false.

    // Add a large offset (kDigitBase * kDigitBase = 150 * 150 = 22500)
    // if the actual dice roll had the lower die rolled first.
    // This distinguishes the two cases mentioned above.
    // Example: Roll (3, 5). Move using 5 then 3. Encoded as (dig1=move5, dig0=move3).
    //          Action = encode(move3) + encode(move5) * 150 + 22500.
    // Example: Roll (5, 3). Move using 5 then 3. Encoded as (dig1=move5, dig0=move3).
    //          Action = encode(move3) + encode(move5) * 150.
    // This offset is NOT added for double pass moves, as the dice order is irrelevant.
    bool is_double_pass = (encoded_moves.size() == 2 && encoded_moves[0].pos == kPassPos && encoded_moves[1].pos == kPassPos);
    if (actual_low_roll_first && !is_double_pass) {
      action += kDigitBase * kDigitBase; // Add offset ~22500
    }

    // Final sanity checks for the non-doubles encoding range.
    SPIEL_CHECK_GE(action, 0);
    // Ensure the action is below the start of the doubles encoding range.
    SPIEL_CHECK_LT(action, kDoublesOffset); // kDoublesOffset is typically 2 * kDigitBase * kDigitBase = 45000
    return action;
  }
}

std::vector<CheckerMove> LongNardeState::SpielMoveToCheckerMoves(
    Player player, Action spiel_move) const {
  // Check if the action falls within the special doubles encoding range.
  if (spiel_move >= kDoublesOffset) {
    // Decode a doubles action (up to 4 moves).
    Action action_double = spiel_move - kDoublesOffset; // Remove the offset
    
    // Extract the 4 encoded positions using base-25 decoding.
    std::vector<int> positions(4);
    for (int i = 0; i < 4; ++i) {
      // positions[i] will be 0-23 for a point, or 24 for a pass.
      positions[i] = action_double % kEncodingBaseDouble;
      action_double /= kEncodingBaseDouble;
    }

    // Determine the die value used for all moves in a doubles turn.
    int die_val = 1;  // Default if dice info isn't available (should not happen).
    if (dice_.size() > 0) {
      die_val = DiceValue(0);  // Use the value of the first die (they are the same).
    }

    // Reconstruct the CheckerMove objects from the decoded positions.
    std::vector<CheckerMove> cmoves;
    for (int i = 0; i < 4; ++i) {
      int pos;
      if (positions[i] == kEncodingBaseDouble - 1) {
        // Encoded value 24 corresponds to kPassPos.
        pos = kPassPos;
      } else {
        // Encoded value 0-23 corresponds to board points.
        pos = positions[i];
      }
      
      if (pos == kPassPos) {
        // Reconstruct a pass move. Note: to_pos is irrelevant for pass.
        cmoves.push_back(CheckerMove(kPassPos, kPassPos, die_val));
      } else {
        // Reconstruct a normal move. Calculate the destination position.
        int to_pos = GetToPos(player, pos, die_val);
        cmoves.push_back(CheckerMove(pos, to_pos, die_val));
      }
    }
    // Note: The returned vector might contain more than the actual number of
    // moves played if fewer than 4 moves were made (padded with passes during encoding).
    // The caller (e.g., DoApplyAction) needs to handle this, typically by stopping
    // after the actual number of moves needed or encountering the first pass.
    return cmoves;
  } else {
    // Decode a standard (non-doubles or doubles <= 2 moves) action.
    // Check if the low-roll-first offset was applied during encoding.
    bool high_roll_first = spiel_move < (kDigitBase * kDigitBase);
    if (!high_roll_first) {
      // Remove the offset if it was present.
      spiel_move -= kDigitBase * kDigitBase;
    }
    
    // Extract the two digits using base kDigitBase (150).
    int dig0 = spiel_move % kDigitBase; // First half-move (least significant)
    int dig1 = spiel_move / kDigitBase; // Second half-move (most significant)

    // Helper function to decode a single digit back into a CheckerMove.
    auto decode_digit = [this, player](int digit) -> CheckerMove {
      if (digit >= kPassOffset) { // Check if it's in the pass range (144-149)
        // Decode a pass move.
        int die = (digit - kPassOffset) + 1; // Extract die value (1-6)
        // Return a pass move. to_pos is irrelevant.
        return CheckerMove(kPassPos, kPassPos, die);
      } else { // Must be in the normal move range (0-143)
        // Decode a normal move.
        int pos = digit / 6;        // Extract source position (0-23)
        int die = (digit % 6) + 1;  // Extract die value (1-6)
        // Calculate the destination position.
        int to_pos = GetToPos(player, pos, die);
        // Return the reconstructed normal move.
        return CheckerMove(pos, to_pos, die);
      }
    };

    // Decode the two digits back into CheckerMoves.
    std::vector<CheckerMove> cmoves;
    cmoves.push_back(decode_digit(dig0)); // Decode first move
    cmoves.push_back(decode_digit(dig1)); // Decode second move
    
    // The order in cmoves matches the order they were encoded (e.g., highest die first if applied).
    // The 'high_roll_first' flag (derived from the offset) indicates the original dice roll order,
    // but the returned moves are in the potentially reordered sequence used for the action.
    return cmoves;
  }
}

int LongNardeState::NumDistinctActions() const {
  // The total number of distinct actions is the sum of the ranges used by the two encoding schemes.
  // 1. Non-doubles (and doubles <= 2 moves) encoding range:
  //    - Actions are encoded using two digits in base kDigitBase (150).
  //    - An offset of kDigitBase*kDigitBase (22500) is added if the original roll was low-die first.
  //    - This results in two potential ranges: [0, 22499] and [22500, 44999].
  //    - The start of the doubles encoding (kDoublesOffset) is set to the end of this range (45000).
  //    - So, this scheme uses the range [0, kDoublesOffset - 1].
  // 2. Doubles (> 2 moves) encoding range:
  //    - Actions are encoded using base kEncodingBaseDouble (25) for 4 positions.
  //    - An offset of kDoublesOffset (45000) is added.
  //    - The size of this range is kEncodingBaseDouble^4 = 25^4 = 390625.
  //    - This scheme uses the range [kDoublesOffset, kDoublesOffset + kEncodingBaseDouble^4 - 1].

  // Calculate the size of the doubles encoding range (kEncodingBaseDouble^4)
  int double_range_size = 1;
  for (int i = 0; i < 4; ++i) {
    double_range_size *= kEncodingBaseDouble;  // 25^4 = 390625
  }
  
  // The total number of distinct actions is the start of the doubles range plus the size of the doubles range.
  // This gives the upper bound for the highest possible action value.
  return kDoublesOffset + double_range_size; // 45000 + 390625 = 435625
}

// ===== Constructor and State Setup =====

LongNardeState::LongNardeState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      x_turns_(0),
      o_turns_(0),
      double_turn_(false),
      is_first_turn_(true),
      moved_from_head_(false),
      is_playing_extra_turn_(false),
      dice_({}),
      initial_dice_({}), // Initialize initial_dice_
      scores_({0, 0}),
      board_({std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
      turn_history_info_({}),
      allow_last_roll_tie_(false),
      scoring_type_(ParseScoringType(
          game->GetParameters().count("scoring_type") > 0 ?
          game->GetParameters().at("scoring_type").string_value() :
          kDefaultScoringType)) {
  SetupInitialBoard();
}

void LongNardeState::SetState(int cur_player, bool double_turn,
                              const std::vector<int>& dice,
                              const std::vector<int>& scores,
                              const std::vector<std::vector<int>>& board) {
  cur_player_ = cur_player;
  prev_player_ = cur_player;
  double_turn_ = double_turn;
  is_playing_extra_turn_ = false;
  dice_ = dice;
  initial_dice_ = dice; // Initialize initial_dice_ in SetState
  scores_ = scores;
  board_ = board;
  
  if (cur_player_ == kChancePlayerId) {
    turns_ = -1;
  } else {
    turns_ = 0;
  }
  
  if (cur_player != kChancePlayerId && cur_player != kTerminalPlayerId) {
    is_first_turn_ = IsFirstTurn(cur_player);
  }
  moved_from_head_ = false;
}

void LongNardeState::SetupInitialBoard() {
  board_[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  board_[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
}

// ===== Core Spiel API Functions =====

Player LongNardeState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

void LongNardeState::DoApplyAction(Action move_id) {
  if (IsChanceNode()) {
    ProcessChanceRoll(move_id);
    return;
  }

  bool rolled_doubles = (dice_.size() == 2 && DiceValue(0) == DiceValue(1));
  bool currently_extra = is_playing_extra_turn_; // Store current state

  is_first_turn_ = IsFirstTurn(cur_player_);
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
      if (is_first_turn_) {
        // Must be double 6,4,3
        bool is_special_double = (rolled_doubles &&
                                 (DiceValue(0) == 6 ||
                                  DiceValue(0) == 4 ||
                                  DiceValue(0) == 3));
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

  // Record history with the current state before modifications
  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, move_id, double_turn_,
                      is_first_turn_, moved_from_head_, currently_extra));

  // Only grant an extra turn if we rolled doubles and are NOT already in an extra turn
  bool grant_extra_turn = rolled_doubles && !currently_extra;

  // Update turn progression
  if (!grant_extra_turn) {
    turns_++;
    if (cur_player_ == kXPlayerId) {
      x_turns_++;
    } else if (cur_player_ == kOPlayerId) {
      o_turns_++;
    }
  }

  // Update state for next turn
  prev_player_ = cur_player_;
  dice_.clear();
  cur_player_ = IsTerminal() ? kTerminalPlayerId : kChancePlayerId;
  double_turn_ = grant_extra_turn;  // Signal for next ProcessChanceRoll
  is_playing_extra_turn_ = false;  // Reset after move completes
  is_first_turn_ = false;
  moved_from_head_ = false;
}

void LongNardeState::UndoAction(Player player, Action action) {
  TurnHistoryInfo info = turn_history_info_.back();
  turn_history_info_.pop_back();
  is_first_turn_ = info.is_first_turn;
  moved_from_head_ = info.moved_from_head;
  cur_player_ = info.player;
  prev_player_ = info.prev_player;
  dice_ = info.dice;
  double_turn_ = info.double_turn;
  is_playing_extra_turn_ = info.is_playing_extra_turn;  // Restore extra turn state

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
    
    if (!double_turn_) {
      turns_--;
      if (player == kXPlayerId) {
        x_turns_--;
      } else if (player == kOPlayerId) {
        o_turns_--;
      }
    }
  }
}

std::vector<Action> LongNardeState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();

  std::unique_ptr<State> cstate = this->Clone();
  LongNardeState* state = dynamic_cast<LongNardeState*>(cstate.get());
  std::set<std::vector<CheckerMove>> movelist;

  // Determine max moves based on dice
  int max_moves = 0;
  if (!dice_.empty()) {
      bool is_doubles = (dice_.size() >= 2 && DiceValue(0) == DiceValue(1));
      max_moves = is_doubles ? 4 : 2; // Allow up to 4 for doubles, 2 otherwise
  }

  // Generate all possible move sequences using the cloned state, limiting depth
  state->RecLegalMoves({}, &movelist, max_moves);

  // DEBUG: Print movelist contents after generation
  if (kDebugging) {
    std::cout << "DEBUG LegalActions (Player " << state->CurrentPlayer() << "): Movelist after RecLegalMoves (" << movelist.size() << " entries):" << std::endl;
    for (const auto& seq : movelist) {
        std::cout << "  Seq: ";
        for(const auto& m : seq) { std::cout << "{" << m.pos << "," << m.to_pos << "," << m.die << "} "; }
        std::cout << std::endl;
    }
  }

  // Find the maximum sequence length achieved
  int longest_sequence = 0;
  for (const auto& moveseq : movelist) {
      longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
  }

  // Find the maximum number of non-pass moves within sequences of the longest length
  int max_non_pass = 0;
  for (const auto& moveseq : movelist) {
      if (moveseq.size() == longest_sequence) {
          int current_non_pass = 0;
          for (const auto& move : moveseq) {
              if (move.pos != kPassPos) {
                  current_non_pass++;
              }
          }
          max_non_pass = std::max(max_non_pass, current_non_pass);
      }
  }

  // DEBUG: Print filtering criteria
  if (kDebugging) {
    std::cout << "DEBUG LegalActions: longest_sequence = " << longest_sequence 
              << ", max_non_pass = " << max_non_pass << std::endl;
  }

  // If max_non_pass is 0, it means only pass moves are possible
  if (max_non_pass == 0) {
      std::vector<CheckerMove> pass_move_seq;
      int p_die1 = (dice_.size() >= 1 && UsableDiceOutcome(dice_[0])) ? DiceValue(0) : 1;
      int p_die2 = (dice_.size() >= 2 && UsableDiceOutcome(dice_[1])) ? DiceValue(1) : p_die1;
      p_die1 = std::max(1, std::min(6, p_die1));
      p_die2 = std::max(1, std::min(6, p_die2));
      pass_move_seq.push_back({kPassPos, kPassPos, p_die1});
      pass_move_seq.push_back({kPassPos, kPassPos, p_die2});
      return {CheckerMovesToSpielMove(pass_move_seq)};
  }

  // Filter the movelist to keep only sequences with max length AND max non-pass moves
  const size_t kMaxActionsToGenerate = 20;
  std::set<Action> unique_actions;

  for (const auto& moveseq : movelist) {
      if (moveseq.size() == longest_sequence) {
          int current_non_pass = 0;
          for (const auto& move : moveseq) {
              if (move.pos != kPassPos) {
                  current_non_pass++;
              }
          }

          if (current_non_pass == max_non_pass) {
              if (unique_actions.size() >= kMaxActionsToGenerate) break;

              Action action = CheckerMovesToSpielMove(moveseq);
              unique_actions.insert(action);
          }
      }
  }

  std::vector<Action> legal_moves;
  legal_moves.assign(unique_actions.begin(), unique_actions.end());

  // Apply the "play higher die" rule if exactly one die was playable (non-doubles)
  bool is_doubles = (dice_.size() == 2 && DiceValue(0) == DiceValue(1));
  if (max_non_pass == 1 && !is_doubles && dice_.size() == 2) {
      int d1 = DiceValue(0);
      int d2 = DiceValue(1);
      int higher_die = std::max(d1, d2);
      int lower_die = std::min(d1, d2);

      // Check if each die was individually playable *anywhere* on the board
      std::unique_ptr<State> temp_state = this->Clone();
      LongNardeState* cloned_state = dynamic_cast<LongNardeState*>(temp_state.get());
      // Ensure the cloned state has the original dice values (unmarked)
      std::vector<int> original_dice;
      if (dice_.size() >= 1) original_dice.push_back(DiceValue(0));
      if (dice_.size() >= 2) original_dice.push_back(DiceValue(1));
      cloned_state->dice_ = original_dice;
      // Reset moved_from_head status on the clone
      cloned_state->moved_from_head_ = false;

      std::set<CheckerMove> all_half_moves = cloned_state->GenerateAllHalfMoves(cur_player_);
      bool higher_die_ever_playable = false;
      bool lower_die_ever_playable = false;
      for(const auto& hm : all_half_moves) {
          if(hm.pos != kPassPos) {
              if (hm.die == higher_die) higher_die_ever_playable = true;
              if (hm.die == lower_die) lower_die_ever_playable = true;
          }
          if (higher_die_ever_playable && lower_die_ever_playable) break;
      }

      // Identify the die used in the single-move sequences found
      std::vector<Action> actions_using_higher;
      std::vector<Action> actions_using_lower;

      for (Action action : legal_moves) {
          std::vector<CheckerMove> decoded_moves = SpielMoveToCheckerMoves(cur_player_, action);
          // Find the single non-pass move represented by this action
          CheckerMove single_played_move = kPassMove;
          int actual_non_pass_count = 0;
          for(const auto& m : decoded_moves) {
              if (m.pos != kPassPos) {
                  if (actual_non_pass_count == 0) {
                     single_played_move = m;
                  }
                  actual_non_pass_count++;
              }
          }

          // This action should represent exactly one non-pass move
          if (actual_non_pass_count == 1) {
              if (single_played_move.die == higher_die) {
                  actions_using_higher.push_back(action);
              } else if (single_played_move.die == lower_die) {
                  actions_using_lower.push_back(action);
              }
          } else if (actual_non_pass_count > 1) {
               SpielFatalError(absl::StrCat("Higher-die rule check: Action ", action, " decoded to ", actual_non_pass_count, " moves, expected 1."));
          }
      }

      // Apply the rule based on which dice were ever playable:
      if (higher_die_ever_playable && lower_die_ever_playable) {
          // Both were playable, must use higher die
          return actions_using_higher;
      } else if (higher_die_ever_playable) {
          // Only higher was playable
          return actions_using_higher;
      } else if (lower_die_ever_playable) {
          // Only lower was playable
          return actions_using_lower;
      } else {
          // This should not happen if max_non_pass == 1
          SpielFatalError("Inconsistent state in LegalActions higher die rule: Neither die playable but max_non_pass=1.");
          return legal_moves; // Should be unreachable
      }
  }

  return legal_moves;
}

std::vector<Action> LongNardeState::IllegalActions() const {
  std::vector<Action> illegal_actions;
  if (IsChanceNode() || IsTerminal()) return illegal_actions;
  
  // Check dice validity before proceeding
  if (dice_.size() < 2) return illegal_actions; // Cannot determine rolls
  
  int high_roll = DiceValue(0);
  int low_roll = DiceValue(1);
  if (high_roll < low_roll) std::swap(high_roll, low_roll);
  int kMaxActionId = NumDistinctActions();
  
  std::vector<Action> legal_actions = LegalActions(); // Get legal actions once
  std::set<Action> legal_set(legal_actions.begin(), legal_actions.end());
  
  for (Action action = 0; action < kMaxActionId; ++action) {
    if (legal_set.count(action)) continue; // Skip known legal actions
    
    // Simple heuristic check: Check if it decodes reasonably.
    // Full validation is complex and already done by LegalActions.
    // This mainly catches encoding ranges that don't make sense.
    try {
  std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
        // Basic check: pass moves should correspond to valid die values if possible
        bool is_pass = true;
        for(const auto& m : moves) {
           if (m.pos != kPassPos) {
               is_pass = false;
               break;
           }
        }
        if (is_pass) {
             // Check if die values encoded in pass make sense with current roll
             if (moves.size() >= 1 && moves[0].die != DiceValue(0) && moves[0].die != DiceValue(1)) {
                  illegal_actions.push_back(action);
                  continue;
             }
              if (moves.size() >= 2 && moves[1].die != DiceValue(0) && moves[1].die != DiceValue(1)) {
                  illegal_actions.push_back(action);
                  continue;
             }
        }
        // Further checks could be added, but might duplicate LegalActions logic.
        // The main goal is to identify actions outside the *possible* range or clearly invalid encodings.
        
    } catch (...) {
        // If decoding itself fails, it's illegal
        illegal_actions.push_back(action);
        continue;
    }
     // If it wasn't caught by simple checks and isn't legal, add it.
     if (!legal_set.count(action)) {
         illegal_actions.push_back(action);
     }

  }
  return illegal_actions;
}

std::string LongNardeState::ActionToString(Player player,
                                            Action move_id) const {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move_id, 0);
    SPIEL_CHECK_LT(move_id, kChanceOutcomes.size());
    if (turns_ >= 0) {
      int d1 = (dice_.size() >=1) ? DiceValue(0) : kChanceOutcomeValues[move_id][0];
      int d2 = (dice_.size() >=2) ? DiceValue(1) : kChanceOutcomeValues[move_id][1];
      if (dice_.empty()) {
          d1 = kChanceOutcomeValues[move_id][0];
          d2 = kChanceOutcomeValues[move_id][1];
      }
      return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", d1, d2, ")");
        } else {
      // Starting roll - This logic seems specific to older backgammon setup, might need review for Long Narde
      const char* starter = (move_id < kNumNonDoubleOutcomes ? "X starts" : "O starts"); // kNumNonDoubleOutcomes needs to be correct
      Action outcome_id = move_id;
      if (outcome_id >= kNumNonDoubleOutcomes) { // Ensure this constant is valid
        outcome_id -= kNumNonDoubleOutcomes;
      }
      SPIEL_CHECK_LT(outcome_id, kChanceOutcomeValues.size());
      return absl::StrCat("chance outcome ", move_id, " ", starter, ", ",
                          "(roll: ", kChanceOutcomeValues[outcome_id][0],
                          kChanceOutcomeValues[outcome_id][1], ")");
    }
  }

  std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(player, move_id);

  std::string returnVal = absl::StrCat(move_id, " -");
  bool any_move = false;
  for (const auto& move : cmoves) {
    if (move.pos == kPassPos) {
      bool all_pass = true;
      for(const auto& m : cmoves) {
          if (m.pos != kPassPos) {
              all_pass = false;
              break;
          }
      }
      if (all_pass) {
          // Only return "Pass" if *all* decoded moves are passes.
          return absl::StrCat(move_id, " - Pass");
      }
      // If some moves are passes and some aren't, don't print the pass part explicitly.
      continue;
    }

    any_move = true;
    int start_hr, end_hr;

    // Human-readable conversion: White 1-24, Black 1-24 (opposite direction)
    if (player == kOPlayerId) { // Black's perspective
       start_hr = move.pos <= 11 ? (12 - move.pos) : (36 - move.pos) ; // 11->1, 0->12, 23->13, 12->24
       if (IsOff(player, move.to_pos)) {
           end_hr = kNumOffPosHumanReadable; // "Off"
       } else {
            SPIEL_CHECK_GE(move.to_pos, 0);
            SPIEL_CHECK_LT(move.to_pos, kNumPoints);
            end_hr = move.to_pos <= 11 ? (12 - move.to_pos) : (36 - move.to_pos);
       }
    } else { // White's perspective (kXPlayerId)
       start_hr = 24 - move.pos; // 23->1, 0->24
        if (IsOff(player, move.to_pos)) {
            end_hr = kNumOffPosHumanReadable; // "Off"
        } else {
            SPIEL_CHECK_GE(move.to_pos, 0);
            SPIEL_CHECK_LT(move.to_pos, kNumPoints);
            end_hr = 24 - move.to_pos;
        }
    }

    absl::StrAppend(&returnVal, " ", PositionToStringHumanReadable(start_hr), "/",
                    PositionToStringHumanReadable(end_hr));
  }

   if (!any_move) {
       // If loop finishes and any_move is false, it must have been all passes (handled above)
       // or an empty move sequence (which shouldn't happen for valid actions).
       // We return Pass as a fallback.
       return absl::StrCat(move_id, " - Pass");
   }

  return returnVal;
}

std::string LongNardeState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

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

std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  // In Long Narde, the chance outcomes (dice rolls) are always the same,
  // regardless of whether it's the starting roll or a regular turn roll.
  return kChanceOutcomes;
}

std::unique_ptr<State> LongNardeState::Clone() const {
  auto new_state = std::make_unique<LongNardeState>(*this);
  // History management (optional, for performance/memory)
  const size_t kMaxSafeHistorySize = 100; 
  if (IsTerminal() || turn_history_info_.size() > kMaxSafeHistorySize) {
    new_state->turn_history_info_.clear(); // Clear history on terminal or large states
    #ifndef NDEBUG // Only print warning in debug builds
    if (turn_history_info_.size() > kMaxSafeHistorySize) {
      std::cout << "Warning: Cloning state with large history (" << turn_history_info_.size() 
                << "), clearing history in clone." << std::endl;
    }
    #endif
  }
  return new_state;
}

// ===== Validation Functions =====

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
      checkers_on_board += board(player, i);
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
       checkers_on_board += board(player, i);
    }
    // Check indices 18-23 (points 19-24)
    for (int i = kBlackHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
       checkers_on_board += board(player, i);
    }
    // Also count checkers *inside* home
    for (int i = kBlackHomeStart; i <= kBlackHomeEnd; ++i) {
        checkers_on_board += board(player, i);
    }
  }
  
  // Final check: Ensure the total count matches (all checkers are either in home or borne off)
  return (checkers_on_board + scores_[player] == kNumCheckersPerPlayer);
}

bool LongNardeState::IsHeadPos(int player, int pos) const {
  return (player == kXPlayerId && pos == kWhiteHeadPos) ||
         (player == kOPlayerId && pos == kBlackHeadPos);
}

bool LongNardeState::IsFirstTurn(int player) const {
  // The first turn is characterized by having all 15 checkers on the head point.
  int head_pos = (player == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;
  return board_[player][head_pos] == kNumCheckersPerPlayer;
}

bool LongNardeState::IsLegalHeadMove(int player, int from_pos) const {
  bool is_head = IsHeadPos(player, from_pos);
  if (!is_head) return true; // Not a head move, always allowed by this rule.

  // Head Rule 5: Only 1 checker may leave the head per turn.
  // Exception: First turn double 6, 4, or 3 allows moving 2 checkers.

  // Use the member variable 'is_first_turn_' which reflects the turn status
  // at the beginning of the turn, not the current simulation state.
  // bool is_this_players_first_turn = IsFirstTurn(player); // DON'T use this
  bool is_special_double_roll = false;

  // *** Use initial_dice_ for this check ***
  if (initial_dice_.size() >= 2) { // Check the roll at the start of the turn
      int die1_val = initial_dice_[0]; // Raw value is fine here (1-6)
      int die2_val = initial_dice_[1];
      if (die1_val == die2_val && (die1_val == 3 || die1_val == 4 || die1_val == 6)) {
          is_special_double_roll = true;
      }
  }

  // Check for first turn special doubles exception
  // Use the member variable 'is_first_turn_' here
  if (is_first_turn_ && is_special_double_roll) {
    // On special first turn doubles, we can move up to two checkers from head.
    // This function checks the validity of a *single* potential move.
    // If the conditions are met, this specific move is allowed by the head rule.
    // The limit of two moves is handled implicitly by the sequence generation
    // (RecLegalMoves) and its depth limit combined with state updates.
    return true; // Allow potential head move during special first turn double.
  }

  // Normal case (not first turn OR not a special double roll):
  // Can only move from head if no checker has moved from head *yet* this turn.
  return !moved_from_head_;
}

bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const {
  // Create a temporary board reflecting the potential move
  std::vector<std::vector<int>> temp_board = board_;
  if (from_pos >= 0 && from_pos < kNumPoints) {
     if (temp_board[player][from_pos] <= 0) {
        // Trying to move from an empty point - should have been caught earlier, but handle defensively.
        // This move itself is invalid, but doesn't inherently form an illegal bridge yet.
        // Consider this case as not forming an *additional* illegal bridge.
        return false; 
     }
     temp_board[player][from_pos]--;
  }
  // Don't check bounds for to_pos yet, might be bearing off
   if (to_pos >= 0 && to_pos < kNumPoints) {
      temp_board[player][to_pos]++;
   } else if (!IsOff(player, to_pos)) {
       // Invalid 'to_pos' that isn't bear off - move is illegal, but not specifically a bridge issue.
       return false;
   }


  int opponent = Opponent(player);
  bool opponent_has_checkers_on_board = false;
  for (int i = 0; i < kNumPoints; ++i) {
    if (temp_board[opponent][i] > 0) {
        opponent_has_checkers_on_board = true;
        break;
      }
  }

  // If opponent has no checkers left on the board, no bridge can possibly trap them.
  if (!opponent_has_checkers_on_board) {
      return false;
  }

  // Check all possible 6-point spans for a block
  for (int start = 0; start < kNumPoints; ++start) {
    bool is_block = true;
    for (int i = 0; i < 6; ++i) {
      int pos = (start + i) % kNumPoints;
      if (temp_board[player][pos] == 0) {
        is_block = false;
        break;
      }
    }

    if (is_block) {
      // Found a 6-block. Check if it's illegal.
      // Rule: Illegal if NO opponent checker is ahead of the block's start (from opponent's perspective).
      // "Ahead" means further along the opponent's path (higher path index).
      int block_path_start_on_opp_path_real_pos = GetBlockPathStartRealPos(opponent, start);

      bool is_legal_bridge = false; // Assume illegal until proven otherwise
      for (int opp_pos_idx = 0; opp_pos_idx < kNumPoints; ++opp_pos_idx) {
        if (temp_board[opponent][opp_pos_idx] > 0) {
            // Check if this opponent checker is ahead of the block's starting point.
            if (IsAhead(opponent, opp_pos_idx, block_path_start_on_opp_path_real_pos)) {
                 // Found an opponent checker ahead. Bridge is legal.
                is_legal_bridge = true;
                break; // No need to check other opponent checkers or other blocks starting here.
            }
        }
      }

      // If after checking all opponent checkers, none were found ahead, the bridge is illegal.
      if (!is_legal_bridge) {
        return true; // Illegal bridge would be formed
      }
      // Otherwise (is_legal_bridge is true), this specific block is legal. Continue checking other potential blocks.
    }
  }

  return false; // No illegal bridge found
}

// Checks the current board state for an illegal bridge for the given player.
bool LongNardeState::HasIllegalBridge(int player) const {
   // This just calls WouldFormBlockingBridge without simulating a move.
   // We pass invalid from/to positions to check the *current* board state.
   return WouldFormBlockingBridge(player, /*from_pos=*/-1, /*to_pos=*/-1);
}

bool LongNardeState::IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule) const {
  // --- Basic Checks ---
  if (from_pos == kPassPos) return true; // Pass is always valid in isolation.
  if (from_pos < 0 || from_pos >= kNumPoints) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid from_pos " << from_pos << std::endl;
    return false;
  }
  if (board(player, from_pos) <= 0) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: No checker at from_pos " << from_pos << std::endl;
    return false;
  }
  if (die_value < 1 || die_value > 6) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid die_value " << die_value << std::endl;
    return false;
  }
  // Check if the provided 'to_pos' matches the calculated destination
  int expected_to_pos = GetToPos(player, from_pos, die_value);
  if (to_pos != expected_to_pos) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: to_pos " << to_pos << " doesn't match expected " << expected_to_pos << " for die " << die_value << " from " << from_pos << std::endl;
    return false;
  }

  // --- Head Rule Check ---
  if (check_head_rule && !IsLegalHeadMove(player, from_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Head rule violation for pos " << from_pos << std::endl;
    return false;
  }

  // --- Bearing Off Checks ---
  bool is_bearing_off = IsOff(player, to_pos);
  if (is_bearing_off) {
    // Check if all checkers are in home *directly* here
    int checkers_outside_home = 0;
    for (int pos_check = 0; pos_check < kNumPoints; ++pos_check) { // Use different loop variable
        if (board(player, pos_check) > 0) {
            bool is_home = (player == kXPlayerId) ? 
                            (pos_check >= kWhiteHomeStart && pos_check <= kWhiteHomeEnd) :
                            (pos_check >= kBlackHomeStart && pos_check <= kBlackHomeEnd);
            if (!is_home) {
                checkers_outside_home += board(player, pos_check);
            }
        }
    }
    if (checkers_outside_home > 0) {
        if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Direct check failed - Cannot bear off, " << checkers_outside_home << " checkers outside home" << std::endl;
        return false;
    }
    // Original check commented out:
    // if (!AllInHome(player)) {
    //     if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Cannot bear off, not all checkers in home" << std::endl;
    //     return false;
    // }

    // Calculate exact pips needed to bear off from 'from_pos'
    int pips_needed;
    if (player == kXPlayerId) {
        // White bears off from pos 0-5. Pips needed = pos + 1.
        SPIEL_CHECK_GE(from_pos, kWhiteHomeStart); // Should be in home
        SPIEL_CHECK_LE(from_pos, kWhiteHomeEnd);
        pips_needed = from_pos + 1;
    } else { // kOPlayerId
        // Black bears off from pos 12-17. Pips needed = pos - 11.
        SPIEL_CHECK_GE(from_pos, kBlackHomeStart); // Should be in home
        SPIEL_CHECK_LE(from_pos, kBlackHomeEnd);
        pips_needed = from_pos - 11; 
    }


    if (die_value == pips_needed) {
        return true; // Exact roll bears off.
    }
    if (die_value > pips_needed) {
        // Higher roll can bear off *only if* no checkers are further back.
        if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Checking higher roll (die=" << die_value << " > needed=" << pips_needed << ") for pos=" << from_pos << std::endl;
        bool further_checker_exists = false;
        // Need to check positions *within the home board* that require *more* pips than 'from_pos'.
        if (player == kXPlayerId) {
            for (int check_pos = from_pos + 1; check_pos <= kWhiteHomeEnd; ++check_pos) {
                 if (board(player, check_pos) > 0) {
                     further_checker_exists = true;
                     break;
                 }
            }
        } else { // kOPlayerId
             // INCORRECT for Black: still checks pos+1 up to home end - Corrected below
             // Correct: Check positions requiring *more* pips, which are *lower* indices for Black within home [12..17]
             for (int check_pos = from_pos - 1; check_pos >= kBlackHomeStart; --check_pos) { // Iterate downwards
                 if (board(player, check_pos) > 0) {
                     further_checker_exists = true;
                     break;
                 }
            }
        }
        // int current_path_idx = GetPathIndex(player, from_pos); // Old logic using path index
        // // Check positions corresponding to path indices GREATER than current_path_idx
        // for (int path_idx = current_path_idx + 1; path_idx < kNumPoints; ++path_idx) {
        //      int check_pos = GetPosFromPathIndex(player, path_idx); // Need this helper
        //      if (board(player, check_pos) > 0) {
        //          further_checker_exists = true;
        //          break;
        //      }
        // }
        if (!further_checker_exists) {
             return true; // Can bear off with higher roll
        }
    }
    // If die_value < pips_needed, it's an invalid bear off move.
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid bearing off move (die=" << die_value << " < needed=" << pips_needed << ")" << std::endl;
    return false;
  }
  
  // --- Regular Move Checks ---
  // Check destination bounds (already implicitly checked by GetToPos if not bearing off)
  if (to_pos < 0 || to_pos >= kNumPoints) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid to_pos " << to_pos << " for non-bearoff" << std::endl;
    return false; // Should be unreachable if GetToPos is correct and not bearing off
  }

  // Check opponent occupancy at destination
  if (board(Opponent(player), to_pos) > 0) {
     if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Cannot land on opponent's checker at " << to_pos << std::endl;
    return false;
  }

  // Check if the move *would* form an illegal blocking bridge
  if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Would form illegal blocking bridge from " << from_pos << " to " << to_pos << std::endl;
    return false;
  }

  // If all checks passed, it's a valid move.
  return true;
}

bool LongNardeState::ValidateAction(Action action) const {
  if (IsChanceNode() || IsTerminal()) return false; // Actions only valid for current player

  if (action < 0 || action >= NumDistinctActions()) {
     if (kDebugging) std::cout << "DEBUG ValidateAction: Action " << action << " out of range [0, " << NumDistinctActions() << ")" << std::endl;
    return false;
  }
  
  // The most reliable validation is checking if it's in the set of legal actions.
  const auto& legal_actions = LegalActions(); // Calculate legal actions
  if (std::find(legal_actions.begin(), legal_actions.end(), action) == legal_actions.end()) {
    if (kDebugging) {
      std::cout << "DEBUG ValidateAction: Action " << action << " not found in legal actions.\n";
      std::cout << "DEBUG: Decoded moves for invalid action " << action << ":\n";
       try {
           std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
            for (const auto& m : moves) {
                std::cout << "  pos=" << m.pos << ", to_pos=" << GetToPos(cur_player_, m.pos, m.die)
                        << ", die=" << m.die << "\n";
            }
       } catch(...) { std::cout << "   <Decoding failed>" << std::endl;}
      std::cout << "DEBUG: Current dice: ";
      for (int d : dice_) { std::cout << DiceValue(d) << " "; } // Show actual value
      std::cout << "\nDEBUG: Board state:\n" << ToString() << "\n";
       std::cout << "DEBUG: Legal actions (" << legal_actions.size() << " total): ";
       for (Action a : legal_actions) { std::cout << a << " "; }
       std::cout << "\n";
    }
    return false;
  }
  
  // Optional: Perform consistency checks on the decoded move sequence itself, 
  // even though it was found in LegalActions. This helps catch bugs in encoding/decoding or LegalActions.
  // Note: This duplicates some logic but can be useful for debugging.
  #ifndef NDEBUG // Only run these checks in debug builds
  try {
        std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
        // Simulate applying the moves on a cloned state to verify step-by-step validity
        std::unique_ptr<State> temp_state_ptr = this->Clone();
        LongNardeState* temp_state = dynamic_cast<LongNardeState*>(temp_state_ptr.get());
        
        bool sequence_valid = true;
        for (const auto& move : moves) {
            if (move.pos == kPassPos) continue; // Skip passes here

            // Check validity *in the context of the temporary state*
            if (!temp_state->IsValidCheckerMove(temp_state->cur_player_, move.pos, move.to_pos, move.die, /*check_head_rule=*/true)) {
                 if (kDebugging) {
                     std::cout << "ERROR ValidateAction: Decoded move [" << move.pos << "->" << move.to_pos << "/" << move.die 
                               << "] from legal action " << action << " is INVALID at its step in sequence!" << std::endl;
                     std::cout << "  Temp State Board:\n" << temp_state->ToString() << std::endl;
                 }
                 sequence_valid = false;
                 break;
            }
            // Apply the move to the temp state for the next check
             temp_state->ApplyCheckerMove(temp_state->cur_player_, move);
        }

        if (!sequence_valid) {
            // Consider logging or asserting here if an action from LegalActions fails validation.
             // SpielFatalError("Inconsistency: Action from LegalActions failed sequence validation.");
             return false; // Treat as invalid if sequence check fails
        }

  } catch (...) {
       if (kDebugging) std::cout << "ERROR ValidateAction: Exception during validation decode/simulation for action " << action << std::endl;
       return false; // Decoding/simulation error means invalid
  }
  #endif // NDEBUG

  return true; // Action is in the legal set
}


bool LongNardeState::IsOff(int player, int pos) const {
  return pos == kBearOffPos; // kBearOffPos is the special value indicating off the board
}


// ===== Move Generation =====

std::set<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player) const {
  std::set<CheckerMove> half_moves;
  bool is_debugging = false; // Keep general debugging off unless needed
  
  if (is_debugging) {
    std::cout << "GenerateAllHalfMoves for player " << player << "\n";
    std::cout << "Dice: "; 
    for(int d : dice_) { std::cout << DiceValue(d) << (UsableDiceOutcome(d)?"":"(used)") << " "; }
    std::cout << "\nBoard:\n";
    std::cout << ToString(); // Use ToString for full board context
    std::cout << "All checkers in home? " << (AllInHome(player) ? "YES" : "NO") << "\n";
    std::cout << "Moved from head this turn? " << (moved_from_head_ ? "YES" : "NO") << "\n";
    std::cout << "Is first turn? " << (is_first_turn_ ? "YES" : "NO") << "\n";
  }
  
  // For each checker belonging to the player
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (board(player, pos) <= 0) continue;
    
    if (is_debugging) {
      std::cout << "  Checking checker at pos " << pos << " (point " << (player==kXPlayerId ? 24-pos : (pos<=11?12-pos:36-pos)) << ")\n";
    }
    
    // For each usable die
    for (int i = 0; i < dice_.size(); ++i) {
      int outcome = dice_[i];
      if (!UsableDiceOutcome(outcome)) {
        if (is_debugging) std::cout << "    Die " << DiceValue(outcome) << " (raw " << outcome <<") not usable, skipping\n";
        continue; // Skip used dice
      }
      
      int die_value = outcome; // Since UsableDiceOutcome passed, outcome is 1-6
      int to_pos = GetToPos(player, pos, die_value);
      
      if (is_debugging) {
        std::cout << "    Checking die " << die_value << ", calculated to_pos=" << to_pos 
                  << (IsOff(player, to_pos) ? " (Bear Off)" : "") << "\n";
      }
      
      // Check if this specific half-move is valid *now*
      // Crucially includes the head rule check based on current `moved_from_head_` state.
      bool is_valid = IsValidCheckerMove(player, pos, to_pos, die_value, /*check_head_rule=*/true); 
      
      if (is_debugging) {
        std::cout << "    IsValidCheckerMove (with head check) returned " << (is_valid ? "true" : "false") << "\n";
      }
      
      if (is_valid) {
        half_moves.insert(CheckerMove(pos, to_pos, die_value));
        if (is_debugging) {
          std::cout << "    Added valid move: pos=" << pos << ", to_pos=" << to_pos 
                    << ", die=" << die_value << "\n";
        }
      }
    }
  }
  
  // If no valid moves were found after checking all checkers and dice,
  // the player *must* pass. We add a pass move placeholder.
  // The encoding function will handle assigning dice values to the pass.
  if (half_moves.empty()) {
       // Add a single pass move. LegalActions/Encoding will handle using correct dice.
       // Use die=1 as a placeholder.
      half_moves.insert(CheckerMove(kPassPos, kPassPos, 1)); 
      if (is_debugging) {
         std::cout << "  No regular moves found. Added placeholder pass move.\n";
      }
  }
  
  if (is_debugging) {
    std::cout << "Generated " << half_moves.size() << " potential half-moves for this step:\n";
    for (const auto& move : half_moves) {
      std::cout << "  - from=" << move.pos << ", to=" << move.to_pos 
                << ", die=" << move.die << "\n";
    }
  }
  
  return half_moves;
}

// Recursive helper for LegalActions. Explores possible move sequences.
int LongNardeState::RecLegalMoves(const std::vector<CheckerMove>& moveseq,
                                  std::set<std::vector<CheckerMove>>* movelist,
                                  int max_depth) {
  // Safety limits to prevent excessive recursion/computation
  const size_t kMaxTotalSequences = 200; 
  const size_t kMaxBranchingFactor = 30; // Max half-moves to explore per step
  const int kMaxRecursionDepth = 6; // Should be > 4 for doubles

  if (movelist->size() >= kMaxTotalSequences || max_depth > kMaxRecursionDepth) {
     #ifndef NDEBUG
      if (movelist->size() >= kMaxTotalSequences) std::cerr << "Warning: RecLegalMoves hit sequence limit (" << kMaxTotalSequences << ")" << std::endl;
      if (max_depth > kMaxRecursionDepth) std::cerr << "Warning: RecLegalMoves hit recursion depth limit (" << kMaxRecursionDepth << ")" << std::endl;
     #endif
    // Add current sequence if non-empty, as it's a valid (though possibly incomplete) sequence end-point due to limits
    if (!moveseq.empty() && movelist->find(moveseq) == movelist->end()) {
        movelist->insert(moveseq);
    }
    return moveseq.size(); 
  }

  // Generate next possible *single* half-moves from the current state
  std::set<CheckerMove> half_moves = GenerateAllHalfMoves(cur_player_);

  // Base Case 1: No moves possible from this state (only pass was generated or set was empty).
  if (half_moves.empty() || (half_moves.size() == 1 && (*half_moves.begin()).pos == kPassPos) ) {
    // Add the sequence found so far (if any moves were made).
    // If moveseq is empty, it means no moves were possible from the start (pass turn).
    if (movelist->find(moveseq) == movelist->end()) { // Avoid duplicates
        movelist->insert(moveseq);
    }
    return moveseq.size();
  }
  
  // Base Case 2: Max number of moves for this turn reached (e.g., 2 for non-doubles, 4 for doubles).
  if (moveseq.size() >= max_depth) {
       if (movelist->find(moveseq) == movelist->end()) { // Avoid duplicates
           movelist->insert(moveseq);
       }
      return moveseq.size();
  }


  // --- Recursive Step ---
  size_t moves_checked_this_level = 0;
  int max_len_found_downstream = moveseq.size(); // Track max length found *from this point*

  for (const auto& move : half_moves) {
    // Skip the placeholder pass move if other moves are available
    if (move.pos == kPassPos) continue; 

    // Check limits before processing the move
    if (movelist->size() >= kMaxTotalSequences) return max_len_found_downstream; // Early exit if limit hit during iteration
    if (moves_checked_this_level >= kMaxBranchingFactor) break; // Limit branching factor
    moves_checked_this_level++;

    // --- Simulate applying the move and recursing ---
    std::vector<CheckerMove> next_moveseq = moveseq; // Copy current sequence
    next_moveseq.push_back(move);                   // Add the chosen half-move

    // DEBUG: Print state BEFORE applying move
    if (kDebugging) {
        std::cout << "DEBUG RecLegalMoves: BEFORE Apply {pos=" << move.pos << ", die=" << move.die << "}: Dice: ";
        for (int d_idx = 0; d_idx < dice_.size(); ++d_idx) { std::cout << DiceValue(d_idx) << (UsableDiceOutcome(dice_[d_idx])?"(U)":"(X)") << " "; }
        std::cout << " Board[12]="<< board(cur_player_, 12) << " Board[13]="<< board(cur_player_, 13) << std::endl;
    }

    bool original_moved_from_head = moved_from_head_; // Save state part not handled by Undo
    ApplyCheckerMove(cur_player_, move); // Apply move to *this* state object (will be undone later)

    // DEBUG: Print state AFTER applying move, BEFORE recursion
    if (kDebugging) {
        std::cout << "DEBUG RecLegalMoves: AFTER Apply {pos=" << move.pos << ", die=" << move.die << "}:  Dice: ";
        for (int d_idx = 0; d_idx < dice_.size(); ++d_idx) { std::cout << DiceValue(d_idx) << (UsableDiceOutcome(dice_[d_idx])?"(U)":"(X)") << " "; }
        std::cout << " Board[12]="<< board(cur_player_, 12) << " Board[13]="<< board(cur_player_, 13) << std::endl;
    }

    // *** Check for momentary illegal bridge ***
    // This check ensures that even intermediate positions within a sequence are valid.
    if (HasIllegalBridge(cur_player_)) {
        // This move temporarily created an illegal bridge. This path is invalid.
        // Backtrack the move and continue to the next possible half-move.
        UndoCheckerMove(cur_player_, move);
        moved_from_head_ = original_moved_from_head; // Restore head move status
        // Do NOT add next_moveseq to movelist
        continue; // Skip the recursive call for this invalid path
    }

    // Recursive call for the next move in the sequence
    int child_max_len = RecLegalMoves(next_moveseq, movelist, max_depth);

    // Backtrack state after recursive call returns
    UndoCheckerMove(cur_player_, move);
    moved_from_head_ = original_moved_from_head; // Restore head move status

    // DEBUG: Print state AFTER undoing move
    if (kDebugging) {
        std::cout << "DEBUG RecLegalMoves: AFTER Undo {pos=" << move.pos << ", die=" << move.die << "}:   Dice: ";
        for (int d_idx = 0; d_idx < dice_.size(); ++d_idx) { std::cout << DiceValue(d_idx) << (UsableDiceOutcome(dice_[d_idx])?"(U)":"(X)") << " "; }
        std::cout << " Board[12]="<< board(cur_player_, 12) << " Board[13]="<< board(cur_player_, 13) << std::endl;
    }

    // Update the maximum sequence length found so far among all branches explored from this node
    // Update the maximum sequence length found so far among all branches explored from this node
    max_len_found_downstream = std::max(child_max_len, max_len_found_downstream);

     // Check limit again after recursive call returns
     if (movelist->size() >= kMaxTotalSequences) return max_len_found_downstream;
  }

   // If we explored moves (moves_checked_this_level > 0) but didn't find any sequence
   // reaching the full max_depth from this point (e.g. only one die playable),
   // then the sequence leading *to* this state is a valid end-point.
   // Add the 'moveseq' (the sequence *before* exploring this level's half_moves)
   // unless it was already added in a base case.
   if (moves_checked_this_level > 0 && max_len_found_downstream < (moveseq.size() + 1) ) {
       if (!moveseq.empty() && movelist->find(moveseq) == movelist->end()) {
            movelist->insert(moveseq);
       }
   }


  return max_len_found_downstream;
}


// Deprecated/Unused: ProcessLegalMoves logic is now integrated into LegalActions.
std::vector<Action> LongNardeState::ProcessLegalMoves(
    int max_moves, const std::set<std::vector<CheckerMove>>& movelist) const {
    // This function is likely no longer needed as LegalActions handles the filtering.
    // Return empty vector or log a warning if called.
    #ifndef NDEBUG
     std::cerr << "Warning: LongNardeState::ProcessLegalMoves called (likely deprecated)." << std::endl;
    #endif
    std::vector<Action> legal_moves; 
    // Replicate essential filtering if necessary, but prefer LegalActions logic.
    // Simplified version: Convert valid sequences to actions
     for (const auto& moveseq : movelist) {
         // Basic filtering based on expected max_moves might be useful if called directly
         if (moveseq.size() <= max_moves && !moveseq.empty()) { 
              Action action = CheckerMovesToSpielMove(moveseq);
              legal_moves.push_back(action);
         }
     }
     // Remove duplicates if needed
     std::sort(legal_moves.begin(), legal_moves.end());
     legal_moves.erase(std::unique(legal_moves.begin(), legal_moves.end()), legal_moves.end());
    return legal_moves;
}

// Deprecated/Unused: LegalCheckerMoves logic replaced by GenerateAllHalfMoves.
std::set<CheckerMove> LongNardeState::LegalCheckerMoves(int player) const {
    // This function is likely no longer needed. GenerateAllHalfMoves is used.
     #ifndef NDEBUG
      std::cerr << "Warning: LongNardeState::LegalCheckerMoves called (likely deprecated)." << std::endl;
     #endif
  return GenerateAllHalfMoves(player); // Delegate to the current function
}

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

void LongNardeState::ApplyCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Nothing to do for a pass move

  // Re-validate the move *without* the head rule check here.
  // The head rule is context-dependent (how many moved *before* this one)
  // and is handled during sequence generation (GenerateAllHalfMoves/RecLegalMoves).
  // This check ensures basic validity (on board, not blocked, valid destination).
  if (!IsValidCheckerMove(player, move.pos, move.to_pos, move.die, /*check_head_rule=*/false)) {
    std::string error_message = absl::StrCat("ApplyCheckerMove: Invalid checker move provided! ",
                                           "Player ", player, " Move: ", move.pos, "->", move.to_pos, "/", move.die);
     error_message += "\nBoard state:\n" + ToString();
     error_message += "\nDice: ";
      for (int d : dice_) { error_message += absl::StrCat(DiceValue(d), UsableDiceOutcome(d)?" ":"u "); }
     error_message += "\nMoved from head? ", (moved_from_head_?"Y":"N");
     error_message += "\nIs first turn? ", (is_first_turn_?"Y":"N");
    SpielFatalError(error_message);
  }

  // Perform the move on the board
  SPIEL_CHECK_GE(move.pos, 0); // Should be guaranteed by IsValidCheckerMove if not pass
  SPIEL_CHECK_LT(move.pos, kNumPoints);
  SPIEL_CHECK_GT(board_[player][move.pos], 0); // Must have a checker to move
  board_[player][move.pos]--;

  // Mark the die used (find the first usable die with that value)
  bool die_marked = false;
  for (int i = 0; i < dice_.size(); ++i) {
    if (UsableDiceOutcome(dice_[i]) && dice_[i] == move.die) {
      dice_[i] += 6; // Mark as used by adding 6
      die_marked = true;
                    break;
                  }
                }
  SPIEL_CHECK_TRUE(die_marked); // Should always find a usable die if the move was valid

  // Update destination (board or score)
  int next_pos = move.to_pos; 
  if (IsOff(player, next_pos)) {
    scores_[player]++;
    SPIEL_CHECK_LE(scores_[player], kNumCheckersPerPlayer);
              } else {
    // Ensure destination is valid board position (should be guaranteed by IsValidCheckerMove)
    SPIEL_CHECK_GE(next_pos, 0);
    SPIEL_CHECK_LT(next_pos, kNumPoints);
    board_[player][next_pos]++;
  }

  // Update head move status *for the current turn's sequence*
  if (IsHeadPos(player, move.pos)) {
    moved_from_head_ = true;
  }
}

void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Nothing to undo for a pass

  // Check consistency: should have a valid starting position
  SPIEL_CHECK_GE(move.pos, 0); 
  SPIEL_CHECK_LT(move.pos, kNumPoints);

  // Restore checker to the starting position
  board_[player][move.pos]++;
  SPIEL_CHECK_LE(board_[player][move.pos], kNumCheckersPerPlayer);

  // Unmark the die used (find the first *used* die matching the value)
   bool die_unmarked = false;
   for (int i = 0; i < dice_.size(); ++i) {
     // Check if dice_[i] represents the used version of move.die
     if (dice_[i] == move.die + 6) { 
       dice_[i] -= 6; // Unmark by subtracting 6
       die_unmarked = true;
       break;
     }
   }
   // If this fails, it indicates a major inconsistency in state/undo logic.
   if (!die_unmarked) {
        std::string error_msg = "UndoCheckerMove: Could not find used die to unmark. ";
        error_msg += absl::StrCat("Player ", player, ", Move ", move.pos, "->", move.to_pos, "/", move.die);
        error_msg += "\nDice state: ";
         for (int d : dice_) { error_msg += absl::StrCat(d, " "); }
         error_msg += "\nBoard:\n" + ToString();
         SpielFatalError(error_msg);
   }

  // Reverse the effect on the destination
  int next_pos = move.to_pos; 
  if (IsOff(player, next_pos)) {
    // If it was a bear-off move, decrement the score
    scores_[player]--;
    SPIEL_CHECK_GE(scores_[player], 0);
  } else {
    // If it was a regular move, remove checker from the destination
    // Ensure destination is valid before decrementing
    SPIEL_CHECK_GE(next_pos, 0);
    SPIEL_CHECK_LT(next_pos, kNumPoints);
    SPIEL_CHECK_GT(board_[player][next_pos], 0); // Must have been a checker there
    board_[player][next_pos]--;
  }

  // Note: Undoing moved_from_head_ is handled by the caller (RecLegalMoves)
  // by restoring the value from before the ApplyCheckerMove call.
}


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

int LongNardeState::GetToPos(int player, int from_pos, int pips) const {
  SPIEL_CHECK_GE(from_pos, 0);
  SPIEL_CHECK_LT(from_pos, kNumPoints);
  SPIEL_CHECK_GE(pips, 1);
  SPIEL_CHECK_LE(pips, 6);

  if (player == kXPlayerId) { // White path: 23 -> 0 (decreasing index)
    int target_idx = from_pos - pips;
    // Check if the move takes the checker off the board (past index 0)
    if (target_idx < 0) {
      return kBearOffPos; // Bear off
    }
    return target_idx; // Regular move
  } else { // kOPlayerId (Black path: 11 -> 0 -> 23 -> 12 (complex index changes))
    int current_pos = from_pos;
    for (int i = 0; i < pips; ++i) {
      if (current_pos == 0) { // Wrap around from 0 to 23
        current_pos = 23;
      } else if (current_pos == 12) { // Trying to move from point 13 (index 12)
         // Any move from index 12 goes off the board for Black
        return kBearOffPos; 
      } else { // Normal move: decrement index
        current_pos--;
      }

      // Additional check: If a move starts at or after point 13 (index >= 12)
      // and *lands* at or before point 12 (index <= 11) *during* the pip count,
      // it means the checker crossed the finish line and should bear off.
      // We need to compare the position *before* this single pip step.
      int pos_before_step; 
       if (current_pos == 23) pos_before_step = 0; // Wrapped around
       else if (current_pos == kBearOffPos) pos_before_step = 12; // Just went off
       else pos_before_step = current_pos + 1; // Normal decrement

      if (pos_before_step >= 12 && current_pos <= 11) {
           // Crossed the finish line (from index 12+ to 11-) mid-move
           return kBearOffPos;
      }
    }

    // After moving 'pips' steps, check final position
    // REMOVED: This check incorrectly treated landing on index 12 as bear off.
    // if (current_pos == 12) { // Landed exactly on the bear-off threshold index
    //     return kBearOffPos;
    // }

    // If still on the board, return the final index.
    SPIEL_CHECK_GE(current_pos, 0); 
    SPIEL_CHECK_LT(current_pos, kNumPoints); 
    return current_pos;
  }
}


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

std::string LongNardeState::ToString() const {
  std::vector<std::string> board_array = {
      "+-------------------------------------+", // Exactly matches expected width
      "|13 14 15 16 17 18| |19 20 21 22 23 24|", // No extra space before final |
      "|                 | |                 |", // Top checker row
      "|                 | |                 |", // Ownership row for double-digit piles
      "|                 | |                 |", // Spacer row
      "|                 | |                 |", // Bottom checker row
      "|12 11 10  9  8  7| |6  5  4  3  2  1 |", // Space between '1' and final |
      "+-------------------------------------+"  // Bottom border
  };
  const int PADDING_TOP = 2;    // Top checker row
  const int OWNERSHIP_ROW = 3;  // New row for double-digit ownership
  const int PADDING_BOT = 5;    // Bottom checker row (incremented from 4 to 5)
  const int BAR_COL = 19;       // Column index where the bar starts

  // Fill the board representation
  for (int player = 0; player < 2; ++player) {
      char symbol = (player == kXPlayerId ? 'X' : 'O');
      
      for (int pos_idx = 0; pos_idx < kNumPoints; ++pos_idx) {
          int count = board(player, pos_idx);
          if (count > 0) {
              int row;
              int col;

              // Determine row based on point index
              if (pos_idx >= 12) { // Points 13-24 go on top row
                  row = PADDING_TOP;
                  col = 1 + (pos_idx - 12) * 3; 
              } else { // Points 1-12 go on bottom row
                  row = PADDING_BOT;
                  col = 1 + (11 - pos_idx) * 3; 
              }
              
              // Adjust for bar separator
              if (col >= BAR_COL) col += 2; // Skip the "| |" bar (2 chars wide)
              
              // Place checker symbol and count
              if (count < 10) {
                  // Single-digit format: Keep "X1" or "O5" format
                  board_array[row][col] = symbol; 
                  board_array[row][col + 1] = '0' + count;
              } else { 
                  // Double-digit format: Show "15" on checker row
                  board_array[row][col] = '0' + (count / 10);
                  board_array[row][col + 1] = '0' + (count % 10);
                  
                  // Show player symbol on ownership row
                  int ownership_row = (row == PADDING_TOP) ? OWNERSHIP_ROW : OWNERSHIP_ROW + 1;
                  board_array[ownership_row][col] = symbol;
              }
          }
      }
  }

  // Use single backslash for newline separator
  std::string board_str = absl::StrJoin(board_array, "\n") + "\n";

  // Add game state information
  absl::StrAppend(&board_str, "Turn: ");
  absl::StrAppend(&board_str, CurPlayerToString(cur_player_));
  if (cur_player_ != kChancePlayerId && cur_player_ != kTerminalPlayerId) {
      absl::StrAppend(&board_str, (is_first_turn_ ? " (First Turn)" : ""));
      absl::StrAppend(&board_str, (is_playing_extra_turn_ ? " (Extra Turn)" : ""));
  }
  absl::StrAppend(&board_str, "\n");
  
  absl::StrAppend(&board_str, "Dice: ");
  if (dice_.empty() && cur_player_ != kChancePlayerId) {
       absl::StrAppend(&board_str, "(None rolled yet)");
  } else if (dice_.empty() && cur_player_ == kChancePlayerId) {
       absl::StrAppend(&board_str, "(Waiting for roll)");
  } else {
      for (size_t i = 0; i < dice_.size(); ++i) {
        if (i > 0) absl::StrAppend(&board_str, " ");
        absl::StrAppend(&board_str, std::to_string(DiceValue(i)));
        if (!UsableDiceOutcome(dice_[i])) absl::StrAppend(&board_str, "(u)"); // Mark used dice
      }
  }
  absl::StrAppend(&board_str, "\n");
  
  absl::StrAppend(&board_str, "Scores: X (White): ", scores_[kXPlayerId]);
  absl::StrAppend(&board_str, ", O (Black): ", scores_[kOPlayerId], "\n");
  
  if (moved_from_head_) {
    absl::StrAppend(&board_str, "Status: Head checker moved this turn.\n");
  }
  if (double_turn_) {
     absl::StrAppend(&board_str, "Status: Next roll is for an extra turn.\n");
  }
   if (allow_last_roll_tie_) {
     absl::StrAppend(&board_str, "Status: Last roll tie attempt allowed.\n");
   }

  return board_str;
}


ScoringType ParseScoringType(const std::string& st_str) {
  if (st_str == "winloss_scoring") {
    return ScoringType::kWinLossScoring;
  } else if (st_str == "winlosstie_scoring") {
    return ScoringType::kWinLossTieScoring;
  } else {
    SpielFatalError("Unrecognized scoring_type parameter: " + st_str);
    return ScoringType::kWinLossScoring; // Should be unreachable
  }
}

// ===== Game Class Methods =====

LongNardeGame::LongNardeGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(ParseScoringType(
          params.count("scoring_type") > 0 ?
          params.at("scoring_type").string_value() :
          kDefaultScoringType)) {}

double LongNardeGame::MaxUtility() const {
  return 2.0; // Max score is 2 for mars/gammon
}


}  // namespace long_narde
}  // namespace open_spiel