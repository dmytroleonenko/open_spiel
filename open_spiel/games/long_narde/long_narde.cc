/**
 *   Long Narde Rules:
 *  1. Setup: White's 15 checkers on point 24; Black's 15 on point 12.
    2. Movement: Both move checkers CCW into home (White 1–6, Black 13–18), then bear off.
    3. Starting: Each rolls 1 die; higher is White and goes first. But in our open_spiel implementation white is always first without the dice roll
    4. Turns: Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves exist, skip; if only one is possible, use the higher die.
    5. Head Rule: Only 1 checker may leave the head (White 24, Black 12) per turn. Exception on the first turn: if you roll double 6, 4, or 3, you can move 2 checkers from the head; after that, no more head moves.
    6. Bearing Off: Once all your checkers reach home, bear them off with exact or higher rolls.
    7. Ending/Scoring: Game ends when someone bears off all. If the loser has none off, winner scores 2 (mars); otherwise 1 (oin). Some events allow a last roll to tie.
    8. Block (Bridge): You cannot form a contiguous block of 6 checkers unless at least 1 opponent checker is still ahead of it. Fully trapping all 15 opponent checkers is banned—even a momentary (going through in a sequence of moves) 6‑block that would leave no opponent checkers in front is disallowed.
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
namespace {

// Constants for encoding half–moves.
// We now encode both the starting board position and the die value.
// For a non‑pass move, we use: digit = pos * 6 + (die - 1)   (range: 0 to 143)
// For a pass move (pos==kPassPos) we reserve digits 144–149 (one for each die value).
constexpr int kDigitBase = 150;   // our "base" for each encoded half–move
constexpr int kPassOffset = 144;    // pass moves: digit = 144 + (die - 1)

// New constants for encoding doubles moves (which can have up to 4 half-moves)
constexpr int kEncodingBaseDouble = 25;  // Base for encoding positions in doubles (0-23 + pass)
constexpr int kDoublesOffset = 2 * kDigitBase * kDigitBase;  // Offset for doubles encoding space

constexpr int kNumOffPosHumanReadable = -2;
constexpr int kNumBarPosHumanReadable = -3;
constexpr int kNumNonDoubleOutcomes = 15;

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
}  // namespace

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
    return PositionToString(pos);
  }
}

int LongNardeState::GetMoveEndPosition(CheckerMove* cmove, int player,
                                        int start) const {
  int end = cmove->die;
  if (cmove->pos != kPassPos) {
    end = start - cmove->die;
    if (end <= 0) {
      end = kNumOffPosHumanReadable;
    }
  }
  return end;
}

//=== NEW ENCODING/DECODING FUNCTIONS FOR ACTIONS ============================
//
// We now encode each half–move as a "digit" that includes both the starting position and die value.
// For non–pass moves: digit = pos * 6 + (die – 1)   (range: 0 to 143)
// For pass moves:       digit = kPassOffset + (die – 1) (range: 144 to 149)
// Then the overall action is encoded as:
//     action = (second_digit * kDigitBase) + first_digit
//
// An extra block is reserved (adding kDigitBase²) for ordering if needed.

Action LongNardeState::CheckerMovesToSpielMove(
    const std::vector<CheckerMove>& moves) const {
  SPIEL_CHECK_LE(moves.size(), 4);  // Allow up to 4 moves for doubles

  // Check if this is a doubles roll
  bool is_doubles = false;
  if (dice_.size() == 2 && DiceValue(0) == DiceValue(1)) {
    is_doubles = true;
  }

  if (is_doubles && moves.size() > 2) {
    // Special encoding for doubles with >2 moves
    // We encode up to 4 positions in base-25 (0-23 for board positions, 24 for pass)
    std::vector<int> positions(4, kEncodingBaseDouble - 1);  // Default to pass (encoded as 24)
    
    // Fill the positions array with actual positions from moves
    for (size_t i = 0; i < moves.size() && i < 4; ++i) {
      if (moves[i].pos == kPassPos) {
        positions[i] = kEncodingBaseDouble - 1;  // kPassPos encoded as 24
      } else {
        SPIEL_CHECK_GE(moves[i].pos, 0);
        SPIEL_CHECK_LT(moves[i].pos, kNumPoints);
        positions[i] = moves[i].pos;
      }
    }
    
    // Encode using base-25: positions[0] is least significant, positions[3] is most significant
    Action action_double = 0;
    for (int i = 3; i >= 0; --i) {
      action_double = action_double * kEncodingBaseDouble + positions[i];
    }
    
    // Add offset to distinguish from non-doubles encoding
    Action action = kDoublesOffset + action_double;
    
    SPIEL_CHECK_GE(action, kDoublesOffset);
    SPIEL_CHECK_LT(action, NumDistinctActions());
    return action;
  } else {
    // Original encoding for non-doubles or doubles with ≤2 moves
    // The sequence 'moves' is guaranteed by LegalActions to be valid in this order.
    // We encode moves[0] as dig0 and moves[1] as dig1 directly.
    std::vector<CheckerMove> encoded_moves = moves; // Use a copy to add padding if needed

    // Ensure we have at least 2 moves for encoding by adding Pass moves
    while (encoded_moves.size() < 2) {
      // Add pass moves as needed
      int die_val = 1;  // Default
      // Try to find an *unused* die value if possible for the pass padding
      int available_die = -1;
      if (dice_.size() >= 1 && UsableDiceOutcome(dice_[0])) available_die = DiceValue(0);
      else if (dice_.size() >= 2 && UsableDiceOutcome(dice_[1])) available_die = DiceValue(1);

      if (available_die != -1) {
          die_val = available_die;
      } else if (!encoded_moves.empty() && encoded_moves[0].die > 0) {
          // Fallback: use the die from the first move if no dice info available
          die_val = encoded_moves[0].die;
      }
      // Ensure die_val is valid (1-6)
      die_val = std::max(1, std::min(6, die_val));
      encoded_moves.push_back(CheckerMove(kPassPos, kPassPos, die_val));
    }

    auto encode_move = [](const CheckerMove& move) -> int {
      if (move.pos == kPassPos) {
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return kPassOffset + (move.die - 1);
      } else {
        SPIEL_CHECK_GE(move.pos, 0);
        SPIEL_CHECK_LT(move.pos, kNumPoints);
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return move.pos * 6 + (move.die - 1);
      }
    };

    // Encode the moves in the order they were provided.
    int dig0 = encode_move(encoded_moves[0]);
    int dig1 = encode_move(encoded_moves[1]);

    // The action is encoded with the second move's digit in the higher base position.
    Action action = dig1 * kDigitBase + dig0;

    // Determine if the *actual* dice roll (if available) had the lower die first.
    bool actual_low_roll_first = false;
    if (dice_.size() >= 2) {
        // Use DiceValue to handle potential encoding (7-12) if dice were marked used.
        // This function should ideally be called on a state *before* moves are applied.
        int die0_val = DiceValue(0);
        int die1_val = DiceValue(1);
        if (die0_val < die1_val) {
            actual_low_roll_first = true;
        }
    }
    // If dice_.size() < 2, actual_low_roll_first remains false.

    // Add offset only if the actual dice roll had low die first.
    // This distinguishes between (e.g.) moving 5 then 3 when roll was 5,3 vs 3,5.
    // This offset should NOT be added when encoding a generic pass move where dice aren't relevant.
    // Let's only add the offset if it's not a double pass move.
    bool is_double_pass = (encoded_moves.size() == 2 && encoded_moves[0].pos == kPassPos && encoded_moves[1].pos == kPassPos);
    if (actual_low_roll_first && !is_double_pass) {
      action += kDigitBase * kDigitBase;
    }

    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kDoublesOffset);
    return action;
  }
}

std::vector<CheckerMove> LongNardeState::SpielMoveToCheckerMoves(
    Player player, Action spiel_move) const {
  // Check if this is a doubles encoding (actions >= kDoublesOffset)
  if (spiel_move >= kDoublesOffset) {
    // Decode doubles action with up to 4 positions
    Action action_double = spiel_move - kDoublesOffset;
    
    // Extract positions using base-25 decoding
    std::vector<int> positions(4);
    for (int i = 0; i < 4; ++i) {
      positions[i] = action_double % kEncodingBaseDouble;
      action_double /= kEncodingBaseDouble;
    }

    // Determine die value to use for all moves (should be the same for doubles)
    int die_val = 1;  // Default
    if (dice_.size() > 0) {
      die_val = DiceValue(0);  // Use first die value
    }

    // Create checker moves
    std::vector<CheckerMove> cmoves;
    for (int i = 0; i < 4; ++i) {
      int pos;
      if (positions[i] == kEncodingBaseDouble - 1) {
        // This position was encoded as 24, which means kPassPos
        pos = kPassPos;
      } else {
        pos = positions[i];
      }
      
      if (pos == kPassPos) {
        cmoves.push_back(CheckerMove(kPassPos, kPassPos, die_val));
      } else {
        int to_pos = GetToPos(player, pos, die_val);
        cmoves.push_back(CheckerMove(pos, to_pos, die_val));
      }
    }
    
    return cmoves;
  } else {
    // Original decoding for non-doubles actions
    bool high_roll_first = spiel_move < (kDigitBase * kDigitBase);
    if (!high_roll_first) {
      spiel_move -= kDigitBase * kDigitBase;
    }
    int dig0 = spiel_move % kDigitBase;
    int dig1 = spiel_move / kDigitBase;

    auto decode_digit = [this, player](int digit) -> CheckerMove {
      if (digit >= kPassOffset) {
        int die = (digit - kPassOffset) + 1;
        return CheckerMove(kPassPos, kPassPos, die);
      } else {
        int pos = digit / 6;
        int die = (digit % 6) + 1;
        int to_pos = GetToPos(player, pos, die);
        return CheckerMove(pos, to_pos, die);
      }
    };

    std::vector<CheckerMove> cmoves;
    cmoves.push_back(decode_digit(dig0));
    cmoves.push_back(decode_digit(dig1));
    return cmoves;
  }
}

// FIX: Added declaration for NumDistinctActions() to match header.
int LongNardeState::NumDistinctActions() const {
  // Total distinct actions = non-doubles range + doubles range
  // Non-doubles range: 2 * (kDigitBase * kDigitBase)
  // Doubles range: kEncodingBaseDouble^4 (for encoding 4 positions in base 25)
  int double_range_size = 1;
  for (int i = 0; i < 4; ++i) {
    double_range_size *= kEncodingBaseDouble;  // kEncodingBaseDouble^4
  }
  return kDoublesOffset + double_range_size;
}
//=== END NEW ENCODING/DECODING FUNCTIONS =====================================

std::string LongNardeState::ActionToString(Player player,
                                            Action move_id) const {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move_id, 0);
    SPIEL_CHECK_LT(move_id, kChanceOutcomes.size());
    if (turns_ >= 0) {
      return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
    } else {
      const char* starter = (move_id < kNumNonDoubleOutcomes ? "X starts" : "O starts");
      Action outcome_id = move_id;
      if (outcome_id >= kNumNonDoubleOutcomes) {
        outcome_id -= kNumNonDoubleOutcomes;
      }
      SPIEL_CHECK_LT(outcome_id, kChanceOutcomeValues.size());
      return absl::StrCat("chance outcome ", move_id, " ", starter, ", ",
                          "(roll: ", kChanceOutcomeValues[outcome_id][0],
                          kChanceOutcomeValues[outcome_id][1], ")");
    }
  }

  std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(player, move_id);

  int cmove0_start, cmove1_start;
  if (player == kOPlayerId) {
    cmove0_start = cmoves[0].pos + 1;
    cmove1_start = cmoves[1].pos + 1;
  } else {
    cmove0_start = kNumPoints - cmoves[0].pos;
    cmove1_start = kNumPoints - cmoves[1].pos;
  }

  int cmove0_end = GetMoveEndPosition(&cmoves[0], player, cmove0_start);
  int cmove1_end = GetMoveEndPosition(&cmoves[1], player, cmove1_start);

  std::string returnVal = "";
  if (cmove0_start == cmove1_start && cmove0_end == cmove1_end) {
    if (cmoves[1].pos == kPassPos) {
      returnVal = "Pass";
    } else {
      returnVal = absl::StrCat(move_id, " - ",
                               PositionToStringHumanReadable(cmove0_start),
                               "/", PositionToStringHumanReadable(cmove0_end),
                               "(2)");
    }
  } else if ((cmove0_start < cmove1_start ||
              (cmove0_start == cmove1_start && cmove0_end < cmove1_end) ||
              cmoves[0].pos == kPassPos) &&
             cmoves[1].pos != kPassPos) {
    if (cmove1_end == cmove0_start) {
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
          PositionToStringHumanReadable(cmove1_end), "/",
          PositionToStringHumanReadable(cmove0_end));
    } else {
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
          PositionToStringHumanReadable(cmove1_end), " ",
          (cmoves[0].pos != kPassPos) ? PositionToStringHumanReadable(cmove0_start) : "",
          (cmoves[0].pos != kPassPos) ? "/" : "",
          PositionToStringHumanReadable(cmove0_end));
    }
  } else {
    if (cmove0_end == cmove1_start) {
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
          PositionToStringHumanReadable(cmove0_end), "/",
          PositionToStringHumanReadable(cmove1_end));
    } else {
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
          PositionToStringHumanReadable(cmove0_end), " ",
          (cmoves[1].pos != kPassPos) ? PositionToStringHumanReadable(cmove1_start) : "",
          (cmoves[1].pos != kPassPos) ? "/" : "",
          PositionToStringHumanReadable(cmove1_end));
    }
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

  for (int i = 0; i < kNumPoints; ++i) {
    *value_it++ = board(player, i);
  }
  for (int i = 0; i < kNumPoints; ++i) {
    *value_it++ = board(opponent, i);
  }

  *value_it++ = scores_[player];
  *value_it++ = (cur_player_ == player) ? 1 : 0;
  *value_it++ = scores_[opponent];
  *value_it++ = (cur_player_ == opponent) ? 1 : 0;
  *value_it++ = (!dice_.empty()) ? dice_[0] : 0;
  *value_it++ = (dice_.size() > 1) ? dice_[1] : 0;

  SPIEL_CHECK_EQ(value_it, values.end());
}

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

int LongNardeState::board(int player, int pos) const {
  // When bearing off, we might check positions outside the board
  // Return 0 for any position that's out of bounds
  if (pos < 0 || pos >= kNumPoints) {
    return 0;
  }
  return board_[player][pos];
}

Player LongNardeState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int LongNardeState::Opponent(int player) const { return 1 - player; }

void LongNardeState::RollDice(int outcome) {
  int die1 = kChanceOutcomeValues[outcome][0];
  int die2 = kChanceOutcomeValues[outcome][1];
  if (die1 != die2 && die1 < die2) {
    dice_.push_back(die2);
    dice_.push_back(die1);
  } else {
    dice_.push_back(die1);
    dice_.push_back(die2);
  }
}

int LongNardeState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());
  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else if (dice_[i] >= 7 && dice_[i] <= 12) {
    return dice_[i] - 6;
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

bool LongNardeState::IsHeadPos(int player, int pos) const {
  return (player == kXPlayerId && pos == kWhiteHeadPos) ||
         (player == kOPlayerId && pos == kBlackHeadPos);
}

bool LongNardeState::IsLegalHeadMove(int player, int from_pos) const {
  bool is_debugging = false;
  bool is_head = IsHeadPos(player, from_pos);
  bool result = !is_head || is_first_turn_ || !moved_from_head_;
  if (is_debugging && from_pos == 6) {
    std::cout << "DEBUG: IsLegalHeadMove for pos 6:" << std::endl;
    std::cout << "DEBUG:   IsHeadPos(" << player << ", " << from_pos << ") = "
              << (is_head ? "true" : "false") << std::endl;
    std::cout << "DEBUG:   head_pos = " << (player == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos) << std::endl;
    std::cout << "DEBUG:   is_first_turn = " << (is_first_turn_ ? "true" : "false") << std::endl;
    std::cout << "DEBUG:   moved_from_head = " << (moved_from_head_ ? "true" : "false") << std::endl;
    std::cout << "DEBUG:   Result = " << (result ? "PASS" : "FAIL") << std::endl;
  }
  return result;
}

bool LongNardeState::IsFirstTurn(int player) const {
  int head_pos = (player == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;
  return board_[player][head_pos] == kNumCheckersPerPlayer;
}

bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const {
  // Create temporary board state
  std::vector<std::vector<int>> temp_board = board_;
  if (from_pos >= 0 && from_pos < kNumPoints) {
    temp_board[player][from_pos]--;
  }
  if (to_pos >= 0 && to_pos < kNumPoints) {
    temp_board[player][to_pos]++;
  }

  // Helper function to check for 6 consecutive checkers
  auto has_six_consecutive = [&](int start) -> bool {
    for (int i = 0; i < 6; ++i) {
      int pos = (start + i) % kNumPoints;
      if (temp_board[player][pos] == 0) return false;
    }
    return true;
  };

  // Check all possible 6-consecutive blocks
  for (int start = 0; start < kNumPoints; ++start) {
    if (!has_six_consecutive(start)) continue;

    int end = (start + 5) % kNumPoints;
    int opponent = Opponent(player);
    bool any_opponent_ahead = false;

    // Iterate through ALL board positions to find opponent checkers
    for (int i = 0; i < kNumPoints; ++i) {
      // Check if the current position 'i' is part of the bridge
      bool is_part_of_bridge;
      if (start <= end) { // Bridge does not wrap around
        is_part_of_bridge = (i >= start && i <= end);
      } else { // Bridge wraps around (e.g., 22, 23, 0, 1, 2, 3)
        is_part_of_bridge = (i >= start || i <= end);
      }

      // If 'i' is NOT part of the bridge AND the opponent occupies it...
      if (!is_part_of_bridge && temp_board[opponent][i] > 0) {
        // ...then there is an opponent checker ahead of the bridge.
        any_opponent_ahead = true;
        break; // No need to check further
      }
    }

    // If no opponent checkers were found ahead, this bridge formation is illegal.
    if (!any_opponent_ahead) {
      return true;
    }
  }

  return false;
}

bool LongNardeState::IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule) const {
  bool is_debugging = false; // Keep general debugging off unless needed
  // Add specific debugging for the problematic move O: 11 -> 12
  bool specific_debug = (player == kOPlayerId && from_pos == 11 && to_pos == 12);

  if (from_pos == kPassPos) return true;
  if (from_pos < 0 || from_pos >= kNumPoints) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Invalid from_pos " << from_pos << std::endl;
    return false;
  }
  if (board(player, from_pos) <= 0) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: No checker at from_pos " << from_pos << std::endl;
    return false;
  }
  if (die_value < 1 || die_value > 6) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Invalid die_value " << die_value << std::endl;
    return false;
  }
  int expected_to_pos = GetToPos(player, from_pos, die_value);
  if (to_pos != expected_to_pos) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: to_pos " << to_pos << " doesn't match expected " << expected_to_pos << std::endl;
    return false;
  }
  if (check_head_rule && !IsLegalHeadMove(player, from_pos)) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Head rule violation for pos " << from_pos << std::endl;
    return false;
  }
  bool is_bearing_off = IsOff(player, to_pos);
  if (is_bearing_off) {
    if (!AllInHome(player)) {
      if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Cannot bear off, not all checkers in home" << std::endl;
      return false;
    }
    int exact_roll = (player == kXPlayerId) ? (from_pos + 1) : (kNumPoints - from_pos);
    if (die_value == exact_roll) return true;
    if (die_value > exact_roll) {
        if (player == kXPlayerId) {
            for (int pos = from_pos + 1; pos <= kWhiteHomeEnd; pos++) {
                if (board(player, pos) > 0) {
                    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Cannot bear off high (X), checker at " << pos << std::endl;
                    return false;
                }
            }
        } else {
            for (int pos = kBlackHomeStart; pos < from_pos; pos++) {
                if (board(player, pos) > 0) {
                    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Cannot bear off high (O), checker at " << pos << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Invalid bearing off move (die too low)" << std::endl;
    return false; // Die too low
  }
  
  // Check bounds for non-bearing off moves
  if (to_pos < 0 || to_pos >= kNumPoints) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Invalid to_pos " << to_pos << std::endl;
    return false;
  }

  // *** Critical Check: Opponent Occupancy ***
  // Only check opponent occupancy if this is a normal move (not bearing off)
  if (specific_debug) {
    std::cout << "DEBUG IsValidCheckerMove (O: 11->12): Checking opponent occupancy at to_pos=" << to_pos << std::endl;
    std::cout << "  board(X, " << to_pos << ") = " << board(1-player, to_pos) << std::endl;
    // Also print neighbors for context
    std::cout << "  Context: O[" << board(player, 11) << "," << board(player, 12) << "," << board(player, 13) << "] "
              << "X[" << board(1-player, 11) << "," << board(1-player, 12) << "," << board(1-player, 13) << "]" << std::endl;
  }
  
  // Only check opponent occupancy if position is on the board
  if (to_pos >= 0 && to_pos < kNumPoints && board(1-player, to_pos) > 0) {
    if (is_debugging || specific_debug) std::cout << "  -> FAILED: Opponent found at " << to_pos << std::endl;
    return false;
  }
  // *** End Critical Check ***

  // *** Critical Check: Blocking Bridge ***
  if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
    if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Would form illegal blocking bridge" << std::endl;
    return false;
  }

  return true;
}

// Process a chance roll (dice roll) action.
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
  RollDice(move_id);

  // Decide which player starts or continues.
  if (turns_ < 0) {
    // White always starts, ignore dice outcomes
    turns_ = 0;
    cur_player_ = prev_player_ = kXPlayerId;
    is_playing_extra_turn_ = false;
  } else if (double_turn_) {
    // Extra turn in progress (from doubles).
    cur_player_ = prev_player_;
    is_playing_extra_turn_ = true;  // Mark that this is an extra turn
  } else {
    // Normal turn progression: pass to the opponent.
    cur_player_ = Opponent(prev_player_);
    is_playing_extra_turn_ = false;  // Reset for normal turn
  }
  
  double_turn_ = false;  // Reset after using it

  // Check special condition for last-roll tie.
  if (scores_[kXPlayerId] == kNumCheckersPerPlayer &&
      scores_[kOPlayerId] >= 14 &&
      scores_[kOPlayerId] < kNumCheckersPerPlayer) {
    allow_last_roll_tie_ = true;
  }
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

bool LongNardeState::IsPosInHome(int player, int pos) const {
  switch (player) {
    case kXPlayerId:
      return (pos >= kWhiteHomeStart && pos <= kWhiteHomeEnd);
    case kOPlayerId:
      return (pos >= kBlackHomeStart && pos <= kBlackHomeEnd);
    default:
      SpielFatalError(absl::StrCat("Unknown player ID: ", player));
  }
}

bool LongNardeState::AllInHome(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LE(player, 1);
  if (player == kXPlayerId) {
    // White's home is points 1-6 (indices 0-5).
    // Check if any checkers are outside this range (indices 6-23).
    // All checkers must be at index <= 5.
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) { // Check 6 to 23
      if (board(player, i) > 0) {
        return false;
      }
    }
  } else { // kOPlayerId
    // Black's home starts at point 13 (index 12).
    // For bearing off eligibility, all checkers must be at index 12 or higher.
    // Check indices 0 to 11 (before the home stretch).
    for (int i = 0; i < kBlackHomeStart; ++i) { // Check 0 to 11
      if (board(player, i) > 0) {
        // Found a checker before the start of the home area (index 12).
        return false;
      }
    }
    // If no checkers are found in indices 0-11, all must be >= 12.
  }
  // If we reach here, all checkers are in the required zone for bearing off.
  return true;
}

bool LongNardeState::IsTerminal() const {
  if (scores_[kXPlayerId] == kNumCheckersPerPlayer ||
      scores_[kOPlayerId] == kNumCheckersPerPlayer) {
    if (scoring_type_ == ScoringType::kWinLossTieScoring) {
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer &&
          scores_[kOPlayerId] >= 14 && scores_[kOPlayerId] < kNumCheckersPerPlayer) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::vector<double> LongNardeState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  }
  if (scoring_type_ == ScoringType::kWinLossTieScoring) {
    if (scores_[kXPlayerId] == kNumCheckersPerPlayer &&
        scores_[kOPlayerId] == kNumCheckersPerPlayer) {
      return {0.0, 0.0};
    }
  }
  int won = (scores_[kXPlayerId] == kNumCheckersPerPlayer ? kXPlayerId : kOPlayerId);
  int lost = Opponent(won);
  double score = (scores_[lost] > 0) ? 1.0 : 2.0;
  return (won == kXPlayerId ? std::vector<double>{score, -score} : std::vector<double>{-score, score});
}

LongNardeGame::LongNardeGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(ParseScoringType(
          params.count("scoring_type") > 0 ?
          params.at("scoring_type").string_value() :
          kDefaultScoringType)) {}

double LongNardeGame::MaxUtility() const {
  return 2;
}

std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (turns_ == -1) {
    return kChanceOutcomes;
  } else {
    return kChanceOutcomes;
  }
}

bool LongNardeState::ValidateAction(Action action) const {
  if (action < 0 || action >= NumDistinctActions()) {
    return false;
  }
  const auto& legal_actions = LegalActions();
  if (std::find(legal_actions.begin(), legal_actions.end(), action) == legal_actions.end()) {
    std::cout << "DEBUG: Action " << action << " not found in legal actions.\n";
    std::cout << "DEBUG: Legal actions (" << legal_actions.size() << " total): ";
    for (Action a : legal_actions) {
      std::cout << a << " ";
    }
    std::cout << "\n";
    std::cout << "DEBUG: Decoded moves for invalid action " << action << ":\n";
    std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
    for (const auto& m : moves) {
      std::cout << "  pos=" << m.pos << ", to_pos=" << GetToPos(cur_player_, m.pos, m.die)
                << ", die=" << m.die << "\n";
    }
    std::cout << "DEBUG: Current dice: ";
    for (int d : dice_) {
      std::cout << d << " ";
    }
    std::cout << "\nDEBUG: Board state:\n" << ToString() << "\n";
    return false;
  }
  std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
  for (const auto& move : moves) {
    if (move.pos == kPassPos) continue;
    int to_pos = GetToPos(cur_player_, move.pos, move.die);
    if (!IsValidCheckerMove(cur_player_, move.pos, to_pos, move.die, true)) {
      std::cout << "ERROR: Decoded move from " << move.pos << " to " << to_pos
                << " with die=" << move.die << " for player " << cur_player_
                << " is not valid but action was in legal actions!" << std::endl;
      std::cout << "Dice: [" << (dice_.size() >= 1 ? std::to_string(dice_[0]) : "none")
                << ", " << (dice_.size() >= 2 ? std::to_string(dice_[1]) : "none") << "]" << std::endl;
      std::cout << "Board state: " << ToString() << std::endl;
      return false;
    }
  }
  return true;
}

std::set<CheckerMove> LongNardeState::LegalCheckerMoves(int player) const {
  std::set<CheckerMove> moves;
  bool is_debugging = false;
  if (is_debugging) {
    std::cout << "DEBUG: LegalCheckerMoves for player " << player << std::endl;
    std::cout << "DEBUG: Board state for player " << player << ": ";
    for (int i = 0; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        std::cout << i << ":" << board(player, i) << " ";
      }
    }
    std::cout << std::endl;
    std::cout << "DEBUG: is_first_turn=" << is_first_turn_ << std::endl;
    std::cout << "DEBUG: dice values: ";
    for (int die : dice_) {
      std::cout << die << " ";
    }
    std::cout << std::endl;
  }
  // Safely check for doubles
  bool is_doubles = false;
  if (dice_.size() == 2) {
      is_doubles = (DiceValue(0) == DiceValue(1));
  }

  for (int i = 0; i < kNumPoints; ++i) {
    if (board(player, i) <= 0) continue;
    for (int outcome : dice_) { // outcome can be 1-6 (unused) or 7-12 (used)
      if (!UsableDiceOutcome(outcome)) { // Check if outcome is 1-6 (i.e., die is available)
        continue;
      }
      int die_value = outcome; // Since UsableDiceOutcome passed, outcome is 1-6
      int to_pos = GetToPos(player, i, die_value);
      if (IsValidCheckerMove(player, i, to_pos, die_value, true)) {
        moves.insert(CheckerMove(i, to_pos, die_value));
      } else if (is_debugging) {
        if (to_pos >= 0 && to_pos < kNumPoints && board(1-player, to_pos) > 0) {
          std::cout << "DEBUG: Cannot land on opponent's checker at " << to_pos << std::endl;
        } else {
          std::cout << "DEBUG: Invalid move from " << i << " to " << to_pos << " with die=" << die_value << " for player " << player << std::endl;
        }
      }
    }
  }
  if (is_debugging) {
    std::cout << "DEBUG: Generated " << moves.size() << " legal moves" << std::endl;
    for (const auto& move : moves) {
      std::cout << "DEBUG: Legal move: pos=" << move.pos << ", to_pos=" << move.to_pos << ", die=" << move.die << std::endl;
    }
  }
  return moves;
}

void LongNardeState::ApplyCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return;
  // Check validity without head rule, as RecLegalMoves handles sequence validity
  // Note: We rely on this check being correct *at the time it's called*.
  if (!IsValidCheckerMove(player, move.pos, move.to_pos, move.die, /*check_head_rule=*/false)) {
    std::string error_message = absl::StrCat("ApplyCheckerMove: Invalid checker move from ", move.pos,
                                           " to ", move.to_pos, " with die=", move.die,
                                           " for player ", player);
    // Add board state to error
     error_message += "\nBoard state:\n" + ToString();
    SpielFatalError(error_message);
    // No return needed after SpielFatalError
  }

  // Add specific debugging for the problematic move O: 11 -> 12 and its context
  bool specific_debug = (player == kOPlayerId && (move.pos == 11 || move.to_pos == 12 || move.pos == 13 || move.to_pos == 13));
  if (specific_debug) {
      std::cout << "DEBUG ApplyCheckerMove (O): move=" << move.pos << "->" << move.to_pos << " die=" << move.die << std::endl;
      std::cout << "  Board state BEFORE apply at [11, 12, 13]: O=["
                << board_[kOPlayerId][11] << "," << board_[kOPlayerId][12] << "," << board_[kOPlayerId][13] << "] X=["
                << board_[kXPlayerId][11] << "," << board_[kXPlayerId][12] << "," << board_[kXPlayerId][13] << "]" << std::endl;
  }

  board_[player][move.pos]--;
  // Mark die used
  for (int i = 0; i < dice_.size(); ++i) {
    if (dice_[i] == move.die) {
      dice_[i] += 6;
      break;
    }
  }
  int next_pos = move.to_pos; // Use the pre-calculated to_pos from CheckerMove
  if (IsOff(player, next_pos)) {
    scores_[player]++;
  } else {
    // We assume next_pos is valid because IsValidCheckerMove passed
    board_[player][next_pos]++;
  }
  if (IsHeadPos(player, move.pos)) {
    moved_from_head_ = true;
  }

  if (specific_debug) {
      std::cout << "  Board state AFTER apply at [11, 12, 13]: O=["
                << board_[kOPlayerId][11] << "," << board_[kOPlayerId][12] << "," << board_[kOPlayerId][13] << "] X=["
                << board_[kXPlayerId][11] << "," << board_[kXPlayerId][12] << "," << board_[kXPlayerId][13] << "]" << std::endl;
  }
}

void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return;

  int next_pos = move.to_pos; // Use the pre-calculated to_pos from CheckerMove

  // Add specific debugging for the problematic move O: 11 -> 12 and its context
  bool specific_debug = (player == kOPlayerId && (move.pos == 11 || next_pos == 12 || move.pos == 13 || next_pos == 13));
   if (specific_debug) {
      std::cout << "DEBUG UndoCheckerMove (O): move=" << move.pos << "->" << next_pos << " die=" << move.die << std::endl;
      std::cout << "  Board state BEFORE undo at [11, 12, 13]: O=["
                << board_[kOPlayerId][11] << "," << board_[kOPlayerId][12] << "," << board_[kOPlayerId][13] << "] X=["
                << board_[kXPlayerId][11] << "," << board_[kXPlayerId][12] << "," << board_[kXPlayerId][13] << "]" << std::endl;
  }


  if (IsOff(player, next_pos)) {
    scores_[player]--;
  } else {
    // Ensure the position exists before decrementing (should always be true if Apply was valid)
    if (next_pos >= 0 && next_pos < kNumPoints) {
       board_[player][next_pos]--;
    } else {
         std::cerr << "Warning: UndoCheckerMove attempting to decrement invalid next_pos " << next_pos << std::endl;
    }
  }
  // Unmark die used
  for (int i = 0; i < dice_.size(); ++i) {
    if (dice_[i] == move.die + 6) {
      dice_[i] -= 6;
      break;
    }
  }
  // Ensure the original position exists before incrementing (should always be true)
  if (move.pos >= 0 && move.pos < kNumPoints) {
      board_[player][move.pos]++;
  } else {
      std::cerr << "Warning: UndoCheckerMove attempting to increment invalid move.pos " << move.pos << std::endl;
  }
  // Note: Undoing moved_from_head_ is handled by restoring the state in RecLegalMoves

   if (specific_debug) {
      std::cout << "  Board state AFTER undo at [11, 12, 13]: O=["
                << board_[kOPlayerId][11] << "," << board_[kOPlayerId][12] << "," << board_[kOPlayerId][13] << "] X=["
                << board_[kXPlayerId][11] << "," << board_[kXPlayerId][12] << "," << board_[kXPlayerId][13] << "]" << std::endl;
  }
}

bool LongNardeState::UsableDiceOutcome(int outcome) const {
  return outcome >= 1 && outcome <= 6;
}

std::vector<Action> LongNardeState::ProcessLegalMoves(
    int max_moves, const std::set<std::vector<CheckerMove>>& movelist) const {
  std::vector<Action> legal_moves;
  if (movelist.empty()) {
    return legal_moves;
  }
  const size_t kMaxToProcess = 20;
  legal_moves.reserve(std::min(movelist.size(), kMaxToProcess));
  int head_pos = cur_player_ == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos;
  int sequences_processed = 0;
  for (const auto& moveseq : movelist) {
    if (sequences_processed >= kMaxToProcess) break;
    if (moveseq.size() == max_moves) {
      if (!is_first_turn_) {
        int head_move_count = 0;
        for (const auto& move : moveseq) {
          if (IsHeadPos(cur_player_, move.pos) && move.pos != kPassPos) {
            head_move_count++;
          }
        }
        if (head_move_count > 1) continue;
      }
      Action action = CheckerMovesToSpielMove(moveseq);
      legal_moves.push_back(action);
      sequences_processed++;
      if (legal_moves.size() >= 10) return legal_moves;
    }
  }
  if (legal_moves.empty() && !movelist.empty()) {
    int longest = 0;
    int scan_count = 0;
    for (const auto& moveseq : movelist) {
      if (scan_count++ >= 20) break;
      longest = std::max(longest, static_cast<int>(moveseq.size()));
    }
    sequences_processed = 0;
    for (const auto& moveseq : movelist) {
      if (sequences_processed >= kMaxToProcess) break;
      if (moveseq.size() == longest) {
        if (!is_first_turn_) {
          int head_move_count = 0;
          for (const auto& move : moveseq) {
            if (IsHeadPos(cur_player_, move.pos) && move.pos != kPassPos) {
              head_move_count++;
            }
          }
          if (head_move_count > 1) continue;
        }
        Action action = CheckerMovesToSpielMove(moveseq);
        legal_moves.push_back(action);
        sequences_processed++;
        if (legal_moves.size() >= 10) return legal_moves;
      }
    }
  }
  if (legal_moves.size() > 1 && legal_moves.size() < 20) {
    std::sort(legal_moves.begin(), legal_moves.end());
    auto new_end = std::unique(legal_moves.begin(), legal_moves.end());
    legal_moves.erase(new_end, legal_moves.end());
  }
  return legal_moves;
}

int LongNardeState::RecLegalMoves(const std::vector<CheckerMove>& moveseq,
                                  std::set<std::vector<CheckerMove>>* movelist,
                                  int max_depth) {
  const size_t kSafeLimit = 50; // Limit recursion depth/breadth for safety
  if (movelist->size() >= kSafeLimit) {
    return moveseq.size(); // Return current depth if limit reached
  }

  // Base case 1: Max depth for this path reached.
  if (max_depth <= 0) {
    // Only add non-empty sequences. Empty sequences shouldn't be valid actions.
    if (!moveseq.empty()) {
      movelist->insert(moveseq);
    }
    return moveseq.size();
  }

  // Generate next possible half-moves from the current state
  std::set<CheckerMove> half_moves = GenerateAllHalfMoves(cur_player_);

  // Base case 2: No moves possible from this state (player must pass/stop).
  if (half_moves.empty()) {
    // Add the sequence found so far. It represents a valid end-point for moves.
    // If moveseq is empty, it means no moves were possible from the start.
    if (movelist->find(moveseq) == movelist->end()) {
        movelist->insert(moveseq);
    }
    return moveseq.size();
  }

  // --- Recursive Step ---
  const size_t kMaxMovesToCheck = 15; // Limit branching factor per step
  size_t moves_checked = 0;
  std::vector<CheckerMove> new_moveseq = moveseq; // Create a mutable copy for the next level
  new_moveseq.reserve(moveseq.size() + max_depth); // Reserve potential max size
  int max_len_found = moveseq.size(); // Track max length found *down this path*

  for (const auto& move : half_moves) {
    // Check limits before processing the move
    if (movelist->size() >= kSafeLimit / 2) return max_len_found; // Early exit if close to limit
    if (moves_checked >= kMaxMovesToCheck) break; // Limit branching
    moves_checked++;

    // --- Try applying the move and recursing ---
    bool old_moved_from_head = moved_from_head_; // Save state part not handled by Undo
    bool head_move_would_be_invalid = false;
    
    // Check head rule constraints (separately from the standard validation in GenerateAllHalfMoves)
    if (!is_first_turn_ && IsHeadPos(cur_player_, move.pos) && move.pos != kPassPos) {
      // For normal turns, limit to one head move per turn
      if (moved_from_head_) {
        // Skip this move - would be a second move from head in a normal turn
        head_move_would_be_invalid = true;
      }
    } else if (is_first_turn_ && IsHeadPos(cur_player_, move.pos) && move.pos != kPassPos) {
      // First turn: track head moves for special doubles check
      int head_move_count = 0;
      for (const auto& m : moveseq) {
        if (IsHeadPos(cur_player_, m.pos) && m.pos != kPassPos) {
          head_move_count++;
        }
      }
      
      if (head_move_count > 0) {
        // Already have one head move - check if this is allowed with special doubles
        bool is_special_double = (dice_.size() == 2 && 
                                 DiceValue(0) == DiceValue(1) &&
                                 (DiceValue(0) == 6 || DiceValue(0) == 4 || DiceValue(0) == 3));
        if (!is_special_double) {
          // Not a special double - second head move is not allowed
          head_move_would_be_invalid = true;
        }
      }
    }
    
    // Skip invalid head moves
    if (head_move_would_be_invalid) {
      continue;
    }

    // Apply the move
    new_moveseq.push_back(move);
    ApplyCheckerMove(cur_player_, move);

    // *** Check for momentary illegal bridge ***
    if (HasIllegalBridge(cur_player_)) {
        // This move created an illegal bridge. This path is invalid.
        // Backtrack the move and continue to the next possible move.
        UndoCheckerMove(cur_player_, move);
        moved_from_head_ = old_moved_from_head;
        new_moveseq.pop_back(); // Remove the invalid move from the sequence
        continue; // Skip the recursive call for this invalid path
    }

    // Check limit again before recursion call
    if (movelist->size() >= kSafeLimit / 2) {
      UndoCheckerMove(cur_player_, move); // Backtrack state
      moved_from_head_ = old_moved_from_head;
      return max_len_found;
    }

    // Recursive call for the next move in the sequence
    int child_max = RecLegalMoves(new_moveseq, movelist, max_depth - 1);

    // Backtrack state after recursive call returns
    UndoCheckerMove(cur_player_, move);
    moved_from_head_ = old_moved_from_head;
    new_moveseq.pop_back(); // Remove the move tried in this iteration

    // Update the maximum sequence length found so far in this branch
    max_len_found = std::max(child_max, max_len_found);
  }

  return max_len_found;
}

std::unique_ptr<State> LongNardeState::Clone() const {
  auto new_state = std::make_unique<LongNardeState>(*this);
  const size_t kMaxSafeHistorySize = 100;
  if (IsTerminal() || turn_history_info_.size() > kMaxSafeHistorySize ||
      (IsChanceNode() && dice_.empty())) {
    new_state->turn_history_info_.clear();
    if (turn_history_info_.size() > kMaxSafeHistorySize) {
      std::cout << "Clearing large history of size " << turn_history_info_.size() << std::endl;
    }
  }
  return new_state;
}

std::vector<Action> LongNardeState::IllegalActions() const {
  std::vector<Action> illegal_actions;
  if (IsChanceNode() || IsTerminal()) return illegal_actions;
  int high_roll = DiceValue(0);
  int low_roll = DiceValue(1);
  if (high_roll < low_roll) std::swap(high_roll, low_roll);
  int kMaxActionId = NumDistinctActions();
  for (Action action = 0; action < kMaxActionId; ++action) {
    std::vector<CheckerMove> moves;
    try {
      moves = SpielMoveToCheckerMoves(cur_player_, action);
    } catch (...) {
      illegal_actions.push_back(action);
      continue;
    }
    std::vector<CheckerMove> pass_move_encoding = {kPassMove, kPassMove};
    Action pass_spiel_action = CheckerMovesToSpielMove(pass_move_encoding);
    if (action == pass_spiel_action) continue;
    bool valid_action = true;
    for (const auto& move : moves) {
      if (move.pos == kPassPos) continue;
      if (move.pos > 0 && board_[cur_player_][move.pos] == 0) {
        valid_action = false;
        break;
      }
      int to_pos = GetToPos(cur_player_, move.pos, move.die);
      if (!IsValidCheckerMove(cur_player_, move.pos, to_pos, move.die, true)) {
        valid_action = false;
        break;
      }
      if (to_pos != kNumPoints && to_pos > 0 && board_[1 - cur_player_][to_pos] > 0) {
        valid_action = false;
        break;
      }
    }
    if (!valid_action) {
      illegal_actions.push_back(action);
    }
  }
  return illegal_actions;
}

bool LongNardeState::IsOff(int player, int pos) const {
  return (player == kXPlayerId) ? pos < 0 : pos >= kNumPoints;
}

inline int CounterClockwisePos(int from, int pips, int num_points) {
  int pos = from - pips;
  pos = (pos % num_points + num_points) % num_points;
  return pos;
}

int LongNardeState::GetToPos(int player, int from_pos, int pips) const {
  if (player == kXPlayerId) {
    // White moves counter-clockwise (decreasing positions)
    // A result < 0 means bearing off.
    return from_pos - pips;
  } else { // kOPlayerId
    // Black moves counter-clockwise (increasing positions towards 23, then off)
    int potential_pos = from_pos + pips;
    // If the potential position is >= kNumPoints, it means the checker bears off.
    // We return kNumPoints (or any value >= kNumPoints) to signify this.
    // The IsOff function checks for pos >= kNumPoints for Black.
    if (potential_pos >= kNumPoints) {
       return kNumPoints; // Or potential_pos, as long as it's >= kNumPoints
    } else {
       return potential_pos; // Normal move within the board
    }
    // Old incorrect logic: return (from_pos + pips) % kNumPoints;
  }
}

int LongNardeState::FurthestCheckerInHome(int player) const {
  if (player == kXPlayerId) {
    for (int i = kWhiteHomeEnd; i >= kWhiteHomeStart; --i) {
      if (board(player, i) > 0) return i;
    }
  } else {
    for (int i = kBlackHomeStart; i <= kBlackHomeEnd; ++i) {
      if (board(player, i) > 0) return i;
    }
  }
  return -1;
}

std::string LongNardeState::ToString() const {
  std::vector<std::string> board_array = {
      "+----------------------------+",
      "|24 23 22 21 20 19 18 17 16 15 14 13|",
      "|                            |",
      "|                            |",
      "|                            |",
      "|                            |",
      "|01 02 03 04 05 06 07 08 09 10 11 12|",
      "+----------------------------+"
  };

  // Fill the board
  for (int pos = 0; pos < kNumPoints; pos++) {
    int row, col;
    
    if (pos < 12) {
      row = 6;
      col = 1 + pos * 3;
    } else {
      row = 1;
      col = 1 + (23 - pos) * 3;
    }
    
    // X player's checkers
    if (board_[kXPlayerId][pos] > 0) {
      char symbol = 'X';
      int count = board_[kXPlayerId][pos];
      if (count < 10) {
        board_array[row - 1][col] = symbol;
        board_array[row - 1][col + 1] = '0' + count;
      } else {
        board_array[row - 1][col] = '0' + (count / 10);
        board_array[row - 1][col + 1] = '0' + (count % 10);
      }
    }
    
    // O player's checkers
    if (board_[kOPlayerId][pos] > 0) {
      char symbol = 'O';
      int count = board_[kOPlayerId][pos];
      int offset = 2; // Offset for O player
      if (count < 10) {
        board_array[row + 1][col] = symbol;
        board_array[row + 1][col + 1] = '0' + count;
      } else {
        board_array[row + 1][col] = '0' + (count / 10);
        board_array[row + 1][col + 1] = '0' + (count % 10);
      }
    }
  }

  std::string board_str = absl::StrJoin(board_array, "\n") + "\n";

  // Extra info like whose turn it is, dice, etc.
  absl::StrAppend(&board_str, "Turn: ");
  absl::StrAppend(&board_str, CurPlayerToString(cur_player_));
  absl::StrAppend(&board_str, "\n");
  
  absl::StrAppend(&board_str, "Dice: ");
  for (size_t i = 0; i < dice_.size(); ++i) {
    if (i > 0) absl::StrAppend(&board_str, " ");
    absl::StrAppend(&board_str, std::to_string(DiceValue(i)));
  }
  absl::StrAppend(&board_str, "\n");
  
  absl::StrAppend(&board_str, "Scores, X: ", scores_[kXPlayerId]);
  absl::StrAppend(&board_str, ", O: ", scores_[kOPlayerId], "\n");
  
  if (double_turn_) {
    absl::StrAppend(&board_str, "Double turn in progress\n");
  }
  
  if (is_first_turn_) {
    absl::StrAppend(&board_str, "First turn for current player\n");
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
          return actions_using_higher;
      } else if (higher_die_ever_playable) {
          return actions_using_higher;
      } else if (lower_die_ever_playable) {
          return actions_using_lower;
      } else {
          SpielFatalError("Inconsistent state in LegalActions higher die rule: Neither die playable but max_non_pass=1.");
          return legal_moves;
      }
  }

  return legal_moves;
}


// *** NEW HELPER FUNCTION: HasIllegalBridge ***
// Checks the current board state for an illegal bridge for the given player.
bool LongNardeState::HasIllegalBridge(int player) const {
  // Check if player has any 6 consecutive positions, handling wrap-around
  for (int start = 0; start < kNumPoints; ++start) {
    bool consecutive = true;
    
    // Check 6 consecutive positions, with wrap-around handling
    for (int offset = 0; offset < 6; ++offset) {
      int pos = (start + offset) % kNumPoints; // Handle wrap-around
      if (board_[player][pos] == 0) {
        consecutive = false;
        break;
      }
    }
    
    if (!consecutive) continue;
    
    // Found 6 consecutive checkers - determine if bridge is illegal
    int end = (start + 5) % kNumPoints;
    int opponent = Opponent(player);
    bool any_opponent_ahead = false;

    // Iterate through ALL board positions to find opponent checkers
    for (int i = 0; i < kNumPoints; ++i) {
      // Check if the current position 'i' is part of the bridge
      bool is_part_of_bridge;
      if (start <= end) { // Bridge does not wrap around
        is_part_of_bridge = (i >= start && i <= end);
      } else { // Bridge wraps around (e.g., 22, 23, 0, 1, 2, 3)
        is_part_of_bridge = (i >= start || i <= end);
      }

      // If 'i' is NOT part of the bridge AND the opponent occupies it...
      if (!is_part_of_bridge && board_[opponent][i] > 0) {
        // ...then there is an opponent checker ahead of the bridge.
        any_opponent_ahead = true;
        break; // No need to check further
      }
    }

    // If no opponent checkers were found ahead, this bridge formation is illegal.
    if (!any_opponent_ahead) {
      return true;
    }
  }
  
  return false;  // No illegal bridge found
}

// Implement the GenerateAllHalfMoves method
std::set<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player) const {
  std::set<CheckerMove> half_moves;
  bool debugging = true;
  
  if (debugging) {
    std::cout << "GenerateAllHalfMoves for player " << player << "\n";
    std::cout << "Dice: " << (dice_.size() > 0 ? std::to_string(dice_[0]) : "none") 
              << ", " << (dice_.size() > 1 ? std::to_string(dice_[1]) : "none") << "\n";
    std::cout << "Board:\n";
    for (int i = 0; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        std::cout << "  Point " << (i + 1) << " (pos " << i << "): " << board(player, i) << " checkers\n";
      }
    }
    std::cout << "Scores: X=" << scores_[kXPlayerId] << ", O=" << scores_[kOPlayerId] << "\n";
    std::cout << "All checkers in home? " << (AllInHome(player) ? "YES" : "NO") << "\n";
  }
  
  // For each checker belonging to the player
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (board(player, pos) <= 0) continue;
    
    if (debugging) {
      std::cout << "  Found checker at pos " << pos << " (point " << pos+1 << ")\n";
    }
    
    // For each usable die
    for (int i = 0; i < dice_.size(); ++i) {
      int outcome = dice_[i];
      if (!UsableDiceOutcome(outcome)) {
        if (debugging) std::cout << "    Die " << outcome << " not usable, skipping\n";
        continue; // Skip used dice
      }
      
      int die_value = outcome; // Since UsableDiceOutcome passed, outcome is 1-6
      int to_pos = GetToPos(player, pos, die_value);
      
      if (debugging) {
        std::cout << "    Checking die " << die_value << ", to_pos=" << to_pos << "\n";
        if (IsOff(player, to_pos)) {
          std::cout << "    This would be a bearing off move\n";
        }
      }
      
      // Check if this would be a valid move (including bearing off)
      bool is_valid = IsValidCheckerMove(player, pos, to_pos, die_value, true);
      
      if (debugging) {
        std::cout << "    IsValidCheckerMove returned " << (is_valid ? "true" : "false") << "\n";
        if (!is_valid) {
          // Log the reason why this move is invalid
          if (board(1-player, to_pos) > 0 && to_pos >= 0 && to_pos < kNumPoints) {
            std::cout << "    -> Invalid because: opponent occupies destination\n";
          }
          else if (IsOff(player, to_pos) && !AllInHome(player)) {
            std::cout << "    -> Invalid because: cannot bear off, not all checkers in home\n";
          }
          else if (IsOff(player, to_pos)) {
            int exact_roll = (player == kXPlayerId) ? (pos + 1) : (kNumPoints - pos);
            if (die_value < exact_roll) {
              std::cout << "    -> Invalid because: die too small for bearing off\n";
            } else if (die_value > exact_roll) {
              // Check if there are checkers further out
              bool has_further_checkers = false;
              if (player == kXPlayerId) {
                for (int p = pos + 1; p <= kWhiteHomeEnd; p++) {
                  if (board(player, p) > 0) {
                    has_further_checkers = true;
                    break;
                  }
                }
              } else {
                for (int p = kBlackHomeStart; p < pos; p++) {
                  if (board(player, p) > 0) {
                    has_further_checkers = true;
                    break;
                  }
                }
              }
              if (has_further_checkers) {
                std::cout << "    -> Invalid because: die larger than needed but checkers exist further out\n";
              }
            }
          }
          else if (!IsLegalHeadMove(player, pos)) {
            std::cout << "    -> Invalid because: violates head rule\n";
          }
        }
      }
      
      if (is_valid) {
        half_moves.insert(CheckerMove(pos, to_pos, die_value));
        if (debugging) {
          std::cout << "    Added valid move: pos=" << pos << ", to_pos=" << to_pos 
                    << ", die=" << die_value << "\n";
        }
      }
    }
  }
  
  // If no valid moves, add a pass move for each usable die
  if (half_moves.empty()) {
    for (int i = 0; i < dice_.size(); ++i) {
      int outcome = dice_[i];
      if (UsableDiceOutcome(outcome)) {
        half_moves.insert(CheckerMove(kPassPos, kPassPos, outcome));
        if (debugging) {
          std::cout << "  Added pass move with die " << outcome << "\n";
        }
      }
    }
  }
  
  if (debugging) {
    std::cout << "Generated " << half_moves.size() << " half-moves:\n";
    for (const auto& move : half_moves) {
      std::cout << "  - from=" << move.pos << ", to=" << move.to_pos 
                << ", die=" << move.die << "\n";
    }
  }
  
  return half_moves;
}

// Helper function: checks if 'player' has any checker in [startPos, endPos] inclusive.

bool LongNardeState::HasAnyChecker(int player, int startPos, int endPos) const {
  for (int p = startPos; p <= endPos; ++p) {
    if (board_[player][p] > 0) return true;
  }
  return false;
}

}  // namespace long_narde
}  // namespace open_spiel