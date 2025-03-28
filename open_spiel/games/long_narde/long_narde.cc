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
constexpr int kBearOffPos = -1;     // Consistent value for bear-off target
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
    // Convert human-readable point (1-24) to internal index (0-23)
    SPIEL_CHECK_GE(pos, 1);
    SPIEL_CHECK_LE(pos, kNumPoints);
    int internal_pos = pos - 1;
    return PositionToString(internal_pos);
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
      int d1 = (dice_.size() >=1) ? DiceValue(0) : kChanceOutcomeValues[move_id][0];
      int d2 = (dice_.size() >=2) ? DiceValue(1) : kChanceOutcomeValues[move_id][1];
      if (dice_.empty()) {
          d1 = kChanceOutcomeValues[move_id][0];
          d2 = kChanceOutcomeValues[move_id][1];
      }
      return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", d1, d2, ")");
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
          return absl::StrCat(move_id, " - Pass");
      }
      continue;
    }

    any_move = true;
    int start_hr, end_hr;

    if (player == kOPlayerId) {
      start_hr = move.pos + 1;
    } else {
      start_hr = kNumPoints - move.pos;
    }

    if (IsOff(player, move.to_pos)) {
        end_hr = kNumOffPosHumanReadable;
    } else {
         SPIEL_CHECK_GE(move.to_pos, 0);
         SPIEL_CHECK_LT(move.to_pos, kNumPoints);
        if (player == kOPlayerId) {
             end_hr = move.to_pos + 1;
        } else {
             end_hr = kNumPoints - move.to_pos;
        }
    }

    absl::StrAppend(&returnVal, " ", PositionToStringHumanReadable(start_hr), "/",
                    PositionToStringHumanReadable(end_hr));
  }

   if (!any_move) {
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
  bool is_head = IsHeadPos(player, from_pos);
  if (!is_head) return true; // Not a head move

  // Use the member variable which tracks if this *turn* is the first turn
  // bool is_this_players_first_turn = IsFirstTurn(player); // <-- OLD: Checks current board
  bool is_this_players_turn_first = is_first_turn_; // <-- NEW: Use member tracking turn status

  if (is_this_players_turn_first) {
    // Check for special doubles (3,3 / 4,4 / 6,6) using current dice values
    bool is_special_double = false;
    if (dice_.size() >= 2) { // Need at least 2 dice values (used or unused)
        int val1 = DiceValue(0); // Get base value (1-6)
        int val2 = DiceValue(1); // Get base value (1-6)
        if (val1 == val2 && (val1 == 3 || val1 == 4 || val1 == 6)) {
             // Check if the number of dice corresponds to a double roll (2 or 4 available/used)
             // Let's simplify: if the values match a special double, assume it's the roll.
             // The sequence generation depth/dice limits overall moves.
             is_special_double = true;
        }
    }

    if (is_special_double) {
      // On the first turn with special doubles, the head move restriction is eased.
      // Return true here to allow IsValidMove to pass the head rule check.
      // The recursive search depth (max_moves) and dice usage will naturally
      // limit the total number of moves (including head moves) possible.
      // The test just needs *one* sequence showing >1 head move is possible.
      return true;
    } else {
      // First turn, but NOT a special double: only the first head move is allowed.
      return !moved_from_head_;
    }
  } else {
    // Not the player's first turn: only the first head move is allowed.
    return !moved_from_head_;
  }
}

bool LongNardeState::IsFirstTurn(int player) const {
  int head_pos = (player == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;
  // Ensure head_pos is valid before accessing board_
  if (head_pos < 0 || head_pos >= kNumPoints) {
       SpielFatalError(absl::StrCat("IsFirstTurn: Invalid head_pos calculated: ", head_pos));
       return false; // Should not happen
  }
  // Check if the player still has exactly the starting number of checkers on their head point.
  return board_[player][head_pos] == kNumCheckersPerPlayer;
}

bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const {
  std::vector<std::vector<int>> temp_board = board_;
  if (from_pos >= 0 && from_pos < kNumPoints) {
     if (temp_board[player][from_pos] == 0) {
        SpielFatalError("WouldFormBlockingBridge: Trying to move from empty point.");
     }
     temp_board[player][from_pos]--;
  }
  if (to_pos >= 0 && to_pos < kNumPoints) {
    temp_board[player][to_pos]++;
  }

  int opponent = Opponent(player);
  bool opponent_exists_on_board = false;
  for (int i = 0; i < kNumPoints; ++i) {
    if (temp_board[opponent][i] > 0) {
        opponent_exists_on_board = true;
        break;
    }
  }

  if (!opponent_exists_on_board) {
      return false;
  }

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
      int block_path_start_on_opp_path_real_pos = GetBlockPathStartRealPos(opponent, start);

      bool opponent_found_ahead = false;
      for (int opp_pos = 0; opp_pos < kNumPoints; ++opp_pos) {
        if (temp_board[opponent][opp_pos] > 0) {
            int opp_real_pos = opp_pos;
            if (IsAhead(opponent, opp_real_pos, block_path_start_on_opp_path_real_pos)) {
                opponent_found_ahead = true;
                break;
            }
        }
      }

      if (!opponent_found_ahead) {
        return true;
      }
    }
  }

  return false;
}

bool LongNardeState::IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule) const {
  if (from_pos == kPassPos) return true;
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
  int calculated_target_pos = GetToPos(player, from_pos, die_value);
  if (to_pos != calculated_target_pos) {
      if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Mismatch to_pos=" << to_pos << " vs calculated=" << calculated_target_pos << std::endl;
      return false; // Provided to_pos doesn't match calculation
  }
  if (check_head_rule && !IsLegalHeadMove(player, from_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Head rule violation for pos " << from_pos << std::endl;
    return false;
  }
  bool is_bearing_off = IsOff(player, to_pos);
  if (is_bearing_off) {
    if (!AllInHome(player)) {
        if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Cannot bear off, not all checkers in home" << std::endl;
        return false;
    }

    // Calculate exact pips needed to bear off from from_pos
    int pips_needed = 0;
    int temp_pos = from_pos;
    while (temp_pos != kBearOffPos && pips_needed <= 6) {
        pips_needed++;
        // Simulate one step using GetToPos logic
        int next_temp_pos = GetToPos(player, temp_pos, 1); // Simulate moving 1 pip
        temp_pos = next_temp_pos;
    }
    // If loop finished and temp_pos is not kBearOffPos, something's wrong (>6 pips needed)
    if (temp_pos != kBearOffPos) pips_needed = 99;

    if (die_value == pips_needed) {
        // Exact roll is valid
    } else if (die_value > pips_needed) {
        // Higher roll only valid if no checkers are further back
        bool further_checker_exists = false;
        int current_path_idx = GetPathIndex(player, from_pos);
        for (int p = 0; p < kNumPoints; ++p) {
            if (p == from_pos) continue;
            if (board(player, p) > 0) {
                // Check if checker 'p' is further back on the path than 'from_pos'
                if (GetPathIndex(player, p) < current_path_idx) {
                    further_checker_exists = true;
                    break;
                }
            }
        }
        if (further_checker_exists) {
             if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Cannot bear off high, checker exists further back on path" << std::endl;
             return false; // Cannot use higher roll
        }
        // Valid high roll bear off
    } else {
        // Die roll too small
        if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid bearing off move (die too low)" << std::endl;
        return false;
    }
    // If we reach here, the bear off move is valid
    return true;
  }
  
  // Check bounds for non-bearing off moves
  if (to_pos < 0 || to_pos >= kNumPoints) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid to_pos " << to_pos << std::endl;
    return false;
  }

  // *** Critical Check: Opponent Occupancy ***
  // Only check opponent occupancy if this is a normal move (not bearing off)
  if (kDebugging) {
    std::cout << "DEBUG IsValidCheckerMove (O: 11->12): Checking opponent occupancy at to_pos=" << to_pos << std::endl;
    std::cout << "  board(X, " << to_pos << ") = " << board(1-player, to_pos) << std::endl;
    // Also print neighbors for context
    std::cout << "  Context: O[" << board(player, 11) << "," << board(player, 12) << "," << board(player, 13) << "] "
              << "X[" << board(1-player, 11) << "," << board(1-player, 12) << "," << board(1-player, 13) << "]" << std::endl;
  }
  
  // Only check opponent occupancy if position is on the board
  if (to_pos >= 0 && to_pos < kNumPoints && board(1-player, to_pos) > 0) {
    return false;
  }
  // *** End Critical Check ***

  // *** Critical Check: Blocking Bridge ***
  if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Would form illegal blocking bridge" << std::endl;
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

  // *** ADD THIS: Set is_first_turn_ based on the player whose turn it is now ***
  if (cur_player_ != kChancePlayerId && cur_player_ != kTerminalPlayerId) {
      is_first_turn_ = IsFirstTurn(cur_player_);
  } else {
      is_first_turn_ = false; // Not applicable for chance/terminal
  }
  // Reset moved_from_head_ for the new turn
  moved_from_head_ = false;


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

  // *** Note: is_first_turn_ is now correctly set by ProcessChanceRoll before this point ***
  // is_first_turn_ = IsFirstTurn(cur_player_); // Can remove this redundant check
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
      if (is_first_turn_) { // Check the member variable correctly set at turn start
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
  // is_first_turn_ = false; // REMOVE THIS LINE
  moved_from_head_ = false; // This is reset in ProcessChanceRoll
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

bool LongNardeState::AllInHome(Player player) const {
  if (player == kXPlayerId) {
    // White's home is points 1-6 (indices 0-5)
    // Check if any checkers are outside this range
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false;
      }
    }
  } else { // kOPlayerId
    // Black's home is points 13-18 (indices 12-17)
    // Check if any checkers are outside this range
    // Check indices 0-11 (first segment before home)
    for (int i = 0; i < kBlackHomeStart; ++i) {
      if (board(player, i) > 0) {
        return false;
      }
    }
    // Check indices 18-23 (second segment before home)
    for (int i = kBlackHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false;
      }
    }
  }
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
    bool is_debugging = false; // Keep general debugging off unless needed
    if (is_debugging) {
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
    }
    return false;
  }
  std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
  for (const auto& move : moves) {
    if (move.pos == kPassPos) continue;
    int to_pos = GetToPos(cur_player_, move.pos, move.die);
    if (!IsValidCheckerMove(cur_player_, move.pos, to_pos, move.die, true)) {
      bool is_debugging = false; // Keep general debugging off unless needed
      if (is_debugging) {
        std::cout << "ERROR: Decoded move from " << move.pos << " to " << to_pos
                  << " with die=" << move.die << " for player " << cur_player_
                  << " is not valid but action was in legal actions!" << std::endl;
        std::cout << "Dice: [" << (dice_.size() >= 1 ? std::to_string(dice_[0]) : "none")
                  << ", " << (dice_.size() >= 2 ? std::to_string(dice_[1]) : "none") << "]" << std::endl;
        std::cout << "Board state: " << ToString() << std::endl;
      }
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
  bool is_debugging = false; // Keep general debugging off unless needed
  bool specific_debug = (player == kOPlayerId && (move.pos == 11 || move.to_pos == 12 || move.pos == 13 || move.to_pos == 13));
  if (is_debugging && specific_debug) {
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

  if (is_debugging && specific_debug) {
      std::cout << "  Board state AFTER apply at [11, 12, 13]: O=["
                << board_[kOPlayerId][11] << "," << board_[kOPlayerId][12] << "," << board_[kOPlayerId][13] << "] X=["
                << board_[kXPlayerId][11] << "," << board_[kXPlayerId][12] << "," << board_[kXPlayerId][13] << "]" << std::endl;
  }
}

void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return;

  int next_pos = move.to_pos; // Use the pre-calculated to_pos from CheckerMove

  // Add specific debugging for the problematic move O: 11 -> 12 and its context
  bool is_debugging = false; // Keep general debugging off unless needed
  bool specific_debug = (player == kOPlayerId && (move.pos == 11 || next_pos == 12 || move.pos == 13 || next_pos == 13));
  if (is_debugging && specific_debug) {
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
      bool is_debugging = false; // Keep general debugging off unless needed
      if (is_debugging) {
         std::cerr << "Warning: UndoCheckerMove attempting to decrement invalid next_pos " << next_pos << std::endl;
      }
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
      bool is_debugging = false; // Keep general debugging off unless needed
      if (is_debugging) {
        std::cerr << "Warning: UndoCheckerMove attempting to increment invalid move.pos " << move.pos << std::endl;
      }
  }
  // Note: Undoing moved_from_head_ is handled by restoring the state in RecLegalMoves
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
  std::vector<CheckerMove> half_moves = GenerateAllHalfMoves(cur_player_);

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

    // Apply the move - GenerateAllHalfMoves should have already filtered based on head rule for *this* step.
    new_moveseq.push_back(move);
    ApplyCheckerMove(cur_player_, move); // This updates moved_from_head_ if move was from head

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
  return pos == kBearOffPos;
}

inline int CounterClockwisePos(int from, int pips, int num_points) {
  int pos = from - pips;
  pos = (pos % num_points + num_points) % num_points;
  return pos;
}

int LongNardeState::GetToPos(int player, int from_pos, int pips) const {
  if (from_pos == kPassPos) return kPassPos;
  SPIEL_CHECK_GE(pips, 1);
  SPIEL_CHECK_LE(pips, 6);

  int current_pos = from_pos;
  for (int i = 0; i < pips; ++i) {
    // Determine the next position based on player path
    int next_pos;
    if (player == kXPlayerId) {
      // White moves towards 0. Bear off if current_pos is 0 and moving further.
      if (current_pos == kWhiteHomeStart) { // At pos 0
        next_pos = kBearOffPos; // Next step is off
      } else {
        next_pos = current_pos - 1;
      }
    } else { // kOPlayerId
      // Black moves 11..0 then 23..12. Bear off if current_pos is 12 and moving further.
      if (current_pos == kBlackHomeStart) { // At pos 12
        next_pos = kBearOffPos; // Next step is off
      } else if (current_pos == 0) {
        next_pos = 23; // Wrap around
      } else {
        next_pos = current_pos - 1;
      }
    }

    // Check if we landed off the board
    if (next_pos == kBearOffPos) {
      // If this is the exact number of pips, return BearOffPos.
      // If this is *before* the last pip, the move still lands off.
      return kBearOffPos;
    }
    current_pos = next_pos; // Update position for next pip
  }

  // If the loop completes without returning kBearOffPos, current_pos is the final landing spot.
  SPIEL_CHECK_GE(current_pos, 0);
  SPIEL_CHECK_LT(current_pos, kNumPoints);
  return current_pos;
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

// ** NEW HELPER FUNCTION: GetDiceInfo **
LongNardeState::DiceInfo LongNardeState::GetDiceInfo() const {
  DiceInfo info{0, false, 0, 0};
  
  if (!dice_.empty()) {
    info.is_doubles = (dice_.size() >= 2 && DiceValue(0) == DiceValue(1));
    info.max_moves = info.is_doubles ? 4 : 2;
    
    // Get actual die values
    info.die1 = (dice_.size() >= 1 && UsableDiceOutcome(dice_[0])) ? DiceValue(0) : 1;
    info.die2 = (dice_.size() >= 2 && UsableDiceOutcome(dice_[1])) ? DiceValue(1) : info.die1;
    
    // Ensure valid die values (1-6)
    info.die1 = std::max(1, std::min(6, info.die1));
    info.die2 = std::max(1, std::min(6, info.die2));
  }
  
  return info;
}

// ** NEW HELPER FUNCTION: GeneratePassMove **
Action LongNardeState::GeneratePassMove() const {
  DiceInfo dice_info = GetDiceInfo();
  std::vector<CheckerMove> pass_move_seq;
  
  // Create pass moves with appropriate dice values
  pass_move_seq.push_back({kPassPos, kPassPos, dice_info.die1});
  pass_move_seq.push_back({kPassPos, kPassPos, dice_info.die2});
  
  // Ensure pass move sequence has correct length for encoding (non-doubles case)
  while (pass_move_seq.size() < 2 && dice_info.max_moves < 3) {
    pass_move_seq.push_back({kPassPos, kPassPos, dice_info.die2});
  }
  
  return CheckerMovesToSpielMove(pass_move_seq);
}

// ** NEW HELPER FUNCTION: ShouldGeneratePassMove **
bool LongNardeState::ShouldGeneratePassMove(const MoveSequenceInfo& info) const {
  return info.max_non_pass == 0;
}

// ** REFACTORED LegalActions FUNCTION **
std::vector<Action> LongNardeState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();

  // Clone state for move generation - needed for the recursive helper
  std::unique_ptr<State> cstate = this->Clone();
  LongNardeState* state = dynamic_cast<LongNardeState*>(cstate.get());
  // Ensure state pointer is valid
  if (!state) {
      SpielFatalError("LegalActions: Failed to cast cloned state to LongNardeState*");
      return {}; // Should not happen
  }


  // Get dice information
  DiceInfo dice_info = GetDiceInfo();

  // Generate and filter move sequences using the recursive helper
  // The function now returns the results directly.
  MoveSequenceInfo move_info = state->FindMaximalMoveSequencesRecursive({}, dice_info.max_moves);


  // *** ADD DEBUG PRINT HERE ***
  if (kDebugging) {
      std::cout << "[DEBUG LegalActions] After Recurse: max_non_pass=" << move_info.max_non_pass // Use value from struct
                << ", num_sequences=" << move_info.sequences.size() << std::endl;
      if (!move_info.sequences.empty()) {
          std::cout << "[DEBUG LegalActions] First sequence found: ";
          for(const auto& m : *move_info.sequences.begin()) std::cout << "(" << m.pos << "," << m.die << ") ";
          std::cout << std::endl;
      }
  }
  // *** END DEBUG PRINT ***


  // Handle pass moves: If no maximal sequences were found AND max_non_pass was truly 0.
  if (move_info.sequences.empty()) {
    // *** ADD DEBUG PRINT HERE ***
    if (kDebugging) {
        std::cout << "[DEBUG LegalActions] sequences empty, checking max_non_pass (" << move_info.max_non_pass << ") for Pass." << std::endl;
    }
    // *** END DEBUG PRINT ***
    // Only generate pass if max_non_pass was indeed 0.
    // If sequences is empty but max_non_pass > 0, it indicates an issue (e.g., limits hit), return empty list.
    if (move_info.max_non_pass == 0) {
       return {GeneratePassMove()};
    } else {
       if (kDebugging) {
            std::cerr << "[WARN LegalActions] sequences empty but max_non_pass=" << move_info.max_non_pass << ". Returning empty list." << std::endl;
       }
       return {};
    }
  }

  // Convert the filtered sequences (already maximal) into action IDs
  // The ConvertSequencesToActions function expects the MoveSequenceInfo struct
  std::vector<Action> legal_moves = ConvertSequencesToActions(move_info.sequences, move_info);


  // Apply the "play higher die" rule if applicable (only 1 non-pass move made)
  // Use the max_non_pass value from the returned struct
  if (move_info.max_non_pass == 1) {
    legal_moves = ApplyHigherDieRule(legal_moves);
  }

  return legal_moves;
}

// ** NEW RECURSIVE HELPER for Generating/Filtering Sequences **
// Change return type to MoveSequenceInfo
// Remove pointer parameters for results
LongNardeState::MoveSequenceInfo LongNardeState::FindMaximalMoveSequencesRecursive(
    const std::vector<CheckerMove>& current_sequence,
    int remaining_moves) {

  MoveSequenceInfo result_info; // Holds best results found *starting from* this node
  result_info.max_non_pass = 0;

  const size_t kGlobalSequenceLimit = 100; // Limit total sequences found

  Player player = CurrentPlayer();
  int approx_depth = GetDiceInfo().max_moves - remaining_moves; // Calculate depth for logging

  // Handle terminal state: If CurrentPlayer is terminal, this path ends here.
  // Evaluate the sequence that *led* to this terminal state.
  if (player == kTerminalPlayerId) {
      if (kDebugging) std::cout << "[DEBUG Recurse Exit] Depth=" << approx_depth << " - Terminal Player State Encountered" << std::endl;
      // Evaluate the 'current_sequence' that resulted in this terminal state.
      int current_non_pass_count = 0;
      for (const auto& m : current_sequence) {
          if (m.pos != kPassPos) current_non_pass_count++;
      }
      // Only record if it's a non-empty sequence.
      if (current_non_pass_count > 0) {
          result_info.max_non_pass = current_non_pass_count;
          result_info.sequences.insert(current_sequence);
           if (kDebugging) std::cout << "[DEBUG Recurse Eval@Terminal] Depth=" << approx_depth << " - Recorded SeqLen=" << current_sequence.size() << " NonPass=" << current_non_pass_count << std::endl;
      } else if (kDebugging) {
          std::cout << "[DEBUG Recurse Eval@Terminal] Depth=" << approx_depth << " - Empty sequence led to terminal? Ignoring." << std::endl;
      }
      // Return this result, as no further exploration is possible.
      return result_info;
  }
  // Check for other invalid player states (should not happen).
  if (player != kXPlayerId && player != kOPlayerId) {
      if (kDebugging) std::cout << "[DEBUG Recurse Exit] Depth=" << approx_depth << " - Invalid Player " << player << std::endl;
      return result_info; // Return empty result
  }


  // *** DEBUGGING START ***
  if (kDebugging) {
    std::cout << "[DEBUG Recurse Start] Depth=" << approx_depth
              << ", Player=" << player
              << ", RemMoves=" << remaining_moves
              << ", CurrentSeqLen=" << current_sequence.size() << std::endl;
  }
  // *** DEBUGGING END ***

  std::vector<CheckerMove> half_moves = GenerateAllHalfMoves(player);
  bool can_move_further = false;
  for(const auto& hm : half_moves) {
      // Check for *any* non-pass move possibility
      if (hm.pos != kPassPos) {
          can_move_further = true;
          break;
      }
  }
  // Add special case: if half_moves contains ONLY pass moves, can_move_further remains false.
  if (!can_move_further && !half_moves.empty()) {
      bool only_pass = true;
      for(const auto& hm : half_moves) {
          if (hm.pos != kPassPos) { only_pass = false; break; }
      }
      if (only_pass) can_move_further = false;
  }


  // *** DEBUGGING START ***
  if (kDebugging) {
      std::cout << "[DEBUG Recurse Info] Depth=" << approx_depth
                << ", GenHalfMoves=" << half_moves.size()
                << ", can_move_further=" << can_move_further << std::endl;
  }
  // *** DEBUGGING END ***

  // --- Recursive Step: Explore children ---
  // Explore only if moves are remaining AND further non-pass moves are possible from current state
  if (remaining_moves > 0 && can_move_further) {
    for (const auto& move : half_moves) {
      // Only branch on actual non-pass moves
      if (move.pos == kPassPos) continue;

      bool old_moved_from_head = moved_from_head_;
      ApplyMove(move); // Updates state including dice and moved_from_head_

      // Skip invalid intermediate states (illegal bridges)
      if (HasIllegalBridge(player)) {
          UndoCheckerMove(player, move); // Backtrack the single move
          moved_from_head_ = old_moved_from_head;
          continue;
      }

      // Recursive call
      std::vector<CheckerMove> next_sequence = current_sequence;
      next_sequence.push_back(move);
      if (kDebugging) std::cout << "[DEBUG Recurse Call] Depth=" << approx_depth << " -> Calling Depth=" << (approx_depth + 1) << " with SeqLen " << next_sequence.size() << std::endl;
      MoveSequenceInfo child_info = FindMaximalMoveSequencesRecursive(next_sequence, remaining_moves - 1);

      // Backtrack state *after* recursive call returns
      UndoCheckerMove(player, move);
      moved_from_head_ = old_moved_from_head;

      // --- Aggregation ---
      // Compare child result with the best result found *so far* from other children or deeper paths
       if (kDebugging) {
           std::cout << "[DEBUG Recurse Aggregation] Depth=" << approx_depth
                     << ", Received ChildMaxLen=" << child_info.max_non_pass
                     << " (num seqs=" << child_info.sequences.size() << ")"
                     << ", CurrentBestSoFar=" << result_info.max_non_pass << std::endl;
       }
      if (child_info.max_non_pass > result_info.max_non_pass) {
          if (kDebugging) std::cout << "[DEBUG Recurse Aggregation] Depth=" << approx_depth << " - Child is new best. Updating result." << std::endl;
          result_info = child_info; // Child branch found a longer sequence
      } else if (child_info.max_non_pass == result_info.max_non_pass && result_info.max_non_pass > 0) {
          // If child length matches current best, merge sequences
          if (!child_info.sequences.empty()) { // Ensure child actually found sequences
             if (kDebugging) std::cout << "[DEBUG Recurse Aggregation] Depth=" << approx_depth << " - Child matches best. Merging sequences." << std::endl;
             result_info.sequences.insert(child_info.sequences.begin(), child_info.sequences.end());
          } else if (kDebugging) {
             // This case might happen if a deeper path hit a limit or error.
             std::cout << "[DEBUG Recurse Aggregation] Depth=" << approx_depth << " - Child matches best length but has no sequences. Ignoring merge." << std::endl;
          }
      } else {
           if (kDebugging) std::cout << "[DEBUG Recurse Aggregation] Depth=" << approx_depth << " - Child is not better (or both 0). Ignoring." << std::endl;
      }
       if (kDebugging) {
           std::cout << "[DEBUG Recurse Aggregation] Depth=" << approx_depth
                     << ", After Aggregation BestSoFar=" << result_info.max_non_pass
                     << ", NumSeqs=" << result_info.sequences.size() << std::endl;
       }

      // Optional: Limit sequence count
      if (result_info.sequences.size() > kGlobalSequenceLimit) {
           if (kDebugging) std::cout << "[DEBUG Recurse Limit] Depth=" << approx_depth << " - Sequence limit reached. Stopping child exploration." << std::endl;
           break; // Stop exploring more children at this level if limit hit
      }
    }
  }

  // --- Evaluation Step (Post-Loop / Path Termination) ---
  // Evaluate the 'current_sequence' IF this path terminates here.
  // Termination occurs if no further non-pass moves were possible OR we ran out of moves.
  bool path_terminates_here = (!can_move_further || remaining_moves == 0);

  if (path_terminates_here) {
      int current_non_pass_count = 0;
      for (const auto& m : current_sequence) {
          if (m.pos != kPassPos) {
              current_non_pass_count++;
          }
      }

      if (kDebugging) {
           std::cout << "[DEBUG Recurse Eval (Post-Loop)] Depth=" << approx_depth
                     << ", Path Terminates Here. SeqLen=" << current_sequence.size()
                     << ", SeqNonPass=" << current_non_pass_count
                     << ", BestFromSubtree=" << result_info.max_non_pass << std::endl;
      }

      // Compare this sequence's length with the best found in any subtree (result_info)
      if (current_non_pass_count > result_info.max_non_pass) {
          // This sequence ending here is longer than any sequence found deeper.
          if (kDebugging) std::cout << "[DEBUG Recurse Eval (Post-Loop)] Depth=" << approx_depth << " - Current sequence is NEW BEST." << std::endl;
          result_info.max_non_pass = current_non_pass_count;
          result_info.sequences.clear(); // Discard shorter results from subtrees
          // Add the current sequence only if it's non-empty
          if (!current_sequence.empty()) {
             result_info.sequences.insert(current_sequence);
          }
      } else if (current_non_pass_count == result_info.max_non_pass) {
          // This sequence matches the best length found below OR is the best if no subtrees existed/found anything.
          // Add it only if it's a non-empty sequence (max_non_pass > 0 implies current_non_pass > 0 here)
          if (current_non_pass_count > 0) { // Equivalent to result_info.max_non_pass > 0
             if (kDebugging) std::cout << "[DEBUG Recurse Eval (Post-Loop)] Depth=" << approx_depth << " - Current sequence MATCHES BEST length." << std::endl;
             result_info.sequences.insert(current_sequence); // Add it to the set
          } else {
              // Case: current_non_pass_count is 0, and best from below was also 0.
              // result_info remains {max_non_pass=0, sequences=empty}.
              if (kDebugging) std::cout << "[DEBUG Recurse Eval (Post-Loop)] Depth=" << approx_depth << " - Current sequence is empty, matches best (0). No change." << std::endl;
          }

      } else {
           // This sequence is shorter than the best found deeper. Ignore it.
           if (kDebugging) std::cout << "[DEBUG Recurse Eval (Post-Loop)] Depth=" << approx_depth << " - Current sequence is shorter than best from subtree. Ignoring." << std::endl;
      }
       if (kDebugging) {
           std::cout << "[DEBUG Recurse Eval (Post-Loop)] Depth=" << approx_depth
                     << ", After Eval FinalBest=" << result_info.max_non_pass
                     << ", NumSeqs=" << result_info.sequences.size() << std::endl;
       }
  }


  // *** DEBUGGING START ***
   if (kDebugging) {
       std::cout << "[DEBUG Recurse Return] Depth=" << approx_depth
                 << ", Returning MaxLen=" << result_info.max_non_pass
                 << ", NumSeqs=" << result_info.sequences.size() << std::endl;
        if (!result_info.sequences.empty() && kDebugging) {
            std::cout << "[DEBUG Recurse Return] Depth=" << approx_depth << " First Seq Returned: ";
             for(const auto& m : *result_info.sequences.begin()) std::cout << "(" << m.pos << "," << m.die << ") ";
             std::cout << std::endl;
        } else if (kDebugging) {
             std::cout << "[DEBUG Recurse Return] Depth=" << approx_depth << " Returning No Sequences." << std::endl;
        }
   }
  // *** DEBUGGING END ***

  return result_info;
}

// ** REFACTORED HELPER FUNCTION: ConvertSequencesToActions **
// Match the header declaration signature (takes two arguments)
std::vector<Action> LongNardeState::ConvertSequencesToActions(
    const std::set<std::vector<CheckerMove>>& sequences,
    const MoveSequenceInfo& info) const { // Add MoveSequenceInfo info back
  std::vector<Action> actions;
  actions.reserve(sequences.size());

  for (const auto& sequence : sequences) {
    // The recursive function already ensures these sequences match max_non_pass
    // So no filtering is needed here based on info.max_non_pass.
    // Just convert the sequence to an action.
    actions.push_back(CheckerMovesToSpielMove(sequence));
  }

  // Optional: Limit the number of actions for performance if needed
  // const size_t kMaxActionsToReturn = 20;
  // if (actions.size() > kMaxActionsToReturn) {
  //   actions.resize(kMaxActionsToReturn);
  // }

  return actions;
}

// ** REFACTORED HELPER FUNCTION: ApplyHigherDieRule **
std::vector<Action> LongNardeState::ApplyHigherDieRule(
    const std::vector<Action>& candidate_actions) const {
  std::vector<Action> filtered_actions;
  DiceInfo dice_info = GetDiceInfo();

  // If dice are equal or there's only one die, no filtering needed
  if (dice_info.die1 == dice_info.die2 || dice_.size() < 2) {
    return candidate_actions;
  }

  // Get the higher die value
  int higher_die = std::max(dice_info.die1, dice_info.die2);

  // Filter actions to only include those using the higher die when possible
  bool higher_die_used_in_any_action = false;
  for (const auto& action : candidate_actions) {
    // Need to pass player to SpielMoveToCheckerMoves
    std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(CurrentPlayer(), action);
    bool uses_higher_die = false;

    for (const auto& move : moves) {
      // Use move.pos and move.die
      if (move.pos != kPassPos && move.die == higher_die) {
        uses_higher_die = true;
        break;
      }
    }

    if (uses_higher_die) {
      filtered_actions.push_back(action);
      higher_die_used_in_any_action = true;
    }
  }

  // If the higher die *could* be used (at least one action used it), return only those actions.
  // Otherwise (if no action could use the higher die), return all original actions (meaning the lower die must be played).
  return higher_die_used_in_any_action ? filtered_actions : candidate_actions;
}

// *** NEW HELPER FUNCTION: HasIllegalBridge ***
// Checks the current board state for an illegal bridge for the given player.
bool LongNardeState::HasIllegalBridge(int player) const {
  int opponent = Opponent(player);
  bool opponent_exists_on_board = false;
  for (int i = 0; i < kNumPoints; ++i) {
    if (board(opponent, i) > 0) {
        opponent_exists_on_board = true;
        break;
    }
  }

  if (!opponent_exists_on_board) {
      return false;
  }

  bool is_debugging = false; // Keep general debugging off unless needed
  if (is_debugging) {
    std::cout << "DEBUG: HasIllegalBridge check for player " << player << "\n";
    std::cout << "DEBUG: Opponent (" << opponent << ") board state (real pos -> count (path idx)):\n";
    std::cout << "  [";
    bool first = true;
    for (int i = 0; i < kNumPoints; ++i) {
      if (board(opponent, i) > 0) {
          if (!first) std::cout << ", ";
          std::cout << i << "->" << board(opponent, i) << "(" << GetPathIndex(opponent, i) << ")";
          first = false;
      }
    }
    std::cout << "]\n";
  }

  for (int start = 0; start < kNumPoints; ++start) {
    bool is_block = true;
    for (int i = 0; i < 6; ++i) {
      int pos = (start + i) % kNumPoints;
      if (board(player, pos) == 0) {
        is_block = false;
        break;
      }
    }

    if (is_block) {
      if (is_debugging) {
        std::cout << "DEBUG: Found potential block starting at real pos " << start << "\n";
      }
      
      int block_path_start_on_opp_path_real_pos = GetBlockPathStartRealPos(opponent, start);
      
      if (is_debugging) {
        std::cout << "DEBUG: Block path start on opponent's path (real pos): " 
                  << block_path_start_on_opp_path_real_pos << "\n";
        std::cout << "DEBUG: Block path start index on opponent's path: " << GetPathIndex(opponent, block_path_start_on_opp_path_real_pos) << "\n";
      }

      bool opponent_found_ahead = false;
      for (int opp_pos = 0; opp_pos < kNumPoints; ++opp_pos) {
        if (board(opponent, opp_pos) > 0) {
            int opp_real_pos = opp_pos;
            if (IsAhead(opponent, opp_real_pos, block_path_start_on_opp_path_real_pos)) {
                if (is_debugging) {
                  std::cout << "DEBUG: Found opponent checker ahead at real pos " << opp_real_pos 
                            << " (path index " << GetPathIndex(opponent, opp_real_pos) << ")\n";
                }
                opponent_found_ahead = true;
                break;
            }
        }
      }

      if (!opponent_found_ahead) {
        if (is_debugging) {
          std::cout << "DEBUG: ILLEGAL BRIDGE DETECTED! No opponent checkers found ahead of block path start\n";
        }
        return true;
      }
    }
  }

  if (is_debugging) {
    std::cout << "DEBUG: No illegal bridges found\n";
  }
  return false;
}

// Implement the GenerateAllHalfMoves method
// std::set<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player) const { // Old signature
std::vector<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player) const { // New signature
  // std::set<CheckerMove> half_moves; // Old type
  std::vector<CheckerMove> half_moves; // New type

  // *** DEBUGGING START ***
  if (kDebugging) {
    std::cout << "[DEBUG GenHalfMoves] Player=" << player << std::endl;
    std::cout << "[DEBUG GenHalfMoves] Dice: ";
    for(int d : dice_) std::cout << d << " ";
    std::cout << std::endl;
    std::cout << "[DEBUG GenHalfMoves] Board: " << std::endl << ToString() << std::endl;
    std::cout << "[DEBUG GenHalfMoves] AllInHome? " << (AllInHome(player) ? "YES" : "NO") << std::endl;
  }
  // *** DEBUGGING END ***

  // For each checker belonging to the player
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (board(player, pos) <= 0) continue;

    // *** DEBUGGING START ***
    if (kDebugging) std::cout << "[DEBUG GenHalfMoves] Checking checker at pos=" << pos << std::endl;
    // *** DEBUGGING END ***

    // For each usable die
    for (int i = 0; i < dice_.size(); ++i) {
      int outcome = dice_[i];
      if (!UsableDiceOutcome(outcome)) {
        continue; // Skip used dice
      }

      int die_value = outcome; // Since UsableDiceOutcome passed, outcome is 1-6
      int to_pos = GetToPos(player, pos, die_value);

      // *** DEBUGGING START ***
      if (kDebugging) std::cout << "[DEBUG GenHalfMoves]  - Checking die=" << die_value << " -> to_pos=" << to_pos << std::endl;
      // *** DEBUGGING END ***

      // Check if this would be a valid move (including bearing off)
      // Use IsValidMove as it's intended for sequence generation context
      CheckerMove potential_move(pos, to_pos, die_value);
      bool is_valid = IsValidMove(potential_move);

      if (is_valid) {
        // *** DEBUGGING START ***
        if (kDebugging) std::cout << "[DEBUG GenHalfMoves]    -> VALID" << std::endl;
        // *** DEBUGGING END ***
        // half_moves.insert(potential_move); // Old insert
        half_moves.push_back(potential_move); // New push_back
      } else {
        // *** DEBUGGING START ***
        if (kDebugging) {
            std::cout << "[DEBUG GenHalfMoves]    -> INVALID (Reason checking below...)" << std::endl;
            // Add reason logging copied from IsValidCheckerMove debugging
            if (board(Opponent(player), to_pos) > 0 && to_pos >= 0 && to_pos < kNumPoints) {
                 std::cout << "[DEBUG GenHalfMoves]       Reason: Opponent block at " << to_pos << std::endl;
            } else if (IsOff(player, to_pos) && !AllInHome(player)) {
                 std::cout << "[DEBUG GenHalfMoves]       Reason: Cannot bear off, not all home" << std::endl;
            } else if (IsOff(player, to_pos)) {
                 // Re-calculate pips needed for logging
                int pips_needed_log = 0;
                int temp_pos_log = pos;
                while (temp_pos_log != kBearOffPos && pips_needed_log <= 6) {
                    pips_needed_log++;
                    temp_pos_log = GetToPos(player, temp_pos_log, 1);
                }
                if (temp_pos_log != kBearOffPos) pips_needed_log = 99;

                if (die_value < pips_needed_log) {
                     std::cout << "[DEBUG GenHalfMoves]       Reason: Bear off die too small (need " << pips_needed_log << ")" << std::endl;
                } else if (die_value > pips_needed_log) {
                    bool further_exists_log = false;
                    int current_path_idx_log = GetPathIndex(player, pos);
                    for(int p_log=0; p_log<kNumPoints; ++p_log) {
                        if (p_log == pos) continue;
                        if (board(player, p_log) > 0 && GetPathIndex(player, p_log) < current_path_idx_log) {
                            further_exists_log = true; break;
                        }
                    }
                    if (further_exists_log) {
                        std::cout << "[DEBUG GenHalfMoves]       Reason: Bear off die too high & further checker exists" << std::endl;
                    } else {
                         std::cout << "[DEBUG GenHalfMoves]       Reason: Bear off die too high (but maybe valid? Check IsValidMove logic)" << std::endl;
                    }
                }
            } else if (!IsLegalHeadMove(player, pos)) {
                 std::cout << "[DEBUG GenHalfMoves]       Reason: Head rule violation" << std::endl;
            } else if (WouldFormBlockingBridge(player, pos, to_pos)) {
                 std::cout << "[DEBUG GenHalfMoves]       Reason: Would form bridge" << std::endl;
            } else if (pos < 0 || pos >= kNumPoints || board(player, pos) == 0) {
                 std::cout << "[DEBUG GenHalfMoves]       Reason: Invalid source position/checker" << std::endl;
            } else {
                 std::cout << "[DEBUG GenHalfMoves]       Reason: Other (check IsValidMove logic)" << std::endl;
            }
        }
         // *** DEBUGGING END ***
      }
    }
  }

  // If no valid moves, add a pass move for each usable die
  if (half_moves.empty()) {
    // *** DEBUGGING START ***
    if (kDebugging) std::cout << "[DEBUG GenHalfMoves] No non-pass half moves found. Adding pass moves." << std::endl;
    // *** DEBUGGING END ***
    for (int i = 0; i < dice_.size(); ++i) {
      int outcome = dice_[i];
      if (UsableDiceOutcome(outcome)) {
        // Use kPassDieValue (or a default like 1) for pass move die encoding?
        // Let's use the actual die value for consistency with encoding/decoding pass moves.
        // half_moves.insert(CheckerMove(kPassPos, kPassPos, outcome)); // Old insert
        half_moves.push_back(CheckerMove(kPassPos, kPassPos, outcome)); // New push_back
         // *** DEBUGGING START ***
         if (kDebugging) std::cout << "[DEBUG GenHalfMoves]  - Added Pass move with die " << outcome << std::endl;
         // *** DEBUGGING END ***
      }
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

int LongNardeState::GetVirtualCoords(int player, int real_pos) const {
  if (real_pos < 0 || real_pos >= kNumPoints) {
    SpielFatalError(absl::StrCat("GetVirtualCoords called with invalid real_pos: ", real_pos));
    return -1;
  }

  if (player == kXPlayerId) {
    return real_pos;
  } else { // kOPlayerId
    if (real_pos >= 0 && real_pos <= 11) { // Segment 1
      return real_pos + 12;
    } else { // Segment 2
      return real_pos - 12;
    }
  }
}

int LongNardeState::GetPathIndex(int player, int real_pos) const {
    if (real_pos < 0 || real_pos >= kNumPoints) {
         SpielFatalError(absl::StrCat("GetPathIndex called with invalid real_pos: ", real_pos));
         return -1;
    }

    if (player == kXPlayerId) {
        return 23 - real_pos;
    } else {
        if (real_pos >= 0 && real_pos <= 11) {
            return 11 - real_pos;
        } else {
            return 12 + (23 - real_pos);
        }
    }
}

bool LongNardeState::IsAhead(int player, int checker_pos, int reference_pos) const {
    // Ensure positions are potentially valid before conversion.
    if (checker_pos < 0 || reference_pos < 0 || checker_pos >= kNumPoints || reference_pos >= kNumPoints) {
        return false;
    }

    // Get the virtual coordinate representation for the player.
    int vcoord_checker = GetVirtualCoords(player, checker_pos);
    int vcoord_ref = GetVirtualCoords(player, reference_pos);

    // On both real (White) and virtual (Black) paths, moving "forward"
    // means decreasing the coordinate value (towards 0).
    // Therefore, a checker is "ahead" if its virtual coordinate is less than the reference coordinate.
    return vcoord_checker < vcoord_ref;
}

int LongNardeState::GetBlockPathStartRealPos(int player_for_path, int block_lowest_real_idx) const {
    if (block_lowest_real_idx < 0 || block_lowest_real_idx >= kNumPoints) {
         SpielFatalError(absl::StrCat("GetBlockPathStartRealPos called with invalid block_lowest_real_idx: ", block_lowest_real_idx));
         return -1;
    }

    int bridge_path_start_pos = block_lowest_real_idx;
    int min_path_idx = GetPathIndex(player_for_path, block_lowest_real_idx);

    for (int i = 1; i < 6; ++i) {
        int current_pos = (block_lowest_real_idx + i) % kNumPoints;
         if (current_pos < 0 || current_pos >= kNumPoints) {
            SpielFatalError(absl::StrCat("GetBlockPathStartRealPos calculated invalid current_pos: ", current_pos));
            continue;
         }
        int current_path_idx = GetPathIndex(player_for_path, current_pos);
        if (current_path_idx < min_path_idx) {
            min_path_idx = current_path_idx;
            bridge_path_start_pos = current_pos;
        }
    }
    return bridge_path_start_pos;
}

// ** NEW HELPER FUNCTION: IsValidMove **
bool LongNardeState::IsValidMove(const CheckerMove& move) const {
  // Pass moves are always valid in the context of sequence building
  if (move.pos == kPassPos && move.to_pos == kPassPos) {
    return true;
  }

  // Use the CurrentPlayer for checks
  Player player = CurrentPlayer();
  // Ensure player is valid (could be terminal during sequence check) - if so, moves are invalid.
  if (player != kXPlayerId && player != kOPlayerId) return false;

  // Check if move uses an *available* die (This might need refinement based on how dice are tracked in sequence search)
  // Basic check: die value must match one of the original dice roll values.
   bool die_exists_and_usable = false; // Check if the die exists AND is currently usable (1-6)
   for (int i = 0; i < dice_.size(); ++i) {
       if (dice_[i] == move.die) { // Check if an *unused* die (1-6) matches
           die_exists_and_usable = true;
           break;
       }
   }
   // This check is crucial: GenerateAllHalfMoves iterates through usable dice,
   // so this check should pass if called from there.
   // If called elsewhere, it ensures the move corresponds to a currently available die.
   if (!die_exists_and_usable) {
        if (kDebugging) std::cout << "DEBUG IsValidMove: Die " << move.die << " not available/usable in current dice_ state." << std::endl;
        return false;
   }


  // Check if calculated target position matches the move's to_pos
  int calculated_target_pos = GetToPos(player, move.pos, move.die);
   // Allow mismatch only if move.to_pos wasn't set correctly initially (legacy?)
   // For robustness, let's assume move.to_pos should match GetToPos result.
  if (move.to_pos != calculated_target_pos) {
       if (kDebugging) std::cout << "DEBUG IsValidMove: Mismatch move.to_pos=" << move.to_pos << " vs calculated=" << calculated_target_pos << std::endl;
       // Allow legacy calls where to_pos might be -1 initially? No, enforce consistency.
       return false;
  }

  // Check if source position has player's checker
  if (move.pos < 0 || move.pos >= kNumPoints || board(player, move.pos) == 0) {
    if (kDebugging) std::cout << "DEBUG IsValidMove: Invalid source pos " << move.pos << " or no checker." << std::endl;
    return false;
  }

  // Check if destination is blocked by opponent (only if not bearing off)
  // Use calculated_target_pos as the reliable destination
  if (!IsOff(player, calculated_target_pos)) {
      // Check bounds before accessing opponent board state
      if (calculated_target_pos >= 0 && calculated_target_pos < kNumPoints && board(Opponent(player), calculated_target_pos) > 0) {
        if (kDebugging) std::cout << "DEBUG IsValidMove: Opponent block at dest " << calculated_target_pos << std::endl;
        return false;
      }
  }


  // Check bearing off rules if applicable
  if (IsOff(player, calculated_target_pos)) {
       if (!AllInHome(player)) {
           if (kDebugging) std::cout << "DEBUG IsValidMove: Cannot bear off, not all home." << std::endl;
           return false;
       }

       // Calculate exact pips needed to bear off from move.pos
        int pips_needed = 0;
        int temp_pos = move.pos;
        while (temp_pos != kBearOffPos && pips_needed <= 6) {
            pips_needed++;
            int next_temp_pos = GetToPos(player, temp_pos, 1); // Simulate moving 1 pip
            temp_pos = next_temp_pos;
        }
        if (temp_pos != kBearOffPos) pips_needed = 99;


       if (move.die == pips_needed) {
           // Exact roll is always valid if all in home
       } else if (move.die > pips_needed) {
           // Higher roll only valid if no checkers are further back
            bool further_checker_exists = false;
            int current_path_idx = GetPathIndex(player, move.pos);
            for (int p = 0; p < kNumPoints; ++p) {
                if (p == move.pos) continue;
                if (board(player, p) > 0) {
                    if (GetPathIndex(player, p) < current_path_idx) {
                        further_checker_exists = true;
                        break;
                    }
                }
            }
           if (further_checker_exists) {
               if (kDebugging) std::cout << "DEBUG IsValidMove: Cannot bear off high, further checker exists." << std::endl;
               return false; // Cannot use higher roll
           }
       } else {
           if (kDebugging) std::cout << "DEBUG IsValidMove: Bear off die too small." << std::endl;
           return false; // Die roll too small
       }
       // If we reach here, bearing off is valid
  }


  // Check head rule (delegated from original IsValidCheckerMove - relies on moved_from_head_ state)
  // IsLegalHeadMove takes the 'from_pos'
  // *** This call now uses the updated IsLegalHeadMove logic ***
  if (!IsLegalHeadMove(player, move.pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidMove: Head rule violation." << std::endl;
    return false;
  }

  // Check for illegal bridge formation (delegated - uses calculated_target_pos)
  // Ensure calculated_target_pos is used here.
  if (WouldFormBlockingBridge(player, move.pos, calculated_target_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidMove: Would form illegal bridge." << std::endl;
    return false;
  }

  // All checks passed
  return true;
}

// ** NEW HELPER FUNCTION: ApplyMove **
// Use move.pos and move.to_pos
void LongNardeState::ApplyMove(const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Pass moves don't change state

  // Get player ID - Use the member variable cur_player_
  // as this function is called within the context of the current player's turn exploration.
  Player player = cur_player_;
  // Basic sanity check: ensure player is valid before proceeding
  if (player != kXPlayerId && player != kOPlayerId) {
      // This might happen if ApplyMove is somehow called in a terminal state search?
      // Should be prevented by checks in FindMaximalMoveSequencesRecursive.
      SpielFatalError("ApplyMove called with invalid player state.");
      return;
  }


  // Ensure source has a checker (basic sanity check)
  SPIEL_CHECK_GT(board(player, move.pos), 0); // Use player

  // Mark die used *before* modifying board, find the *first* available die matching value
  bool die_marked = false;
  for (int i = 0; i < dice_.size(); ++i) {
    if (dice_[i] == move.die) { // Find an unused die (1-6) with the correct value
      dice_[i] += 6; // Mark as used (7-12)
      die_marked = true;
      break;
    }
  }
  // This should always find a die if IsValidMove/IsValidCheckerMove checks were correct
  // and GenerateAllHalfMoves only provides moves for available dice.
  SPIEL_CHECK_TRUE(die_marked);


  // Remove checker from source
  board_[player][move.pos]--; // Use player

  // Add checker to destination or score
  if (IsOff(player, move.to_pos)) { // Use player
    scores_[player]++; // Use player
  } else {
    // Ensure destination is valid before incrementing
    SPIEL_CHECK_GE(move.to_pos, 0);
    SPIEL_CHECK_LT(move.to_pos, kNumPoints);
    board_[player][move.to_pos]++; // Use player
  }

  // Update head move status
  if (IsHeadPos(player, move.pos)) { // Use player
    moved_from_head_ = true;
  }
}

}  // namespace long_narde
}  // namespace open_spiel