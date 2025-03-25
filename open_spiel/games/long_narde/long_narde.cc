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
  SPIEL_CHECK_LE(moves.size(), 2);

  // Determine dice values using our helper.
  int high_roll = DiceValue(0) >= DiceValue(1) ? DiceValue(0) : DiceValue(1);
  int low_roll = DiceValue(0) < DiceValue(1) ? DiceValue(0) : DiceValue(1);
  bool high_roll_first = true;  // Our convention.

  // Order moves so that the move using the higher die comes first.
  std::vector<CheckerMove> ordered_moves = moves;
  if (ordered_moves.size() >= 2) {
    if (ordered_moves[0].die < ordered_moves[1].die) {
      std::swap(ordered_moves[0], ordered_moves[1]);
    }
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

  int dig0 = ordered_moves.empty() ? kPassOffset : encode_move(ordered_moves[0]);
  int dig1 = (ordered_moves.size() > 1) ? encode_move(ordered_moves[1]) : kPassOffset;

  Action action = dig1 * kDigitBase + dig0;
  if (!high_roll_first) {
    action += kDigitBase * kDigitBase;
  }
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, NumDistinctActions());
  return action;
}

std::vector<CheckerMove> LongNardeState::SpielMoveToCheckerMoves(
    Player player, Action spiel_move) const {
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

// FIX: Added declaration for NumDistinctActions() to match header.
int LongNardeState::NumDistinctActions() const {
  // Total distinct actions = 2 * (kDigitBase * kDigitBase)
  return 2 * kDigitBase * kDigitBase;
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
  dice_ = dice;
  scores_ = scores;
  board_ = board;
  // Reset turn counter to simulate a fresh start.
  turns_ = -1;
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
  SPIEL_CHECK_GE(pos, 0);
  SPIEL_CHECK_LT(pos, kNumPoints);
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
  bool is_debugging = false;
  std::vector<std::vector<int>> temp_board = board_;
  if (from_pos >= 0 && from_pos < kNumPoints) {
    temp_board[player][from_pos]--;
  }
  if (to_pos >= 0 && to_pos < kNumPoints) {
    temp_board[player][to_pos]++;
  } else {
    return false;
  }
  int consecutive_count = 0;
  int start_pos, end_pos, direction;
  if (player == kXPlayerId) {
    start_pos = 0;
    end_pos = kNumPoints;
    direction = 1;
  } else {
    start_pos = kNumPoints - 1;
    end_pos = -1;
    direction = -1;
  }
  for (int pos = start_pos; pos != end_pos; pos += direction) {
    if (temp_board[player][pos] > 0) {
      consecutive_count++;
    } else {
      consecutive_count = 0;
      continue;
    }
    if (consecutive_count == 6) {
      bool any_opponent_ahead = false;
      int opponent = Opponent(player);
      if (player == kXPlayerId) {
        int bridge_end = pos;
        for (int i = bridge_end + 1; i < kNumPoints; i++) {
          if (board(opponent, i) > 0) {
            any_opponent_ahead = true;
            break;
          }
        }
      } else {
        int bridge_end = pos;
        for (int i = 0; i < bridge_end; i++) {
          if (board(opponent, i) > 0) {
            any_opponent_ahead = true;
            break;
          }
        }
      }
      if (!any_opponent_ahead) {
        return true;
      }
    }
  }
  return false;
}

bool LongNardeState::ValidateAction(Action action) const {
  if (action < 0 || action >= NumDistinctActions()) {
    return false;
  }
  bool is_debugging = false;
  const auto& legal_actions = LegalActions();
  if (std::find(legal_actions.begin(), legal_actions.end(), action) == legal_actions.end()) {
    if (is_debugging) {
      std::cout << "DEBUG: Action " << action << " not in legal actions list of size " << legal_actions.size() << std::endl;
      for (auto a : legal_actions) {
        std::cout << "  " << a << std::endl;
      }
    }
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

// Process a chance roll (dice roll) action.
void LongNardeState::ProcessChanceRoll(Action move_id) {
  SPIEL_CHECK_GE(move_id, 0);
  SPIEL_CHECK_LT(move_id, game_->MaxChanceOutcomes());

  // Record the chance outcome in turn history.
  turn_history_info_.push_back(
      TurnHistoryInfo(kChancePlayerId, prev_player_, dice_,
                      move_id, double_turn_, is_first_turn_, moved_from_head_));

  // Ensure we have no dice set yet, then apply this new roll.
  SPIEL_CHECK_TRUE(dice_.empty());
  RollDice(move_id);

  // Decide which player starts or continues.
  if (turns_ < 0) {
    // White always starts, ignore dice outcomes
    turns_ = 0;
    cur_player_ = prev_player_ = kXPlayerId;
    return;  // Skip all other chance logic for the first turn
  } else if (double_turn_) {
    // Extra turn in progress (from doubles).
    cur_player_ = prev_player_;
    double_turn_ = false;  // Reset once new dice are rolled.
  } else {
    // Normal turn progression: pass to the opponent.
    cur_player_ = Opponent(prev_player_);
  }

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

  if (!ValidateAction(move_id)) {
    SpielFatalError(absl::StrCat("Invalid action: ", move_id));
  }
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
    int to_pos = GetToPos(cur_player_, m.pos, m.die);
    if (!IsValidCheckerMove(cur_player_, m.pos, to_pos, m.die, false)) {
      std::cout << "DEBUG: Invalid move from " << m.pos << " to " << to_pos
                << " with die=" << m.die << " for player " << cur_player_ << std::endl;
      filtered_moves.push_back(kPassMove);
      continue;
    }
    if (!is_first_turn_ && IsHeadPos(cur_player_, m.pos) && used_head_move) {
      // Allow a second head move if the dice roll is doubles.
      if (!(original_moves.size() == 2 && original_moves[0].die == original_moves[1].die)) {
        filtered_moves.push_back(kPassMove);
        continue;
      }
    }
    if (IsHeadPos(cur_player_, m.pos)) {
      used_head_move = true;
      moved_from_head_ = true;
    }
    filtered_moves.push_back(m);
  }
  for (const auto& m : filtered_moves) {
    if (m.pos != kPassPos) {
      ApplyCheckerMove(cur_player_, m);
    }
  }
  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, move_id, double_turn_,
                      is_first_turn_, moved_from_head_));
  bool extra_turn = false;
  if (!double_turn_ && dice_.size() == 2 && dice_[0] == dice_[1]) {
    int dice_used = 0;
    for (int i = 0; i < 2; i++) {
      if (dice_[i] > 6) {
        dice_[i] -= 6;
        dice_used++;
      }
      SPIEL_CHECK_GE(dice_[i], 1);
      SPIEL_CHECK_LE(dice_[i], 6);
    }
    if (dice_used == 2) {
      extra_turn = true;
    }
  }
  // Always update turn progression after a move.
  if (extra_turn) {
    // Mark that an extra (double) turn is in progress.
    double_turn_ = true;
  } else {
    // Count the completed turn.
    turns_++;
    if (cur_player_ == kXPlayerId) {
      x_turns_++;
    } else if (cur_player_ == kOPlayerId) {
      o_turns_++;
    }
  }
  prev_player_ = cur_player_;
  dice_.clear();
  // Always move to chance node so a new dice roll is generated.
  if (IsTerminal()) {
    cur_player_ = kTerminalPlayerId;
  } else {
    cur_player_ = kChancePlayerId;
  }
  // For non–extra-turn moves, ensure the double_turn_ flag is reset.
  if (!extra_turn) {
    double_turn_ = false;
  }
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
    UndoCheckerMove(player, moves[1]);
    UndoCheckerMove(player, moves[0]);
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
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false;
      }
    }
  } else {
    for (int i = 0; i < kBlackHomeStart; ++i) {
      if (board(player, i) > 0) {
        return false;
      }
    }
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

bool LongNardeState::IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule) const {
  bool is_debugging = false;
  if (from_pos == kPassPos) return true;
  if (from_pos < 0 || from_pos >= kNumPoints) {
    if (is_debugging) std::cout << "DEBUG: Invalid from_pos " << from_pos << std::endl;
    return false;
  }
  if (board(player, from_pos) <= 0) {
    if (is_debugging) std::cout << "DEBUG: No checker at from_pos " << from_pos << std::endl;
    return false;
  }
  if (die_value < 1 || die_value > 6) {
    if (is_debugging) std::cout << "DEBUG: Invalid die_value " << die_value << std::endl;
    return false;
  }
  int expected_to_pos = GetToPos(player, from_pos, die_value);
  if (to_pos != expected_to_pos) {
    if (is_debugging) std::cout << "DEBUG: to_pos " << to_pos << " doesn't match expected " << expected_to_pos << std::endl;
    return false;
  }
  if (check_head_rule && !IsLegalHeadMove(player, from_pos)) {
    if (is_debugging) std::cout << "DEBUG: Head rule violation for pos " << from_pos << std::endl;
    return false;
  }
  bool is_bearing_off = IsOff(player, to_pos);
  if (is_bearing_off) {
    if (!AllInHome(player)) {
      if (is_debugging) std::cout << "DEBUG: Cannot bear off, not all checkers in home" << std::endl;
      return false;
    }
    int furthest = FurthestCheckerInHome(player);
    bool is_exact_roll = false;
    if (player == kXPlayerId) {
      is_exact_roll = (from_pos + 1 == die_value);
    } else {
      is_exact_roll = (kNumPoints - from_pos == die_value);
    }
    if (from_pos == furthest || is_exact_roll) {
      return true;
    }
    if (is_debugging) std::cout << "DEBUG: Invalid bearing off move" << std::endl;
    return false;
  }
  if (to_pos < 0 || to_pos >= kNumPoints) {
    if (is_debugging) std::cout << "DEBUG: Invalid to_pos " << to_pos << std::endl;
    return false;
  }
  if (board(1-player, to_pos) > 0) {
    if (is_debugging) std::cout << "DEBUG: Cannot land on opponent's checker at " << to_pos << std::endl;
    return false;
  }
  if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
    if (is_debugging) std::cout << "DEBUG: Would form illegal blocking bridge" << std::endl;
    return false;
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
  bool is_doubles = dice_.size() == 2 && dice_[0] == dice_[1];
  std::vector<int> die_usage(7, 0);
  for (int die : dice_) {
    if (die > 6) {
      die_usage[die - 6]++;
    }
  }
  for (int i = 0; i < kNumPoints; ++i) {
    if (board(player, i) <= 0) continue;
    for (int outcome : dice_) {
      if (!UsableDiceOutcome(outcome)) {
        continue;
      }
      if (is_doubles && die_usage[outcome] >= 2) {
        continue;
      }
      int to_pos = GetToPos(player, i, outcome);
      if (IsValidCheckerMove(player, i, to_pos, outcome, true)) {
        moves.insert(CheckerMove(i, to_pos, outcome));
      } else if (is_debugging) {
        if (to_pos >= 0 && to_pos < kNumPoints && board(1-player, to_pos) > 0) {
          std::cout << "DEBUG: Cannot land on opponent's checker at " << to_pos << std::endl;
        } else {
          std::cout << "DEBUG: Invalid move from " << i << " to " << to_pos << " with die=" << outcome << " for player " << player << std::endl;
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
  if (!IsValidCheckerMove(player, move.pos, move.to_pos, move.die, false)) {
    bool is_debugging = false;
    std::string error_message = absl::StrCat("Invalid checker move from ", move.pos, " to ", move.to_pos, " with die ", move.die, " for player ", player);
    if (is_debugging) {
      std::cout << "DEBUG: " << error_message << std::endl;
    }
    return;
  }
  board_[player][move.pos]--;
  for (int i = 0; i < 2; ++i) {
    if (dice_[i] == move.die) {
      dice_[i] += 6;
      break;
    }
  }
  int next_pos = GetToPos(player, move.pos, move.die);
  if (IsOff(player, next_pos)) {
    scores_[player]++;
  } else {
    board_[player][next_pos]++;
  }
  if (IsHeadPos(player, move.pos)) {
    moved_from_head_ = true;
  }
}

void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return;
  int next_pos = GetToPos(player, move.pos, move.die);
  if (IsOff(player, next_pos)) {
    scores_[player]--;
  } else {
    board_[player][next_pos]--;
  }
  for (int i = 0; i < 2; ++i) {
    if (dice_[i] == move.die + 6) {
      dice_[i] -= 6;
      break;
    }
  }
  board_[player][move.pos]++;
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
  const size_t kSafeLimit = 50;
  if (movelist->size() >= kSafeLimit) {
    return moveseq.size();
  }
  if (max_depth <= 0 || moveseq.size() >= 2) {
    if (!moveseq.empty()) {
      movelist->insert(moveseq);
    }
    return moveseq.size();
  }
  std::set<CheckerMove> moves_here = LegalCheckerMoves(cur_player_);
  if (moves_here.empty()) {
    movelist->insert(moveseq);
    return moveseq.size();
  }
  bool used_head_move = false;
  if (!is_first_turn_ && !moveseq.empty()) {
    int head_pos = cur_player_ == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos;
    for (const auto& prev_move : moveseq) {
      if (prev_move.pos == head_pos) {
        used_head_move = true;
        break;
      }
    }
  }
  const size_t kMaxMovesToCheck = 3;
  size_t moves_checked = 0;
  std::vector<CheckerMove> new_moveseq;
  new_moveseq.reserve(2);
  new_moveseq = moveseq;
  int max_moves = -1;
  for (const auto& move : moves_here) {
    if (movelist->size() >= kSafeLimit / 2) return moveseq.size();
    if (moves_checked >= kMaxMovesToCheck) break;
    moves_checked++;
    if (!is_first_turn_ && used_head_move && IsHeadPos(cur_player_, move.pos)) continue;
    new_moveseq.push_back(move);
    ApplyCheckerMove(cur_player_, move);
    if (movelist->size() >= kSafeLimit / 2) {
      UndoCheckerMove(cur_player_, move);
      new_moveseq.pop_back();
      return max_moves > 0 ? max_moves : moveseq.size();
    }
    int child_max = RecLegalMoves(new_moveseq, movelist, max_depth - 1);
    UndoCheckerMove(cur_player_, move);
    new_moveseq.pop_back();
    max_moves = std::max(child_max, max_moves);
  }
  return max_moves;
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
    return from_pos - pips;
  } else {
    return CounterClockwisePos(from_pos, pips, kNumPoints);
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

  // Compute dice values.
  int high_roll = DiceValue(0);
  int low_roll = DiceValue(1);
  if (high_roll < low_roll) std::swap(high_roll, low_roll);

  std::vector<Action> moves;
  bool found_any_valid_move = false;
  
  // Instead of directly using {high_roll, low_roll} which causes deduction errors,
  // we explicitly create a temporary vector.
  std::vector<int> dice_vals = {high_roll, low_roll};
  
  for (int pos1 = 1; pos1 < kNumPoints; ++pos1) {
    if (board_[cur_player_][pos1] == 0) continue;
    for (int die1 : dice_vals) {
      int to_pos1 = GetToPos(cur_player_, pos1, die1);
      if (to_pos1 >= 0 && to_pos1 < kNumPoints && board_[1 - cur_player_][to_pos1] > 0) {
        continue;
      }
      if (!IsValidCheckerMove(cur_player_, pos1, to_pos1, die1, true)) {
        continue;
      }
      found_any_valid_move = true;
      int die2 = (die1 == high_roll) ? low_roll : high_roll;
      std::vector<CheckerMove> moves_encoding = {{pos1, die1}, kPassMove};
      moves.push_back(CheckerMovesToSpielMove(moves_encoding));
      for (int pos2 = 1; pos2 < kNumPoints; ++pos2) {
        if (pos2 == pos1 && board_[cur_player_][pos1] <= 1) continue;
        if (pos2 != pos1 && board_[cur_player_][pos2] == 0) continue;
        int to_pos2 = GetToPos(cur_player_, pos2, die2);
        if (to_pos2 >= 0 && to_pos2 < kNumPoints && board_[1 - cur_player_][to_pos2] > 0) {
          continue;
        }
        if (!IsValidCheckerMove(cur_player_, pos2, to_pos2, die2, true)) {
          continue;
        }
        std::vector<CheckerMove> moves_encoding = {{pos1, die1}, {pos2, die2}};
        moves.push_back(CheckerMovesToSpielMove(moves_encoding));
      }
    }
  }
  if (!found_any_valid_move) {
    std::vector<CheckerMove> pass_move_encoding = {kPassMove, kPassMove};
    Action pass_spiel_action = CheckerMovesToSpielMove(pass_move_encoding);
    moves.push_back(pass_spiel_action);
  }
  std::vector<Action> illegal_actions = IllegalActions();
  if (!illegal_actions.empty()) {
    std::unordered_set<Action> illegal_set(illegal_actions.begin(), illegal_actions.end());
    std::vector<Action> filtered_moves;
    for (Action move : moves) {
      if (illegal_set.find(move) == illegal_set.end()) {
        filtered_moves.push_back(move);
      }
    }
    moves = filtered_moves;
  }
  
  // --- NEW: Filter out head moves if any non-head move exists ---
  // Determine the head position for the current player.
  int head_pos = (cur_player_ == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;

  // Check if there is any legal move that does not originate from the head.
  bool hasNonHeadMove = false;
  for (Action a : moves) {
    std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(cur_player_, a);
    // Ignore pass moves.
    if (!cmoves.empty() && cmoves[0].pos != head_pos && cmoves[0].pos != kPassPos) {
      hasNonHeadMove = true;
      break;
    }
  }

  // If a non-head move exists, filter out any move that originates from the head.
  if (hasNonHeadMove) {
    std::vector<Action> filtered_moves;
    for (Action a : moves) {
      std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(cur_player_, a);
      if (!cmoves.empty() && cmoves[0].pos == head_pos) continue;
      filtered_moves.push_back(a);
    }
    moves = filtered_moves;
  }
  // --- End of new filtering code ---

  return moves;
}

}  // namespace long_narde
}  // namespace open_spiel