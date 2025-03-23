// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// A few constants to help with the conversion to human-readable string formats.
constexpr int kNumOffPosHumanReadable = -2;
// Long Narde doesn't use the bar concept, but this is kept for compatibility with the code structure
constexpr int kNumBarPosHumanReadable = -3;
constexpr int kNumNonDoubleOutcomes = 15;

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 18),
    std::pair<Action, double>(1, 1.0 / 18),
    std::pair<Action, double>(2, 1.0 / 18),
    std::pair<Action, double>(3, 1.0 / 18),
    std::pair<Action, double>(4, 1.0 / 18),
    std::pair<Action, double>(5, 1.0 / 18),
    std::pair<Action, double>(6, 1.0 / 18),
    std::pair<Action, double>(7, 1.0 / 18),
    std::pair<Action, double>(8, 1.0 / 18),
    std::pair<Action, double>(9, 1.0 / 18),
    std::pair<Action, double>(10, 1.0 / 18),
    std::pair<Action, double>(11, 1.0 / 18),
    std::pair<Action, double>(12, 1.0 / 18),
    std::pair<Action, double>(13, 1.0 / 18),
    std::pair<Action, double>(14, 1.0 / 18),
    std::pair<Action, double>(15, 1.0 / 36),
    std::pair<Action, double>(16, 1.0 / 36),
    std::pair<Action, double>(17, 1.0 / 36),
    std::pair<Action, double>(18, 1.0 / 36),
    std::pair<Action, double>(19, 1.0 / 36),
    std::pair<Action, double>(20, 1.0 / 36),
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

// Facts about the game
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
// 1. Internally as kPassPos (-1) in the game logic
// 2. As position 24 when encoding/decoding actions (since valid board positions are 0-23)
std::string PositionToString(int pos) {
  if (pos == kPassPos || pos == 24) return "Pass";
  SPIEL_CHECK_GE(pos, 0);
  SPIEL_CHECK_LT(pos, kNumPoints);  // kNumPoints should be 24 for Long Narde
  return absl::StrCat(pos + 1);
}

std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case kXPlayerId:
      return "x";
    case kOPlayerId:
      return "o";
    case kChancePlayerId:
      return "*";
    case kTerminalPlayerId:
      return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}

std::string PositionToStringHumanReadable(int pos) {
  if (pos == kNumOffPosHumanReadable) {
    return "Off";
  } else if (pos == kPassPos || pos == 24) {
    return "Pass";
  } else {
    return PositionToString(pos);
  }
}

int LongNardeState::GetMoveEndPosition(CheckerMove* cmove, int player,
                                        int start) const {
  int end = cmove->num;
  if (end != kPassPos) {
    // Not a pass, so work out where the piece finished
    end = start - cmove->num;
    if (end <= 0) {
      end = kNumOffPosHumanReadable;  // Off
    }
  }
  return end;
}

// Note: We've removed EncodedBarMove and EncodedPassMove functions and 
// hardcoded their values (24 and 25) in the functions that use them since
// Long Narde doesn't use the bar concept

std::vector<CheckerMove> LongNardeState::SpielMoveToCheckerMoves(
    int player, Action spiel_move) const {
  SPIEL_CHECK_GE(spiel_move, 0);
  SPIEL_CHECK_LT(spiel_move, kNumDistinctActions);

  bool high_roll_first = spiel_move < 625;  // base-25: 25*25 = 625
  if (!high_roll_first) {
    spiel_move -= 625;
  }

  std::vector<Action> digits = {spiel_move % 25, spiel_move / 25};  // base-25
  std::vector<CheckerMove> cmoves;
  int high_roll = DiceValue(0) >= DiceValue(1) ? DiceValue(0) : DiceValue(1);
  int low_roll = DiceValue(0) < DiceValue(1) ? DiceValue(0) : DiceValue(1);

  for (int i = 0; i < 2; ++i) {
    SPIEL_CHECK_GE(digits[i], 0);
    SPIEL_CHECK_LE(digits[i], 24);  // 0-23 for board positions, 24 for pass

    int num = -1;
    if (i == 0) {
      num = high_roll_first ? high_roll : low_roll;
    } else {
      num = high_roll_first ? low_roll : high_roll;
    }
    SPIEL_CHECK_GE(num, 1);
    SPIEL_CHECK_LE(num, 6);

    if (digits[i] == 24) {  // Pass move is now 24 instead of 25
      cmoves.push_back(kPassMove);
    } else {
      cmoves.push_back(CheckerMove(digits[i], num));
    }
  }

  return cmoves;
}

Action LongNardeState::CheckerMovesToSpielMove(
    const std::vector<CheckerMove>& moves) const {
  SPIEL_CHECK_LE(moves.size(), 2);

  int high_roll = DiceValue(0) >= DiceValue(1) ? DiceValue(0) : DiceValue(1);
  int low_roll = DiceValue(0) < DiceValue(1) ? DiceValue(0) : DiceValue(1);
  bool high_roll_first = true;

  Action dig0 = 24;  // Pass move is now 24 instead of 25
  Action dig1 = 24;  // Pass move is now 24 instead of 25

  int pos0 = moves[0].pos;
  if (pos0 != kPassPos) {
    dig0 = pos0;
  }

  if (moves.size() > 1) {
    int pos1 = moves[1].pos;
    if (pos1 != kPassPos) {
      dig1 = pos1;
    }
  }

  Action move = dig1 * 25 + dig0;  // base-25 encoding
  if (!high_roll_first) {
    move += 625;  // 25*25 = 625
  }
  SPIEL_CHECK_GE(move, 0);
  SPIEL_CHECK_LT(move, kNumDistinctActions);
  return move;
}

std::string LongNardeState::ActionToString(Player player,
                                            Action move_id) const {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move_id, 0);
    SPIEL_CHECK_LT(move_id, kChanceOutcomes.size());
    if (turns_ >= 0) {
      // Normal chance roll.
      return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
    } else {
      // Initial roll to determine who starts.
      const char* starter = (move_id < kNumNonDoubleOutcomes ?
                             "X starts" : "O starts");
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

  int cmove0_start;
  int cmove1_start;
  if (player == kOPlayerId) {
    cmove0_start = cmoves[0].pos + 1;
    cmove1_start = cmoves[1].pos + 1;
  } else {
    // swap the board numbering for Player X
    cmove0_start = kNumPoints - cmoves[0].pos;
    cmove1_start = kNumPoints - cmoves[1].pos;
  }

  // Add hit information and compute whether the moves go off the board.
  int cmove0_end = GetMoveEndPosition(&cmoves[0], player, cmove0_start);
  int cmove1_end = GetMoveEndPosition(&cmoves[1], player, cmove1_start);

  std::string returnVal = "";
  if (cmove0_start == cmove1_start && cmove0_end == cmove1_end) {     // same move, show as (2).
    if (cmoves[1].num == kPassPos) {  // Player can't move at all!
      returnVal = "Pass";
    } else {
      returnVal = absl::StrCat(move_id, " - ",
                               PositionToStringHumanReadable(cmove0_start),
                               "/", PositionToStringHumanReadable(cmove0_end),
                               "(2)");
    }
  } else if ((cmove0_start < cmove1_start ||
              (cmove0_start == cmove1_start && cmove0_end < cmove1_end) ||
              cmoves[0].num == kPassPos) &&
             cmoves[1].num != kPassPos) {
    // tradition to start with higher numbers first, so swap moves round if this not the case.
    // If there is a pass move, put it last.
    if (cmove1_end == cmove0_start) {
      // Check to see if the same piece is moving for both moves, as this changes the format of the output.
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
          PositionToStringHumanReadable(cmove1_end), "/",
          PositionToStringHumanReadable(cmove0_end));
    } else {
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
          PositionToStringHumanReadable(cmove1_end), " ",
          (cmoves[0].num != kPassPos) ? PositionToStringHumanReadable(cmove0_start) : "",
          (cmoves[0].num != kPassPos) ? "/" : "",
          PositionToStringHumanReadable(cmove0_end));
    }
  } else {
    if (cmove0_end == cmove1_start) {
      // Check to see if the same piece is moving for both moves, as this changes the format of the output.
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
          PositionToStringHumanReadable(cmove0_end), "/",
          PositionToStringHumanReadable(cmove1_end));
    } else {
      returnVal = absl::StrCat(
          move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
          PositionToStringHumanReadable(cmove0_end), " ",
          (cmoves[1].num != kPassPos) ? PositionToStringHumanReadable(cmove1_start) : "",
          (cmoves[1].num != kPassPos) ? "/" : "",
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
  
  // Simplified encoding for Long Narde - just store the number of checkers
  // on each point for each player
  for (int i = 0; i < kNumPoints; ++i) {
    *value_it++ = board(player, i);  // Number of current player's checkers
  }
  for (int i = 0; i < kNumPoints; ++i) {
    *value_it++ = board(opponent, i);  // Number of opponent's checkers
  }
  
  *value_it++ = (scores_[player]);
  *value_it++ = ((cur_player_ == player) ? 1 : 0);

  *value_it++ = (scores_[opponent]);
  *value_it++ = ((cur_player_ == opponent) ? 1 : 0);

  *value_it++ = ((!dice_.empty()) ? dice_[0] : 0);
  *value_it++ = ((dice_.size() > 1) ? dice_[1] : 0);

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
      board_(
          {std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
      turn_history_info_({}),
      allow_last_roll_tie_(false),
      scoring_type_(ParseScoringType(
          game->GetParameters().count("scoring_type") > 0 ? 
          game->GetParameters().at("scoring_type").string_value() : 
          kDefaultScoringType)) {
  SetupInitialBoard();
}

void LongNardeState::SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                              const std::vector<int>& scores,
                              const std::vector<std::vector<int>>& board) {
  cur_player_ = cur_player;
  double_turn_ = double_turn;
  dice_ = dice;
  scores_ = scores;
  board_ = board;
}

void LongNardeState::SetupInitialBoard() {
  // Long Narde initial setup 
  // White's 15 checkers on point 24 (index 23)
  board_[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  // Black's 15 checkers on point 12 (index 11)
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
  dice_.push_back(kChanceOutcomeValues[outcome][0]);
  dice_.push_back(kChanceOutcomeValues[outcome][1]);
}

int LongNardeState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());

  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else if (dice_[i] >= 7 && dice_[i] <= 12) {
    // This die is marked as chosen, so return its proper value.
    // Note: dice are only marked as chosen during the legal moves enumeration.
    return dice_[i] - 6;
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

// Long Narde specific function to check if a position is a head position for a player
bool LongNardeState::IsHeadPos(int player, int pos) const {
  return (player == kXPlayerId && pos == kWhiteHeadPos) || 
         (player == kOPlayerId && pos == kBlackHeadPos);
}

// Check if a move from a head position is legal according to the head rule
bool LongNardeState::IsLegalHeadMove(int player, int from_pos) const {
  // If not moving from head, the move is allowed
  if (!IsHeadPos(player, from_pos)) {
    return true;
  }

  // First turn always allows moves from head
  if (is_first_turn_) {
    return true;
  }

  // If we already moved from head in this turn, no more moves from head are allowed
  if (moved_from_head_) {
    return false;
  }

  return true;  // One move from head is always allowed
}

// Check if a move would form a blocking bridge (6 consecutive checkers)
bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const {
  // Clone the board to simulate the move
  std::vector<std::vector<int>> temp_board = board_;
  
  // Simulate the move
  if (from_pos >= 0 && from_pos < kNumPoints) {
    temp_board[player][from_pos]--;
  }
  
  // Skip if the move is bearing off (to_pos is off the board)
  if (to_pos >= 0 && to_pos < kNumPoints) {
    temp_board[player][to_pos]++;
  } else {
    return false;  // Bearing off doesn't create a bridge
  }
  
  // Check for 6 consecutive checkers
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
      
      // Check if we have 6 consecutive points with checkers
      if (consecutive_count == 6) {
        // Now check if all opponent checkers are trapped behind this bridge
        bool any_opponent_ahead = false;
        int opponent = Opponent(player);
        
        // For white, check if any black checkers are beyond the bridge
        if (player == kXPlayerId) {
          int bridge_end = pos;
          for (int i = bridge_end + 1; i < kNumPoints; i++) {
            if (board(opponent, i) > 0) {
              any_opponent_ahead = true;
              break;
            }
          }
        } 
        // For black, check if any white checkers are beyond the bridge
        else {
          int bridge_end = pos;
          for (int i = bridge_end - 1; i >= 0; i--) {
            if (board(opponent, i) > 0) {
              any_opponent_ahead = true;
              break;
            }
          }
        }
        
        // If no opponent checkers are ahead of the bridge, this is illegal
        if (!any_opponent_ahead) {
          return true;
        }
      }
    } else {
      consecutive_count = 0;
    }
  }
  
  return false;
}

void LongNardeState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    turn_history_info_.push_back(
        TurnHistoryInfo(kChancePlayerId, prev_player_, dice_, move, double_turn_,
                        is_first_turn_, moved_from_head_));

    if (turns_ == -1) {
      // In Long Narde, White (X) always goes first - no initial roll needed
      SPIEL_CHECK_TRUE(dice_.empty());
      cur_player_ = prev_player_ = kXPlayerId;
      RollDice(move);
      turns_ = 0;

      // Check if this is the "last roll" for a potential tie
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer && scores_[kOPlayerId] >= 14 && 
          scores_[kOPlayerId] < kNumCheckersPerPlayer) {
        allow_last_roll_tie_ = true;
      }

      return;
    } else {
      // Normal chance node.
      SPIEL_CHECK_TRUE(dice_.empty());
      RollDice(move);
      cur_player_ = Opponent(prev_player_);

      // Check if this is the "last roll" for a potential tie
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer && scores_[kOPlayerId] >= 14 && 
          scores_[kOPlayerId] < kNumCheckersPerPlayer) {
        allow_last_roll_tie_ = true;
      }

      return;
    }
  }

  // Normal move action.
  std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, move);
  
  // Track if we move from the head position
  bool first_move_from_head = IsHeadPos(cur_player_, moves[0].pos);
  bool second_move_from_head = IsHeadPos(cur_player_, moves[1].pos);
  
  // Apply checker moves (hitting not supported, so no hit info is returned)
  ApplyCheckerMove(cur_player_, moves[0]);
  ApplyCheckerMove(cur_player_, moves[1]);

  // If either move was from head, update moved_from_head_
  if (first_move_from_head || second_move_from_head) {
    moved_from_head_ = true;
  }

  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, move, double_turn_,
                      is_first_turn_, moved_from_head_));

  if (!double_turn_) {
    turns_++;
    if (cur_player_ == kXPlayerId) {
      x_turns_++;
    } else if (cur_player_ == kOPlayerId) {
      o_turns_++;
    }
  }

  prev_player_ = cur_player_;

  // Doubles don't get extra turns in Long Narde
  
  // Clear the dice and set up for the next move.
  dice_.clear();
  if (IsTerminal()) {
    cur_player_ = kTerminalPlayerId;
  } else {
    cur_player_ = kChancePlayerId;
  }
  
  // Reset first turn and moved_from_head flags for next turn
  is_first_turn_ = false;
  moved_from_head_ = false;
  double_turn_ = false;

  // Check if this is the "last roll" for a potential tie
  if (scores_[kXPlayerId] == kNumCheckersPerPlayer && scores_[kOPlayerId] >= 14 && 
      scores_[kOPlayerId] < kNumCheckersPerPlayer) {
    allow_last_roll_tie_ = true;
  }
}

void LongNardeState::UndoAction(Player player, Action action) {
  // Remove the last history item.
  TurnHistoryInfo info = turn_history_info_.back();
  turn_history_info_.pop_back();

  is_first_turn_ = info.is_first_turn;
  moved_from_head_ = info.moved_from_head;
  cur_player_ = info.player;
  prev_player_ = info.prev_player;
  dice_ = info.dice;
  double_turn_ = info.double_turn;

  if (player == kChancePlayerId && info.dice.empty()) {
    // Undoing first roll, reset to initial state
    cur_player_ = kChancePlayerId;
    prev_player_ = kChancePlayerId;
    turns_ = -1;
    return;
  }

  if (player != kChancePlayerId) {
    // Undoing the move from a terminal state.
    if (cur_player_ == kTerminalPlayerId) {
      cur_player_ = player;
    }

    // Recreate the checker moves
    std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(player, action);

    // Undo in reverse order to handle a case of one checker moving twice.
    // (Second move comes before first move here, following the policy also
    // implemented in UndoCheckerMove.)
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

// Modified for Long Narde home/bearing off rules
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

// Checks if all checkers are in home
bool LongNardeState::AllInHome(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LE(player, 1);

  // Looking for any checkers outside home.
  if (player == kXPlayerId) {
    // Check areas outside home (points 7-24)
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false;
      }
    }
  } else {  // kOPlayerId
    // Check areas outside home (points 1-12 and 19-24)
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

// Check if the game has ended
bool LongNardeState::IsTerminal() const {
  // The game is over if one player has born off all checkers.
  if (scores_[kXPlayerId] == kNumCheckersPerPlayer ||
      scores_[kOPlayerId] == kNumCheckersPerPlayer) {
    
    // If white has borne off all checkers and black has at least 14 off,
    // black gets one more roll to potentially achieve a tie, but only in winlosstie mode
    if (scoring_type_ == ScoringType::kWinLossTieScoring) {
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer && 
          scores_[kOPlayerId] >= 14 && scores_[kOPlayerId] < kNumCheckersPerPlayer) {
        return false;  // Not terminal yet, black gets last roll
      }
    }
    
    return true;
  }
  return false;
}

// Implement Long Narde scoring
std::vector<double> LongNardeState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  // Check for tie condition (only possible in winlosstie mode)
  if (scoring_type_ == ScoringType::kWinLossTieScoring) {
    // If both players have all checkers off the board, it's a tie
    if (scores_[kXPlayerId] == kNumCheckersPerPlayer && 
        scores_[kOPlayerId] == kNumCheckersPerPlayer) {
      return {0.0, 0.0};
    }
  }

  // First determine who won
  int won = scores_[kXPlayerId] == kNumCheckersPerPlayer
                ? kXPlayerId
                : kOPlayerId;
  int lost = Opponent(won);

  // Check if opponent has any checkers off (Oyn vs Mars)
  double score = (scores_[lost] > 0) ? 1.0 : 2.0;  // 1 for Oyn, 2 for Mars
  
  return (won == kXPlayerId ? std::vector<double>{score, -score}
                           : std::vector<double>{-score, score});
}

// Long Narde game implementation
LongNardeGame::LongNardeGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(ParseScoringType(
          params.count("scoring_type") > 0 ? params.at("scoring_type").string_value() : kDefaultScoringType)) {}

double LongNardeGame::MaxUtility() const {
  return 2;  // Mars scores 2 points
}

// Keep other implementation details the same as backgammon for now,
// especially for the basic game mechanics that aren't changing
// ...

std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  
  if (turns_ == -1) {
    // In Long Narde, White (X) always goes first
    // First roll is still needed but only for the dice values, not for determining the starter
    return kChanceOutcomes;
  } else {
    // Regular turns use all possible dice combinations
    return kChanceOutcomes;
  }
}

std::set<CheckerMove> LongNardeState::LegalCheckerMoves(int player) const {
  std::set<CheckerMove> moves;
  /*
  *** Commented out for now, because it's too verbose. We can uncomment it if we want to debug.
  *
  std::cout << "LegalCheckerMoves for player " << player << ", is_first_turn=" << is_first_turn_ 
            << ", head pos=" << (player == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos) 
            << ", board[player][head]=" << board(player, (player == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos))
            << ", dice: " << dice_[0] << "," << dice_[1] << std::endl;
  */

  // Regular board moves
  bool all_in_home = AllInHome(player);
  for (int i = 0; i < kNumPoints; ++i) {
    if (board(player, i) <= 0) continue;

    // Check if this is a legal position to move from (head rule)
    if (!IsLegalHeadMove(player, i)) {
      continue;
    }

    for (int outcome : dice_) {
      if (!UsableDiceOutcome(outcome)) {
        continue;
      }

      int to_pos = GetToPos(player, i, outcome);
      
      // Handle bearing off
      if ((player == kXPlayerId && to_pos < 0) || 
          (player == kOPlayerId && to_pos >= kNumPoints)) {
        if (all_in_home) {
          // Check if this is a legal bear-off move
          if (i == FurthestCheckerInHome(player) || 
              (player == kXPlayerId && i + outcome == -1) ||
              (player == kOPlayerId && i + outcome == kNumPoints)) {
            moves.insert(CheckerMove(i, outcome));
          }
        }
        continue;
      }

      // Regular move validation
      if (to_pos >= 0 && to_pos < kNumPoints) {
        // Check if landing position is free and won't create illegal blocking bridge
        if (board(1-player, to_pos) == 0 && !WouldFormBlockingBridge(player, i, to_pos)) {
          moves.insert(CheckerMove(i, outcome));
        }
      }
    }
  }

  return moves;
}

std::vector<Action> LongNardeState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  SPIEL_CHECK_GE(cur_player_, 0);
  SPIEL_CHECK_LE(cur_player_, 1);

  // Get all legal single checker moves
  std::set<CheckerMove> legal_moves = LegalCheckerMoves(cur_player_);

  // If no legal moves available, return pass move
  if (legal_moves.empty()) {
    // Create a vector with two pass moves
    std::vector<CheckerMove> pass_moves = {kPassMove, kPassMove};
    return {CheckerMovesToSpielMove(pass_moves)};
  }

  // Create a state copy for move generation
  std::unique_ptr<State> cstate = Clone();
  LongNardeState* state = dynamic_cast<LongNardeState*>(cstate.get());
  std::set<std::vector<CheckerMove>> movelist;
  int max_moves = state->RecLegalMoves({}, &movelist);

  // Convert move sequences to actions
  return ProcessLegalMoves(max_moves, movelist);
}

int LongNardeState::RecLegalMoves(std::vector<CheckerMove> moveseq,
                                 std::set<std::vector<CheckerMove>>* movelist) {
  // Base case: if we've used both dice, add the sequence
  if (moveseq.size() == 2) {
    movelist->insert(moveseq);
    return moveseq.size();  // Return the actual size, not hardcoded 2
  }

  std::set<CheckerMove> moves_here = LegalCheckerMoves(cur_player_);

  if (moves_here.empty()) {
    movelist->insert(moveseq);
    return moveseq.size();
  }

  int max_moves = -1;
  for (const auto& move : moves_here) {
    moveseq.push_back(move);
    ApplyCheckerMove(cur_player_, move);
    int child_max = RecLegalMoves(moveseq, movelist);
    UndoCheckerMove(cur_player_, move);
    max_moves = std::max(child_max, max_moves);
    moveseq.pop_back();
  }

  return max_moves;
}

std::unique_ptr<State> LongNardeState::Clone() const {
  return std::unique_ptr<State>(new LongNardeState(*this));
}

// Get the To position for this play given the from position and number of
// pips on the die.
int LongNardeState::GetToPos(int player, int from_pos, int pips) const {
  // Both players move counter-clockwise in Long Narde, which means:
  // - White moves from higher positions to lower (e.g., 23→22→...→0)
  //   and bears off when going below 0
  // - Black also moves counter-clockwise (e.g., 11→10→...→0→23→22→...→12)
  //   and bears off when going above 23
  
  // For Black, implement counter-clockwise movement with position wrapping
  if (player == kOPlayerId) {
    // Apply counter-clockwise movement
    int pos = from_pos - pips;
    // Handle position wrapping
    if (pos < 0) {
      pos += kNumPoints; // Wrap around from 0 to 23
    }
    return pos;
  } else {
    // For White, counter-clockwise is simply decreasing position
    return from_pos - pips;
  }
}

// Returns the position of the furthest checker in the home of this player.
// For White (kXPlayerId), this is the highest position in 0-5 that has a checker.
// For Black (kOPlayerId), this is the lowest position in 12-17 that has a checker.
// Returns -1 if none found.
int LongNardeState::FurthestCheckerInHome(int player) const {
  if (player == kXPlayerId) {
    // White's home is positions 0-5
    // The furthest checker is the one at the highest position (closest to 5)
    for (int i = kWhiteHomeEnd; i >= kWhiteHomeStart; --i) {
      if (board(player, i) > 0) {
        return i;
      }
    }
  } else {  // kOPlayerId
    // Black's home is positions 12-17
    // The furthest checker is the one at the lowest position (closest to 12)
    for (int i = kBlackHomeStart; i <= kBlackHomeEnd; ++i) {
      if (board(player, i) > 0) {
        return i;
      }
    }
  }
  
  return -1;  // No checkers found in home
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

// Apply a checker move: remove from the source position and add to the destination
bool LongNardeState::ApplyCheckerMove(int player, const CheckerMove& move) {
  // Pass does nothing
  if (move.pos == kPassPos) {
    return false;
  }

  // First, remove the checker from source
  board_[player][move.pos]--;

  // Mark the die as used
  for (int i = 0; i < 2; ++i) {
    if (dice_[i] == move.num) {
      dice_[i] += 6;
      break;
    }
  }

  // Calculate where the checker ends up
  int next_pos = GetToPos(player, move.pos, move.num);

  // Now add the checker (or score)
  if ((player == kXPlayerId && next_pos < 0) || 
      (player == kOPlayerId && next_pos >= kNumPoints)) {
    scores_[player]++;
  } else {
    board_[player][next_pos]++;
  }

  return false;  // Long Narde doesn't support hitting
}

// Undo a checker move
void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) {
  // Undoing a pass does nothing
  if (move.pos == kPassPos) {
    return;
  }

  // Calculate where the checker ended up
  int next_pos = GetToPos(player, move.pos, move.num);

  // Remove the moved checker or decrement score
  if ((player == kXPlayerId && next_pos < 0) || 
      (player == kOPlayerId && next_pos >= kNumPoints)) {
    scores_[player]--;
  } else {
    board_[player][next_pos]--;
  }

  // Mark the die as unused
  for (int i = 0; i < 2; ++i) {
    if (dice_[i] == move.num + 6) {
      dice_[i] -= 6;
      break;
    }
  }

  // Return the checker to its original position
  board_[player][move.pos]++;
}

// Check if a dice outcome is still usable (not marked as used)
bool LongNardeState::UsableDiceOutcome(int outcome) const {
  // A die value of 7-12 means it's been used (marked by adding 6)
  return outcome >= 1 && outcome <= 6;
}

// Process legal moves into actions
std::vector<Action> LongNardeState::ProcessLegalMoves(
    int max_moves, const std::set<std::vector<CheckerMove>>& movelist) const {
  std::vector<Action> legal_moves;
  
  // Convert each move sequence to an action
  for (const auto& moveseq : movelist) {
    if (moveseq.size() == max_moves) {
      Action action = CheckerMovesToSpielMove(moveseq);
      legal_moves.push_back(action);
    }
  }
  
  // Sort the actions in ascending order
  std::sort(legal_moves.begin(), legal_moves.end());
  
  // Remove duplicate actions
  auto new_end = std::unique(legal_moves.begin(), legal_moves.end());
  legal_moves.erase(new_end, legal_moves.end());
  
  return legal_moves;
}

// Convert the current game state to a human-readable string
std::string LongNardeState::ToString() const {
  std::string rv;
  
  // Add current player
  absl::StrAppend(&rv, "Current player: ", CurPlayerToString(cur_player_), "\n");
  
  // Add dice info if any
  if (!dice_.empty()) {
    absl::StrAppend(&rv, "Dice: ");
    for (int die : dice_) {
      if (die <= 6) {  // Only show unused dice
        absl::StrAppend(&rv, die, " ");
      }
    }
    absl::StrAppend(&rv, "\n");
  }
  
  // Add scores
  absl::StrAppend(&rv, "Scores - White: ", scores_[kXPlayerId],
                  ", Black: ", scores_[kOPlayerId], "\n");
  
  // Add board state
  absl::StrAppend(&rv, "Board:\n");
  // White's pieces (X)
  absl::StrAppend(&rv, "White (X): ");
  for (int i = 0; i < kNumPoints; ++i) {
    if (board_[kXPlayerId][i] > 0) {
      absl::StrAppend(&rv, i + 1, ":", board_[kXPlayerId][i], " ");
    }
  }
  absl::StrAppend(&rv, "\n");
  
  // Black's pieces (O)
  absl::StrAppend(&rv, "Black (O): ");
  for (int i = 0; i < kNumPoints; ++i) {
    if (board_[kOPlayerId][i] > 0) {
      absl::StrAppend(&rv, i + 1, ":", board_[kOPlayerId][i], " ");
    }
  }
  absl::StrAppend(&rv, "\n");
  
  return rv;
}

// Check if a position is off the board for a given player
bool LongNardeState::IsOff(int player, int pos) const {
  // For White (kXPlayerId), positions < 0 are off the board (bearing off)
  // For Black (kOPlayerId), positions >= kNumPoints (24) are off the board
  return (player == kXPlayerId) ? pos < 0 : pos >= kNumPoints;
}

}  // namespace long_narde
}  // namespace open_spiel

