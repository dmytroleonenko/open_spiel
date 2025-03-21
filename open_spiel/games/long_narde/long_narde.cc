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
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

// A few constants to help with the conversion to human-readable string formats.
constexpr int kNumBarPosHumanReadable = 25;
constexpr int kNumOffPosHumanReadable = -2;
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
    /*parameter_specification=*/
    {{"scoring_type",
      GameParameter(static_cast<std::string>(kDefaultScoringType))}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LongNardeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

ScoringType ParseScoringType(const std::string& st_str) {
  if (st_str == "winloss_scoring") {
    return ScoringType::kWinLossScoring;
  } else if (st_str == "enable_gammons") {
    return ScoringType::kEnableGammons;
  } else if (st_str == "full_scoring") {
    return ScoringType::kFullScoring;
  } else {
    SpielFatalError("Unrecognized scoring_type parameter: " + st_str);
  }
}

std::string PositionToString(int pos) {
  switch (pos) {
    case kBarPos:
      return "Bar";
    case kScorePos:
      return "Score";
    case -1:
      return "Pass";
    default:
      return absl::StrCat(pos);
  }
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
  if (pos == kNumBarPosHumanReadable) {
    return "Bar";
  } else if (pos == kNumOffPosHumanReadable) {
    return "Off";
  } else {
    return PositionToString(pos);
  }
}

int LongNardeState::AugmentCheckerMove(CheckerMove* cmove, int player,
                                        int start) const {
  int end = cmove->num;
  if (end != kPassPos) {
    // Not a pass, so work out where the piece finished
    end = start - cmove->num;
    if (end <= 0) {
      end = kNumOffPosHumanReadable;  // Off
    } else if (board_[Opponent(player)]
                     [player == kOPlayerId ? (end - 1) : (kNumPoints - end)] ==
               1) {
      cmove->hit = true;  // Check to see if move is a hit
    }
  }
  return end;
}

std::string LongNardeState::ActionToString(Player player,
                                            Action move_id) const {
  if (player == kChancePlayerId) {
    if (turns_ >= 0) {
      // Normal chance roll.
      return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
    } else {
      // Initial roll to determine who starts.
      const char* starter = (move_id < kNumNonDoubleOutcomes ?
                             "X starts" : "O starts");
      if (move_id >= kNumNonDoubleOutcomes) {
        move_id -= kNumNonDoubleOutcomes;
      }
      return absl::StrCat("chance outcome ", move_id, " ", starter, ", ",
                          "(roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
    }
  } else {
    std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(player, move_id);

    int cmove0_start;
    int cmove1_start;
    if (player == kOPlayerId) {
      cmove0_start = (cmoves[0].pos == kBarPos ? kNumBarPosHumanReadable
                                               : cmoves[0].pos + 1);
      cmove1_start = (cmoves[1].pos == kBarPos ? kNumBarPosHumanReadable
                                               : cmoves[1].pos + 1);
    } else {
      // swap the board numbering round for Player X so player is moving
      // from 24->0
      cmove0_start = (cmoves[0].pos == kBarPos ? kNumBarPosHumanReadable
                                               : kNumPoints - cmoves[0].pos);
      cmove1_start = (cmoves[1].pos == kBarPos ? kNumBarPosHumanReadable
                                               : kNumPoints - cmoves[1].pos);
    }

    // Add hit information and compute whether the moves go off the board.
    int cmove0_end = AugmentCheckerMove(&cmoves[0], player, cmove0_start);
    int cmove1_end = AugmentCheckerMove(&cmoves[1], player, cmove1_start);

    // check for 2 pieces hitting on the same point.
    bool double_hit =
        (cmoves[1].hit && cmoves[0].hit && cmove1_end == cmove0_end);

    std::string returnVal = "";
    if (cmove0_start == cmove1_start &&
        cmove0_end == cmove1_end) {     // same move, show as (2).
      if (cmoves[1].num == kPassPos) {  // Player can't move at all!
        returnVal = "Pass";
      } else {
        returnVal = absl::StrCat(move_id, " - ",
                                 PositionToStringHumanReadable(cmove0_start),
                                 "/", PositionToStringHumanReadable(cmove0_end),
                                 cmoves[0].hit ? "*" : "", "(2)");
      }
    } else if ((cmove0_start < cmove1_start ||
                (cmove0_start == cmove1_start && cmove0_end < cmove1_end) ||
                cmoves[0].num == kPassPos) &&
               cmoves[1].num != kPassPos) {
      // tradition to start with higher numbers first,
      // so swap moves round if this not the case. If
      // there is a pass move, put it last.
      if (cmove1_end == cmove0_start) {
        // Check to see if the same piece is moving for both
        // moves, as this changes the format of the output.
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
            PositionToStringHumanReadable(cmove1_end), cmoves[1].hit ? "*" : "",
            "/", PositionToStringHumanReadable(cmove0_end),
            cmoves[0].hit ? "*" : "");
      } else {
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
            PositionToStringHumanReadable(cmove1_end), cmoves[1].hit ? "*" : "",
            " ",
            (cmoves[0].num != kPassPos)
                ? PositionToStringHumanReadable(cmove0_start)
                : "",
            (cmoves[0].num != kPassPos) ? "/" : "",
            PositionToStringHumanReadable(cmove0_end),
            (cmoves[0].hit && !double_hit) ? "*" : "");
      }
    } else {
      if (cmove0_end == cmove1_start) {
        // Check to see if the same piece is moving for both
        // moves, as this changes the format of the output.
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
            PositionToStringHumanReadable(cmove0_end), cmoves[0].hit ? "*" : "",
            "/", PositionToStringHumanReadable(cmove1_end),
            cmoves[1].hit ? "*" : "");
      } else {
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
            PositionToStringHumanReadable(cmove0_end), cmoves[0].hit ? "*" : "",
            " ",
            (cmoves[1].num != kPassPos)
                ? PositionToStringHumanReadable(cmove1_start)
                : "",
            (cmoves[1].num != kPassPos) ? "/" : "",
            PositionToStringHumanReadable(cmove1_end),
            (cmoves[1].hit && !double_hit) ? "*" : "");
      }
    }

    return returnVal;
  }
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
  
  for (int count : board_[player]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  for (int count : board_[opponent]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  *value_it++ = (bar_[player]);
  *value_it++ = (scores_[player]);
  *value_it++ = ((cur_player_ == player) ? 1 : 0);

  *value_it++ = (bar_[opponent]);
  *value_it++ = (scores_[opponent]);
  *value_it++ = ((cur_player_ == opponent) ? 1 : 0);

  *value_it++ = ((!dice_.empty()) ? dice_[0] : 0);
  *value_it++ = ((dice_.size() > 1) ? dice_[1] : 0);

  SPIEL_CHECK_EQ(value_it, values.end());
}

LongNardeState::LongNardeState(std::shared_ptr<const Game> game,
                                 ScoringType scoring_type)
    : State(game),
      scoring_type_(scoring_type),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      x_turns_(0),
      o_turns_(0),
      double_turn_(false),
      is_first_turn_(true),
      moved_from_head_(false),
      dice_({}),
      bar_({0, 0}),
      scores_({0, 0}),
      board_(
          {std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
      turn_history_info_({}),
      allow_last_roll_tie_(false) {
  SetupInitialBoard();
}

void LongNardeState::SetupInitialBoard() {
  // Long Narde initial setup 
  // White's 15 checkers on point 24 (index 23)
  board_[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  // Black's 15 checkers on point 12 (index 11)
  board_[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
}

int LongNardeState::board(int player, int pos) const {
  if (pos == kBarPos) {
    return bar_[player];
  } else {
    SPIEL_CHECK_GE(pos, 0);
    SPIEL_CHECK_LT(pos, kNumPoints);
    return board_[player][pos];
  }
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

  // If we already moved from head in this turn, no more moves from head are allowed
  if (moved_from_head_) {
    return false;
  }

  // First turn special case for doubles 6, 4, or 3
  if (is_first_turn_ && dice_[0] == dice_[1] && 
      (dice_[0] == 6 || dice_[0] == 4 || dice_[0] == 3)) {
    return true;
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
        
        // For white, check if any black checkers are beyond the bridge
        if (player == kXPlayerId) {
          int bridge_end = pos;
          for (int i = bridge_end + 1; i < kNumPoints; i++) {
            if (temp_board[kOPlayerId][i] > 0) {
              any_opponent_ahead = true;
              break;
            }
          }
        } 
        // For black, check if any white checkers are beyond the bridge
        else {
          int bridge_end = pos;
          for (int i = bridge_end - 1; i >= 0; i--) {
            if (temp_board[kXPlayerId][i] > 0) {
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
    turn_history_info_.push_back(TurnHistoryInfo(kChancePlayerId, prev_player_,
                                                 dice_, move, double_turn_,
                                                 false, false, is_first_turn_,
                                                 moved_from_head_));

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
  
  bool first_move_hit = ApplyCheckerMove(cur_player_, moves[0]);
  bool second_move_hit = ApplyCheckerMove(cur_player_, moves[1]);

  // If either move was from head, update moved_from_head_
  if (first_move_from_head || second_move_from_head) {
    moved_from_head_ = true;
  }

  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, move, double_turn_,
                      first_move_hit, second_move_hit, is_first_turn_,
                      moved_from_head_));

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
    moves = AugmentWithHitInfo(player, moves);

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

// Modified to check Long Narde rules
bool LongNardeState::IsLegalFromTo(int player, int from_pos, int to_pos, 
                                  int my_checkers_from, int opp_checkers_to) const {
  // Common basic checks
  if (my_checkers_from <= 0) {
    return false;  // No checkers to move
  }
  
  // Check head rule
  if (!IsLegalHeadMove(player, from_pos)) {
    return false;
  }
  
  // No landing on opponent checkers in Long Narde
  if (opp_checkers_to > 0) {
    return false;
  }
  
  // Check for blocking bridge
  if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
    return false;
  }
  
  // If bearing off, must be exact or higher roll when all checkers in home
  if ((player == kXPlayerId && to_pos >= kNumPoints) || 
      (player == kOPlayerId && to_pos < 0)) {
    if (AllInHome(player)) {
      int furthest = FurthestCheckerInHome(player);
      if (furthest == -1) {
        return true;  // No checkers in home (unlikely but valid)
      }
      
      // For bearing off, ensure we use exact die values when possible
      // or higher values when no exact match is available
      int distance;
      if (player == kXPlayerId) {
        distance = kNumPoints - furthest;
      } else {
        distance = furthest + 1;
      }
      
      int die_value = GetDistance(player, from_pos, to_pos);
      
      if (furthest == from_pos) {
        return true;  // Furthest checker can always bear off with any value
      } else {
        // Must use exact values for non-furthest checkers
        return die_value == GetDistance(player, from_pos, 
                                       player == kXPlayerId ? kNumPoints : -1);
      }
    } else {
      return false;  // Can't bear off until all checkers are in home
    }
  }
  
  return true;
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
  if (bar_[player] > 0) {
    return false;
  }

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LE(player, 1);

  // Looking for any checkers outside home.
  if (player == kXPlayerId) {
    // Check areas outside home (points 7-24)
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) {
      if (board_[player][i] > 0) {
        return false;
      }
    }
  } else {  // kOPlayerId
    // Check areas outside home (points 1-12 and 19-24)
    for (int i = 0; i < kBlackHomeStart; ++i) {
      if (board_[player][i] > 0) {
        return false;
      }
    }
    for (int i = kBlackHomeEnd + 1; i < kNumPoints; ++i) {
      if (board_[player][i] > 0) {
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
    // black gets one more roll to potentially achieve a tie
    if (allow_last_roll_tie_) {
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

  // Check for tie condition (last roll rule)
  if (allow_last_roll_tie_) {
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

  // Then, the score is determined by the scoring system.
  if (scoring_type_ == ScoringType::kWinLossScoring) {
    return (won == kXPlayerId ? std::vector<double>{1.0, -1.0}
                              : std::vector<double>{-1.0, 1.0});
  } else if (scoring_type_ == ScoringType::kEnableGammons) {
    int gammon = IsGammoned(lost);
    double score = 1.0 + gammon;
    return (won == kXPlayerId ? std::vector<double>{score, -score}
                              : std::vector<double>{-score, score});
  } else {
    // Must be full scoring
    SPIEL_CHECK_EQ(scoring_type_, ScoringType::kFullScoring);
    int gammon = IsGammoned(lost);
    int backgammon = IsBackgammoned(lost);
    double score = 1.0 + gammon + backgammon;
    return (won == kXPlayerId ? std::vector<double>{score, -score}
                              : std::vector<double>{-score, score});
  }
}

// Long Narde game implementation
LongNardeGame::LongNardeGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(
          ParseScoringType(ParameterValue<std::string>("scoring_type"))) {}

double LongNardeGame::MaxUtility() const {
  if (scoring_type_ == ScoringType::kWinLossScoring) {
    return 1;
  } else {
    // Mars scores 2 points
    return 2;
  }
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

std::vector<Action> LongNardeState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  SPIEL_CHECK_GE(cur_player_, 0);
  SPIEL_CHECK_LE(cur_player_, 1);

  // Generate all possible legal moves based on the current dice
  std::vector<Action> legal_actions;
  
  // Create action based on each possible starting position
  for (int from_pos = 0; from_pos < kNumPoints; ++from_pos) {
    // Skip positions with no checkers
    if (board_[cur_player_][from_pos] <= 0) {
      continue;
    }
    
    // Try the first die value
    int die_value1 = dice_[0];
    int to_pos1 = GetToPos(cur_player_, from_pos, die_value1);
    bool move1_valid = false;
    
    // Check if move is legal
    if (cur_player_ == kXPlayerId) {
      move1_valid = IsLegalFromTo(cur_player_, from_pos, to_pos1, 
                                  board_[cur_player_][from_pos], 
                                  (to_pos1 < kNumPoints) ? board_[1-cur_player_][to_pos1] : 0);
    } else {  // kOPlayerId
      move1_valid = IsLegalFromTo(cur_player_, from_pos, to_pos1, 
                                  board_[cur_player_][from_pos], 
                                  (to_pos1 >= 0) ? board_[1-cur_player_][to_pos1] : 0);
    }
    
    if (move1_valid) {
      // Create a single move action
      std::vector<CheckerMove> move_seq = {
        {from_pos, die_value1, false}
      };
      legal_actions.push_back(CheckerMovesToSpielMove(move_seq));
    }
    
    // Try the second die value
    int die_value2 = dice_[1];
    int to_pos2 = GetToPos(cur_player_, from_pos, die_value2);
    bool move2_valid = false;
    
    // Check if move is legal
    if (cur_player_ == kXPlayerId) {
      move2_valid = IsLegalFromTo(cur_player_, from_pos, to_pos2, 
                                 board_[cur_player_][from_pos], 
                                 (to_pos2 < kNumPoints) ? board_[1-cur_player_][to_pos2] : 0);
    } else {  // kOPlayerId
      move2_valid = IsLegalFromTo(cur_player_, from_pos, to_pos2, 
                                 board_[cur_player_][from_pos], 
                                 (to_pos2 >= 0) ? board_[1-cur_player_][to_pos2] : 0);
    }
    
    if (move2_valid) {
      // Create a single move action
      std::vector<CheckerMove> move_seq = {
        {from_pos, die_value2, false}
      };
      legal_actions.push_back(CheckerMovesToSpielMove(move_seq));
    }
  }
  
  // Add pass action if no legal moves are available
  if (legal_actions.empty()) {
    std::vector<CheckerMove> pass_move = {
      {kPassPos, -1, false}, 
      {kPassPos, -1, false}
    };
    legal_actions.push_back(CheckerMovesToSpielMove(pass_move));
  }
  
  return legal_actions;
}

std::unique_ptr<State> LongNardeState::Clone() const {
  return std::unique_ptr<State>(new LongNardeState(*this));
}

}  // namespace long_narde
}  // namespace open_spiel

