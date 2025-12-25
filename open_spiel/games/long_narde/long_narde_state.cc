// Copyright 2025 DeepMind Technologies Limited
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
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/long_narde/long_narde_internal.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

namespace {

int CountCheckers(const std::array<int, kNumPoints>& row) {
  int total = 0;
  for (int v : row) {
    total += v;
  }
  return total;
}

int CanonicalSrcFromReal(int real_pos, Player player) {
  if (real_pos == kPassPos) {
    return kActionPassSrc;
  }
  if (player == kXPlayerId) {
    return 23 - real_pos;
  }
  return (11 - real_pos + 24) % 24;
}

int RealPosFromCanonical(int canonical_pos, Player player) {
  if (player == kXPlayerId) {
    return 23 - canonical_pos;
  }
  return (11 - canonical_pos + 24) % 24;
}

int OppRelativeIndex(int canonical_pos) { return (canonical_pos + 12) % 24; }

std::string MoveToString(int src, int die) {
  if (src == kActionPassSrc) {
    return "pass";
  }
  int dst = src + die;
  if (dst >= kNumPoints) {
    return absl::StrCat(src, "->off(", die, ")");
  }
  return absl::StrCat(src, "->", dst, "(", die, ")");
}

}  // namespace

LongNardeState::LongNardeState(std::shared_ptr<const Game> game,
                               ScoringType scoring_type)
    : State(std::move(game)),
      scoring_type_(scoring_type) {
  SetupInitialBoard();
  current_player_ = kXPlayerId;
  awaiting_roll_ = true;
  initial_roll_ = true;
  is_doubles_ = false;
  phase_ = 0;
  head_moved_count_ = 0;
  is_first_turn_ = true;
  terminal_ = false;
  dice_ = {0, 0};
}

void LongNardeState::SetupInitialBoard() {
  for (auto& row : board_) {
    row.fill(0);
  }
  board_[0][0] = kNumCheckersPerPlayer;
  board_[1][12] = kNumCheckersPerPlayer;
}

Player LongNardeState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (awaiting_roll_) {
    return kChancePlayerId;
  }
  return current_player_;
}

std::vector<Action> LongNardeState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }

  int max_die = std::max(dice_[0], dice_[1]);
  if (max_die <= 0) {
    return {internal::kOrderActionCount - 1};
  }

  const internal::ActionTables& tables = internal::GetActionTables();
  internal::ActionBitset raw_bits = internal::GenerateLegalActionBits(
      board_, dice_, head_moved_count_, is_doubles_, is_first_turn_);

  bool keep_pass_order1 = dice_[0] < dice_[1];
  bool has_move = false;
  internal::ActionBitset forced_bits = internal::ApplyForceRulesBits(
      raw_bits, dice_, keep_pass_order1, &has_move);
  internal::ActionBitset pass_bits =
      keep_pass_order1 ? tables.pass_action_order1_bits
                       : tables.pass_action_order0_bits;

  const internal::ActionBitset& out_bits = has_move ? forced_bits : pass_bits;

  std::vector<Action> actions;
  actions.reserve(kNumDistinctActions);
  for (int action = 0; action < kNumDistinctActions; ++action) {
    int word = action / 32;
    int bit = action % 32;
    if ((out_bits[word] >> bit) & 1u) {
      actions.push_back(action);
    }
  }
  return actions;
}

std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  int count = initial_roll_ ? kNumInitialChanceOutcomes : kNumChanceOutcomes;
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(count);
  const double prob = 1.0 / count;
  for (Action a = 0; a < count; ++a) {
    outcomes.push_back({a, prob});
  }
  return outcomes;
}

std::string LongNardeState::ActionToString(Player player,
                                           Action action_id) const {
  if (player == kChancePlayerId) {
    const auto& values = internal::ChanceOutcomeValues();
    int count = initial_roll_ ? kNumInitialChanceOutcomes : kNumChanceOutcomes;
    SPIEL_CHECK_GE(action_id, 0);
    SPIEL_CHECK_LT(action_id, count);
    auto dice = values[action_id];
    return absl::StrCat("roll(", dice[0], ",", dice[1], ")");
  }

  internal::DecodedAction decoded = internal::DecodeAction(action_id);
  int d_min = std::min(dice_[0], dice_[1]);
  int d_max = std::max(dice_[0], dice_[1]);
  int d1 = (decoded.order == 0) ? d_max : d_min;
  int d2 = (decoded.order == 0) ? d_min : d_max;
  return absl::StrCat(MoveToString(decoded.src1, d1), ", ",
                      MoveToString(decoded.src2, d2));
}

std::string LongNardeState::ToString() const {
  std::ostringstream ss;
  ss << "P" << current_player_;
  ss << " roll=" << (awaiting_roll_ ? "*" : "")
     << dice_[0] << "," << dice_[1];
  ss << " phase=" << phase_;
  ss << " head=" << head_moved_count_;
  ss << " first=" << (is_first_turn_ ? "1" : "0");
  ss << " board0=[";
  for (int i = 0; i < kNumPoints; ++i) {
    if (i) ss << ",";
    ss << board_[0][i];
  }
  ss << "] board1=[";
  for (int i = 0; i < kNumPoints; ++i) {
    if (i) ss << ",";
    ss << board_[1][i];
  }
  ss << "]";
  return ss.str();
}

bool LongNardeState::IsTerminal() const {
  if (terminal_) {
    return true;
  }
  return CountCheckers(board_[0]) == 0 || CountCheckers(board_[1]) == 0;
}

std::vector<double> LongNardeState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(kNumPlayers, 0.0);
  }
  int cur_left = CountCheckers(board_[0]);
  int opp_left = CountCheckers(board_[1]);
  if (cur_left == 0 && opp_left == 0 &&
      scoring_type_ == ScoringType::kWinLossTieScoring) {
    return std::vector<double>(kNumPlayers, 0.0);
  }

  Player winner = kInvalidPlayer;
  Player loser = kInvalidPlayer;
  if (cur_left == 0) {
    winner = current_player_;
    loser = 1 - current_player_;
  } else {
    winner = 1 - current_player_;
    loser = current_player_;
  }

  int loser_left = (winner == current_player_) ? opp_left : cur_left;
  int loser_borne = kNumCheckersPerPlayer - loser_left;
  int win_score = (loser_borne > 0) ? 1 : 2;
  std::vector<double> returns(kNumPlayers, 0.0);
  returns[winner] = win_score;
  returns[loser] = -win_score;
  return returns;
}

std::string LongNardeState::ObservationString(Player player) const {
  return ToString();
}

void LongNardeState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  int idx = 0;
  for (int p = 0; p < kNumPlayers; ++p) {
    for (int i = 0; i < kNumPoints; ++i) {
      values[idx++] = static_cast<float>(board_[p][i]);
    }
  }
  int cur_score = kNumCheckersPerPlayer - CountCheckers(board_[0]);
  int opp_score = kNumCheckersPerPlayer - CountCheckers(board_[1]);
  values[idx++] = static_cast<float>(cur_score);
  values[idx++] = static_cast<float>(opp_score);
  values[idx++] = static_cast<float>(current_player_);
  values[idx++] = static_cast<float>(dice_[0]);
  values[idx++] = static_cast<float>(dice_[1]);
  values[idx++] = static_cast<float>(head_moved_count_);
  values[idx++] = is_first_turn_ ? 1.0f : 0.0f;
  values[idx++] = static_cast<float>(phase_);
  values[idx++] = is_doubles_ ? 1.0f : 0.0f;
  values[idx++] = awaiting_roll_ ? 1.0f : 0.0f;
  SPIEL_CHECK_EQ(idx, values.size());
}

std::unique_ptr<State> LongNardeState::Clone() const {
  return std::unique_ptr<State>(new LongNardeState(*this));
}

int LongNardeState::GetCount(Player player, int real_pos) const {
  int canonical_pos = CanonicalSrcFromReal(real_pos, player);
  if (canonical_pos == kActionPassSrc) {
    return 0;
  }
  if (player == current_player_) {
    return board_[0][canonical_pos];
  }
  return board_[1][OppRelativeIndex(canonical_pos)];
}

int LongNardeState::GetToPos(Player player, int real_pos, int die) const {
  int src = CanonicalSrcFromReal(real_pos, player);
  if (src == kActionPassSrc) {
    return kPassPos;
  }
  int target = src + die;
  if (target >= kNumPoints) {
    return kBearOffPos;
  }
  return RealPosFromCanonical(target, player);
}

bool LongNardeState::IsOff(Player player, int real_pos) const {
  return real_pos == kBearOffPos;
}

bool LongNardeState::IsFirstTurn(Player player) const {
  return is_first_turn_ && !awaiting_roll_ && player == current_player_;
}

std::vector<LongNardeCheckerMove>
LongNardeState::LongNardeSpielMoveToCheckerMoves(Player player,
                                                 Action action) const {
  SPIEL_CHECK_EQ(player, current_player_);
  internal::DecodedAction decoded = internal::DecodeAction(action);
  int d_min = std::min(dice_[0], dice_[1]);
  int d_max = std::max(dice_[0], dice_[1]);
  int d1 = (decoded.order == 0) ? d_max : d_min;
  int d2 = (decoded.order == 0) ? d_min : d_max;

  std::vector<LongNardeCheckerMove> moves;
  moves.reserve(2);
  auto add_move = [&](int src, int die) {
    if (src == kActionPassSrc) {
      moves.push_back({kPassPos, kPassPos, die});
      return;
    }
    int real_src = RealPosFromCanonical(src, player);
    int target = src + die;
    int real_dst = (target >= kNumPoints)
                       ? kBearOffPos
                       : RealPosFromCanonical(target, player);
    moves.push_back({real_src, real_dst, die});
  };
  add_move(decoded.src1, d1);
  add_move(decoded.src2, d2);
  return moves;
}

Action LongNardeState::LongNardeCheckerMovesToSpielMove(
    const std::vector<LongNardeCheckerMove>& moves) const {
  SPIEL_CHECK_GE(moves.size(), 1);
  int src1 = kActionPassSrc;
  int src2 = kActionPassSrc;
  int die1 = dice_[0];
  if (!moves.empty()) {
    src1 = CanonicalSrcFromReal(moves[0].pos, current_player_);
    die1 = moves[0].die;
  }
  if (moves.size() >= 2) {
    src2 = CanonicalSrcFromReal(moves[1].pos, current_player_);
  }

  int d_min = std::min(dice_[0], dice_[1]);
  int d_max = std::max(dice_[0], dice_[1]);
  int order = 0;
  if (d_min != d_max && die1 == d_min) {
    order = 1;
  }
  return src1 + 25 * src2 + (order ? 625 : 0);
}

void LongNardeState::SetStateForTesting(
    const std::array<std::array<int, kNumPoints>, 2>& board,
    Player current_player, const std::array<int, 2>& dice, bool awaiting_roll,
    bool initial_roll, bool is_first_turn, int head_moved_count, int phase) {
  board_ = board;
  current_player_ = current_player;
  dice_ = dice;
  awaiting_roll_ = awaiting_roll;
  initial_roll_ = initial_roll;
  is_first_turn_ = is_first_turn;
  head_moved_count_ = head_moved_count;
  phase_ = phase;
  is_doubles_ = dice_[0] == dice_[1];
  terminal_ = CountCheckers(board_[0]) == 0 || CountCheckers(board_[1]) == 0;
}

void LongNardeState::PushUndoState() {
  undo_stack_.push_back(
      UndoState{board_, dice_, current_player_, awaiting_roll_, initial_roll_,
                is_doubles_, phase_, head_moved_count_, is_first_turn_,
                terminal_});
}

void LongNardeState::PopUndoState() {
  SPIEL_CHECK_FALSE(undo_stack_.empty());
  UndoState last = undo_stack_.back();
  undo_stack_.pop_back();
  board_ = last.board;
  dice_ = last.dice;
  current_player_ = last.current_player;
  awaiting_roll_ = last.awaiting_roll;
  initial_roll_ = last.initial_roll;
  is_doubles_ = last.is_doubles;
  phase_ = last.phase;
  head_moved_count_ = last.head_moved_count;
  is_first_turn_ = last.is_first_turn;
  terminal_ = last.terminal;
}

void LongNardeState::UndoAction(Player player, Action action) {
  PopUndoState();
  SPIEL_CHECK_FALSE(history_.empty());
  history_.pop_back();
  --move_number_;
}

void LongNardeState::DoApplyAction(Action action_id) {
  PushUndoState();
  if (IsChanceNode()) {
    const auto& values = internal::ChanceOutcomeValues();
    if (initial_roll_) {
      SPIEL_CHECK_GE(action_id, 0);
      SPIEL_CHECK_LT(action_id, kNumInitialChanceOutcomes);
      auto dice = values[action_id];
      current_player_ = (action_id % 2 == 0) ? kXPlayerId : kOPlayerId;
      if (current_player_ == kOPlayerId) {
        internal::FlipBoard(&board_);
      }
      dice_ = {dice[0], dice[1]};
      is_doubles_ = dice_[0] == dice_[1];
      phase_ = 0;
      head_moved_count_ = 0;
      awaiting_roll_ = false;
      initial_roll_ = false;
      return;
    }

    SPIEL_CHECK_GE(action_id, 0);
    SPIEL_CHECK_LT(action_id, kNumChanceOutcomes);
    auto dice = values[action_id];
    dice_ = {dice[0], dice[1]};
    is_doubles_ = dice_[0] == dice_[1];
    phase_ = 0;
    head_moved_count_ = 0;
    awaiting_roll_ = false;
    return;
  }

  internal::DecodedAction decoded = internal::DecodeAction(action_id);
  int d_min = std::min(dice_[0], dice_[1]);
  int d_max = std::max(dice_[0], dice_[1]);
  int d1 = (decoded.order == 0) ? d_max : d_min;
  int d2 = (decoded.order == 0) ? d_min : d_max;

  auto apply_move = [&](int src, int die) {
    if (src == kActionPassSrc) {
      return;
    }
    int target = src + die;
    int src_read = std::min(src, 23);
    SPIEL_CHECK_GT(board_[0][src_read], 0);
    board_[0][src_read] -= 1;
    if (target < kNumPoints) {
      board_[0][target] += 1;
    }
  };

  apply_move(decoded.src1, d1);
  head_moved_count_ += (decoded.src1 == 0) ? 1 : 0;
  apply_move(decoded.src2, d2);
  head_moved_count_ += (decoded.src2 == 0) ? 1 : 0;

  terminal_ = CountCheckers(board_[0]) == 0;
  bool continue_turn = is_doubles_ && (phase_ == 0) && !terminal_;
  if (continue_turn) {
    phase_ = 1;
    return;
  }

  if (terminal_) {
    return;
  }

  internal::FlipBoard(&board_);
  current_player_ = 1 - current_player_;
  awaiting_roll_ = true;
  dice_ = {0, 0};
  is_doubles_ = false;
  phase_ = 0;
  head_moved_count_ = 0;
  is_first_turn_ = false;
}

}  // namespace long_narde
}  // namespace open_spiel
