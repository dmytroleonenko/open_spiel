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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace long_narde {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumPoints = 24;
inline constexpr int kNumCheckersPerPlayer = 15;
inline constexpr int kNumDistinctActions = 1250;
inline constexpr int kNumChanceOutcomes = 36;
inline constexpr int kNumInitialChanceOutcomes = 30;
inline constexpr int kXPlayerId = 0;
inline constexpr int kOPlayerId = 1;
inline constexpr int kObservationTensorSize = 58;
inline constexpr int kPassPos = -1;
inline constexpr int kBearOffPos = -2;
inline constexpr int kActionPassSrc = 24;

enum class ScoringType {
  kWinLossScoring,
  kWinLossTieScoring,
};

ScoringType ParseScoringType(const std::string& st_str);

struct LongNardeCheckerMove {
  int pos;
  int to_pos;
  int die;
};

class LongNardeState : public State {
 public:
  LongNardeState(const LongNardeState&) = default;
  explicit LongNardeState(std::shared_ptr<const Game> game,
                          ScoringType scoring_type);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  const std::array<std::array<int, kNumPoints>, 2>& board() const {
    return board_;
  }
  int dice(int idx) const { return dice_.at(idx); }
  const std::array<int, 2>& dice() const { return dice_; }
  Player current_player_id() const { return current_player_; }
  bool awaiting_roll() const { return awaiting_roll_; }
  bool initial_roll() const { return initial_roll_; }
  bool is_doubles() const { return is_doubles_; }
  int phase() const { return phase_; }
  int head_moved_count() const { return head_moved_count_; }
  bool is_first_turn() const { return is_first_turn_; }

  int GetCount(Player player, int real_pos) const;
  int GetToPos(Player player, int real_pos, int die) const;
  bool IsOff(Player player, int real_pos) const;
  bool IsFirstTurn(Player player) const;
  std::vector<LongNardeCheckerMove> LongNardeSpielMoveToCheckerMoves(
      Player player, Action action) const;
  Action LongNardeCheckerMovesToSpielMove(
      const std::vector<LongNardeCheckerMove>& moves) const;

  void SetStateForTesting(
      const std::array<std::array<int, kNumPoints>, 2>& board,
      Player current_player, const std::array<int, 2>& dice,
      bool awaiting_roll, bool initial_roll, bool is_first_turn,
      int head_moved_count, int phase);

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void SetupInitialBoard();
  void PushUndoState();
  void PopUndoState();

  std::array<std::array<int, kNumPoints>, 2> board_{};
  std::array<int, 2> dice_{};
  Player current_player_ = kInvalidPlayer;
  bool awaiting_roll_ = true;
  bool initial_roll_ = true;
  bool is_doubles_ = false;
  int phase_ = 0;
  int head_moved_count_ = 0;
  bool is_first_turn_ = true;
  bool terminal_ = false;
  ScoringType scoring_type_;

  struct UndoState {
    std::array<std::array<int, kNumPoints>, 2> board;
    std::array<int, 2> dice;
    Player current_player;
    bool awaiting_roll;
    bool initial_roll;
    bool is_doubles;
    int phase;
    int head_moved_count;
    bool is_first_turn;
    bool terminal;
  };
  std::vector<UndoState> undo_stack_;
};

class LongNardeGame : public Game {
 public:
  explicit LongNardeGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new LongNardeState(shared_from_this(), scoring_type_));
  }
  int MaxChanceOutcomes() const override { return kNumChanceOutcomes; }
  int MaxGameLength() const override { return 1000; }
  int MaxChanceNodesInHistory() const override {
    return MaxGameLength() + 1;
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -MaxUtility(); }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 2.0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }

 private:
  ScoringType scoring_type_;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_H_
