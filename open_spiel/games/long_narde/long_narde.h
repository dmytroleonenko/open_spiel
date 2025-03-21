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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of Long Narde, a variant of backgammon.
// See game rules in the README.md or in the code comments.
//
// Parameters:
//   "scoring_type"      string  Type of scoring for the game: "winloss_scoring"
//                               (default), "enable_gammons", or "full_scoring"

namespace open_spiel {
namespace long_narde {

inline constexpr const int kNumPlayers = 2;
inline constexpr const int kNumChanceOutcomes = 21;
inline constexpr const int kNumPoints = 24;
inline constexpr const int kNumDiceOutcomes = 6;
inline constexpr const int kXPlayerId = 0;  // White player
inline constexpr const int kOPlayerId = 1;  // Black player
inline constexpr const int kPassPos = -1;

// Number of checkers per player
inline constexpr const int kNumCheckersPerPlayer = 15;

// Head positions for each player
inline constexpr const int kWhiteHeadPos = 23;  // Point 24 (0-indexed)
inline constexpr const int kBlackHeadPos = 11;  // Point 12 (0-indexed)

// Home regions for each player
inline constexpr const int kWhiteHomeStart = 0;   // Point 1 (0-indexed)
inline constexpr const int kWhiteHomeEnd = 5;     // Point 6 (0-indexed)
inline constexpr const int kBlackHomeStart = 12;  // Point 13 (0-indexed)
inline constexpr const int kBlackHomeEnd = 17;    // Point 18 (0-indexed)

inline constexpr const int kBarPos = 100;
inline constexpr const int kScorePos = 101;

// The action encoding stores a number in { 0, 1, ..., 1351 }. If the high
// roll is to move first, then the number is encoded as a 2-digit number in
// base 26 ({0, 1, .., 23, kBarPos, Pass}) (=> first 676 numbers). Otherwise,
// the low die is to move first and, 676 is subtracted and then again the
// number is encoded as a 2-digit number in base 26.
inline constexpr const int kNumDistinctActions = 1352;

// See ObservationTensorShape for details.
inline constexpr const int kBoardEncodingSize = 4 * kNumPoints * kNumPlayers;
inline constexpr const int kStateEncodingSize =
    3 * kNumPlayers + kBoardEncodingSize + 2;
inline constexpr const char* kDefaultScoringType = "winloss_scoring";

// Game scoring type, whether to score gammons/backgammons specially.
enum class ScoringType {
  kWinLossScoring,  // "winloss_scoring": Score only 1 point per player win.
  kEnableGammons,   // "enable_gammons": Score 2 points for a "gammon".
  kFullScoring,     // "full_scoring": Score gammons as well as 3 points for a
                    // "backgammon".
};

struct CheckerMove {
  // Pass is encoded as (pos, num, hit) = (-1, -1, false).
  int pos;  // 0-24  (0-23 for locations on the board and kBarPos)
  int num;  // 1-6
  bool hit;
  CheckerMove(int _pos, int _num, bool _hit)
      : pos(_pos), num(_num), hit(_hit) {}
  bool operator<(const CheckerMove& rhs) const {
    return (pos * 6 + (num - 1)) < (rhs.pos * 6 + rhs.num - 1);
  }
};

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  int player;
  int prev_player;
  std::vector<int> dice;
  Action action;
  bool double_turn;
  bool first_move_hit;
  bool second_move_hit;
  // Added for Long Narde head rule tracking
  bool is_first_turn;
  bool moved_from_head;
  TurnHistoryInfo(int _player, int _prev_player, std::vector<int> _dice,
                  int _action, bool _double_turn, bool fmh, bool smh,
                  bool _is_first_turn, bool _moved_from_head)
      : player(_player),
        prev_player(_prev_player),
        dice(_dice),
        action(_action),
        double_turn(_double_turn),
        first_move_hit(fmh),
        second_move_hit(smh),
        is_first_turn(_is_first_turn),
        moved_from_head(_moved_from_head) {}
};

class LongNardeGame;

class LongNardeState : public State {
 public:
  LongNardeState(const LongNardeState&) = default;
  LongNardeState(std::shared_ptr<const Game>, ScoringType scoring_type);

  Player CurrentPlayer() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // Setter function used for debugging and tests. Note: this does not set the
  // historical information properly, so Undo likely will not work on states
  // set this way!
  void SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                const std::vector<int>& bar, const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

  // Returns the opponent of the specified player.
  int Opponent(int player) const;

  // Compute a distance between 'from' and 'to'. The from can be kBarPos. The
  // to can be a number below 0 or above 23, but do not use kScorePos directly.
  int GetDistance(int player, int from, int to) const;

  // Is this position off the board, i.e. >23 or <0?
  bool IsOff(int player, int pos) const;

  // Returns whether pos2 is further (closer to scoring) than pos1 for the
  // specifed player.
  bool IsFurther(int player, int pos1, int pos2) const;

  // Is this a legal from -> to checker move? Here, the to_pos can be a number
  // that is outside {0, ..., 23}; if so, it is counted as "off the board" for
  // the corresponding player (i.e. >23 is a bear-off move for XPlayerId, and
  // <0 is a bear-off move for OPlayerId).
  bool IsLegalFromTo(int player, int from_pos, int to_pos, int my_checkers_from,
                     int opp_checkers_to) const;

  // Get the To position for this play given the from position and number of
  // pips on the die. This function simply adds the values: the return value
  // will be a position that might be off the the board (<0 or >23).
  int GetToPos(int player, int from_pos, int pips) const;

  // Count the total number of checkers for this player (on the board, in the
  // bar, and have borne off). Should be 15 for the standard game.
  int CountTotalCheckers(int player) const;

  // Returns if moving from the position for the number of spaces is a hit.
  bool IsHit(Player player, int from_pos, int num) const;

  // Accessor functions for some of the specific data.
  int player_turns() const { return turns_; }
  int player_turns(int player) const {
    return (player == kXPlayerId ? x_turns_ : o_turns_);
  }
  int bar(int player) const { return bar_[player]; }
  int score(int player) const { return scores_[player]; }
  int dice(int i) const { return dice_[i]; }
  bool double_turn() const { return double_turn_; }
  bool is_first_turn() const { return is_first_turn_; }
  bool moved_from_head() const { return moved_from_head_; }

  // Get player's head position
  int GetHeadPos(int player) const {
    return (player == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos);
  }

  // Get the number of checkers on the board in the specified position belonging
  // to the specified player. The position can be kBarPos or any valid position
  // on the main part of the board, but kScorePos (use score() to get the number
  // of checkers born off).
  int board(int player, int pos) const;

  // Check if a position is in the home area of the player
  bool IsPosInHome(int player, int pos) const;

  // Action encoding / decoding functions. Note, the converted checker moves
  // do not contain the hit information; use the AddHitInfo function to get the
  // hit information.
  Action CheckerMovesToSpielMove(const std::vector<CheckerMove>& moves) const;
  std::vector<CheckerMove> SpielMoveToCheckerMoves(int player,
                                                   Action spiel_move) const;
  Action TranslateAction(int from1, int from2, bool use_high_die_first) const;

  // Return checker moves with extra hit information.
  std::vector<CheckerMove>
  AugmentWithHitInfo(Player player,
                     const std::vector<CheckerMove> &cmoves) const;

  // Long Narde specific functions
  bool WouldFormBlockingBridge(int player, int from_pos, int to_pos) const;
  bool IsHeadPos(int player, int pos) const;
  bool IsLegalHeadMove(int player, int from_pos) const;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);
  bool AllInHome(int player) const;
  int CheckersInHome(int player) const;
  bool UsableDiceOutcome(int outcome) const;
  int PositionFromBar(int player, int spaces) const;
  int PositionFrom(int player, int pos, int spaces) const;
  int NumOppCheckers(int player, int pos) const;
  std::string DiceToString(int outcome) const;
  int IsGammoned(int player) const;
  int IsBackgammoned(int player) const;
  int DiceValue(int i) const;
  int HighestUsableDiceOutcome() const;
  Action EncodedPassMove() const;
  Action EncodedBarMove() const;

  // A helper function used by ActionToString to add necessary hit information
  // and compute whether the move goes off the board.
  int AugmentCheckerMove(CheckerMove* cmove, int player, int start) const;

  // Returns the position of the furthest checker in the home of this player.
  // Returns -1 if none found.
  int FurthestCheckerInHome(int player) const;

  bool ApplyCheckerMove(int player, const CheckerMove& move);
  void UndoCheckerMove(int player, const CheckerMove& move);
  std::set<CheckerMove> LegalCheckerMoves(int player) const;
  int RecLegalMoves(std::vector<CheckerMove> moveseq,
                    std::set<std::vector<CheckerMove>>* movelist);
  std::vector<Action> ProcessLegalMoves(
      int max_moves, const std::set<std::vector<CheckerMove>>& movelist) const;

  ScoringType scoring_type_;  // Which rules apply when scoring the game.

  Player cur_player_;
  Player prev_player_;
  int turns_;
  int x_turns_;
  int o_turns_;
  bool double_turn_;
  bool is_first_turn_;        // Tracks if this is the first turn
  bool moved_from_head_;      // Tracks if a checker was moved from the head this turn
  std::vector<int> dice_;     // Current dice.
  std::vector<int> bar_;      // Checkers of each player in the bar.
  std::vector<int> scores_;   // Checkers returned home by each player.
  std::vector<std::vector<int>> board_;  // Checkers for each player on points.
  std::vector<TurnHistoryInfo> turn_history_info_;  // Info needed for Undo.
  bool allow_last_roll_tie_;  // Tracks if a last roll for tie is allowed.
};

// Add an overload for the << operator for ScoringType to fix compilation errors
inline std::ostream& operator<<(std::ostream& os, const ScoringType& type) {
  switch (type) {
    case ScoringType::kWinLossScoring:
      os << "kWinLossScoring";
      break;
    case ScoringType::kEnableGammons:
      os << "kEnableGammons";
      break;
    case ScoringType::kFullScoring:
      os << "kFullScoring";
      break;
  }
  return os;
}

class LongNardeGame : public Game {
 public:
  explicit LongNardeGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new LongNardeState(shared_from_this(), scoring_type_));
  }
  int MaxChanceOutcomes() const override { return 30; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -MaxUtility(); }
  double MaxUtility() const override;
  std::vector<int> ObservationTensorShape() const override {
    return {kStateEncodingSize};
  }
  int MaxGameLength() const override { return 1000; }
  int MaxChanceNodesInHistory() const override { return MaxGameLength() + 1; }

 private:
  ScoringType scoring_type_;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_H_
