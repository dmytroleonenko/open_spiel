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
//                               (default) or "winlosstie_scoring"

namespace open_spiel {
namespace long_narde {

inline constexpr const int kNumPlayers = 2;
inline constexpr const int kNumChanceOutcomes = 21;
inline constexpr const int kNumPoints = 24;
inline constexpr const int kNumDiceOutcomes = 6;
inline constexpr const int kXPlayerId = 0;  // White player
inline constexpr const int kOPlayerId = 1;  // Black player
inline constexpr const int kPassPos = -1;

// Define the die value to use for pass moves
inline constexpr const int kPassDieValue = 1;

// Move CheckerMove struct definition before its usage
struct CheckerMove {
  // Pass is encoded as pos = -1 (kPassPos)
  int pos;      // Valid board locations: 0-23; -1 represents a pass.
  int to_pos;   // Destination position (or -1 for pass)
  int die;      // Die value used (1-6, or -1 for pass)
  
  // Default constructor
  constexpr CheckerMove() 
      : pos(kPassPos), to_pos(kPassPos), die(kPassDieValue) {}
  
  // Constructor
  constexpr CheckerMove(int _pos, int _to_pos, int _die) 
      : pos(_pos), to_pos(_to_pos), die(_die) {}
  
  // Legacy constructor for compatibility
  constexpr CheckerMove(int _pos, int _die) 
      : pos(_pos), to_pos(-1), die(_die) {}
  
  bool operator<(const CheckerMove& rhs) const {
    if (pos != rhs.pos) return pos < rhs.pos;
    if (to_pos != rhs.to_pos) return to_pos < rhs.to_pos;
    return die < rhs.die;
  }
};

// Constant pass move to avoid repeated construction
inline constexpr const CheckerMove kPassMove(kPassPos, kPassPos, kPassDieValue);

// Constant vector of two pass moves
inline const std::vector<CheckerMove> kDoublePassMove = {kPassMove, kPassMove};

// Number of checkers per player
inline constexpr const int kNumCheckersPerPlayer = 15;

// Debugging flag
inline constexpr const bool kDebugging = false;

// Head positions for each player
inline constexpr const int kWhiteHeadPos = 23;  // Point 24 (0-indexed)
inline constexpr const int kBlackHeadPos = 11;  // Point 12 (0-indexed)

// Home regions for each player
inline constexpr const int kWhiteHomeStart = 0;   // Point 1 (0-indexed)
inline constexpr const int kWhiteHomeEnd = 5;     // Point 6 (0-indexed)
inline constexpr const int kBlackHomeStart = 12;  // Point 13 (0-indexed)
inline constexpr const int kBlackHomeEnd = 17;    // Point 18 (0-indexed)

inline constexpr const int kScorePos = 101;  // Special sentinel value for scored checkers
inline constexpr const int kBearOffPos = -1; // Canonical value used internally for bearing off. Note: NOT the same as kPassPos.
inline constexpr const int kNumNonDoubleOutcomes = 15; // Number of non-double dice outcomes (e.g., 1-2, 1-3, ..., 5-6)

// Action Encoding Space:
// The action encoding aims to represent all possible moves within a single integer.
// However, the current constant kNumDistinctActions (1250) seems inconsistent with
// the actual encoding logic implemented in long_narde.cc, which uses:
// 1. A base-150 system (`kDigitBase`) for combining two half-moves in non-double scenarios.
//    (Max action value roughly 2 * 150 * 150 = 45000)
// 2. A base-25 system (`kEncodingBaseDouble`) for encoding up to 4 checker positions during doubles.
//    (Max action value depends on kDoublesOffset + 25^4)
// The NumDistinctActions() method calculates the true maximum based on the implementation.
// This constant might be legacy or require revision.
inline constexpr const int kNumDistinctActions = 1250;  // Potential maximum number of distinct actions. See comment above.

// Base used to combine two half-move "digits" in the non-doubles encoding scheme.
// Must be >= 150 to accommodate the max digit value (149).
// inline constexpr const int kDigitBase = 150; // Moved to long_narde.cc

// Since Long Narde doesn't have hitting, we only need to track:
// 1) If a point is occupied (and how many checkers)
// The simplified encoding uses 1 value per point per player
inline constexpr const int kBoardEncodingSize = kNumPoints * kNumPlayers;

// The state encoding size includes:
// - Board encoding: kBoardEncodingSize
// - Scores for each player: 2 (1 per player)
// - Current player indicator: 2 (1 per player)
// - Dice values: 2
inline constexpr const int kStateEncodingSize =
    2 * kNumPlayers + kBoardEncodingSize + 2;
inline constexpr const char* kDefaultScoringType = "winloss_scoring";

// Game scoring type, whether to allow final black move for potential tie
enum class ScoringType {
  kWinLossScoring,    // "winloss_scoring": Standard scoring without final black move
  kWinLossTieScoring  // "winlosstie_scoring": Allows black one last move to try for tie
};

// Add ParseScoringType declaration before its usage
ScoringType ParseScoringType(const std::string& st_str);

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  int player;
  int prev_player;
  std::vector<int> dice;
  Action action;
  bool double_turn;
  bool is_first_turn;
  bool moved_from_head;
  bool is_playing_extra_turn;  // Added: tracks if this turn was an extra turn
  TurnHistoryInfo(int _player, int _prev_player, std::vector<int> _dice,
                  int _action, bool _double_turn,
                  bool _is_first_turn, bool _moved_from_head,
                  bool _is_playing_extra_turn)  // Added parameter
      : player(_player),
        prev_player(_prev_player),
        dice(_dice),
        action(_action),
        double_turn(_double_turn),
        is_first_turn(_is_first_turn),
        moved_from_head(_moved_from_head),
        is_playing_extra_turn(_is_playing_extra_turn) {}  // Added initialization
};

class LongNardeGame;

class LongNardeState : public State {
 public:
  LongNardeState(const LongNardeState&) = default;
  LongNardeState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  virtual int NumDistinctActions() const;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // Sets the game state
  void SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

  // Returns the opponent of the specified player.
  int Opponent(int player) const;

  // Is this position off the board, i.e. >23 or <0?
  bool IsOff(int player, int pos) const;

  // Get the To position for this play given the from position and number of
  // pips on the die. This function simply adds the values: the return value
  // will be a position that might be off the the board (<0 or >23).
  int GetToPos(int player, int from_pos, int pips) const;

  // Count the total number of checkers for this player (on the board
  // and have borne off). Should be 15 for the standard game.
  int CountTotalCheckers(int player) const;

  // Accessor functions for some of the specific data.
  int player_turns() const { return turns_; }
  int player_turns(int player) const {
    return (player == kXPlayerId ? x_turns_ : o_turns_);
  }
  int score(int player) const { return scores_[player]; }
  int dice(int i) const { return dice_[i]; }
  bool double_turn() const { return double_turn_; }
  bool is_first_turn() const { return is_first_turn_; }
  bool moved_from_head() const { return moved_from_head_; }

  // Get the number of checkers on the board in the specified position belonging
  // to the specified player. The position can be kScorePos, but use score() to get the number
  // of checkers born off.
  int board(int player, int pos) const;

  // Check if a position is in the home area of the player
  bool IsPosInHome(int player, int pos) const;

  // Action encoding / decoding functions. Note, the converted checker moves
  // do not contain the hit information; use the AddHitInfo function to get the
  // hit information.
  Action CheckerMovesToSpielMove(const std::vector<CheckerMove>& moves) const;
  std::vector<CheckerMove> SpielMoveToCheckerMoves(Player player,
                                                   Action spiel_move) const;
  Action TranslateAction(int from1, int from2, bool use_high_die_first) const;

  bool WouldFormBlockingBridge(int player, int from_pos, int to_pos) const;
  bool IsHeadPos(int player, int pos) const;
  bool IsLegalHeadMove(int player, int from_pos) const;
  bool IsFirstTurn(int player) const;
  bool& MutableIsFirstTurn() { return is_first_turn_; }

  // Centralized function to check if a checker move is valid
  bool IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule = true) const;

  // Returns the position of the furthest checker in the home of this player.
  // Returns -1 if none found.
  int FurthestCheckerInHome(int player) const;

  void ApplyCheckerMove(int player, const CheckerMove& move);
  void UndoCheckerMove(int player, const CheckerMove& move);

  bool UsableDiceOutcome(int outcome) const;
  std::vector<Action> ProcessLegalMoves(int max_moves,
                                      const std::set<std::vector<CheckerMove>>& movelist) const;

  // Tests if a bridge (illegal formation) would be created by applying a move.
  // Returns true if a bridge would be formed, false otherwise.
  bool WouldFormBridge(Player player, int from_pos, int to_pos) const;

  // Validate that an action is legal and decodes to valid moves
  bool ValidateAction(Action action) const;

  // Returns all illegal actions for the given board state.
  std::vector<Action> IllegalActions() const;

  // Process a chance roll (dice roll) action.
  void ProcessChanceRoll(Action move_id);

  bool AllInHome(int player) const;

  // Checks if the current board state contains an illegal bridge for the player.
  bool HasIllegalBridge(int player) const;
  
  // Helper function to find the real position index within a block
  // that is encountered first on a given player's path.
  int GetBlockPathStartRealPos(int player_for_path, int block_lowest_real_idx) const;

  // Generate all possible half-moves from the current state
  std::set<CheckerMove> GenerateAllHalfMoves(int player) const;

  // Helper function: checks if 'player' has any checker in [startPos, endPos] inclusive.
  bool HasAnyChecker(int player, int startPos, int endPos) const;

  // Translates a real board position to a virtual coordinate for path comparison.
  int GetVirtualCoords(int player, int real_pos) const;

  // Get the 0-based index along the player's path
  int GetPathIndex(int player, int real_pos) const;

  // Check if checker_pos is ahead of reference_pos on player's path
  bool IsAhead(int player, int checker_pos, int reference_pos) const;

  // Directly expose board_ for testing/debugging
  std::vector<std::vector<int>> board_;  // Checkers for each player on points.
  std::vector<int> dice_; // Current dice roll.
  std::vector<int> scores_; // Number of checkers borne off by each player.
  Player cur_player_; // Player whose turn it is.
  bool is_first_turn_;
  bool moved_from_head_;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);
  int CheckersInHome(int player) const;
  int NumOppCheckers(int player, int pos) const;
  std::string DiceToString(int outcome) const;
  int DiceValue(int i) const;
  int HighestUsableDiceOutcome() const;

  // A helper function used by ActionToString to compute the end position
  // of a move and determine whether it goes off the board.
  int GetMoveEndPosition(CheckerMove* cmove, int player, int start) const;

  std::set<CheckerMove> LegalCheckerMoves(int player) const;
  int RecLegalMoves(const std::vector<CheckerMove>& moveseq,
                    std::set<std::vector<CheckerMove>>* movelist,
                    int max_depth = 10);

  ScoringType scoring_type_;  // Which rules apply when scoring the game.

  Player prev_player_;
  int turns_;
  int x_turns_;
  int o_turns_;
  bool double_turn_;
  bool is_playing_extra_turn_; // Added: tracks if current turn is an extra turn from doubles
  std::vector<int> initial_dice_; // Dice rolled at the start of the current player's turn.
  std::vector<TurnHistoryInfo> turn_history_info_;  // Info needed for Undo.
  bool allow_last_roll_tie_;  // Tracks if a last roll for tie is allowed.

  // Helper function to generate all valid move sequences.
  std::set<std::vector<CheckerMove>> GenerateMoveSequences(
      Player player, int max_moves) const;

  // Helper function to filter generated sequences for the best ones
  // (longest sequence length, max non-pass moves within that length).
  // Returns the filtered list and the calculated max_non_pass count.
  std::pair<std::set<std::vector<CheckerMove>>, int> FilterBestMoveSequences(
      const std::set<std::vector<CheckerMove>>& movelist) const;

  // Helper function to apply the "play higher die" rule if necessary.
  std::vector<Action> ApplyHigherDieRuleIfNeeded(
      const std::vector<Action>& current_legal_moves,
      const std::set<std::vector<CheckerMove>>& original_movelist) const;

  friend class LongNardeGame;

  // Friend declarations for test helper functions
  friend void SetupBoardState(LongNardeState* state, Player player,
                              const std::vector<std::vector<int>>& board_config,
                              const std::vector<int>& scores);
  friend void SetupDice(LongNardeState* state, const std::vector<int>& dice,
                        bool double_turn);
};

// Add an overload for the << operator for ScoringType to fix compilation errors
inline std::ostream& operator<<(std::ostream& os, const ScoringType& type) {
  switch (type) {
    case ScoringType::kWinLossScoring:
      os << "kWinLossScoring";
      break;
    case ScoringType::kWinLossTieScoring:
      os << "kWinLossTieScoring";
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
        new LongNardeState(shared_from_this()));
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
