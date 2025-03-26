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

  // Order the input moves according to the encoding convention:
  // The move corresponding to the higher die value should be encoded first (dig0).
  // If dice values are equal (doubles or single move), order doesn't matter.
  // Pass moves are treated specially.
  std::vector<CheckerMove> ordered_moves = moves;
  if (ordered_moves.size() >= 2) {
    bool move0_is_pass = (ordered_moves[0].pos == kPassPos);
    bool move1_is_pass = (ordered_moves[1].pos == kPassPos);

    // If one is a pass and the other isn't, the non-pass move effectively uses the "higher" die slot (dig0)
    // because pass encodings (>=144) are higher than regular move encodings (0-143).
    // The encoding logic later handles putting the larger digit (dig1) in the higher base.
    // Let's simplify: Always encode moves[0] as dig0 and moves[1] as dig1 initially.
    // The base-kDigitBase encoding handles the order.

    // However, the ActionToString and validation might rely on a specific order.
    // Let's stick to the original intent: order by die value for encoding.
    if (!move0_is_pass && !move1_is_pass) {
        // Both regular moves: ensure move with higher die is first in ordered_moves
        if (ordered_moves[0].die < ordered_moves[1].die) {
            std::swap(ordered_moves[0], ordered_moves[1]);
        }
    } else if (move0_is_pass && !move1_is_pass) {
        // Move 0 is pass, Move 1 is regular. Swap so regular move (lower encoding value) is first.
         std::swap(ordered_moves[0], ordered_moves[1]);
         // Now ordered_moves[0] is the regular move, ordered_moves[1] is the pass.
    }
    // else: move1 is pass, or both are passes. Keep original order.
    // If both passes, order doesn't matter unless dice values differ, then higher die first.
    else if (move0_is_pass && move1_is_pass) {
        if (ordered_moves[0].die < ordered_moves[1].die) {
             std::swap(ordered_moves[0], ordered_moves[1]);
        }
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

  // Encode the (potentially reordered) moves. ordered_moves[0] uses the higher die value (or is the non-pass move).
  int dig0 = ordered_moves.empty() ? kPassOffset : encode_move(ordered_moves[0]);
  int dig1 = (ordered_moves.size() > 1) ? encode_move(ordered_moves[1]) : kPassOffset;

  // The action is encoded with the second move's digit (lower die or pass) in the higher base position.
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
  bool is_double_pass = (ordered_moves.size() == 2 && ordered_moves[0].pos == kPassPos && ordered_moves[1].pos == kPassPos);
  if (actual_low_roll_first && !is_double_pass) {
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
  bool is_debugging = false; // General debug flag
  // Specific debug for the failing bridge test case (White, 4->3)
  bool specific_debug = (player == kXPlayerId && from_pos == 4 && to_pos == 3);

  std::vector<std::vector<int>> temp_board = board_;
  if (from_pos >= 0 && from_pos < kNumPoints) {
    // Ensure we don't decrement below zero, although this shouldn't happen for valid moves
    if (temp_board[player][from_pos] > 0) temp_board[player][from_pos]--;
  }
  // Check if to_pos is valid before incrementing
  if (to_pos >= 0 && to_pos < kNumPoints) {
    temp_board[player][to_pos]++;
  } else {
     // If moving off the board, it cannot form a bridge on the board
    return false;
  }

  int consecutive_count = 0;
  int start_pos, end_pos, direction;

  // Determine iteration direction based on player
  if (player == kXPlayerId) {
    start_pos = 0; end_pos = kNumPoints; direction = 1;
  } else {
    start_pos = kNumPoints - 1; end_pos = -1; direction = -1;
  }

  for (int pos = start_pos; pos != end_pos; pos += direction) {
    if (temp_board[player][pos] > 0) {
      consecutive_count++;
    } else {
      consecutive_count = 0; // Reset count if gap found
      // Continue checking the rest of the board for other potential bridges
      continue;
    }

    // Check if a 6-block is formed
    if (consecutive_count >= 6) { // Use >= 6 for safety, though exactly 6 is the rule trigger
      bool any_opponent_ahead = false;
      int opponent = Opponent(player);
      int bridge_end = pos; // The index where the 6th consecutive checker was found

      if (specific_debug) {
          std::cout << "DEBUG WouldFormBlockingBridge(X, 4->3): Found 6-block ending at " << bridge_end << ". Checking opponent ahead..." << std::endl;
      }

      // Check if any opponent checker is ahead of the bridge
      if (player == kXPlayerId) {
        // For White, 'ahead' means higher indices (towards Black's side)
        for (int i = bridge_end + 1; i < kNumPoints; i++) {
          if (board(opponent, i) > 0) { // Check ORIGINAL board_
             if (specific_debug) {
                 std::cout << "  -> Opponent found at index " << i << " (Value: " << board(opponent, i) << "). Bridge is LEGAL." << std::endl;
             }
            any_opponent_ahead = true;
            break;
          }
        }
      } else { // player == kOPlayerId
        // For Black, 'ahead' means lower indices (towards White's side)
        for (int i = 0; i < bridge_end; i++) {
          if (board(opponent, i) > 0) { // Check ORIGINAL board_
             if (specific_debug) {
                 std::cout << "  -> Opponent found at index " << i << " (Value: " << board(opponent, i) << "). Bridge is LEGAL." << std::endl;
             }
            any_opponent_ahead = true;
            break;
          }
        }
      }

      // If a 6-block was found AND no opponent was ahead, it's an illegal bridge
      if (!any_opponent_ahead) {
         if (specific_debug) {
             std::cout << "  -> NO Opponent found ahead. Bridge is ILLEGAL." << std::endl;
         }
        return true; // Illegal bridge detected
      }
      // If an opponent was found ahead, this specific 6-block is legal.
      // Reset count to potentially find other illegal bridges starting after this one.
      // Example: [1,1,1,1,1,1, 0, 1,1,1,1,1,1] - first block is legal if opponent ahead,
      // but second block might still be illegal.
      // However, the rule is usually interpreted as *any* 6-block trapping all opponents is illegal.
      // Let's stick to returning true immediately if an illegal one is found.
      // If this block was legal (opponent ahead), we continue the outer loop.
      // We need to reset consecutive_count here? No, the outer loop continues from pos+1.
      // If the next pos also has a checker, consecutive_count becomes 7, which is fine.
    }
  } // End loop through positions

  // If loop completes without returning true, no illegal bridge was formed
  if (specific_debug) {
      std::cout << "DEBUG WouldFormBlockingBridge(X, 4->3): No illegal bridge found." << std::endl;
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
  if (specific_debug) {
    std::cout << "DEBUG IsValidCheckerMove (O: 11->12): Checking opponent occupancy at to_pos=" << to_pos << std::endl;
    std::cout << "  board(X, " << to_pos << ") = " << board(1-player, to_pos) << std::endl;
    // Also print neighbors for context
    std::cout << "  Context: O[" << board(player, 11) << "," << board(player, 12) << "," << board(player, 13) << "] "
              << "X[" << board(1-player, 11) << "," << board(1-player, 12) << "," << board(1-player, 13) << "]" << std::endl;
  }
  if (board(1-player, to_pos) > 0) {
    if (is_debugging || specific_debug) std::cout << "  -> FAILED: Opponent found at " << to_pos << std::endl;
    return false;
  }
  // *** End Critical Check ***

  // REMOVED: Bridge check is deferred to LegalActions filtering
  // if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
  //  if (is_debugging || specific_debug) std::cout << "DEBUG IsValidCheckerMove: Would form illegal blocking bridge" << std::endl;
  //  return false;
  // }
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
        bool is_special_double = (original_moves.size() == 2 &&
                                  original_moves[0].die == original_moves[1].die &&
                                  (original_moves[0].die == 6 ||
                                   original_moves[0].die == 4 ||
                                   original_moves[0].die == 3));
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

  // Get valid single moves from the current state.
  // LegalCheckerMoves uses IsValidCheckerMove(..., check_head_rule=true),
  // considering the current state including is_first_turn_ and moved_from_head_.
  std::set<CheckerMove> moves_here = LegalCheckerMoves(cur_player_);

  // Base case 2: No moves possible from this state (player must pass/stop).
  if (moves_here.empty()) {
    // Add the sequence found so far. It represents a valid end-point for moves.
    // If moveseq is empty, it means pass was the only option from the start.
    // Only insert if it's not already present to avoid duplicates if multiple paths lead here.
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

  for (const auto& move : moves_here) {
    // Check limits before processing the move
    if (movelist->size() >= kSafeLimit / 2) return max_len_found; // Early exit if close to limit
    if (moves_checked >= kMaxMovesToCheck) break; // Limit branching
    moves_checked++;

    // --- Try applying the move and recursing ---
    new_moveseq.push_back(move);
    bool old_moved_from_head = moved_from_head_; // Save state part not handled by Undo
    ApplyCheckerMove(cur_player_, move);

    // *** Check for momentary illegal bridge ***
    if (HasIllegalBridge(cur_player_)) {
        // This move created an illegal bridge. This path is invalid.
        // Backtrack the move and continue to the next possible move from the *previous* state.
        UndoCheckerMove(cur_player_, move);
        moved_from_head_ = old_moved_from_head;
        new_moveseq.pop_back(); // Remove the invalid move from the sequence being built
        continue; // Skip the recursive call for this invalid path
    }
    // *** END CHECK ***


    // Check limit again before recursion call
    if (movelist->size() >= kSafeLimit / 2) {
      UndoCheckerMove(cur_player_, move); // Backtrack state
      moved_from_head_ = old_moved_from_head;
      // Don't pop new_moveseq as we are returning early
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
    // --- End of move processing ---
  }

  // After trying all moves from this state, if no valid sequence was extended
  // (e.g., all moves led to bridges or dead ends), we might need to add the
  // current sequence 'moveseq' if it represents a valid stopping point.
  // This is handled by the base case check `if (moves_here.empty())` at the start
  // and implicitly by returning max_len_found.

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

  // Use a mutable copy ONLY for RecLegalMoves state modifications
  auto state_copy = std::make_unique<LongNardeState>(*this);
  std::set<std::vector<CheckerMove>> movelist;

  int max_possible_moves = 0;
  // Use state_copy->dice_ as RecLegalMoves modifies the dice in the copy
  if (state_copy->dice_.size() == 2 && state_copy->DiceValue(0) == state_copy->DiceValue(1)) {
      max_possible_moves = 4; // Doubles potentially use 4 dice
  } else {
      // Count usable dice (value 1-6)
      for(int outcome : state_copy->dice_) {
          if (state_copy->UsableDiceOutcome(outcome)) {
              max_possible_moves++;
          }
      }
  }
   // Ensure max_possible_moves is not negative if dice_ is empty (shouldn't happen in player turn)
   if (max_possible_moves < 0) max_possible_moves = 0;

  // Generate all possible move sequences up to the max possible dice usage
  // RecLegalMoves now prunes sequences that create momentary illegal bridges
  state_copy->RecLegalMoves({}, &movelist, max_possible_moves);

  std::vector<Action> legal_moves;
  int longest_sequence = 0;

  // If movelist is empty after RecLegalMoves (no moves possible or all pruned), it's a pass.
  // Check if the only entry is the empty sequence (inserted by base case 2 in RecLegalMoves)
  bool forced_pass = movelist.empty() || (movelist.size() == 1 && movelist.count({}) > 0);

  if (!forced_pass) {
      // Find the length of the longest actual move sequence found
      for (const auto& moveseq : movelist) {
          // Ignore the empty sequence if it exists alongside actual moves
          if (!moveseq.empty()) {
             longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
          }
      }
  }
  // If longest_sequence is still 0, it's a forced pass.

  // Filter sequences: only keep those matching the longest possible length
  for (const auto& moveseq : movelist) {
    // Only consider sequences that match the longest length found.
    // An empty sequence is only valid if longest_sequence is 0 (forced pass).
    if (moveseq.size() == longest_sequence) {
        // Skip adding the empty sequence representation unless it's a forced pass
        if (longest_sequence == 0 && !moveseq.empty()) continue;
        // If longest_sequence > 0, skip the empty sequence if it's present
        if (longest_sequence > 0 && moveseq.empty()) continue;

        // *** NEW: Verify this sequence doesn't create any bridge violations ***
        // Create a clean copy of the current state to verify the sequence
        auto verify_state = std::make_unique<LongNardeState>(*this);
        bool sequence_valid = true;
        
        // Apply each move in the sequence and check for bridge violations
        for (const auto& move : moveseq) {
            if (move.pos == kPassPos) continue; // Skip pass moves
            
            // Apply the move
            verify_state->ApplyCheckerMove(verify_state->cur_player_, move);
            
            // Check if this creates an illegal bridge
            if (verify_state->HasIllegalBridge(verify_state->cur_player_)) {
                sequence_valid = false;
                break;
            }
        }
        
        // Only add sequences that pass the final verification
        if (sequence_valid) {
            // If sequence is valid (correct length and no bridge violations), add it.
            // Use the original state (*this*) to encode the move.
            legal_moves.push_back(CheckerMovesToSpielMove(moveseq));
        }
        // *** END NEW ***
    }
  }

  // If after filtering, legal_moves is empty, it implies a forced pass.
  // This handles cases where movelist was initially empty or only contained shorter sequences.
  if (legal_moves.empty()) {
    std::vector<CheckerMove> pass_move_seq;
    // Use original state's dice here
    if (!dice_.empty()) {
        int die1 = 1, die2 = 1;
        int usable_count = 0;
        if (UsableDiceOutcome(dice_[0])) { die1 = DiceValue(0); usable_count++; } // Use DiceValue
        if (dice_.size() > 1 && UsableDiceOutcome(dice_[1])) {
            die2 = DiceValue(1); usable_count++; // Use DiceValue
            if (usable_count == 1) die1 = die2;
        } else if (usable_count == 1) {
            die2 = die1;
        }
        pass_move_seq.push_back({kPassPos, kPassPos, die1});
        pass_move_seq.push_back({kPassPos, kPassPos, die2});
    } else {
         pass_move_seq.push_back({kPassPos, kPassPos, 1});
         pass_move_seq.push_back({kPassPos, kPassPos, 1});
    }
    legal_moves.push_back(CheckerMovesToSpielMove(pass_move_seq));
  }

  // Deduplicate final list
  if (legal_moves.size() > 1) {
      std::sort(legal_moves.begin(), legal_moves.end());
      auto last = std::unique(legal_moves.begin(), legal_moves.end());
      legal_moves.erase(last, legal_moves.end());
  }

  return legal_moves;
}


// *** NEW HELPER FUNCTION: HasIllegalBridge ***
// Checks the current board state for an illegal bridge for the given player.
bool LongNardeState::HasIllegalBridge(int player) const {
    // Add specific debug for the failing bridge test case (White, 4->3 context)
    bool specific_debug = (player == kXPlayerId); // Broaden debug for X player

    if (specific_debug) {
        std::cout << "DEBUG HasIllegalBridge(Player " << player << ") called. Current Board:" << std::endl;
        // Print relevant part of the board for White (indices 0-7)
        std::cout << "  X: [";
        for(int i=0; i<=7; ++i) std::cout << board(kXPlayerId, i) << (i==7 ? "" : ",");
        std::cout << "]" << std::endl;
        std::cout << "  O: [";
        for(int i=0; i<=7; ++i) std::cout << board(kOPlayerId, i) << (i==7 ? "" : ",");
        std::cout << "]" << std::endl;
    }

    int consecutive_count = 0;
    int start_pos, end_pos, direction;
    int opponent = Opponent(player);

    if (player == kXPlayerId) {
        start_pos = 0; end_pos = kNumPoints; direction = 1;
    } else {
        start_pos = kNumPoints - 1; end_pos = -1; direction = -1;
    }

    for (int pos = start_pos; pos != end_pos; pos += direction) {
        if (board(player, pos) > 0) { // Check CURRENT board state
            consecutive_count++;
        } else {
            consecutive_count = 0;
            continue;
        }

        if (consecutive_count >= 6) {
            bool any_opponent_ahead = false;
            int bridge_end = pos; // Index where the 6th checker is found

            if (specific_debug) {
                std::cout << "  -> Found 6-block ending at index " << bridge_end << ". Checking opponent ahead..." << std::endl;
            }

            if (player == kXPlayerId) {
                // For White, 'ahead' means higher indices
                for (int i = bridge_end + 1; i < kNumPoints; i++) {
                    // *** IMPORTANT: Check ORIGINAL board state for opponent position ***
                    // This assumes 'board_' member holds the state *before* the sequence began.
                    // However, HasIllegalBridge is called *after* ApplyCheckerMove,
                    // so 'board_' reflects the *temporary* state.
                    // The rule implies checking opponent position relative to the potential block.
                    // Let's stick to checking the current state's opponent position.
                    if (board(opponent, i) > 0) { // Check CURRENT board state
                         if (specific_debug) {
                             std::cout << "    -> Opponent found ahead at index " << i << " (Value: " << board(opponent, i) << "). Bridge is LEGAL." << std::endl;
                         }
                        any_opponent_ahead = true;
                        break;
                    }
                }
            } else { // kOPlayerId
                // For Black, 'ahead' means lower indices
                for (int i = 0; i < bridge_end; i++) { // Check indices before the bridge end
                    if (board(opponent, i) > 0) { // Check CURRENT board state
                         if (specific_debug) {
                             std::cout << "    -> Opponent found ahead at index " << i << " (Value: " << board(opponent, i) << "). Bridge is LEGAL." << std::endl;
                         }
                        any_opponent_ahead = true;
                        break;
                    }
                }
            }

            if (!any_opponent_ahead) {
                 if (specific_debug) {
                     std::cout << "    -> NO Opponent found ahead. Bridge is ILLEGAL." << std::endl;
                 }
                return true; // Found an illegal bridge
            }
            // If this 6-block was legal, continue checking the rest of the board
        }
    }

    if (specific_debug) {
        std::cout << "  -> No illegal bridge found in this state." << std::endl;
    }
    return false; // No illegal bridge found
}

}  // namespace long_narde
}  // namespace open_spiel