#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include <string>
#include <vector>

namespace open_spiel {
namespace long_narde {

// ===== General Utility Functions =====

std::string PositionToString(int pos) {
  if (pos == kPassPos) return "Pass";
  SPIEL_CHECK_GE(pos, 0);
  SPIEL_CHECK_LT(pos, kNumPoints);
  return absl::StrCat(pos + 1);
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

std::string LongNardeState::ToString() const {
  std::vector<std::string> board_array = {
      "+-------------------------------------+", // Exactly matches expected width
      "|13 14 15 16 17 18| |19 20 21 22 23 24|", // No extra space before final |
      "|                 | |                 |", // Top checker row
      "|                 | |                 |", // Ownership row for double-digit piles
      "|                 | |                 |", // Spacer row
      "|                 | |                 |", // Bottom checker row
      "|12 11 10  9  8  7| |6  5  4  3  2  1 |", // Space between '1' and final |
      "+-------------------------------------+"  // Bottom border
  };
  const int PADDING_TOP = 2;    // Top checker row
  const int OWNERSHIP_ROW = 3;  // New row for double-digit ownership
  const int PADDING_BOT = 5;    // Bottom checker row (incremented from 4 to 5)
  const int BAR_COL = 19;       // Column index where the bar starts

  // Fill the board representation
  for (int player = 0; player < 2; ++player) {
      char symbol = (player == kXPlayerId ? 'X' : 'O');

      for (int pos_idx = 0; pos_idx < kNumPoints; ++pos_idx) {
          int count = board(player, pos_idx);
          if (count > 0) {
              int row;
              int col;

              // Determine row based on point index
              if (pos_idx >= 12) { // Points 13-24 go on top row
                  row = PADDING_TOP;
                  col = 1 + (pos_idx - 12) * 3;
              } else { // Points 1-12 go on bottom row
                  row = PADDING_BOT;
                  col = 1 + (11 - pos_idx) * 3;
              }

              // Adjust for bar separator
              if (col >= BAR_COL) col += 2; // Skip the "| |" bar (2 chars wide)

              // Place checker symbol and count
              if (count < 10) {
                  // Single-digit format: Keep "X1" or "O5" format
                  board_array[row][col] = symbol;
                  board_array[row][col + 1] = '0' + count;
              } else {
                  // Double-digit format: Show "15" on checker row
                  board_array[row][col] = '0' + (count / 10);
                  board_array[row][col + 1] = '0' + (count % 10);

                  // Show player symbol on ownership row
                  int ownership_row = (row == PADDING_TOP) ? OWNERSHIP_ROW : OWNERSHIP_ROW + 1;
                  board_array[ownership_row][col] = symbol;
              }
          }
      }
  }

  // Use single backslash for newline separator
  std::string board_str = absl::StrJoin(board_array, "\n") + "\n";

  // Add game state information
  absl::StrAppend(&board_str, "Turn: ");
  absl::StrAppend(&board_str, CurPlayerToString(cur_player_));
  if (cur_player_ != kChancePlayerId && cur_player_ != kTerminalPlayerId) {
      absl::StrAppend(&board_str, (is_first_turn_ ? " (First Turn)" : ""));
      absl::StrAppend(&board_str, (is_playing_extra_turn_ ? " (Extra Turn)" : ""));
  }
  absl::StrAppend(&board_str, "\n");

  absl::StrAppend(&board_str, "Dice: ");
  if (dice_.empty() && cur_player_ != kChancePlayerId) {
       absl::StrAppend(&board_str, "(None rolled yet)");
  } else if (dice_.empty() && cur_player_ == kChancePlayerId) {
       absl::StrAppend(&board_str, "(Waiting for roll)");
  } else {
      for (size_t i = 0; i < dice_.size(); ++i) {
        if (i > 0) absl::StrAppend(&board_str, " ");
        absl::StrAppend(&board_str, std::to_string(DiceValue(i)));
        if (!UsableDiceOutcome(dice_[i])) absl::StrAppend(&board_str, "(u)"); // Mark used dice
      }
  }
  absl::StrAppend(&board_str, "\n");

  absl::StrAppend(&board_str, "Scores: X (White): ", scores_[kXPlayerId]);
  absl::StrAppend(&board_str, ", O (Black): ", scores_[kOPlayerId], "\n");

  if (moved_from_head_) {
    absl::StrAppend(&board_str, "Status: Head checker moved this turn.\n");
  }
  if (double_turn_) {
     absl::StrAppend(&board_str, "Status: Next roll is for an extra turn.\n");
  }
   if (allow_last_roll_tie_) {
     absl::StrAppend(&board_str, "Status: Last roll tie attempt allowed.\n");
   }

  return board_str;
}

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
      // Starting roll - This logic seems specific to older backgammon setup, might need review for Long Narde
      const char* starter = (move_id < kNumNonDoubleOutcomes ? "X starts" : "O starts"); // kNumNonDoubleOutcomes needs to be correct
      Action outcome_id = move_id;
      if (outcome_id >= kNumNonDoubleOutcomes) { // Ensure this constant is valid
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
          // Only return "Pass" if *all* decoded moves are passes.
          return absl::StrCat(move_id, " - Pass");
      }
      // If some moves are passes and some aren't, don't print the pass part explicitly.
      continue;
    }

    any_move = true;
    int start_hr, end_hr;

    // Human-readable conversion: White 1-24, Black 1-24 (opposite direction)
    if (player == kOPlayerId) { // Black's perspective
       start_hr = move.pos <= 11 ? (12 - move.pos) : (36 - move.pos) ; // 11->1, 0->12, 23->13, 12->24
       if (IsOff(player, move.to_pos)) {
           end_hr = kNumOffPosHumanReadable; // "Off"
       } else {
            SPIEL_CHECK_GE(move.to_pos, 0);
            SPIEL_CHECK_LT(move.to_pos, kNumPoints);
            end_hr = move.to_pos <= 11 ? (12 - move.to_pos) : (36 - move.to_pos);
       }
    } else { // White's perspective (kXPlayerId)
       start_hr = 24 - move.pos; // 23->1, 0->24
        if (IsOff(player, move.to_pos)) {
            end_hr = kNumOffPosHumanReadable; // "Off"
        } else {
            SPIEL_CHECK_GE(move.to_pos, 0);
            SPIEL_CHECK_LT(move.to_pos, kNumPoints);
            end_hr = 24 - move.to_pos;
        }
    }

    absl::StrAppend(&returnVal, " ", PositionToStringHumanReadable(start_hr), "/",
                    PositionToStringHumanReadable(end_hr));
  }

   if (!any_move) {
       // If loop finishes and any_move is false, it must have been all passes (handled above)
       // or an empty move sequence (which shouldn't happen for valid actions).
       // We return Pass as a fallback.
       return absl::StrCat(move_id, " - Pass");
   }

  return returnVal;
}

ScoringType ParseScoringType(const std::string& st_str) {
  if (st_str == "winloss_scoring") {
    return ScoringType::kWinLossScoring;
  } else if (st_str == "winlosstie_scoring") {
    return ScoringType::kWinLossTieScoring;
  } else {
    SpielFatalError("Unrecognized scoring_type parameter: " + st_str);
    return ScoringType::kWinLossScoring; // Should be unreachable
  }
}

} // namespace long_narde
} // namespace open_spiel 