#include "open_spiel/games/long_narde/long_narde.h"

#include <string>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

// ===== Movement Functions =====

/**
 * @brief Applies a single checker move (half-move) to the board state.
 *
 * Updates the board by removing a checker from `from_pos` and adding it to `to_pos`.
 * Handles bearing off by incrementing the player's score instead of placing on the board.
 * Also manages the `moved_from_head_` flag if the move originates from a head position.
 *
 * @param player The player making the move.
 * @param move The CheckerMove struct containing from_pos, to_pos, and die value.
 */
void LongNardeState::ApplyCheckerMove(Player player, const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Nothing to do for a pass move

  // RE-ENABLED: Validation check.
  if (!IsValidCheckerMove(player, move, /*moved_from_head_this_sequence=*/false)) {
    std::string error_message = "Invalid checker move: ";
    error_message += "Player: ", std::to_string(player);
    error_message += absl::StrCat(", Move: ", move.pos, "->", move.to_pos, "/", move.die);
    error_message += "\nIs first turn? ", (IsFirstTurn(player)?"Y":"N");
    error_message += ", Moved from head? ", (moved_from_head_?"Y":"N");
    error_message += ", Dice: " + DiceToString();
    error_message += "\nBoard:\n" + ToString();
    SpielFatalError(error_message);
  }

  // Apply move
  board_[player][move.pos]--;
  if (!IsOff(player, move.to_pos)) { 
    board_[player][move.to_pos]++;
  } else {
    // Bearing off - increment score
    scores_[player]++;
  }

  // Mark die as used. Find the *first available* slot matching the die value.
  bool found_die = false;
  for (int i = 0; i < dice_.size(); ++i) { // dice_.size() should be 4
    if (IsDieUsable(i) && DiceValue(i) == move.die) {
      dice_[i] += kNumDiceOutcomes; // Mark as used (e.g., 3 -> 9)
      found_die = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(found_die); // Should always find a usable die if the move was generated correctly

  // Update moved_from_head status if applicable
  if (IsHeadPos(player, move.pos)) {
    moved_from_head_ = true;
  }
}

/**
 * @brief Undoes a single checker move (half-move) from the board state.
 *
 * Reverts the board changes made by ApplyCheckerMove. Removes a checker from `to_pos`
 * (or decrements score if it was a bear-off) and adds it back to `from_pos`.
 * Note: This function does NOT revert the `moved_from_head_` flag, as that depends on the whole turn's sequence.
 *
 * @param player The player whose move is being undone.
 * @param move The CheckerMove struct containing from_pos, to_pos, and die value.
 */
void LongNardeState::UndoCheckerMove(Player player, const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Nothing to undo for a pass

  // Check consistency: should have a valid starting position
  SPIEL_CHECK_GE(move.pos, 0); 
  SPIEL_CHECK_LT(move.pos, kNumPoints);

  // Reverse the move application
  board_[player][move.pos]++;
  if (!IsOff(player, move.to_pos)) { 
    board_[player][move.to_pos]--;
  } else {
    // Undo bearing off - decrement score
    scores_[player]--;
  }

  // Unmark die as used. Find the *first used* slot matching the die value.
  bool found_die = false;
  for (int i = 0; i < dice_.size(); ++i) { // dice_.size() should be 4
    if (!IsDieUsable(i) && DiceValue(i) == move.die) { // Check !IsDieUsable
      dice_[i] -= kNumDiceOutcomes; // Unmark (e.g., 9 -> 3)
      found_die = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(found_die); // Should always find a used die to unmark if undoing correctly

  // moved_from_head_ is restored from TurnHistoryInfo in UndoAction.
}

/**
 * @brief Calculates the destination position for a move.
 *
 * Given a starting position and a die roll, determines the resulting board position
 * index after moving counter-clockwise. Handles bearing off by returning a special value
 * (kBearOffPosWhite or kBearOffPosBlack).
 *
 * @param player The player making the move.
 * @param from_pos The starting position index (0-23).
 * @param die_value The value of the die roll (1-6).
 * @return The destination position index (0-23), or a special bear-off value (-1 or -2).
 */
int LongNardeState::GetToPos(int player, int from_pos, int pips) const {
  SPIEL_CHECK_GE(from_pos, 0);
  SPIEL_CHECK_LT(from_pos, kNumPoints);
  SPIEL_CHECK_GE(pips, 1);
  SPIEL_CHECK_LE(pips, 6);

  if (player == kXPlayerId) { // White path: 23 -> 0 (decreasing index)
    int target_idx = from_pos - pips;
    // Check if the move takes the checker off the board (past index 0)
    if (target_idx < 0) {
      return kBearOffPos; // Bear off
    }
    return target_idx; // Regular move
  } else { // kOPlayerId (Black path: 11 -> 0 -> 23 -> 12 (complex index changes))
    int current_pos = from_pos;
    for (int i = 0; i < pips; ++i) {
      if (current_pos == 0) { // Wrap around from 0 to 23
        current_pos = 23;
      } else if (current_pos == 12) { // Trying to move from point 13 (index 12)
         // Any move from index 12 goes off the board for Black
        return kBearOffPos; 
      } else { // Normal move: decrement index
        current_pos--;
      }

      // Additional check: If a move starts at or after point 13 (index >= 12)
      // and *lands* at or before point 12 (index <= 11) *during* the pip count,
      // it means the checker crossed the finish line and should bear off.
      // We need to compare the position *before* this single pip step.
      int pos_before_step; 
       if (current_pos == 23) pos_before_step = 0; // Wrapped around
       else if (current_pos == kBearOffPos) pos_before_step = 12; // Just went off
       else pos_before_step = current_pos + 1; // Normal decrement

      if (pos_before_step >= 12 && current_pos <= 11) {
           // Crossed the finish line (from index 12+ to 11-) mid-move
           return kBearOffPos;
      }
    }

    // After moving 'pips' steps, check final position
    // REMOVED: This check incorrectly treated landing on index 12 as bear off.
    // if (current_pos == 12) { // Landed exactly on the bear-off threshold index
    //     return kBearOffPos;
    // }

    // If still on the board, return the final index.
    SPIEL_CHECK_GE(current_pos, 0); 
    SPIEL_CHECK_LT(current_pos, kNumPoints); 
    return current_pos;
  }
}

} // namespace long_narde
} // namespace open_spiel
