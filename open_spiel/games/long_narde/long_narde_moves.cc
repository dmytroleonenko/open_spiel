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
void LongNardeState::ApplyCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Nothing to do for a pass move

  // Re-validate the move *without* the head rule check here.
  // The head rule is context-dependent (how many moved *before* this one)
  // and is handled during sequence generation (GenerateAllHalfMoves/RecLegalMoves).
  // This check ensures basic validity (on board, not blocked, valid destination).
  if (!IsValidCheckerMove(player, move.pos, move.to_pos, move.die, /*check_head_rule=*/false)) {
    std::string error_message = absl::StrCat("ApplyCheckerMove: Invalid checker move provided! ",
                                           "Player ", player, " Move: ", move.pos, "->", move.to_pos, "/", move.die);
     error_message += "\nBoard state:\n" + ToString();
     error_message += "\nDice: ";
      for (int d : dice_) { error_message += absl::StrCat(DiceValue(d), UsableDiceOutcome(d)?" ":"u "); }
     error_message += "\nMoved from head? ", (moved_from_head_?"Y":"N");
     error_message += "\nIs first turn? ", (is_first_turn_?"Y":"N");
    SpielFatalError(error_message);
  }

  // Perform the move on the board
  SPIEL_CHECK_GE(move.pos, 0); // Should be guaranteed by IsValidCheckerMove if not pass
  SPIEL_CHECK_LT(move.pos, kNumPoints);
  SPIEL_CHECK_GT(board_[player][move.pos], 0); // Must have a checker to move
  board_[player][move.pos]--;

  // Mark the die used (find the first usable die with that value)
  bool die_marked = false;
  for (int i = 0; i < dice_.size(); ++i) {
    if (UsableDiceOutcome(dice_[i]) && dice_[i] == move.die) {
      dice_[i] += 6; // Mark as used by adding 6
      die_marked = true;
                    break;
                  }
                }
  SPIEL_CHECK_TRUE(die_marked); // Should always find a usable die if the move was valid

  // Update destination (board or score)
  int next_pos = move.to_pos; 
  if (IsOff(player, next_pos)) {
    scores_[player]++;
    SPIEL_CHECK_LE(scores_[player], kNumCheckersPerPlayer);
              } else {
    // Ensure destination is valid board position (should be guaranteed by IsValidCheckerMove)
    SPIEL_CHECK_GE(next_pos, 0);
    SPIEL_CHECK_LT(next_pos, kNumPoints);
    board_[player][next_pos]++;
  }

  // Update head move status *for the current turn's sequence*
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
void LongNardeState::UndoCheckerMove(int player, const CheckerMove& move) {
  if (move.pos == kPassPos) return; // Nothing to undo for a pass

  // Check consistency: should have a valid starting position
  SPIEL_CHECK_GE(move.pos, 0); 
  SPIEL_CHECK_LT(move.pos, kNumPoints);

  // Restore checker to the starting position
  board_[player][move.pos]++;
  SPIEL_CHECK_LE(board_[player][move.pos], kNumCheckersPerPlayer);

  // Unmark the die used (find the first *used* die matching the value)
   bool die_unmarked = false;
   for (int i = 0; i < dice_.size(); ++i) {
     // Check if dice_[i] represents the used version of move.die
     if (dice_[i] == move.die + 6) { 
       dice_[i] -= 6; // Unmark by subtracting 6
       die_unmarked = true;
       break;
     }
   }
   // If this fails, it indicates a major inconsistency in state/undo logic.
   if (!die_unmarked) {
        std::string error_msg = "UndoCheckerMove: Could not find used die to unmark. ";
        error_msg += absl::StrCat("Player ", player, ", Move ", move.pos, "->", move.to_pos, "/", move.die);
        error_msg += "\nDice state: ";
         for (int d : dice_) { error_msg += absl::StrCat(d, " "); }
         error_msg += "\nBoard:\n" + ToString();
         SpielFatalError(error_msg);
   }

  // Reverse the effect on the destination
  int next_pos = move.to_pos; 
  if (IsOff(player, next_pos)) {
    // If it was a bear-off move, decrement the score
    scores_[player]--;
    SPIEL_CHECK_GE(scores_[player], 0);
  } else {
    // If it was a regular move, remove checker from the destination
    // Ensure destination is valid before decrementing
    SPIEL_CHECK_GE(next_pos, 0);
    SPIEL_CHECK_LT(next_pos, kNumPoints);
    SPIEL_CHECK_GT(board_[player][next_pos], 0); // Must have been a checker there
    board_[player][next_pos]--;
  }

  // Note: Undoing moved_from_head_ is handled by the caller (RecLegalMoves)
  // by restoring the value from before the ApplyCheckerMove call.
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
