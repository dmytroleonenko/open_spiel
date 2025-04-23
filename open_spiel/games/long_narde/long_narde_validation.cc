#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> // For std::find
#include <iostream>  // For kDebugging cout
#include <memory>    // For unique_ptr in ValidateAction

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" // For debug output

namespace open_spiel {
namespace long_narde {

// ===== Validation Functions =====

bool LongNardeState::IsHeadPos(int player, int pos) const {
  return (player == kXPlayerId && pos == kWhiteHeadPos) ||
         (player == kOPlayerId && pos == kBlackHeadPos);
}

bool LongNardeState::IsFirstTurn(int player) const {
  // The first turn is characterized by having all 15 checkers on the head point.
  int head_pos = (player == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;
  return board_[player][head_pos] == kNumCheckersPerPlayer;
  // NOTE: This function checks the *current* state. The *member variable* `is_first_turn_` 
  // holds the status determined at the *beginning* of the player's turn.
}

bool LongNardeState::IsLegalHeadMove(int player, int from_pos, bool moved_from_head_this_sequence) const {
  bool is_head = IsHeadPos(player, from_pos);
  if (!is_head) return true; // Not a head move, always allowed by this rule.

  // Head Rule 5: Only 1 checker may leave the head per turn.
  // Exception: First turn double 6, 4, or 3 allows moving 2 checkers.

  // Use the member variable 'is_first_turn_' which reflects the turn status
  // at the beginning of the turn, not the current simulation state.
  bool is_special_double_roll = false;

  // *** Use initial_dice_ for this check ***
  if (initial_dice_.size() >= 2) { // Check the roll at the start of the turn
      int die1_val = initial_dice_[0]; // Raw value is fine here (1-6)
      int die2_val = initial_dice_[1];
      if (die1_val == die2_val && (die1_val == 3 || die1_val == 4 || die1_val == 6)) {
          is_special_double_roll = true;
      }
  }

  // Check for first turn special doubles exception
  // *** Use the MEMBER VARIABLE 'is_on_first_turn_' instead of the method IsFirstTurn(player) ***
  if (is_on_first_turn_ && is_special_double_roll) {
    // On special first turn doubles, we can move up to two checkers from head.
    // This function checks the validity of a *single* potential move.
    // The limit of two moves is handled implicitly by the sequence generation
    // (RecLegalMoves) and its depth limit combined with state updates.
    return true; // Allow potential head move during special first turn double.
  }

  // Normal case (not first turn OR not a special double roll):
  // Can only move from head if no checker has moved from head *yet* this sequence.
  return !moved_from_head_this_sequence;
}

/**
 * @brief Checks if a given move would result in an illegal 6-point blocking bridge.
 *
 * Rationale: In Long Narde, forming a prime (a block of 6 consecutive points)
 * is illegal if it completely traps all of the opponent's checkers behind it.
 * A bridge is considered legal only if at least one opposing checker is ahead of
 * (further along the opponent's path than) the block's starting point.
 *
 * This function simulates the move on a temporary board and then checks all possible
 * 6-point spans. For each 6-block found, it verifies if any opponent checker
 * exists ahead of the block's effective start point (relative to the opponent's path).
 *
 * @param player The player proposing the move.
 * @param from_pos The starting position of the move (0-23), or -1 to check the current board state.
 * @param to_pos The destination position of the move (0-23 or bear-off), or -1 to check the current board state.
 * @return True if the move (or current state if from/to are -1) results in an illegal blocking bridge, false otherwise.
 */
bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const {
  // Create a temporary board reflecting the potential move
  std::vector<std::vector<int>> temp_board = board_;
  if (from_pos >= 0 && from_pos < kNumPoints) {
     if (temp_board[player][from_pos] <= 0) {
        // Trying to move from an empty point - should have been caught earlier, but handle defensively.
        // This move itself is invalid, but doesn't inherently form an illegal bridge yet.
        // Consider this case as not forming an *additional* illegal bridge.
        return false; 
     }
     temp_board[player][from_pos]--;
  }
  // Don't check bounds for to_pos yet, might be bearing off
   if (to_pos >= 0 && to_pos < kNumPoints) {
      temp_board[player][to_pos]++;
   } else if (!IsOff(player, to_pos)) {
       // Invalid 'to_pos' that isn't bear off - move is illegal, but not specifically a bridge issue.
       return false;
   }

  int opponent = Opponent(player);
  bool opponent_has_checkers_on_board = false;
  for (int i = 0; i < kNumPoints; ++i) {
    if (temp_board[opponent][i] > 0) {
        opponent_has_checkers_on_board = true;
        break;
      }
  }

  // If opponent has no checkers left on the board, no bridge can possibly trap them.
  if (!opponent_has_checkers_on_board) {
      return false;
  }

  // Check all possible 6-point spans for a block
  for (int start = 0; start < kNumPoints; ++start) {
    bool is_block = true;
    for (int i = 0; i < 6; ++i) {
      int pos = (start + i) % kNumPoints;
      if (temp_board[player][pos] == 0) {
        is_block = false;
        break;
      }
    }

    if (is_block) {
      // Found a 6-block. Check if it's illegal.
      // Rule: Illegal if NO opponent checker is ahead of the block's start (from opponent's perspective).
      // "Ahead" means further along the opponent's path (higher path index).
      int block_path_start_on_opp_path_real_pos = GetBlockPathStartRealPos(opponent, start);

      bool is_legal_bridge = false; // Assume illegal until proven otherwise
      for (int opp_pos_idx = 0; opp_pos_idx < kNumPoints; ++opp_pos_idx) {
        if (temp_board[opponent][opp_pos_idx] > 0) {
            // Check if this opponent checker is ahead of the block's starting point.
            if (IsAhead(opponent, opp_pos_idx, block_path_start_on_opp_path_real_pos)) {
                 // Found an opponent checker ahead. Bridge is legal.
                is_legal_bridge = true;
                break; // No need to check other opponent checkers or other blocks starting here.
            }
        }
      }

      // If after checking all opponent checkers, none were found ahead, the bridge is illegal.
      if (!is_legal_bridge) {
        return true; // Illegal bridge would be formed
      }
      // Otherwise (is_legal_bridge is true), this specific block is legal. Continue checking other potential blocks.
    }
  }

  return false; // No illegal bridge found
}

// Checks the current board state for an illegal bridge for the given player.
bool LongNardeState::HasIllegalBridge(int player) const {
   // This just calls WouldFormBlockingBridge without simulating a move.
   // We pass invalid from/to positions to check the *current* board state.
   return WouldFormBlockingBridge(player, /*from_pos=*/-1, /*to_pos=*/-1);
}

/**
 * @brief Checks if a single proposed checker move is valid according to Long Narde rules.
 *
 * This function validates a single step (half-move) of moving one checker based on a die roll.
 * It performs several checks:
 * 1.  Basic validity (non-pass move, valid positions, checker exists at start).
 * 2.  Correctness of the destination position based on the die roll.
 * 3.  Head Rule: Ensures the move doesn't violate restrictions on moving from the head position (if check_head_rule is true).
 * 4.  Bearing Off: Validates bear-off moves, checking if all checkers are home, if the roll is exact or higher, and if higher rolls are permitted (no checkers further back).
 * 5.  Opponent Occupancy: Ensures the destination point is not occupied by an opponent's checker.
 * 6.  Bridging Rule: Checks if making this move would create an illegal 6-point block that traps the opponent.
 *
 * @param player The player making the move (kXPlayerId or kOPlayerId).
 * @param from_pos The starting board position index (0-23), or kPassPos (-1).
 * @param to_pos The target board position index (0-23), or kBearOffPos (-1 or -2 depending on player).
 * @param die_value The value of the die used for this move (1-6).
 * @param check_head_rule If true, enforces the head movement rule for this move. Should generally be true, except when validating individual steps within a pre-validated sequence.
 * @param moved_from_head_this_sequence Boolean flag indicating if a checker has already moved from the head in the current sequence being explored.
 * @return True if the single checker move is valid, false otherwise.
 */
bool LongNardeState::IsValidCheckerMove(int player, const CheckerMove& move,
                                         bool moved_from_head_this_sequence) const {
  // Check basic move properties
  if (move.pos == kPassPos) return true; // Pass is always valid conceptually

  // Validate inputs
  SPIEL_CHECK_GE(move.pos, 0);
  SPIEL_CHECK_LT(move.pos, kNumPoints);
  SPIEL_CHECK_GT(board_[player][move.pos], 0); // Must have a checker to move
  SPIEL_CHECK_GE(move.die, 1);
  SPIEL_CHECK_LE(move.die, 6);

  // Calculate the potential destination position first.
  int to_pos = GetToPos(player, move.pos, move.die);

  // Check Bear Off conditions
  bool all_checkers_home = AllInHome(player); // Calculate once

  // ADDED DEBUGGING
  if (kDebugging) std::cout << "    [DEBUG IsValidCheckerMove] pos=" << move.pos << " die=" << move.die << " -> to_pos=" << to_pos
                           << " | calculated_to_pos=" << to_pos << " | all_checkers_home=" << all_checkers_home << std::endl;

  // First, check if the player is in the bearing off phase
  if (all_checkers_home) {
      // --- Bear Off Validation ---
      int pips_needed = (player == kXPlayerId) ? (move.pos + 1) : (move.pos - 12 + 1);

      // Check if the die roll EXACTLY matches the pips needed for bear-off
      if (move.die == pips_needed) {
          if (kDebugging) std::cout << "    VALID: Exact bear off check passed (pos=" << move.pos << ", die=" << move.die << ")." << std::endl;
          return true; // Exact bear-off is always valid if all checkers are home
      }

      // Check if die roll is sufficient
      if (move.die < pips_needed) {
          // This move is only invalid for bear-off; it might be a valid regular move within the home board.
          // We will let the regular move checks handle this later.
          if (kDebugging) std::cout << "    INFO: Die (" << move.die << ") < pips needed (" << pips_needed << ") for bear-off. Will check as regular move." << std::endl;
          // Continue to regular move checks
      } else { // move.die > pips_needed
          // If die roll is higher than needed, check if it's the furthest checker.
          int furthest_pos = FurthestCheckerInHome(player);
          if (move.pos != furthest_pos) {
               if (kDebugging) std::cout << "    INVALID: Cannot use higher die roll (" << move.die << ") for non-furthest checker (pos=" << move.pos << ", furthest=" << furthest_pos << ")." << std::endl;
               return false; // Invalid bear-off (higher roll on non-furthest)
          }
          // Valid bear-off move using a higher die roll on the furthest checker.
          if (kDebugging) std::cout << "    VALID: Higher die bear off check passed (pos=" << move.pos << ", die=" << move.die << ")." << std::endl;
          return true;
      }
  }

  // --- Regular Move Check (or move within home board if bear-off conditions not met) ---
  // Check if the calculated destination is off the board (and wasn't a valid bear-off)
  bool is_target_off_board = IsOff(player, to_pos);
  if (is_target_off_board) {
       if (kDebugging) std::cout << "    INVALID: Calculated to_pos (" << to_pos << ") is off-board, but not a valid bear-off." << std::endl;
       return false;
  }
  // Check destination validity (must be within 0-23)
  if (to_pos < 0 || to_pos >= kNumPoints) {
       if (kDebugging) std::cout << "    INVALID: Calculated to_pos (" << to_pos << ") is outside valid board range [0, 23]." << std::endl;
       return false;
  }

  // Check opponent occupancy at the calculated on-board destination.
  // ADDED DEBUG LOGGING HERE
  if (player == 1 && move.pos == 11 && move.die == 1) {
      std::cout << "[DEBUG ILM CHECK] player=" << player
                << ", move.pos=" << move.pos
                << ", move.die=" << move.die
                << ", to_pos=" << to_pos
                << ", Opponent(player)=" << Opponent(player)
                << ", board(Opponent(player), to_pos)=" << board(Opponent(player), to_pos)
                << std::endl;
  }
  // ADDED DEBUG LOGGING for NoLandingOnOpponentTest case
  if (player == 1 && move.pos == 15 && move.die == 3) {
      std::cout << "[DEBUG NLO CHECK] player=" << player
                << ", move.pos=" << move.pos
                << ", move.die=" << move.die
                << ", to_pos=" << to_pos
                << ", Opponent(player)=" << Opponent(player)
                << ", board(Opponent(player), to_pos)=" << board(Opponent(player), to_pos)
                << std::endl;
  }
  if (board(Opponent(player), to_pos) > 0) {
    if (kDebugging) std::cout << "[DEBUG ICMV " << player << "] Invalid move: Opponent block at " << to_pos << " for move " << move.pos << " -> " << to_pos << std::endl;
    return false;
  }

  // Check Head Rule
  if (IsHeadPos(player, move.pos)) {
      // Ensure GetToPos calculated correctly for head moves initially
      SPIEL_CHECK_TRUE(to_pos >= 0 && to_pos < kNumPoints);
      if (!IsLegalHeadMove(player, move.pos, moved_from_head_this_sequence)) {
          if (kDebugging) std::cout << "    INVALID: Head rule violation (already moved from head this sequence)." << std::endl;
          return false;
      }
  }

  // ADDED CHECK: Black cannot move past index 12 once inside home
  if (player == kOPlayerId && move.pos >= kBlackHomeStart && to_pos < kBlackHomeStart) {
       if (kDebugging) std::cout << "    INVALID: Black cannot move past index 12 from home (pos=" << move.pos << ", to=" << to_pos << ")." << std::endl;
       return false;
  }

  // Check bridge rule
  if (WouldFormBlockingBridge(player, move.pos, to_pos)) {
    if (kDebugging) std::cout << "    INVALID: Move would form illegal bridge." << std::endl;
    return false; // Move would create an illegal bridge
  }

  if (kDebugging) std::cout << "    VALID: Regular move check passed (pos=" << move.pos << ", to=" << to_pos << ", die=" << move.die << ")." << std::endl;
  return true; // Regular move (or move within home) is valid
}

bool LongNardeState::ValidateAction(Action action) const {
  if (IsChanceNode() || IsTerminal()) return false; // Actions only valid for current player

  if (action < 0 || action >= NumDistinctActions()) {
     if (kDebugging) std::cout << "DEBUG ValidateAction: Action " << action << " out of range [0, " << NumDistinctActions() << ")" << std::endl;
    return false;
  }
  
  // Perform consistency checks by decoding and simulating the action.
  // This was previously inside #ifndef NDEBUG, now it's the main validation logic.
  try {
        std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
        // Simulate applying the moves on a cloned state to verify step-by-step validity
        std::unique_ptr<State> temp_state_ptr = this->Clone();
        LongNardeState* temp_state = dynamic_cast<LongNardeState*>(temp_state_ptr.get());

        // We will manipulate the dice_usage_count_ of the temp_state directly.

        bool sequence_valid = true;
        for (const auto& move : moves) {
            // Check if the move requires a die (i.e., not a pass)
            if (move.pos != kPassPos) {
                // Check if the required die (move.die) is available in the *temp_state* 
                // according to its current dice_ and dice_usage_count_.
                bool found_usable_die_slot = false;
                for (int i=0; i < temp_state->dice_.size(); ++i) {
                    if (temp_state->dice_[i] == move.die && temp_state->IsDieUsable(i)) {
                        found_usable_die_slot = true;
                        // Do not mark usage here; ApplyCheckerMove inside the loop will do it.
                        break;
                    }
                }
                
                if (!found_usable_die_slot) {
                     if (kDebugging) {
                         std::cout << "ERROR ValidateAction: Action " << action << " requires die " << move.die
                                   << " but no usable slot found in temp_state at this step." << std::endl;
                         std::cout << "  Temp State Dice: ";
                         for(size_t i=0; i < temp_state->dice_.size(); ++i) {
                             std::cout << temp_state->DiceValue(i) << " ";
                         }
                         std::cout << std::endl;
                     }
                    sequence_valid = false;
                    break;
                }
            }

            // Check move validity *in the context of the temporary state*
             // The moved_from_head state needs to be tracked across the loop for ValidateAction.
             // For now, passing true here might be incorrect, as it assumes the head rule applies
             // independently for each step rather than sequentially.
             // TODO: Refactor ValidateAction to track moved_from_head state sequentially.
            if (move.pos != kPassPos && !temp_state->IsValidCheckerMove(temp_state->cur_player_, move, /*moved_from_head_this_sequence=*/true)) {
                 if (kDebugging) {
                     std::cout << "ERROR ValidateAction: Decoded move [" << move.pos << "->" << move.to_pos << "/" << move.die
                               << "] from action " << action << " is INVALID at its step in sequence!" << std::endl;
                     std::cout << "  Temp State Board:\n" << temp_state->ToString() << std::endl;
                 }
                 sequence_valid = false;
                 break;
            }
            // Apply the move to the temp state for the next check
             // No longer need to track usage locally; 
             // ApplyCheckerMove below will update temp_state->dice_usage_count_
              // Apply the move (ApplyCheckerMove handles marking dice used *within the temp state*)
             temp_state->ApplyCheckerMove(temp_state->cur_player_, move);
        }

        if (!sequence_valid) {
            // If any step was invalid, the whole action is invalid.
             return false; 
        }
        
        // Optional: Add checks for dice usage rules (e.g., using max possible dice)?
        // For now, focus on breaking the recursion. If LegalActions filters correctly,
        // actions reaching here should already represent valid dice usage patterns.


  } catch (const std::exception& e) {
       if (kDebugging) std::cout << "ERROR ValidateAction: Exception during validation decode/simulation for action " << action << ": " << e.what() << std::endl;
       return false; // Decoding/simulation error means invalid
  } catch (...) {
       if (kDebugging) std::cout << "ERROR ValidateAction: Unknown exception during validation decode/simulation for action " << action << std::endl;
       return false; // Decoding/simulation error means invalid
  }
  // REMOVED: #ifndef NDEBUG and #endif

  return true; // Action survived simulation and decoding.
}

bool LongNardeState::IsOff(int player, int pos) const {
  return pos == kBearOffPos; // kBearOffPos is the special value indicating off the board
}

// ===== Home Region Checks =====

bool LongNardeState::AllInHome(Player player) const {
  int checkers_on_board = 0;
  if (player == kXPlayerId) {
    // White's home is points 1-6 (indices 0-5)
    // Check if any checkers are *outside* this range (points 7-24, indices 6-23)
    for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
      // Note: We don't need to sum checkers_on_board here as the return false above handles it.
    }
     // Also count checkers *inside* home
     for (int i = kWhiteHomeStart; i <= kWhiteHomeEnd; ++i) {
        checkers_on_board += board(player, i);
     }

  } else { // kOPlayerId
    // Black's home is points 13-18 (indices 12-17)
    // Check if any checkers are *outside* this range
    // Check indices 0-11 (points 1-12)
    for (int i = 0; i < kBlackHomeStart; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
       // Note: We don't need to sum checkers_on_board here.
    }
    // Check indices 18-23 (points 19-24)
    for (int i = kBlackHomeEnd + 1; i < kNumPoints; ++i) {
      if (board(player, i) > 0) {
        return false; // Found checker outside home
      }
        // Note: We don't need to sum checkers_on_board here.
    }
    // Also count checkers *inside* home
    for (int i = kBlackHomeStart; i <= kBlackHomeEnd; ++i) {
        checkers_on_board += board(player, i);
    }
  }
  
  // If we reached here, no checkers were found outside the home board.
  // All checkers currently on the board are within the home region.
  return true; // Allow bearing off
}

// Finds the position of the furthest checker in the home board. Returns -1 if empty.
int LongNardeState::FurthestCheckerInHome(Player player) const {
  if (player == kXPlayerId) { // White home: 0-5. Furthest is highest index.
    for (int pos = kWhiteHomeEnd; pos >= kWhiteHomeStart; --pos) {
      if (board_[player][pos] > 0) {
        return pos;
      }
    }
  } else { // Black home: 12-17. Furthest is lowest index.
    for (int pos = kBlackHomeStart; pos <= kBlackHomeEnd; ++pos) {
      if (board_[player][pos] > 0) {
        return pos;
      }
    }
  }
  return -1; // No checkers in home
}

// ===== Bridge Rule Checks =====

// Returns the actual die value (1-6) for a given index in dice_.
// Handles used dice markers.
int LongNardeState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size()); // dice_.size() is now 4
  int val = dice_[i];
  if (val == 0) return 0; // 0 represents an invalid/unused slot
  return (val > kNumDiceOutcomes) ? (val - kNumDiceOutcomes) : val;
}

// Checks if the die at the specified index in dice_ is usable.
bool LongNardeState::IsDieUsable(int index) const {
  SPIEL_CHECK_GE(index, 0);
  SPIEL_CHECK_LT(index, dice_.size()); // dice_.size() is now 4
  int val = dice_[index];
  // A die is usable if its value is > 0 (not an empty slot) 
  // and <= kNumDiceOutcomes (not marked as used).
  return val > 0 && val <= kNumDiceOutcomes;
}

// Checks if a given die *outcome* value (potentially marked as used) is usable.
// This function might be less relevant now with 4 slots, but keep for compatibility/potential use.
bool LongNardeState::UsableDiceOutcome(int outcome) const {
  // An outcome is usable if it's between 1 and 6 (inclusive).
  // Values 0 (empty slot) or > 6 (used marker) are not usable outcomes.
  return outcome >= 1 && outcome <= kNumDiceOutcomes;
}

} // namespace long_narde
} // namespace open_spiel

 