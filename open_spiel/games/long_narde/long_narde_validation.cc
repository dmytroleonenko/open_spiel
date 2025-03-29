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

bool LongNardeState::IsLegalHeadMove(int player, int from_pos) const {
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
  // Use the member variable 'is_first_turn_' here
  if (is_first_turn_ && is_special_double_roll) {
    // On special first turn doubles, we can move up to two checkers from head.
    // This function checks the validity of a *single* potential move.
    // The limit of two moves is handled implicitly by the sequence generation
    // (RecLegalMoves) and its depth limit combined with state updates.
    return true; // Allow potential head move during special first turn double.
  }

  // Normal case (not first turn OR not a special double roll):
  // Can only move from head if no checker has moved from head *yet* this turn.
  return !moved_from_head_;
}

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

bool LongNardeState::IsValidCheckerMove(int player, int from_pos, int to_pos, int die_value, bool check_head_rule) const {
  // --- Basic Checks ---
  if (from_pos == kPassPos) return true; // Pass is always valid in isolation.
  if (from_pos < 0 || from_pos >= kNumPoints) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid from_pos " << from_pos << std::endl;
    return false;
  }
  if (board(player, from_pos) <= 0) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: No checker at from_pos " << from_pos << std::endl;
    return false;
  }
  if (die_value < 1 || die_value > 6) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid die_value " << die_value << std::endl;
    return false;
  }
  // Check if the provided 'to_pos' matches the calculated destination
  int expected_to_pos = GetToPos(player, from_pos, die_value);
  if (to_pos != expected_to_pos) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: to_pos " << to_pos << " doesn't match expected " << expected_to_pos << " for die " << die_value << " from " << from_pos << std::endl;
    return false;
  }

  // --- Head Rule Check ---
  if (check_head_rule && !IsLegalHeadMove(player, from_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Head rule violation for pos " << from_pos << std::endl;
    return false;
  }

  // --- Bearing Off Checks ---
  bool is_bearing_off = IsOff(player, to_pos);
  if (is_bearing_off) {
    // Check if all checkers are in home *directly* here
    int checkers_outside_home = 0;
    for (int pos_check = 0; pos_check < kNumPoints; ++pos_check) { // Use different loop variable
        if (board(player, pos_check) > 0) {
            bool is_home = (player == kXPlayerId) ? 
                            (pos_check >= kWhiteHomeStart && pos_check <= kWhiteHomeEnd) :
                            (pos_check >= kBlackHomeStart && pos_check <= kBlackHomeEnd);
            if (!is_home) {
                checkers_outside_home += board(player, pos_check);
            }
        }
    }
    if (checkers_outside_home > 0) {
        if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Direct check failed - Cannot bear off, " << checkers_outside_home << " checkers outside home" << std::endl;
        return false;
    }

    // Calculate exact pips needed to bear off from 'from_pos'
    int pips_needed;
    if (player == kXPlayerId) {
        // White bears off from pos 0-5. Pips needed = pos + 1.
        SPIEL_CHECK_GE(from_pos, kWhiteHomeStart); // Should be in home
        SPIEL_CHECK_LE(from_pos, kWhiteHomeEnd);
        pips_needed = from_pos + 1;
    } else { // kOPlayerId
        // Black bears off from pos 12-17. Pips needed = pos - 11.
        SPIEL_CHECK_GE(from_pos, kBlackHomeStart); // Should be in home
        SPIEL_CHECK_LE(from_pos, kBlackHomeEnd);
        pips_needed = from_pos - 11; 
    }

    if (die_value == pips_needed) {
        return true; // Exact roll bears off.
    }
    if (die_value > pips_needed) {
        // Higher roll can bear off *only if* no checkers are further back.
        if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Checking higher roll (die=" << die_value << " > needed=" << pips_needed << ") for pos=" << from_pos << std::endl;
        bool further_checker_exists = false;
        // Need to check positions *within the home board* that require *more* pips than 'from_pos'.
        if (player == kXPlayerId) {
            for (int check_pos = from_pos + 1; check_pos <= kWhiteHomeEnd; ++check_pos) {
                 if (board(player, check_pos) > 0) {
                     further_checker_exists = true;
                     break;
                 }
            }
        } else { // kOPlayerId
             // Correct: Check positions requiring *more* pips, which are *lower* indices for Black within home [12..17]
             for (int check_pos = from_pos - 1; check_pos >= kBlackHomeStart; --check_pos) { // Iterate downwards
                 if (board(player, check_pos) > 0) {
                     further_checker_exists = true;
                     break;
                 }
            }
        }
        if (!further_checker_exists) {
             return true; // Can bear off with higher roll
        }
    }
    // If die_value < pips_needed, it's an invalid bear off move.
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid bearing off move (die=" << die_value << " < needed=" << pips_needed << ")" << std::endl;
    return false;
  }
  
  // --- Regular Move Checks ---
  // Check destination bounds (already implicitly checked by GetToPos if not bearing off)
  if (to_pos < 0 || to_pos >= kNumPoints) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Invalid to_pos " << to_pos << " for non-bearoff" << std::endl;
    return false; // Should be unreachable if GetToPos is correct and not bearing off
  }

  // Check opponent occupancy at destination
  if (board(Opponent(player), to_pos) > 0) {
     if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Cannot land on opponent's checker at " << to_pos << std::endl;
    return false;
  }

  // Check if the move *would* form an illegal blocking bridge
  if (WouldFormBlockingBridge(player, from_pos, to_pos)) {
    if (kDebugging) std::cout << "DEBUG IsValidCheckerMove: Would form illegal blocking bridge from " << from_pos << " to " << to_pos << std::endl;
    return false;
  }

  // If all checks passed, it's a valid move.
  return true;
}

bool LongNardeState::ValidateAction(Action action) const {
  if (IsChanceNode() || IsTerminal()) return false; // Actions only valid for current player

  if (action < 0 || action >= NumDistinctActions()) {
     if (kDebugging) std::cout << "DEBUG ValidateAction: Action " << action << " out of range [0, " << NumDistinctActions() << ")" << std::endl;
    return false;
  }
  
  // The most reliable validation is checking if it's in the set of legal actions.
  const auto& legal_actions = LegalActions(); // Calculate legal actions
  if (std::find(legal_actions.begin(), legal_actions.end(), action) == legal_actions.end()) {
    if (kDebugging) {
      std::cout << "DEBUG ValidateAction: Action " << action << " not found in legal actions.\n";
      std::cout << "DEBUG: Decoded moves for invalid action " << action << ":\n";
       try {
           std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
            for (const auto& m : moves) {
                std::cout << "  pos=" << m.pos << ", to_pos=" << GetToPos(cur_player_, m.pos, m.die)
                        << ", die=" << m.die << "\n";
            }
       } catch(...) { std::cout << "   <Decoding failed>" << std::endl;}
      std::cout << "DEBUG: Current dice: ";
      // Correctly display dice values using DiceValue
      for (size_t i = 0; i < dice_.size(); ++i) { std::cout << DiceValue(i) << " "; }
      std::cout << "\nDEBUG: Board state:\n" << ToString() << "\n";
       std::cout << "DEBUG: Legal actions (" << legal_actions.size() << " total): ";
       for (Action a : legal_actions) { std::cout << a << " "; }
       std::cout << "\n";
    }
    return false;
  }
  
  // Optional: Perform consistency checks on the decoded move sequence itself, 
  // even though it was found in LegalActions. This helps catch bugs in encoding/decoding or LegalActions.
  // Note: This duplicates some logic but can be useful for debugging.
  #ifndef NDEBUG // Only run these checks in debug builds
  try {
        std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
        // Simulate applying the moves on a cloned state to verify step-by-step validity
        std::unique_ptr<State> temp_state_ptr = this->Clone();
        LongNardeState* temp_state = dynamic_cast<LongNardeState*>(temp_state_ptr.get());
        
        bool sequence_valid = true;
        for (const auto& move : moves) {
            if (move.pos == kPassPos) continue; // Skip passes here

            // Check validity *in the context of the temporary state*
            // Need to ensure temp_state is correctly managing its internal state like moved_from_head_
            // Assuming ApplyCheckerMove updates moved_from_head_ correctly within temp_state
            if (!temp_state->IsValidCheckerMove(temp_state->cur_player_, move.pos, move.to_pos, move.die, /*check_head_rule=*/true)) {
                 if (kDebugging) {
                     std::cout << "ERROR ValidateAction: Decoded move [" << move.pos << "->" << move.to_pos << "/" << move.die 
                               << "] from legal action " << action << " is INVALID at its step in sequence!" << std::endl;
                     std::cout << "  Temp State Board:\n" << temp_state->ToString() << std::endl;
                 }
                 sequence_valid = false;
                 break;
            }
            // Apply the move to the temp state for the next check
             temp_state->ApplyCheckerMove(temp_state->cur_player_, move);
        }

        if (!sequence_valid) {
            // Consider logging or asserting here if an action from LegalActions fails validation.
             // SpielFatalError("Inconsistency: Action from LegalActions failed sequence validation.");
             return false; // Treat as invalid if sequence check fails
        }

  } catch (...) {
       if (kDebugging) std::cout << "ERROR ValidateAction: Exception during validation decode/simulation for action " << action << std::endl;
       return false; // Decoding/simulation error means invalid
  }
  #endif // NDEBUG

  return true; // Action is in the legal set
}

bool LongNardeState::IsOff(int player, int pos) const {
  return pos == kBearOffPos; // kBearOffPos is the special value indicating off the board
}


} // namespace long_narde
} // namespace open_spiel

 