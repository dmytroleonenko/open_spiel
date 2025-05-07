#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> // For std::find
#include <iostream>  // For kDebugging cout
#include <memory>    // For unique_ptr in ValidateAction

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" // For debug output

namespace open_spiel
{
  namespace long_narde
  {

    // ===== Validation Functions =====

    bool LongNardeState::IsHeadPos(int player, int pos) const
    {
      return (player == kXPlayerId && pos == kWhiteHeadPos) ||
             (player == kOPlayerId && pos == kBlackHeadPos);
    }

    bool LongNardeState::IsFirstTurn(int player) const
    {
      // The first turn is characterized by having all 15 checkers on the head point.
      int head_pos = (player == kXPlayerId) ? kWhiteHeadPos : kBlackHeadPos;
      return board_[player][head_pos] == kNumCheckersPerPlayer;
      // NOTE: This function checks the *current* state. The *member variable* `is_first_turn_`
      // holds the status determined at the *beginning* of the player's turn.
    }

    bool LongNardeState::IsLegalHeadMove(int player, int from_pos, bool moved_from_head_this_sequence) const
    {
      if (!IsHeadPos(player, from_pos))
        return true; // Not a head move, always allowed by this rule.

      // If this is a head move, apply new logic incorporating head_move_occurred_this_full_turn_
      
      if (is_on_first_turn_) {
        bool is_special_double_roll = false;
        if (initial_dice_.size() >= 4 && // Need all 4 dice for initial check of doubles
            initial_dice_[0] > 0 && initial_dice_[0] <= kNumDiceOutcomes && /* Check it's a real die value */
            initial_dice_[0] == initial_dice_[1] && 
            initial_dice_[0] == initial_dice_[2] && 
            initial_dice_[0] == initial_dice_[3] && 
            (initial_dice_[0] == 3 || initial_dice_[0] == 4 || initial_dice_[0] == 6)) {
          is_special_double_roll = true;
        }

        if (is_special_double_roll) {
          // On special first turn doubles, up to two head moves are allowed in the entire turn.
          if (!head_move_occurred_this_full_turn_) {
            return true; // This is the first head move of the turn.
          } else {
            // One head move already occurred this turn.
            // Allow this one if it's the first head move of the current sub-sequence.
            return !moved_from_head_this_sequence; 
          }
        } else {
          // First turn, but not a special double: only one head move allowed in the turn.
          // It must also be the first in the current sub-sequence.
          return !head_move_occurred_this_full_turn_ && !moved_from_head_this_sequence;
        }
      } else {
        // Not on the first turn: only one head move allowed in the turn.
        // It must also be the first in the current sub-sequence.
        return !head_move_occurred_this_full_turn_ && !moved_from_head_this_sequence;
      }
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
    bool LongNardeState::WouldFormBlockingBridge(int player, int from_pos, int to_pos) const
    {
      // Create a temporary board reflecting the potential move
      std::vector<std::vector<int>> temp_board = board_;
      if (from_pos >= 0 && from_pos < kNumPoints)
      {
        if (temp_board[player][from_pos] <= 0)
        {
          // Trying to move from an empty point - should have been caught earlier, but handle defensively.
          // This move itself is invalid, but doesn't inherently form an illegal bridge yet.
          // Consider this case as not forming an *additional* illegal bridge.
          return false;
        }
        temp_board[player][from_pos]--;
      }
      // Don't check bounds for to_pos yet, might be bearing off
      if (to_pos >= 0 && to_pos < kNumPoints)
      {
        temp_board[player][to_pos]++;
      }
      else if (!IsOff(player, to_pos))
      {
        // Invalid 'to_pos' that isn't bear off - move is illegal, but not specifically a bridge issue.
        return false;
      }

      int opponent = Opponent(player);
      bool opponent_has_checkers_on_board = false;
      for (int i = 0; i < kNumPoints; ++i)
      {
        if (temp_board[opponent][i] > 0)
        {
          opponent_has_checkers_on_board = true;
          break;
        }
      }

      // If opponent has no checkers left on the board, no bridge can possibly trap them.
      if (!opponent_has_checkers_on_board)
      {
        return false;
      }

      // Check all possible 6-point spans for a block
      for (int start = 0; start < kNumPoints; ++start)
      {
        bool is_block = true;
        for (int i = 0; i < 6; ++i)
        {
          int pos = (start + i) % kNumPoints;
          if (temp_board[player][pos] == 0)
          {
            is_block = false;
            break;
          }
        }

        if (is_block)
        {
          // Found a 6-block. Check if it's illegal.
          // Rule: Illegal if NO opponent checker is ahead of the block's start (from opponent's perspective).
          // "Ahead" means further along the opponent's path (higher path index).
          int block_path_start_on_opp_path_real_pos = GetBlockPathStartRealPos(opponent, start);

          bool is_legal_bridge = false; // Assume illegal until proven otherwise
          for (int opp_pos_idx = 0; opp_pos_idx < kNumPoints; ++opp_pos_idx)
          {
            if (temp_board[opponent][opp_pos_idx] > 0)
            {
              // Check if this opponent checker is ahead of the block's starting point.
              if (IsAhead(opponent, opp_pos_idx, block_path_start_on_opp_path_real_pos))
              {
                // Found an opponent checker ahead. Bridge is legal.
                is_legal_bridge = true;
                break; // No need to check other opponent checkers or other blocks starting here.
              }
            }
          }

          // If after checking all opponent checkers, none were found ahead, the bridge is illegal.
          if (!is_legal_bridge)
          {
            return true; // Illegal bridge would be formed
          }
          // Otherwise (is_legal_bridge is true), this specific block is legal. Continue checking other potential blocks.
        }
      }

      return false; // No illegal bridge found
    }

    // Checks the current board state for an illegal bridge for the given player.
    bool LongNardeState::HasIllegalBridge(int player) const
    {
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
    bool LongNardeState::LongNardeIsValidCheckerMove(int player, const LongNardeCheckerMove &move,
                                            bool moved_from_head_this_sequence) const
    {
      // ADDING HYPER-SPECIFIC DEBUG FOR SCBO-1W scenario
      if (player == kXPlayerId && move.pos == 0 && (move.die == 1 || move.die == 6) && AllInHome(player)) {
        std::cout << "[DEBUG SCBO-1W IsValidMove ENTRY] P" << player << " Pos:" << move.pos << " Die:" << move.die << " ToPos:" << move.to_pos << " AllHome:true" << std::endl;
      }

      if (kDebugging) {
        std::cout << "[DEBUG IsValidMove] Entry: P" << player << " Move:{pos:" << move.pos 
                  << ", to:" << move.to_pos << ", die:" << move.die 
                  << "}, moved_head_seq: " << moved_from_head_this_sequence << std::endl;
      }

      // Check basic move properties
      if (move.pos == kPassPos) {
        if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: true (Pass)" << std::endl;
        return true; // Pass is always valid conceptually
      }

      // Validate inputs
      if (!(move.pos >= 0)) { return false; }
      if (!(move.pos < kNumPoints)) { return false; }
      if (!(board_[player][move.pos] > 0)) { 
        return false; 
      }
      if (!(move.die >= 1)) { return false; }
      if (!(move.die <= 6)) { return false; }

      // Calculate the potential destination position first.
      int to_pos_calc = GetToPos(player, move.pos, move.die);

      // Check Bear Off conditions
      bool all_checkers_home = AllInHome(player); // Calculate once

      // First, check if the player is in the bearing off phase
      if (all_checkers_home)
      {
        // --- Bear Off Validation ---
        int pips_needed = (player == kXPlayerId) ? (move.pos + 1) : (GetCanonicalPoint(player, move.pos) + 1);
        if (kDebugging) {
            std::cout << "[DEBUG IsValidMove] BearOffCheck: P" << player << " Pos:" << move.pos << " Die:" << move.die 
                      << " PipsNeeded:" << pips_needed << " AllHome:true" << std::endl;
        }

        if (move.die == pips_needed) // Exact bear-off
        {
          if (kDebugging) std::cout << "[DEBUG IsValidMove] ExactBearOff branch. move.die: " << move.die << " == pips_needed: " << pips_needed << std::endl;
          // ADDING HYPER-SPECIFIC DEBUG FOR SCBO-1W scenario
          if (player == kXPlayerId && move.pos == 0 && (move.die == 1 || move.die == 6) && AllInHome(player)) {
            std::cout << "[DEBUG SCBO-1W IsValidMove EXACT BEAROFF CHECK] P" << player << " Pos:" << move.pos << " Die:" << move.die << " PipsNeeded:" << pips_needed << std::endl;
          }
          for (int i = 0; i < dice_.size(); ++i) {
            if (IsDieUsable(i)) { 
              int other_die_val = DiceValue(i);
              if (kDebugging) {
                std::cout << "[DEBUG IsValidMove] ExactBearOff HigherDieCheck: move.die:" << move.die 
                          << ", other_die_slot:" << i << " (val:" << other_die_val << ", usable:true)" << std::endl;
              }
              if (other_die_val > move.die && other_die_val >= pips_needed) {
                if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: false (Higher die " << other_die_val << " could make exact/overkill bear_off for die " << move.die << ")" << std::endl;
                // ADDING HYPER-SPECIFIC DEBUG FOR SCBO-1W scenario
                if (player == kXPlayerId && move.pos == 0 && move.die == 1 && other_die_val == 6 && AllInHome(player)) {
                    std::cout << "[DEBUG SCBO-1W IsValidMove DECISION] Die 1 invalidated by Die 6 for P0, Pos0. Returning false." << std::endl;
                }
                return false; 
              }
            }
          }
          if (move.to_pos != kBearOffPos) { 
              if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: false (Exact bear-off but to_pos != kBearOffPos)" << std::endl;
              return false; 
          }
          if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: true (Exact bear-off, no higher die conflict)" << std::endl;
          // ADDING HYPER-SPECIFIC DEBUG FOR SCBO-1W scenario
          if (player == kXPlayerId && move.pos == 0 && (move.die == 1 || move.die == 6) && AllInHome(player)) {
            std::cout << "[DEBUG SCBO-1W IsValidMove FINAL RETURN TRUE] For P" << player << " Pos:" << move.pos << " Die:" << move.die << std::endl;
          }
          return true; 
        }
        else if (move.die > pips_needed) // Using a higher die than needed
        {
          if (kDebugging) std::cout << "[DEBUG IsValidMove] OverkillBearOff branch. move.die: " << move.die << " > pips_needed: " << pips_needed << std::endl;
          if (move.pos != FurthestCheckerInHome(player)) {
            if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: false (Overkill on non-furthest)" << std::endl;
            return false; 
          }
          if (move.to_pos != kBearOffPos) { 
              if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: false (Overkill bear-off but to_pos != kBearOffPos)" << std::endl;
              return false; 
          }
          if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: true (Overkill on furthest)" << std::endl;
          return true;
        }
        else
        { // move.die < pips_needed
          if (kDebugging) std::cout << "[DEBUG IsValidMove] Die < PipsNeeded for bear-off. move.die: " << move.die << " < pips_needed: " << pips_needed << std::endl;
          if (move.to_pos == kBearOffPos) {
             if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: false (Attempted bear-off with insufficient die)" << std::endl;
             return false;
          }
        }
      }
      // else (not all checkers home), if move.to_pos is kBearOffPos, it's an error caught later.

      // If we reach here, it's either a regular move, or a move within home that wasn't a direct bear-off.
      // Ensure move.to_pos is consistent with to_pos_calc for non-bear-off, and not off-board.
      if (move.to_pos == kBearOffPos) {
          // If trying to bear_off but all_checkers_home was false, it's invalid.
          if (!all_checkers_home) return false;
          // If all_checkers_home is true, but previous logic didn't return true, it's an invalid bear-off.
          return false; 
      }
      
      // Regular move checks (destination must be on board)
      if (IsOff(player, move.to_pos) || move.to_pos < 0 || move.to_pos >= kNumPoints) {
          return false; // Calculated to_pos is off board or invalid, but not a valid bear-off context
      }
      // And calculated to_pos from GetToPos must match move.to_pos for non-bear-off moves.
      if (to_pos_calc != move.to_pos) {
          return false;
      }

      // Head Rule
      bool legal_head_move = IsLegalHeadMove(player, move.pos, moved_from_head_this_sequence);
      if (!legal_head_move) {
        return false;
      }

      // Opponent Occupancy
      if (board_[Opponent(player)][move.to_pos] > 0) {
        return false;
      }

      // Bridge Rule
      bool would_block = WouldFormBlockingBridge(player, move.pos, move.to_pos);
      if (would_block) {
        if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: false (WouldFormBlockingBridge)" << std::endl;
        return false;
      }

      if (kDebugging) std::cout << "[DEBUG IsValidMove] Return: true (All checks passed)" << std::endl;
      // ADDING HYPER-SPECIFIC DEBUG FOR SCBO-1W scenario
      if (player == kXPlayerId && move.pos == 0 && (move.die == 1 || move.die == 6) && AllInHome(player)) {
            std::cout << "[DEBUG SCBO-1W IsValidMove FALLTHROUGH RETURN TRUE] For P" << player << " Pos:" << move.pos << " Die:" << move.die << std::endl;
      }
      return true; // All checks passed
    }

    bool LongNardeState::ValidateAction(Action action) const
    {
      if (IsChanceNode() || IsTerminal())
        return false; // Actions only valid for current player

      if (action < 0 || action >= NumDistinctActions())
      {
        return false;
      }

      // Perform consistency checks by decoding and simulating the action.
      try
      {
        std::vector<LongNardeCheckerMove> moves = LongNardeSpielMoveToCheckerMoves(cur_player_, action);
        // Simulate applying the moves on a cloned state to verify step-by-step validity
        std::unique_ptr<State> temp_state_ptr = this->Clone();
        LongNardeState *temp_state = dynamic_cast<LongNardeState *>(temp_state_ptr.get());

        // We will manipulate the dice_usage_count_ of the temp_state directly.

        bool sequence_valid = true;
        for (const auto &move : moves)
        {
          if (move.pos != kPassPos)
          {
            // Check if the required die (move.die) is available in the *temp_state*
            // according to its current dice_ and dice_usage_count_.
            bool found_usable_die_slot = false;
            for (int i = 0; i < temp_state->dice_.size(); ++i)
            {
              if (temp_state->dice_[i] == move.die && temp_state->IsDieUsable(i))
              {
                found_usable_die_slot = true;
                // Do not mark usage here; ApplyCheckerMove inside the loop will do it.
                break;
              }
            }

            if (!found_usable_die_slot)
            {
              sequence_valid = false;
              break;
            }
          }

          // Check move validity *in the context of the temporary state*
          // The moved_from_head state needs to be tracked across the loop for ValidateAction.
          // For now, passing true here might be incorrect, as it assumes the head rule applies
          // independently for each step rather than sequentially.
          // TODO: Refactor ValidateAction to track moved_from_head state sequentially.
          if (move.pos != kPassPos && !temp_state->LongNardeIsValidCheckerMove(temp_state->cur_player_, move, /*moved_from_head_this_sequence=*/true))
          {
            sequence_valid = false;
            break;
          }
          // Apply the move to the temp state for the next check
          // No longer need to track usage locally;
          // ApplyCheckerMove below will update temp_state->dice_usage_count_
          // Apply the move (LongNardeApplyCheckerMove handles marking dice used *within the temp state*)
          temp_state->LongNardeApplyCheckerMove(temp_state->cur_player_, move);
        }

        if (!sequence_valid)
        {
          // If any step was invalid, the whole action is invalid.
          return false;
        }

        // Optional: Add checks for dice usage rules (e.g., using max possible dice)?
        // For now, focus on breaking the recursion. If LegalActions filters correctly,
        // actions reaching here should already represent valid dice usage patterns.
      }
      catch (const std::exception &e)
      {
        return false; // Decoding/simulation error means invalid
      }
      catch (...)
      {
        return false; // Decoding/simulation error means invalid
      }

      return true; // Action survived simulation and decoding.
    }

    bool LongNardeState::IsOff(int player, int pos) const
    {
      return pos == kBearOffPos; // kBearOffPos is the special value indicating off the board
    }

    // ===== Home Region Checks =====

    bool LongNardeState::AllInHome(Player player) const
    {
      int checkers_on_board = 0;
      if (player == kXPlayerId)
      {
        // White's home is points 1-6 (indices 0-5)
        for (int i = kWhiteHomeEnd + 1; i < kNumPoints; ++i)
        {
          if (board(player, i) > 0)
          {
            return false; // Found checker outside home
          }
          // Note: We don't need to sum checkers_on_board here as the return false above handles it.
        }
        for (int i = kWhiteHomeStart; i <= kWhiteHomeEnd; ++i)
        {
          checkers_on_board += board(player, i);
        }
      }
      else
      { // kOPlayerId
        // Black's home is points 13-18 (indices 12-17)
        for (int i = 0; i < kBlackHomeStart; ++i)
        {
          if (board(player, i) > 0)
          {
            return false; // Found checker outside home
          }
          // Note: We don't need to sum checkers_on_board here.
        }
        for (int i = kBlackHomeEnd + 1; i < kNumPoints; ++i)
        {
          if (board(player, i) > 0)
          {
            return false; // Found checker outside home
          }
          // Note: We don't need to sum checkers_on_board here.
        }
        for (int i = kBlackHomeStart; i <= kBlackHomeEnd; ++i)
        {
          checkers_on_board += board(player, i);
        }
      }

      // If we reached here, no checkers were found outside the home board.
      // All checkers currently on the board are within the home region.
      return true; // Allow bearing off
    }

    // Finds the position of the furthest checker in the home board. Returns -1 if empty.
    int LongNardeState::FurthestCheckerInHome(Player player) const
    {
      if (player == kXPlayerId)
      { // White home: 0-5. Furthest is highest index.
        for (int pos = kWhiteHomeEnd; pos >= kWhiteHomeStart; --pos)
        {
          if (board_[player][pos] > 0)
          {
            return pos;
          }
        }
      }
      else
      { // Black home: 12-17. Furthest is lowest index.
        for (int pos = kBlackHomeStart; pos <= kBlackHomeEnd; ++pos)
        {
          if (board_[player][pos] > 0)
          {
            return pos;
          }
        }
      }
      return -1; // No checkers in home
    }

    // ===== Bridge Rule Checks =====

    // Checks if the die at the specified index in dice_ is usable.
    bool LongNardeState::IsDieUsable(int index) const
    {
      SPIEL_CHECK_GE(index, 0);
      SPIEL_CHECK_LT(index, dice_.size()); // dice_.size() is now 4
      int val = dice_[index];
      // A die is usable if its value is > 0 (not an empty slot)
      // and <= kNumDiceOutcomes (not marked as used).
      return val > 0 && val <= kNumDiceOutcomes;
    }

    // Checks if a given die *outcome* value (potentially marked as used) is usable.
    // This function might be less relevant now with 4 slots, but keep for compatibility/potential use.
    bool LongNardeState::UsableDiceOutcome(int outcome) const
    {
      // An outcome is usable if it's between 1 and 6 (inclusive).
      // Values 0 (empty slot) or > 6 (used marker) are not usable outcomes.
      return outcome >= 1 && outcome <= kNumDiceOutcomes;
    }

  } // namespace long_narde
} // namespace open_spiel

