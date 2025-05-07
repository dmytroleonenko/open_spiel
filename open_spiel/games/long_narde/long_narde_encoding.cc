#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <string>
#include <algorithm> // For std::max, std::min
#include <array>     // For std::array (needed for kDoublesBasePower)

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
  namespace long_narde
  {

    // ===== Encoding/Decoding Constants =====

    // Long Narde uses a complex encoding scheme to represent a player's full turn
    // (potentially involving multiple checker movements) as a single integer Action.
    // There are two main schemes used:

    // --- Scheme 1: Encoding for Non-Doubles Turns (or Doubles with <= 2 moves) ---
    // Each individual half-move (moving one checker by one die's value) is encoded
    // into a "digit".
    // - Regular move: digit = `pos * 6 + (die - 1)`. `pos` is 0-23, `die` is 1-6.
    //   Range: 0 * 6 + (1 - 1) = 0  to  23 * 6 + (6 - 1) = 138 + 5 = 143.
    // - Pass move: digit = `kPassOffset + (die - 1)`. `die` is 1-6.
    //   `kPassOffset` is chosen to be 144, so the range is 144 to 149.
    // A full turn consists of up to two such half-moves. These two "digits" (d0, d1)
    // are combined using a base, `kDigitBase`.
    // The action is roughly `d1 * kDigitBase + d0`.
    // An additional offset (`kDigitBase * kDigitBase`) can be added to indicate
    // if the higher or lower die was used first, if necessary.

    constexpr int kDigitBase = 150;  // Base used to combine two half-move "digits".
                                     // Must be >= 150 to accommodate the max digit value (149).
    constexpr int kPassOffset = 144; // Offset for encoding pass half-moves.
                                     // Starts after the max regular move digit (143).

    // --- Scheme 2: Encoding for Doubles Turns (with > 2 moves) ---
    // When a player rolls doubles and can make more than two moves (up to four),
    // a different encoding is used. This scheme encodes the *starting positions*
    // of the checkers being moved.
    // It uses base-25 (`kEncodingBaseDouble`) because there are 24 board points (0-23)
    // plus a special value (24) to represent a pass or unused move slot.
    // The four positions (p0, p1, p2, p3, with p0 being the least significant)
    // are combined: `p3*B^3 + p2*B^2 + p1*B^1 + p0*B^0`, where B = `kEncodingBaseDouble`.
    // An offset (`kDoublesOffset`) is added to distinguish these doubles actions
    // from the non-doubles actions encoded using Scheme 1.

    constexpr int kEncodingBaseDouble = 25;                     // Base for encoding the *positions* in doubles moves (0-23 for points, 24 for pass).
    constexpr int kDoublesOffset = 2 * kDigitBase * kDigitBase; // Offset added to doubles actions.
                                                                // Chosen to be larger than the maximum possible non-doubles action
                                                                // (which is approx. `1 * kDigitBase^2 + (kDigitBase-1)*kDigitBase + (kDigitBase-1)`).

    // ===== Encoding/Decoding Helper Functions (Internal) =====
    namespace
    {

      // Precomputed powers of kEncodingBaseDouble (25) for efficient encoding/decoding.
      // kDoublesBasePower[0] = 25^0 = 1
      // kDoublesBasePower[1] = 25^1 = 25
      // kDoublesBasePower[2] = 25^2 = 625
      // kDoublesBasePower[3] = 25^3 = 15625
      // kDoublesBasePower[4] = 25^4 = 390625 (used for encoding the die value in DecodeDoubles)
      const std::array<Action, 5> kDoublesBasePower = {
          1L, 25L, 625L, 15625L, 390625L};

      /**
       * @brief Encodes a single CheckerMove (normal or pass) into an integer digit.
       *
       * This is used by the standard encoding scheme (non-doubles or doubles <= 2 moves).
       * - Normal move: digit = `pos * 6 + (die - 1)`, range [0, 143].
       * - Pass move: digit = `kPassOffset + (die - 1)`, range [144, 149].
       *
       * @param move The CheckerMove to encode.
       * @return The encoded integer digit.
       */
      int EncodeSingleMove(const LongNardeCheckerMove &move)
      {
        if (move.pos == kPassPos)
        {
          // Encode a pass move: kPassOffset + (die - 1), range 144-149.
          SPIEL_CHECK_GE(move.die, 1);
          SPIEL_CHECK_LE(move.die, 6);
          return kPassOffset + (move.die - 1);
        }
        else
        {
          // Encode a normal move: pos * 6 + (die - 1), range 0-143.
          SPIEL_CHECK_GE(move.pos, 0);
          SPIEL_CHECK_LT(move.pos, kNumPoints);
          SPIEL_CHECK_GE(move.die, 1);
          SPIEL_CHECK_LE(move.die, 6);
          return move.pos * 6 + (move.die - 1);
        }
      }

      /**
       * @brief Decodes a single integer digit back into a CheckerMove.
       *
       * Used by the standard decoding scheme.
       * Requires the state context to calculate the `to_pos` for normal moves.
       *
       * @param digit The integer digit to decode (0-149).
       * @param player The player making the move (needed for GetToPos).
       * @param state A pointer to the current game state (needed for GetToPos).
       * @return The decoded CheckerMove.
       */
      LongNardeCheckerMove DecodeSingleDigit(int digit, Player player, const LongNardeState *state)
      {
        if (digit >= kPassOffset)
        { // Pass range (144-149)
          int die = (digit - kPassOffset) + 1;
          return LongNardeCheckerMove(kPassPos, kPassPos, die);
        }
        else
        { // Normal move range (0-143)
          int pos = digit / 6;
          int die = (digit % 6) + 1;
          // Need the state context to calculate to_pos
          SPIEL_CHECK_TRUE(state != nullptr);
          // Check if this move would result in a bear-off
          int calculated_to_pos = state->GetToPos(player, pos, die);
          if (calculated_to_pos < 0)
          {
            // If it's any type of bear-off, return with kBearOffPos
            return LongNardeCheckerMove(pos, kBearOffPos, die);
          }
          else
          {
            // Otherwise, it's a regular move on the board
            return LongNardeCheckerMove(pos, calculated_to_pos, die);
          }
        }
      }

      /**
       * @brief Encodes up to four CheckerMoves (for a doubles roll with > 2 moves) into a single integer Action.
       *
       * Uses a special base-25 encoding scheme for the source positions:
       * - Each position (0-23) is encoded as `pos`.
       * - A pass move (kPassPos) is encoded as `kEncodingBaseDouble - 1` (24).
       * - The four encoded positions (p0, p1, p2, p3) are combined: `p3*B^3 + p2*B^2 + p1*B^1 + p0*B^0`, where B = kEncodingBaseDouble.
       * - The implicit die value (1-6) is not directly encoded in the positions part.
       * - The final action adds kDoublesOffset to distinguish it from the standard encoding.
       *
       * @param moves A vector containing 3 or 4 CheckerMoves (padding with passes if necessary).
       * @param die The die value rolled (the doubles value, 1-6).
       * @return The encoded Spiel Action, guaranteed to be >= kDoublesOffset.
       */
      Action EncodeDoubles(const std::vector<LongNardeCheckerMove> &moves, int die)
      {
        // Doubles encoding: Base 25 encoding for up to 4 moves.
        // Each move is encoded as pos + 1 (1-24), or 0 for pass/unused.
        Action encoded_action = 0;
        for (int i = 0; i < 4; ++i)
        {
          int val = 0;
          if (i < moves.size() && moves[i].pos != kPassPos)
          {
            // Encode actual move: pos + 1 (range 1-24)
            SPIEL_CHECK_GE(moves[i].pos, 0);
            SPIEL_CHECK_LT(moves[i].pos, kNumPoints);
            val = moves[i].pos + 1;
          } // Otherwise, val remains 0 for pass/unused move

          // Add the encoded move value to the total action, scaled by base 25^i.
          encoded_action += val * kDoublesBasePower[i];
        }

        // Add the die value (0-5) scaled by the highest power of 25.
        encoded_action += (die - 1) * kDoublesBasePower[4];

        // Apply the final offset for the doubles range.
        return encoded_action + kDoublesOffset;
      }

      /**
       * @brief Decodes a Spiel action (in the special doubles range) back into a vector of CheckerMoves.
       *
       * Reverses the EncodeDoubles process.
       * Extracts the 4 encoded positions and reconstructs the CheckerMoves.
       * The die value is implicit (same for all moves) and must be provided or inferred.
       * Note: The returned vector might contain pass moves if fewer than 4 actual moves were encoded.
       *
       * @param spiel_move The Spiel Action to decode (must be >= kDoublesOffset).
       * @param player The player who made the move.
       * @param state A pointer to the state (needed for GetToPos calculation).
       * @return A vector containing up to 4 CheckerMoves (potentially including passes).
       */
      std::vector<LongNardeCheckerMove> DecodeDoubles(Action spiel_move, Player player, const LongNardeState *state)
      {
        // Adjust the action value by removing the doubles offset.
        Action adjusted_action = spiel_move - kDoublesOffset;

        // Extract the die value (1-6) encoded with the highest power of 25.
        int die = (adjusted_action / kDoublesBasePower[4]) + 1;
        SPIEL_CHECK_GE(die, 1);
        SPIEL_CHECK_LE(die, 6);

        // Extract the encoded move values (0-24) for each of the 4 potential moves.
        std::vector<LongNardeCheckerMove> cmoves;
        Action remainder = adjusted_action % kDoublesBasePower[4];

        for (int i = 0; i < 4; ++i)
        {
          // Calculate the index for accessing powers in reverse order (3 down to 0)
          int power_index = 3 - i;
          // Extract the encoded value (0-24) for this move position.
          int val = remainder / kDoublesBasePower[power_index];
          remainder %= kDoublesBasePower[power_index]; // Update remainder

          if (val > 0)
          { // Encoded value > 0 corresponds to a normal move
            // Decode the source position (pos = val - 1).
            int pos = val - 1;
            SPIEL_CHECK_GE(pos, 0);
            SPIEL_CHECK_LT(pos, kNumPoints);
            // Calculate the destination position using the state context.
            SPIEL_CHECK_TRUE(state != nullptr);
            int calculated_to_pos = state->GetToPos(player, pos, die);
            if (calculated_to_pos < 0)
            {
              cmoves.push_back(LongNardeCheckerMove(pos, kBearOffPos, die));
            }
            else
            {
              cmoves.push_back(LongNardeCheckerMove(pos, calculated_to_pos, die));
            }
          }
          else
          {
            // Encoded value 0 means this move slot was unused or a pass.
            // We don't add pass moves explicitly here; the absence indicates pass/unused.
            // If no moves are decoded, it implies all 4 were passed.
          }
        }
        return cmoves;
      }

    } // namespace

    // ===== Encoding/Decoding Functions =====

    /**
     * @brief Encodes a sequence of 1 or 2 checker moves for the current phase into a single Spiel Action (int).
     *
     * This function implements the new action encoding scheme targeting a 1250 ID space.
     * It takes 1 or 2 checker moves for the current phase of the turn.
     * - Moves are padded with Pass moves to ensure exactly two moves are encoded.
     * - Positions are encoded (0-23 for points, 24 for Pass).
     * - A base action ID is calculated: `encoded_pos1 * 25 + encoded_pos0`.
     * - An offset of 625 is added if the first actual move did not use the
     *   higher of two distinct available dice for the phase, and it's not a double pass.
     *
     * @param phase_moves A vector containing 0, 1, or 2 `LongNardeCheckerMove`s for the current phase.
     *                    If 0 or 1, it will be padded with appropriate Pass moves.
     * @return The encoded Spiel Action (0-1249).
     */
    Action LongNardeState::LongNardeCheckerMovesToSpielMove(
        const std::vector<LongNardeCheckerMove> &initial_phase_moves) const
    {
      // Only handle up to two moves per phase; remove legacy doubles encoding.
      // This function should only be called with 0, 1, or 2 moves,
      // representing a single phase of a turn. Doubles (4 moves) will be
      // handled as two separate phases/actions by the game logic.
      SPIEL_CHECK_LE(initial_phase_moves.size(), 2);

      std::vector<LongNardeCheckerMove> current_phase_moves = initial_phase_moves;

      // Determine available dice for the current phase
      // bool is_second_phase = this->is_handling_second_phase_of_doubles_; (Will be used when state is updated)
      // For now, assume dice_ has 4 elements: d0, d1 for first phase; d2, d3 for second.
      // And that DiceValue(i) gives the die if >0 and available, or 0 if used/unavailable.

      // Placeholder: these would be derived from this->dice_ and is_second_phase
      // For example:
      // int die_idx1_for_phase = is_second_phase ? 2 : 0;
      // int die_idx2_for_phase = is_second_phase ? 3 : 1;
      // int d1_val = DiceValue(die_idx1_for_phase);
      // int d2_val = DiceValue(die_idx2_for_phase);

      // Simplified dice for padding (NEEDS ACCURATE STATE LOGIC)
      // This logic is a placeholder and must be replaced with logic that correctly
      // identifies available, unused dice from `this->dice_` for the current phase.
      int pad_die_val1 = 1; // Default if no specific dice info
      int pad_die_val2 = 1; // Default

      // Attempt to get real dice values for padding if available
      // This is highly speculative without the actual state variables and logic for current phase dice.
      if (DiceValue(is_handling_second_phase_of_doubles_ ? 2 : 0) > 0) {
          pad_die_val1 = DiceValue(is_handling_second_phase_of_doubles_ ? 2 : 0);
      }
      if (DiceValue(is_handling_second_phase_of_doubles_ ? 3 : 1) > 0) {
          pad_die_val2 = DiceValue(is_handling_second_phase_of_doubles_ ? 3 : 1);
      }


      if (current_phase_moves.empty())
      {
        current_phase_moves.push_back({kPassPos, kPassPos, pad_die_val1});
        current_phase_moves.push_back({kPassPos, kPassPos, pad_die_val2});
      }
      else if (current_phase_moves.size() == 1)
      {
        int die_for_padding_pass = pad_die_val1; // default
        // If the first move exists and used one of the pad_die_vals, use the other for padding.
        if (current_phase_moves[0].pos != kPassPos) {
            if (current_phase_moves[0].die == pad_die_val1 && DiceValue(is_handling_second_phase_of_doubles_ ? 3 : 1) > 0) {
                die_for_padding_pass = pad_die_val2;
            } else if (current_phase_moves[0].die == pad_die_val2 && DiceValue(is_handling_second_phase_of_doubles_ ? 2 : 0) > 0) {
                die_for_padding_pass = pad_die_val1;
            } else {
                // If the move's die doesn't match either, or only one pad_die_val is valid, pick one.
                // Prefer pad_die_val2 if pad_die_val1 was "conceptually" used or if pad_die_val2 is valid.
                 if (DiceValue(is_handling_second_phase_of_doubles_ ? 3 : 1) > 0) die_for_padding_pass = pad_die_val2;
                 else if (DiceValue(is_handling_second_phase_of_doubles_ ? 2 : 0) > 0) die_for_padding_pass = pad_die_val1;
            }
        } else { // The single move is a pass itself.
            // Pad with the "other" die if possible, or the same if only one is notionally available.
            if (DiceValue(is_handling_second_phase_of_doubles_ ? 3 : 1) > 0 && current_phase_moves[0].die != pad_die_val2) {
                 die_for_padding_pass = pad_die_val2;
            } else if (DiceValue(is_handling_second_phase_of_doubles_ ? 2 : 0) > 0) {
                 die_for_padding_pass = pad_die_val1;
            }
        }
        current_phase_moves.push_back({kPassPos, kPassPos, die_for_padding_pass});
      }

      // Encode positions: 0-23 for points, 24 for kPassPos.
      int encoded_pos0 = (current_phase_moves[0].pos == kPassPos) ? 24 : current_phase_moves[0].pos;
      int encoded_pos1 = (current_phase_moves[1].pos == kPassPos) ? 24 : current_phase_moves[1].pos;

      SPIEL_CHECK_GE(encoded_pos0, 0);
      SPIEL_CHECK_LE(encoded_pos0, 24);
      SPIEL_CHECK_GE(encoded_pos1, 0);
      SPIEL_CHECK_LE(encoded_pos1, 24);

      Action base_action_id = encoded_pos1 * 25 + encoded_pos0;

      bool first_move_used_actual_high_die_for_phase = true; // Default to true (no offset)
      bool both_moves_are_passes = (current_phase_moves[0].pos == kPassPos && current_phase_moves[1].pos == kPassPos);

      if (!both_moves_are_passes) {
        // Determine `first_move_used_actual_high_die_for_phase` based on `this->dice_` for the current phase
        // and `current_phase_moves`.
        // This requires knowing which dice from `this->dice_` correspond to `current_phase_moves[0].die`
        // and `current_phase_moves[1].die` (if it's not a pass).

        // Simplified placeholder logic:
        // Get the two active dice for the phase from state->dice_
        // bool is_second_phase = this->is_handling_second_phase_of_doubles_;
        // int d_phase_1_idx = is_second_phase ? 2 : 0;
        // int d_phase_2_idx = is_second_phase ? 3 : 1;
        // int die_val_phase_1 = this->DiceValue(d_phase_1_idx);
        // int die_val_phase_2 = this->DiceValue(d_phase_2_idx);

        // This is a critical placeholder and needs accurate implementation
        // based on how available dice are identified for the phase.
        // The logic should be:
        // 1. Identify the two distinct dice values available for this phase (e.g., phase_die_A, phase_die_B).
        // 2. If only one distinct die value is available (e.g. doubles, or only one die left), then
        //    `first_move_used_actual_high_die_for_phase` is true (no concept of higher/lower).
        // 3. If two distinct dice (phase_die_A, phase_die_B with phase_die_A != phase_die_B):
        //    Let higher_phase_die = max(phase_die_A, phase_die_B).
        //    Let lower_phase_die = min(phase_die_A, phase_die_B).
        //    If current_phase_moves[0] is not a pass:
        //      If current_phase_moves[0].die == lower_phase_die, then first_move_used_actual_high_die_for_phase = false.
        //      Else (it used higher_phase_die), it's true.
        //    Else (current_phase_moves[0] is a pass, but move 1 is not):
        //      If current_phase_moves[1].die == lower_phase_die, then first_move_used_actual_high_die_for_phase = false. (Arguably, this case means the "first actual move" used the low die)
        //      This part of the logic needs careful handling: the rule is about the *first move that used a die*.

        // For now, using a very simplified placeholder.
        // This assumes dice1 and dice2 parameters were somehow correctly passed or derived for the phase.
        // And that first_move_used_higher_die was correctly set based on *those phase dice*.
        // This will be wrong until state integration.
        // Let's assume `dice1_for_phase` and `dice2_for_phase` are available
        // And `move0_die` is `current_phase_moves[0].die`
        
        // Placeholder values for phase dice
        int d1_phase = 1, d2_phase = 1;
        if (this->dice_.size() >=2 ) { // Basic safety
            d1_phase = this->DiceValue(this->is_handling_second_phase_of_doubles_ ? 2 : 0);
            d2_phase = this->DiceValue(this->is_handling_second_phase_of_doubles_ ? 3 : 1);
        }


        if (current_phase_moves[0].pos != kPassPos) { // If first move is an actual move
            if (d1_phase > 0 && d2_phase > 0 && d1_phase != d2_phase) { // Two distinct available dice
                if (current_phase_moves[0].die == std::min(d1_phase, d2_phase)) {
                    first_move_used_actual_high_die_for_phase = false;
                } else {
                    first_move_used_actual_high_die_for_phase = true;
                }
            } else { // One distinct die or less, so no "lower die first" penalty
                first_move_used_actual_high_die_for_phase = true;
            }
        } else if (current_phase_moves.size() > 1 && current_phase_moves[1].pos != kPassPos) { // First is pass, second is actual
             if (d1_phase > 0 && d2_phase > 0 && d1_phase != d2_phase) { // Two distinct available dice
                if (current_phase_moves[1].die == std::min(d1_phase, d2_phase)) {
                    // If the only actual move used the lower die, it implies "lower die first" scenario
                    first_move_used_actual_high_die_for_phase = false;
                } else {
                     first_move_used_actual_high_die_for_phase = true;
                }
            } else {
                first_move_used_actual_high_die_for_phase = true;
            }
        } else { // Both passes (already handled) or only one pass move (no dice preference)
            first_move_used_actual_high_die_for_phase = true;
        }
      }


      Action final_action_id = base_action_id;
      if (!first_move_used_actual_high_die_for_phase && !both_moves_are_passes) {
          final_action_id += (25 * 25); // 625
      }
      
      return final_action_id; // Range 0 to 1249. Total 1250 distinct actions.
    }

    /**
     * @brief Decodes a Spiel Action (0-1249) back into a pair of `LongNardeCheckerMove`s for the current phase.
     *
     * This function reverses the encoding performed by `LongNardeCheckerMovesToSpielMove`.
     * It determines if an offset was applied (indicating the lower of two distinct dice was used first).
     * It decodes the two positions (0-23 for points, 24 for Pass).
     * Crucially, it infers the dice values for these moves based on the state's current `dice_`
     * array for the active phase and the `first_move_used_actual_high_die` flag derived from the action ID.
     *
     * @param player The player making the move.
     * @param spiel_move_id The Spiel Action ID to decode (0-1249).
     * @return A vector containing exactly two `LongNardeCheckerMove`s. These may include Pass moves.
     *         The `to_pos` for actual moves is calculated using `GetToPos`.
     */
    std::vector<LongNardeCheckerMove> LongNardeState::LongNardeSpielMoveToCheckerMoves(
        Player player, Action spiel_move_id) const
    {
      SPIEL_CHECK_GE(spiel_move_id, 0);
      SPIEL_CHECK_LT(spiel_move_id, kNumDistinctActions); // kNumDistinctActions will be 1250

      Action base_action_id = spiel_move_id;
      bool first_move_used_actual_high_die_for_phase = true; // Default: no offset means high die first or not applicable

      if (spiel_move_id >= (25 * 25)) // Offset for lower die first is 625
      {
        base_action_id = spiel_move_id - (25 * 25);
        first_move_used_actual_high_die_for_phase = false; // Offset was applied, so lower die was used first by a non-pass move.
      }

      int encoded_pos1 = base_action_id / 25;
      int encoded_pos0 = base_action_id % 25;

      SPIEL_CHECK_GE(encoded_pos0, 0);
      SPIEL_CHECK_LE(encoded_pos0, 24);
      SPIEL_CHECK_GE(encoded_pos1, 0);
      SPIEL_CHECK_LE(encoded_pos1, 24);

      int actual_pos0 = (encoded_pos0 == 24) ? kPassPos : encoded_pos0;
      int actual_pos1 = (encoded_pos1 == 24) ? kPassPos : encoded_pos1;

      // Infer dice for move0 and move1 based on state->dice_ for the current phase
      // and first_move_used_actual_high_die_for_phase.
      // This is a critical part and relies on state variables being correctly set.

      // Placeholder: these would be derived from this->dice_ and this->is_handling_second_phase_of_doubles_
      // int die_idx1_for_phase = this->is_handling_second_phase_of_doubles_ ? 2 : 0;
      // int die_idx2_for_phase = this->is_handling_second_phase_of_doubles_ ? 3 : 1;
      // int d1_val_phase = this->DiceValue(die_idx1_for_phase);
      // int d2_val_phase = this->DiceValue(die_idx2_for_phase);
      
      // Simplified placeholder for dice values available in the phase
      int d1_phase = 1, d2_phase = 1; // Default values
      if (this->dice_.size() >=2 ) { // Basic safety
          d1_phase = this->DiceValue(this->is_handling_second_phase_of_doubles_ ? 2 : 0);
          d2_phase = this->DiceValue(this->is_handling_second_phase_of_doubles_ ? 3 : 1);
      }
      // Ensure they are actual die values (1-6) if > 0, otherwise treat as unavailable (e.g. 0)
      if (d1_phase > 6) d1_phase = 0; 
      if (d2_phase > 6) d2_phase = 0;


      int die_for_move0 = 0;
      int die_for_move1 = 0;

      bool move0_is_pass = (actual_pos0 == kPassPos);
      bool move1_is_pass = (actual_pos1 == kPassPos);

      // Determine dice based on availability and the high_die_first flag
      if (d1_phase > 0 && d2_phase > 0 && d1_phase != d2_phase) { // Two distinct, available dice for the phase
        int higher_die = std::max(d1_phase, d2_phase);
        int lower_die = std::min(d1_phase, d2_phase);

        if (!move0_is_pass && !move1_is_pass) { // Both are actual moves
          if (first_move_used_actual_high_die_for_phase) {
            die_for_move0 = higher_die;
            die_for_move1 = lower_die;
          } else {
            die_for_move0 = lower_die;
            die_for_move1 = higher_die;
          }
        } else if (!move0_is_pass) { // Move 0 is actual, Move 1 is pass
          // If high_die_first is true, move0 used higher. If false, move0 used lower.
          die_for_move0 = first_move_used_actual_high_die_for_phase ? higher_die : lower_die;
          die_for_move1 = first_move_used_actual_high_die_for_phase ? lower_die : higher_die; // Pass uses the other die
        } else if (!move1_is_pass) { // Move 0 is pass, Move 1 is actual
          // If high_die_first is true, it means the conceptual first die (higher) was for the pass.
          // So the actual move (move1) must have used the lower die.
          // If high_die_first is false, it means the conceptual first die (lower) was for the pass.
          // So the actual move (move1) must have used the higher die.
          die_for_move0 = first_move_used_actual_high_die_for_phase ? higher_die : lower_die; // Pass uses this
          die_for_move1 = first_move_used_actual_high_die_for_phase ? lower_die : higher_die; // Actual move uses this
        } else { // Both are passes
          // Order doesn't matter as much, but assign consistently.
          die_for_move0 = higher_die; 
          die_for_move1 = lower_die;
        }
      } else if (d1_phase > 0) { // Only d1_phase is available (or d1_phase == d2_phase)
        die_for_move0 = d1_phase;
        die_for_move1 = d1_phase; // If d2 was 0, effectively d1 is the only option for both.
                                  // If d1 == d2, then this is correct.
      } else if (d2_phase > 0) { // Only d2_phase is available
        die_for_move0 = d2_phase;
        die_for_move1 = d2_phase;
      } else {
        // No dice available in phase? Should not happen if legal actions were generated.
        // Or, this might be a state where only pass-pass is possible.
        // For pass moves, we still need a die value. Default to 1 if none from state.
        die_for_move0 = move0_is_pass ? 1 : 0; 
        die_for_move1 = move1_is_pass ? 1 : 0;
        if (!move0_is_pass || !move1_is_pass) {
             SpielFatalError(absl::StrCat("LongNardeSpielMoveToCheckerMoves: No dice available in phase for non-pass move. spiel_move_id: ", spiel_move_id));
        }
      }
      
      // Safety check: ensure pass moves have a die.
      if (move0_is_pass && die_for_move0 == 0) die_for_move0 = 1; 
      if (move1_is_pass && die_for_move1 == 0) die_for_move1 = 1;


      std::vector<LongNardeCheckerMove> decoded_moves;
      decoded_moves.reserve(2);

      LongNardeCheckerMove move0(actual_pos0, actual_pos0 == kPassPos ? kPassPos : GetToPos(player, actual_pos0, die_for_move0), die_for_move0);
      LongNardeCheckerMove move1(actual_pos1, actual_pos1 == kPassPos ? kPassPos : GetToPos(player, actual_pos1, die_for_move1), die_for_move1);

      decoded_moves.push_back(move0);
      decoded_moves.push_back(move1);

      return decoded_moves;
    }

  } // namespace long_narde
} // namespace open_spiel
