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

    constexpr int kEncodingBase = 25; // 24 points + 1 for pass

    // ===== Encoding/Decoding Functions =====

    /**
     * @brief Encodes a sequence of checker moves (up to 4) into a single Spiel Action (int).
     *
     * Selects the appropriate encoding scheme based on whether it's a doubles roll
     * and how many moves are being made.
     * 1. Standard Scheme (non-doubles, or doubles <= 2 moves): Encodes two half-moves
     *    (padded with passes if needed) using base kDigitBase. Adds an offset if the
     *    original dice roll was low-die first. Range: [0, kDoublesOffset - 1].
     * 2. Special Doubles Scheme (doubles > 2 moves): Encodes up to 4 source positions
     *    using base kEncodingBaseDouble and adds kDoublesOffset.
     *    Range: [kDoublesOffset, NumDistinctActions() - 1].
     *
     * @param moves A vector of CheckerMove objects representing the full turn.
     * @return The encoded Spiel Action.
     */
    Action LongNardeState::LongNardeCheckerMovesToSpielMove(const std::vector<LongNardeCheckerMove> &moves) const {
      // Always encode two slots per phase (pad with pass if needed)
      std::vector<LongNardeCheckerMove> padded_moves = moves;
      while (padded_moves.size() < 2) padded_moves.push_back(kPassMove);

      // Determine high_pip_first_for_phase based on move ordering (higher pip first)
      bool high_pip_first_for_phase = (padded_moves[0].die >= padded_moves[1].die);

      // Encode positions (0-23 for points, 24 for pass)
      int slot0_pos = (padded_moves[0].pos == kPassPos) ? (kEncodingBase - 1) : padded_moves[0].pos;
      int slot1_pos = (padded_moves[1].pos == kPassPos) ? (kEncodingBase - 1) : padded_moves[1].pos;

      int action = slot0_pos + kEncodingBase * slot1_pos;
      if (!high_pip_first_for_phase) {
        action += kEncodingBase * kEncodingBase;
      }
      return action;
    }

    /**
     * @brief Decodes a Spiel Action (int) back into a sequence of checker moves.
     *
     * Determines which encoding scheme was used based on the action value and calls
     * the appropriate internal decoding helper (DecodeSingleDigit or DecodeDoubles).
     *
     * @param player The player whose action is being decoded.
     * @param spiel_move The Spiel Action to decode.
     * @return A vector of CheckerMove objects representing the turn. May contain passes.
     */
    std::vector<LongNardeCheckerMove> LongNardeState::LongNardeSpielMoveToCheckerMoves(Player player, Action spiel_move) const {
      // Decode two-slot action
      bool high_pip_first_for_phase = (spiel_move < (kEncodingBase * kEncodingBase));
      if (!high_pip_first_for_phase) {
        spiel_move -= kEncodingBase * kEncodingBase;
      }
      int slot0_pos = spiel_move % kEncodingBase;
      int slot1_pos = (spiel_move / kEncodingBase) % kEncodingBase;

      // Determine which dice are active for this phase
      int die0, die1;
      bool is_doubles_roll = (initial_dice_.size() >= 4 && initial_dice_[0] == initial_dice_[1] && initial_dice_[0] > 0);
      if (is_doubles_roll) {
        // Doubles: use DiceValue() to get actual pip values (1-6) even if marked used
        if (is_first_phase_of_doubles_) {
          die0 = DiceValue(0);
          die1 = DiceValue(1);
        } else {
          die0 = DiceValue(2);
          die1 = DiceValue(3);
        }
      } else {
        // Non-doubles: use DiceValue() for the two dice
        die0 = DiceValue(0);
        die1 = DiceValue(1);
      }
      
      // Determine higher and lower die values
      int higher_die = std::max(die0, die1);
      int lower_die = std::min(die0, die1);
      
      // Order dice according to high_pip_first_for_phase
      int pip0 = high_pip_first_for_phase ? higher_die : lower_die;
      int pip1 = high_pip_first_for_phase ? lower_die : higher_die;

      std::vector<LongNardeCheckerMove> result;
      // Slot 0
      if (slot0_pos == kEncodingBase - 1) {
        result.push_back(LongNardeCheckerMove(kPassPos, kPassPos, pip0));
      } else {
        int to_pos = GetToPos(player, slot0_pos, pip0);
        if (to_pos < 0 || to_pos >= kNumPoints) to_pos = kBearOffPos;
        result.push_back(LongNardeCheckerMove(slot0_pos, to_pos, pip0));
      }
      // Slot 1
      if (slot1_pos == kEncodingBase - 1) {
        result.push_back(LongNardeCheckerMove(kPassPos, kPassPos, pip1));
      } else {
        int to_pos = GetToPos(player, slot1_pos, pip1);
        if (to_pos < 0 || to_pos >= kNumPoints) to_pos = kBearOffPos;
        result.push_back(LongNardeCheckerMove(slot1_pos, to_pos, pip1));
      }
      return result;
    }

    /**
     * @brief Returns the total number of distinct actions possible in the game.
     *
     * This is the maximum value an action can take + 1.
     * It's calculated based on the sizes of the two encoding ranges:
     * - Standard range size: kDoublesOffset
     * - Doubles range size: kEncodingBaseDouble^4
     *
     * @return The total number of distinct actions.
     */
    int LongNardeState::NumDistinctActions() const
    {
      // Two phases, two slots per phase, each slot: 25 (24 points + 1 pass)
      // 2 * 25 * 25 = 1250
      return 2 * kEncodingBase * kEncodingBase;
    }

  } // namespace long_narde
} // namespace open_spiel
