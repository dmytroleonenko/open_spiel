#include "open_spiel/games/long_narde/long_narde.h"

#include <string>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
  namespace long_narde
  {

    // ===== Movement Functions =====

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
    int LongNardeState::GetToPos(int player, int from_pos, int pips) const
    {
      SPIEL_CHECK_GE(from_pos, 0);
      SPIEL_CHECK_LT(from_pos, kNumPoints);
      SPIEL_CHECK_GE(pips, 1);
      SPIEL_CHECK_LE(pips, 6);

      if (player == kXPlayerId)
      {
        int target_idx = from_pos - pips;
        return target_idx;
      }
      else
      {
        // Black: check if move starts within or reaches the bear-off zone (home board 12-17)
        if (from_pos >= kBlackHomeStart && from_pos <= kBlackHomeEnd)
        {
          int pips_needed_to_bear_off = (from_pos - kBlackHomeStart + 1);
          if (pips >= pips_needed_to_bear_off)
          {
            return kBearOffPos;
          }
        }

        // Normal move calculation (counter-clockwise, wrap 0->23)
        int current_pos = from_pos;
        for (int i = 0; i < pips; ++i)
        {
          if (current_pos == 0)
          {
            current_pos = 23;
          }
          else
          {
            current_pos--;
          }
        }
        SPIEL_CHECK_GE(current_pos, 0);
        SPIEL_CHECK_LT(current_pos, kNumPoints);
        return current_pos;
      }
    }

  } // namespace long_narde
} // namespace open_spiel
