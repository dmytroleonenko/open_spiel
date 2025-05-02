#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

// Encode exactly one half-move (source position or pass) into an action ID (0..23 for points, 24 for pass).
Action LongNardeState::LongNardeCheckerMovesToSpielMove(
    const std::vector<LongNardeCheckerMove>& moves) const {
  SPIEL_CHECK_EQ(moves.size(), 1);
  const auto& move = moves[0];
  if (move.pos == kPassPos) {
    return kNumPoints;  // Pass action ID = 24
  }
  SPIEL_CHECK_GE(move.pos, 0);
  SPIEL_CHECK_LT(move.pos, kNumPoints);
  return move.pos;
}

// Decode a half-move action ID back into a corresponding LongNardeCheckerMove.
// Returns a vector of size 1.
std::vector<LongNardeCheckerMove> LongNardeState::LongNardeSpielMoveToCheckerMoves(
    Player player, Action action) const {
  SPIEL_CHECK_LE(action, kNumPoints);
  std::vector<LongNardeCheckerMove> result;
  if (action == kNumPoints) {
    // Pass action: Find the *highest* remaining usable die.
    int remaining_die = -1;
    int usable_dice_count = 0;
    for (int die_val : dice_) {
      if (die_val > 0) {
        remaining_die = std::max(remaining_die, die_val); // Find max usable die
        usable_dice_count++;
      }
    }
    // Ensure *at least one* die remains when decoding a pass
    SPIEL_CHECK_GE(usable_dice_count, 1);
    SPIEL_CHECK_GT(remaining_die, 0);
    result.emplace_back(kPassPos, kPassPos, remaining_die);
  } else {
    // Find the matching half-move among legal moves (highest die if multiple).
    auto half_moves = LongNardeGenerateAllHalfMoves(player, moved_from_head_);
    bool found = false;
    LongNardeCheckerMove best_move;
    for (const auto& m : half_moves) {
      if (m.pos == action && (!found || m.die > best_move.die)) {
        best_move = m;
        found = true;
      }
    }
    if (!found) {
      // This check is redundant now, but kept the log message for context
      SPIEL_CHECK_TRUE(found);
    }
    result.push_back(best_move);
  }
  return result;
}

// Return the fixed number of half-move actions in the game (kNumDistinctActions = 25).
int LongNardeState::NumDistinctActions() const {
  return kNumDistinctActions;
}

}  // namespace long_narde
}  // namespace open_spiel
