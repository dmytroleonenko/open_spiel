/**
 *   Long Narde Rules:
 *  1. Setup: White's 15 checkers on point 24; Black's 15 on point 12.
 *  2. Movement: Both move checkers CCW into home (White 1–6, Black 13–18), then bear off.
 *  3. Starting: Each rolls 1 die; higher is White and goes first. But in our open_spiel implementation white is always first without the dice roll
 *  4. Turns: Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves exist, skip; if only one is possible, use the higher die.
 *  5. Head Rule: Only 1 checker may leave the head (White 24, Black 12) per turn. Exception on the first turn: if you roll double 6, 4, or 3, you can move 2 checkers from the head; after that, no more head moves.
 *  6. Bearing Off: Once all your checkers reach home, bear them off with exact or higher rolls.
 *  7. Ending/Scoring: Game ends when someone bears off all. If the loser has none off, winner scores 2 (mars); otherwise 1 (oin). Some events allow a last roll to tie.
 *  8. Block (Bridge): You cannot form a contiguous block of 6 checkers unless at least 1 opponent checker is still ahead of it. Fully trapping all 15 opponent checkers is banned—even a momentary (going through in a sequence of moves) 6‑block that would leave no opponent checkers in front is disallowed.
 */
#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>
#include <queue>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace { // Anonymous namespace

// ===== Game Definition and Constants =====
// All content moved to other files.

// ===== Encoding/Decoding Constants =====
// All content moved to long_narde_encoding.cc

} // End anonymous namespace

// ===== State Manipulation Helpers =====
// All content moved to other files.

// ===== Utility Functions =====
// All content moved to other files.

// ===== Validation Functions =====
// All content moved to other files.

}  // namespace long_narde
}  // namespace open_spiel