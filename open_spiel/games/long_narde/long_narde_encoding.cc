#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <string>
#include <algorithm> // For std::max, std::min

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

// ===== Encoding/Decoding Constants (Implementation Details) =====

// Base used to combine two half-move "digits" in the non-doubles encoding scheme.
// Must be >= 150 to accommodate the max digit value (149).
constexpr int kDigitBase = 150;

// Offset used to distinguish pass moves from normal moves in the non-doubles encoding.
// Must be > max normal move value (143).
constexpr int kPassOffset = 144;

// Base used for encoding up to 4 checker positions in the special doubles encoding scheme.
// Value must be >= kNumPoints + 1 (24 + 1 = 25) to represent 24 points + pass.
constexpr int kEncodingBaseDouble = 25;

// Offset added to actions encoded using the special doubles scheme to distinguish them
// from the standard (non-doubles) encoding scheme.
// This is set to 2 * kDigitBase * kDigitBase, which is the theoretical max of the standard scheme + 1.
constexpr Action kDoublesOffset = 2 * kDigitBase * kDigitBase; // 2 * 150 * 150 = 45000


// ===== Encoding/Decoding Functions =====

Action LongNardeState::CheckerMovesToSpielMove(
    const std::vector<CheckerMove>& moves) const {
  SPIEL_CHECK_LE(moves.size(), 4);  // Allow up to 4 moves for doubles

  // Check if this is a doubles roll based on the current dice state.
  bool is_doubles = false;
  if (dice_.size() == 2 && DiceValue(0) == DiceValue(1)) {
    is_doubles = true;
  }

  // Use a separate, higher-range encoding for doubles when more than 2 moves are made (up to 4).
  // This is necessary because the standard encoding only supports two half-moves.
  if (is_doubles && moves.size() > 2) {
    // Encode up to 4 checker source positions using a base-25 system.
    // 0-23 represent board points, 24 represents a pass (kPassPos).
    // The die value is implicit (it's the doubles value).
    std::vector<int> positions(4, kEncodingBaseDouble - 1);  // Default to pass (encoded as 24)
    
    // Fill the positions array with actual positions from the provided moves.
    for (size_t i = 0; i < moves.size() && i < 4; ++i) {
      if (moves[i].pos == kPassPos) {
        positions[i] = kEncodingBaseDouble - 1;  // kPassPos encoded as 24
      } else {
        SPIEL_CHECK_GE(moves[i].pos, 0);
        SPIEL_CHECK_LT(moves[i].pos, kNumPoints);
        positions[i] = moves[i].pos; // Store the 'from' position
      }
    }
    
    // Encode the 4 positions into a single integer using base-25.
    // positions[0] is the least significant digit, positions[3] is the most significant.
    Action action_double = 0;
    for (int i = 3; i >= 0; --i) {
      action_double = action_double * kEncodingBaseDouble + positions[i];
    }
    
    // Add kDoublesOffset to distinguish this encoding from the non-doubles scheme.
    // The final action value will be >= kDoublesOffset.
    Action action = kDoublesOffset + action_double;
    
    SPIEL_CHECK_GE(action, kDoublesOffset); // Ensure it's in the doubles range
    // No need to check upper bound here, as NumDistinctActions depends on constants defined here
    return action;
  } else {
    // Standard encoding for non-doubles rolls or doubles rolls with 0, 1, or 2 moves.
    // This scheme encodes two "half-moves" (CheckerMove) into a single action.
    // The sequence 'moves' is guaranteed by LegalActions to be valid in this order.
    // We encode moves[0] as dig0 and moves[1] as dig1 directly.
    std::vector<CheckerMove> encoded_moves = moves; // Use a copy to add padding if needed

    // Ensure we always encode exactly two half-moves by adding Pass moves if necessary.
    while (encoded_moves.size() < 2) {
      // Add pass moves as padding.
      int die_val = 1;  // Default die value for padding.
      // Try to find an *unused* die value if possible for the pass padding.
      // This helps preserve information if decoding is done without full state context,
      // although full state context is generally assumed.
      int available_die = -1;
      if (dice_.size() >= 1 && UsableDiceOutcome(dice_[0])) available_die = DiceValue(0);
      else if (dice_.size() >= 2 && UsableDiceOutcome(dice_[1])) available_die = DiceValue(1);

      if (available_die != -1) {
          die_val = available_die;
      } else if (!encoded_moves.empty() && encoded_moves[0].die > 0) {
          // Fallback: use the die from the first move if no dice info available (should not happen in normal flow).
          die_val = encoded_moves[0].die;
      }
      // Ensure die_val is valid (1-6).
      die_val = std::max(1, std::min(6, die_val));
      // Add a pass move with the chosen die value.
      encoded_moves.push_back(CheckerMove(kPassPos, kPassPos, die_val));
    }

    // Helper function to encode a single half-move (CheckerMove) into an integer digit.
    // This digit represents either a normal move or a pass move.
    auto encode_move = [](const CheckerMove& move) -> int {
      if (move.pos == kPassPos) {
        // Encode a pass move. Uses a specific offset (kPassOffset).
        // The value is kPassOffset + (die - 1), ranging from 144 to 149.
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return kPassOffset + (move.die - 1); // 144 + 0..5
      } else {
        // Encode a normal move from a board position.
        // The value is pos * 6 + (die - 1).
        // pos is 0-23, die is 1-6.
        // Max value is 23 * 6 + 5 = 138 + 5 = 143.
        // This ensures no overlap with the pass encoding range (144-149).
        SPIEL_CHECK_GE(move.pos, 0);
        SPIEL_CHECK_LT(move.pos, kNumPoints);
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return move.pos * 6 + (move.die - 1); // 0..143
      }
    };

    // Encode the first two (potentially padded) moves.
    int dig0 = encode_move(encoded_moves[0]); // First half-move
    int dig1 = encode_move(encoded_moves[1]); // Second half-move

    // Combine the two digits into a single action using base kDigitBase (150).
    // dig0 is the least significant digit, dig1 is the most significant.
    // Max value is 149 * 150 + 149 = 22350 + 149 = 22499.
    Action action = dig1 * kDigitBase + dig0;

    // Determine if the *actual* dice roll (if available) had the lower die first.
    // This is needed because LegalActions might reorder moves (e.g., highest die first).
    // We need to distinguish action (5, 3) from roll (5, 3) vs action (5, 3) from roll (3, 5).
    bool actual_low_roll_first = false;
    if (dice_.size() >= 2) {
        // Use DiceValue to handle potential internal encoding (7-12) if dice were marked used.
        int die0_val = DiceValue(0);
        int die1_val = DiceValue(1);
        if (die0_val < die1_val) {
            actual_low_roll_first = true;
        }
    }
    // If dice_.size() < 2 (e.g., chance node), actual_low_roll_first remains false.

    // Add a large offset (kDigitBase * kDigitBase = 150 * 150 = 22500)
    // if the actual dice roll had the lower die rolled first.
    // This distinguishes the two cases mentioned above.
    // Example: Roll (3, 5). Move using 5 then 3. Encoded as (dig1=move5, dig0=move3).
    //          Action = encode(move3) + encode(move5) * 150 + 22500.
    // Example: Roll (5, 3). Move using 5 then 3. Encoded as (dig1=move5, dig0=move3).
    //          Action = encode(move3) + encode(move5) * 150.
    // This offset is NOT added for double pass moves, as the dice order is irrelevant.
    bool is_double_pass = (encoded_moves.size() == 2 && encoded_moves[0].pos == kPassPos && encoded_moves[1].pos == kPassPos);
    if (actual_low_roll_first && !is_double_pass) {
      action += kDigitBase * kDigitBase; // Add offset ~22500
    }

    // Final sanity checks for the non-doubles encoding range.
    SPIEL_CHECK_GE(action, 0);
    // Ensure the action is below the start of the doubles encoding range.
    SPIEL_CHECK_LT(action, kDoublesOffset); // kDoublesOffset is typically 2 * kDigitBase * kDigitBase = 45000
    return action;
  }
}

std::vector<CheckerMove> LongNardeState::SpielMoveToCheckerMoves(
    Player player, Action spiel_move) const {
  // Check if the action falls within the special doubles encoding range.
  if (spiel_move >= kDoublesOffset) {
    // Decode a doubles action (up to 4 moves).
    Action action_double = spiel_move - kDoublesOffset; // Remove the offset
    
    // Extract the 4 encoded positions using base-25 decoding.
    std::vector<int> positions(4);
    for (int i = 0; i < 4; ++i) {
      // positions[i] will be 0-23 for a point, or 24 for a pass.
      positions[i] = action_double % kEncodingBaseDouble;
      action_double /= kEncodingBaseDouble;
    }

    // Determine the die value used for all moves in a doubles turn.
    int die_val = 1;  // Default if dice info isn't available (should not happen).
    if (dice_.size() > 0) {
      die_val = DiceValue(0);  // Use the value of the first die (they are the same).
    }

    // Reconstruct the CheckerMove objects from the decoded positions.
    std::vector<CheckerMove> cmoves;
    for (int i = 0; i < 4; ++i) {
      int pos;
      if (positions[i] == kEncodingBaseDouble - 1) {
        // Encoded value 24 corresponds to kPassPos.
        pos = kPassPos;
      } else {
        // Encoded value 0-23 corresponds to board points.
        pos = positions[i];
      }
      
      if (pos == kPassPos) {
        // Reconstruct a pass move. Note: to_pos is irrelevant for pass.
        cmoves.push_back(CheckerMove(kPassPos, kPassPos, die_val));
      } else {
        // Reconstruct a normal move. Calculate the destination position.
        int to_pos = GetToPos(player, pos, die_val);
        cmoves.push_back(CheckerMove(pos, to_pos, die_val));
      }
    }
    // Note: The returned vector might contain more than the actual number of
    // moves played if fewer than 4 moves were made (padded with passes during encoding).
    // The caller (e.g., DoApplyAction) needs to handle this, typically by stopping
    // after the actual number of moves needed or encountering the first pass.
    return cmoves;
  } else {
    // Decode a standard (non-doubles or doubles <= 2 moves) action.
    // Check if the low-roll-first offset was applied during encoding.
    bool high_roll_first = spiel_move < (kDigitBase * kDigitBase);
    if (!high_roll_first) {
      // Remove the offset if it was present.
      spiel_move -= kDigitBase * kDigitBase;
    }
    
    // Extract the two digits using base kDigitBase (150).
    int dig0 = spiel_move % kDigitBase; // First half-move (least significant)
    int dig1 = spiel_move / kDigitBase; // Second half-move (most significant)

    // Helper function to decode a single digit back into a CheckerMove.
    auto decode_digit = [this, player](int digit) -> CheckerMove {
      if (digit >= kPassOffset) { // Check if it's in the pass range (144-149)
        // Decode a pass move.
        int die = (digit - kPassOffset) + 1; // Extract die value (1-6)
        // Return a pass move. to_pos is irrelevant.
        return CheckerMove(kPassPos, kPassPos, die);
      } else { // Must be in the normal move range (0-143)
        // Decode a normal move.
        int pos = digit / 6;        // Extract source position (0-23)
        int die = (digit % 6) + 1;  // Extract die value (1-6)
        // Calculate the destination position.
        int to_pos = GetToPos(player, pos, die);
        // Return the reconstructed normal move.
        return CheckerMove(pos, to_pos, die);
      }
    };

    // Decode the two digits back into CheckerMoves.
    std::vector<CheckerMove> cmoves;
    cmoves.push_back(decode_digit(dig0)); // Decode first move
    cmoves.push_back(decode_digit(dig1)); // Decode second move
    
    // The order in cmoves matches the order they were encoded (e.g., highest die first if applied).
    // The 'high_roll_first' flag (derived from the offset) indicates the original dice roll order,
    // but the returned moves are in the potentially reordered sequence used for the action.
    return cmoves;
  }
}

int LongNardeState::NumDistinctActions() const {
  // The total number of distinct actions is the sum of the ranges used by the two encoding schemes.
  // 1. Non-doubles (and doubles <= 2 moves) encoding range:
  //    - Actions are encoded using two digits in base kDigitBase (150).
  //    - An offset of kDigitBase*kDigitBase (22500) is added if the original roll was low-die first.
  //    - This results in two potential ranges: [0, 22499] and [22500, 44999].
  //    - The start of the doubles encoding (kDoublesOffset) is set to the end of this range (45000).
  //    - So, this scheme uses the range [0, kDoublesOffset - 1].
  // 2. Doubles (> 2 moves) encoding range:
  //    - Actions are encoded using base kEncodingBaseDouble (25) for 4 positions.
  //    - An offset of kDoublesOffset (45000) is added.
  //    - The size of this range is kEncodingBaseDouble^4 = 25^4 = 390625.
  //    - This scheme uses the range [kDoublesOffset, kDoublesOffset + kEncodingBaseDouble^4 - 1].

  // Calculate the size of the doubles encoding range (kEncodingBaseDouble^4)
  int double_range_size = 1;
  for (int i = 0; i < 4; ++i) {
    double_range_size *= kEncodingBaseDouble;  // 25^4 = 390625
  }
  
  // The total number of distinct actions is the start of the doubles range plus the size of the doubles range.
  // This gives the upper bound for the highest possible action value.
  return kDoublesOffset + double_range_size; // 45000 + 390625 = 435625
}


} // namespace long_narde
} // namespace open_spiel
