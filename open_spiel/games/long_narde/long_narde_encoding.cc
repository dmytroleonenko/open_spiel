#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <string>
#include <algorithm> // For std::max, std::min

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

// ===== Encoding/Decoding Constants (Implementation Details) =====

// Base for standard encoding (2 moves max). Based on 24 points * 6 dice + 6 pass moves.
const int kDigitBase = kNumPoints * 6 + 6; // 24 * 6 + 6 = 144 + 6 = 150

// Offset within the standard encoding range to distinguish pass moves.
const int kPassOffset = kNumPoints * 6; // 24 * 6 = 144

// Base for doubles encoding (4 moves max). Need 25 values: 0 for pass/unused, 1-24 for points.
const int kDoublesBase = kNumPoints + 1; // 24 + 1 = 25

// Precomputed powers of kDoublesBase (25) for efficient encoding/decoding.
// kDoublesBasePower[0] = 25^0 = 1
// kDoublesBasePower[1] = 25^1 = 25
// kDoublesBasePower[2] = 25^2 = 625
// kDoublesBasePower[3] = 25^3 = 15625
// kDoublesBasePower[4] = 25^4 = 390625 (used for encoding the die value)
const std::array<Action, 5> kDoublesBasePower = {
    1L, 25L, 625L, 15625L, 390625L
};

// Offset to distinguish doubles encoding range from standard encoding range.
// Calculated as the maximum standard action value + 1.
// Max standard action = (kDigitBase - 1) * kDigitBase + (kDigitBase - 1)
//                    = (149 * 150) + 149 = 22350 + 149 = 22499
// We also add the low_roll_first offset, which is kDigitBase^2 = 150^2 = 22500
// So, max standard action = 22499 + 22500 = 44999.
// kDoublesOffset must be >= 45000.
constexpr Action kDoublesOffset = 45000;

// --- Special Doubles Encoding (> 2 moves) ---

// Base used for encoding up to 4 checker source positions in the special doubles encoding scheme.
// Each position digit represents a source point (0-23) or a pass (encoded as 24).
// Value must be >= kNumPoints + 1 (24 + 1 = 25) to represent 24 points + pass.
constexpr int kEncodingBaseDouble = 25;

// --- Low-Roll-First Offset (Standard Encoding Only) ---
// In the standard encoding, an additional offset (kDigitBase * kDigitBase = 22500) is added
// if the *original* dice roll had the lower die value rolled first (e.g., 3 then 5).
// This distinguishes sequences resulting from low-roll-first vs. high-roll-first, even if
// the sequence of moves played is the same (e.g., playing the 5 first).
// This offset ensures that LegalActions which might always list the higher-die move first
// still produce distinct action IDs based on the underlying dice roll order.
// Max standard action = (149 * 150 + 149) + 22500 = 22499 + 22500 = 44999.
// kDoublesOffset (45000) starts just above this range.

// ===== Encoding/Decoding Helper Functions (Internal) =====
namespace {

// Encodes a single CheckerMove (normal or pass) into an integer digit.
// Used by the standard encoding scheme.
int EncodeSingleMove(const CheckerMove& move) {
  if (move.pos == kPassPos) {
    // Encode a pass move: kPassOffset + (die - 1), range 144-149.
    SPIEL_CHECK_GE(move.die, 1);
    SPIEL_CHECK_LE(move.die, 6);
    return kPassOffset + (move.die - 1);
  } else {
    // Encode a normal move: pos * 6 + (die - 1), range 0-143.
    SPIEL_CHECK_GE(move.pos, 0);
    SPIEL_CHECK_LT(move.pos, kNumPoints);
    SPIEL_CHECK_GE(move.die, 1);
    SPIEL_CHECK_LE(move.die, 6);
    return move.pos * 6 + (move.die - 1);
  }
}

// Decodes a single integer digit back into a CheckerMove.
// Used by the standard decoding scheme.
CheckerMove DecodeSingleDigit(int digit, Player player, const LongNardeState* state) {
  if (digit >= kPassOffset) { // Pass range (144-149)
    int die = (digit - kPassOffset) + 1;
    return CheckerMove(kPassPos, kPassPos, die);
  } else { // Normal move range (0-143)
    int pos = digit / 6;
    int die = (digit % 6) + 1;
    // Need the state context to calculate to_pos
    SPIEL_CHECK_TRUE(state != nullptr);
    int to_pos = state->GetToPos(player, pos, die);
    return CheckerMove(pos, to_pos, die);
  }
}

// Encodes up to four CheckerMoves (for a doubles roll) into a single integer.
// Uses a base-25 encoding scheme.
Action EncodeDoubles(const std::vector<CheckerMove>& moves, int die)
{
  // Doubles encoding: Base 25 encoding for up to 4 moves.
  // Each move is encoded as pos + 1 (1-24), or 0 for pass/unused.
  Action encoded_action = 0;
  for (int i = 0; i < 4; ++i) {
    int val = 0;
    if (i < moves.size() && moves[i].pos != kPassPos) {
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

// Decodes a Spiel action (in the doubles range) back into a vector of CheckerMoves.
std::vector<CheckerMove> DecodeDoubles(Action spiel_move, Player player, const LongNardeState* state)
{
  // Adjust the action value by removing the doubles offset.
  Action adjusted_action = spiel_move - kDoublesOffset;

  // Extract the die value (1-6) encoded with the highest power of 25.
  int die = (adjusted_action / kDoublesBasePower[4]) + 1;
  SPIEL_CHECK_GE(die, 1);
  SPIEL_CHECK_LE(die, 6);

  // Extract the encoded move values (0-24) for each of the 4 potential moves.
  std::vector<CheckerMove> cmoves;
  Action remainder = adjusted_action % kDoublesBasePower[4];

  for (int i = 0; i < 4; ++i) {
    // Calculate the index for accessing powers in reverse order (3 down to 0)
    int power_index = 3 - i;
    // Extract the encoded value (0-24) for this move position.
    int val = remainder / kDoublesBasePower[power_index];
    remainder %= kDoublesBasePower[power_index]; // Update remainder

    if (val > 0) { // Encoded value > 0 corresponds to a normal move
      // Decode the source position (pos = val - 1).
      int pos = val - 1;
      SPIEL_CHECK_GE(pos, 0);
      SPIEL_CHECK_LT(pos, kNumPoints);
      // Calculate the destination position using the state context.
      SPIEL_CHECK_TRUE(state != nullptr);
      int to_pos = state->GetToPos(player, pos, die);
      // Add the decoded move to the list.
      cmoves.push_back(CheckerMove(pos, to_pos, die));
    } else {
      // Encoded value 0 means this move slot was unused or a pass.
      // We don't add pass moves explicitly here; the absence indicates pass/unused.
      // If no moves are decoded, it implies all 4 were passed.
    }
  }
  return cmoves;
}

} // namespace

// ===== Encoding/Decoding Functions =====

// Encodes a sequence of checker moves (up to 4) into a single Spiel Action (int).
// It uses two distinct schemes:
// 1. Standard Scheme: For non-doubles rolls, or doubles rolls resulting in <= 2 moves.
//    - Encodes exactly two CheckerMoves (padding with Passes if needed).
//    - Uses kDigitBase and potentially a low-roll-first offset.
//    - Resulting action is in the range [0, kDoublesOffset - 1].
// 2. Special Doubles Scheme: For doubles rolls resulting in > 2 moves (typically 3 or 4).
//    - Encodes up to 4 source positions (0-23, or 24 for Pass).
//    - Uses kEncodingBaseDouble and adds kDoublesOffset.
//    - Resulting action is in the range [kDoublesOffset, NumDistinctActions() - 1].
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
        // Max value is 23 * 6 + 5 = 143.
        // This ensures no overlap with the pass encoding range (144-149).
        SPIEL_CHECK_GE(move.pos, 0);
        SPIEL_CHECK_LT(move.pos, kNumPoints);
        SPIEL_CHECK_GE(move.die, 1);
        SPIEL_CHECK_LE(move.die, 6);
        return move.pos * 6 + (move.die - 1); // 0..143
      }
    };

    // Encode the first two (potentially padded) moves using the helper.
    int dig0 = EncodeSingleMove(encoded_moves[0]); // First half-move
    int dig1 = EncodeSingleMove(encoded_moves[1]); // Second half-move

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

// Decodes a Spiel Action (int) back into a sequence of checker moves.
// Handles both the standard and special doubles encoding schemes based on the action value.
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

    // Decode the two digits using the helper.
    std::vector<CheckerMove> cmoves;
    cmoves.push_back(DecodeSingleDigit(dig0, player, this));
    cmoves.push_back(DecodeSingleDigit(dig1, player, this));

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
