#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> // For std::max, std::min, std::sort, std::unique
#include <iostream>  // For kDebugging cout/cerr
#include <memory>    // For unique_ptr
#include <utility>   // For std::pair
#include <stack>     // For iterative approach

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" // For error messages

namespace open_spiel {
namespace long_narde {

// ===== Legal/Illegal Action Generation =====

std::vector<Action> LongNardeState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  Player player = CurrentPlayer();

  // 1. Generate all possible raw move sequences using the iterative helper.
  std::vector<std::vector<CheckerMove>> movelist;
  // The initial_moveseq and max_moves parameters are unused in IterativeLegalMoves
  int max_non_pass_moves = IterativeLegalMoves({}, &movelist, 0);

  // 2. Filter the sequences to find the best ones (longest, most non-pass).
  // *** Update the call to match the new signature ***
  std::vector<std::vector<CheckerMove>> filtered_movelist;
  FilterBestMoveSequences(movelist, &filtered_movelist, max_non_pass_moves);

  // 3. Convert the best CheckerMove sequences into Spiel Actions.
  std::vector<Action> current_legal_moves;
  // *** Use the filtered_movelist directly ***
  for (const auto& moveseq : filtered_movelist) {
    Action action = CheckerMovesToSpielMove(moveseq);
    // Basic validation (optional but good practice)
    if (ValidateAction(action)) {
        current_legal_moves.push_back(action);
    } else {
        SpielFatalError(absl::StrCat("Generated invalid action: ", action, " from sequence."));
    }
  }

  // 4. Apply the "must play higher die" rule if applicable.
  // Note: This uses the *original* full movelist to determine if a higher die
  // could have been played.
  // *** Pass filtered_movelist to the higher die rule helper ***
  current_legal_moves = ApplyHigherDieRuleIfNeeded(current_legal_moves, movelist); // Pass original movelist here

  if (current_legal_moves.empty()) {
      SpielFatalError("LegalActions resulted in empty list after filtering/rules.");
  }

  return current_legal_moves;
}

std::vector<Action> LongNardeState::IllegalActions() const {
  std::vector<Action> illegal_actions;
  if (IsChanceNode() || IsTerminal()) return illegal_actions;
  
  // Check dice validity before proceeding
  if (dice_.size() < 2) return illegal_actions; // Cannot determine rolls
  
  int high_roll = DiceValue(0);
  int low_roll = DiceValue(1);
  if (high_roll < low_roll) std::swap(high_roll, low_roll);
  int kMaxActionId = NumDistinctActions();
  
  std::vector<Action> legal_actions = LegalActions(); // Get legal actions once
  std::set<Action> legal_set(legal_actions.begin(), legal_actions.end());
  
  for (Action action = 0; action < kMaxActionId; ++action) {
    if (legal_set.count(action)) continue; // Skip known legal actions
    
    // Simple heuristic check: Check if it decodes reasonably.
    // Full validation is complex and already done by LegalActions.
    // This mainly catches encoding ranges that don't make sense.
    try {
      std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
        // Basic check: pass moves should correspond to valid die values if possible
        bool is_pass = true;
        for(const auto& m : moves) {
           if (m.pos != kPassPos) {
               is_pass = false;
               break;
           }
        }
        if (is_pass) {
             // Check if die values encoded in pass make sense with current roll
             if (moves.size() >= 1 && moves[0].die != DiceValue(0) && moves[0].die != DiceValue(1)) {
                  illegal_actions.push_back(action);
                  continue;
             }
              if (moves.size() >= 2 && moves[1].die != DiceValue(0) && moves[1].die != DiceValue(1)) {
                  illegal_actions.push_back(action);
                  continue;
             }
        }
        // Further checks could be added, but might duplicate LegalActions logic.
        // The main goal is to identify actions outside the *possible* range or clearly invalid encodings.
        
    } catch (...) {
        // If decoding itself fails, it's illegal
        illegal_actions.push_back(action);
        continue;
    }
     // If it wasn't caught by simple checks and isn't legal, add it.
     if (!legal_set.count(action)) {
         illegal_actions.push_back(action);
     }

  }
  return illegal_actions;
}

// Helper function to generate all valid move sequences.
std::vector<std::vector<CheckerMove>> LongNardeState::GenerateMoveSequences(
    Player player, int max_moves) const {
  std::vector<std::vector<CheckerMove>> movelist; // Changed from std::set
  // Use the new iterative function
  IterativeLegalMoves({}, &movelist, max_moves); 

  // Sort and remove duplicates to mimic std::set behavior
  std::sort(movelist.begin(), movelist.end());
  movelist.erase(std::unique(movelist.begin(), movelist.end()), movelist.end());

  // DEBUG: Print movelist contents after generation
  if (kDebugging) {
    std::cout << "DEBUG GenerateMoveSequences (Player " << player << "): Movelist size = " << movelist.size() << " (after sort/unique)" << std::endl;
    // Optionally print details if needed
    /*
    for (const auto& seq : movelist) {
        std::cout << "  Seq: ";
        for(const auto& m : seq) { std::cout << "{" << m.pos << "," << m.to_pos << "," << m.die << "} "; }
        std::cout << std::endl;
    }
    */
  }
  return movelist;
}

std::set<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player) const {
  std::set<CheckerMove> half_moves;
  bool is_debugging = false; // Keep general debugging off unless needed
  
  if (is_debugging) {
    std::cout << "GenerateAllHalfMoves for player " << player << "\n";
    std::cout << "Dice: "; 
    for(int d : dice_) { std::cout << DiceValue(d) << (UsableDiceOutcome(d)?"":"(used)") << " "; }
    std::cout << "\nBoard:\n";
    std::cout << ToString(); // Use ToString for full board context
    std::cout << "All checkers in home? " << (AllInHome(player) ? "YES" : "NO") << "\n";
    std::cout << "Moved from head this turn? " << (moved_from_head_ ? "YES" : "NO") << "\n";
    std::cout << "Is first turn? " << (is_first_turn_ ? "YES" : "NO") << "\n";
  }
  
  // For each checker belonging to the player
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (board(player, pos) <= 0) continue;
    
    if (is_debugging) {
      std::cout << "  Checking checker at pos " << pos << " (point " << (player==kXPlayerId ? 24-pos : (pos<=11?12-pos:36-pos)) << ")\n";
    }
    
    // For each usable die
    for (int i = 0; i < dice_.size(); ++i) {
      int outcome = dice_[i];
      if (!UsableDiceOutcome(outcome)) {
        if (is_debugging) std::cout << "    Die " << DiceValue(i) << " (raw " << outcome <<") not usable, skipping\n";
        continue; // Skip used dice
      }
      
      int die_value = outcome; // Since UsableDiceOutcome passed, outcome is 1-6
      int to_pos = GetToPos(player, pos, die_value);
      
      if (is_debugging) {
        std::cout << "    Checking die " << die_value << ", calculated to_pos=" << to_pos 
                  << (IsOff(player, to_pos) ? " (Bear Off)" : "") << "\n";
      }
      
      // Check if this specific half-move is valid *now*
      // Crucially includes the head rule check based on current `moved_from_head_` state.
      bool is_valid = IsValidCheckerMove(player, pos, to_pos, die_value, /*check_head_rule=*/true); 
      
      if (is_valid) {
        half_moves.insert(CheckerMove(pos, to_pos, die_value));
        if (is_debugging) {
          std::cout << "    Added valid move: pos=" << pos << ", to_pos=" << to_pos 
                    << ", die=" << die_value << "\n";
        }
      }
    }
  }
  
  // If no valid moves were found after checking all checkers and dice,
  // the player *must* pass. We add a pass move placeholder.
  // The encoding function will handle assigning dice values to the pass.
  if (half_moves.empty()) {
       // Add a single pass move. LegalActions/Encoding will handle using correct dice.
       // Use die=1 as a placeholder.
      half_moves.insert(CheckerMove(kPassPos, kPassPos, 1)); 
      if (is_debugging) {
         std::cout << "  No regular moves found. Added placeholder pass move.\n";
      }
  }
  
  if (is_debugging) {
    std::cout << "Generated " << half_moves.size() << " potential half-moves for this step:\n";
    for (const auto& move : half_moves) {
      std::cout << "  - from=" << move.pos << ", to=" << move.to_pos 
                << ", die=" << move.die << "\n";
    }
  }
  
  return half_moves;
}

// ----- Iterative Implementation -----

// Struct to hold state for the iterative exploration (simpler version)
struct IterativeFrame {
  std::vector<CheckerMove> current_sequence;
  int depth; // To track recursion depth limit
};

// Iterative helper for LegalActions. Explores possible move sequences using DFS.
int LongNardeState::IterativeLegalMoves(const std::vector<CheckerMove>& /*initial_moveseq - unused*/,
                                        std::vector<std::vector<CheckerMove>>* movelist, // Changed from std::set*
                                        int /*max_moves_param - unused, depth limit controls*/) const {
  // Safety limits (same as recursive version)
  const size_t kMaxTotalSequences = 200;
  const size_t kMaxBranchingFactor = 30; // Limit branches explored from one node
  const int kMaxIterationDepth = 6; // Equivalent to kMaxRecursionDepth

  // *** Determine max moves allowed based on initial dice ***
  int max_allowed_moves = 0;
  if (this->dice_.size() >= 2) { // Check initial dice from the const 'this' state
      // Pass the index (0 or 1), not the die value, to DiceValue
      max_allowed_moves = (DiceValue(0) == DiceValue(1)) ? 4 : 2;
  }
  // If dice_.size() < 2, something is wrong, but max_allowed_moves=0 will prevent moves.

  std::stack<IterativeFrame> exploration_stack;

  // Clone the state *once* at the beginning to work with a mutable copy.
  auto current_state = absl::WrapUnique(static_cast<LongNardeState*>(this->Clone().release()));

  // Push the initial frame (empty sequence, depth 0)
  exploration_stack.push({{}, 0});

  int max_non_pass_found = 0; // Track the overall maximum non-pass moves

  while (!exploration_stack.empty()) {
    IterativeFrame current_frame = exploration_stack.top();
    exploration_stack.pop();

    // Check limits *before* generating moves for this frame
    bool sequence_limit_hit = (movelist->size() >= kMaxTotalSequences);
    if (sequence_limit_hit || current_frame.depth >= kMaxIterationDepth) {
      #ifndef NDEBUG
      if (sequence_limit_hit) std::cerr << "Warning: IterativeLegalMoves hit sequence limit (" << kMaxTotalSequences << ")" << std::endl;
      if (current_frame.depth >= kMaxIterationDepth) std::cerr << "Warning: IterativeLegalMoves hit depth limit (" << kMaxIterationDepth << ")" << std::endl;
      #endif
      // If a limit is hit, we consider the sequence leading *to* this point as a potential final sequence
      // Only add if it's non-empty (avoids adding empty sequence from initial state if limits are 0)
      if (!current_frame.current_sequence.empty()) {
         movelist->push_back(current_frame.current_sequence);
         int non_pass = 0;
         for(const auto& m : current_frame.current_sequence) if(m.pos != kPassPos) non_pass++;
         max_non_pass_found = std::max(max_non_pass_found, non_pass);
      }
      continue; // Stop exploring this path
    }

    // Generate all valid *single* half-moves from the *current state*
    // Note: cur_player_ changes during apply/undo, so get it now.
    Player player_to_move = current_state->cur_player_;
    std::set<CheckerMove> possible_next_moves = current_state->GenerateAllHalfMoves(player_to_move);

    // Check if the *current state* allows no further moves (pass or empty)
    bool no_moves_possible = possible_next_moves.empty() ||
                             (possible_next_moves.size() == 1 && possible_next_moves.begin()->pos == kPassPos);

    if (no_moves_possible) {
      // Base case for this path: No more moves from current state.
      // Add the sequence that led here if it's not empty.
      if (!current_frame.current_sequence.empty()) {
        movelist->push_back(current_frame.current_sequence);
        int non_pass = 0;
        for (const auto& m : current_frame.current_sequence) if (m.pos != kPassPos) non_pass++;
        max_non_pass_found = std::max(max_non_pass_found, non_pass);
        if (kDebugging) std::cout << "  Iterative: End of path (no moves). Added seq. Non-pass: " << non_pass << std::endl;
      }
      continue; // Finished exploring this path
    }

    // Explore potential next moves from the current state
    int branches_pushed = 0;
    // Iterate in reverse to maintain DFS order similar to recursion when pushing to stack
    for (auto it = possible_next_moves.rbegin(); it != possible_next_moves.rend(); ++it) {
        const CheckerMove& next_move = *it;
        if (next_move.pos == kPassPos) continue; // Skip placeholder pass moves here

        // --- Pre-move Check ---
        // Check bridge rule *before* applying the move.
        bool would_block = current_state->WouldFormBlockingBridge(player_to_move, next_move.pos, next_move.to_pos);
        if (would_block) {
             if (kDebugging) std::cout << "  Iterative: Skipping move {" << next_move.pos << "," << next_move.to_pos << "} due to bridge rule." << std::endl;
            continue; // Skip this move if it forms an illegal bridge
        }

        // --- Apply / Check / Recurse (Push) / Undo Cycle ---

        // 1. Apply the move to the single working state
        // *** Pass the correct player argument ***
        current_state->ApplyCheckerMove(player_to_move, next_move);

        // 2. Check validity *after* applying - No longer needed here as bridge checked before.
        // bool is_valid_after_apply = true; // Assuming other checks done in GenerateAllHalfMoves

        // Always proceed if bridge rule passed
        std::vector<CheckerMove> next_sequence = current_frame.current_sequence;
        next_sequence.push_back(next_move);

        // 3. Check termination conditions for this new sequence/state
        //    Is the game over *now*? Are dice used up?
        //    *** AND check if max allowed moves reached ***
        bool has_usable_dice = false;
        for (int die : current_state->dice_) {
            if (current_state->UsableDiceOutcome(die)) { // UsableDiceOutcome checks if die <= 6
                has_usable_dice = true;
                break;
            }
        }
        // Check sequence length against the limit determined at the start
        bool max_moves_reached = next_sequence.size() >= max_allowed_moves;
        bool sequence_complete = !has_usable_dice || current_state->IsTerminal() || max_moves_reached;

        if (sequence_complete) {
            // Path ends here, add the completed sequence
            movelist->push_back(next_sequence);
            int non_pass = 0;
            for(const auto& m : next_sequence) if(m.pos != kPassPos) non_pass++;
            max_non_pass_found = std::max(max_non_pass_found, non_pass);
            if (kDebugging) std::cout << "  Iterative: End of path (complete seq). Added seq. Non-pass: " << non_pass << std::endl;

        } else {
            // 4. Sequence is not complete, explore further if limits allow
            if (movelist->size() < kMaxTotalSequences && branches_pushed < kMaxBranchingFactor) {
                 exploration_stack.push({next_sequence, current_frame.depth + 1});
                 branches_pushed++;
            } else {
                // Hit limits, stop adding branches from this node for this iteration
                #ifndef NDEBUG
                // if (movelist->size() >= kMaxTotalSequences) std::cerr << "Warning: IterativeLegalMoves hit sequence limit during branching." << std::endl;
                // if (branches_pushed >= kMaxBranchingFactor) std::cerr << "Warning: IterativeLegalMoves hit branching factor limit." << std::endl;
                #endif
            }
        }

        // 5. *** CRITICAL: Undo the move to backtrack ***
        //    This restores current_state for the next iteration (sibling move).
        // *** Pass the correct player argument ***
        current_state->UndoCheckerMove(player_to_move, next_move);

    } // End loop over possible_next_moves
  } // End while stack not empty

  return max_non_pass_found;
}

// Filters the generated move sequences based on the number of dice used.
// If any sequence uses both dice, only those sequences are kept.
// Otherwise, sequences using the higher die (if possible) are preferred.
// Changed movelist from const std::set& to const std::vector&
void LongNardeState::FilterBestMoveSequences(
    const std::vector<std::vector<CheckerMove>>& movelist,
    std::vector<std::vector<CheckerMove>>* filtered_movelist,
    int max_non_pass_moves) const {
  if (movelist.empty()) {
      return;
  }

  // Find the maximum sequence length achieved
  int longest_sequence = 0;
  for (const auto& moveseq : movelist) {
      longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
  }

  // Find the maximum number of non-pass moves within sequences of the longest length
  int max_non_pass = 0;
  for (const auto& moveseq : movelist) {
      if (moveseq.size() == longest_sequence) {
          int current_non_pass = 0;
          for (const auto& move : moveseq) {
              if (move.pos != kPassPos) {
                  current_non_pass++;
              }
          }
          max_non_pass = std::max(max_non_pass, current_non_pass);
      }
  }

  // Filter the sequences: keep only those with the longest length AND max non-pass moves
  *filtered_movelist = {};
  bool pass_possible = false; // Track if pass is a potentially valid "best" move

  for (const auto& moveseq : movelist) {
      int current_non_pass = 0;
      bool is_pass_sequence = true;
      for (const auto& move : moveseq) {
          if (move.pos != kPassPos) {
              current_non_pass++;
              is_pass_sequence = false;
          }
      }

      if (moveseq.size() == longest_sequence && current_non_pass == max_non_pass) {
          (*filtered_movelist).push_back(moveseq);
          if (is_pass_sequence) {
             pass_possible = true; // A pass sequence is among the best
          }
      } else if (is_pass_sequence && max_non_pass == 0 && longest_sequence <= 1) {
         // Special case: If the best move is "pass" (max_non_pass = 0) and
         // the longest sequences are size 0 or 1, ensure the explicit pass
         // sequence {kPassMove} is included if it exists in the original list.
         // This handles the scenario where the *only* possible action is Pass.
         if (moveseq.size() == 1 && moveseq[0].pos == kPassPos) {
           (*filtered_movelist).push_back(moveseq);
           pass_possible = true;
         }
      }
  }

  // If the filtered list is empty AND the original list only contained sequences
  // ending because no moves were possible from the start, we need to check
  // if a single pass move is valid.
  if ((*filtered_movelist).empty() && max_non_pass == 0 && longest_sequence == 0) {
      if (kDebugging) std::cout << "FilterBest: Filtered list empty, checking for pass validity." << std::endl;

      // Avoid cloning: Save relevant state, call GenerateAllHalfMoves, restore state.
      // Store potentially modified state variables
      auto original_dice = this->dice_;
      bool original_moved_from_head = this->moved_from_head_;
      Player current_player = this->cur_player_; // Use member variable

      // Temporarily modify 'this' state for the check
      // Need a non-const version of 'this' to modify members and call non-const GenerateAllHalfMoves
      LongNardeState* mutable_this = const_cast<LongNardeState*>(this);

      // Reset potentially affected state for the check
      mutable_this->moved_from_head_ = false; // Reset head move status for the check

      // Generate moves directly on the (temporarily modified) current state
      std::set<CheckerMove> all_half_moves = mutable_this->GenerateAllHalfMoves(current_player);

      // Restore the original state immediately after the call
      mutable_this->dice_ = original_dice;
      mutable_this->moved_from_head_ = original_moved_from_head;

      if (all_half_moves.size() == 1 && all_half_moves.begin()->pos == kPassPos) {
          if (kDebugging) std::cout << "FilterBest: Only pass move is valid. Adding pass sequence." << std::endl;
          (*filtered_movelist).push_back({kPassMove});
          pass_possible = true; // Pass is the only option
      } else if (kDebugging) {
          std::cout << "FilterBest: Pass check - found " << all_half_moves.size() << " half moves. Pass not added." << std::endl;
          for(const auto& mv : all_half_moves) {
             std::cout << "  - Move:{" << mv.pos << "," << mv.to_pos << "," << mv.die << "}" << std::endl;
          }
      }
  }
}

// Helper function to apply the "play higher die" rule if necessary.
std::vector<Action> LongNardeState::ApplyHigherDieRuleIfNeeded(
    const std::vector<Action>& current_legal_moves,
    const std::vector<std::vector<CheckerMove>>& original_movelist /* Unused for now, keep for potential future refactors */) const {

  // This logic requires knowing max_non_pass, which is calculated *before* this would be called.
  // Re-calculate it here based on the original_movelist (passed as arg, though maybe inefficiently).
  // TODO: Refactor LegalActions further to avoid recalculating max_non_pass or pass it directly.
  int longest_sequence = 0;
  if (!original_movelist.empty()) {
    for (const auto& moveseq : original_movelist) {
        longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
    }
  }
  int max_non_pass = 0;
  if(longest_sequence > 0) {
    for (const auto& moveseq : original_movelist) {
        if (moveseq.size() == longest_sequence) {
            int current_non_pass = 0;
            for (const auto& move : moveseq) {
                if (move.pos != kPassPos) {
                    current_non_pass++;
                }
            }
            max_non_pass = std::max(max_non_pass, current_non_pass);
        }
    }
  }

  bool is_doubles = (dice_.size() == 2 && DiceValue(0) == DiceValue(1));

  // Only apply the rule if exactly one die was playable (max_non_pass == 1)
  // AND it's not doubles AND we have exactly 2 dice.
  if (max_non_pass == 1 && !is_doubles && dice_.size() == 2) {
      int d1 = DiceValue(0);
      int d2 = DiceValue(1);
      int higher_die = std::max(d1, d2);
      int lower_die = std::min(d1, d2);

      // Check if each die was individually playable *anywhere* on the board
      // Use a clone to avoid modifying the actual state during this check.
      std::unique_ptr<State> temp_state = this->Clone();
      LongNardeState* cloned_state = dynamic_cast<LongNardeState*>(temp_state.get());
      // Ensure the cloned state has the original dice values (unmarked)
      std::vector<int> original_dice_vals;
      if (dice_.size() >= 1) original_dice_vals.push_back(DiceValue(0));
      if (dice_.size() >= 2) original_dice_vals.push_back(DiceValue(1));
      // Convert dice values back to raw outcomes if needed (assuming they are 1-6 already)
      std::vector<int> raw_original_dice = original_dice_vals; // Assuming dice_ already stores 1-6
      cloned_state->dice_ = raw_original_dice;
      cloned_state->moved_from_head_ = false; // Reset head move status for the check

      std::set<CheckerMove> all_half_moves = cloned_state->GenerateAllHalfMoves(cur_player_);
      bool higher_die_ever_playable = false;
      bool lower_die_ever_playable = false;
      for(const auto& hm : all_half_moves) {
          if(hm.pos != kPassPos) {
              if (hm.die == higher_die) higher_die_ever_playable = true;
              if (hm.die == lower_die) lower_die_ever_playable = true;
          }
          if (higher_die_ever_playable && lower_die_ever_playable) break; // Optimization
      }

      // Identify which of the current legal actions use the higher/lower die
      std::vector<Action> actions_using_higher;
      std::vector<Action> actions_using_lower;

      for (Action action : current_legal_moves) {
          std::vector<CheckerMove> decoded_moves = SpielMoveToCheckerMoves(cur_player_, action);
          CheckerMove single_played_move = kPassMove;
          int actual_non_pass_count = 0;
          for(const auto& m : decoded_moves) {
              if (m.pos != kPassPos) {
                  // Ensure we only find one non-pass move as expected by max_non_pass==1
                  if (actual_non_pass_count == 0) {
                     single_played_move = m;
                  }
                  actual_non_pass_count++;
               }
           }

          if (actual_non_pass_count == 1) {
              if (single_played_move.die == higher_die) {
                  actions_using_higher.push_back(action);
              } else if (single_played_move.die == lower_die) {
                  actions_using_lower.push_back(action);
              }
          } else if (actual_non_pass_count > 1) {
               // This indicates an inconsistency between max_non_pass and the filtered moves.
               SpielFatalError(absl::StrCat("ApplyHigherDieRule: Action ", action, " decoded to ", actual_non_pass_count, " moves, expected 1 based on max_non_pass."));
          }
      }

      // Apply the rule based on which dice were ever playable:
      if (higher_die_ever_playable && lower_die_ever_playable) {
          // Both were playable, must use higher die
          if (kDebugging) std::cout << "ApplyHigherDieRule: Both dice playable, forcing higher die (" << higher_die << ")" << std::endl;
          return actions_using_higher;
      } else if (higher_die_ever_playable) {
          // Only higher was playable
           if (kDebugging) std::cout << "ApplyHigherDieRule: Only higher die (" << higher_die << ") playable." << std::endl;
          return actions_using_higher;
      } else if (lower_die_ever_playable) {
          // Only lower was playable
           if (kDebugging) std::cout << "ApplyHigherDieRule: Only lower die (" << lower_die << ") playable." << std::endl;
          return actions_using_lower;
      } else {
          // This state should not be reachable if max_non_pass == 1
          SpielFatalError("Inconsistent state in ApplyHigherDieRule: Neither die playable but max_non_pass=1.");
          return current_legal_moves; // Should be unreachable, return original as fallback
      }
  }

  // If the rule didn't apply, return the original set of legal moves
  return current_legal_moves;
}

} // namespace long_narde
} // namespace open_spiel
