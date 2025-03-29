#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> // For std::max, std::min, std::sort, std::unique
#include <iostream>  // For kDebugging cout/cerr
#include <memory>    // For unique_ptr
#include <utility>   // For std::pair

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" // For error messages

namespace open_spiel {
namespace long_narde {

// ===== Legal/Illegal Action Generation =====

std::vector<Action> LongNardeState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();

  // Determine max moves based on dice
  int max_moves = 0;
  if (!dice_.empty()) {
      bool is_doubles = (dice_.size() >= 2 && DiceValue(0) == DiceValue(1));
      max_moves = is_doubles ? 4 : 2; // Allow up to 4 for doubles, 2 otherwise
  }

  // Generate all possible move sequences
  std::set<std::vector<CheckerMove>> movelist = GenerateMoveSequences(CurrentPlayer(), max_moves);

  // DEBUG: Print movelist contents after generation
  if (kDebugging) {
    std::cout << "DEBUG LegalActions (Player " << CurrentPlayer() << "): Movelist after GenerateMoveSequences (" << movelist.size() << " entries):" << std::endl;
    /* Optional detail printout - keep commented unless needed 
    for (const auto& seq : movelist) {
        std::cout << "  Seq: ";
        for(const auto& m : seq) { std::cout << "{" << m.pos << "," << m.to_pos << "," << m.die << "} "; }
        std::cout << std::endl;
    }
    */
  }

  // Filter for the best move sequences (longest, max non-pass)
  auto filter_result = FilterBestMoveSequences(movelist);
  std::set<std::vector<CheckerMove>> filtered_movelist = filter_result.first;
  int max_non_pass = filter_result.second;

  // If filtering resulted in only a pass sequence (and original was pass-only), convert and return it.
  if (max_non_pass == 0 && !filtered_movelist.empty()) {
    // FilterBestMoveSequences guarantees the set contains one canonical pass sequence here.
     return {CheckerMovesToSpielMove(*filtered_movelist.begin())};
  }
  
  // Convert filtered move sequences to Spiel Actions
  const size_t kMaxActionsToGenerate = 20;
  std::set<Action> unique_actions;

  for (const auto& moveseq : filtered_movelist) {
    // The filtering logic previously here is now in FilterBestMoveSequences
    if (unique_actions.size() >= kMaxActionsToGenerate) break;
    Action action = CheckerMovesToSpielMove(moveseq);
    unique_actions.insert(action);
  }

  std::vector<Action> legal_moves;
  legal_moves.assign(unique_actions.begin(), unique_actions.end());

  // Apply the "play higher die" rule if necessary.
  // Pass the original generated movelist for context needed by the rule.
  legal_moves = ApplyHigherDieRuleIfNeeded(legal_moves, movelist); 

  return legal_moves;
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
std::set<std::vector<CheckerMove>> LongNardeState::GenerateMoveSequences(
    Player player, int max_moves) const {
  std::set<std::vector<CheckerMove>> movelist;
  std::unique_ptr<State> cstate = this->Clone();
  LongNardeState* state = dynamic_cast<LongNardeState*>(cstate.get());

  // Generate all possible move sequences using the cloned state
  state->RecLegalMoves({}, &movelist, max_moves);

  // DEBUG: Print movelist contents after generation
  if (kDebugging) {
    std::cout << "DEBUG GenerateMoveSequences (Player " << player << "): Movelist size = " << movelist.size() << std::endl;
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
        if (is_debugging) std::cout << "    Die " << DiceValue(outcome) << " (raw " << outcome <<") not usable, skipping\n";
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

// Recursive helper for LegalActions. Explores possible move sequences.
int LongNardeState::RecLegalMoves(const std::vector<CheckerMove>& moveseq,
                                  std::set<std::vector<CheckerMove>>* movelist,
                                  int max_depth) {
  // Safety limits to prevent excessive recursion/computation
  const size_t kMaxTotalSequences = 200; 
  const size_t kMaxBranchingFactor = 30; // Max half-moves to explore per step
  const int kMaxRecursionDepth = 6; // Should be > 4 for doubles

  if (movelist->size() >= kMaxTotalSequences || max_depth > kMaxRecursionDepth) {
     #ifndef NDEBUG
      if (movelist->size() >= kMaxTotalSequences) std::cerr << "Warning: RecLegalMoves hit sequence limit (" << kMaxTotalSequences << ")" << std::endl;
      if (max_depth > kMaxRecursionDepth) std::cerr << "Warning: RecLegalMoves hit recursion depth limit (" << kMaxRecursionDepth << ")" << std::endl;
     #endif
    // Add current sequence if non-empty, as it's a valid (though possibly incomplete) sequence end-point due to limits
    if (!moveseq.empty() && movelist->find(moveseq) == movelist->end()) {
        movelist->insert(moveseq);
    }
    return moveseq.size(); 
  }

  // Generate next possible *single* half-moves from the current state
  std::set<CheckerMove> half_moves = GenerateAllHalfMoves(cur_player_);

  // Base Case 1: No moves possible from this state (only pass was generated or set was empty).
  if (half_moves.empty() || (half_moves.size() == 1 && (*half_moves.begin()).pos == kPassPos) ) {
    // Add the sequence found so far (if any moves were made).
    // If moveseq is empty, it means no moves were possible from the start (pass turn).
    if (movelist->find(moveseq) == movelist->end()) { // Avoid duplicates
        movelist->insert(moveseq);
    }
    return moveseq.size();
  }
  
  // Base Case 2: Max number of moves for this turn reached (e.g., 2 for non-doubles, 4 for doubles).
  if (moveseq.size() >= max_depth) {
       if (movelist->find(moveseq) == movelist->end()) { // Avoid duplicates
           movelist->insert(moveseq);
       }
      return moveseq.size();
  }


  // --- Recursive Step ---
  size_t moves_checked_this_level = 0;
  int max_len_found_downstream = moveseq.size(); // Track max length found *from this point*

  for (const auto& move : half_moves) {
    // Skip the placeholder pass move if other moves are available
    if (move.pos == kPassPos) continue; 

    // Check limits before processing the move
    if (movelist->size() >= kMaxTotalSequences) return max_len_found_downstream; // Early exit if limit hit during iteration
    if (moves_checked_this_level >= kMaxBranchingFactor) break; // Limit branching factor
    moves_checked_this_level++;

    // --- Simulate applying the move and recursing ---
    std::vector<CheckerMove> next_moveseq = moveseq; // Copy current sequence
    next_moveseq.push_back(move);                   // Add the chosen half-move

    // DEBUG: Print state BEFORE applying move
    if (kDebugging) {
        std::cout << "DEBUG RecLegalMoves: BEFORE Apply {pos=" << move.pos << ", die=" << move.die << "}: Dice: ";
        for (int d_idx = 0; d_idx < dice_.size(); ++d_idx) { std::cout << DiceValue(d_idx) << (UsableDiceOutcome(dice_[d_idx])?"(U)":"(X)") << " "; }
        std::cout << " Board[12]="<< board(cur_player_, 12) << " Board[13]="<< board(cur_player_, 13) << std::endl;
    }

    bool original_moved_from_head = moved_from_head_; // Save state part not handled by Undo
    ApplyCheckerMove(cur_player_, move); // Apply move to *this* state object (will be undone later)

    // DEBUG: Print state AFTER applying move, BEFORE recursion
    if (kDebugging) {
        std::cout << "DEBUG RecLegalMoves: AFTER Apply {pos=" << move.pos << ", die=" << move.die << "}:  Dice: ";
        for (int d_idx = 0; d_idx < dice_.size(); ++d_idx) { std::cout << DiceValue(d_idx) << (UsableDiceOutcome(dice_[d_idx])?"(U)":"(X)") << " "; }
        std::cout << " Board[12]="<< board(cur_player_, 12) << " Board[13]="<< board(cur_player_, 13) << std::endl;
    }

    // *** Check for momentary illegal bridge ***
    // This check ensures that even intermediate positions within a sequence are valid.
    if (HasIllegalBridge(cur_player_)) {
        // This move temporarily created an illegal bridge. This path is invalid.
        // Backtrack the move and continue to the next possible half-move.
        UndoCheckerMove(cur_player_, move);
        moved_from_head_ = original_moved_from_head; // Restore head move status
        // Do NOT add next_moveseq to movelist
        continue; // Skip the recursive call for this invalid path
    }

    // Recursive call for the next move in the sequence
    int child_max_len = RecLegalMoves(next_moveseq, movelist, max_depth);

    // Backtrack state after recursive call returns
    UndoCheckerMove(cur_player_, move);
    moved_from_head_ = original_moved_from_head; // Restore head move status

    // DEBUG: Print state AFTER undoing move
    if (kDebugging) {
        std::cout << "DEBUG RecLegalMoves: AFTER Undo {pos=" << move.pos << ", die=" << move.die << "}:   Dice: ";
        for (int d_idx = 0; d_idx < dice_.size(); ++d_idx) { std::cout << DiceValue(d_idx) << (UsableDiceOutcome(dice_[d_idx])?"(U)":"(X)") << " "; }
        std::cout << " Board[12]="<< board(cur_player_, 12) << " Board[13]="<< board(cur_player_, 13) << std::endl;
    }

    // Update the maximum sequence length found so far among all branches explored from this node
    max_len_found_downstream = std::max(child_max_len, max_len_found_downstream);

     // Check limit again after recursive call returns
     if (movelist->size() >= kMaxTotalSequences) return max_len_found_downstream;
  }

   // If we explored moves (moves_checked_this_level > 0) but didn't find any sequence
   // reaching the full max_depth from this point (e.g. only one die playable),
   // then the sequence leading *to* this state is a valid end-point.
   // Add the 'moveseq' (the sequence *before* exploring this level's half_moves)
   // unless it was already added in a base case.
   if (moves_checked_this_level > 0 && max_len_found_downstream < (moveseq.size() + 1) ) {
       if (!moveseq.empty() && movelist->find(moveseq) == movelist->end()) {
            movelist->insert(moveseq);
       }
   }


  return max_len_found_downstream;
}


// Helper function to filter generated sequences for the best ones.
std::pair<std::set<std::vector<CheckerMove>>, int> LongNardeState::FilterBestMoveSequences(
    const std::set<std::vector<CheckerMove>>& movelist) const {
  if (movelist.empty()) {
      return {{}, 0};
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

  // DEBUG: Print filtering criteria
  if (kDebugging) {
    std::cout << "DEBUG FilterBestMoveSequences: longest_sequence = " << longest_sequence 
              << ", max_non_pass = " << max_non_pass << std::endl;
  }

  // If max_non_pass is 0, it means only pass moves are possible (or movelist was empty initially)
  if (max_non_pass == 0) {
      // Create a canonical pass sequence (length 2, using dice or defaults)
      std::vector<CheckerMove> pass_move_seq;
      int p_die1 = (dice_.size() >= 1 && UsableDiceOutcome(dice_[0])) ? DiceValue(0) : kPassDieValue;
      int p_die2 = (dice_.size() >= 2 && UsableDiceOutcome(dice_[1])) ? DiceValue(1) : p_die1;
      // Ensure dice are valid (1-6), clamp if necessary (should ideally not happen with kPassDieValue)
      p_die1 = std::max(1, std::min(6, p_die1));
      p_die2 = std::max(1, std::min(6, p_die2));

      // Use the actual dice values when creating the pass moves for correct encoding.
      pass_move_seq.push_back({kPassPos, kPassPos, p_die1}); 
      pass_move_seq.push_back({kPassPos, kPassPos, p_die2}); 
      
      // Return a set containing just this pass sequence and max_non_pass = 0
      return {{pass_move_seq}, 0};
  }

  // Filter the movelist to keep only sequences with max length AND max non-pass moves
  std::set<std::vector<CheckerMove>> filtered_movelist;
  for (const auto& moveseq : movelist) {
      if (moveseq.size() == longest_sequence) {
          int current_non_pass = 0;
          for (const auto& move : moveseq) {
              if (move.pos != kPassPos) {
                  current_non_pass++;
              }
          }
          if (current_non_pass == max_non_pass) {
              filtered_movelist.insert(moveseq);
          }
      }
  }

  return {filtered_movelist, max_non_pass};
}

// Helper function to apply the "play higher die" rule if necessary.
std::vector<Action> LongNardeState::ApplyHigherDieRuleIfNeeded(
    const std::vector<Action>& current_legal_moves,
    const std::set<std::vector<CheckerMove>>& original_movelist /* Unused for now, keep for potential future refactors */) const {

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
      std::vector<int> original_dice;
      if (dice_.size() >= 1) original_dice.push_back(DiceValue(0));
      if (dice_.size() >= 2) original_dice.push_back(DiceValue(1));
      cloned_state->dice_ = original_dice;
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
          return actions_using_higher;
      } else if (higher_die_ever_playable) {
          // Only higher was playable
          return actions_using_higher;
      } else if (lower_die_ever_playable) {
          // Only lower was playable
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
