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
  // Use the new iterative function
  IterativeLegalMoves({}, &movelist, max_moves); 

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

// ----- Iterative Implementation -----

// Struct to hold state for the iterative exploration
struct ExplorationState {
  std::unique_ptr<LongNardeState> state;
  std::vector<CheckerMove> current_sequence;
  int depth; // To track recursion depth limit

  // Constructor for initial state
  ExplorationState(std::unique_ptr<LongNardeState> s, int d) 
    : state(std::move(s)), depth(d) {}

  // Constructor for subsequent states
  ExplorationState(std::unique_ptr<LongNardeState> s, 
                   const std::vector<CheckerMove>& seq, 
                   int d) 
    : state(std::move(s)), current_sequence(seq), depth(d) {}
};

// Iterative helper for LegalActions. Explores possible move sequences using DFS.
int LongNardeState::IterativeLegalMoves(const std::vector<CheckerMove>& initial_moveseq,
                                        std::set<std::vector<CheckerMove>>* movelist,
                                        int /*max_moves_param - unused, depth limit controls*/) const {
  // Safety limits (same as recursive version)
  const size_t kMaxTotalSequences = 200; 
  const size_t kMaxBranchingFactor = 30; 
  const int kMaxIterationDepth = 6; // Equivalent to kMaxRecursionDepth

  std::stack<ExplorationState> exploration_stack;

  // Push the initial state onto the stack
  exploration_stack.emplace(
      std::unique_ptr<LongNardeState>(static_cast<LongNardeState*>(this->Clone().release())), 
      0 // Initial depth is 0
  );
  
  int max_non_pass_found = 0; // Track the overall maximum non-pass moves

  while (!exploration_stack.empty()) {
    ExplorationState current = std::move(exploration_stack.top());
    exploration_stack.pop();

    LongNardeState* current_state = current.state.get();
    const std::vector<CheckerMove>& current_sequence = current.current_sequence;
    int current_depth = current.depth;

    // --> ADD DEBUG CHECK HERE <--
    #ifndef NDEBUG // Only include in debug builds
    // Use temporary variables to avoid calling CurrentPlayer() multiple times if it has side effects (it shouldn't)
    bool is_term = current_state->IsTerminal();
    Player actual_player = Player{current_state->cur_player_}; // Get raw player ID
    Player reported_player = current_state->CurrentPlayer(); // Get player via method

    if (reported_player == kTerminalPlayerId || actual_player < 0) {
        std::cerr << "!!! IterativeLegalMoves: Popped suspect state at depth " << current_depth << ".\n"
                  << "    Reported Player (CurrentPlayer()): " << reported_player << "\n"
                  << "    Actual Player (cur_player_):       " << actual_player << "\n"
                  << "    IsTerminal():                    " << (is_term ? "TRUE" : "FALSE") << "\n"
                  << "    Current Sequence Size:           " << current_sequence.size() << "\n"
                  << "State:\n" << current_state->ToString() << std::endl;
        // Optionally add SPIEL_CHECK here if player is invalid to halt earlier
        // SPIEL_CHECK_GE(actual_player, 0);
    }
    #endif

    // --- Check Limits and Base Cases ---
    if (movelist->size() >= kMaxTotalSequences || current_depth > kMaxIterationDepth) {
      #ifndef NDEBUG
      if (movelist->size() >= kMaxTotalSequences) std::cerr << "Warning: IterativeLegalMoves hit sequence limit (" << kMaxTotalSequences << ")" << std::endl;
      if (current_depth > kMaxIterationDepth) std::cerr << "Warning: IterativeLegalMoves hit depth limit (" << kMaxIterationDepth << ")" << std::endl;
      #endif
      // Add sequence if non-empty, as it's a valid endpoint due to limits
      if (!current_sequence.empty()) {
         movelist->insert(current_sequence);
         int non_pass = 0;
         for(const auto& m : current_sequence) if(m.pos != kPassPos) non_pass++;
         max_non_pass_found = std::max(max_non_pass_found, non_pass);
      }
      continue; // Stop exploring this path
    }

    // --> ADDED: Check for terminal state BEFORE generating moves <--
    if (current_state->IsTerminal()) {
      if (!current_sequence.empty()) {
         movelist->insert(current_sequence);
         int non_pass = 0;
         for(const auto& m : current_sequence) if(m.pos != kPassPos) non_pass++;
         max_non_pass_found = std::max(max_non_pass_found, non_pass);
         if (kDebugging) std::cout << "  Iterative: End of path (Terminal state). Added seq. Non-pass: " << non_pass << std::endl;
      } else {
         if (kDebugging) std::cout << "  Iterative: End of path (Terminal state from start). Not adding." << std::endl;
      }
      continue; // Stop exploring this path
    }
    // --> END ADDED CHECK <--

    // Generate all valid *single* moves from the *current* state
    std::set<CheckerMove> half_moves = current_state->GenerateAllHalfMoves(current_state->CurrentPlayer());

    bool only_pass_available = half_moves.size() == 1 && half_moves.begin()->pos == kPassPos;
    // Replicate HasUsableDice logic:
    bool has_usable_dice = false;
    for (int die_outcome : current_state->dice_) {
        if (current_state->UsableDiceOutcome(die_outcome)) {
            has_usable_dice = true;
            break;
        }
    }
    bool no_dice_left = current_state->dice_.empty() || !has_usable_dice;

    // --- Base Case Check: End of a sequence path? (Excluding terminal check, done above) ---
    if (only_pass_available || no_dice_left || half_moves.empty() || current_sequence.size() >= game_->MaxGameLength()) {
      if (!current_sequence.empty()) {
          movelist->insert(current_sequence);
          int non_pass = 0;
          for(const auto& m : current_sequence) if(m.pos != kPassPos) non_pass++;
          max_non_pass_found = std::max(max_non_pass_found, non_pass);
          if (kDebugging) std::cout << "  Iterative: End of path (pass/no dice/no moves/max len). Added seq. Non-pass: " << non_pass << std::endl;
      } else if (only_pass_available) {
          // If sequence is empty and only pass is available, add the pass sequence
          movelist->insert({*half_moves.begin()}); 
          // max_non_pass_found remains 0
          if (kDebugging) std::cout << "  Iterative: End of path (only pass available from start). Added pass sequence." << std::endl;
      } else {
          // No moves possible from start, or other terminal condition with empty sequence
           if (kDebugging) std::cout << "  Iterative: End of path (no moves from start or other). Not adding." << std::endl;
      }
      continue; // Finished exploring this path
    }

    // --- Explore Next Moves ---
    size_t explored_branches = 0;
    bool found_move_in_iteration = false; 
    for (const CheckerMove& next_move : half_moves) {
      if (next_move.pos == kPassPos) continue; // Skip placeholder pass if other moves exist

      if (explored_branches >= kMaxBranchingFactor) {
        #ifndef NDEBUG
        std::cerr << "Warning: IterativeLegalMoves hit branching factor limit (" << kMaxBranchingFactor << ") at depth " << current_depth << std::endl;
        #endif
        break; // Stop exploring further branches from this node
      }

      // Clone the current state *before* applying the move
      std::unique_ptr<LongNardeState> next_state_ptr(static_cast<LongNardeState*>(current_state->Clone().release()));
      
      // Apply the move to the *cloned* state
      next_state_ptr->ApplyCheckerMove(next_state_ptr->CurrentPlayer(), next_move);
      found_move_in_iteration = true; 

      // Create the new sequence
      std::vector<CheckerMove> next_sequence = current_sequence;
      next_sequence.push_back(next_move);

      // Push the new state and sequence onto the stack
      exploration_stack.emplace(std::move(next_state_ptr), next_sequence, current_depth + 1);
      
      if (kDebugging) std::cout << "  Iterative: Pushed state for move {" << next_move.pos << "," << next_move.to_pos << "," << next_move.die << "} at depth " << current_depth + 1 << std::endl;

      explored_branches++;
    }
     // If no actual moves were pushed (e.g., only pass was generated but skipped, or branching limit hit immediately)
     // and the current sequence is valid, add it.
     if (!found_move_in_iteration && !current_sequence.empty()) {
         movelist->insert(current_sequence);
         int non_pass = 0;
         for(const auto& m : current_sequence) if(m.pos != kPassPos) non_pass++;
         max_non_pass_found = std::max(max_non_pass_found, non_pass);
         if (kDebugging) std::cout << "  Iterative: End of path (no branches explored). Added current seq. Non-pass: " << non_pass << std::endl;
     }

  } // End while loop

  if (kDebugging) std::cout << "IterativeLegalMoves finished. Total sequences: " << movelist->size() << ", Max non-pass found: " << max_non_pass_found << std::endl;
  
  // The return value isn't strictly used by GenerateMoveSequences anymore, 
  // but we maintain it for potential future use or consistency.
  return max_non_pass_found; 
}


// ----- End Iterative Implementation -----


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
