#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> // For std::max, std::min, std::sort, std::unique
#include <iostream>  // For kDebugging cout/cerr
#include <memory>    // For unique_ptr
#include <utility>   // For std::pair
#include <stack>     // For iterative approach
#include <sstream>   // Added for logging

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" // For error messages

namespace open_spiel
{
    namespace long_narde
    {

        // ===== Legal/Illegal Action Generation =====

        std::vector<Action> LongNardeState::LegalActions() const
        {
            if (IsTerminal()) return {};
            if (IsChanceNode()) return LegalChanceOutcomes();


            std::vector<Action> legal_actions;
            std::set<std::vector<LongNardeCheckerMove>> unique_phase_moves; // To store unique {m1, m2} for current phase

            if (is_handling_second_phase_of_doubles_) {
                // Second phase of doubles: find sequences in cache that match the first phase moves.
                for (const auto& full_sequence : current_turn_full_legal_sequences_cache_) {
                    if (full_sequence.size() >= 2 && 
                        full_sequence[0] == first_phase_selected_move1_ && 
                        full_sequence[1] == first_phase_selected_move2_) {
                        
                        std::vector<LongNardeCheckerMove> second_phase_pair;
                        if (full_sequence.size() >= 3) {
                            second_phase_pair.push_back(full_sequence[2]);
                        }
                        if (full_sequence.size() >= 4) {
                            second_phase_pair.push_back(full_sequence[3]);
                        } else if (full_sequence.size() == 3) {
                            // If only 3 moves in total, pad 4th with pass for encoding
                            // The die for pass needs to be an available die from dice_[2] or dice_[3]
                            int pass_die = DiceValue(3) > 0 ? DiceValue(3) : (DiceValue(2) > 0 ? DiceValue(2) : 1);
                            if(second_phase_pair.size()==1 && second_phase_pair[0].die == pass_die && DiceValue(2) > 0 && DiceValue(2) != pass_die) pass_die = DiceValue(2); 
                            else if (second_phase_pair.size()==1 && second_phase_pair[0].die == pass_die && DiceValue(3) > 0 && DiceValue(3) != pass_die) pass_die = DiceValue(3); 
                            second_phase_pair.push_back({kPassPos, kPassPos, pass_die});
                        }
                        // If full_sequence.size() == 2 (meaning only 2 moves in a double roll)
                        // then this is an empty second_phase_pair, which will become two passes.
                        
                        unique_phase_moves.insert(second_phase_pair); // Will be padded to 2 by encoder if <2
                    }
                }
                // If no matching sequences found (e.g. first phase was pass pass on a 2-move double roll)
                // or if the first phase resulted in no more moves possible for the second phase,
                // we must still offer a pass-pass for the second phase.
                if (unique_phase_moves.empty()) {
                    unique_phase_moves.insert({}); // Encoder will make this {Pass, Pass}
                }

            } else {
                // First phase of doubles, or a non-double turn.
                for (const auto& full_sequence : current_turn_full_legal_sequences_cache_) {
                    std::vector<LongNardeCheckerMove> phase_pair;
                    if (!full_sequence.empty()) {
                        phase_pair.push_back(full_sequence[0]);
                    }
                    if (full_sequence.size() >= 2) {
                        phase_pair.push_back(full_sequence[1]);
                    }
                    // If sequence has 0 or 1 move, it will be padded to 2 by encoder.
                    // If it's a double roll (original sequence > 2), we only take first 2 for this phase.
                    unique_phase_moves.insert(phase_pair);
                }
                 // If cache is empty (e.g. no moves possible after roll), must offer Pass, Pass
                if (current_turn_full_legal_sequences_cache_.empty()) {
                    unique_phase_moves.insert({});
                }
            }

            for (const auto& phase_m_pair : unique_phase_moves) {
                // LongNardeCheckerMovesToSpielMove expects a const ref to vector,
                // and it will pad it to 2 moves if necessary. It now takes state.
                legal_actions.push_back(LongNardeCheckerMovesToSpielMove(phase_m_pair));
            }
            
            // The Higher Die Rule should be implicitly handled by:
            // 1. LongNardeGenerateMoveSequences only generating valid sequences adhering to it.
            // 2. LongNardeFilterBestMoveSequences preferring sequences that use more dice.
            // 3. LongNardeCheckerMovesToSpielMove using first_move_used_actual_high_die_for_phase for encoding.
            // Thus, the explicit call to LongNardeApplyHigherDieRuleIfNeeded is removed.

            // Ensure pass is always an option if no other moves found
            // The logic above for unique_phase_moves.insert({}); should cover this.
            if (legal_actions.empty()) {
                // This case should ideally be covered by unique_phase_moves.insert({}) logic
                // when current_turn_full_legal_sequences_cache_ is empty or second phase has no continuations.
                // Adding explicit double pass just in case.
                legal_actions.push_back(LongNardeCheckerMovesToSpielMove({}));
            }
            
            // Remove duplicates that might arise if different phase_m_pair vectors encode to the same Action ID
            // (e.g. if padding pass dice are chosen differently but pos leads to same ID).
            // This is less likely with the new pos-based encoding but good for safety.
            std::sort(legal_actions.begin(), legal_actions.end());
            legal_actions.erase(std::unique(legal_actions.begin(), legal_actions.end()), legal_actions.end());

            return legal_actions;
        }

        std::vector<Action> LongNardeState::IllegalActions() const
        {
            std::vector<Action> illegal_actions;
            if (IsChanceNode() || IsTerminal())
                return illegal_actions;

            if (dice_.size() < 2)
                return illegal_actions;

            int high_roll = DiceValue(0);
            int low_roll = DiceValue(1);
            if (high_roll < low_roll)
                std::swap(high_roll, low_roll);
            int kMaxActionId = NumDistinctActions();

            std::vector<Action> legal_actions = LegalActions();
            std::set<Action> legal_set(legal_actions.begin(), legal_actions.end());

            for (Action action = 0; action < kMaxActionId; ++action)
            {
                if (legal_set.count(action))
                    continue;
                try
                {
                    std::vector<LongNardeCheckerMove> moves = LongNardeSpielMoveToCheckerMoves(cur_player_, action);
                    bool is_pass = true;
                    for (const auto &m : moves)
                    {
                        if (m.pos != kPassPos)
                        {
                            is_pass = false;
                            break;
                        }
                    }
                    if (is_pass)
                    {
                        if (moves.size() >= 1 && moves[0].die != DiceValue(0) && moves[0].die != DiceValue(1))
                        {
                            illegal_actions.push_back(action);
                            continue;
                        }
                        if (moves.size() >= 2 && moves[1].die != DiceValue(0) && moves[1].die != DiceValue(1))
                        {
                            illegal_actions.push_back(action);
                            continue;
                        }
                    }
                }
                catch (...)
                {
                    illegal_actions.push_back(action);
                    continue;
                }
                if (!legal_set.count(action))
                {
                    illegal_actions.push_back(action);
                }
            }
            return illegal_actions;
        }

        std::vector<std::vector<LongNardeCheckerMove>> LongNardeState::LongNardeGenerateMoveSequences(
            Player player) const
        {
            std::vector<std::vector<LongNardeCheckerMove>> movelist;
            LongNardeIterativeLegalMoves({}, &movelist);
            std::sort(movelist.begin(), movelist.end());
            movelist.erase(std::unique(movelist.begin(), movelist.end()), movelist.end());
            if (movelist.empty())
            {
                const auto &initial = this->LongNardeInitialDice();
                std::vector<LongNardeCheckerMove> pass_seq;
                if (initial.size() >= 2 && initial[0] == initial[1])
                {
                    pass_seq.clear();
                    pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                    pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                }
                else if (initial.size() >= 2)
                {
                    pass_seq.clear();
                    pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                    pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[1]));
                }
                else if (!initial.empty())
                {
                    pass_seq.clear();
                    pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                    pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                }
                movelist.push_back(pass_seq);
            }
            return movelist;
        }

        std::set<LongNardeCheckerMove> LongNardeState::LongNardeGenerateAllHalfMoves(int player, bool moved_from_head_this_sequence) const
        {
            std::set<LongNardeCheckerMove> half_moves;
            for (int pos = 0; pos < kNumPoints; ++pos)
            {
                if (board(player, pos) <= 0)
                    continue;
                for (int die_idx = 0; die_idx < dice_.size(); ++die_idx)
                {
                    if (IsDieUsable(die_idx))
                    {
                        int die_value = DiceValue(die_idx);
                        int to_pos = GetToPos(player, pos, die_value);
                        LongNardeCheckerMove current_move(pos, to_pos, die_value);
                        if (current_move.to_pos < 0)
                        {
                            current_move.to_pos = kBearOffPos;
                        }
                        bool is_valid = LongNardeIsValidCheckerMove(player, current_move, moved_from_head_this_sequence);
                        if (is_valid)
                        {
                            half_moves.insert(current_move);
                        }
                    }
                }
            }
            if (half_moves.empty())
            {
                half_moves.insert(LongNardeCheckerMove(kPassPos, kPassPos, 1));
            }
            return half_moves;
        }

        // ----- Iterative Implementation -----

        // Struct to hold state for the iterative exploration
        struct ExplorationState
        {
            std::unique_ptr<LongNardeState> state; // Represents the state *after* the move is applied
            std::vector<LongNardeCheckerMove> current_sequence;
            LongNardeCheckerMove move_applied;         // The move that led to this state (or a dummy {kPassPos, kPassPos, 0} for root)
            int depth;                        // To track recursion depth limit
            bool moved_from_head_in_sequence; // Tracks if head move occurred *in this path*

            // Constructor for initial state
            ExplorationState(std::unique_ptr<LongNardeState> s, int d, bool initial_home_status)
                : state(std::move(s)), move_applied({kPassPos, kPassPos, 0}), depth(d), moved_from_head_in_sequence(false) {} // Initialize flags

            // Constructor for subsequent states
            ExplorationState(std::unique_ptr<LongNardeState> s,
                             const std::vector<LongNardeCheckerMove> &seq,
                             const LongNardeCheckerMove &move, // Pass the move that was applied
                             int d,
                             bool moved_head) // Added parameter
                : state(std::move(s)), current_sequence(seq), move_applied(move), depth(d), moved_from_head_in_sequence(moved_head)
            {
            } // Use passed flags
        };

        // Iterative helper for LegalActions. Explores possible move sequences using DFS.
        int LongNardeState::LongNardeIterativeLegalMoves(
            const std::vector<LongNardeCheckerMove> &current_sequence, // Changed from initial_moveseq
            std::vector<std::vector<LongNardeCheckerMove>> *moves_list // Changed from movelist
            /* int max_moves_param - Removed */) const
        {
            // Safety limits (reuse from previous recursive version or define new)
            const size_t kMaxTotalSequences = 500;
            const size_t kMaxBranchingFactor = 30;
            const int kMaxIterationDepth = 4; // Equivalent to kMaxRecursionDepth

            std::stack<ExplorationState> exploration_stack;

            // Determine AllInHome status at the very beginning of the turn
            bool initial_all_in_home = this->AllInHome(this->CurrentPlayer());

            // Push the initial state onto the stack (cloned once)
            exploration_stack.emplace(
                std::unique_ptr<LongNardeState>(static_cast<LongNardeState *>(this->Clone().release())),
                0,                  // Initial depth is 0
                initial_all_in_home // Pass initial home status
            );

            int max_non_pass_found = 0; // Track the overall maximum non-pass moves

            while (!exploration_stack.empty())
            {
                // Move ownership of the popped state data
                ExplorationState current_exploration = std::move(exploration_stack.top());
                exploration_stack.pop();

                // Extract data (state pointer is now owned by current_exploration)
                std::unique_ptr<LongNardeState> current_state_ptr = std::move(current_exploration.state);
                LongNardeState *current_state = current_state_ptr.get(); // Get raw pointer for use
                const std::vector<LongNardeCheckerMove> &current_sequence = current_exploration.current_sequence;
                const LongNardeCheckerMove &move_applied_to_reach_this = current_exploration.move_applied;
                int current_depth = current_exploration.depth;

                // --- Check Limits and Base Cases ---
                // Check sequence limit *before* adding potentially large number of sequences
                bool sequence_limit_hit = (moves_list->size() >= kMaxTotalSequences);

                if (sequence_limit_hit || current_depth > kMaxIterationDepth)
                {
#ifndef NDEBUG
                    if (sequence_limit_hit) {
                        std::cerr << "Warning: IterativeLegalMoves hit sequence limit (" << kMaxTotalSequences << ")" << std::endl;
                        // Added debugging for the state that caused the limit to be hit
                        std::cerr << "  Problematic State Details for IterativeLegalMoves limit hit:" << std::endl;
                        std::cerr << "    Player: " << this->CurrentPlayer() << std::endl;
                        std::cerr << "    Board: " << std::endl << this->BoardToString() << std::endl;
                        std::cerr << "    Dice (current turn, dice_): " << this->DiceToString() << std::endl;
                        std::string initial_dice_str = "    InitialDice (initial_dice_): {";
                        const auto& initial_d = this->LongNardeInitialDice();
                        for (size_t i = 0; i < initial_d.size(); ++i) {
                            initial_dice_str += std::to_string(initial_d[i]);
                            if (i < initial_d.size() - 1) initial_dice_str += ",";
                        }
                        initial_dice_str += "}";
                        std::cerr << initial_dice_str << std::endl;
                    }
                    if (current_depth > kMaxIterationDepth)
                        std::cerr << "Warning: IterativeLegalMoves hit depth limit (" << kMaxIterationDepth << ")" << std::endl;
#endif
                    // Add sequence if non-empty, as it's a valid endpoint due to limits
                    if (!current_sequence.empty())
                    {
                        moves_list->push_back(current_sequence); // Changed from insert
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    // No need to undo here, state ptr goes out of scope
                    continue; // Stop exploring this path
                }

                // --> ADDED: Check for terminal state BEFORE generating moves <--
                if (current_state->IsTerminal())
                {
                    if (!current_sequence.empty())
                    {
                        moves_list->push_back(current_sequence); // Changed from insert
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    else
                    {
                        // if (kDebugging) std::cout << "  Iterative: End of path (Terminal state from start). Not adding." << std::endl;
                    }
                    // No need to undo here
                    continue; // Stop exploring this path
                }
                // --> END ADDED CHECK <--

                // Pass the current sequence's head move status to influence head rule checking
                bool current_moved_from_head = current_exploration.moved_from_head_in_sequence;

                // Generate possible next half-moves from the current state
                // Pass the current_moved_from_head status.
                std::set<LongNardeCheckerMove> half_moves = current_state->LongNardeGenerateAllHalfMoves(current_state->CurrentPlayer(), current_moved_from_head);

                // --- Base Case Check: End of a sequence path? (Excluding terminal check, done above) ---
                if (half_moves.empty() || current_sequence.size() >= game_->MaxGameLength())
                {
                    if (!current_sequence.empty())
                    {
                        moves_list->push_back(current_sequence); // Changed from insert
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    else if (half_moves.size() == 1 && half_moves.begin()->pos == kPassPos)
                    {
                        // This means GenerateAllHalfMoves found no valid checker moves and added a placeholder pass.
                        // Construct the actual pass sequence using the initial dice for this turn.
                        std::vector<LongNardeCheckerMove> actual_pass_sequence;
                        const std::vector<int> &initial_dice = this->LongNardeInitialDice();

                        SPIEL_CHECK_GE(initial_dice.size(), 2); // Ensure we have dice info

                        // Handle Doubles Pass (needs 4 pass moves with the same die)
                        if (initial_dice.size() >= 2 && initial_dice[0] == initial_dice[1])
                        {
                            int die_value = initial_dice[0];
                            // For doubles pass, encode just two pass moves with the die value.
                            // The encoding/decoding logic handles the 4-move implication.
                            actual_pass_sequence.push_back({kPassPos, kPassPos, die_value});
                            actual_pass_sequence.push_back({kPassPos, kPassPos, die_value});
                        }
                        else if (initial_dice.size() >= 2)
                        {
                            // Handle Non-Doubles Pass (needs 2 pass moves, one for each die)
                            actual_pass_sequence.push_back({kPassPos, kPassPos, initial_dice[0]});
                            actual_pass_sequence.push_back({kPassPos, kPassPos, initial_dice[1]});
                        }
                        // Else: Should not happen if dice_ were properly set earlier

                        // Only add if a valid pass sequence was constructed
                        if (!actual_pass_sequence.empty())
                        {
                            moves_list->push_back(actual_pass_sequence); // Add the correctly formed sequence
                        }
                    }
                    else
                    {
                        // No moves possible from start, or other terminal condition with empty sequence
                        // if (kDebugging) std::cout << "  Iterative: End of path (no moves from start or other). Not adding." << std::endl;
                    }
                    // No need to undo here
                    continue; // Finished exploring this path
                }

                // --- Explore Next Moves ---
                size_t explored_branches = 0;
                bool found_move_in_iteration = false;
                Player player = current_state->CurrentPlayer(); // Get player once

                // Convert to vector to easily check the index for the last move optimization
                std::vector<LongNardeCheckerMove> moves_to_explore;
                for (const auto &move : half_moves)
                {
                    if (move.pos != kPassPos)
                    { // Filter out placeholder pass
                        moves_to_explore.push_back(move);
                    }
                }

                bool ownership_transferred = false; // Track if the unique_ptr was moved
                for (int i = 0; i < moves_to_explore.size(); ++i)
                {
                    const LongNardeCheckerMove &next_move = moves_to_explore[i];
                    // No need to check for kPassPos here, already filtered

                    if (explored_branches >= kMaxBranchingFactor)
                    {
#ifndef NDEBUG
                        std::cerr << "Warning: IterativeLegalMoves hit branching factor limit (" << kMaxBranchingFactor << ")" << std::endl;
#endif
                        break; // Stop exploring further branches from this node
                    }

                    // --- Apply Move ---
                    current_state->LongNardeApplyCheckerMove(player, next_move);
                    found_move_in_iteration = true;

                    // Create the new sequence
                    std::vector<LongNardeCheckerMove> next_sequence = current_sequence;
                    next_sequence.push_back(next_move);

                    // Calculate the head move status for the next state
                    bool next_moved_from_head = current_exploration.moved_from_head_in_sequence || current_state->IsHeadPos(player, next_move.pos);

                    // **** Check if applying this move resulted in a terminal state ****
                    if (current_state->IsTerminal())
                    {
                        // If terminal, add this completed sequence and don't push state to stack.
                        moves_list->push_back(next_sequence);
                        int non_pass = 0;
                        for (const auto &m : next_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        // Need to undo the move if we are not transferring ownership (i.e., not the last move)
                        if (i != moves_to_explore.size() - 1)
                        {
                            current_state->LongNardeUndoCheckerMove(player, next_move);
                        }
                        // If it *was* the last move, ownership will be transferred implicitly when current_state_ptr goes out of scope if not moved.
                    }
                    else
                    {
                        // **** If not terminal, proceed with pushing to stack ****
                        bool is_last_move = (i == moves_to_explore.size() - 1);

                        if (!is_last_move)
                        {
                            // --- Push Cloned State (Not Last Move) ---
                            std::unique_ptr<LongNardeState> next_state_for_stack(
                                static_cast<LongNardeState *>(current_state->Clone().release()));
                            exploration_stack.emplace(std::move(next_state_for_stack), next_sequence, next_move, current_depth + 1, next_moved_from_head);
                            // --- Undo Move ---
                            current_state->LongNardeUndoCheckerMove(player, next_move);
                        }
                        else
                        {
                            // --- Push Original State (Last Move) ---
                            exploration_stack.emplace(std::move(current_state_ptr), next_sequence, next_move, current_depth + 1, next_moved_from_head);
                            ownership_transferred = true;
                        }
                    }

                    explored_branches++;
                } // End for loop over moves_to_explore

                // If no actual moves were pushed (e.g., only pass was generated initially, or branching limit hit immediately)
                if (!found_move_in_iteration && !current_sequence.empty())
                {
                    // If the FIX block is commented out, we might need to ensure current_sequence is still added if it's a valid stopping point.
                    // This was the implicit behavior if the `else` part of the `if(forced_pass)` was hit.
                    // Adding the original behavior here if the `else` part of the `if(forced_pass)` is hit:
                    moves_list->push_back(current_sequence); // Add sequence as is
                    int non_pass = 0;
                    for (const auto &m : current_sequence)
                        if (m.pos != kPassPos)
                            non_pass++;
                    max_non_pass_found = std::max(max_non_pass_found, non_pass);
                }

                // If ownership wasn't transferred in the loop, the unique_ptr (current_state_ptr)
                // goes out of scope here, deleting the state object automatically.

            } // End while loop

            // The return value isn't strictly used by GenerateMoveSequences anymore,
            // but we maintain it for potential future use or consistency.
            return max_non_pass_found;
        }

        // ----- End Iterative Implementation -----

        // Helper function to filter generated sequences for the best ones.
        std::pair<std::vector<std::vector<LongNardeCheckerMove>>, int> LongNardeState::LongNardeFilterBestMoveSequences(
            const std::vector<std::vector<LongNardeCheckerMove>> &movelist) const
        {
            if (movelist.empty())
            {
                return {{}, 0};
            }

            // Find the maximum sequence length achieved
            int longest_sequence = 0;
            for (const auto &moveseq : movelist)
            {
                longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
            }

            // Find the maximum number of non-pass moves within sequences of the longest length
            int max_non_pass = 0;
            for (const auto &moveseq : movelist)
            {
                if (moveseq.size() == longest_sequence)
                {
                    int current_non_pass = 0;
                    for (const auto &move : moveseq)
                    {
                        if (move.pos != kPassPos)
                        {
                            current_non_pass++;
                        }
                    }
                    max_non_pass = std::max(max_non_pass, current_non_pass);
                }
            }

            // Filter the sequences: keep only those with the longest length AND max non-pass moves
            std::vector<std::vector<LongNardeCheckerMove>> filtered_movelist;
            bool pass_possible = false; // Track if pass is a potentially valid "best" move

            // Check if any sequence leads to a terminal state
            bool has_terminal_sequence = false;
            std::vector<std::vector<LongNardeCheckerMove>> terminal_sequences;

            // First pass: identify terminal sequences
            for (const auto &moveseq : movelist)
            {
                // Clone the current state to test if this sequence leads to a terminal state
                std::unique_ptr<LongNardeState> test_state = std::unique_ptr<LongNardeState>(static_cast<LongNardeState *>(this->Clone().release()));

                // Apply all moves in the sequence
                for (const auto &move : moveseq)
                {
                    if (move.pos != kPassPos)
                    {
                        test_state->LongNardeApplyCheckerMove(test_state->CurrentPlayer(), move);
                    }
                }

                // Check if this sequence leads to a terminal state
                if (test_state->IsTerminal())
                {
                    has_terminal_sequence = true;
                    terminal_sequences.push_back(moveseq);
                }
            }

            // If we have terminal sequences, include all of them regardless of length
            if (has_terminal_sequence)
            {
                filtered_movelist = terminal_sequences;
            }
            else
            {
                // Standard filtering for non-terminal sequences
                for (const auto &moveseq : movelist)
                {
                    int current_non_pass = 0;
                    bool is_pass_sequence = true;
                    for (const auto &move : moveseq)
                    {
                        if (move.pos != kPassPos)
                        {
                            current_non_pass++;
                            is_pass_sequence = false;
                        }
                    }

                    if (moveseq.size() == longest_sequence && current_non_pass == max_non_pass)
                    {
                        filtered_movelist.push_back(moveseq);
                        if (is_pass_sequence)
                        {
                            pass_possible = true; // A pass sequence is among the best
                        }
                    }
                    else if (is_pass_sequence && max_non_pass == 0 && longest_sequence <= 1)
                    {
                        // Special case: If the best move is "pass" (max_non_pass = 0) and
                        // the longest sequences are size 0 or 1, ensure the explicit pass
                        // sequence {kPassMove} is included if it exists in the original list.
                        // This handles the scenario where the *only* possible action is Pass.
                        if (moveseq.size() == 1 && moveseq[0].pos == kPassPos)
                        {
                            filtered_movelist.push_back(moveseq);
                            pass_possible = true;
                        }
                    }
                }
            }

            // Apply "play higher die if only one actual move is made" rule for non-doubles
            bool is_doubles_roll_for_filter = (DiceValue(0) > 0 && DiceValue(0) == DiceValue(1) && DiceValue(0) == DiceValue(2) && DiceValue(0) == DiceValue(3));

            if (max_non_pass == 1 && (longest_sequence == 1 || longest_sequence == 2) && !is_doubles_roll_for_filter && filtered_movelist.size() > 1) {
                int d1_val = DiceValue(0); // Higher die of the roll (assuming dice_ is sorted or 0/1 are the relevant ones)
                int d2_val = DiceValue(1); // Lower die of the roll

                // Ensure dice are distinct and valid before proceeding
                if (d1_val > 0 && d2_val > 0 && d1_val != d2_val) {
                    int higher_rolled_die = std::max(d1_val, d2_val);
                    int lower_rolled_die = std::min(d1_val, d2_val);

                    std::vector<std::vector<LongNardeCheckerMove>> sequences_using_higher_die_for_the_move;
                    std::vector<std::vector<LongNardeCheckerMove>> sequences_using_lower_die_for_the_move;
                    std::vector<std::vector<LongNardeCheckerMove>> other_sequences; // For safety, if logic misses a case

                    for (const auto& seq : filtered_movelist) {
                        if (seq.empty()) { other_sequences.push_back(seq); continue; }

                        LongNardeCheckerMove actual_move = {kPassPos, kPassPos, 0};
                        int non_pass_in_seq = 0;
                        for(const auto& m : seq) {
                            if (m.pos != kPassPos) {
                                actual_move = m;
                                non_pass_in_seq++;
                            }
                        }
                        
                        if (non_pass_in_seq == 1) { // Ensure it's truly a single actual move sequence
                            if (actual_move.die == higher_rolled_die) {
                                sequences_using_higher_die_for_the_move.push_back(seq);
                            }
                            else if (actual_move.die == lower_rolled_die) {
                                sequences_using_lower_die_for_the_move.push_back(seq);
                            }
                            else {
                                // This case should not happen if dice are d1_val/d2_val and one is used
                                other_sequences.push_back(seq); 
                            }
                        }
                        else {
                            other_sequences.push_back(seq); // Not a single actual move sequence, keep it as is for now
                        }
                    }

                    // If any sequence uses the higher die, prioritize them.
                    // Otherwise, if only lower die sequences exist, they are the only option.
                    if (!sequences_using_higher_die_for_the_move.empty()) {
                        filtered_movelist = sequences_using_higher_die_for_the_move;
                        // Append any 'other_sequences' that were not single-move sequences, if they should still be considered.
                        // For this specific rule (max_non_pass == 1), other_sequences should ideally be empty or only contain passes.
                        // However, to be safe and maintain any all-pass sequences if they got here:
                        for(const auto& os_seq : other_sequences) {
                            bool is_all_pass = true;
                            for(const auto& m : os_seq) if(m.pos != kPassPos) is_all_pass = false;
                            if(is_all_pass) filtered_movelist.push_back(os_seq); // only add back if it was an all-pass sequence
                        }

                    } else if (!sequences_using_lower_die_for_the_move.empty()) {
                        filtered_movelist = sequences_using_lower_die_for_the_move;
                        // Append 'other_sequences' similarly
                         for(const auto& os_seq : other_sequences) {
                            bool is_all_pass = true;
                            for(const auto& m : os_seq) if(m.pos != kPassPos) is_all_pass = false;
                            if(is_all_pass) filtered_movelist.push_back(os_seq); 
                        }
                    } else {
                        // No single move sequences found, or they didn't match dice. Keep original other_sequences.
                        filtered_movelist = other_sequences;
                    }
                     // Re-sort and unique if changes were made and new duplicates could arise
                    std::sort(filtered_movelist.begin(), filtered_movelist.end());
                    filtered_movelist.erase(std::unique(filtered_movelist.begin(), filtered_movelist.end()), filtered_movelist.end());
                }
            }

            // If the filtered list is empty AND the original list only contained sequences
            // ending because no moves were possible from the start, we need to check
            // if a single pass move is valid.
            if (filtered_movelist.empty() && max_non_pass == 0 && longest_sequence == 0)
            {
                // Avoid cloning: Save relevant state, call GenerateAllHalfMoves, restore state.
                // Store potentially modified state variables
                std::vector<int> original_dice = this->dice_;
                bool original_moved_from_head = this->moved_from_head_;
                Player current_player = this->cur_player_; // Use member variable

                // Temporarily modify 'this' state for the check
                // Need a non-const version of 'this' to modify members and call non-const GenerateAllHalfMoves
                LongNardeState *mutable_this = const_cast<LongNardeState *>(this);

                // Reset potentially affected state for the check
                mutable_this->moved_from_head_ = false; // Reset head move status for the check

                // Use GenerateAllHalfMoves to check validity from the current state.
                // Determine initial home status for this specific check
                bool check_initial_home_status = mutable_this->AllInHome(current_player);
                std::set<LongNardeCheckerMove> all_half_moves = mutable_this->LongNardeGenerateAllHalfMoves(current_player, mutable_this->moved_from_head_);

                // Restore the original state immediately after the call
                mutable_this->dice_ = original_dice;
                mutable_this->moved_from_head_ = original_moved_from_head;

                if (all_half_moves.size() == 1 && all_half_moves.begin()->pos == kPassPos)
                {
                    filtered_movelist.push_back({kPassMove});
                    pass_possible = true; // Pass is the only option
                }
            }

            // Canonicalize sequences with one actual move and one pass
            if (max_non_pass == 1 && longest_sequence == 2) {
                for (auto& seq : filtered_movelist) {
                    if (seq.size() == 2) {
                        bool move0_is_pass = (seq[0].pos == kPassPos);
                        bool move1_is_pass = (seq[1].pos == kPassPos);
                        // If one is a pass and the other is not, ensure the non-pass is first.
                        if (move0_is_pass && !move1_is_pass) {
                            std::swap(seq[0], seq[1]);
                        }
                    }
                }
                // After canonicalizing, re-sort and unique to remove duplicates arising from different orderings.
                std::sort(filtered_movelist.begin(), filtered_movelist.end());
                filtered_movelist.erase(std::unique(filtered_movelist.begin(), filtered_movelist.end()), filtered_movelist.end());
            }

            return {filtered_movelist, max_non_pass};
        }

        // Helper function to apply the "play higher die" rule if necessary.
        std::vector<Action> LongNardeState::LongNardeApplyHigherDieRuleIfNeeded(
            const std::vector<Action> &current_legal_moves,
            const std::vector<std::vector<LongNardeCheckerMove>> &original_movelist /* Keep arg for now */) const
        {
            // Recalculate max_non_pass (consider refactoring LegalActions to pass this)
            int longest_sequence = 0;
            if (!original_movelist.empty())
            {
                for (const auto &moveseq : original_movelist)
                {
                    longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
                }
            }
            int max_non_pass = 0;
            if (longest_sequence > 0)
            {
                for (const auto &moveseq : original_movelist)
                {
                    if (moveseq.size() == longest_sequence)
                    {
                        int current_non_pass = 0;
                        for (const auto &move : moveseq)
                        {
                            if (move.pos != kPassPos)
                            {
                                current_non_pass++;
                            }
                        }
                        max_non_pass = std::max(max_non_pass, current_non_pass);
                    }
                }
            }

            // Correct doubles check for 4-element dice_
            bool is_doubles = (dice_.size() == 4 && DiceValue(0) > 0 && DiceValue(0) == DiceValue(1));

            // Apply the rule if exactly one die was playable (max_non_pass == 1) and it wasn't doubles.
            if (max_non_pass == 1 && !is_doubles)
            {
                SPIEL_CHECK_GE(dice_.size(), 2); // Should have at least 2 dice defined
                int d1 = DiceValue(0);
                int d2 = DiceValue(1);
                // Ensure d1/d2 are valid die values (1-6) if max_non_pass is 1
                SPIEL_CHECK_GE(d1, 1);
                SPIEL_CHECK_LE(d1, 6);
                SPIEL_CHECK_GE(d2, 1);
                SPIEL_CHECK_LE(d2, 6);

                int higher_die = std::max(d1, d2);
                int lower_die = std::min(d1, d2);

                std::vector<Action> actions_using_higher;
                std::vector<Action> actions_using_lower;
                bool found_higher = false;
                bool found_lower = false;

                for (Action action : current_legal_moves)
                {
                    std::vector<LongNardeCheckerMove> decoded_moves = LongNardeSpielMoveToCheckerMoves(cur_player_, action);
                    // Find the single non-pass move
                    for (const auto &m : decoded_moves)
                    {
                        if (m.pos != kPassPos)
                        {
                            if (m.die == higher_die)
                            {
                                actions_using_higher.push_back(action);
                                found_higher = true;
                            }
                            else if (m.die == lower_die)
                            {
                                actions_using_lower.push_back(action);
                                found_lower = true;
                            }
                            break; // Stop checking moves for this action
                        }
                    }
                }

                // Determine which list to return based on what was found.
                // If only higher die moves were found OR if both were found (rule mandates higher),
                // return the higher die actions.
                if (found_higher)
                {
                    return actions_using_higher;
                }
                // If only lower die moves were found, those are the only playable ones.
                else if (found_lower)
                {
                    return actions_using_lower;
                }
                // If neither was found (and current_legal_moves wasn't empty), something is wrong.
                else if (!current_legal_moves.empty())
                {
                    SpielFatalError(absl::StrCat("ApplyHigherDieRule: No moves found matching dice when max_non_pass=1. Higher: ", higher_die, ", Lower: ", lower_die));
                    return {}; // Should be unreachable
                }
                else
                {
                    // Input list was empty, return empty.
                    return {};
                }
            }

            // If the rule didn't apply, return the original set of legal moves
            return current_legal_moves;
        }

    } // namespace long_narde
} // namespace open_spiel
