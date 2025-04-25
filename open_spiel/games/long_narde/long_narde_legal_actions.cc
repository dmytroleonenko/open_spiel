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
            std::vector<Action> legal_actions;

            if (IsTerminal())
                return {};
            if (IsChanceNode())
                return LegalChanceOutcomes();

            // Generate all possible move sequences
            std::vector<std::vector<CheckerMove>> movelist = GenerateMoveSequences(CurrentPlayer());

            // Filter for the best move sequences (longest, max non-pass)
            auto filter_result = FilterBestMoveSequences(movelist);
            std::vector<std::vector<CheckerMove>> filtered_movelist = filter_result.first;
            int max_non_pass = filter_result.second;

            // If filtering resulted in only a pass sequence (and original was pass-only), convert and return it.
            if (max_non_pass == 0 && !filtered_movelist.empty())
            {
                // FilterBestMoveSequences guarantees the set contains one canonical pass sequence here.
                // HOWEVER, that sequence might just be the placeholder {kPassPos, kPassPos, 0} or {kPassPos, kPassPos, 1}.
                // We need to construct the *correct* pass sequence using the actual dice for encoding.
                SPIEL_CHECK_GE(dice_.size(), 2); // Should have dice if we reached here needing to pass.
                std::vector<CheckerMove> actual_pass_sequence;
                actual_pass_sequence.push_back({kPassPos, kPassPos, DiceValue(0)}); // Use first die value
                actual_pass_sequence.push_back({kPassPos, kPassPos, DiceValue(1)}); // Use second die value
                return {CheckerMovesToSpielMove(actual_pass_sequence)};             // Encode the correct sequence
            }

            // Convert filtered move sequences to Spiel Actions
            const size_t kMaxActionsToGenerate = 20;
            std::set<Action> unique_actions;

            for (const auto &moveseq : filtered_movelist)
            {
                // The filtering logic previously here is now in FilterBestMoveSequences
                if (unique_actions.size() >= kMaxActionsToGenerate)
                    break;
                Action action = CheckerMovesToSpielMove(moveseq);
                unique_actions.insert(action);
            }

            std::vector<Action> legal_moves;
            legal_moves.assign(unique_actions.begin(), unique_actions.end());

            // **** Apply Higher Die Rule ****
            std::vector<Action> final_actions = ApplyHigherDieRuleIfNeeded(legal_moves, filtered_movelist);
            return final_actions;
        }

        std::vector<Action> LongNardeState::IllegalActions() const
        {
            std::vector<Action> illegal_actions;
            if (IsChanceNode() || IsTerminal())
                return illegal_actions;

            // Check dice validity before proceeding
            if (dice_.size() < 2)
                return illegal_actions; // Cannot determine rolls

            int high_roll = DiceValue(0);
            int low_roll = DiceValue(1);
            if (high_roll < low_roll)
                std::swap(high_roll, low_roll);
            int kMaxActionId = NumDistinctActions();

            std::vector<Action> legal_actions = LegalActions(); // Get legal actions once
            std::set<Action> legal_set(legal_actions.begin(), legal_actions.end());

            for (Action action = 0; action < kMaxActionId; ++action)
            {
                if (legal_set.count(action))
                    continue; // Skip known legal actions

                // Simple heuristic check: Check if it decodes reasonably.
                // Full validation is complex and already done by LegalActions.
                // This mainly catches encoding ranges that don't make sense.
                try
                {
                    std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, action);
                    // Basic check: pass moves should correspond to valid die values if possible
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
                        // Check if die values encoded in pass make sense with current roll
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
                    // Further checks could be added, but might duplicate LegalActions logic.
                    // The main goal is to identify actions outside the *possible* range or clearly invalid encodings.
                }
                catch (...)
                {
                    // If decoding itself fails, it's illegal
                    illegal_actions.push_back(action);
                    continue;
                }
                // If it wasn't caught by simple checks and isn't legal, add it.
                if (!legal_set.count(action))
                {
                    illegal_actions.push_back(action);
                }
            }
            return illegal_actions;
        }

        // Helper function to generate all valid move sequences.
        std::vector<std::vector<CheckerMove>> LongNardeState::GenerateMoveSequences(
            Player player /* Removed: int max_moves */) const
        {
            std::vector<std::vector<CheckerMove>> movelist; // Changed from std::set
            // Use the new iterative function - remove max_moves argument
            IterativeLegalMoves({}, &movelist);

            // Sort and remove duplicates to mimic std::set behavior
            std::sort(movelist.begin(), movelist.end());
            movelist.erase(std::unique(movelist.begin(), movelist.end()), movelist.end());

            // Add pass move sequence if no moves possible
            if (movelist.empty())
            {
                const auto &initial = this->InitialDice();
                std::vector<CheckerMove> pass_seq;
                if (initial.size() >= 2 && initial[0] == initial[1])
                {
                    pass_seq = {CheckerMove(kPassPos, kPassPos, initial[0]),
                                CheckerMove(kPassPos, kPassPos, initial[0])};
                }
                else if (initial.size() >= 2)
                {
                    pass_seq = {CheckerMove(kPassPos, kPassPos, initial[0]),
                                CheckerMove(kPassPos, kPassPos, initial[1])};
                }
                else if (!initial.empty())
                {
                    pass_seq = {CheckerMove(kPassPos, kPassPos, initial[0]),
                                CheckerMove(kPassPos, kPassPos, initial[0])};
                }
                movelist.push_back(pass_seq);
            }

            return movelist;
        }

        std::set<CheckerMove> LongNardeState::GenerateAllHalfMoves(int player, bool moved_from_head_this_sequence) const
        {
            std::set<CheckerMove> half_moves;

            // For each checker belonging to the player
            for (int pos = 0; pos < kNumPoints; ++pos)
            {
                if (board(player, pos) <= 0)
                    continue; // Skip points with no checkers for this player

                // For each usable die slot (always 4 slots)
                for (int die_idx = 0; die_idx < dice_.size(); ++die_idx)
                {
                    if (IsDieUsable(die_idx))
                    {                                       // Check if the die slot itself is usable
                        int die_value = DiceValue(die_idx); // Get the actual die value (1-6)

                        // Calculate destination based on the *checker's position* (pos)
                        int to_pos = GetToPos(player, pos, die_value);

                        // Check if this specific half-move is valid *now*
                        // Use the checker's position (pos) as the starting point
                        CheckerMove current_move(pos, to_pos, die_value); // Create the move struct

                        // If the calculated to_pos indicates any type of bear-off (is < 0),
                        // standardize it to kBearOffPos for consistency before validation/insertion.
                        if (current_move.to_pos < 0)
                        {
                            current_move.to_pos = kBearOffPos;
                        }

                        bool is_valid = IsValidCheckerMove(player, current_move, moved_from_head_this_sequence);

                        if (is_valid)
                        {
                            half_moves.insert(current_move);
                        }
                    }
                }
            }

            // If no valid moves were found after checking all checkers and dice,
            // the player *must* pass. We add a pass move placeholder.
            // The encoding function will handle assigning dice values to the pass.
            if (half_moves.empty())
            {
                // Add a single pass move. LegalActions/Encoding will handle using correct dice.
                // Use die=1 as a placeholder.
                half_moves.insert(CheckerMove(kPassPos, kPassPos, 1));
            }

            return half_moves;
        }

        // ----- Iterative Implementation -----

        // Struct to hold state for the iterative exploration
        struct ExplorationState
        {
            std::unique_ptr<LongNardeState> state; // Represents the state *after* the move is applied
            std::vector<CheckerMove> current_sequence;
            CheckerMove move_applied;         // The move that led to this state (or a dummy {kPassPos, kPassPos, 0} for root)
            int depth;                        // To track recursion depth limit
            bool moved_from_head_in_sequence; // Tracks if head move occurred *in this path*

            // Constructor for initial state
            ExplorationState(std::unique_ptr<LongNardeState> s, int d, bool initial_home_status)
                : state(std::move(s)), move_applied({kPassPos, kPassPos, 0}), depth(d), moved_from_head_in_sequence(false) {} // Initialize flags

            // Constructor for subsequent states
            ExplorationState(std::unique_ptr<LongNardeState> s,
                             const std::vector<CheckerMove> &seq,
                             const CheckerMove &move, // Pass the move that was applied
                             int d,
                             bool moved_head) // Added parameter
                : state(std::move(s)), current_sequence(seq), move_applied(move), depth(d), moved_from_head_in_sequence(moved_head)
            {
            } // Use passed flags
        };

        // Iterative helper for LegalActions. Explores possible move sequences using DFS.
        int LongNardeState::IterativeLegalMoves(
            const std::vector<CheckerMove> &current_sequence, // Changed from initial_moveseq
            std::vector<std::vector<CheckerMove>> *moves_list // Changed from movelist
            /* int max_moves_param - Removed */) const
        {

            // *** ADDED: Unconditional Entry Log ***
            if (kDebugging)
            {
                std::cerr << "[DEBUG ILM ENTRY] IterativeLegalMoves function entered.\n"
                          << std::flush;
            }
            // *** END Unconditional Entry Log ***

            // Safety limits (reuse from previous recursive version or define new)
            const size_t kMaxTotalSequences = 200;
            const size_t kMaxBranchingFactor = 30;
            const int kMaxIterationDepth = 6; // Equivalent to kMaxRecursionDepth

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
                const std::vector<CheckerMove> &current_sequence = current_exploration.current_sequence;
                const CheckerMove &move_applied_to_reach_this = current_exploration.move_applied;
                int current_depth = current_exploration.depth;

                // *** Log State Immediately After Pop ***
                if (kDebugging)
                {
                    std::cerr << "[DEBUG ILM Popped State] Depth: " << current_depth
                              << " | Player: " << current_state->CurrentPlayer()
                              << " | Dice: " << current_state->DiceToString()
                              << " | Initial: [";
                    for (int d : current_state->InitialDice())
                    {
                        std::cerr << d << " ";
                    }
                    std::cerr << "] | Seq Size: " << current_sequence.size() << "\n"
                              << std::flush;
                }
                // *** End Log ***

                // *** START DEBUG LOGGING BLOCK FOR ConsecutiveMovesTest ***
                bool is_target_scenario_intermediate = false;
                // Check for the specific intermediate state in ConsecutiveMovesTest Scenario 1
                if (current_exploration.current_sequence.size() == 1)
                {
                    const CheckerMove &first_move = current_exploration.current_sequence[0];
                    // Check if the first move applied was {pos=8, to=5, die=3}
                    // AND ensure we are Player 0 with initial dice [5, 3]
                    const auto &initial_dice = current_state->InitialDice();
                    bool dice_match = initial_dice.size() >= 2 && initial_dice[0] == 5 && initial_dice[1] == 3;
                    if (current_state->CurrentPlayer() == kXPlayerId && dice_match &&
                        first_move.pos == 8 && first_move.to_pos == 5 && first_move.die == 3)
                    {
                        is_target_scenario_intermediate = true;
                        if (kDebugging)
                        {
                            std::cerr << "\n[DEBUG ILM] Target Intermediate State Detected (ConsecutiveMovesTest):" << std::flush;
                            std::cerr << "\n  Sequence so far: {pos=" << first_move.pos << ", to=" << first_move.to_pos << ", die=" << first_move.die << "}\n"
                                      << std::flush;
                            std::cerr << "  Current State Player: " << current_state->CurrentPlayer() << "\n"
                                      << std::flush;
                            std::cerr << "  Current State Dice (Usable): " << current_state->DiceToString() << "\n"
                                      << std::flush; // Keep this line
                            // Display initial dice as stored in the *current* state object
                            std::cerr << "  Initial Dice (as stored): [";
                            for (size_t i = 0; i < initial_dice.size(); ++i)
                            { // This loop should now work
                                std::cerr << initial_dice[i] << (i == initial_dice.size() - 1 ? "" : ", ");
                            }
                            std::cerr << "]\n"
                                      << std::flush;
                            std::cerr << "  Moved from head in sequence: " << (current_exploration.moved_from_head_in_sequence ? "true" : "false") << "\n"
                                      << std::flush;
                        }
                    }
                }
// *** END DEBUG LOGGING BLOCK ***

// --> ADD DEBUG CHECK HERE <--
#ifndef NDEBUG // Only include in debug builds
               // Use temporary variables to avoid calling CurrentPlayer() multiple times if it has side effects (it shouldn't)
                bool is_term = current_state->IsTerminal();
                Player actual_player = Player{current_state->cur_player_}; // Get raw player ID
                Player reported_player = current_state->CurrentPlayer();   // Get player via method

                if (reported_player == kTerminalPlayerId || actual_player < 0)
                {
                    std::cerr << "!!! IterativeLegalMoves: Popped suspect state at depth " << current_depth << ".\n"
                              << "    Reported Player (CurrentPlayer()): " << reported_player << "\n"
                              << "    Actual Player (cur_player_):       " << actual_player << "\n"
                              << "    IsTerminal():                    " << (is_term ? "TRUE" : "FALSE") << "\n"
                              << "    Current Sequence Size:           " << current_sequence.size() << "\n"
                              << "State:\n"
                              << current_state->ToString() << std::endl;
                    // Optionally add SPIEL_CHECK here if player is invalid to halt earlier
                    // SPIEL_CHECK_GE(actual_player, 0);
                }
#endif

                // --- Check Limits and Base Cases ---
                // Check sequence limit *before* adding potentially large number of sequences
                bool sequence_limit_hit = (moves_list->size() >= kMaxTotalSequences);

                if (sequence_limit_hit || current_depth > kMaxIterationDepth)
                {
#ifndef NDEBUG
                    if (sequence_limit_hit)
                        std::cerr << "Warning: IterativeLegalMoves hit sequence limit (" << kMaxTotalSequences << ")" << std::endl;
                    if (current_depth > kMaxIterationDepth)
                        std::cerr << "Warning: IterativeLegalMoves hit depth limit (" << kMaxIterationDepth << ")" << std::endl;
#endif
                    // Add sequence if non-empty, as it's a valid endpoint due to limits
                    if (!current_sequence.empty())
                    {
                        // ADDED: Log sequence before adding due to limits
                        if (kDebugging)
                        {
                            std::stringstream ss_limit;
                            ss_limit << "[DEBUG ILM ADD SEQ LIMIT] Depth=" << current_depth << ":";
                            for (const auto &m : current_sequence)
                            {
                                ss_limit << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                            }
                            std::cout << ss_limit.str() << std::endl;
                        }
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
                        // ADDED: Log sequence before adding due to terminal state
                        if (kDebugging)
                        {
                            std::stringstream ss_term;
                            ss_term << "[DEBUG ILM ADD SEQ TERM] Depth=" << current_depth << ":";
                            for (const auto &m : current_sequence)
                            {
                                ss_term << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                            }
                            std::cout << ss_term.str() << std::endl;
                        }
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
                std::set<CheckerMove> half_moves = current_state->GenerateAllHalfMoves(current_state->CurrentPlayer(), current_moved_from_head);

                // ADDED: Log the content of half_moves, specifically checking for the illegal move
                if (kDebugging && current_state->CurrentPlayer() == 1)
                { // Log only for player 1 for relevance
                    std::stringstream ss;
                    ss << "[DEBUG ILM POST-GAHM] Half-moves generated for P1 state (depth=" << current_depth << "):";
                    bool found_illegal = false;
                    for (const auto &hm : half_moves)
                    {
                        ss << " {" << hm.pos << "," << hm.to_pos << "," << hm.die << "}";
                        if (hm.pos == 11 && hm.to_pos == 12 && hm.die == 1)
                        {
                            found_illegal = true;
                        }
                    }
                    if (found_illegal)
                    {
                        ss << " <<< ILLEGAL MOVE (11->12/1) FOUND IN SET! >>>";
                    }
                    std::cout << ss.str() << std::endl;
                }

                // *** START DEBUG LOGGING BLOCK FOR ConsecutiveMovesTest ***
                if (is_target_scenario_intermediate)
                {
                    if (kDebugging)
                    {
                        std::cerr << "[DEBUG ILM] Result of GenerateAllHalfMoves (Target Intermediate State):"
                                  << "\n"
                                  << std::flush;
                        if (half_moves.empty())
                        {
                            std::cerr << "  <No half moves generated>\n"
                                      << std::flush;
                        }
                        else
                        {
                            std::cerr << "  Generated " << half_moves.size() << " half-moves:\n"
                                      << std::flush;
                            for (const auto &hm : half_moves)
                            {
                                std::cerr << "    {pos=" << hm.pos << ", to=" << hm.to_pos << ", die=" << hm.die << "}\n"
                                          << std::flush;
                            }
                        }
                        std::cerr << "[DEBUG ILM] ---- End Target Intermediate State Log ----\n"
                                  << std::flush;
                    }
                }
                // *** END DEBUG LOGGING BLOCK ***

                if (kDebugging)
                {
                    std::cout << "  Generated " << half_moves.size() << " half-moves: ";
                    for (const auto &hm : half_moves)
                    {
                        std::cout << "[" << hm.pos << "->" << hm.to_pos << "/" << hm.die << "] ";
                    }
                    std::cout << std::endl;
                }

                // --- Base Case Check: End of a sequence path? (Excluding terminal check, done above) ---
                if (half_moves.empty() || current_sequence.size() >= game_->MaxGameLength())
                {
                    if (!current_sequence.empty())
                    {
                        // ADDED: Log sequence before adding due to base case (empty half_moves/max len)
                        if (kDebugging)
                        {
                            std::stringstream ss_base;
                            ss_base << "[DEBUG ILM ADD SEQ BASE] Depth=" << current_depth << ":";
                            for (const auto &m : current_sequence)
                            {
                                ss_base << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                            }
                            std::cout << ss_base.str() << std::endl;
                        }
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
                        std::vector<CheckerMove> actual_pass_sequence;
                        // Access initial_dice_ from the *original* state (this) as current_state might have changed dice
                        const std::vector<int> &initial_dice = this->InitialDice();

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

                        // ADDED: Log the *actual* pass sequence being added
                        if (kDebugging)
                        {
                            std::stringstream ss_pass_actual;
                            ss_pass_actual << "[DEBUG ILM ADD SEQ PASS ACTUAL] Depth=" << current_depth << ":";
                            for (const auto &m : actual_pass_sequence)
                            {
                                ss_pass_actual << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                            }
                            std::cout << ss_pass_actual.str() << std::endl;
                        }
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
                std::vector<CheckerMove> moves_to_explore;
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
                    const CheckerMove &next_move = moves_to_explore[i];
                    // No need to check for kPassPos here, already filtered

                    if (explored_branches >= kMaxBranchingFactor)
                    {
#ifndef NDEBUG
                        std::cerr << "Warning: IterativeLegalMoves hit branching factor limit (" << kMaxBranchingFactor << ")" << std::endl;
#endif
                        break; // Stop exploring further branches from this node
                    }

                    // --- Apply Move ---
                    current_state->ApplyCheckerMove(player, next_move);
                    found_move_in_iteration = true;

                    // Create the new sequence
                    std::vector<CheckerMove> next_sequence = current_sequence;
                    next_sequence.push_back(next_move);

                    // Calculate the head move status for the next state
                    bool next_moved_from_head = current_exploration.moved_from_head_in_sequence || current_state->IsHeadPos(player, next_move.pos);

                    // **** Check if applying this move resulted in a terminal state ****
                    if (current_state->IsTerminal())
                    {
                        // If terminal, add this completed sequence and don't push state to stack.
                        // ADDED: Log sequence before adding due to terminal state *after* move
                        if (kDebugging)
                        {
                            std::stringstream ss_term_post;
                            ss_term_post << "[DEBUG ILM ADD SEQ TERM POST] Depth=" << current_depth + 1 << ":";
                            for (const auto &m : next_sequence)
                            {
                                ss_term_post << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                            }
                            std::cout << ss_term_post.str() << std::endl;
                        }
                        moves_list->push_back(next_sequence);
                        int non_pass = 0;
                        for (const auto &m : next_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        // Need to undo the move if we are not transferring ownership (i.e., not the last move)
                        if (i != moves_to_explore.size() - 1)
                        {
                            current_state->UndoCheckerMove(player, next_move);
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
                            // *** Log Before Push (Clone) ***
                            if (kDebugging)
                            {
                                std::cerr << "[DEBUG ILM Before Push (Clone)] Player: " << current_state->CurrentPlayer()
                                          << " | Dice: " << current_state->DiceToString()
                                          << " | Initial: [";
                                for (int d : current_state->InitialDice())
                                {
                                    std::cerr << d << " ";
                                }
                                std::cerr << "]";
                            }
                            // *** End Log ***
                            std::unique_ptr<LongNardeState> next_state_for_stack(
                                static_cast<LongNardeState *>(current_state->Clone().release()));
                            exploration_stack.emplace(std::move(next_state_for_stack), next_sequence, next_move, current_depth + 1, next_moved_from_head);
                            // *** Log After Push (Clone) ***
                            if (kDebugging)
                            {
                                if (!exploration_stack.empty())
                                {
                                    LongNardeState *top_state = exploration_stack.top().state.get();
                                    std::cerr << "\n[DEBUG ILM After Push (Clone)] Top State - Player: " << top_state->CurrentPlayer()
                                              << " | Dice: " << top_state->DiceToString()
                                              << " | Initial: [";
                                    for (int d : top_state->InitialDice())
                                    {
                                        std::cerr << d << " ";
                                    }
                                    std::cerr << "]";
                                }
                            }
                            // *** End Log ***
                            // if (kDebugging) std::cout << "  Iterative (Clone): Pushed state for move {" << next_move.pos << "," << next_move.to_pos << "," << next_move.die << "} at depth " << current_depth + 1 << std::endl;

                            // --- Undo Move ---
                            current_state->UndoCheckerMove(player, next_move);
                        }
                        else
                        {
                            // --- Push Original State (Last Move) ---
                            // *** Log Before Push (Move) ***
                            if (kDebugging)
                            {
                                std::cerr << "[DEBUG ILM Before Push (Move)] Player: " << current_state->CurrentPlayer()
                                          << " | Dice: " << current_state->DiceToString()
                                          << " | Initial: [";
                                for (int d : current_state->InitialDice())
                                {
                                    std::cerr << d << " ";
                                }
                                std::cerr << "]";
                            }
                            // *** End Log ***
                            exploration_stack.emplace(std::move(current_state_ptr), next_sequence, next_move, current_depth + 1, next_moved_from_head);
                            ownership_transferred = true;
                            // *** Log After Push (Move) ***
                            if (kDebugging)
                            {
                                if (!exploration_stack.empty())
                                {
                                    LongNardeState *top_state = exploration_stack.top().state.get();
                                    std::cerr << "\n[DEBUG ILM After Push (Move)] Top State - Player: " << top_state->CurrentPlayer()
                                              << " | Dice: " << top_state->DiceToString()
                                              << " | Initial: [";
                                    for (int d : top_state->InitialDice())
                                    {
                                        std::cerr << d << " ";
                                    }
                                    std::cerr << "]";
                                }
                            }
                            // *** End Log ***
                            // if (kDebugging) std::cout << "  Iterative (Move): Pushed state for move {" << next_move.pos << "," << next_move.to_pos << "," << next_move.die << "} at depth " << current_depth + 1 << std::endl;\n";
                        }
                    }

                    explored_branches++;
                } // End for loop over moves_to_explore

                // If no actual moves were pushed (e.g., only pass was generated initially, or branching limit hit immediately)
                if (!found_move_in_iteration && !current_sequence.empty())
                {
                    // *** START FIX ***
                    // Check if we stopped because GenerateAllHalfMoves returned only a pass,
                    // AND if there were still usable dice in the state we popped.
                    bool had_usable_dice = false;
                    for (int i = 0; i < current_state->dice_.size(); ++i)
                    {
                        if (current_state->IsDieUsable(i))
                        {
                            had_usable_dice = true;
                            break;
                        }
                    }

                    // Check if the reason we stopped was a forced pass.
                    bool forced_pass = had_usable_dice && half_moves.size() == 1 && half_moves.begin()->pos == kPassPos;

                    if (forced_pass)
                    {
                        // Find a usable die value to associate with the pass
                        int pass_die_value = -1;
                        for (int i = 0; i < current_state->dice_.size(); ++i)
                        {
                            if (current_state->IsDieUsable(i))
                            {
                                pass_die_value = current_state->DiceValue(i);
                                break; // Use the first usable die found
                            }
                        }
                        if (pass_die_value != -1)
                        {
                            // Append the pass move to the sequence before adding it
                            std::vector<CheckerMove> sequence_with_pass = current_sequence;
                            sequence_with_pass.push_back(CheckerMove(kPassPos, kPassPos, pass_die_value));
                            // Log and add the sequence *with* the pass
                            if (kDebugging)
                            {
                                std::stringstream ss_forcedpass;
                                ss_forcedpass << "[DEBUG ILM ADD SEQ FORCED_PASS] Depth=" << current_depth << ":";
                                for (const auto &m : sequence_with_pass)
                                {
                                    ss_forcedpass << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                                }
                                std::cout << ss_forcedpass.str() << std::endl;
                            }
                            moves_list->push_back(sequence_with_pass); // Add sequence with pass
                            int non_pass = 0;
                            for (const auto &m : sequence_with_pass)
                                if (m.pos != kPassPos)
                                    non_pass++;
                            max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        }
                        else
                        {
                            // Should not happen if had_usable_dice was true, but log error just in case.
                            std::cerr << "ERROR: IterativeLegalMoves - had usable dice but couldn't find value for pass!" << std::endl;
                            // Fall back to adding the sequence without pass (original behavior)
                            if (kDebugging)
                            {
                                std::stringstream ss_errorcase;
                                ss_errorcase << "[DEBUG ILM ADD SEQ ERROR_FALLBACK] Depth=" << current_depth << ":";
                                for (const auto &m : current_sequence)
                                {
                                    ss_errorcase << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                                }
                                std::cout << ss_errorcase.str() << std::endl;
                            }
                            moves_list->push_back(current_sequence); // Add sequence without pass
                            int non_pass = 0;
                            for (const auto &m : current_sequence)
                                if (m.pos != kPassPos)
                                    non_pass++;
                            max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        }
                    }
                    else
                    {
                        // Original behavior: add the sequence as is (e.g., limits hit, or truly no usable dice left)
                        if (kDebugging)
                        {
                            std::stringstream ss_nobranch;
                            ss_nobranch << "[DEBUG ILM ADD SEQ NOBRANCH/LIMIT] Depth=" << current_depth << ":";
                            for (const auto &m : current_sequence)
                            {
                                ss_nobranch << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                            }
                            std::cout << ss_nobranch.str() << std::endl;
                        }
                        moves_list->push_back(current_sequence); // Add sequence as is
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    // *** END FIX ***
                }

                // If ownership wasn't transferred in the loop, the unique_ptr (current_state_ptr)
                // goes out of scope here, deleting the state object automatically.

            } // End while loop

            // if (kDebugging) std::cout << "IterativeLegalMoves finished. Total sequences added: " << moves_list->size() << ", Max non-pass found: " << max_non_pass_found << std::endl;

            // The return value isn't strictly used by GenerateMoveSequences anymore,
            // but we maintain it for potential future use or consistency.
            return max_non_pass_found;
        }

        // ----- End Iterative Implementation -----

        // Helper function to filter generated sequences for the best ones.
        std::pair<std::vector<std::vector<CheckerMove>>, int> LongNardeState::FilterBestMoveSequences(
            const std::vector<std::vector<CheckerMove>> &movelist) const
        {

            // *** ADDED DEBUG LOG ***
            if (kDebugging)
            {
                std::cout << "[DEBUG FBS Entry] Player=" << CurrentPlayer() << " Input movelist (" << movelist.size() << "):\n";
                for (const auto &seq : movelist)
                {
                    std::cout << "  Seq: ";
                    for (const auto &m : seq)
                    {
                        std::cout << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                    }
                    std::cout << std::endl;
                }
            }
            // *** END ADDED DEBUG LOG ***

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

            // *** ADDED DEBUG LOG ***
            if (kDebugging)
            {
                std::cout << "[DEBUG FBS Stats] longest_sequence=" << longest_sequence << ", max_non_pass=" << max_non_pass << std::endl;
            }
            // *** END ADDED DEBUG LOG ***

            // Filter the sequences: keep only those with the longest length AND max non-pass moves
            std::vector<std::vector<CheckerMove>> filtered_movelist;
            bool pass_possible = false; // Track if pass is a potentially valid "best" move

            // Check if any sequence leads to a terminal state
            bool has_terminal_sequence = false;
            std::vector<std::vector<CheckerMove>> terminal_sequences;

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
                        test_state->ApplyCheckerMove(test_state->CurrentPlayer(), move);
                    }
                }

                // Check if this sequence leads to a terminal state
                if (test_state->IsTerminal())
                {
                    has_terminal_sequence = true;
                    terminal_sequences.push_back(moveseq);

                    if (kDebugging)
                    {
                        std::cout << "[DEBUG FBS Terminal] Found terminal sequence:";
                        for (const auto &m : moveseq)
                        {
                            std::cout << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                        }
                        std::cout << std::endl;
                    }
                }
            }

            // If we have terminal sequences, include all of them regardless of length
            if (has_terminal_sequence)
            {
                if (kDebugging)
                {
                    std::cout << "[DEBUG FBS Terminal] Including " << terminal_sequences.size() << " terminal sequences" << std::endl;
                }
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

            // If the filtered list is empty AND the original list only contained sequences
            // ending because no moves were possible from the start, we need to check
            // if a single pass move is valid.
            if (filtered_movelist.empty() && max_non_pass == 0 && longest_sequence == 0)
            {
                // if (kDebugging) std::cout << "FilterBest: Filtered list empty, checking for pass validity." << std::endl;

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
                std::set<CheckerMove> all_half_moves = mutable_this->GenerateAllHalfMoves(current_player, mutable_this->moved_from_head_);

                // Restore the original state immediately after the call
                mutable_this->dice_ = original_dice;
                mutable_this->moved_from_head_ = original_moved_from_head;

                if (all_half_moves.size() == 1 && all_half_moves.begin()->pos == kPassPos)
                {
                    // if (kDebugging) std::cout << "FilterBest: Only pass move is valid. Adding pass sequence." << std::endl;
                    filtered_movelist.push_back({kPassMove});
                    pass_possible = true; // Pass is the only option
                }
                /*else if (kDebugging) { // Remove debug block
                    std::cout << "FilterBest: Pass check - found " << all_half_moves.size() << " half moves. Pass not added." << std::endl;
                    for(const auto& mv : all_half_moves) {
                       std::cout << "  - Move:{" << mv.pos << "," << mv.to_pos << "," << mv.die << "}" << std::endl;
                    }
                }*/
                // Remove debug block
            }

            // *** ADDED DEBUG LOG ***
            if (kDebugging)
            {
                std::cout << "[DEBUG FBS Exit] Returning filtered_movelist (" << filtered_movelist.size() << "):\n";
                for (const auto &seq : filtered_movelist)
                {
                    std::cout << "  Seq: ";
                    for (const auto &m : seq)
                    {
                        std::cout << " {" << m.pos << "," << m.to_pos << "," << m.die << "}";
                    }
                    std::cout << std::endl;
                }
            }
            // *** END ADDED DEBUG LOG ***

            return {filtered_movelist, max_non_pass};
        }

        // Helper function to apply the "play higher die" rule if necessary.
        std::vector<Action> LongNardeState::ApplyHigherDieRuleIfNeeded(
            const std::vector<Action> &current_legal_moves,
            const std::vector<std::vector<CheckerMove>> &original_movelist /* Keep arg for now */) const
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
                    std::vector<CheckerMove> decoded_moves = SpielMoveToCheckerMoves(cur_player_, action);
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
