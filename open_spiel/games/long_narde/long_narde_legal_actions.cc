#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> 
#include <iostream>  
#include <memory>    
#include <utility>   
#include <stack>     
#include <sstream>   

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" 

namespace open_spiel
{
    namespace long_narde
    {

        // ===== Legal/Illegal Action Generation =====

        std::vector<Action> LongNardeState::LegalActions() const
        {
            if (IsTerminal())
                return {};
            if (IsChanceNode())
                return LegalChanceOutcomes();

            std::set<Action> unique_actions;
            if (moves_remaining_ > 0)
            {
                auto all_sequences = LongNardeGenerateMoveSequences(CurrentPlayer());
                auto filter_result = LongNardeFilterBestMoveSequences(all_sequences);
                for (const auto &seq : filter_result.first)
                {
                    if (seq.empty()){
                        continue;
                    }
                    const LongNardeCheckerMove &move = seq[0];
                    Action a = (move.pos == kPassPos ? kNumPoints : move.pos);
                    unique_actions.insert(a);
                }
            }
            else
            {
                return {};
            }

            // Ensure there's always at least a pass move if no other moves are valid.
            if (unique_actions.empty()) {
                unique_actions.insert(kNumPoints); // kNumPoints represents the pass action here
            }

            return std::vector<Action>(unique_actions.begin(), unique_actions.end());
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
            std::unique_ptr<LongNardeState> state;
            std::vector<LongNardeCheckerMove> current_sequence;
            LongNardeCheckerMove move_applied;
            int depth;
            bool moved_from_head_in_sequence;
            ExplorationState(std::unique_ptr<LongNardeState> s, int d, bool initial_home_status)
                : state(std::move(s)), move_applied({kPassPos, kPassPos, 0}), depth(d), moved_from_head_in_sequence(initial_home_status) {}
            ExplorationState(std::unique_ptr<LongNardeState> s,
                             const std::vector<LongNardeCheckerMove> &seq,
                             const LongNardeCheckerMove &move,
                             int d,
                             bool moved_head)
                : state(std::move(s)), current_sequence(seq), move_applied(move), depth(d), moved_from_head_in_sequence(moved_head)
            {
            }
        };

        // Iterative helper for LegalActions. Explores possible move sequences using DFS.
        int LongNardeState::LongNardeIterativeLegalMoves(
            const std::vector<LongNardeCheckerMove> &current_sequence,
            std::vector<std::vector<LongNardeCheckerMove>> *moves_list
            ) const
        {

            const size_t kMaxTotalSequences = 200;
            const size_t kMaxBranchingFactor = 30;
            const int kMaxIterationDepth = 6;

            std::stack<ExplorationState> exploration_stack;

            bool initial_moved_from_head = this->moved_from_head_;

            exploration_stack.emplace(
                std::unique_ptr<LongNardeState>(static_cast<LongNardeState *>(this->Clone().release())),
                0,
                initial_moved_from_head
            );

            int max_non_pass_found = 0;

            while (!exploration_stack.empty())
            {
                ExplorationState current_exploration = std::move(exploration_stack.top());
                exploration_stack.pop();

                std::unique_ptr<LongNardeState> current_state_ptr = std::move(current_exploration.state);
                LongNardeState *current_state = current_state_ptr.get();
                const std::vector<LongNardeCheckerMove> &current_sequence = current_exploration.current_sequence;
                const LongNardeCheckerMove &move_applied_to_reach_this = current_exploration.move_applied;
                int current_depth = current_exploration.depth;

                bool sequence_limit_hit = (moves_list->size() >= kMaxTotalSequences);

                if (sequence_limit_hit || current_depth > kMaxIterationDepth)
                {
                    if (!current_sequence.empty())
                    {
                        moves_list->push_back(current_sequence);
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    continue;
                }

                if (current_state->IsTerminal())
                {
                    if (!current_sequence.empty())
                    {
                        moves_list->push_back(current_sequence);
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    else
                    {
                    }
                    continue;
                }

                bool current_moved_from_head = current_exploration.moved_from_head_in_sequence;

                std::set<LongNardeCheckerMove> half_moves = current_state->LongNardeGenerateAllHalfMoves(current_state->CurrentPlayer(), current_moved_from_head);

                if (half_moves.empty() || current_sequence.size() >= game_->MaxGameLength())
                {
                    if (!current_sequence.empty())
                    {
                        moves_list->push_back(current_sequence);
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                    else if (half_moves.size() == 1 && half_moves.begin()->pos == kPassPos)
                    {
                        std::vector<LongNardeCheckerMove> actual_pass_sequence;
                        const std::vector<int> &initial_dice = this->LongNardeInitialDice();

                        SPIEL_CHECK_GE(initial_dice.size(), 2);

                        if (initial_dice.size() >= 2 && initial_dice[0] == initial_dice[1])
                        {
                            int die_value = initial_dice[0];
                            actual_pass_sequence.push_back({kPassPos, kPassPos, die_value});
                            actual_pass_sequence.push_back({kPassPos, kPassPos, die_value});
                        }
                        else if (initial_dice.size() >= 2)
                        {
                            actual_pass_sequence.push_back({kPassPos, kPassPos, initial_dice[0]});
                            actual_pass_sequence.push_back({kPassPos, kPassPos, initial_dice[1]});
                        }

                        if (!actual_pass_sequence.empty())
                        {
                            moves_list->push_back(actual_pass_sequence);
                        }
                    }
                    else
                    {
                    }
                    continue;
                }

                size_t explored_branches = 0;
                bool found_move_in_iteration = false;
                Player player = current_state->CurrentPlayer();

                std::vector<LongNardeCheckerMove> moves_to_explore;
                for (const auto &move : half_moves)
                {
                    if (move.pos != kPassPos)
                    {
                        moves_to_explore.push_back(move);
                    }
                }

                bool ownership_transferred = false;
                for (int i = 0; i < moves_to_explore.size(); ++i)
                {
                    const LongNardeCheckerMove &next_move = moves_to_explore[i];

                    current_state->LongNardeApplyCheckerMove(player, next_move);
                    found_move_in_iteration = true;

                    std::vector<LongNardeCheckerMove> next_sequence = current_sequence;
                    next_sequence.push_back(next_move);

                    bool next_moved_from_head = current_exploration.moved_from_head_in_sequence || current_state->IsHeadPos(player, next_move.pos);

                    if (current_state->IsTerminal())
                    {
                        moves_list->push_back(next_sequence);
                        int non_pass = 0;
                        for (const auto &m : next_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        if (i != moves_to_explore.size() - 1)
                        {
                            current_state->LongNardeUndoCheckerMove(player, next_move);
                        }
                    }
                    else
                    {
                        bool is_last_move = (i == moves_to_explore.size() - 1);

                        if (!is_last_move)
                        {
                            std::unique_ptr<LongNardeState> next_state_for_stack(
                                static_cast<LongNardeState *>(current_state->Clone().release()));
                            exploration_stack.emplace(std::move(next_state_for_stack), next_sequence, next_move, current_depth + 1, next_moved_from_head);
                            current_state->LongNardeUndoCheckerMove(player, next_move);
                        }
                        else
                        {
                            exploration_stack.emplace(std::move(current_state_ptr), next_sequence, next_move, current_depth + 1, next_moved_from_head);
                            ownership_transferred = true;
                        }
                    }

                    explored_branches++;
                }

                if (!found_move_in_iteration && !current_sequence.empty())
                {
                    bool had_usable_dice = false;
                    for (int i = 0; i < current_state->dice_.size(); ++i)
                    {
                        if (current_state->IsDieUsable(i))
                        {
                            had_usable_dice = true;
                            break;
                        }
                    }

                    bool forced_pass = had_usable_dice && half_moves.size() == 1 && half_moves.begin()->pos == kPassPos;

                    if (forced_pass)
                    {
                        int pass_die_value = -1;
                        for (int i = 0; i < current_state->dice_.size(); ++i)
                        {
                            if (current_state->IsDieUsable(i))
                            {
                                pass_die_value = current_state->DiceValue(i);
                                break;
                            }
                        }
                        if (pass_die_value != -1)
                        {
                            std::vector<LongNardeCheckerMove> sequence_with_pass = current_sequence;
                            sequence_with_pass.push_back(LongNardeCheckerMove(kPassPos, kPassPos, pass_die_value));
                            moves_list->push_back(sequence_with_pass);
                            int non_pass = 0;
                            for (const auto &m : sequence_with_pass)
                                if (m.pos != kPassPos)
                                    non_pass++;
                            max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        }
                        else
                        {
                            moves_list->push_back(current_sequence);
                            int non_pass = 0;
                            for (const auto &m : current_sequence)
                                if (m.pos != kPassPos)
                                    non_pass++;
                            max_non_pass_found = std::max(max_non_pass_found, non_pass);
                        }
                    }
                    else
                    {
                        moves_list->push_back(current_sequence);
                        int non_pass = 0;
                        for (const auto &m : current_sequence)
                            if (m.pos != kPassPos)
                                non_pass++;
                        max_non_pass_found = std::max(max_non_pass_found, non_pass);
                    }
                }

            }

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

            int longest_sequence = 0;
            for (const auto &moveseq : movelist)
            {
                longest_sequence = std::max(longest_sequence, static_cast<int>(moveseq.size()));
            }

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

            std::vector<std::vector<LongNardeCheckerMove>> filtered_movelist;
            bool pass_possible = false;

            bool has_terminal_sequence = false;
            std::vector<std::vector<LongNardeCheckerMove>> terminal_sequences;

            for (const auto &moveseq : movelist)
            {
                std::unique_ptr<LongNardeState> test_state = std::unique_ptr<LongNardeState>(static_cast<LongNardeState *>(this->Clone().release()));

                for (const auto &move : moveseq)
                {
                    if (move.pos != kPassPos)
                    {
                        test_state->LongNardeApplyCheckerMove(test_state->CurrentPlayer(), move);
                    }
                }

                if (test_state->IsTerminal())
                {
                    has_terminal_sequence = true;
                    terminal_sequences.push_back(moveseq);
                }
            }

            if (has_terminal_sequence)
            {
                filtered_movelist = terminal_sequences;
            }
            else
            {
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
                            pass_possible = true;
                        }
                    }
                    else if (is_pass_sequence && max_non_pass == 0 && longest_sequence <= 1)
                    {
                        if (moveseq.size() == 1 && moveseq[0].pos == kPassPos)
                        {
                            filtered_movelist.push_back(moveseq);
                            pass_possible = true;
                        }
                    }
                }
            }

            return {filtered_movelist, max_non_pass};
        }

        // Helper function to apply the "play higher die" rule if necessary.
        std::vector<Action> LongNardeState::LongNardeApplyHigherDieRuleIfNeeded(
            const std::vector<Action> &current_legal_moves,
            const std::vector<std::vector<LongNardeCheckerMove>> &original_movelist ) const
        {

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

            bool is_doubles = (dice_.size() == 4 && DiceValue(0) > 0 && DiceValue(0) == DiceValue(1));

            if (max_non_pass == 1 && !is_doubles)
            {
                SPIEL_CHECK_GE(dice_.size(), 2);
                int d1 = DiceValue(0);
                int d2 = DiceValue(1);
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
                            break;
                        }
                    }
                }

                if (found_higher)
                {
                    return actions_using_higher;
                }
                else if (found_lower)
                {
                    return actions_using_lower;
                }
                else if (!current_legal_moves.empty())
                {
                    SpielFatalError(absl::StrCat("ApplyHigherDieRule: No moves found matching dice when max_non_pass=1. Higher: ", higher_die, ", Lower: ", lower_die));
                    return {};
                }
                else
                {
                    return {};
                }
            }

            return current_legal_moves;
        }

    } // namespace long_narde
} // namespace open_spiel
