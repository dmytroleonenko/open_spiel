#include "open_spiel/games/long_narde/long_narde.h"

#include <vector>
#include <set>
#include <algorithm> // For std::max, std::min, std::sort, std::unique
#include <iostream>  // For kDebugging cout/cerr
#include <memory>    // For unique_ptr
#include <utility>   // For std::pair
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

            // --- Two-Phase Dice Override: use only two active dice for this phase ---
            LongNardeState* mutable_this = const_cast<LongNardeState*>(this);
            std::vector<int> saved_dice = mutable_this->dice_;
            bool is_doubles = (initial_dice_[0] == initial_dice_[1] && initial_dice_[0] > 0);
            bool is_phase1 = is_doubles && is_first_phase_of_doubles_;
            mutable_this->dice_.assign(4, 0);
            if (is_phase1 || !is_doubles) {
                mutable_this->dice_[0] = initial_dice_[0];
                mutable_this->dice_[1] = initial_dice_[1];
            } else {
                // Phase 2 of doubles
                mutable_this->dice_[0] = initial_dice_[2];
                mutable_this->dice_[1] = initial_dice_[3];
            }
            // Generate all possible move sequences via recursive generator
            std::set<std::pair<std::vector<LongNardeCheckerMove>, bool>> movelist_set;
            RecLegalMoveSequences({}, &movelist_set, false);
            // Restore original dice after generation
            mutable_this->dice_ = saved_dice;
            std::vector<std::pair<std::vector<LongNardeCheckerMove>, bool>> movelist_with_flags(
                movelist_set.begin(), movelist_set.end());

            // Filter for the best move sequences (longest, max non-pass)
            auto filter_result = LongNardeFilterBestMoveSequences(movelist_with_flags);
            std::vector<std::vector<LongNardeCheckerMove>> filtered_movelist = filter_result.first;
            int max_non_pass = filter_result.second;

            // If filtering resulted in only a pass sequence (and original was pass-only), convert and return it.
            if (max_non_pass == 0 && !filtered_movelist.empty())
            {
                SPIEL_CHECK_GE(dice_.size(), 2);
                std::vector<LongNardeCheckerMove> actual_pass_sequence;
                if (is_doubles && !is_first_phase_of_doubles_) {
                  actual_pass_sequence.push_back({kPassPos, kPassPos, initial_dice_[2]});
                  actual_pass_sequence.push_back({kPassPos, kPassPos, initial_dice_[3]});
                } else {
                  actual_pass_sequence.push_back({kPassPos, kPassPos, DiceValue(0)});
                  actual_pass_sequence.push_back({kPassPos, kPassPos, DiceValue(1)});
                }
                return {LongNardeCheckerMovesToSpielMove(actual_pass_sequence)};
            }

            // Convert filtered move sequences to Spiel Actions
            const size_t kMaxActionsToGenerate = 20;
            std::set<Action> unique_actions;

            for (const auto &moveseq : filtered_movelist)
            {
                if (unique_actions.size() >= kMaxActionsToGenerate)
                    break;
                Action action = LongNardeCheckerMovesToSpielMove(moveseq);
                unique_actions.insert(action);
            }

            std::vector<Action> legal_moves;
            legal_moves.assign(unique_actions.begin(), unique_actions.end());

            std::vector<Action> final_actions = LongNardeApplyHigherDieRuleIfNeeded(legal_moves, filtered_movelist);
            return final_actions;
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

        std::vector<std::vector<LongNardeCheckerMove>> LongNardeState::LongNardeGenerateMoveSequences(Player player) const {
            // Use the recursive generator for move sequences
            std::set<std::pair<std::vector<LongNardeCheckerMove>, bool>> movelist_set;
            RecLegalMoveSequences({}, &movelist_set, false);
            std::vector<std::pair<std::vector<LongNardeCheckerMove>, bool>> movelist_with_flags(
                movelist_set.begin(), movelist_set.end());
            std::vector<std::vector<LongNardeCheckerMove>> movelist;
            movelist.reserve(movelist_with_flags.size());
            for (const auto& entry : movelist_with_flags) {
                movelist.push_back(entry.first);
            }
            // Sort for deterministic ordering
            std::sort(movelist.begin(), movelist.end());
            return movelist;
        }

        std::set<LongNardeCheckerMove> LongNardeState::LongNardeGenerateAllHalfMoves(int player, bool moved_from_head_this_sequence) const {
            // Reserve buffer for at most kNumPoints * dice_.size() moves
            std::vector<LongNardeCheckerMove> buf;
            buf.reserve(kNumPoints * static_cast<int>(dice_.size()));
            // Generate candidates with inline deduplication
            for (int pos = 0; pos < kNumPoints; ++pos) {
              if (GetCount(player, pos) <= 0) continue;
              for (int die_idx = 0; die_idx < dice_.size(); ++die_idx) {
                if (!IsDieUsable(die_idx)) continue;
                int die_value = DiceValue(die_idx);
                int to_pos = GetToPos(player, pos, die_value);
                LongNardeCheckerMove move(pos, to_pos < 0 ? kBearOffPos : to_pos, die_value);
                if (!LongNardeIsValidCheckerMove(player, move, moved_from_head_this_sequence)) continue;
                // Inline deduplication
                bool exists = false;
                for (const auto &m : buf) {
                  if (m == move) { exists = true; break; }
                }
                if (!exists) buf.push_back(move);
              }
            }
            // If no moves, insert pass placeholder
            if (buf.empty()) {
              buf.emplace_back(kPassPos, kPassPos, 1);
            }
            // Build and return set
            return std::set<LongNardeCheckerMove>(buf.begin(), buf.end());
        }

        // ----- End Iterative Implementation -----

        // Recursive helper for move sequence generation.
        int LongNardeState::RecLegalMoveSequences(std::vector<LongNardeCheckerMove> moveseq,
                                                  std::set<std::pair<std::vector<LongNardeCheckerMove>, bool>>* movelist,
                                                  bool moved_from_head_this_sequence) const {
          // Count usable dice
          int usable_dice = 0;
          for (int i = 0; i < dice_.size(); ++i) {
            if (IsDieUsable(i)) ++usable_dice;
          }
          // Always use max_moves = 2 for two-phase system
          int max_moves = 2;
          const auto& initial = this->LongNardeInitialDice();
          bool is_doubles = (initial.size() >= 2 && initial[0] == initial[1] && initial[0] > 0);
          bool is_phase1 = is_doubles && is_first_phase_of_doubles_;
          bool is_phase2 = is_doubles && !is_first_phase_of_doubles_;
          // Base case: no dice left or max moves reached
          if (usable_dice == 0 || moveseq.size() >= max_moves) {
            bool is_term = this->IsTerminal();
            movelist->insert({moveseq, is_term});
            int non_pass = 0;
            for (const auto& m : moveseq) if (m.pos != kPassPos) ++non_pass;
            return non_pass;
          }

          // Generate all possible half-moves from this state
          std::set<LongNardeCheckerMove> half_moves =
              LongNardeGenerateAllHalfMoves(cur_player_, moved_from_head_this_sequence);

          // Pass-only case: use correct dice for phase 2 of doubles
          if (half_moves.size() == 1 && half_moves.begin()->pos == kPassPos) {
            std::vector<LongNardeCheckerMove> pass_seq = moveseq;
            if (moveseq.empty()) {
              if (is_doubles && is_phase2) {
                pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[2]));
                pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[3]));
              } else if (is_doubles) {
                pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[1]));
              } else {
                pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[0]));
                pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, initial[1]));
              }
              { bool is_term = this->IsTerminal(); movelist->insert({pass_seq, is_term}); }
              int non_pass = 0;
              for (const auto& m : pass_seq) if (m.pos != kPassPos) ++non_pass;
              return non_pass;
            } else {
              // Forced pass after some moves: one pass for first usable die
              for (int i = 0; i < dice_.size(); ++i) {
                if (IsDieUsable(i)) {
                  pass_seq.push_back(LongNardeCheckerMove(kPassPos, kPassPos, DiceValue(i)));
                  break;
                }
              }
              { bool is_term = this->IsTerminal(); movelist->insert({pass_seq, is_term}); }
              int non_pass = 0;
              for (const auto& m : pass_seq) if (m.pos != kPassPos) ++non_pass;
              return non_pass;
            }
          }

          // Recursive case: apply each half-move
          int max_non_pass = -1;
          for (const auto& move : half_moves) {
            // Apply move
            LongNardeState* mutable_this = const_cast<LongNardeState*>(this);
            mutable_this->LongNardeApplyCheckerMove(cur_player_, move);
            moveseq.push_back(move);
            bool next_moved_from_head = moved_from_head_this_sequence ||
                IsHeadPos(cur_player_, move.pos);
            int child_max = mutable_this->RecLegalMoveSequences(
                moveseq, movelist, next_moved_from_head);
            max_non_pass = std::max(child_max, max_non_pass);
            // Undo move
            moveseq.pop_back();
            mutable_this->LongNardeUndoCheckerMove(cur_player_, move);
          }
          return max_non_pass;
        }

        // Helper function to filter generated sequences for the best ones.
        std::pair<std::vector<std::vector<LongNardeCheckerMove>>, int> LongNardeState::LongNardeFilterBestMoveSequences(
            const std::vector<std::pair<std::vector<LongNardeCheckerMove>, bool>> &movelist_with_flags) const
        {
            if (movelist_with_flags.empty()) return {{}, 0};

            int longest_sequence = 0;
            for (const auto &entry : movelist_with_flags) {
                longest_sequence = std::max(longest_sequence, static_cast<int>(entry.first.size()));
            }

            int max_non_pass = 0;
            for (const auto &entry : movelist_with_flags) {
                if (entry.first.size() == longest_sequence) {
                    int current_non_pass = 0;
                    for (const auto &move : entry.first) if (move.pos != kPassPos) ++current_non_pass;
                    max_non_pass = std::max(max_non_pass, current_non_pass);
                }
            }

            std::vector<std::vector<LongNardeCheckerMove>> filtered_movelist;
            bool has_terminal_sequence = false;
            std::vector<std::vector<LongNardeCheckerMove>> terminal_sequences;
            // Collect terminal sequences from flags
            for (const auto &entry : movelist_with_flags) {
                if (entry.second) {
                    has_terminal_sequence = true;
                    terminal_sequences.push_back(entry.first);
                }
            }
            if (has_terminal_sequence) {
                filtered_movelist = terminal_sequences;
            } else {
                for (const auto &entry : movelist_with_flags) {
                    const auto &moveseq = entry.first;
                    int current_non_pass = 0;
                    bool is_pass_sequence = true;
                    for (const auto &move : moveseq) {
                        if (move.pos != kPassPos) {
                            current_non_pass++;
                            is_pass_sequence = false;
                        }
                    }
                    if (moveseq.size() == longest_sequence && current_non_pass == max_non_pass) {
                        filtered_movelist.push_back(moveseq);
                    } else if (is_pass_sequence && max_non_pass == 0 && longest_sequence <= 1) {
                        if (moveseq.size() == 1 && moveseq[0].pos == kPassPos) {
                            filtered_movelist.push_back(moveseq);
                        }
                    }
                }
            }
            // Pass fallback without cloning
            if (filtered_movelist.empty() && max_non_pass == 0 && longest_sequence == 0) {
                LongNardeState *mutable_this = const_cast<LongNardeState *>(this);
                auto original_dice = mutable_this->dice_;
                bool original_moved_from_head = mutable_this->moved_from_head_;
                Player current_player = this->cur_player_;
                mutable_this->moved_from_head_ = false;
                std::set<LongNardeCheckerMove> all_half_moves =
                    mutable_this->LongNardeGenerateAllHalfMoves(current_player, mutable_this->moved_from_head_);
                mutable_this->dice_ = original_dice;
                mutable_this->moved_from_head_ = original_moved_from_head;
                if (all_half_moves.size() == 1 && all_half_moves.begin()->pos == kPassPos) {
                    filtered_movelist.push_back({kPassMove});
                }
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
