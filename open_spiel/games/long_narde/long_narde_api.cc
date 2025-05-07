#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel_utils.h"
#include <vector>
#include <memory>   // For std::unique_ptr, std::make_unique
#include <iostream> // For std::cout (used in Clone #ifndef NDEBUG)
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include <algorithm>

namespace open_spiel
{
  namespace long_narde
  {

    // ===== Core Spiel API Implementations =====

    /**
     * @brief Returns the player whose turn it is.
     *
     * Returns kTerminalPlayerId if the game has ended.
     *
     * @return The player ID (0 or 1), or kTerminalPlayerId.
     */
    Player LongNardeState::CurrentPlayer() const
    {
      return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
    }

    /**
     * @brief Applies the given action (Spiel move) to the current state.
     *
     * Handles both chance node rolls and player moves.
     * For player moves, it decodes the Spiel action into checker moves, validates
     * them (including head rule checks), applies the valid moves, updates turn
     * history, and advances the game state (player turn, dice, etc.).
     *
     * @param move_id The encoded Spiel action to apply.
     */
    void LongNardeState::DoApplyAction(Action move_id)
    {
      if (IsChanceNode())
      {
        ProcessChanceRoll(move_id);
        return;
      }

      std::vector<LongNardeCheckerMove> decoded_moves = LongNardeSpielMoveToCheckerMoves(cur_player_, move_id);
      SPIEL_CHECK_EQ(decoded_moves.size(), 2); // New encoding always gives 2 moves (passes if needed)

      bool local_head_move_occurred_this_phase = false;
      for (const auto &m : decoded_moves)
      {
        if (m.pos != kPassPos) { // Only apply non-pass moves
            LongNardeApplyCheckerMove(cur_player_, m);
            if (IsHeadPos(cur_player_, m.pos)) {
                local_head_move_occurred_this_phase = true;
            }
        }
      }
      if (local_head_move_occurred_this_phase) {
          head_move_occurred_this_full_turn_ = true;
      }

      turn_history_info_.push_back(
          TurnHistoryInfo(cur_player_, prev_player_, dice_, move_id,
                          head_move_occurred_this_full_turn_)); // Use the full turn flag

      bool was_doubles_roll = (initial_dice_.size() == 4 && 
                               initial_dice_[0] > 0 && 
                               initial_dice_[0] == initial_dice_[1] && 
                               initial_dice_[0] == initial_dice_[2] && 
                               initial_dice_[0] == initial_dice_[3]);

      int usable_dice_remaining = 0;
      for(int i = 0; i < dice_.size(); ++i) {
          if (DiceValue(i) > 0 && !IsUsed(i)) {
              usable_dice_remaining++;
          }
      }

      if (was_doubles_roll && !is_handling_second_phase_of_doubles_ && usable_dice_remaining > 0)
      {
        is_handling_second_phase_of_doubles_ = true;
        first_phase_selected_move1_ = decoded_moves[0];
        first_phase_selected_move2_ = decoded_moves[1];
      }
      else
      {
        is_handling_second_phase_of_doubles_ = false;
        dice_.assign(4, 0); 
        LongNardeAdvanceToNextPlayer(decoded_moves, move_id);
      }
    }

    /**
     * @brief Undoes the last action applied to the state.
     *
     * Restores the game state (player, dice, turn count, history flags) from the
     * most recent entry in the turn history.
     * If the action being undone was a player move, it also undoes the individual
     * checker moves in reverse order.
     *
     * @param player The player who made the action to undo.
     * @param action The Spiel action that was applied.
     */
    void LongNardeState::UndoAction(Player player, Action action)
    {
      SPIEL_CHECK_FALSE(turn_history_info_.empty());
      TurnHistoryInfo info = turn_history_info_.back();
      turn_history_info_.pop_back();

      moved_from_head_ = info.moved_from_head;
      cur_player_ = info.player;
      prev_player_ = info.prev_player;
      
      SPIEL_CHECK_EQ(info.dice.size(), 4);
      dice_ = info.dice; 

      if (player == kChancePlayerId)
      {
        if (info.prev_player == kChancePlayerId && turns_ == 0) { 
            turns_ = -1;
            initial_dice_.assign(4,0);
        }
        return;
      }

      if (cur_player_ == kTerminalPlayerId) {
        cur_player_ = player;
      }
      std::vector<LongNardeCheckerMove> moves = LongNardeSpielMoveToCheckerMoves(player, action);

      for (int i = moves.size() - 1; i >= 0; --i)
      {
        LongNardeUndoCheckerMove(player, moves[i]);
      }

      bool was_doubles_roll = info.dice[0] > 0 &&
                              info.dice[0] <= kNumDiceOutcomes && 
                              info.dice[0] == info.dice[1] &&
                              info.dice[0] == info.dice[2] &&
                              info.dice[0] == info.dice[3];

      if (!was_doubles_roll)
      {
        if (turns_ > 0) turns_--;
        if (player == kXPlayerId && x_turns_ > 0) x_turns_--;
        else if (player == kOPlayerId && o_turns_ > 0) o_turns_--;
      }
      initial_dice_ = info.dice; 
    }

    /**
     * @brief Returns a string representation of the current game state from the perspective of the specified player.
     *
     * @param player The player for whom the observation is requested (0 or 1).
     * @return A string describing the current board state, dice, and scores.
     */
    std::string LongNardeState::ObservationString(Player player) const
    {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);
      return ToString();
    }

    /**
     * @brief Fills a provided tensor with the observation data for the specified player.
     *
     * The tensor encoding includes:
     * - Board representation from the player's perspective (24 values).
     * - Board representation from the opponent's perspective (24 values).
     * - Player's score (1 value).
     * - Opponent's score (1 value).
     * - Indicator for player's turn (1 value).
     * - Indicator for opponent's turn (1 value).
     * - Dice values (2 values, 0 if not rolled).
     * The total size is kStateEncodingSize.
     *
     * @param player The player for whom the observation tensor is requested (0 or 1).
     * @param values A span of floats to be filled with the observation data. Its size must equal kStateEncodingSize.
     */
    void LongNardeState::ObservationTensor(Player player,
                                           absl::Span<float> values) const
    {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);

      int opponent = Opponent(player);
      SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
      auto value_it = values.begin();

      // Board representation: Player's checkers perspective
      for (int i = 0; i < kNumPoints; ++i)
      {
        // Map board index i to the player's path index (0=farthest, 23=closest to home)
        int path_idx = GetPathIndex(player, i);
        *(values.begin() + path_idx) = board(player, i);
      }
      value_it += kNumPoints; // Move iterator past player's board section

      for (int i = 0; i < kNumPoints; ++i)
      {
        // Map board index i to the opponent's path index (0=farthest, 23=closest to home)
        int path_idx = GetPathIndex(opponent, i);
        *(values.begin() + kNumPoints + path_idx) = board(opponent, i);
      }
      value_it += kNumPoints; // Move iterator past opponent's board section

      // Scores and turn indicator
      *value_it++ = scores_[player];
      *value_it++ = scores_[opponent];
      *value_it++ = (cur_player_ == player) ? 1.0f : 0.0f;   // Player's turn?
      *value_it++ = (cur_player_ == opponent) ? 1.0f : 0.0f; // Opponent's turn? (Should be redundant if not chance/terminal)

      // Dice values for the current phase
      if (is_handling_second_phase_of_doubles_) {
        *value_it++ = (dice_.size() >= 3) ? DiceValue(2) : 0.0f; // Die 3 for current phase
        *value_it++ = (dice_.size() >= 4) ? DiceValue(3) : 0.0f; // Die 4 for current phase
      } else {
        *value_it++ = (dice_.size() >= 1) ? DiceValue(0) : 0.0f; // Die 1 for current phase
        *value_it++ = (dice_.size() >= 2) ? DiceValue(1) : 0.0f; // Die 2 for current phase
      }

      // Second phase of doubles indicator
      *value_it++ = is_handling_second_phase_of_doubles_ ? 1.0f : 0.0f;

      // Check if iterator reached the end
      SPIEL_CHECK_EQ(value_it, values.end());
    }

    /**
     * @brief Determines if the game has reached a terminal state.
     *
     * The game ends when either player has borne off all 15 checkers.
     * Special handling exists for the WinLossTie scoring rule, where the game
     * might not be terminal immediately if one player finishes, allowing the
     * opponent a potential last roll to tie (if they have 14 or 15 checkers off).
     *
     * @return True if the game is over, false otherwise.
     */
    bool LongNardeState::IsTerminal() const
    {
      if (scores_[kXPlayerId] == kNumCheckersPerPlayer ||
          scores_[kOPlayerId] == kNumCheckersPerPlayer)
      {
        // Check for potential tie scenario if using that scoring rule
        if (scoring_type_ == ScoringType::kWinLossTieScoring)
        {
          // If White finished, but Black has 14 or 15 checkers borne off,
          // Black might get a last roll to tie. Game isn't terminal yet.
          if (scores_[kXPlayerId] == kNumCheckersPerPlayer &&
              scores_[kOPlayerId] >= 14 && scores_[kOPlayerId] < kNumCheckersPerPlayer &&
              !allow_last_roll_tie_)
          {               // Check if the tie roll has already been allowed/processed
                          // This state might need refinement: how do we know if Black's *turn* is next?
                          // If White just finished, cur_player_ should be Chance.
                          // We need to ensure Black actually gets the chance roll.
            return false; // Potential tie possible, not terminal yet.
          }
          // Symmetrically for Black finishing
          if (scores_[kOPlayerId] == kNumCheckersPerPlayer &&
              scores_[kXPlayerId] >= 14 && scores_[kXPlayerId] < kNumCheckersPerPlayer &&
              !allow_last_roll_tie_)
          {               // Check if the tie roll has already been allowed/processed
            return false; // Potential tie possible, not terminal yet.
          }
        }
        // If no tie is possible or the tie roll is handled, it's terminal.
        return true;
      }
      return false;
    }

    /**
     * @brief Returns the final scores for each player if the game is terminal.
     *
     * Returns {0, 0} if the game is not terminal.
     * Otherwise, returns the scores based on the game outcome:
     * - Tie (WinLossTieScoring only): {0, 0}
     * - White Wins (Mars): {2, -2}
     * - White Wins (Oin): {1, -1}
     * - Black Wins (Mars): {-2, 2}
     * - Black Wins (Oin): {-1, 1}
     *
     * @return A vector containing the final return for each player.
     */
    std::vector<double> LongNardeState::Returns() const
    {
      if (!IsTerminal())
      {
        return {0.0, 0.0};
      }

      bool x_won = scores_[kXPlayerId] == kNumCheckersPerPlayer;
      bool o_won = scores_[kOPlayerId] == kNumCheckersPerPlayer;

      if (x_won && o_won)
      { // Tie occurred
        return {0.0, 0.0};
      }
      else if (x_won)
      {
        double score = (scores_[kOPlayerId] > 0) ? 1.0 : 2.0; // 1 for oin, 2 for mars
        return {score, -score};
      }
      else if (o_won)
      {
        double score = (scores_[kXPlayerId] > 0) ? 1.0 : 2.0; // 1 for oin, 2 for mars
        return {-score, score};
      }
      else
      {
        // Should not happen if IsTerminal() is true and it's not a tie
        SpielFatalError("Returns() called on non-terminal or inconsistent state.");
        return {0.0, 0.0};
      }
    }

    /**
     * @brief Returns the set of possible chance outcomes (dice rolls) and their probabilities.
     *
     * In Long Narde, the standard 30 non-double dice rolls (1-2, 1-3, ..., 5-6) are possible.
     * Doubles are excluded from the initial roll but handled during regular turns.
     * This function returns the fixed set of 30 non-double outcomes, each with equal probability.
     *
     * @return A vector of pairs, where each pair contains a chance action (encoded dice roll)
     *         and its probability (1.0 / 30.0).
     */
    std::vector<std::pair<Action, double>> LongNardeState::ChanceOutcomes() const
    {
      SPIEL_CHECK_TRUE(IsChanceNode());
      // In Long Narde, the chance outcomes (dice rolls) are always the same,
      // regardless of whether it's the starting roll or a regular turn roll.
      return kChanceOutcomes;
    }

    /**
     * @brief Creates a deep copy of the current game state.
     *
     * Note: Currently implements a simple copy constructor. History management logic
     * previously present has been removed.
     *
     * @return A unique pointer to the newly created clone of the state.
     */
    std::unique_ptr<State> LongNardeState::Clone() const
    {
      // Use the default copy constructor
      auto cloned_state = std::make_unique<LongNardeState>(*this);
      return cloned_state; // Return the unique_ptr directly
    }

    // ===== Chance Node Handling =====

    /**
     * @brief Processes a chance outcome (dice roll) and updates the game state.
     *
     * This function is called when the current player is kChancePlayerId.
     * It takes the encoded chance action (dice roll), updates the internal dice_ state,
     * stores the roll history, determines the next player based on game rules (initial
     * turn, normal alternation), updates turn flags (is_first_turn_),
     * and handles the setup for a potential
     * last roll tie under the WinLossTieScoring rule.
     *
     * @param move_id The encoded chance action representing the dice roll.
     */
    void LongNardeState::ProcessChanceRoll(Action outcome)
    {
      SPIEL_CHECK_GE(outcome, 0);
      SPIEL_CHECK_LT(outcome, kNumChanceOutcomes);

      // Determine which player this roll is for.
      Player player_for_this_roll;
      // In Long Narde, White (kXPlayerId) always takes the first turn after the initial roll determination phase.
      // The 'turns_ == -1' condition signifies the game's very first roll (for determining who goes first / their dice).
      // 'this->prev_player_' stores the player who made the last *actual* (non-chance) move.
      if (turns_ == -1 || this->prev_player_ == kChancePlayerId) { 
          player_for_this_roll = kXPlayerId; // White (Player 0) gets the first turn.
      } else {
          player_for_this_roll = NextPlayerRoundRobin(this->prev_player_, num_players_);
      }
      
      this->prev_player_ = this->cur_player_; // Current cur_player_ is kChancePlayerId; store that Chance was chronologically previous to the active player's turn.
      this->cur_player_ = player_for_this_roll; // Set the actual player for this turn.

      if (turns_ == -1) { 
        turns_ = 0;
      }
      
      is_on_first_turn_ = (board(cur_player_, (cur_player_ == kXPlayerId ? kWhiteHeadPos : kBlackHeadPos)) == kNumCheckersPerPlayer);

      current_turn_full_legal_sequences_cache_.clear();
      first_phase_selected_move1_ = {kPassPos, kPassPos, 0};
      first_phase_selected_move2_ = {kPassPos, kPassPos, 0};
      is_handling_second_phase_of_doubles_ = false;
      head_move_occurred_this_full_turn_ = false;

      const std::vector<int>& roll = kChanceOutcomeValues[outcome];
      SPIEL_CHECK_EQ(roll.size(), 2);

      dice_.assign(4, 0);
      initial_dice_.assign(4, 0);

      if (roll[0] == roll[1]) {
        dice_[0] = roll[0];
        dice_[1] = roll[0];
        dice_[2] = roll[0];
        dice_[3] = roll[0];
        initial_dice_ = dice_;
      } else {
        // Non-double roll: use std::max/min for high/low die assignment
        dice_[0] = std::max(roll[0], roll[1]);
        dice_[1] = std::min(roll[0], roll[1]);
        dice_[2] = 0;
        dice_[3] = 0;
        initial_dice_ = dice_;
      }

      GenerateAndCacheFullLegalSequences();
    }

    void LongNardeState::LongNardeAdvanceToNextPlayer(const std::vector<LongNardeCheckerMove> &applied_moves,
                                             Action spiel_action)
    {
      Player moving_player = cur_player_; // Player who just finished moving

      bool was_doubles_roll = (initial_dice_.size() == 4 && initial_dice_[0] > 0 && initial_dice_[0] == initial_dice_[1] && initial_dice_[1] == initial_dice_[2] && initial_dice_[2] == initial_dice_[3]);

      int num_actual_moves = 0;
      for (const auto &m : applied_moves)
      {
        if (m.pos != kPassPos)
        {
          num_actual_moves++;
        }
      }
      int potential_moves = was_doubles_roll ? 4 : 2;

      Player next_player_id; // Who will play AFTER the next dice roll?

      next_player_id = NextPlayerRoundRobin(moving_player, num_players_);
      is_on_first_turn_ = false; // Can never be the first turn after the first move sequence
      turns_++;
      if (next_player_id == kXPlayerId)
        x_turns_++;
      else
        o_turns_++; // Increment for the player whose turn is STARTING

      moved_from_head_ = false;

      prev_player_ = moving_player; // Correctly set prev_player_ to the player who just moved

      cur_player_ = kChancePlayerId;
    }

    /**
     * @brief Returns the dice values as a string.
     *
     * Formats the current dice roll (e.g., "3 5", "6 6 (u)") indicating used dice.
     *
     * @return A string representation of the dice state.
     */
    std::string LongNardeState::DiceToString() const
    {
      std::string dice_str = "";
      if (dice_.empty())
      {
        dice_str = "(None)";
      }
      else
      {
        bool first_die = true;
        for (size_t i = 0; i < dice_.size(); ++i)
        {
          int die_val = DiceValue(i); // Get actual value (1-6)
          if (die_val == 0)
            continue; // Skip unused slots

          if (!first_die)
            dice_str += " ";
          dice_str += std::to_string(die_val);
          if (!IsDieUsable(i))
            dice_str += "(u)";
          first_die = false;
        }
        if (first_die)
        { // Means all slots were 0 (shouldn't happen if dice_ not empty)
          dice_str = "(Invalid Dice State)";
        }
      }
      return dice_str;
    }

    void LongNardeState::RollDice(Action outcome)
    {
      const std::vector<int> &roll = kChanceOutcomeValues[outcome];
      dice_.resize(4, 0); // Always resize to 4 elements, initialize with 0
      if (roll[0] == roll[1])
      { // Doubles
        dice_[0] = roll[0];
        dice_[1] = roll[0]; // Keep index 1 for potential use in IsDieUsable logic if needed
        dice_[2] = roll[0];
        dice_[3] = roll[0];
      }
      else
      { // Not doubles
        // Ensure higher die is first
        if (roll[0] > roll[1])
        {
          dice_[0] = roll[0];
          dice_[1] = roll[1];
        }
        else
        {
          dice_[0] = roll[1]; // Put higher die (roll[1]) first
          dice_[1] = roll[0]; // Put lower die (roll[0]) second
        }
        // dice_[2] and dice_[3] remain 0
      }
    }

  } // namespace long_narde
} // namespace open_spiel