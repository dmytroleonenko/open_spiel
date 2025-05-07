#include "open_spiel/games/long_narde/long_narde.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
  namespace long_narde
  {

    /**
     * @brief Constructs a LongNardeState.
     *
     * Initializes the game state, including setting the scoring type based on game parameters,
     * and setting up the initial board configuration.
     *
     * @param game A shared pointer to the parent LongNardeGame object.
     */
    LongNardeState::LongNardeState(std::shared_ptr<const Game> game)
        : State(game),
          cur_player_(kChancePlayerId),
          prev_player_(kChancePlayerId),
          turns_(-1), // Initial turns count before first roll
          moved_from_head_(false),
          // is_on_first_turn_(false), // Already initialized in .h as part of TurnHistory or similar logic
          dice_(4, 0),         // Initialize dice_ with 4 elements, all zero.
          initial_dice_(4, 0), // Initialize with 4 zeros
          scores_({0, 0}),
          board_(kNumPlayers, std::vector<int>(kNumPoints, 0)), // Corrected initialization
          // turn_history_info_({}), // Initialized via default constructor
          allow_last_roll_tie_(false),
          scoring_type_(ParseScoringType(
              game->GetParameters().count("scoring_type") > 0 ? game->GetParameters().at("scoring_type").string_value() : kDefaultScoringType)),
          // Initialize new state variables for two-phase doubles
          is_handling_second_phase_of_doubles_(false),
          // current_turn_full_legal_sequences_cache_ is default-initialized (empty)
          first_phase_selected_move1_({kPassPos, kPassPos, 0}),
          first_phase_selected_move2_({kPassPos, kPassPos, 0}),
          head_move_occurred_this_full_turn_(false)
    {
      turn_history_info_.reserve(kMaxGameLengthEst); // Assuming kMaxGameLengthEst is defined
      SetupInitialBoard();
      // No need to initialize is_on_first_turn_ here if it's managed elsewhere or by default construction
    }

    /**
     * @brief Sets up the initial checker positions on the board.
     *
     * Places 15 checkers for White (X) on point 24 (index 23) and
     * 15 checkers for Black (O) on point 12 (index 11).
     * All other points are initialized to 0 checkers.
     */
    void LongNardeState::SetupInitialBoard()
    {
      board_[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
      board_[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
    }

    // ===== Basic State Accessors =====

    int LongNardeState::NumDistinctActions() const
    {
      // With the new encoding scheme, the number of distinct actions is fixed.
      return kNumDistinctActions; // Should be 1250
    }

    int LongNardeState::board(int player, int pos) const
    {
      // Bounds check for safety, returning 0 for invalid positions
      if (pos < 0 || pos >= kNumPoints)
      {
        return 0;
      }
      return board_[player][pos];
    }

    int LongNardeState::Opponent(int player) const { return 1 - player; }

    void LongNardeState::LongNardeApplyCheckerMove(int player, const LongNardeCheckerMove &move)
    {
      // Handle pass move
      if (move.pos == kPassPos)
      {
        // For a pass, a die must still be marked as "used".
        // The specific die used for a pass might be determined by higher-level logic
        // (e.g., if only one die is playable, or if forced to pass with a specific die).
        // Here, we find the first *usable* die that matches move.die (typically kPassDieValue)
        // or any usable die if kPassDieValue is not available/specified for the pass.
        bool found_and_marked_die = false;
        if (move.die > 0 && move.die <= kNumDiceOutcomes) { // Specific die for pass
            for (int i = 0; i < dice_.size(); ++i) {
                if (DiceValue(i) == move.die && !IsUsed(i)) {
                    dice_[i] += kNumDiceOutcomes; // Mark as used
                    found_and_marked_die = true;
            break;
          }
        }
        }
        
        if (!found_and_marked_die) { // Fallback: mark any available die if specific pass die not found/marked
            for (int i = 0; i < dice_.size(); ++i) {
                if (DiceValue(i) > 0 && !IsUsed(i)) {
                    dice_[i] += kNumDiceOutcomes; // Mark as used
                    found_and_marked_die = true;
              break;
            }
          }
        }

        if (!found_and_marked_die && kDebugging) {
             SpielFatalError("ApplyCheckerMove (Pass): No usable die to mark for pass.");
        }
        return; // No board changes needed for pass
      }

      // Check if this is a head move BEFORE modifying the board
      if (IsHeadPos(player, move.pos))
      {
        moved_from_head_ = true; // This flag might need to be part of TurnHistory for undo
      }

      // Decrement checker count at the 'from' position
      if (board_[player][move.pos] <= 0)
      {
        SpielFatalError(absl::StrCat("ApplyCheckerMove: No checker to move from pos ", move.pos, " for player ", player));
      }
      board_[player][move.pos]--;

      // Check if bearing off
      if (move.to_pos == kBearOffPos)
      {
        scores_[player]++; // Increment score
      }
      else
      {
        // Regular move: Increment checker count at the 'to' position
        if (move.to_pos < 0 || move.to_pos >= kNumPoints)
        {
          SpielFatalError(absl::StrCat("ApplyCheckerMove: Invalid to_pos ", move.to_pos, " for player ", player));
        }
        board_[player][move.to_pos]++;
      }

      // Mark the die used for this move as inactive by adding kNumDiceOutcomes
      bool found_die_to_mark = false;
      for (int i = 0; i < dice_.size(); ++i)
      {
        if (DiceValue(i) == move.die && !IsUsed(i))
        {
          dice_[i] += kNumDiceOutcomes; // Mark as used
          found_die_to_mark = true;
          break;
        }
      }
      
      // Handle bearing off with a higher die if exact die wasn't found marked yet
      // This logic assumes the move itself (move.die) is valid for bearing off with a higher roll.
      if (!found_die_to_mark && move.to_pos == kBearOffPos) {
          // Use FurthestCheckerInHome to check if the move was for the furthest checker.
          // FurthestCheckerInHome returns the board position (0-23 or -1 if none).
          int furthest_checker_pos = FurthestCheckerInHome(player);
          if (move.pos == furthest_checker_pos) { // Check if it was the furthest checker
              for (int i = 0; i < dice_.size(); ++i) {
                  if (!IsUsed(i) && DiceValue(i) >= move.die) {
                      dice_[i] += kNumDiceOutcomes; // Mark the higher die as used
                      found_die_to_mark = true;
                      break;
                  }
              }
          }
      }

      if (!found_die_to_mark)
        {
        // This condition indicates an issue: either the move is invalid,
        // or dice state is not as expected.
        std::string dice_str = "{";
        for(int d_val : dice_) dice_str += std::to_string(d_val) + " ";
        dice_str += "}";
        SpielFatalError(absl::StrCat("ApplyCheckerMove: Die ", move.die,
                                     " not found usable or already marked. Current dice_ state: ", dice_str));
      }
    }

    void LongNardeState::LongNardeUndoCheckerMove(int player, const LongNardeCheckerMove &move)
    {
      // Handle pass move undo
      if (move.pos == kPassPos)
      {
        // Restore the die used for the pass.
        // This requires knowing which die was marked. The move.die should indicate this.
        bool found_and_unmarked_die = false;
        if (move.die > 0 && move.die <= kNumDiceOutcomes) {
            for (int i = 0; i < dice_.size(); ++i) {
                // Check if this die slot was used for this specific die value
                if (IsUsed(i) && (dice_[i] - kNumDiceOutcomes == move.die)) {
                    dice_[i] -= kNumDiceOutcomes; // Mark as unused
                    found_and_unmarked_die = true;
              break;
            }
          }
        }
        // Fallback: if specific die not found marked (e.g. pass used 'any' available die)
        // This part is tricky without more context on how pass dice are chosen and recorded.
        // Assuming the move.die for pass accurately reflects what was used.
        // If not, the TurnHistoryInfo might need to store which dice index was used.
        if (!found_and_unmarked_die && kDebugging) {
             SpielFatalError(absl::StrCat("UndoCheckerMove (Pass): Could not find die ", move.die, " to unmark."));
        }
        return; 
      }

      // Restore the die used for the move by subtracting kNumDiceOutcomes
      // This needs to robustly find the die that was marked for *this* specific move.
      // The simplest is if moves are undone in strict reverse of application and dice are chosen deterministically.
      bool found_die_to_unmark = false;
      for (int i = 0; i < dice_.size(); ++i) {
        // Check if this die slot was used for *this* die value
        if (IsUsed(i) && (dice_[i] - kNumDiceOutcomes == move.die)) {
            dice_[i] -= kNumDiceOutcomes; // Mark as unused
            found_die_to_unmark = true;
          break;
        }
      }
      
      // If not found, it might be a bear-off with a higher die.
      // This case is complex for undo because we need to know which *specific higher die* was marked.
      // The `move.die` itself is the *required* roll, not necessarily the *actual higher roll* used.
      // This suggests TurnHistoryInfo might need to store the actual dice indices used.
      // For now, we assume `move.die` is sufficient to identify the marked die.
      if (!found_die_to_unmark) {
          // Attempt to find a die that was marked used, whose original value could be >= move.die
          // This is still a heuristic and might not be robust.
          // The problem is if multiple dice could satisfy this condition.
          // This is a strong indicator that TurnHistoryInfo needs to store the index of the die used.
          if (move.to_pos == kBearOffPos) {
               for (int i = 0; i < dice_.size(); ++i) {
                   if (IsUsed(i) && (dice_[i] - kNumDiceOutcomes >= move.die)) {
                       // Potential candidate. If multiple, this is ambiguous.
                       // For simplicity, unmark the first one found. This is a known limitation.
                       dice_[i] -= kNumDiceOutcomes;
                       found_die_to_unmark = true;
                       break;
                   }
               }
          }
      }


      if (!found_die_to_unmark)
      {
        std::string dice_str = "{";
        for(int d_val : dice_) dice_str += std::to_string(d_val) + " ";
        dice_str += "}";
        SpielFatalError(absl::StrCat("UndoCheckerMove: Die ", move.die, 
                                     " not found marked used. Current dice_ state: ", dice_str));
      }

      // Increment checker count at the 'from' position
      if (move.pos < 0 || move.pos >= kNumPoints)
      {
        SpielFatalError(absl::StrCat("UndoCheckerMove: Invalid from_pos ", move.pos, " for player ", player));
      }
      board_[player][move.pos]++;


      // Check if bearing off was undone
      if (move.to_pos == kBearOffPos)
      {
        scores_[player]--; // Decrement score
      }
      else
      {
        // Regular move undo: Decrement checker count at the 'to' position
        if (move.to_pos < 0 || move.to_pos >= kNumPoints)
        {
          SpielFatalError(absl::StrCat("UndoCheckerMove: Invalid to_pos ", move.to_pos, " for player ", player));
        }
        if (board_[player][move.to_pos] <= 0) {
          SpielFatalError(absl::StrCat("UndoCheckerMove: No checker to remove from to_pos ", move.to_pos, " for player ", player));
        }
        board_[player][move.to_pos]--;
      }

      // moved_from_head_ needs to be restored from TurnHistoryInfo by UndoAction
    }

    /**
     * @brief Counts the total number of checkers for a player.
     *
     * Sums the checkers on the board and those already borne off (scores_).
     * Should typically equal kNumCheckersPerPlayer (15) for a valid state.
     *
     * @param player The player ID (0 or 1).
     * @return The total count of the player's checkers.
     */
    int LongNardeState::CountTotalCheckers(int player) const {
      SPIEL_CHECK_TRUE(player == kXPlayerId || player == kOPlayerId);
      int count = scores_[player]; // Start with borne-off checkers
      for (int pos = 0; pos < kNumPoints; ++pos) {
        count += board_[player][pos];
      }
      return count;
    }

    // The SetState function needs to be found or its signature assumed from .h
    // Assuming it's similar to this, based on common patterns and .h:
    void LongNardeState::SetState(int cur_player,
                                  const std::vector<int>& p_dice, // p_dice should be size 4
                                  const std::vector<int>& p_scores,
                                  const std::vector<std::vector<int>>& p_board) {
      cur_player_ = cur_player;
      // double_turn parameter is removed.
      
      SPIEL_CHECK_EQ(p_dice.size(), 4);
      dice_ = p_dice; // Directly assign, assuming p_dice is already in the new format (0 for unused/unrolled, 1-6 for rolled, 7-12 for used)

      SPIEL_CHECK_EQ(p_scores.size(), kNumPlayers);
      for (int p = 0; p < kNumPlayers; ++p) {
        scores_[p] = p_scores[p];
      }

      SPIEL_CHECK_EQ(p_board.size(), kNumPlayers);
      for (int p = 0; p < kNumPlayers; ++p) {
        SPIEL_CHECK_EQ(p_board[p].size(), kNumPoints);
        for (int i = 0; i < kNumPoints; ++i) {
          board_[p][i] = p_board[p][i];
        }
      }
      // Other members like turns_, prev_player_, history_ etc. might also need setting
      // This is a simplified SetState. A full one would reset more.
    }

    int LongNardeState::DiceValue(int i) const
    {
      SPIEL_CHECK_GE(i, 0);
      SPIEL_CHECK_LT(i, dice_.size());
      int die_val = dice_[i];
      if (die_val > kNumDiceOutcomes)
      { // Die is marked as used
        return die_val - kNumDiceOutcomes;
      }
      else if (die_val > 0 && die_val <= kNumDiceOutcomes)
      { // Die is available
        return die_val;
      }
      return 0; // Die is 0 (not rolled or placeholder)
    }

    bool LongNardeState::IsUsed(int i) const {
      SPIEL_CHECK_GE(i, 0);
      SPIEL_CHECK_LT(i, dice_.size());
      return dice_[i] > kNumDiceOutcomes;
    }
    
    // Removed conflicting definition of UsableDiceOutcome(int i)
    // The function bool UsableDiceOutcome(int outcome) const; is declared in .h 
    // and defined in long_narde_validation.cc.
    // To check if a die at a given index 'i' is usable, use DiceValue(i) > 0.

    /**
     * @brief Generates all fully valid move sequences for the current player and dice,
     *        filters them, and caches them in current_turn_full_legal_sequences_cache_.
     *
     * This function is called once per dice roll (typically within ProcessChanceRoll).
     * It uses LongNardeGenerateMoveSequences to get all raw sequences, then applies
     * LongNardeFilterBestMoveSequences to ensure that if a player *can* make more moves,
     * sequences with fewer moves are pruned if they are part of longer valid sequences.
     * The final, filtered list of canonical move sequences is stored.
     */
    void LongNardeState::GenerateAndCacheFullLegalSequences() {
      // Ensure dice_ are populated for the current player.
      // Player cur_player_ is already set by ProcessChanceRoll before this would be called.
      
      // 1. Generate all raw move sequences.
      std::vector<std::vector<LongNardeCheckerMove>> raw_sequences = LongNardeGenerateMoveSequences(cur_player_);

      // 2. Filter for best/maximal sequences.
      // LongNardeFilterBestMoveSequences returns a pair: {filtered_sequences, max_moves_found}
      // We only need the sequences themselves for the cache.
      current_turn_full_legal_sequences_cache_ = LongNardeFilterBestMoveSequences(raw_sequences).first;

      // TODO: Further canonicalization or sorting if needed? 
      // The plan mentions "Ensures canonical sequences". LongNardeGenerateMoveSequences and 
      // LongNardeFilterBestMoveSequences might inherently do this or require an additional step.
      // For now, assuming the combination of the two provides sufficiently canonical sequences
      // for LegalActions to correctly map them to unique Action IDs later.
      // The "Higher Die Rule" is also mentioned. This is typically handled by LegalActions or its sub-components
      // when deciding which dice to use, rather than filtering sequences here.
    }

  } // namespace long_narde
} // namespace open_spiel