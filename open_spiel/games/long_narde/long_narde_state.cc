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
          is_on_first_turn_(false),    // Default to false
          dice_({0, 0, 0, 0}),         // Initialize dice_ with 4 elements
          initial_dice_({0, 0, 0, 0}), // Initialize with 4 zeros
          scores_({0, 0}),
          board_({std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
          turn_history_info_({}),
          allow_last_roll_tie_(false),
          // Initialize scoring_type_ based on game parameters
          scoring_type_(ParseScoringType(
              game->GetParameters().count("scoring_type") > 0 ? game->GetParameters().at("scoring_type").string_value() : kDefaultScoringType))
    {
      turn_history_info_.reserve(kMaxGameLengthEst);
      SetupInitialBoard();
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

    void LongNardeState::ApplyCheckerMove(int player, const CheckerMove &move)
    {
      // Handle pass move
      if (move.pos == kPassPos)
      {
        // Mark a die as used (placeholder logic, assumes pass uses die 1)
        // Need robust logic to mark the *correct* die used if pass is forced.
        // Mark a die corresponding to kPassDieValue (which is 1) if available
        bool found_die = false;
        for (int i = 0; i < dice_.size(); ++i)
        {
          if (dice_[i] == kPassDieValue)
          {                // Find a '1'
            dice_[i] = -1; // Mark as used
            found_die = true;
            break;
          }
        }
        // If die 1 wasn't available, mark the lowest available die (if any)
        if (!found_die)
        {
          for (int i = 0; i < dice_.size(); ++i)
          {
            if (dice_[i] > 0)
            {
              dice_[i] = -1; // Mark as used
              break;
            }
          }
        }
        // No board changes needed for pass
        return;
      }

      // Check if this is a head move BEFORE modifying the board
      if (IsHeadPos(player, move.pos))
      {
        moved_from_head_ = true;
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

      // Mark the die used for this move as inactive
      bool found_die = false;
      for (int i = 0; i < dice_.size(); ++i)
      {
        if (dice_[i] == move.die)
        {
          dice_[i] = -1; // Mark as used
          found_die = true;
          break;
        }
      }
      if (!found_die)
      {
        // This might happen legitimately if a higher die was used to bear off the furthest checker
        // Or if a pass move used the die needed.
        // Try to find *any* usable die that *could* have been used (e.g. a higher die for bear off)
        bool found_alternative = false;
        if (move.to_pos == kBearOffPos)
        {
          int furthest_pos = FurthestCheckerInHome(player); // Re-check furthest after potential move
          if (move.pos == furthest_pos)
          { // Was this move bearing off the furthest checker?
            for (int i = 0; i < dice_.size(); ++i)
            {
              if (dice_[i] > 0 && dice_[i] >= move.die)
              { // Found a usable die >= the die value needed
                dice_[i] = -1;
                found_alternative = true;
                break;
              }
            }
          }
        }
        if (!found_alternative)
        {
          // Still haven't found a die. This shouldn't happen for valid moves.
          std::cerr << "Current Dice: " << DiceToString() << std::endl;
          std::cerr << "Initial Dice: { ";
          for (int d : initial_dice_)
            std::cerr << d << " ";
          std::cerr << "}\n";
          std::cerr << "Attempted Move: P" << player << " Pos:" << move.pos << " To:" << move.to_pos << " Die:" << move.die << std::endl;
          SpielFatalError(absl::StrCat("ApplyCheckerMove: Die ", move.die, " not found or already used."));
        }
      }
    }

    void LongNardeState::UndoCheckerMove(int player, const CheckerMove &move)
    {
      // Handle pass move undo
      if (move.pos == kPassPos)
      {
        // Restore the die used for the pass (placeholder logic)
        // Need robust way to know *which* die was marked by ApplyCheckerMove for pass.
        // For now, try restoring the placeholder kPassDieValue (1) if it's marked used.
        bool found_used_die = false;
        for (int i = 0; i < dice_.size(); ++i)
        {
          if (dice_[i] == -1)
          { // Find a used die
            // Was this the pass die?
            // Heuristic: If initial dice had kPassDieValue, restore that.
            // Otherwise restore the lowest value die? This is fragile.
            // Assume for now the pass used kPassDieValue (1) if possible.
            bool had_pass_die_initially = false;
            for (int initial_d : initial_dice_)
            {
              if (initial_d == kPassDieValue)
              {
                had_pass_die_initially = true;
                break;
              }
            }
            if (had_pass_die_initially)
            {
              dice_[i] = kPassDieValue; // Restore '1'
              found_used_die = true;
              break;
            } // Else: Need better logic to know which die the pass *actually* consumed.
              // As a fallback, restore the lowest initial die value? Assume 1 for now.
            else
            {
              dice_[i] = kPassDieValue; // Fallback restore '1'
              found_used_die = true;
              break;
            }
          }
        }
        if (!found_used_die)
        {
          // This implies pass was undone but no die was marked - shouldn't happen
          SpielFatalError("UndoCheckerMove: Attempted to undo pass, but no die was marked as used.");
        }
        return; // No board changes needed
      }

      // Restore the die used for the move
      // This needs to handle the case where a higher die was used for bear-off.
      // Find the first occurrence of -1 in dice_ and restore move.die.
      // This assumes moves are undone in reverse order and dice are consumed deterministically.
      bool found_used_die_slot = false;
      for (int i = 0; i < dice_.size(); ++i)
      {
        if (dice_[i] == -1)
        {
          dice_[i] = move.die;
          found_used_die_slot = true;
          break;
        }
      }
      if (!found_used_die_slot)
      {
        SpielFatalError(absl::StrCat("UndoCheckerMove: Could not find used die slot (-1) to restore die ", move.die));
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
        if (board_[player][move.to_pos] <= 0)
        {
          SpielFatalError(absl::StrCat("UndoCheckerMove: No checker to remove from to_pos ", move.to_pos, " for player ", player));
        }
        board_[player][move.to_pos]--;
      }

      // Note: Undoing moved_from_head_ requires history tracking, which is handled
      // by the ExplorationState in IterativeLegalMoves or TurnHistoryInfo in ApplyAction.
      // We don't reset moved_from_head_ here directly.
    }

  } // namespace long_narde
} // namespace open_spiel