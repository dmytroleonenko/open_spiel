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
          moves_remaining_(0),
          turns_(-1),
          moved_from_head_(false),
          is_on_first_turn_(false),
          dice_({0, 0, 0, 0}),
          initial_dice_({0, 0, 0, 0}),
          scores_({0, 0}),
          board_({std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
          turn_history_info_({}),
          allow_last_roll_tie_(false),
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
      if (pos < 0 || pos >= kNumPoints)
      {
        return 0;
      }
      return board_[player][pos];
    }

    int LongNardeState::Opponent(int player) const { return 1 - player; }

    void LongNardeState::LongNardeApplyCheckerMove(int player, const LongNardeCheckerMove &move)
    {
      if (move.pos == kPassPos)
      {
        bool found_die = false;
        for (int i = 0; i < dice_.size(); ++i)
        {
          if (dice_[i] == move.die)
          {
            dice_[i] = -1;
            found_die = true;
            break;
          }
        }
        if (!found_die)
        {
          SpielFatalError(absl::StrCat("ApplyCheckerMove: Pass move die ", move.die, " not found or already used."));
        }
        return;
      }

      if (IsHeadPos(player, move.pos))
      {
        moved_from_head_ = true;
      }

      if (board_[player][move.pos] <= 0)
      {
        SpielFatalError(absl::StrCat("ApplyCheckerMove: No checker to move from pos ", move.pos, " for player ", player));
      }
      board_[player][move.pos]--;

      if (move.to_pos == kBearOffPos)
      {
        scores_[player]++;
      }
      else
      {
        if (move.to_pos < 0 || move.to_pos >= kNumPoints)
        {
          SpielFatalError(absl::StrCat("ApplyCheckerMove: Invalid to_pos ", move.to_pos, " for player ", player));
        }
        board_[player][move.to_pos]++;
      }

      bool found_die = false;
      for (int i = 0; i < dice_.size(); ++i)
      {
        if (dice_[i] == move.die)
        {
          dice_[i] = -1;
          found_die = true;
          break;
        }
      }
      if (!found_die)
      {
        bool found_alternative = false;
        if (move.to_pos == kBearOffPos)
        {
          int furthest_pos = FurthestCheckerInHome(player);
          if (move.pos == furthest_pos)
          {
            for (int i = 0; i < dice_.size(); ++i)
            {
              if (dice_[i] > 0 && dice_[i] >= move.die)
              {
                dice_[i] = -1;
                found_alternative = true;
                break;
              }
            }
          }
        }
        if (!found_alternative)
        {
          SpielFatalError(absl::StrCat("ApplyCheckerMove: Die ", move.die, " not found or already used."));
        }
      }
    }

    void LongNardeState::LongNardeUndoCheckerMove(int player, const LongNardeCheckerMove &move)
    {
      if (move.pos == kPassPos)
      {
        bool found_used_die = false;
        for (int i = 0; i < dice_.size(); ++i)
        {
          if (dice_[i] == -1)
          {
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
              dice_[i] = kPassDieValue;
              found_used_die = true;
              break;
            }
            else
            {
              dice_[i] = kPassDieValue;
              found_used_die = true;
              break;
            }
          }
        }
        if (!found_used_die)
        {
          SpielFatalError("UndoCheckerMove: Attempted to undo pass, but no die was marked as used.");
        }
        return;
      }

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

      if (move.pos < 0 || move.pos >= kNumPoints)
      {
        SpielFatalError(absl::StrCat("UndoCheckerMove: Invalid from_pos ", move.pos, " for player ", player));
      }
      board_[player][move.pos]++;

      if (move.to_pos == kBearOffPos)
      {
        scores_[player]--;
      }
      else
      {
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

    /**
     * @brief Counts the total number of checkers for a player.
     *
     * Sums the checkers on the board and those already borne off (scores_).
     * Should typically equal kNumCheckersPerPlayer (15) for a valid state.
     *
     * @param player The player ID (0 or 1).
     * @return The total count of the player's checkers.
     */
    int LongNardeState::CountTotalCheckers(int player) const
    {
      SPIEL_CHECK_TRUE(player == kXPlayerId || player == kOPlayerId);
      int count = scores_[player];
      for (int pos = 0; pos < kNumPoints; ++pos)
      {
        count += board_[player][pos];
      }
      return count;
    }

  } // namespace long_narde
} // namespace open_spiel