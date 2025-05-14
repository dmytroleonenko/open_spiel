#include "open_spiel/games/long_narde/long_narde.h"
#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
  namespace long_narde
  {

    // Precomputed masks of opponent positions ahead of each point for each player.
    uint32_t opponent_ahead_mask_[kNumPlayers][kNumPoints];
    // Static initializer to fill in opponent_ahead_mask_
    struct OpponentAheadMaskInitializer {
      OpponentAheadMaskInitializer() {
        for (int p = 0; p < kNumPlayers; ++p) {
          for (int pos = 0; pos < kNumPoints; ++pos) {
            uint32_t mask = 0;
            if (p == kXPlayerId) {
              // White moves decreasing indices; ahead are positions < pos
              for (int q = 0; q < pos; ++q) mask |= (1u << q);
            } else {
              // Black moves increasing indices; ahead are positions > pos
              for (int q = pos + 1; q < kNumPoints; ++q) mask |= (1u << q);
            }
            opponent_ahead_mask_[p][pos] = mask;
          }
        }
      }
    } opponent_ahead_mask_initializer;

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
          // board_({std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
          turn_history_info_({}),
          allow_last_roll_tie_(false),
          is_first_phase_of_doubles_(false), // NEW: Initialize to false
          // Initialize scoring_type_ based on game parameters
          scoring_type_(ParseScoringType(
              game->GetParameters().count("scoring_type") > 0 ? game->GetParameters().at("scoring_type").string_value() : kDefaultScoringType))
    {
      turn_history_info_.reserve(kMaxGameLengthEst);
      std::fill_n(board_data_, kNumPlayers * kNumPoints, 0); // Initialize board_data_
      SetupInitialBoard();
      // Initialize bitboard occupancy and checker counts from board_data_
      for (int p = 0; p < kNumPlayers; ++p) {
        uint32_t occ = 0;
        for (int pos = 0; pos < kNumPoints; ++pos) {
          uint8_t point_count = board_data_[p * kNumPoints + pos];
          if (point_count > 0) {
            occ |= (1u << pos);
          }
        }
        player_occupancy_[p] = occ;
      }
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
      board_data_[kXPlayerId * kNumPoints + kWhiteHeadPos] = kNumCheckersPerPlayer;
      board_data_[kOPlayerId * kNumPoints + kBlackHeadPos] = kNumCheckersPerPlayer;
    }

    // ===== Basic State Accessors =====

    int LongNardeState::Opponent(int player) const { return 1 - player; }

    void LongNardeState::LongNardeApplyCheckerMove(int player, const LongNardeCheckerMove &move)
    {
      // Handle pass move
      if (move.pos == kPassPos)
      {
        bool found_die = false;
        for (int i = 0; i < dice_.size(); ++i)
        {
          if (dice_[i] == kPassDieValue)
          {
            dice_[i] += 6; // Mark as used by adding 6
            found_die = true;
            break;
          }
        }
        // If die 1 wasn't available, mark the lowest available die
        if (!found_die)
        {
          for (int i = 0; i < dice_.size(); ++i)
          {
            if (dice_[i] > 0)
            {
              dice_[i] += 6; // Mark as used by adding 6
              break;
            }
          }
        }
        // No board changes needed for pass
        return;
      }

      // Set moved_from_head_ if a checker moves from the player's head position.
      // This flag is used by LegalActions to enforce head movement rules.
      if ((player == kXPlayerId && move.pos == kWhiteHeadPos) || 
          (player == kOPlayerId && move.pos == kBlackHeadPos)) {
        moved_from_head_ = true;
      }

      // Decrement checker count at the 'from' position
      if (GetCount(player, move.pos) <= 0)
      {
        SpielFatalError(absl::StrCat("ApplyCheckerMove: No checker to move from pos ", move.pos, " for player ", player));
      }
      DecrementPoint(player, move.pos);

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
        IncrementPoint(player, move.to_pos);
      }

      // Mark the die used for this move as inactive (NEW: use +6 marking)
      bool found_die = false;
      for (int i = 0; i < dice_.size(); ++i)
      {
        if (dice_[i] == move.die)
        {
          dice_[i] = move.die + 6; // Mark as used by adding 6
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
              if (dice_[i] > 0 && dice_[i] >= move.die && dice_[i] <= 6)
              { // Found a usable die >= the die value needed
                dice_[i] = dice_[i] + 6;
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

    void LongNardeState::LongNardeUndoCheckerMove(int player, const LongNardeCheckerMove &move)
    {
      // Handle pass move undo
      if (move.pos == kPassPos)
      {
        // Restore the die used for the pass by reverting +6 marking
        bool found_used_die = false;
        for (int i = 0; i < dice_.size(); ++i)
        {
          if (dice_[i] == move.die + 6)
          {
            dice_[i] = move.die;
            found_used_die = true;
            break;
          }
        }
        if (!found_used_die)
        {
          SpielFatalError(absl::StrCat("UndoCheckerMove: Could not find used die slot (", move.die + 6, ") to restore die ", move.die));
        }
        return;
      }

      // Restore the die used for the move (NEW: use -6 to restore)
      bool found_used_die_slot = false;
      for (int i = 0; i < dice_.size(); ++i)
      {
        if (dice_[i] == move.die + 6)
        {
          dice_[i] = move.die;
          found_used_die_slot = true;
          break;
        }
      }
      if (!found_used_die_slot)
      {
        SpielFatalError(absl::StrCat("UndoCheckerMove: Could not find used die slot (", move.die + 6, ") to restore die ", move.die));
      }

      // Increment checker count at the 'from' position
      if (move.pos < 0 || move.pos >= kNumPoints)
      {
        SpielFatalError(absl::StrCat("UndoCheckerMove: Invalid from_pos ", move.pos, " for player ", player));
      }
      IncrementPoint(player, move.pos);

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
        if (GetCount(player, move.to_pos) <= 0)
        {
          SpielFatalError(absl::StrCat("UndoCheckerMove: No checker to remove from to_pos ", move.to_pos, " for player ", player));
        }
        DecrementPoint(player, move.to_pos);
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
    int LongNardeState::CountTotalCheckers(int player) const {
      SPIEL_CHECK_TRUE(player == kXPlayerId || player == kOPlayerId);
      int count = scores_[player]; // Start with borne-off checkers
      for (int pos = 0; pos < kNumPoints; ++pos) {
        count += GetCount(player, pos);
      }
      return count;
    }

    void LongNardeState::SetPointCount(int player, int pos, uint8_t count) {
      if (pos < 0 || pos >= kNumPoints) {
        SpielFatalError("SetPointCount: Invalid position");
      }
      uint8_t prev = board_data_[player * kNumPoints + pos];
      board_data_[player * kNumPoints + pos] = count;
      // Update occupancy
      if (count > 0) {
        player_occupancy_[player] |= (1u << pos);
      } else {
        player_occupancy_[player] &= ~(1u << pos);
      }
    }

    void LongNardeState::IncrementPoint(int player, int pos) {
      if (pos < 0 || pos >= kNumPoints) {
        SpielFatalError("IncrementPoint: Invalid position");
      }
      uint8_t prev = board_data_[player * kNumPoints + pos];
      board_data_[player * kNumPoints + pos]++;
      if (prev == 0) {
        player_occupancy_[player] |= (1u << pos);
      }
    }

    void LongNardeState::DecrementPoint(int player, int pos) {
      if (pos < 0 || pos >= kNumPoints) {
        SpielFatalError("DecrementPoint: Invalid position");
      }
      uint8_t prev = board_data_[player * kNumPoints + pos];
      if (prev == 0) {
        SpielFatalError("DecrementPoint: No checker to decrement");
      }
      board_data_[player * kNumPoints + pos]--;
      if (prev == 1) {
        player_occupancy_[player] &= ~(1u << pos);
      }
    }

  } // namespace long_narde
} // namespace open_spiel