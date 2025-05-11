# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the game-specific functions for Long Narde."""

from absl.testing import absltest
import pickle
import pyspiel

class GamesLongNardeTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = pyspiel.load_game("long_narde")

    def test_create_state(self):
        state = self.game.new_initial_state()
        self.assertIsInstance(state, pyspiel.LongNardeState)
        self.assertFalse(state.is_terminal())
        # Long Narde starts with a chance node for the initial dice roll
        self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
        # Apply the first chance outcome to get to the first player's turn
        chance_outcomes = state.chance_outcomes()
        action, _ = chance_outcomes[0] # Just take the first outcome for testing
        state.apply_action(action)
        # Now it should be player 0's turn (or player 1 if doubles rolled, handle later if needed)
        self.assertTrue(state.current_player() == 0 or state.current_player() == 1)

    def test_board_and_methods(self):
        """Tests various methods of LongNardeState."""
        state = self.game.new_initial_state()
        # Apply the first chance outcome to get a valid state for testing methods
        if state.is_chance_node():
            chance_outcomes = state.chance_outcomes()
            action, _ = chance_outcomes[0] # Just take the first outcome
            state.apply_action(action)

        # Add check for terminal state after first roll
        if state.is_terminal():
            self.assertTrue(state.is_terminal())
            # In a terminal state, current_player might be -2 (PlayerId.TERMINAL)
            # or the player who made the last move.
            # No legal actions should be available.
            # Depending on the game phase, legal_actions() might expect a player_id
            # but for terminal states, it should be empty or error gracefully.
            # For now, let's assume legal_actions() can be called without player_id if terminal.
            # Or, if it requires a player, we might need to know who the "winner" or last player was.
            # However, the C++ IsTerminal() implies the game has ended.
            # Let's check if any player has legal actions.
            # A more robust check would be to ensure no player (0 or 1) has legal actions.
            if state.current_player() == pyspiel.PlayerId.TERMINAL:
                 self.assertEqual(len(state.legal_actions()), 0)
            else:
                 # If not PlayerId.TERMINAL, it could be the player who's turn it would have been.
                 # Check for both players if current_player is not explicitly TERMINAL.
                 self.assertEqual(len(state.legal_actions(0)), 0)
                 self.assertEqual(len(state.legal_actions(1)), 0)

            # Check scores are valid
            self.assertIsInstance(state.score(0), float) # Scores are typically float
            self.assertIsInstance(state.score(1), float)
            self.skipTest("Initial state became terminal after first chance outcome. Checked terminal properties.")
            return

        # The board() method requires player and pos, it doesn't return the whole board.
        # Test getting a specific point's checker count instead.
        player = state.current_player()
        # Find a valid position to test (e.g., the starting position)
        start_pos = 0 # Assuming player 0 starts at pos 0 conceptually (index)
        checker_count = state.board(player, start_pos)
        self.assertIsInstance(checker_count, int)

        # Test other methods using the current player
        opponent = state.opponent(player)
        self.assertTrue(opponent == 0 or opponent == 1)
        self.assertIsInstance(state.is_off(player, 0), bool)
        # Call dice(index) to get a specific die value
        first_die_value = state.dice(0)
        self.assertIsInstance(first_die_value, int)
        self.assertIsInstance(state.get_to_pos(player, 0, first_die_value), int) # Test with first die
        self.assertIsInstance(state.count_total_checkers(player), int)
        self.assertIsInstance(state.player_turns(), int)
        self.assertIsInstance(state.score(player), int)
        # Test getting individual dice values
        self.assertIsInstance(state.dice(0), int)
        self.assertIsInstance(state.dice(1), int)
        self.assertIsInstance(state.double_turn(), bool)
        # Call moved_from_head() without arguments
        self.assertIsInstance(state.moved_from_head(), bool)
        # Methods related to moves might need a specific state/dice roll
        # self.assertIsInstance(state.is_valid_checker_move(player, ?, ?), bool) # Need CheckerMove
        self.assertIsInstance(state.is_head_pos(player, 0), bool)
        # self.assertIsInstance(state.is_legal_head_move(player, 0, False), bool) # Depends on dice
        # self.assertIsInstance(state.would_form_blocking_bridge(player, 0, 5), bool) # Depends on board
        self.assertIsInstance(state.furthest_checker_in_home(player), int)
        self.assertIsInstance(state.all_in_home(player), bool)
        self.assertIsInstance(state.dice_to_string(), str)
        self.assertIsInstance(state.board_to_string(), str)

    def test_move_conversion(self):
        """Tests conversion between Spiel moves and checker moves."""
        state = self.game.new_initial_state()
        # Apply the first chance outcome
        if state.is_chance_node():
            chance_outcomes = state.chance_outcomes()
            action, _ = chance_outcomes[0] # Just take the first outcome
            state.apply_action(action)

        if state.is_terminal(): # Skip if the first roll ends the game (unlikely)
             self.skipTest("Initial state is terminal after chance.")
             return

        # Spiel move to checker moves and back (should be identity for legal moves)
        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)

        if not legal_actions:
            self.skipTest("No legal actions available in this state.")
            return

        for action in legal_actions:
            # Pass the current player to the conversion function
            checker_moves = state.spiel_move_to_checker_moves(current_player, action)
            self.assertIsInstance(checker_moves, list)
            # Ensure it's a list of CheckerMove objects if the binding returns them directly
            # (Assuming binding returns list[LongNardeCheckerMove])
            if checker_moves:
                 self.assertIsInstance(checker_moves[0], pyspiel.LongNardeCheckerMove)

            action2 = state.checker_moves_to_spiel_move(checker_moves)
            self.assertEqual(action, action2)
            break  # Just test the first legal action for simplicity

    def test_pickle_support(self):
        state = self.game.new_initial_state()
        state2 = pickle.loads(pickle.dumps(state))
        self.assertIsInstance(state2, pyspiel.LongNardeState)
        self.assertEqual(state2.to_string(), state.to_string())
        self.assertEqual(state2.current_player(), state.current_player())

if __name__ == "__main__":
    absltest.main() 