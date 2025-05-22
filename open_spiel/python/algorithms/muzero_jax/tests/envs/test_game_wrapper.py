import jax
import jax.numpy as jnp
import pytest
import pyspiel # Required for pyspiel.Game
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper # Added import
"""Removed the unconditional skip so that game wrapper tests execute when pyspiel is available."""


def test_tic_tac_toe_wrapper():
    # Test basic non-chance game wrapper functionality
    wrapper = GameWrapper("tic_tac_toe")
    # Reset and initial observation
    obs = wrapper.reset()
    assert isinstance(obs, list)
    # Legal actions should match OpenSpiel's legal_actions
    initial_state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    expected_actions = initial_state.legal_actions()
    assert wrapper.legal_actions() == list(expected_actions)
    # Number of distinct actions matches game
    assert wrapper.num_distinct_actions() == pyspiel.load_game("tic_tac_toe").num_distinct_actions()
    # TicTacToe has no chance nodes
    assert wrapper.is_chance_node() is False
    # Test is_terminal() is False after reset on Tic-Tac-Toe
    assert not wrapper.is_terminal(), "Game should not be terminal after reset"
    # Take a valid action and step
    action = expected_actions[0]
    obs2, rewards, done = wrapper.step(action)
    assert isinstance(obs2, list)
    assert isinstance(rewards, list)
    assert isinstance(done, bool)
    # After step, legal_actions should update
    assert wrapper.legal_actions() != initial_state.legal_actions()


def test_kuhn_poker_chance_wrapper():
    # Test chance node functionality using a game with chance nodes
    wrapper = GameWrapper("kuhn_poker")
    obs = wrapper.reset()
    # Assert reset() returns [] for Kuhn Poker (chance node at start)
    assert obs == [], "reset() should return an empty list for an initial chance node"
    # At initial state of Kuhn Poker, should be a chance node
    assert wrapper.is_chance_node() is True
    # Assert legal_actions() on Kuhn Poker's initial chance node matches pyspiel
    kuhn_game = pyspiel.load_game("kuhn_poker")
    kuhn_initial_state = kuhn_game.new_initial_state()
    assert wrapper.legal_actions() == list(kuhn_initial_state.legal_actions()), \
        "legal_actions() at initial chance node should match OpenSpiel"
    # Chance outcomes should return a list of (action, prob)
    outcomes = wrapper.chance_outcomes()
    assert isinstance(outcomes, list)
    assert len(outcomes) > 0
    for outcome in outcomes:
        assert isinstance(outcome, tuple)
        assert len(outcome) == 2
        action, prob = outcome
        assert isinstance(action, int)
        assert isinstance(prob, float)
    # Assert chance_outcomes() probabilities sum to 1.0
    if outcomes: # Ensure outcomes is not empty before summing
        total_p = sum(prob for _, prob in outcomes)
        assert abs(total_p - 1.0) < 1e-6, f"Probabilities in chance_outcomes should sum to 1.0, got {total_p}"

    # Applying a chance action should transition to next state
    first_action = outcomes[0][0]
    # For chance node, use step to apply a chance action
    obs_ch, rewards_ch, done_ch = wrapper.step(first_action)
    assert isinstance(obs_ch, list)
    assert isinstance(rewards_ch, list)
    assert isinstance(done_ch, bool)
    # After the first chance event (dealing one card), it's still a chance node (dealing second card)
    assert wrapper.is_chance_node() is True

    # Apply second chance action
    second_action = wrapper.chance_outcomes()[0][0]
    wrapper.step(second_action)
    # After both chance events (dealing two cards), it should be a player node
    assert wrapper.is_chance_node() is False


def test_current_observation_and_legal_actions_after_reset_and_step():
    # Test current_observation and legal_actions for a non-chance game
    wrapper = GameWrapper("tic_tac_toe")
    # Reset to initial state
    init_obs = wrapper.reset()
    # current_observation should match reset's observation
    assert wrapper.current_observation() == init_obs
    # Legal actions should be a list of ints
    actions = wrapper.legal_actions()
    assert isinstance(actions, list)
    assert all(isinstance(a, int) for a in actions)
    # Step with first action and test current_observation updates
    if actions:
        wrapper.step(actions[0])
        obs_after = wrapper.current_observation()
        assert isinstance(obs_after, list)
        # Observation can change after a move
        assert obs_after != init_obs or len(init_obs) == 0 


def test_chance_outcomes_off_chance_node():
    # Test chance_outcomes on a non-chance node (e.g., TicTacToe)
    wrapper = GameWrapper("tic_tac_toe")
    wrapper.reset()
    # TicTacToe initial state is not a chance node
    assert wrapper.is_chance_node() is False
    # chance_outcomes should be empty for non-chance nodes
    assert wrapper.chance_outcomes() == []
    # Play to a terminal state
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # A sequence of moves leading to a terminal state in TicTacToe
    # Player 1: (0,0)
    # Player 2: (1,0)
    # Player 1: (0,1)
    # Player 2: (1,1)
    # Player 1: (0,2) -> Player 1 wins
    actions_to_terminal = [0, 3, 1, 4, 2]
    for action in actions_to_terminal:
        wrapper.step(action)
        state.apply_action(action) # Keep track with a manual state

    assert wrapper.is_terminal() # Should be terminal now
    assert state.is_terminal() # Manual state also terminal
    # chance_outcomes should be empty for terminal states
    assert wrapper.chance_outcomes() == [] 


def test_reset_idempotence():
    # Test that reset restores the game to its initial state properties
    wrapper = GameWrapper("tic_tac_toe")

    # Initial state properties
    initial_obs = wrapper.reset()
    initial_legal_actions = wrapper.legal_actions()
    initial_is_chance = wrapper.is_chance_node()
    initial_current_obs = wrapper.current_observation()

    # Take a step
    if initial_legal_actions:
        wrapper.step(initial_legal_actions[0])
        # Verify state has changed (or could be terminal)
        assert wrapper.current_observation() != initial_current_obs or wrapper.is_terminal()

    # Reset again
    obs_after_reset = wrapper.reset()
    legal_actions_after_reset = wrapper.legal_actions()
    is_chance_after_reset = wrapper.is_chance_node()
    current_obs_after_reset = wrapper.current_observation()

    # Assertions
    assert obs_after_reset == initial_obs
    assert legal_actions_after_reset == initial_legal_actions
    assert is_chance_after_reset == initial_is_chance
    assert current_obs_after_reset == initial_current_obs


def test_terminal_transitions_and_rewards():
    # Test terminal state properties and rewards
    wrapper = GameWrapper("tic_tac_toe")
    game = pyspiel.load_game("tic_tac_toe") # For num_players
    wrapper.reset()

    # Sequence of moves to a terminal state (Player 1 wins)
    # P1: (0,0) -> 0
    # P2: (1,0) -> 3
    # P1: (0,1) -> 1
    # P2: (1,1) -> 4
    # P1: (0,2) -> 2 (winning move)
    actions_to_win = [0, 3, 1, 4, 2]
    obs, rews, done = [], [], False
    for i, action in enumerate(actions_to_win):
        obs, rews, done = wrapper.step(action)
        if done:
            break
    
    assert done is True, "Game should be terminal after winning sequence"
    assert isinstance(rews, list), "Rewards should be a list"
    assert len(rews) == game.num_players(), "Rewards list length should match num_players"
    assert all(isinstance(r, (int, float)) for r in rews), "Rewards should be numeric"
    # In TicTacToe, for a win, one player gets 1.0, other -1.0 (or 0.0 if not their turn to lose)
    # The exact reward depends on the perspective. OpenSpiel returns for current player then others.
    # We expect one 1.0 and one -1.0 in some order for a win/loss scenario.
    assert 1.0 in rews and -1.0 in rews, f"Expected win/loss rewards [1.0, -1.0], got {rews}"
    assert wrapper.legal_actions() == [], "Legal actions should be empty at a terminal state"
    assert wrapper.is_terminal() is True, "Wrapper should report terminal state"

    # Test a draw scenario
    wrapper_draw = GameWrapper("tic_tac_toe")
    wrapper_draw.reset()
    # P1: (1,1) -> 4
    # P2: (0,0) -> 0
    # P1: (0,1) -> 1
    # P2: (2,1) -> 7
    # P1: (2,0) -> 6
    # P2: (0,2) -> 2
    # P1: (1,0) -> 3
    # P2: (1,2) -> 5
    # P1: (2,2) -> 8 (draw)
    actions_to_draw = [4, 0, 1, 7, 6, 2, 3, 5, 8]
    obs_d, rews_d, done_d = [], [], False
    for action in actions_to_draw:
        obs_d, rews_d, done_d = wrapper_draw.step(action)
        if done_d:
            break
    
    assert done_d is True, "Game should be terminal after draw sequence"
    assert all(r == 0.0 for r in rews_d), f"Expected all zero rewards for a draw, got {rews_d}"
    assert wrapper_draw.legal_actions() == [], "Legal actions should be empty at a terminal state (draw)"
    assert wrapper_draw.is_terminal() is True, "Wrapper should report terminal state (draw)"


def test_current_observation_at_chance_node():
    # Test current_observation at a chance node
    wrapper = GameWrapper("kuhn_poker")
    wrapper.reset() # Kuhn Poker starts with a chance node
    assert wrapper.is_chance_node() is True
    # current_observation should return an empty list for chance nodes
    assert wrapper.current_observation() == []

    # Step through chance nodes until it's a player node
    while wrapper.is_chance_node():
        action = wrapper.chance_outcomes()[0][0]
        wrapper.step(action)
    
    # Now it should be a player node, current_observation should not be empty
    assert wrapper.is_chance_node() is False
    current_obs = wrapper.current_observation()
    assert isinstance(current_obs, list)
    assert len(current_obs) > 0, "Current observation should not be empty at a player node"


def test_observation_content_and_changes():
    # Test observation content and changes for a simple game like TicTacToe
    wrapper = GameWrapper("tic_tac_toe")
    game = pyspiel.load_game("tic_tac_toe")
    initial_obs = wrapper.reset()

    # Check initial observation length
    assert len(initial_obs) == game.observation_tensor_size(), \
        f"Observation length {len(initial_obs)} does not match game's tensor size {game.observation_tensor_size()}"
    
    # Initial observation content depends on the game, removed strict all-zero check.
    # The more important check is that it changes predictably and matches a manual state.

    # Take a specific action and check how observation changes
    # Action 0 corresponds to (0,0) on the board for player 1
    action_to_take = 0
    if action_to_take in wrapper.legal_actions():
        obs_after_step, _, _ = wrapper.step(action_to_take)
        current_obs_after_step = wrapper.current_observation()

        assert obs_after_step == current_obs_after_step, \
            "Observation from step() should match subsequent current_observation()"

        # Verify the observation tensor reflects the move
        # OpenSpiel TicTacToe observation: board (3x3=9) + player turn (1) = 10 elements.
        # The board part is usually flattened. A move by player 1 at (0,0) should mark that spot.
        # The exact representation can vary, this is a conceptual check.
        # For TicTacToe, a common representation is 1 for player 1, -1 for player 2.
        # Assuming the first element of the board part corresponds to (0,0)
        # and player 1 is represented by 1.0
        # This depends on the specific game's observation_tensor implementation.
        # We will check that *something* changed from initial_obs.
        assert obs_after_step != initial_obs, \
             "Observation should change after a valid move."
        
        # More specific check if we know the representation:
        # For pyspiel.tic_tac_toe, the observation is a flat list of board cells + current player.
        # Cells are 0 for empty, 1 for current player, -1 for opponent.
        # After P1 moves to (0,0), obs[0] should be 1 (current player), others 0.
        # And the last element (player turn) might change or stay indicating next player.
        expected_board_part_after_action_0 = [0.0] * 9 # Assuming 3x3 board
        # If player 1 made the move (action 0), cell 0 should be 1.0.
        # The observation tensor in OpenSpiel represents the board from the perspective of the current player.
        # So after player 1 moves, it's player 2's turn. Player 1's piece is 1.0, Player 2's is -1.0.
        # If action 0 is (0,0), then obs[0] would be 1.0 from P1's perspective after their move, but P2 sees it as -1.0.
        # The state observation_tensor() is from the perspective of the *current* player to play.
        # So, after P1 (current_player=0) plays action 0, state switches to P2 (current_player=1).
        # P2 sees P1's piece at (0,0). P1's piece is 1.0 *for P1*. For P2, P1 is an opponent, so P2 sees -1.0.
        # Let's test this based on how `state.observation_tensor()` works for `tic_tac_toe`.
        manual_state = game.new_initial_state()
        manual_state.apply_action(action_to_take)
        expected_obs_from_manual_state = list(manual_state.observation_tensor())

        assert obs_after_step == expected_obs_from_manual_state, \
            f"Wrapper observation {obs_after_step} did not match manual state observation {expected_obs_from_manual_state}"
    else:
        pytest.skip(f"Action {action_to_take} not available for content check.")


def test_invalid_action_handling():
    # Test that attempting to take an illegal action raises an error
    wrapper = GameWrapper("tic_tac_toe")
    wrapper.reset()
    game = pyspiel.load_game("tic_tac_toe")
    num_actions = game.num_distinct_actions()

    # Find an invalid action
    # An action is invalid if it's not in legal_actions()
    # Or simply pick an action outside the valid range [0, num_actions - 1]
    invalid_action = -1 # Definitely invalid
    with pytest.raises(RuntimeError): # OpenSpiel typically raises RuntimeError for illegal actions
        wrapper.step(invalid_action)

    invalid_action_large = num_actions # Also definitely invalid
    with pytest.raises(RuntimeError):
        wrapper.step(invalid_action_large)

    # Try an action that might be initially valid but becomes invalid
    wrapper.reset() # Start fresh
    # In TicTacToe, action 0 is (0,0). Take it.
    if 0 in wrapper.legal_actions():
        wrapper.step(0)
        # Now, action 0 should be illegal (cell is occupied).
        assert 0 not in wrapper.legal_actions()
        with pytest.raises(RuntimeError):
            wrapper.step(0) # Attempting to play on an occupied cell
    else:
        pytest.skip("Action 0 was not initially legal, cannot test re-playing it.")


def test_jax_compatibility_smoke_test():
    # Test that outputs can be converted to JAX arrays without error
    wrapper = GameWrapper("tic_tac_toe")
    initial_obs_list = wrapper.reset()

    try:
        initial_obs_jax = jnp.array(initial_obs_list)
        assert initial_obs_jax.shape is not None # Basic check that conversion worked
    except Exception as e:
        pytest.fail(f"Failed to convert initial observation to JAX array: {e}")

    legal_actions_list = wrapper.legal_actions()
    if legal_actions_list:
        action = legal_actions_list[0]
        obs_list, rewards_list, done_bool = wrapper.step(action)

        try:
            obs_jax = jnp.array(obs_list)
            assert obs_jax.shape is not None
        except Exception as e:
            pytest.fail(f"Failed to convert observation from step() to JAX array: {e}")

        try:
            rewards_jax = jnp.array(rewards_list)
            assert rewards_jax.shape is not None
        except Exception as e:
            pytest.fail(f"Failed to convert rewards from step() to JAX array: {e}")
        
        # done_bool is just a Python bool, jnp.array(True) is fine.
        try:
            done_jax = jnp.array(done_bool)
            assert done_jax.shape is not None # Should be a scalar array
        except Exception as e:
            pytest.fail(f"Failed to convert done flag to JAX array: {e}")

    current_obs_list = wrapper.current_observation()
    try:
        current_obs_jax = jnp.array(current_obs_list)
        assert current_obs_jax.shape is not None
    except Exception as e:
        pytest.fail(f"Failed to convert current_observation to JAX array: {e}") 