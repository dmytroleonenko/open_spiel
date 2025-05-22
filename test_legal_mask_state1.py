import jax.numpy as jnp
from long_narde import LongNarde, State, NUM_POINTS, NUM_CHECKERS, PLAYER_HEAD_POS_RELATIVE, HEAD_POS_P1, NUM_DICE_PLAY_OPTIONS, NO_OP_ACTION_IDX, ACTION_SPACE_SIZE

def test_playthrough_state1_legal_actions():
    print("Testing Legal Action Mask for OpenSpiel Playthrough State 1 equivalent...")

    env = LongNarde()

    # Constructing State 1: P1 starts, dice (1,1)
    # Player 1's turn (_turn_player = 1, current_player = 1)
    # Dice roll is 1,1 (stored as 0,0 in _dice)
    # Playable dice are four 1s (stored as four 0s in _playable_dice)
    # Board is from P1's perspective:
    #   P1 has 15 checkers on their head (P1's point 23, which is _board[0, PLAYER_HEAD_POS_RELATIVE])
    #   P0 has 15 checkers on P0's head (P0's point 23 from P0 view), which is P1's point 11 (_board[1, HEAD_POS_P1])
    
    _turn_player = jnp.int32(1)
    _current_player = jnp.int32(1)
    _dice_values = jnp.array([0, 0], dtype=jnp.int32) # Roll of 1,1
    _playable_dice_values = jnp.array([0, 0, 0, 0], dtype=jnp.int32) # Four 1s to play

    board_p1_pov_row0 = jnp.zeros(NUM_POINTS, dtype=jnp.int32).at[PLAYER_HEAD_POS_RELATIVE].set(NUM_CHECKERS)
    board_p1_pov_row1 = jnp.zeros(NUM_POINTS, dtype=jnp.int32).at[HEAD_POS_P1].set(NUM_CHECKERS)
    _board_state = jnp.stack([board_p1_pov_row0, board_p1_pov_row1])

    _player_off_count_state = jnp.array([0, 0], dtype=jnp.int32) # [P1_off_count, P0_off_count] from P1's view
    _played_dice_num_state = jnp.int32(0)
    _moved_from_head_this_play_state = jnp.bool_(False)
    _is_first_turn_for_player_state = jnp.array([True, True], dtype=jnp.bool_) # P0 hasn't played, P1 is on first turn

    test_state = State(
        current_player=_current_player,
        _turn_player=_turn_player,
        _board=_board_state,
        _dice=_dice_values,
        _playable_dice=_playable_dice_values,
        _player_off_count=_player_off_count_state,
        _played_dice_num=_played_dice_num_state,
        _moved_from_head_this_play=_moved_from_head_this_play_state,
        _is_first_turn_for_player=_is_first_turn_for_player_state,
        # Default values for other fields not critical for this mask computation
        observation=jnp.zeros(58, dtype=jnp.float32),
        rewards=jnp.float32([0.0, 0.0]),
        terminated=jnp.bool_(False),
        truncated=jnp.bool_(False),
        _step_count=jnp.int32(0),
        legal_action_mask=jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.bool_) # This will be ignored by the function
    )

    # Compute the legal action mask
    computed_mask = env._legal_action_mask(test_state)

    # Verification
    # Actions for moving from head (from_source_idx = NUM_POINTS = 24)
    # Action = from_source_idx * NUM_DICE_PLAY_OPTIONS + die_idx
    # from_head_source_idx = NUM_POINTS (which is 24)
    
    action_head_die0 = NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 0 # 24 * 4 + 0 = 96
    action_head_die1 = NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 1 # 24 * 4 + 1 = 97
    action_head_die2 = NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 2 # 24 * 4 + 2 = 98
    action_head_die3 = NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 3 # 24 * 4 + 3 = 99

    print(f"Computed mask shape: {computed_mask.shape}")
    
    # Check expected TRUE values
    assert computed_mask[action_head_die0] == TRUE, f"Action {action_head_die0} (move from head, die 0) should be TRUE"
    assert computed_mask[action_head_die1] == TRUE, f"Action {action_head_die1} (move from head, die 1) should be TRUE"
    assert computed_mask[action_head_die2] == TRUE, f"Action {action_head_die2} (move from head, die 2) should be TRUE"
    assert computed_mask[action_head_die3] == TRUE, f"Action {action_head_die3} (move from head, die 3) should be TRUE"
    print("Assertions for actions 96-99 passed.")

    # Check NO_OP action
    assert computed_mask[NO_OP_ACTION_IDX] == FALSE, f"Action {NO_OP_ACTION_IDX} (NO_OP) should be FALSE"
    print(f"Assertion for NO_OP action ({NO_OP_ACTION_IDX}) passed.")

    # Check all other actions (0-95) are FALSE
    all_others_false = True
    for i in range(NUM_POINTS * NUM_DICE_PLAY_OPTIONS): # Iterate from 0 to 95
        if computed_mask[i] == TRUE:
            all_others_false = False
            print(f"Error: Action {i} should be FALSE but is TRUE.")
            break
    assert all_others_false, "Not all actions from 0-95 are FALSE."
    print("Assertion for actions 0-95 being FALSE passed.")

    # Print all TRUE indices
    true_indices = jnp.where(computed_mask)[0]
    print(f"Indices of all TRUE values in computed_mask: {true_indices}")
    
    # Final check on the number of true actions
    assert len(true_indices) == 4, f"Expected 4 legal actions, got {len(true_indices)}"
    print("Final count of TRUE actions is correct (4).")
    
    print("Test completed successfully.")

if __name__ == "__main__":
    # Manually import TRUE and FALSE if they are not automatically available
    # This might be needed if running outside a context where long_narde is fully loaded
    try:
        TRUE
    except NameError:
        TRUE = jnp.bool_(True)
        FALSE = jnp.bool_(False)
        
    test_playthrough_state1_legal_actions()
