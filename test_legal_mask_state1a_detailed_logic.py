import jax.numpy as jnp
from long_narde import LongNarde, State, NUM_POINTS, NUM_CHECKERS, PLAYER_HEAD_POS_RELATIVE, HEAD_POS_P1, NUM_DICE_PLAY_OPTIONS, NO_OP_ACTION_IDX, ACTION_SPACE_SIZE

# Define TRUE/FALSE for direct use if not available in test context
try:
    TRUE
except NameError:
    TRUE = jnp.bool_(True)
    FALSE = jnp.bool_(False)

def test_playthrough_state1a_detailed_logic_legal_actions():
    print("Testing Legal Action Mask for State 1a (Detailed Head Logic)...")

    env = LongNarde()

    # Constructing State 1a:
    # _turn_player = 1 (Player O).
    # current_player = 1.
    # _dice = jnp.array([0, 0]) (original roll was 1,1).
    # _playable_dice = jnp.array([-1, 0, 0, 0]) (one '1' used, three '1's left, value '0').
    # _played_dice_num = 1.
    # _moved_from_head_this_play = TRUE.
    # _is_first_turn_for_player = jnp.array([TRUE, TRUE]) 
    #   (P0 (index 0) hasn't played, P1 (index 1) is on their first turn)
    # Board (P1's perspective):
    #   Player 1 (_board[0]): 14 checkers on head (relative point 23), 1 checker on point 22.
    #   Player 0 (_board[1]): 15 checkers on P0's head (P1's relative point 11).
    
    _turn_player = jnp.int32(1)
    _current_player = jnp.int32(1)
    _dice_values = jnp.array([0, 0], dtype=jnp.int32) 
    _playable_dice_values = jnp.array([-1, 0, 0, 0], dtype=jnp.int32) 
    _played_dice_num_state = jnp.int32(1)
    _moved_from_head_this_play_state = TRUE
    # _is_first_turn_for_player is indexed by absolute player ID.
    # P1 (player index 1) is on their first turn. P0 (player index 0) has not played.
    _is_first_turn_for_player_state = jnp.array([TRUE, TRUE], dtype=jnp.bool_) 

    board_p1_pov_row0 = jnp.zeros(NUM_POINTS, dtype=jnp.int32) \
        .at[PLAYER_HEAD_POS_RELATIVE].set(NUM_CHECKERS - 1) \
        .at[22].set(1) # P1: 14 on head (23), 1 on pt 22
    board_p1_pov_row1 = jnp.zeros(NUM_POINTS, dtype=jnp.int32).at[HEAD_POS_P1].set(NUM_CHECKERS) # P0: 15 on head (seen at P1's pt 11)
    _board_state = jnp.stack([board_p1_pov_row0, board_p1_pov_row1])

    _player_off_count_state = jnp.array([0, 0], dtype=jnp.int32) # [P1_off, P0_off]

    test_state_1a = State(
        current_player=_current_player,
        _turn_player=_turn_player,
        _board=_board_state,
        _dice=_dice_values,
        _playable_dice=_playable_dice_values,
        _player_off_count=_player_off_count_state,
        _played_dice_num=_played_dice_num_state,
        _moved_from_head_this_play=_moved_from_head_this_play_state,
        _is_first_turn_for_player=_is_first_turn_for_player_state,
        observation=jnp.zeros(58, dtype=jnp.float32),
        rewards=jnp.float32([0.0, 0.0]),
        terminated=FALSE,
        truncated=FALSE,
        _step_count=jnp.int32(0),
        legal_action_mask=jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.bool_)
    )

    computed_mask = env._legal_action_mask(test_state_1a)
    true_indices = jnp.where(computed_mask)[0]
    
    print(f"Computed TRUE indices: {true_indices}")

    # Expected TRUE indices for State 1a with detailed head movement logic:
    # - already_moved_from_head is TRUE.
    # - any_board_move_in_base is TRUE (from pt 22 to 21).
    # - So, apply_board_priority_filter is TRUE.
    # - apply_head_priority_filter should be FALSE because already_moved_from_head is TRUE.
    #   (force_head_due_to_first_turn requires ~already_moved_from_head)
    #   (force_head_due_to_later_turn_policy requires ~already_moved_from_head)
    # - Therefore, filtered_mask becomes base_mask & ~is_head_move_type_mask (board moves only).
    # - Board moves from pt 22 to 21 are legal with available dice (indices 1, 2, 3).
    expected_true_indices = sorted([
        22 * NUM_DICE_PLAY_OPTIONS + 1, # 89
        22 * NUM_DICE_PLAY_OPTIONS + 2, # 90
        22 * NUM_DICE_PLAY_OPTIONS + 3  # 91
    ])
    
    print(f"Expected TRUE indices: {expected_true_indices}")

    computed_true_indices_list = sorted(list(true_indices))

    assert computed_true_indices_list == expected_true_indices, \
        f"Mismatch in legal actions. Expected: {expected_true_indices}, Got: {computed_true_indices_list}"
    
    # Check NO_OP action is FALSE
    assert computed_mask[NO_OP_ACTION_IDX] == FALSE, f"Action {NO_OP_ACTION_IDX} (NO_OP) should be FALSE"
    print(f"Assertion for NO_OP action ({NO_OP_ACTION_IDX}) passed.")

    # Verify total number of legal moves
    assert len(computed_true_indices_list) == len(expected_true_indices), \
        f"Expected {len(expected_true_indices)} legal actions, got {len(computed_true_indices_list)}"

    print("Test for State 1a (Detailed Head Logic) completed successfully.")

if __name__ == "__main__":
    test_playthrough_state1a_detailed_logic_legal_actions()
