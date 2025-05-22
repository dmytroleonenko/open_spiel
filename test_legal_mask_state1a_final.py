import jax.numpy as jnp
from long_narde import LongNarde, State, NUM_POINTS, NUM_CHECKERS, PLAYER_HEAD_POS_RELATIVE, HEAD_POS_P1, NUM_DICE_PLAY_OPTIONS, NO_OP_ACTION_IDX, ACTION_SPACE_SIZE

# Define TRUE/FALSE for direct use if not available in test context
try:
    TRUE
except NameError:
    TRUE = jnp.bool_(True)
    FALSE = jnp.bool_(False)

def test_playthrough_state1a_final_legal_actions():
    print("Testing Legal Action Mask for State 1a (Final Logic Check)...")

    env = LongNarde()

    # Constructing State 1a as per subtask:
    # _turn_player = 1 (Player O).
    # current_player = 1.
    # _dice = jnp.array([0, 0]) (original roll was 1,1).
    # _playable_dice = jnp.array([-1, 0, 0, 0]) (one '1' used, three '1's left, value '0').
    # _played_dice_num = 1.
    # _moved_from_head_this_play = TRUE.
    # _is_first_turn_for_player = jnp.array([TRUE, FALSE]) 
    #   (P0 (idx 0) has not completed turn, P1 (idx 1) has completed first turn - this means P1 is NOT on first turn)
    # Board (P1's perspective):
    #   Player 1 (_board[0]): 14 checkers on head (relative point 23), 1 checker on point 22.
    #   Player 0 (_board[1]): 15 checkers on P0's head (P1's relative point 11).
    
    _turn_player = jnp.int32(1)
    _current_player = jnp.int32(1)
    _dice_values = jnp.array([0, 0], dtype=jnp.int32) 
    _playable_dice_values = jnp.array([-1, 0, 0, 0], dtype=jnp.int32) 
    _played_dice_num_state = jnp.int32(1)
    _moved_from_head_this_play_state = TRUE
    # As per subtask: "ensure _is_first_turn_for_player is jnp.array([TRUE, FALSE])"
    # This means for player 1 (current player), is_on_first_turn will be FALSE.
    _is_first_turn_for_player_state = jnp.array([TRUE, FALSE], dtype=jnp.bool_) 

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

    # Expected TRUE indices for State 1a with detailed head movement logic from prompt_QN5G79:
    # 1. is_on_first_turn = _is_first_turn_for_player[1] = FALSE.
    # 2. head_is_not_empty = TRUE (14 checkers on head).
    # 3. already_moved_from_head = TRUE.
    # 4. any_head_move_in_base: Moves from head (23) to pt 22 are legal with available dice. TRUE.
    # 5. any_board_move_in_base: Moves from pt 22 to pt 21 are legal with available dice. TRUE.
    #
    # Forcing conditions:
    # - force_head_due_to_first_turn = FALSE (is_on_first_turn is FALSE).
    # - force_head_due_to_later_turn_policy = ~is_on_first_turn & head_is_not_empty & ~already_moved_from_head & any_head_move_in_base & any_board_move_in_base
    #                                       = TRUE & TRUE & FALSE & TRUE & TRUE = FALSE.
    # - apply_head_priority_filter = FALSE | FALSE = FALSE.
    #
    # - apply_board_priority_filter = already_moved_from_head & any_board_move_in_base
    #                                 = TRUE & TRUE = TRUE.
    #
    # Filtering:
    # - Head priority filter not applied. filtered_mask = base_mask.
    # - Board priority filter IS applied (~apply_head_priority_filter & apply_board_priority_filter is TRUE).
    #   filtered_mask = _force_board_only_filter_func(base_mask).
    #   This keeps only board moves from base_mask.
    #   Board moves from pt 22 to 21 (actions 89,90,91 for available dice 1,2,3) are kept.
    #   Board moves from pt 23 (board point, from_src_idx=23) to 22 (actions 93,94,95 for available dice 1,2,3) are kept.
    #   Head moves (from_src_idx=24, actions 97,98,99) are filtered out.
    #   So, filtered_mask = [89,90,91,93,94,95].
    #
    # Reversion:
    #   base_sum = 9 (individually legal: 89,90,91, 93,94,95, 97,98,99).
    #   current_sum = 6.
    #   was_filtered_by_head_priority_and_changed_sum = FALSE.
    #   was_filtered_by_board_priority_and_changed_sum = TRUE & (sum(_force_board_only_filter_func(base_mask)) != base_sum)
    #                                                    = TRUE & (6 != 9) = TRUE.
    #   revert_condition = (6 == 0) & (9 > 0) & (FALSE | TRUE) = FALSE. No reversion.
    #   final_filtered_moves_no_pad = [89,90,91,93,94,95].
    
    # The subtask states: "Expected Output for State 1a (with stricter head logic from prompt_QN5G79):
    # The _legal_action_mask should only have TRUE for actions corresponding to moving the checker from point 22...
    # These are actions 89, 90, 91."
    # This implies that moves from board point 23 (actions 93,94,95) should be illegal under this condition.
    # The current _force_board_only_filter_func does NOT exclude moves from point 23 if from_source_idx=23.
    # To match the subtask's expectation of [89,90,91], the _force_board_only_filter_func would need to be more restrictive.
    # However, the test below will assert against the *current code's actual behavior*.
    
    expected_true_indices_from_code = sorted([
        22 * NUM_DICE_PLAY_OPTIONS + 1, # 89
        22 * NUM_DICE_PLAY_OPTIONS + 2, # 90
        22 * NUM_DICE_PLAY_OPTIONS + 3, # 91
        23 * NUM_DICE_PLAY_OPTIONS + 1, # 93 (board move from pt 23, die_idx=1)
        23 * NUM_DICE_PLAY_OPTIONS + 2, # 94 (board move from pt 23, die_idx=2)
        23 * NUM_DICE_PLAY_OPTIONS + 3, # 95 (board move from pt 23, die_idx=3)
    ])
    
    # This is the expectation from the subtask description text.
    subtask_expected_indices = sorted([89, 90, 91])

    print(f"Expected TRUE indices (based on code logic in long_narde.py): {expected_true_indices_from_code}")
    print(f"Expected TRUE indices (based on subtask text for State 1a): {subtask_expected_indices}")

    computed_true_indices_list = sorted(list(true_indices))

    # Assert against the subtask's specific expectation for this test
    assert computed_true_indices_list == subtask_expected_indices, \
        f"Mismatch in legal actions. Expected based on subtask text: {subtask_expected_indices}, Got from code: {computed_true_indices_list}"
    
    assert computed_mask[NO_OP_ACTION_IDX] == FALSE, f"Action {NO_OP_ACTION_IDX} (NO_OP) should be FALSE"
    print(f"Assertion for NO_OP action ({NO_OP_ACTION_IDX}) passed.")

    assert len(computed_true_indices_list) == len(subtask_expected_indices), \
        f"Expected {len(subtask_expected_indices)} legal actions, got {len(computed_true_indices_list)}"

    print("Test for State 1a (Detailed Head Logic) completed based on subtask's direct expectation.")

if __name__ == "__main__":
    test_playthrough_state1a_final_legal_actions()
