import jax.numpy as jnp
from long_narde import LongNarde, State, NUM_POINTS, NUM_CHECKERS, PLAYER_HEAD_POS_RELATIVE, HEAD_POS_P1, NUM_DICE_PLAY_OPTIONS, NO_OP_ACTION_IDX, ACTION_SPACE_SIZE

# Define TRUE/FALSE for direct use if not available in test context (e.g. if not running via pgx framework)
try:
    TRUE
except NameError:
    TRUE = jnp.bool_(True)
    FALSE = jnp.bool_(False)

def test_playthrough_state1a_legal_actions():
    print("Testing Legal Action Mask for State 1a...")

    env = LongNarde()

    # Constructing State 1a:
    # _turn_player = 1 (Player O).
    # current_player = 1.
    # _dice = jnp.array([0, 0]) (original roll was 1,1).
    # _playable_dice = jnp.array([-1, 0, 0, 0]) (one '1' used, three '1's left).
    # _played_dice_num = 1.
    # _moved_from_head_this_play = TRUE.
    # _is_first_turn_for_player = jnp.array([TRUE, TRUE]).
    # Board (P1's perspective):
    #   Player 1 (_board[0]): 14 checkers on head (relative point 23), 1 checker on point 22.
    #   Player 0 (_board[1]): 15 checkers on P0's head (P1's relative point 11).
    
    _turn_player = jnp.int32(1)
    _current_player = jnp.int32(1)
    _dice_values = jnp.array([0, 0], dtype=jnp.int32) 
    _playable_dice_values = jnp.array([-1, 0, 0, 0], dtype=jnp.int32) 
    _played_dice_num_state = jnp.int32(1)
    _moved_from_head_this_play_state = TRUE
    _is_first_turn_for_player_state = jnp.array([TRUE, TRUE], dtype=jnp.bool_)

    board_p1_pov_row0 = jnp.zeros(NUM_POINTS, dtype=jnp.int32) \
        .at[PLAYER_HEAD_POS_RELATIVE].set(NUM_CHECKERS - 1) \
        .at[22].set(1)
    board_p1_pov_row1 = jnp.zeros(NUM_POINTS, dtype=jnp.int32).at[HEAD_POS_P1].set(NUM_CHECKERS)
    _board_state = jnp.stack([board_p1_pov_row0, board_p1_pov_row1])

    _player_off_count_state = jnp.array([0, 0], dtype=jnp.int32)

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
    print(f"Indices of all TRUE values in computed_mask: {true_indices}")

    # Expected TRUE indices based on subtask description and current logic
    # (must move from head if possible and checkers on head)
    # Checkers on head: Yes (14)
    # Legal moves from head (to pt 22 with dice 1,2,3): Yes
    # So, only moves from head should be legal.
    # Action (head, die_idx=0) is INVALID because _playable_dice[0] is -1.
    # Action (head, die_idx=1): 24*4+1 = 97. to_pos = 23-1=22. state._board[0,22]=1, state._board[1,22]=0. target_point_valid = (1==0)&(0==0) = F. This should be illegal.
    
    # Let's re-evaluate based on the code for _is_move_legal:
    # target_point_empty_of_own = (player_board[to_pos] == 0)
    # target_point_empty_of_opp = (state._board[1, to_pos] == 0)
    # target_point_valid = target_point_empty_of_own & target_point_empty_of_opp
    
    # For moving from head (23) to point 22 (action 97, 98, 99 with available dice):
    # from_pos = 23 (PLAYER_HEAD_POS_RELATIVE), to_pos = 22.
    # player_board[to_pos=22] is 1. So target_point_empty_of_own is FALSE.
    # target_point_valid is FALSE. So these head moves (97,98,99) should be FALSE.
    
    # For moving from point 22 to point 21 (actions 89, 90, 91 with available dice):
    # from_pos = 22, to_pos = 21.
    # player_board[to_pos=21] is 0. target_point_empty_of_own is TRUE.
    # state._board[1, to_pos=21] is 0. target_point_empty_of_opp is TRUE.
    # target_point_valid is TRUE.
    # Bridge check: moving 1 checker from 22 to 21. Initial board has 14 on head, 1 on 22.
    # temp_board: 14 on head, 1 on 21. No bridge. bridge_ok = TRUE.
    # So actions 89, 90, 91 should be TRUE.
    # Action using die_idx = 0 (e.g. 22*4+0 = 88) will be false because _playable_dice[0] = -1.
    
    # "Must move from head" rule in _legal_action_mask:
    #   - Checkers on head: Yes (14 on state._board[0, PLAYER_HEAD_POS_RELATIVE]).
    #   - Legal moves from head:
    #     - Moving from head (23) to point 22 with a '1' die.
    #     - _is_move_legal checks target_point_empty_of_own = (player_board[to_pos=22] == 0).
    #     - player_board[22] is 1 (P1 has a checker there). So target_point_empty_of_own is FALSE.
    #     - Thus, moves from head to point 22 are ILLEGAL.
    #   - So, num_legal_from_head_moves will be 0.
    #   - The condition (state._board[0, PLAYER_HEAD_POS_RELATIVE] > 0) & (num_legal_from_head_moves > 0) is FALSE.
    #   - Therefore, active_moves_mask becomes all_individually_legal_moves_mask.
    
    # Expected individual legal moves:
    # After _is_move_legal change to allow stacking on own checkers:
    # - Moves from head (23) to point 22 are now individually LEGAL. Actions 97,98,99 use available dice.
    # - Moves from point 22 to point 21 are still individually LEGAL. Actions 89,90,91 use available dice.
    
    # "Must move from head" rule in _legal_action_mask:
    #   - Checkers on head: Yes (14 on state._board[0, PLAYER_HEAD_POS_RELATIVE]).
    #   - Legal moves from head: Yes (actions 97, 98, 99 are now individually legal).
    #   - So, num_legal_from_head_moves will be 3.
    #   - The condition (state._board[0, PLAYER_HEAD_POS_RELATIVE] > 0) & (num_legal_from_head_moves > 0) is TRUE.
    #   - Therefore, active_moves_mask becomes legal_from_head_moves_mask.
    #   - This means only actions 97, 98, 99 should be TRUE.
    
    expected_true_indices = sorted([
        NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 1, # 97: from head, die_idx=1
        NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 2, # 98: from head, die_idx=2
        NUM_POINTS * NUM_DICE_PLAY_OPTIONS + 3, # 99: from head, die_idx=3
    ])
    
    print(f"Expected TRUE indices based on current code logic (with head move priority): {expected_true_indices}")

    # Convert true_indices to a sorted list for comparison
    computed_true_indices_list = sorted(list(true_indices)) # type: ignore

    assert computed_true_indices_list == expected_true_indices, \
        f"Mismatch in legal actions. Expected: {expected_true_indices}, Got: {computed_true_indices_list}"

    print("Test for State 1a completed successfully.")

if __name__ == "__main__":
    test_playthrough_state1a_legal_actions()
