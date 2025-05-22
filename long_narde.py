import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from typing import Optional
# Removed: from functools import partial, as it's not used.

# Constants
TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

NUM_POINTS = 24
NUM_CHECKERS = 15
HEAD_POS_P0 = 23  # Player 0's head from Player 0's perspective (canonical P0 view)
HEAD_POS_P1 = 11  # Player 1's head from Player 0's perspective (canonical P0 view)

# With player-relative board:
PLAYER_HOME_START_IDX = 0
PLAYER_HOME_END_IDX = 5
PLAYER_HEAD_POS_RELATIVE = NUM_POINTS - 1 # 23
BEAR_OFF_POS = -1 

# Action space:
NUM_SRC_POSITIONS = NUM_POINTS + 1
NUM_DICE_PLAY_OPTIONS = 4 
NUM_MOVE_ACTIONS = NUM_SRC_POSITIONS * NUM_DICE_PLAY_OPTIONS
ACTION_SPACE_SIZE = NUM_MOVE_ACTIONS + 1 
NO_OP_ACTION_IDX = ACTION_SPACE_SIZE - 1


PGX_OS_CHANCE_OUTCOMES_RAW = jnp.array([
    (1,2), (1,3), (1,4), (1,5), (1,6), 
    (2,3), (2,4), (2,5), (2,6),     
    (3,4), (3,5), (3,6),           
    (4,5), (4,6),                 
    (5,6),                       
    (1,1), (2,2), (3,3), (4,4), (5,5), (6,6)
], dtype=jnp.int32)
PGX_OS_CHANCE_OUTCOMES = PGX_OS_CHANCE_OUTCOMES_RAW - 1 

_starter_for_chance_outcome = jnp.array(
    [0]*15 + [1]*6, dtype=jnp.int32 
)
STARTER_AND_DICE_OPTIONS = jnp.concatenate(
    [
        _starter_for_chance_outcome[:, None], 
        PGX_OS_CHANCE_OUTCOMES
    ], axis=1
)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(58, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)

    _board: Array = jnp.zeros((2, NUM_POINTS), dtype=jnp.int32)
    _player_off_count: Array = jnp.zeros(2, dtype=jnp.int32)
    _dice: Array = jnp.zeros(2, dtype=jnp.int32)
    _playable_dice: Array = jnp.full(4, -1, dtype=jnp.int32)
    _played_dice_num: Array = jnp.int32(0)
    _turn_player: Array = jnp.int32(0)
    _moved_from_head_this_play: Array = FALSE
    _is_first_turn_for_player: Array = jnp.array([TRUE, TRUE], dtype=jnp.bool_)

    @property
    def env_id(self) -> str:
        return "long_narde"


class LongNarde(core.Env):
    def __init__(self):
        super().__init__()

    def _flip_board_perspective(self, board_to_flip: Array) -> Array:
        new_board = jnp.zeros_like(board_to_flip)
        flipped_row_for_new_player0 = jnp.flip(board_to_flip[1]) 
        flipped_row_for_new_player1 = jnp.flip(board_to_flip[0])
        new_board = new_board.at[0].set(flipped_row_for_new_player0)
        new_board = new_board.at[1].set(flipped_row_for_new_player1)
        return new_board

    def _init(self, key: PRNGKey, debug_outcome_idx: Optional[int] = None) -> State:
        if debug_outcome_idx is not None:
            outcome_idx = jnp.int32(debug_outcome_idx)
        else:
            outcome_idx = jax.random.randint(key, (), 0, STARTER_AND_DICE_OPTIONS.shape[0])

        starter_player = STARTER_AND_DICE_OPTIONS[outcome_idx, 0]
        d1_val_0_5 = STARTER_AND_DICE_OPTIONS[outcome_idx, 1]
        d2_val_0_5 = STARTER_AND_DICE_OPTIONS[outcome_idx, 2]
        
        current_player = starter_player
        turn_player = starter_player
        dice_for_turn = jnp.array([d1_val_0_5, d2_val_0_5], dtype=jnp.int32)
        playable_dice = self._get_playable_dice_from_roll(dice_for_turn)

        initial_board_p0_view = jnp.zeros((2, NUM_POINTS), dtype=jnp.int32)
        initial_board_p0_view = initial_board_p0_view.at[0, HEAD_POS_P0].set(NUM_CHECKERS)
        initial_board_p0_view = initial_board_p0_view.at[1, HEAD_POS_P1].set(NUM_CHECKERS)

        final_board_for_state = jax.lax.cond(
            starter_player == 1,
            lambda: self._flip_board_perspective(initial_board_p0_view),
            lambda: initial_board_p0_view
        )
        
        player_off_count = jnp.zeros(2, dtype=jnp.int32)
        played_dice_num = jnp.int32(0)
        moved_from_head_this_play = FALSE
        is_first_turn_for_player = jnp.array([TRUE, TRUE], dtype=jnp.bool_)

        temp_state_for_mask = State(
            current_player=current_player,
            _turn_player=turn_player,
            _board=final_board_for_state,
            _dice=dice_for_turn,
            _playable_dice=playable_dice,
            _player_off_count=player_off_count,
            _played_dice_num=played_dice_num,
            _moved_from_head_this_play=moved_from_head_this_play,
            observation=jnp.zeros(58, dtype=jnp.float32), 
            rewards=jnp.float32([0.0, 0.0]),
            terminated=FALSE,
            truncated=FALSE,
            _step_count=jnp.int32(0),
            legal_action_mask=jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.bool_),
            _is_first_turn_for_player=is_first_turn_for_player 
        )
        legal_action_mask = self._legal_action_mask(temp_state_for_mask)

        return State(
            current_player=current_player,
            observation=jnp.zeros(58, dtype=jnp.float32), 
            rewards=jnp.float32([0.0, 0.0]),
            terminated=FALSE,
            truncated=FALSE,
            legal_action_mask=legal_action_mask,
            _step_count=jnp.int32(0),
            _board=final_board_for_state,
            _player_off_count=player_off_count,
            _dice=dice_for_turn,
            _playable_dice=playable_dice,
            _played_dice_num=played_dice_num,
            _turn_player=turn_player,
            _moved_from_head_this_play=moved_from_head_this_play,
            _is_first_turn_for_player=is_first_turn_for_player
        )

    # TODO: Implement complex bridge rule as per Long Narde specifications.
    # Current implementation is a placeholder and permits all bridge formations.
    # The correct rule: A 6-point bridge is illegal if no opponent checker
    # is ahead of the entire bridge, relative to the opponent's path.
    def _would_form_illegal_bridge(self, proposed_board_own_checkers: Array, proposed_board_opp_checkers_relative_view: Array) -> Array:
        """Placeholder for bridge rule. Currently permits all bridges."""
        return FALSE # Permits all bridges for now

    def _all_checkers_in_home(self, player_board_row: Array, player_head_pos: Array) -> Array:
        no_checkers_on_head = (player_board_row[player_head_pos] == 0)
        checkers_outside_home = jnp.sum(player_board_row[PLAYER_HOME_END_IDX + 1 : player_head_pos])
        no_checkers_outside_home = (checkers_outside_home == 0)
        return no_checkers_on_head & no_checkers_outside_home

    def _no_checkers_on_higher_points_in_home(self, player_board_row: Array, from_pos_in_home: Array) -> Array:
        def loop_body(i, current_sum):
            return current_sum + jax.lax.cond(
                i > from_pos_in_home,
                lambda: player_board_row[i],
                lambda: jnp.int32(0)
            )
        sum_higher_points = jax.lax.fori_loop(
            PLAYER_HOME_START_IDX,
            PLAYER_HOME_END_IDX + 1,
            loop_body,
            jnp.int32(0)
        )
        return sum_higher_points == 0

    def _is_move_legal(self, state: State, from_pos: Array, to_pos: Array, die_value: Array, is_from_head: Array) -> Array:
        player_board = state._board[0]
        has_checker = jax.lax.cond(
            is_from_head,
            lambda: player_board[PLAYER_HEAD_POS_RELATIVE] > 0,
            lambda: player_board[from_pos] > 0
        )
        target_point_empty_of_opp = (state._board[1, to_pos] == 0) 
        target_point_valid = target_point_empty_of_opp
        
        can_bear_off_phase = self._all_checkers_in_home(player_board, PLAYER_HEAD_POS_RELATIVE)
        exact_bear_off = ((from_pos + 1) == die_value)
        greater_die_bear_off = (die_value > (from_pos + 1)) & \
                               self._no_checkers_on_higher_points_in_home(player_board, from_pos)
        bear_off_rules_met = can_bear_off_phase & (exact_bear_off | greater_die_bear_off)

        is_valid_target_or_bear_off = jax.lax.cond(
            to_pos == BEAR_OFF_POS,
            lambda: bear_off_rules_met,
            lambda: target_point_valid
        )

        # Hypothetical board update for bridge check
        temp_board0 = player_board
        temp_board0 = jax.lax.cond(is_from_head, lambda: temp_board0.at[PLAYER_HEAD_POS_RELATIVE].add(-1), lambda: temp_board0.at[from_pos].add(-1))
        temp_board0 = jax.lax.cond(to_pos != BEAR_OFF_POS, lambda: temp_board0.at[to_pos].add(1), lambda: temp_board0)
        
        # Bridge check (currently a placeholder returning FALSE)
        bridge_check_fails = self._would_form_illegal_bridge(temp_board0, state._board[1])
        bridge_ok = ~bridge_check_fails
        
        return has_checker & is_valid_target_or_bear_off & bridge_ok
        
    def _calculate_to_pos(self, from_pos: Array, die_value: Array) -> Array:
        calculated_dest = from_pos - die_value
        return jax.lax.cond(calculated_dest < PLAYER_HOME_START_IDX, lambda: BEAR_OFF_POS, lambda: calculated_dest)

    def _handle_win(self, state: State, winner_player: Array) -> State:
        reward = jnp.zeros(2, dtype=jnp.float32).at[winner_player].set(1.0).at[1 - winner_player].set(-1.0)
        return state.replace(terminated=TRUE, rewards=reward)

    def _is_turn_over(self, state: State) -> Array:
        return (state._playable_dice.sum() == -NUM_DICE_PLAY_OPTIONS)

    def _roll_dice(self, key: PRNGKey) -> Array:
        return jax.random.randint(key, shape=(2,), minval=0, maxval=6, dtype=jnp.int32)

    def _get_playable_dice_from_roll(self, dice_roll: Array) -> Array:
        playable = jnp.full(NUM_DICE_PLAY_OPTIONS, -1, dtype=jnp.int32)
        return jax.lax.cond(dice_roll[0] == dice_roll[1], lambda: playable.at[:4].set(dice_roll[0]), lambda: playable.at[:2].set(dice_roll))

    def _start_new_turn(self, state: State, key: PRNGKey) -> State:
        next_player_id = 1 - state._turn_player
        dice_key, _ = jax.random.split(key) 
        new_dice = self._roll_dice(dice_key)
        new_playable = self._get_playable_dice_from_roll(new_dice)
        new_is_first_turn = state._is_first_turn_for_player.at[next_player_id].set(FALSE)
        flipped_board = self._flip_board_perspective(state._board)
        new_off_count = jnp.array([state._player_off_count[1], state._player_off_count[0]])
        
        next_s_for_mask = state.replace(
            current_player=next_player_id, _turn_player=next_player_id, _board=flipped_board, 
            _player_off_count=new_off_count, _dice=new_dice, _playable_dice=new_playable,
            _played_dice_num=jnp.int32(0), _moved_from_head_this_play=FALSE, 
            _is_first_turn_for_player=new_is_first_turn,
            rewards=jnp.float32([0.0, 0.0]),
            terminated=FALSE,
            truncated=FALSE
        )
        return next_s_for_mask.replace(legal_action_mask=self._legal_action_mask(next_s_for_mask))

    def _handle_turn_end(self, state: State, key: PRNGKey) -> State:
        return self._start_new_turn(state, key)

    def _apply_valid_move_and_continue(self, state: State, turn_player: Array, from_pos: Array, to_pos: Array, die_idx: Array, is_from_head: Array, key: PRNGKey) -> State:
        board0 = state._board[0]
        off0 = state._player_off_count[0]
        board0_decr = jax.lax.cond(is_from_head, lambda: board0.at[PLAYER_HEAD_POS_RELATIVE].add(-1), lambda: board0.at[from_pos].add(-1))
        
        ops = (board0_decr, off0)
        final_b0, final_off0 = jax.lax.cond(
            to_pos == BEAR_OFF_POS,
            lambda o: (o[0], o[1] + 1),
            lambda o: (o[0].at[to_pos].add(1), o[1]),
            ops
        )
        new_board = state._board.at[0].set(final_b0)
        new_off_count = state._player_off_count.at[0].set(final_off0)
        
        new_playable = state._playable_dice.at[die_idx].set(-1)
        new_played_num = state._played_dice_num + 1
        new_moved_head = state._moved_from_head_this_play | is_from_head
        
        intermediate_s = state.replace(
            _board=new_board, _player_off_count=new_off_count, _playable_dice=new_playable,
            _played_dice_num=new_played_num, _moved_from_head_this_play=new_moved_head
        )
        has_won = (intermediate_s._player_off_count[0] == NUM_CHECKERS)
        
        current_turn_continues = ~self._is_turn_over(intermediate_s)
        
        return jax.lax.cond(
            has_won, lambda: self._handle_win(intermediate_s, turn_player),
            lambda: jax.lax.cond(
                current_turn_continues,
                lambda: intermediate_s.replace(legal_action_mask=self._legal_action_mask(intermediate_s)),
                lambda: self._start_new_turn(intermediate_s, key)
            )
        )

    def _legal_action_mask(self, state: State) -> Array:
        # Step 0: Compute base_mask (all individually valid moves) & identify head moves
        move_props = jnp.zeros((NUM_MOVE_ACTIONS, 2), dtype=jnp.bool_) # Stores [is_legal, is_from_head]

        def check_micro_action(idx, current_props_acc):
            die_idx = idx % NUM_DICE_PLAY_OPTIONS
            from_src_idx = idx // NUM_DICE_PLAY_OPTIONS
            actual_die_idx_val = state._playable_dice[die_idx]
            is_die_avail = (actual_die_idx_val != -1)
            actual_die_val = actual_die_idx_val + 1
            is_head_flag = (from_src_idx == NUM_POINTS) # NUM_POINTS is 24, so from_src_idx=24 is head
            from_pos_val = jax.lax.cond(is_head_flag, lambda: PLAYER_HEAD_POS_RELATIVE, lambda: from_src_idx)
            to_pos_val = self._calculate_to_pos(from_pos_val, actual_die_val)
            is_legal_micro = self._is_move_legal(state, from_pos_val, to_pos_val, actual_die_val, is_head_flag)
            final_legality = is_die_avail & is_legal_micro
            return current_props_acc.at[idx].set(jnp.array([final_legality, is_head_flag]))

        move_properties_computed = jax.lax.fori_loop(0, NUM_MOVE_ACTIONS, check_micro_action, move_props)
        base_mask = move_properties_computed[:, 0]  # Boolean mask of individually legal moves (length NUM_MOVE_ACTIONS)
        is_head_move_type_mask = move_properties_computed[:, 1] # True if action index corresponds to a head move type

        # Step 1: Gather State Flags
        is_on_first_turn = state._is_first_turn_for_player[state._turn_player] # Uses absolute player index
        head_is_not_empty = state._board[0, PLAYER_HEAD_POS_RELATIVE] > 0 # Relative to current player
        already_moved_from_head = state._moved_from_head_this_play # Relative to current player

        # Step 2: Analyze base_mask Content
        any_head_move_in_base = jnp.any(base_mask & is_head_move_type_mask)
        any_board_move_in_base = jnp.any(base_mask & ~is_head_move_type_mask)
        base_sum = jnp.sum(base_mask)

        # Step 3: Determine Forcing Conditions
        # Condition A1: First turn, head has checkers, not yet moved from head, and head moves are possible in base_mask.
        force_head_due_to_first_turn = is_on_first_turn & head_is_not_empty & ~already_moved_from_head & any_head_move_in_base
        # Condition A2: Later turn, head has checkers, not yet moved from head, head moves possible, AND board moves also possible.
        force_head_due_to_later_turn_policy = ~is_on_first_turn & head_is_not_empty & ~already_moved_from_head & any_head_move_in_base & any_board_move_in_base
        apply_head_priority_filter = force_head_due_to_first_turn | force_head_due_to_later_turn_policy

        # Condition B: Must prioritize playing from board (already moved from head, and board moves are possible)
        apply_board_priority_filter = already_moved_from_head & any_board_move_in_base
        
        # Step 4: Apply Filters Sequentially
        # Define filter functions using boolean masking
        def _force_head_only_filter_func(m): return m & is_head_move_type_mask
        def _force_board_only_filter_func(m): return m & ~is_head_move_type_mask

        filtered_mask = base_mask # Start with all individually legal moves
        
        # Apply head priority filter if condition met
        filtered_mask = jax.lax.cond(apply_head_priority_filter,
                                     _force_head_only_filter_func, # Apply the filter
                                     lambda m: m,             # Identity if condition false
                                     filtered_mask)
        
        # Apply board priority filter only if head priority was not applied.
        filtered_mask = jax.lax.cond(~apply_head_priority_filter & apply_board_priority_filter,
                                     _force_board_only_filter_func, # Apply the filter
                                     lambda m: m,              # Identity
                                     filtered_mask)
        
        # Step 5: Reversion Logic
        current_sum = jnp.sum(filtered_mask)
        
        # was_filtered_by_head_priority: True if head priority was supposed to apply AND it would have changed the base_mask's sum
        hypothetical_after_head_only = _force_head_only_filter_func(base_mask)
        was_filtered_by_head_priority_and_changed_sum = apply_head_priority_filter & (jnp.sum(hypothetical_after_head_only) != base_sum)
        
        # was_filtered_by_board_priority: True if board priority was supposed to apply (and head didn't) 
        # AND it would have changed the base_mask's sum.
        hypothetical_after_board_only = _force_board_only_filter_func(base_mask)
        was_filtered_by_board_priority_and_changed_sum = \
            (~apply_head_priority_filter & apply_board_priority_filter) & \
            (jnp.sum(hypothetical_after_board_only) != base_sum)

        revert_condition = (current_sum == 0) & (base_sum > 0) & \
                           (was_filtered_by_head_priority_and_changed_sum | was_filtered_by_board_priority_and_changed_sum)
        
        final_filtered_moves_no_pad = jax.lax.cond(revert_condition,
                                                   lambda: base_mask, # Revert to the original base_mask
                                                   lambda: filtered_mask)

        # Step 6: Pad for NO_OP and Final NO_OP Logic
        # final_filtered_moves_no_pad is of length NUM_MOVE_ACTIONS
        final_filtered_moves_padded = jnp.pad(final_filtered_moves_no_pad, (0, 1), constant_values=FALSE) # Pad to ACTION_SPACE_SIZE
        
        no_moves_possible_after_reversion = (jnp.sum(final_filtered_moves_padded) == 0)
        
        mask_with_noop = jax.lax.cond(
            no_moves_possible_after_reversion,
            lambda: final_filtered_moves_padded.at[NO_OP_ACTION_IDX].set(TRUE),
            lambda: final_filtered_moves_padded
        )
        return mask_with_noop

    def _step(self, state: State, action: Array, key: PRNGKey) -> State:
        is_no_op = (action == NO_OP_ACTION_IDX)
        
        # If NO_OP is chosen, it must be because no other moves were legal.
        # The _legal_action_mask function ensures NO_OP is only true if no other moves are possible.
        # So, if is_no_op is true, we directly handle turn end.
        
        # For non-NO_OP actions:
        die_idx = action % NUM_DICE_PLAY_OPTIONS
        src_idx = action // NUM_DICE_PLAY_OPTIONS
        turn_p = state._turn_player
        die_val_idx = state._playable_dice[die_idx] 
        actual_die_val = die_val_idx + 1
        is_head = (src_idx == NUM_POINTS)
        from_p = jax.lax.cond(is_head, lambda: PLAYER_HEAD_POS_RELATIVE, lambda: src_idx)
        to_p = self._calculate_to_pos(from_p, actual_die_val)
        
        # Legality of the specific chosen action should already be true if it's not NO_OP,
        # because the agent must choose from the legal_action_mask.
        # We can include safeguards, but fundamentally PGX relies on agent choosing a legal action.
        # For robustness, one might re-check state.legal_action_mask[action] here.
        # If an illegal action is passed, current PGX practice is often undefined behavior or error.
        # Assuming action is legal if not NO_OP (as it was chosen from mask):
        
        return jax.lax.cond(
            is_no_op, 
            lambda: self._handle_turn_end(state, key),
            # If not NO_OP, then it's a move action.
            # We assume the caller (agent) picked a legal move action from the mask.
            lambda: self._apply_valid_move_and_continue(state, turn_p, from_p, to_p, die_idx, is_head, key)
        )

    def _observe(self, state: State, player_id: Array) -> Array:
        board_for_obs = jax.lax.cond(state._turn_player == player_id, lambda: state._board, lambda: self._flip_board_perspective(state._board))
        off_count_for_obs = jax.lax.cond(state._turn_player == player_id, lambda: state._player_off_count, lambda: jnp.array([state._player_off_count[1], state._player_off_count[0]]))
        
        canonical_board = jax.lax.cond(player_id == 0, lambda: board_for_obs, lambda: self._flip_board_perspective(board_for_obs))
        canonical_off_count = jax.lax.cond(player_id == 0, lambda: off_count_for_obs, lambda: jnp.array([off_count_for_obs[1], off_count_for_obs[0]]))

        b0_ch = canonical_board[0].astype(jnp.float32)
        b1_ch = canonical_board[1].astype(jnp.float32)
        p0_off = canonical_off_count[0].astype(jnp.float32).reshape(1)
        p1_off = canonical_off_count[1].astype(jnp.float32).reshape(1)

        playable_dice_obs = jax.lax.cond(
            player_id == state._turn_player,
            lambda: jnp.clip((state._playable_dice + 1), 0, 6).astype(jnp.float32),
            lambda: jnp.zeros(NUM_DICE_PLAY_OPTIONS, dtype=jnp.float32)
        )
        turn_p_one_hot = jax.nn.one_hot(state._turn_player, 2).astype(jnp.float32)
        obs_p_one_hot = jax.nn.one_hot(player_id, 2).astype(jnp.float32)

        return jnp.concatenate([b0_ch, b1_ch, p0_off, p1_off, playable_dice_obs, turn_p_one_hot, obs_p_one_hot])

    @property
    def id(self) -> str: return "long_narde"
    @property
    def version(self) -> str: return "v0"
    @property
    def num_players(self) -> int: return 2
    @property
    def _illegal_action_penalty(self) -> float: return -1.0
