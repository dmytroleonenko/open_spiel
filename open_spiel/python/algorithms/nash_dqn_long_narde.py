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

"""Nash Deep Q Network (Nash-DQN) agent implemented for Long Narde."""

import collections
import os
from typing import Optional

from absl import logging
import numpy as np
import scipy.optimize # Added for saddle point calculation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from open_spiel.python import rl_agent
# TODO: Add imports for network definition, replay buffer, saddle-point solver etc.
# from open_spiel.python import simple_nets # Example
from open_spiel.python.utils.replay_buffer import ReplayBuffer # Imported for use

# Define Transition tuple structure for canonical Nash-DQN
# All state/action components are in the *canonical* view.
# 'action' and 'opponent_action' are *indices* within legal_a/legal_b.
Transition = collections.namedtuple(
    "Transition",
    "state legal_a legal_b action reward next_state next_legal_a next_legal_b terminal")

# TODO: Define necessary data structures like Transition if using a replay buffer internally


# Placeholder for the saddle point solver function
# Updated saddle_point function using scipy.optimize.linprog
def saddle_point(q_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Computes the saddle point (minimax value and strategies) for a Q-matrix.

    Solves the linear program for the row player (maximizer) to find the
    game value and their optimal strategy. Then solves the dual for the column
    player (minimizer).

    Args:
        q_matrix: A 2D numpy array representing the Q-values for joint actions.
                  Rows correspond to player 1's actions, columns to player 2's.
                  Assumes zero-sum game. Illegal actions should have large negative values.
                  Matrix shape: (num_actions_p1, num_actions_p2)

    Returns:
        A tuple (p_star, q_star, value):
          p_star: Optimal mixed strategy for player 1 (row player). Shape (num_actions_p1,).
          q_star: Optimal mixed strategy for player 2 (column player). Shape (num_actions_p2,).
          value: The minimax value of the game matrix.
    """
    num_actions_p1, num_actions_p2 = q_matrix.shape

    # --- Pre-check for fully masked rows/columns --- 
    MASK_THRESHOLD = -1e8 # Value below which an action is considered masked
    valid_rows = np.any(q_matrix > MASK_THRESHOLD, axis=1)
    valid_cols = np.any(q_matrix > MASK_THRESHOLD, axis=0)

    if not np.all(valid_rows): # If any row is fully masked
        logging.warning("Saddle point called with fully masked row(s). Returning degenerate strategy.")
        p_star = np.zeros(num_actions_p1)
        # If there are *any* valid rows, play uniformly over them
        if np.any(valid_rows):
             p_star[valid_rows] = 1.0 / np.sum(valid_rows)
        else: # All rows masked
             p_star.fill(1.0 / num_actions_p1) # Fallback: Uniform (though LP would likely fail)
        # Opponent strategy needs to be uniform over *their* valid actions
        q_star = np.zeros(num_actions_p2)
        if np.any(valid_cols):
            q_star[valid_cols] = 1.0 / np.sum(valid_cols)
        else: # All cols masked too
            q_star.fill(1.0 / num_actions_p2)
        return p_star, q_star, 0.0 # Value is ill-defined, return 0

    if not np.all(valid_cols): # If any column is fully masked
        logging.warning("Saddle point called with fully masked column(s). Returning degenerate strategy.")
        q_star = np.zeros(num_actions_p2)
        q_star[valid_cols] = 1.0 / np.sum(valid_cols) # Play uniform over valid cols
        # Row player strategy uniform over *their* valid actions
        p_star = np.zeros(num_actions_p1)
        p_star[valid_rows] = 1.0 / np.sum(valid_rows) # Should be all valid here based on first check
        return p_star, q_star, 0.0 # Value ill-defined

    # --- Proceed with LP if matrix seems valid --- 

    # Handle trivial case: only one action for each player
    if num_actions_p1 == 1 and num_actions_p2 == 1:
        return np.array([1.0]), np.array([1.0]), q_matrix[0, 0]
    # Handle cases where one player has no actions (should not happen if called correctly)
    if num_actions_p1 == 0 or num_actions_p2 == 0:
        logging.error("Saddle point called with zero actions for a player.")
        p_star = np.ones(num_actions_p1) / num_actions_p1 if num_actions_p1 > 0 else np.array([])
        q_star = np.ones(num_actions_p2) / num_actions_p2 if num_actions_p2 > 0 else np.array([])
        return p_star, q_star, 0.0 # Or raise error?

    # --- Solve for Row Player (Maximizer) ---
    # Objective: Maximize v (or minimize -v)
    # Variables: [p_1, p_2, ..., p_m, v] where m = num_actions_p1
    c = np.zeros(num_actions_p1 + 1)
    c[-1] = -1.0  # Minimize -v

    # Constraints:
    # 1. sum_i (p_i * -Q_ij) + v <= 0  for each j (column)
    #    Rewritten from: sum_i (p_i * Q_ij) >= v
    A_ub = np.hstack((-q_matrix.T, np.ones((num_actions_p2, 1))))
    b_ub = np.zeros(num_actions_p2)

    # 2. sum_i p_i = 1
    A_eq = np.ones((1, num_actions_p1 + 1))
    A_eq[0, -1] = 0.0 # Coefficient for v is 0
    b_eq = np.array([1.0])

    # Bounds: p_i >= 0, v can be anything (-inf, inf)
    bounds = [(0, None)] * num_actions_p1 + [(None, None)]

    try:
        # Using 'highs' solver as it's generally recommended and robust
        result_p1 = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result_p1.success:
            p_star = result_p1.x[:-1]
            value = -result_p1.fun # Remember we minimized -v
            # Clean up potential floating point inaccuracies
            p_star = np.maximum(p_star, 0) # Ensure non-negative
            p_star /= np.sum(p_star) # Ensure sums to 1
        else:
            logging.warning(f"LP for row player failed: {result_p1.message}. Falling back to uniform strategy.")
            p_star = np.ones(num_actions_p1) / num_actions_p1
            # Estimate value from uniform strategy? Difficult. Return 0.
            value = 0.0 # Or NaN?

    except Exception as e:
        logging.error(f"Error during LP solve for row player: {e}. Q-matrix:
{q_matrix}")
        p_star = np.ones(num_actions_p1) / num_actions_p1
        value = 0.0 # Fallback value


    # --- Solve for Column Player (Minimizer) ---
    # Objective: Minimize v (the game value from their perspective)
    # Variables: [q_1, q_2, ..., q_n, v] where n = num_actions_p2
    c_p2 = np.zeros(num_actions_p2 + 1)
    c_p2[-1] = 1.0 # Minimize v

    # Constraints:
    # 1. sum_j (Q_ij * q_j) - v <= 0   for each i (row)
    #    Rewritten from: sum_j (Q_ij * q_j) <= v
    A_ub_p2 = np.hstack((q_matrix, -np.ones((num_actions_p1, 1))))
    b_ub_p2 = np.zeros(num_actions_p1)

    # 2. sum_j q_j = 1
    A_eq_p2 = np.ones((1, num_actions_p2 + 1))
    A_eq_p2[0, -1] = 0.0 # Coefficient for v is 0
    b_eq_p2 = np.array([1.0])

    # Bounds: q_j >= 0, v can be anything
    bounds_p2 = [(0, None)] * num_actions_p2 + [(None, None)]

    try:
        result_p2 = scipy.optimize.linprog(c_p2, A_ub=A_ub_p2, b_ub=b_ub_p2, A_eq=A_eq_p2, b_eq=b_eq_p2, bounds=bounds_p2, method='highs')

        if result_p2.success:
            q_star = result_p2.x[:-1]
            # Value from this LP should match the row player's value in theory
            value_p2 = result_p2.fun
            # Optional: Check if value and value_p2 are close
            if not np.isclose(value, value_p2, atol=1e-4):
                 logging.warning(f"Minimax values from row ({value:.4f}) and col ({value_p2:.4f}) LPs differ slightly.")
                 # Trust the row player's value? Or average? Let's stick with row player's.

            # Clean up strategy
            q_star = np.maximum(q_star, 0)
            q_star /= np.sum(q_star)
        else:
            logging.warning(f"LP for column player failed: {result_p2.message}. Falling back to uniform strategy.")
            q_star = np.ones(num_actions_p2) / num_actions_p2
            # If row player succeeded, we use that value, otherwise it's already 0

    except Exception as e:
        logging.error(f"Error during LP solve for column player: {e}. Q-matrix:
{q_matrix}")
        q_star = np.ones(num_actions_p2) / num_actions_p2
        # Keep value from row player's attempt if possible


    # Final check for NaN in strategies (can happen if LP fails badly or input was weird)
    if np.isnan(p_star).any():
        logging.warning("NaN detected in p_star, falling back to uniform.")
        p_star = np.ones(num_actions_p1) / num_actions_p1
    if np.isnan(q_star).any():
        logging.warning("NaN detected in q_star, falling back to uniform.")
        q_star = np.ones(num_actions_p2) / num_actions_p2


    return p_star, q_star, value

# Function to compute saddle point values for a batch of Q-matrices
def saddle_point_batch(q_batch: np.ndarray) -> np.ndarray:
    """Computes the saddle point value for each Q-matrix in a batch.

    Args:
        q_batch: A 3D numpy array of Q-matrices, shape (batch_size, num_actions_p1, num_actions_p2).
                 Illegal actions should have large negative values.

    Returns:
        A 1D numpy array of saddle point values, shape (batch_size,).
    """
    batch_size = q_batch.shape[0]
    v_list = []
    for i in range(batch_size):
        q_matrix = q_batch[i, :, :]
        try:
            # Reuse the scalar saddle_point function
            _, _, value = saddle_point(q_matrix)
            v_list.append(value)
        except Exception as e:
            # Handle potential errors during saddle point calculation for a single matrix
            logging.error(f"Error computing saddle point for matrix {i} in batch: {e}. Q-matrix:\n{q_matrix}")
            # Append a default value (e.g., 0) or handle as appropriate
            v_list.append(0.0)

    return np.array(v_list, dtype=np.float32)


# ---------- Canonicalization Helpers ----------
BOARD_POINTS = 24
SHIFT = BOARD_POINTS // 2       # 12

def _rot12(idx: int) -> int:
    """Rotate a single board point +12 mod 24 (black -> canonical)."""
    # Ensure index is non-negative before modulo, handles potential negative results if idx+SHIFT is negative
    # Although standard python % handles negative numbers correctly for this case, being explicit adds clarity.
    return (idx + SHIFT) % BOARD_POINTS
# ----------------------------------------------

def canonical_state(obs: np.ndarray, player_id: int) -> np.ndarray:
    """Return a canonical view (current player perspective matches player 0)."""
    if player_id == 0:
        return obs # Already canonical

    state = obs.copy()
    # Board part: first 48 floats = [white_canonical[0:24], black_canonical[0:24]]
    # For player 1 (black), their pieces are originally in black[0:24], opponent's in white[0:24]
    opponent_pieces = state[:BOARD_POINTS]
    player_pieces = state[BOARD_POINTS:2 * BOARD_POINTS]

    # Rotate and swap: opponent becomes canonical white, player becomes canonical black
    canonical_white = np.roll(player_pieces, SHIFT)
    canonical_black = np.roll(opponent_pieces, SHIFT)

    state[:BOARD_POINTS] = canonical_white
    state[BOARD_POINTS:2 * BOARD_POINTS] = canonical_black
    # Rest of the state (scores, dice, etc. indices >= 48) remains unchanged as per review.
    return state

def convert_action_canonical(action: tuple[int, int, int], player_id: int) -> tuple[int, int, int]:
    """Convert action coordinates between player's real view and canonical view.

    For player_id 0, this is an identity operation.
    For player_id 1, this rotates the board points by +12 mod 24.
    Since rot12 is its own inverse, this function handles both to/from canonical.
    """
    if player_id == 0:
        return action
    # Player 1: Apply rotation
    fr, to, d = action
    return (_rot12(fr), _rot12(to), d)

def convert_legal_actions_canonical(legal_actions: list[tuple[int, int, int]],
                                  player_id: int) -> list[tuple[int, int, int]]:
    """Convert a list of legal actions between player's real view and canonical view.

    Applies `convert_action_canonical` to each action in the list.
    For player_id 0, this is an identity operation.
    """
    if player_id == 0:
        return legal_actions # No change needed for player 0

    # Player 1: Apply rotation to each action
    return [convert_action_canonical(a, player_id) for a in legal_actions]

# Utility: scatter_nd for PyTorch (since torch.scatter_nd is not built-in)
def scatter_nd(indices, updates, shape):
    """PyTorch equivalent of TensorFlow's scatter_nd."""
    out = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    if indices.numel() == 0:
        return out
    indices = indices.long()
    if indices.shape[1] == 2:
        out[indices[:, 0], indices[:, 1]] = updates
    else:
        raise NotImplementedError("scatter_nd only supports 2D indices here.")
    return out

# QNet: PyTorch implementation for Nash-DQN (uses custom scatter_nd for masking)
class QNet(nn.Module):
    def __init__(self, input_size, num_actions_p1_max, num_actions_p2_max, embedding_dim=64, hidden_sizes=[256, 256, 128]):
        super(QNet, self).__init__()
        # --- Shared Torso ---
        self.torso_layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.torso_layers.append(nn.Linear(prev_size, size))
            prev_size = size
        self.torso_output_size = prev_size

        # --- Player 1 Head (Embeddings e_i) ---
        self.p1_head_embedding = nn.Linear(self.torso_output_size, embedding_dim * num_actions_p1_max)
        self.num_actions_p1_max = num_actions_p1_max
        self.embedding_dim = embedding_dim

        # --- Player 2 Head (Embeddings f_j) ---
        self.p2_head_embedding = nn.Linear(self.torso_output_size, embedding_dim * num_actions_p2_max)
        self.num_actions_p2_max = num_actions_p2_max

        # --- Normalization Layer (Optional but recommended) ---
        self.input_norm = nn.LayerNorm(self.torso_output_size)

    def forward(self, state_vec, legal_actions_p1, legal_actions_p2):
        """Computes the Q-matrix for a given state and legal action sets."""
        batch_size = state_vec.shape[0]

        # Normalize input state
        normalized_state = self.input_norm(state_vec)

        # --- Pass through shared torso ---
        x = normalized_state
        for layer in self.torso_layers:
            x = F.relu(layer(x))
        torso_output = x # Shape: (batch_size, torso_output_size)

        # --- Compute Player 1 action embeddings ---
        p1_embeddings_flat = self.p1_head_embedding(torso_output)
        p1_embeddings = p1_embeddings_flat.view(batch_size, self.num_actions_p1_max, self.embedding_dim)

        # --- Compute Player 2 action embeddings ---
        p2_embeddings_flat = self.p2_head_embedding(torso_output)
        p2_embeddings = p2_embeddings_flat.view(batch_size, self.num_actions_p2_max, self.embedding_dim)

        # --- Compute Q-matrix using bilinear dot product ---
        q_matrix_full = torch.einsum('bij,bkj->bik', p1_embeddings, p2_embeddings)

        # --- Masking (using custom scatter_nd) ---
        batch_indices_p1 = torch.arange(batch_size, dtype=legal_actions_p1.dtype, device=legal_actions_p1.device)
        batch_indices_p2 = torch.arange(batch_size, dtype=legal_actions_p2.dtype, device=legal_actions_p2.device)
        batch_indices_p1 = batch_indices_p1.view(-1, 1).expand(batch_size, legal_actions_p1.shape[1])
        batch_indices_p2 = batch_indices_p2.view(-1, 1).expand(batch_size, legal_actions_p2.shape[1])

        p1_indices = torch.stack([batch_indices_p1, legal_actions_p1], dim=-1)
        p2_indices = torch.stack([batch_indices_p2, legal_actions_p2], dim=-1)

        p1_indices_flat = p1_indices.view(-1, 2)
        p2_indices_flat = p2_indices.view(-1, 2)
        p1_valid_indices = p1_indices_flat[p1_indices_flat[:, 1] >= 0]
        p2_valid_indices = p2_indices_flat[p2_indices_flat[:, 1] >= 0]

        p1_mask = scatter_nd(
            p1_valid_indices,
            torch.ones(p1_valid_indices.shape[0], dtype=torch.float32, device=state_vec.device),
            [batch_size, self.num_actions_p1_max])
        p2_mask = scatter_nd(
            p2_valid_indices,
            torch.ones(p2_valid_indices.shape[0], dtype=torch.float32, device=state_vec.device),
            [batch_size, self.num_actions_p2_max])

        p1_mask_expanded = p1_mask.view(batch_size, self.num_actions_p1_max, 1)
        p2_mask_expanded = p2_mask.view(batch_size, 1, self.num_actions_p2_max)
        joint_mask = p1_mask_expanded * p2_mask_expanded

        masked_q_matrix = torch.where(joint_mask > 0.5, q_matrix_full, -1e9 * torch.ones_like(q_matrix_full))
        return masked_q_matrix


class NashDQNLongNarde(rl_agent.AbstractAgent):
    """Nash-DQN Agent adapted for Long Narde."""

    def __init__(self,
                 player_id,
                 state_representation_size,
                 num_actions, # Max number of actions possible in the game
                 # Network parameters (match blueprint)
                 hidden_layers_sizes=[256, 256, 128],
                 embedding_dim=64, # Or determine based on network head design
                 # Learning parameters (match blueprint)
                 replay_buffer_capacity=200_000,
                 batch_size=256,
                 replay_buffer_class=None, # TODO: Choose/Import ReplayBuffer
                 learning_rate=3e-4,
                 target_update_tau=0.005, # Tau for soft updates
                 update_target_network_every=1, # Steps between soft updates
                 learn_every=4, # Steps between learning updates
                 discount_factor=0.995,
                 min_buffer_size_to_learn=5_000,
                 # Exploration parameters (match blueprint)
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_duration=200_000, # In terms of agent steps
                 optimizer_str="adam", # As per blueprint
                 loss_str="huber", # As per blueprint
                 max_actions_per_player=20, # As per blueprint
                 device=None,
                 ):
        """Initialize the Nash-DQN agent (PyTorch version)."""
        self._kwargs = locals() # Store args for copying if needed

        self.player_id = player_id
        self._num_actions = num_actions # Max possible actions
        self._state_size = state_representation_size
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._learn_every = learn_every
        self._min_buffer_size_to_learn = min_buffer_size_to_learn
        self._target_update_tau = target_update_tau
        self._update_target_network_every = update_target_network_every

        # Device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Exploration
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration
        self._step_counter = 0

        # Opponent action cache (for symmetric self-play simulation)
        self._legal_actions_opponent_cached = [] # Store legal actions from previous step

        # Instantiate Replay Buffer
        self._replay_buffer = ReplayBuffer(replay_buffer_capacity)

        # Instantiate Networks (Online and Target) in PyTorch
        self._online_net = QNet(self._state_size, max_actions_per_player, max_actions_per_player, embedding_dim, hidden_layers_sizes).to(self.device)
        self._target_net = QNet(self._state_size, max_actions_per_player, max_actions_per_player, embedding_dim, hidden_layers_sizes).to(self.device)
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._target_net.eval()

        # Set up optimizer (only 'adam' supported, loss_str respected for later)
        if optimizer_str == "adam":
            self._optimizer = optim.Adam(self._online_net.parameters(), lr=learning_rate)
        else:
            raise ValueError("Only 'adam' optimizer is supported in this implementation.")
        if loss_str == "huber":
            self.loss_class = F.smooth_l1_loss
        elif loss_str == "mse":
            self.loss_class = F.mse_loss
        else:
            raise ValueError("Only 'huber' and 'mse' loss_str are supported.")

    def step(self, time_step, is_evaluation=False, add_transition_record=True) -> Optional[int]:
        """Returns the action to be taken and updates the agent if needed."""

        # Handle vectorized input if necessary (like DQNLongNarde)
        if isinstance(time_step, list):
            # TODO: Implement batch step logic if using vector envs
            raise NotImplementedError("Batch step not yet implemented for NashDQN.")

        # --- Handle Single TimeStep Input ---
        if (not time_step.last()) and (
            time_step.is_simultaneous_move() or # Should not happen in Long Narde?
            self.player_id == time_step.current_player()):

            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]

            if not legal_actions: # Forced pass
                # Cache empty list for next opponent step? Or does env handle this?
                self._legal_actions_opponent_cached = []
                # No transition needs to be added if no action is taken
                # Update prev_timestep/action?
                self._prev_timestep = time_step
                self._prev_action = None # Indicate pass turn
                return None

            # 1. Get opponent's action set (cached from *previous* player's step)
            opp_actions = self._legal_actions_opponent_cached
            if not opp_actions:
                # Handle case where opponent had no moves (e.g., first move of game for P2)
                # Plan: Assume symmetry for the first move
                 opp_actions = legal_actions[:]
                 # logging.warning("Opponent cached actions empty, using dummy opponent action [0]")
                 # opp_actions = [0] # Dummy action if cache is empty (needs refinement)


            # 2. Prepare state input (normalize if network doesn't do it)
            # Network includes LayerNorm, so pass raw state
            state_vector = np.array(info_state, dtype=np.float32).reshape(1, -1)

            # 3. Compute Q-matrix using online network
            # Need to pass legal actions for masking within the network call
            # Assuming network handles mapping/padding internally based on max_actions
            q_matrix_tf = self._online_net(state_vector,
                                           np.array([legal_actions], dtype=np.int32),
                                           np.array([opp_actions], dtype=np.int32))
            q_matrix = self._session.run(q_matrix_tf)[0] # Get numpy matrix [0] as batch size is 1

            # Slice the q_matrix to only include actual legal actions
            # Ensure slicing bounds are valid
            num_legal_a = len(legal_actions)
            num_opp_a = len(opp_actions)
            q_matrix_sliced = q_matrix[:num_legal_a, :num_opp_a]

            # 4. Compute saddle point
            try:
                p_star, q_star, _ = saddle_point(q_matrix_sliced) # Use sliced matrix
            except Exception as e:
                 logging.error(f"Saddle point computation failed: {e}")
                 logging.error(f"Q-matrix (sliced) shape: {q_matrix_sliced.shape}")
                 logging.error(f"Q-matrix (sliced): 
{q_matrix_sliced}")
                 # Fallback: Choose random action?
                 action = np.random.choice(legal_actions)
                 p_star = np.zeros_like(legal_actions, dtype=float) # Dummy probs
                 p_star[legal_actions.index(action)] = 1.0

            # 5. Sample action using epsilon-greedy from p_star
            epsilon = self._get_epsilon(is_evaluation)
            if np.random.rand() < epsilon:
                action = np.random.choice(legal_actions)
            else:
                # Sample from the computed mixed strategy p_star
                # Ensure p_star has the correct length corresponding to legal_actions
                if len(p_star) == len(legal_actions):
                     action_index = np.random.choice(len(legal_actions), p=p_star)
                     action = legal_actions[action_index]
                else:
                     logging.error(f"p_star length ({len(p_star)}) != legal_actions length ({len(legal_actions)}). Falling back to random.")
                     action = np.random.choice(legal_actions)


            # 6. Cache current legal actions for the *next* opponent step
            self._legal_actions_opponent_cached = legal_actions[:] # Store a copy

            # --- Learning / Replay Buffer Logic ---
            if not is_evaluation:
                self._step_counter += 1

                # Add transition *before* potential learning/target updates
                # Ensure prev_timestep and prev_action are valid before adding
                if self._prev_timestep is not None and self._prev_action is not None and add_transition_record:
                    try:
                        self.add_transition(self._prev_timestep, self._prev_action, time_step)
                    except Exception as e:
                        logging.error(f"Error adding transition: {e}")
                        # Decide how to handle - skip adding? Log and continue?
                # else: # Optional: Log why transition wasn't added
                #     if not add_transition_record:
                #         logging.debug("Skipping add_transition as add_transition_record=False")
                #     else:
                #         logging.debug("Skipping add_transition due to missing prev_timestep/prev_action")

                # Trigger learning step
                # Check replay buffer size *inside* learn() method
                if self._step_counter % self._learn_every == 0:
                     self.learn() # learn() handles buffer size check

                # Trigger target network update (moved to inside learn())
                # if self._step_counter % self._update_target_network_every == 0:
                #      self._session.run(self._update_target_network_op)


            # Store current state and action for the next transition
            self._prev_timestep = time_step
            self._prev_action = action # Store the chosen action

            return action

        else: # Terminal state or not our turn
             # Need to reset opponent cache?
             self._legal_actions_opponent_cached = []
             self._prev_timestep = time_step
             self._prev_action = None
             return None # No action taken


    def _get_epsilon(self, is_evaluation, power=1.0):
        """Returns the evaluation or decayed epsilon value."""
        if is_evaluation:
          return 0.0
        decay_steps = min(self._step_counter, self._epsilon_decay_duration)
        decayed_epsilon = (
            self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
            (1 - decay_steps / self._epsilon_decay_duration)**power)
        return decayed_epsilon

    # Implement add_transition based on plan (symmetric self-play)
    def add_transition(self, prev_time_step, prev_action, time_step):
        """Adds the transition to the replay buffer.

        Assumes the transition components correspond to the `Transition` namedtuple:
        (state, legal_a, legal_b, action, reward, next_state, next_legal_a, next_legal_b, terminal)
        where legal_b are the opponent actions *assumed* when action `a` was chosen.
        """
        # Ensure we have the necessary previous state information
        if prev_time_step is None or prev_action is None:
             logging.debug("Skipping add_transition due to missing prev_time_step or prev_action.")
             return

        state = prev_time_step.observations["info_state"][self.player_id][:]
        legal_a = prev_time_step.observations["legal_actions"][self.player_id][:]
        # Retrieve the cached opponent actions used for the Q(s) calculation leading to prev_action
        legal_b = self._legal_actions_opponent_cached[:] # Use the cached list
        action = prev_action # This is agent's action 'a'

        reward = time_step.rewards[self.player_id]
        terminal = float(time_step.last())
        next_state = time_step.observations["info_state"][self.player_id][:]
        next_legal_a = time_step.observations["legal_actions"][self.player_id][:]
        # next_legal_b assumes the current player becomes the opponent in the next state
        next_legal_b = time_step.observations["legal_actions"][self.player_id][:]

        transition = Transition(
            state=state,
            legal_a=legal_a,
            legal_b=legal_b,
            action=action,
            reward=reward,
            next_state=next_state,
            next_legal_a=next_legal_a,
            next_legal_b=next_legal_b,
            terminal=terminal)

        self._replay_buffer.add(transition)

    def learn(self):
        """Samples from the replay buffer and performs a learning update."""
        # Ensure buffer is sufficiently populated
        if (self._replay_buffer is None or len(self._replay_buffer) < self._batch_size or
            len(self._replay_buffer) < self._min_buffer_size_to_learn):
          logging.debug("Skipping learn step: Replay buffer not ready.")
          return None

        # Sample transitions
        transitions = self._replay_buffer.sample(self._batch_size)

        # Unpack transitions into NumPy arrays
        # NOTE: This assumes the Transition tuple contains opponent_action ('b')
        #       which conflicts with current add_transition and tuple definition.
        #       This needs resolution.
        states = np.array([t.state for t in transitions])
        legal_as = [t.legal_a for t in transitions] # List of lists
        legal_bs = [t.legal_b for t in transitions] # List of lists
        actions = np.array([t.action for t in transitions])
        try:
            # Attempt to unpack opponent_action assuming it exists in the tuple
            # If this fails, it highlights the conflict that needs fixing.
            opponent_actions = np.array([t.opponent_action for t in transitions])
        except AttributeError:
            logging.error("CRITICAL: Transition tuple does not contain 'opponent_action', but it's required by the learn graph. Using dummy value 0. FIX Transition definition and add_transition.")
            opponent_actions = np.zeros_like(actions)

        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        next_legal_as = [t.next_legal_a for t in transitions] # List of lists
        next_legal_bs = [t.next_legal_b for t in transitions] # List of lists
        terminals = np.array([t.terminal for t in transitions])

        # Pad legal action lists
        max_legal_len = self._num_actions # Use agent's max_actions param

        def pad_legal_actions(list_of_lists, max_len):
            padded = np.full((len(list_of_lists), max_len), -1, dtype=np.int32)
            for i, sublist in enumerate(list_of_lists):
                if sublist: # Handle empty lists
                   length = min(len(sublist), max_len)
                   try:
                       padded[i, :length] = sublist[:length]
                   except ValueError as e:
                        logging.error(f"Error padding list {i}: {sublist} with length {len(sublist)} into max_len {max_len}. Error: {e}")
                        # Handle error - maybe skip this sample or use defaults?
                        # For now, leave the row as -1 padding.
                        pass # Keep default padding
            return padded

        legal_actions_p1_padded = pad_legal_actions(legal_as, max_legal_len)
        legal_actions_p2_padded = pad_legal_actions(legal_bs, max_legal_len)
        next_legal_actions_p1_padded = pad_legal_actions(next_legal_as, max_legal_len)
        next_legal_actions_p2_padded = pad_legal_actions(next_legal_bs, max_legal_len)

        # Prepare feed dictionary
        feed_dict = {
            self._info_state_ph: states,
            self._legal_actions_p1_ph: legal_actions_p1_padded,
            self._legal_actions_p2_ph: legal_actions_p2_padded,
            self._action_ph: actions,
            self._opponent_action_ph: opponent_actions, # Critical assumption: 'b' is available
            self._reward_ph: rewards,
            self._is_final_step_ph: terminals,
            self._next_info_state_ph: next_states,
            self._next_legal_actions_p1_ph: next_legal_actions_p1_padded,
            self._next_legal_actions_p2_ph: next_legal_actions_p2_padded,
        }

        # Run the learning step
        try:
            loss, _ = self._session.run([self._loss, self._learn_step], feed_dict=feed_dict)
        except tf.errors.InvalidArgumentError as e:
             logging.error(f"TensorFlow InvalidArgumentError during learn step: {e}")
             logging.error(f"Feed dict sample (first item):")
             for key, val in feed_dict.items():
                  if isinstance(val, np.ndarray):
                       logging.error(f"  {key.name}: shape={val.shape}, first_item={val[0]}")
                  else:
                       logging.error(f"  {key.name}: {val}")
             return None # Indicate failure
        except Exception as e:
             logging.error(f"Unexpected error during learn step: {e}")
             return None # Indicate failure

        # Trigger target network update periodically
        if self._step_counter > 0 and self._step_counter % self._update_target_network_every == 0:
            self._session.run(self._update_target_network_op)

        return loss


    def save(self, checkpoint_dir):
        """Saves the agent's weights to a checkpoint directory."""
        path = os.path.join(checkpoint_dir, 'nash_dqn_agent')
        # Use global_step to keep track of different checkpoints
        self._saver.save(self._session, path, global_step=self._step_counter)
        logging.info(f"Saved checkpoint to {path} at step {self._step_counter}")
        # logging.warning("Save method not implemented.")

    def restore(self, checkpoint_dir):
        """Restores the agent's weights from a checkpoint directory."""
        # Find the latest checkpoint in the directory
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            self._saver.restore(self._session, latest_checkpoint)
            logging.info(f"Restored checkpoint from {latest_checkpoint}")
        else:
            logging.warning(f"No checkpoint found in directory: {checkpoint_dir}. Agent weights remain initialized.")
        # path = os.path.join(checkpoint_dir, 'nash_dqn_agent')
        # self._saver.restore(self._session, path)
        # logging.warning("Restore method not implemented.")

    # Other properties/methods if needed (e.g., get_weights)
    @property
    def step_counter(self):
        return self._step_counter
