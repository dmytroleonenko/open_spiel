# TODO List: Implementing PyTorch Canonical NashDQN for Long Narde (`nash_dqn_long_narde.py`)

This document tracks the remaining steps to complete the implementation of the `NashDQNLongNarde` agent using PyTorch and the canonical view approach.

**Checklist (PyTorch Canonical Nash-DQN):**

1.  **PyTorch Migration - Setup:**
    *   [X] Remove `import tensorflow.compat.v1 as tf`.
    *   [X] Remove `tf.disable_v2_behavior()`.
    *   [X] Add PyTorch imports: `import torch`, `import torch.nn as nn`, `import torch.optim as optim`, `import torch.nn.functional as F`.
    *   [X] Remove TF session management (`self._session`, `session.run`). Agent `__init__` will no longer take `session` as an argument.
    *   [X] Add device handling (e.g., `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`) in `__init__` and ensure tensors/models are moved to the device.
2.  **PyTorch Migration - QNet:**
    *   [X] Rewrite the `QNet` class to inherit from `nn.Module`.
    *   [X] Replace `tf.keras.layers.Dense` with `nn.Linear`.
    *   [X] Replace `tf.keras.layers.LayerNormalization` with `nn.LayerNorm`.
    *   [X] Implement the `forward` method instead of `call`.
    *   [X] Adjust tensor operations (reshape, einsum, scatter_nd equivalent for masking, where) using PyTorch equivalents. `torch.scatter_`, `torch.where`, `torch.einsum`.
    *   [X] Ensure `forward` accepts canonical state tensor and padded canonical legal action tensors, and returns the masked joint Q-matrix tensor `(batch, max_a, max_a)`.
3.  **PyTorch Migration - Agent Skeleton (`NashDQNLongNarde`):**
    *   [X] Update `__init__` signature (remove `session`).
    *   [X] Instantiate `QNet` (PyTorch version) for online and target networks and move them to `self.device`.
    *   [X] Replace TF optimizer (`tf.train.AdamOptimizer`) with PyTorch optimizer (`optim.Adam(self._online_net.parameters(), lr=learning_rate)`).
    *   [X] Rewrite target network update logic (`_create_target_network_update_op`, `_initialize_target_network`) using PyTorch parameter iteration and `target_param.data.copy_(...)` / `target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)`.
    *   [X] Remove TF placeholders.
    *   [X] Remove `tf.variable_scope`.
4.  **Canonicalization - Functions (`nash_dqn_long_narde.py`):**
    *   [X] Implement `canonical_state(obs, player_id)` as described in the previous plan.
    *   [X] Define and implement `map_action_to_canonical(action, player_id, ...)`.
    *   [X] Define and implement `map_action_from_canonical(canonical_action, player_id, ...)`.
    *   [X] Define and implement `map_legal_actions(legal_actions, player_id, to_canonical=True, ...)`.
5.  **Canonicalization - Transition Tuple (`nash_dqn_long_narde.py`):**
    *   [X] Modify the `Transition` namedtuple definition: `Transition = namedtuple("Transition", "state legal_a legal_b action reward next_state next_legal_a next_legal_b terminal")`.
        *   All states/actions/lists are *canonical*.
        *   `action`, are *indices* within `legal_a`, .
        *   Removed `opponent_action` field as it's not needed for the standard learning update.
6.  **Opponent Action Inference (`nash_dqn_long_narde.py`):**
    *   [X] ~~Define and implement `infer_opponent_action_index(prev_canonical_state, current_canonical_state, opponent_canonical_legal_actions) -> opponent_canonical_action_idx`.~~ Removed this function as it's not required for the standard Nash-DQN update in sequential games.
7.  **Agent Logic - Step Method (`NashDQNLongNarde` - PyTorch):**
    *   [ ] Rewrite `step` method entirely.
    *   [ ] Add agent attributes for temporary storage: `self._last_canonical_state`, etc.
    *   [ ] **Delayed Transition Adding:**
        *   If previous state exists, infer opponent action index `b_idx` using `infer_opponent_action_index`.
        *   Canonicalize current `time_step` components (`next_state`, `next_legal_a`, `next_legal_b`).
        *   Construct the *canonical* `Transition` tuple.
        *   Add to `self._replay_buffer`.
    *   [ ] **Action Selection:**
        *   Get raw state/actions.
        *   Canonicalize state (`current_canonical_state`) and legal actions (`current_canonical_legal_a`).
        *   Prepare inputs for PyTorch `_online_net` (canonical state tensor, padded canonical legal actions tensor assuming symmetry). Move to `self.device`.
        *   Call `with torch.no_grad(): q_matrix = self._online_net(...)`.
        *   Convert `q_matrix` tensor to NumPy (`.cpu().numpy()`).
        *   Slice `q_matrix` NumPy array.
        *   Call `saddle_point` on the sliced matrix to get `p_star`.
        *   Sample `canonical_action_idx` from `p_star` (epsilon-greedy).
        *   Map back: `actual_action = map_action_from_canonical(canonical_action_idx, ...)`.
    *   [ ] **Store for Next Transition:** Update `self._last_*` attributes with *canonical* data.
    *   [ ] Handle terminal state reset.
    *   [ ] Return `actual_action`.
8.  **Agent Logic - Learn Method (`NashDQNLongNarde` - PyTorch):**
    *   [ ] Rewrite `learn` method entirely.
    *   [ ] Sample batch of *canonical* `Transition` tuples.
    *   [ ] Unpack fields. Convert NumPy arrays to PyTorch tensors on `self.device`. Pad legal action lists and convert to tensors.
    *   [ ] **Target Calculation:**
        *   `with torch.no_grad(): q_next = self._target_net(...)`.
        *   Convert `q_next` tensor to NumPy.
        *   `v_star_next_np = saddle_point_batch(q_next_np)`.
        *   Convert `v_star_next_np` back to tensor.
        *   Compute target `y`.
    *   [ ] **Prediction Calculation:**
        *   `q_now = self._online_net(...)`.
        *   Extract `q_sa` using `torch.gather` or equivalent based on *indices* `action_idx_tensor` and `opponent_action_idx_tensor`.
    *   [ ] **Loss & Optimization:**
        *   Compute loss (e.g., `F.huber_loss(q_sa, y)`).
        *   `self._optimizer.zero_grad()`.
        *   `loss.backward()`.
        *   `self._optimizer.step()`.
    *   [ ] Trigger target network update periodically (using PyTorch logic).
    *   [ ] Return loss value (`loss.item()`).
9.  **PyTorch Migration - Save/Load:**
    *   [ ] Implement `save(self, checkpoint_dir)` using `torch.save` for `_online_net.state_dict()`, `_optimizer.state_dict()`, etc.
    *   [ ] Implement `restore(self, checkpoint_dir)` using `torch.load` and `load_state_dict()`. Update target network after load.
10. **Testing:**
    *   [ ] Adapt `saddle_point` tests (NumPy/SciPy based).
    *   [ ] Add tests for canonicalization functions.
    *   [ ] Add tests for `infer_opponent_action`.
    *   [ ] Add basic tests for PyTorch `QNet` forward pass and masking.
11. **Final TODO Update:**
    *   [ ] Mark checklist items as complete ([X]) in `TODO_nash_dqn.md`. 