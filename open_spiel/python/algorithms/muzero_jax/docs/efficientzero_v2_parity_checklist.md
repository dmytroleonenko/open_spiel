# EfficientZeroV2 Parity Checklist (JAX)

Purpose: Track gaps between the JAX MuZero implementation and the reference PyTorch EfficientZeroV2 code. Each item is a TODO with a concrete deliverable and a test/validation method. Items are ordered by **likelihood of impacting training quality**.

Legend:
- P0 = highest impact on training quality
- P1 = high impact
- P2 = medium impact
- P3 = lower impact / polish

Reference codebase: `EfficientZeroV2/` (PyTorch)

---

## Consolidated TODO_JAX_MUZERO Mapping (Validated)

This section consolidates `TODO_JAX_MUZERO.md` into this single file. Each item is validated for parity relevance.

- 0.x Verification snapshot / coverage / orchestration / infra: **Not parity tasks** (engineering/coverage only).
- 1 (GAE/TD-Lambda dynamic targets): **Implemented**; no parity action needed.
- 1.1/1.1a/1.1b (GAE vectorization/perf): **Implemented**; no parity action needed.
- 1.2 (GAE delta formula): **Validated**; no parity action needed.
- 2 (Policy reanalysis with mctx): **Implemented**; no parity action needed.
- 3 (Categorical value loss + IQL): **Implemented**; no parity action needed.
- 4 (Categorical reward loss): **Implemented**; no parity action needed.
- 5 (Symlog loss/base handling): **Implemented**; no parity action needed.
- 6 (SSL projection consistency loss): **Implemented**; no parity action needed.
- 7 (Gradient scaling): **Implemented**; no parity action needed.
- 8 (Discrete support transform): **Implemented for DMC/Gym only**; OpenSpiel parity still needs non-DMC/Gym branch (see TODO 11).
- 9 (Continuous action distribution_type): **Out of scope for OpenSpiel parity**.
- 10 (Batch content alignment): **Implemented**; no parity action needed.
- 11 (Half-gradient in unroll): **Implemented**; no parity action needed.
- 12 (Entropy regularization): **Implemented**; no parity action needed.
- 13 (use_IQL/IQL_weight): **Implemented**; no parity action needed.
- 14 (Head output dims per loss type): **Implemented**, but **multi-head v_num** parity still missing (see TODO 7).
- 15 (Config consistency): **Implemented**; no parity action needed.
- 16 (td_steps/auto_td_steps): **Implemented**; no parity action needed.
- 17 (Consistency loss coeff naming): **Implemented**; no parity action needed.
- 18 (symlog IQL error in scalar space): **Implemented**; no parity action needed.
- 19 (value_prefix target logic): **Implemented but still mismatched**; see TODO 1.
- 20 (Temperature schedule utilities): **Implemented**, but actor sampling still deterministic; see TODO 13.
- 21 (top_new_masks/mixed_value_threshold): **Implemented**, but mixed target inputs missing; see TODO 2.
- 22 (DMC/Gym vs Atari support transforms): **Out of scope there**, but OpenSpiel parity still needs non-DMC/Gym branch; see TODO 11.
- 23 (Optimizer choice AdamW/Adam): **Implemented**; no parity action needed.
- 24 (Checkpointing + EMA resume): **Implemented**; no parity action needed.
- 25 (Noisy networks): **Implemented**; no parity action needed.
- 26 (moveaxis for continuous policy loss): **Out of scope for OpenSpiel parity**.
- 27 (Gradient clipping): **Implemented**; no parity action needed.

---

## P0 - Value Prefix / Reward Targets (Training-Critical)

TODO 1 (P0): Align value-prefix target computation with EfficientZeroV2
- Gap: JAX applies `apply_value_prefix_reward_accumulation` to replace targets with LSTM-predicted rewards (self-referential). PyTorch uses cumulative environment rewards as value-prefix targets and uses LSTM only in inference for value-prefix prediction.
- JAX location: `open_spiel/python/algorithms/muzero_jax/training/trainer.py` in `apply_value_prefix_reward_accumulation` and the call site in `_compute_total_loss_static`.
- PyTorch location: `EfficientZeroV2/ez/worker/batch_worker.py` (value_prefix accumulation on env rewards), and `EfficientZeroV2/ez/agents/models/__init__.py` (reward prediction in inference).
- Note: `TODO_JAX_MUZERO.md` item 19 marks this as done, but the current JAX implementation still uses model-predicted rewards as targets, so parity is not achieved.
- Deliverable:
  - Change JAX value-prefix targets to be cumulative reward sums (reset by `lstm_horizon_length`) computed from env rewards in the batch, not from the model.
  - Use LSTM reward network only for predicted reward/value-prefix outputs (forward pass), not as the target generator.
- Test/validation:
  - Unit test: Create a deterministic reward sequence and verify JAX `value_prefix` targets match PyTorch `batch_worker` logic for identical rewards and horizon.
  - Integration test: Compare JAX vs PyTorch value-prefix targets for a fixed trajectory and config (same `lstm_horizon_length`).

TODO 2 (P0): Ensure mixed value targets use real search/SARSA sources
- Gap: JAX `Actor` only returns `value_targets` and lacks `target_search_value`, `target_sarsa_value`, and `sample_indices`/`collected_transitions`, so mixed targets are effectively degenerate.
- JAX location: `open_spiel/python/algorithms/muzero_jax/self_play/actor.py` (trajectory output), `open_spiel/python/algorithms/muzero_jax/training/trainer.py` (mixed target selection).
- PyTorch location: `EfficientZeroV2/ez/worker/batch_worker.py` (search vs TD targets & mixed selection).
- Note: `TODO_JAX_MUZERO.md` item 21 implemented `top_new_masks` logic, but the upstream targets are still missing in JAX data generation.
- Deliverable:
  - Extend JAX trajectory/batch to include search values and TD/SARSA values (and sample indices) so `value_target="mixed"` is meaningful.
- Test/validation:
  - Unit test: Build a batch with explicit `target_search_value` and `target_sarsa_value`, verify JAX selects the correct target based on `top_new_masks` and `mixed_value_threshold`.
  - Integration test: Compare JAX mixed target output vs PyTorch batch_worker for same trajectory and config.

---

## P0 - MCTS Search Parity (Training-Critical)

TODO 3 (P0): Align MCTS implementation with EfficientZeroV2 sequential halving + value min/max normalization
- Gap: JAX uses `mctx.gumbel_muzero_policy` directly and ignores the EfficientZeroV2 sequential-halving logic and `value_minmax_delta` normalization.
- JAX location: `open_spiel/python/algorithms/muzero_jax/mcts/mctx_wrapper.py`
- PyTorch location: `EfficientZeroV2/ez/mcts/py_mcts.py` and `EfficientZeroV2/ez/mcts/cy_mcts.py`
- Deliverable:
  - Either port EfficientZeroV2 sequential-halving logic to JAX or add a compatibility layer to mimic it within `mctx` (including min/max normalization).
- Test/validation:
  - Parity test: For a fixed root prior/value and RNG seed, compare root action distribution between PyTorch MCTS and JAX MCTS over multiple runs; measure KL divergence (target < 1e-3 for deterministic cases).

TODO 4 (P0): Convert categorical/symlog values to scalar before MCTS
- Gap: JAX passes raw value head outputs to MCTS even when value loss type is categorical/symlog.
- JAX location: `open_spiel/python/algorithms/muzero_jax/self_play/actor.py` (root value), `open_spiel/python/algorithms/muzero_jax/mcts/mctx_wrapper.py`.
- PyTorch location: `EfficientZeroV2/ez/agents/models/__init__.py` (value conversion in `initial_inference` / `recurrent_inference`).
- Deliverable:
  - Add conversion logic in JAX inference outputs: support-to-scalar or symexp conversion prior to MCTS usage.
- Test/validation:
  - Unit test: Create categorical value logits, confirm MCTS receives scalar values identical to PyTorch conversion.

---

## P1 - Network Architecture Parity

TODO 5 (P1): Implement full EfficientZeroV2 `DownSample` path
- Gap: JAX `DownSample` is simplified (missing downsample block, resblocks, and pooling stages).
- JAX location: `open_spiel/python/algorithms/muzero_jax/models/network.py` class `DownSample`.
- PyTorch location: `EfficientZeroV2/ez/agents/models/base_model.py` class `DownSample`.
- Deliverable:
  - Port the full downsample stack (conv1 + resblock; conv2 + downsample block + resblock; pooling + resblock; pooling).
- Test/validation:
  - Shape parity test: For identical input size, match feature map sizes and channel counts between PyTorch and JAX.
  - Output parity test (after weight translation): max absolute diff < 1e-4.

TODO 6 (P1): Implement action embedding path in dynamics
- Gap: JAX dynamics uses a single action plane only; no action embedding conv+LN path.
- JAX location: `open_spiel/python/algorithms/muzero_jax/models/network.py` class `DynamicsNetwork`.
- PyTorch location: `EfficientZeroV2/ez/agents/models/base_model.py` class `DynamicsNetwork`.
- Deliverable:
  - Implement optional action embedding (conv1x1 + layer norm) and use it when `action_embedding` is enabled.
- Test/validation:
  - Unit test: For `action_embedding=True`, check that JAX action embedding path matches PyTorch output after weight translation.

TODO 7 (P1): Support multiple value heads (v_num)
- Gap: JAX network outputs single value head even when `v_num > 1`.
- JAX location: `open_spiel/python/algorithms/muzero_jax/models/network.py` class `PredictionNetwork`.
- PyTorch location: `EfficientZeroV2/ez/agents/models/base_model.py` class `ValuePolicyNetwork`.
- Deliverable:
  - Add multiple value heads and propagate `v_num` through config and losses.
- Test/validation:
  - Unit test: Verify output shape `[v_num, batch, ...]` matches PyTorch and that value aggregation matches PyTorch inference logic.

TODO 8 (P1): Add projection head for SSL
- Gap: JAX uses `ProjectionNetwork` only; PyTorch applies `projection_head_model` in the with-gradient branch.
- JAX location: `open_spiel/python/algorithms/muzero_jax/models/network.py` in `initial_inference`/`recurrent_inference`.
- PyTorch location: `EfficientZeroV2/ez/agents/models/__init__.py` method `do_projection`.
- Deliverable:
  - Add `ProjectionHeadNetwork` usage in inference when `use_projection=True`.
- Test/validation:
  - Unit test: Compare projection output shapes and values to PyTorch after weight translation.

---

## P1 - Target Generation & Replay Semantics

TODO 9 (P1): Align replay buffer sampling granularity and priorities
- Gap: JAX prioritizes whole trajectories; PyTorch prioritizes transitions (sampling from transition index list).
- JAX location: `open_spiel/python/algorithms/muzero_jax/replay_buffer/replay_buffer.py`.
- PyTorch location: `EfficientZeroV2/ez/data/replay_buffer.py`.
- Deliverable:
  - Add transition-level sampling and priority updates or justify and document the difference.
- Test/validation:
  - Statistical test: Compare sampling distribution over transition indices for the same priorities.

TODO 10 (P1): Stochastic MCTS chance node handling
- Gap: JAX uses placeholder assumptions (binary outcomes, no state changes at chance nodes).
- JAX location: `open_spiel/python/algorithms/muzero_jax/mcts/mctx_wrapper.py`.
- PyTorch location: EfficientZeroV2 uses env-specific chance/transition logic; see `EfficientZeroV2/ez/mcts/*`.
- Deliverable:
  - Implement real chance outcome integration using OpenSpiel chance outcomes and state transitions.
- Test/validation:
  - Integration test: For a stochastic game (e.g., Leduc Poker), verify that chance node transitions match OpenSpiel dynamics and improve policy stability.

---

## P2 - Loss/Support Transform Parity

TODO 11 (P2): Support transform selection based on environment
- Gap: JAX `scalar_to_support` / `support_to_scalar` always use the DMC/Gym transform; PyTorch uses a separate branch for non-DMC/Gym based on `range/scale`.
- JAX location: `open_spiel/python/algorithms/muzero_jax/training/losses.py`.
- PyTorch location: `EfficientZeroV2/ez/utils/format.py`.
- Note: `TODO_JAX_MUZERO.md` item 22 marks this as out-of-scope for Atari/DMC/Gym, but OpenSpiel parity still requires the non-DMC/Gym branch.
- Deliverable:
  - Add env-aware transform selection and parity with PyTorch for non-DMC/Gym environments (OpenSpiel).
- Test/validation:
  - Unit test: Compare JAX transform with PyTorch `DiscreteSupport` for OpenSpiel configs.

---

## P2 - Configuration Wiring

TODO 12 (P2): Wire `state_norm`, `use_batch_norm`, and `init_zero` into JAX network
- Gap: JAX config exposes these flags but the network ignores them (BN always on; init_zero not wired; no state norm).
- JAX location: `open_spiel/python/algorithms/muzero_jax/models/network.py`, `open_spiel/python/algorithms/muzero_jax/models/layers.py`.
- PyTorch location: `EfficientZeroV2/ez/agents/models/__init__.py` (state_norm) and model building (init_zero).
- Deliverable:
  - Apply `state_norm` after representation/dynamics.
  - Respect `use_batch_norm` and `init_zero` flags in network creation.
- Test/validation:
  - Unit tests: When toggled, verify outputs differ in expected ways and match PyTorch behavior.

---

## P3 - Actor Behavior

TODO 13 (P3): Sample actions stochastically when temperature > 0
- Gap: JAX actor uses argmax even when temperature > 0; PyTorch samples from visit count distribution.
- JAX location: `open_spiel/python/algorithms/muzero_jax/self_play/actor.py` in `_select_action`.
- PyTorch location: `EfficientZeroV2/ez/worker/data_worker.py` and MCTS selection logic.
- Note: `TODO_JAX_MUZERO.md` item 20 added temperature scheduling utilities, but actor sampling is still deterministic.
- Deliverable:
  - Implement actual sampling from softmax(visit counts / temperature).
- Test/validation:
- Unit test: For a fixed policy distribution and temperature, verify sampling frequencies approximate expected probabilities.

---

## Cross-check with TODO_JAX_MUZERO.md (validated)

Items marked DONE in `TODO_JAX_MUZERO.md` that are already implemented for parity and therefore not repeated as TODOs here:
- Dynamic GAE/TD-Lambda target computation and vectorized inference.
- Policy reanalysis with real `mctx` search in trainer.
- Categorical value/reward loss with KL divergence, symlog loss handling, and IQL weighting.
- SSL projection consistency loss, half-gradient on hidden state, entropy regularization, and gradient clipping.
- Support range plumbing in trainer, configuration consistency checks, optimizer selection, noisy networks, and checkpoint/EMA handling.

Items marked DONE in `TODO_JAX_MUZERO.md` but still require parity fixes (tracked above):
- Value-prefix target computation (see TODO 1).
- Temperature-based action sampling in actors (see TODO 13).

Items marked OUT OF SCOPE in `TODO_JAX_MUZERO.md` but still relevant for OpenSpiel parity:
- Environment-specific support transforms for non-DMC/Gym cases (see TODO 11).

# Parity Test Plan (PyTorch vs JAX)

Prerequisite: weight-translation bridge (PyTorch -> JAX parameter mapping).

1) **Module-level forward parity**
- Target modules: representation, dynamics, reward head, value/policy head, projection head.
- Test: feed fixed inputs, compare outputs after weight translation.
- Metrics: max absolute diff < 1e-4; relative error < 1e-3.

2) **Inference parity (initial + recurrent)**
- Test: run `initial_inference` and `recurrent_inference` for same observation/action sequences.
- Metrics: policy logits cosine similarity > 0.999; value/reward diff < 1e-3.

3) **Target generation parity**
- Test: for a stored trajectory, compare:
  - value_prefix targets
  - search vs sarsa value targets
  - mixed target selection masks
- Metrics: exact match for deterministic config; otherwise per-step diff < 1e-5.

4) **MCTS parity**
- Test: for a fixed root (prior logits, value, embedding), run MCTS with same RNG seed.
- Metrics: KL divergence between root action distributions < 1e-3 for deterministic cases.

5) **End-to-end parity smoke test**
- Test: tiny game (e.g., TicTacToe), single episode, same seeds.
- Metrics: identical trajectories (actions, rewards, targets) for deterministic configs.

---

# Weight Translation Checklist (for parity tests)

- Conv: PyTorch (O, I, H, W) -> JAX (H, W, I, O)
- Linear: PyTorch (O, I) -> JAX (I, O)
- BatchNorm: copy gamma/beta/running mean/running var; align eps/momentum
- LSTM: map gate ordering (PyTorch i, f, g, o) to JAX implementation; ensure biases match
