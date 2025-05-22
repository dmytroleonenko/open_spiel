# TODO: JAX-MuZero Implementation Plan (Based on EfficientZeroV2)

## Overview
Implementation of a MuZero-style agent in JAX/Flax NNX, drawing heavily from the architecture and optimizations found in `EfficientZeroV2`. Stochastic environment handling will be integrated, referencing `Stochastic-muzero` and OpenSpiel's AlphaZero where necessary.

**Key Goals:**
*   Leverage `EfficientZeroV2`'s performant network structures and training pipeline. (Reference: `@EfficientZeroV2` codebase)
*   Implement in JAX/Flax NNX for performance and differentiability.
*   Support distributed training (`pjit`) and data generation (Reverb, S3).
*   Incorporate robust handling of stochastic environments. (Reference: `@Stochastic-muzero`, OpenSpiel AlphaZero)
*   Include tools for performance analysis (JAX Profiler) and batch optimization.

**Guiding Development Principles:**
*   **Test-Driven Development (TDD):** Write Pytest test cases FIRST for every function/module, defining expected behavior (golden tests) before implementation.
*   **100% Test Coverage:** Aim for 100% code coverage, verified using `coverage report run -m pytest` and `coverage report report` (or `coverage html`).
*   **Running Tests:** Execute tests using the command: `source venv/bin/activate && python -m pytest path/to/your_test_file.py` (replace `path/to/your_test_file.py` with the actual test file path).
*   **Reference Implementation:** Closely follow the `EfficientZeroV2` codebase (`@EfficientZeroV2`) for algorithmic structure, network design, and training pipeline. Cite specific reference files in commit messages and comments where applicable.
*   **Stochastic Integration:** The primary deviation from a direct `EfficientZeroV2` port will be the integration of stochastic environment handling, drawing from `@Stochastic-muzero` or OpenSpiel AlphaZero for MCTS and game wrapper adaptations.

---

### Phase 1: Core MuZero JAX Implementation (Local - adapting EfficientZeroV2)

**Goal:** Implement the fundamental MuZero algorithm components in JAX/Flax NNX, based on `EfficientZeroV2`'s structure, running locally. All implementation to follow TDD.
(Test execution command: `source venv/bin/activate && python -m pytest path/to/your_test_file.py`)

[DONE] 1.  **Setup JAX/Flax Environment & Dependencies:**
    *   Install JAX, Flax (NNX), Optax, Orbax, Flashbax, Pytest, Coverage, Hydra, WandB.
    *   Create `requirements.txt` for the JAX project.
    *   Set up basic project structure (e.g., `open_spiel/python/algorithms/muzero_jax/`, `open_spiel/python/algorithms/muzero_jax/models`, `open_spiel/python/algorithms/muzero_jax/mcts`, `open_spiel/python/algorithms/muzero_jax/replay_buffer`, `open_spiel/python/algorithms/muzero_jax/self_play`, `open_spiel/python/algorithms/muzero_jax/training`, `open_spiel/python/algorithms/muzero_jax/utils`, `open_spiel/python/algorithms/muzero_jax/envs`, `open_spiel/python/algorithms/muzero_jax/tests/`).
    *   Configure `pytest.ini` if needed.

[DONE] 2.  **Define MuZero Network Architecture (Flax NNX):**
    *   **TDD:** Write Pytest tests for each network component's expected output shapes and basic functionality before implementation. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/models/test_network.py`)
    *   Create `open_spiel/python/algorithms/muzero_jax/models/network.py`.
    *   (Reference: `@EfficientZeroV2/ez/agents/models/base_model.py`, `@EfficientZeroV2/ez/agents/models/layer.py`)
    *   **Representation Function (h):** [DONE - Implemented in `RepresentationNetwork`]
        *   Input: Observation (Tensor).
        *   Output: Initial Hidden State (Tensor).
        *   Implement using Flax NNX layers (e.g., Conv, BatchNorm, Residual Blocks).
        *   Include optional `DownSample` module similar to `EfficientZeroV2` for image-based inputs.
        *   Ensure JAX-compatible handling of varying observation shapes from OpenSpiel games.
    *   **Dynamics Function (g):** [DONE - Implemented in `DynamicsNetwork`]
        *   Input: Hidden State (Tensor), Action (Tensor - one-hot encoded or embedded).
        *   Output: Next Hidden State (Tensor). (Reward prediction will be separate).
        *   Implement using Flax NNX layers, mirroring `EfficientZeroV2`'s structure.
    *   **Prediction Function (f):** [DONE - Implemented in `PredictionNetwork`]
        *   Input: Hidden State (Tensor).
        *   Output: Policy (Logits over actions), Value (Scalar or Categorical Distribution).
        *   Implement using Flax NNX layers, similar to `EfficientZeroV2`'s `ValuePolicyNetwork`.
    *   **Reward Prediction Head/Network:** [DONE - Implemented in `RewardNetwork`]
        *   Input: Hidden State (Tensor).
        *   Output: Reward (Scalar or Categorical Distribution).
        *   Implement based on `EfficientZeroV2`'s approach (e.g., `SupportNetwork`).
    *   **(Optional) Self-Supervised Projection Heads:** (Based on `EfficientZeroV2`'s `ProjectionNetwork`) [TODO - Defer to Phase 3 or if specifically requested earlier]
        *   Input: Hidden State (Tensor).
        *   Output: Projected representation (Tensor).
    *   **Combined Model (`nnx.Module`):** [DONE - Implemented as `MuZeroNetwork`]
        *   Encapsulate `h`, `g`, `f`, reward head, and optional projection heads.
        *   Provide `init` and `apply` methods. The `apply` method should allow flexible calling of individual components.
    *   Helper layers (`DownSample`, `conv3x3`, `ResidualBlock`, `FCResidualBlock`, `MLP`) are implemented in `layers.py` and tested in `test_layers.py`. [DONE]

*The MCTS implementation is now delegated to the [`mctx`](https://github.com/deepmind/mctx) package's GumbelMuZero routines. We no longer maintain a custom JAX MCTS; instead, we wrap and call `mctx.gumbel_muzero_policy`, benefiting from its fully JIT-compatible, GPU-native implementation and comprehensive tests.*

[DONE] 4.  **Game Wrapper for OpenSpiel (JAX):**
    *   **TDD:** Write Pytest tests for all wrapper methods against a known OpenSpiel game. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/envs/test_game_wrapper.py`)
    *   Create `open_spiel/python/algorithms/muzero_jax/envs/game_wrapper.py`.
    *   Wrap an OpenSpiel game (`pyspiel.Game`).
    *   Provide methods: `reset()`, `step(action)`, `legal_actions()`, `current_observation()`, `num_distinct_actions()`, `is_chance_node()`, `chance_outcomes()`.
    *   Ensure observations/actions are JAX-compatible.

[DONE] 5.  **Replay Buffer (Flashbax):**
    *   **TDD:** Write Pytest tests for Flashbax buffer classes (FlatBuffer, TrajectoryBuffer, PrioritisedFlatBuffer) and Vault-based persistence. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_replay_buffer.py`)
    *   Create `open_spiel/python/algorithms/muzero_jax/replay_buffer.py` wrapping Flashbax buffers.
    *   (Reference: Flashbax FlatBuffer, TrajectoryBuffer, PrioritisedFlatBuffer, Vault for disk persistence: https://instadeepai.github.io/flashbax/)
    *   Use Flashbax APIs (`make_flat_buffer`, `make_trajectory_buffer`, `make_prioritised_flat_buffer`) to manage in-memory buffers.
    *   Implement `add_trajectory` and `sample_batch` by delegating to the underlying Flashbax buffer methods.
    *   Integrate prioritized experience replay using Flashbax's prioritised buffers.
    *   Use Vault for persistence when needed (e.g., `Buffer.vault` backed on disk).
    *   **Tests:** Add tests for buffer functionality and Vault persistence (empty samples, negative batch size, disk-backed loading).

[TODO] 6.  **Training Loop (JAX):**
    *   **TDD:** Write Pytest tests for loss components and the overall training step function, verifying gradient computation and parameter updates on mock data. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_trainer.py`)
    *   Create `open_spiel/python/algorithms/muzero_jax/training/trainer.py`.
    *   (Reference: `@EfficientZeroV2/ez/agents/base.py` (especially `update_weights` method), `@EfficientZeroV2/ez/utils/loss.py`)
    *   **Loss Function (MuZero specific, with `EfficientZeroV2` additions):**
        *   Policy, Value, Reward losses.
        *   (Optional) Consistency/Self-Supervised Loss (from `EfficientZeroV2`).
        *   L2 regularization.
    *   **Training Step Function (`@jax.jit`):**
        *   Input: Model `nnx.State`, Optax `optimizer_state`, batch.
        *   Unroll model, calculate losses, compute gradients, update parameters.
        *   Output: New model `nnx.State`, new `optimizer_state`, metrics.
    *   **Main Training Orchestration:** (Inspired by `@EfficientZeroV2/ez/agents/base.py`'s `train` method`)
        *   Initialize model, optimizer. Loop: Sample batch, train step, log (using WandB), checkpoint.
        *   Manage target network updates (EMA or periodic copy, per `EfficientZeroV2`).

[TODO] 6.1.  **Connect Batch Optimizer Hooks to MuZeroNetwork:**
    *   Implement `init_muzero_model_and_params` to initialize the actual `MuZeroNetwork` (using `nnx.Rngs`).
    *   Implement `forward_and_backward_muzero` with `nnx.value_and_grad` computing the combined MuZero loss.
    *   This task depends on Task 2 (MuZeroNetwork) and Task 6 (loss function definition and training step).

[TODO] 7.  **Self-Play Loop (JAX):**
    *   **TDD:** Write Pytest tests for the actor loop, ensuring correct interaction with MCTS, game wrapper, and trajectory generation. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/self_play/test_actor.py`)
    *   Create `open_spiel/python/algorithms/muzero_jax/self_play/actor.py`.
    *   (Reference: `@EfficientZeroV2/ez/worker/actor_worker.py`, `@EfficientZeroV2/ez/worker/self_play_worker.py`)
    *   **Actor Loop:** Load model, play games with MCTS, store trajectories, compute targets, add to Flashbax.

[TODO] 8.  **Main Orchestration Script (`run_muzero_jax.py`):**
    *   **TDD:** Write integration tests for the local setup (actor, learner, Flashbax buffer communication). (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_run_muzero_jax.py`)
    *   Initialize Flashbax buffers (in-memory and Vault persistence), start actor(s) (local), start training.
    *   Manage configuration using Hydra (inspired by `@EfficientZeroV2/ez/train.py`).
    *   Integrate WandB for experiment tracking.

[TODO] 9.  **Checkpointing (Orbax):**
    *   **TDD:** Write Pytest tests for saving and loading model state and optimizer state. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/utils/test_checkpointing.py`)
    *   Integrate Orbax for saving/loading Flax NNX `State` and Optax `optimizer_state`.

[TODO] 10. **Initial Testing & Debugging & Coverage Check:**
    *   Use simple OpenSpiel game (e.g., CartPole, TicTacToe).
    *   Run `source venv/bin/activate && coverage report run -m pytestopen_spiel/python/algorithms/muzero_jax/tests/` and `coverage report report` to ensure >95% coverage for Phase 1 components before proceeding. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`)

---

### Phase 2: Distributed Training and Data Generation (adapting `EfficientZeroV2`'s distributed setup)

**Note:** Phase 2 tasks are now **deferred** until single-device training (Phase 1) is complete. We will revisit distributed training once local training is fully implemented.

**Goal (Deferred):** Scale using JAX's distributed capabilities after Phase 1 progress.
(Test execution command: `source venv/bin/activate && python -m pytest path/to/your_test_file.py`)

[DEFERRED] 11. **Distributed Replay Buffer (Flashbax Vault):**
    *   **TDD:** Write tests for Flashbax Vault in distributed scenarios. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_distributed_replay_buffer.py`)
    *   Set up Flashbax Vault with disk-backed store accessible by multiple processes.

[DEFERRED] 12. **Distributed Data Generation (Actors - JAX processes):**
    *   **TDD:** Write tests for actor processes writing to Flashbax Vault. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/self_play/test_distributed_actor.py`)
    *   Actors as independent JAX processes poll for new Vault entries, run self-play, and write trajectories to Flashbax Vault.

[DEFERRED] 13. **Distributed Training (Learner - `pjit`):**
    *   **TDD:** Tests for `pjit` sharding, distributed checkpointing, and correct gradient aggregation across devices. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_distributed_trainer.py`)
    *   (Reference: `@EfficientZeroV2/ez/agents/base.py` DDP setup, `EfficientZeroV2/ez/train.py` DDP orchestration).
    *   Modify `open_spiel/python/algorithms/muzero_jax/training/trainer.py`. Use `pjit` for data/model parallelism. Define mesh, sharding for `nnx.State` and data.
    *   Distributed Orbax checkpointing to S3.

[DEFERRED] 14. **JAX Profiler Integration:**
    *   **TDD:** (Difficult to TDD directly, but ensure profiler calls are in place and can be activated).
    *   Add hooks for `jax.profiler`.

[DONE] 15. **Gradient Accumulation & Optimal Batch Sizing Script:**
    *   **TDD:** Write Pytest tests for each function in the script (`find_max_batch`, `estimate_grad_var`, `sweep_accum`) using a mock JAX/NNX model and synthetic data. Tests should cover both `float32` and `bfloat16` data types. (Test execution after `pip install -e .`: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/utils/test_batch_optimizer.py`)
    *   The script is located at `open_spiel/python/algorithms/muzero_jax/utils/batch_optimizer.py`.
    *   Ensure `init_params` initializes Flax NNX model, `forward_and_backward` uses `nnx.value_and_grad`, `flatten_grads` handles `nnx.State`.
    *   Support `float32` and `bfloat16` data types throughout the script for creating batches, performing computations, and finding optimal sizes.
    *   **Note:** `sweep_accum` currently runs its core logic in op-by-op mode to avoid JAX tracer issues; JIT for its internal step was removed. Tests for `main_batch_optimizer_workflow` are currently skipped and need implementation, including mixed precision aspects.
    *   **Status:** Script structure and core logic exist. Placeholder model (`SimpleNNXModel`) is used. Needs to be connected to the actual MuZero NNX model once developed (see [TODO] 6.1). Tests for the script itself need to be written/completed to ensure its own correctness with the mock model.

[DEFERRED] 16. **Resilience and Fault Tolerance (Basic):**
    *   **TDD:** Tests for restarting actors/learner from checkpoints. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_resilience.py`)
    *   Ensure frequent checkpointing. Actors/learner can restart from latest checkpoint.

[DEFERRED] 17. **Phase 2 Coverage Check:**
    *   Run `coverage report run -m pytest` and `coverage report report` to ensure >95% coverage for distributed components and batch optimizer script. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`)

---

### Phase 3: Advanced Features & Refinements (inspired by EfficientZeroV2)

**Goal:** Incorporate advanced techniques from `EfficientZeroV2` and productionize, maintaining TDD and high test coverage.
(Test execution command: `source venv/bin/activate && python -m pytest path/to/your_test_file.py`)

[TODO] 18. **Stochastic Environment Handling - Refined:**
    *   **Integration Point:** Implement a thin wrapper around `mctx.gumbel_muzero_policy` in `open_spiel/python/algorithms/muzero_jax/mcts/mctx_wrapper.py` that:
        - Pads the root `prior_logits` with `-inf` entries for chance outcomes and wraps `root.embedding` in `StochasticRecurrentState`.
        - Builds a `recurrent_fn` via `_make_stochastic_recurrent_fn(decision_recurrent_fn, chance_recurrent_fn, num_actions, num_chance_outcomes)`.
        - Calls `gumbel_muzero_policy` unmodified, passing in the stochastic recurrent function.
    *   **TDD:** Write Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/test_stochastic_mcts.py` that:
        - Instantiate the wrapper with mock `decision_recurrent_fn` and `chance_recurrent_fn` on a simple OpenSpiel chance-node game.
        - Verify correct action selection, visit counts, and embedding alternation for both decision and chance nodes.
        - Ensure deterministic behavior when seeding the RNG and correct handling of invalid actions.
    *   **Location:** All wrapper code lives under `open_spiel/python/algorithms/muzero_jax/mcts/mctx_wrapper.py`; tests under `open_spiel/python/algorithms/muzero_jax/tests/test_stochastic_mcts.py`.

[TODO] 19. **Reanalyze Implementation (EfficientZeroV2 style):**
    *   **TDD:** Tests for reanalyze worker logic, target updates in Flashbax, and interaction with main training loop. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_reanalyze.py`)
    *   (Reference: `@EfficientZeroV2/ez/worker/reanalyze_worker.py`, `ez/agents/base.py` reanalyze intervals).
    *   Adapt `EfficientZeroV2`'s reanalyze mechanism.

[TODO] 20. **Self-Supervised Learning (if adopted from EfficientZeroV2):**
    *   **TDD:** Tests for the self-supervised loss component and its integration into the main loss. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_ssl_loss.py`)
    *   (Reference: `@EfficientZeroV2/ez/agents/models/base_model.py` Projection Networks, relevant loss terms in `update_weights`).
    *   If projection heads were implemented, add corresponding SSL loss to main training loss.

[TODO] 21. **Support for wider range of OpenSpiel games:**
    *   **TDD:** Add test suites for new game types as they are supported. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/games/`)
    *   Test and adapt for image-based games (e.g., Atari if `EfficientZeroV2` features are fully ported).

[TODO] 22. **Hyperparameter Tuning:**
    *   (Reference: `@EfficientZeroV2` Hydra configs in `@EfficientZeroV2/ez/config/`)
    *   Use `EfficientZeroV2`'s configurations (managed with Hydra) as starting point.

[TODO] 23. **Evaluation Pipeline:**
    *   **TDD:** Tests for the evaluation loop, metric calculation, and agent loading. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_evaluation.py`)
    *   (Reference: `@EfficientZeroV2/ez/eval.py`)
    *   Implement evaluation similar to `EfficientZeroV2`.

[TODO] 24. **Code Refinement and Documentation:**
    *   Ensure all new code meets TDD and coverage standards.
    *   Update README and add docstrings.

[TODO] 25. **Final Coverage Check & Polish:**
    *   Aim for 100% test coverage across the entire project. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`)
    *   Final review of code quality, documentation, and examples.

---
### Later / Optional

[TODO] 26. **Advanced Network Architectures:**
    *   (Reference: `EfficientZeroV2` model variants, e.g., LSTM use).

[TODO] 27. **Support for Continuous Actions:**
    *   (Reference: `EfficientZeroV2/ez/agents/models/base_model.py` `is_continuous` flags and logic).

[TODO] 28. **More Sophisticated Distributed Setup:**
    *   (Reference: `EfficientZeroV2` Ray actor/server model for replay, storage).
