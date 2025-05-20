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
*   **100% Test Coverage:** Aim for 100% code coverage, verified using `coverage run -m pytest` and `coverage report` (or `coverage html`).
*   **Running Tests:** Execute tests using the command: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest path/to/your_test_file.py` (replace `path/to/your_test_file.py` with the actual test file path).
*   **Reference Implementation:** Closely follow the `EfficientZeroV2` codebase (`@EfficientZeroV2`) for algorithmic structure, network design, and training pipeline. Cite specific reference files in commit messages and comments where applicable.
*   **Stochastic Integration:** The primary deviation from a direct `EfficientZeroV2` port will be the integration of stochastic environment handling, drawing from `@Stochastic-muzero` or OpenSpiel AlphaZero for MCTS and game wrapper adaptations.

---

### Phase 1: Core MuZero JAX Implementation (Local - adapting EfficientZeroV2)

**Goal:** Implement the fundamental MuZero algorithm components in JAX/Flax NNX, based on `EfficientZeroV2`'s structure, running locally. All implementation to follow TDD.
(Test execution command: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest path/to/your_test_file.py`)

[TODO] 1.  **Setup JAX/Flax Environment & Dependencies:**
    *   Install JAX, Flax (NNX), Optax, Orbax, Reverb, Pytest, Coverage, Hydra, WandB.
    *   Create `requirements.txt` for the JAX project.
    *   Set up basic project structure (e.g., `muzero_jax/`, `muzero_jax/models`, `muzero_jax/mcts`, `muzero_jax/replay_buffer`, `muzero_jax/self_play`, `muzero_jax/training`, `muzero_jax/utils`, `muzero_jax/envs`, `tests/`).
    *   Configure `pytest.ini` if needed.

[TODO] 2.  **Define MuZero Network Architecture (Flax NNX):**
    *   **TDD:** Write Pytest tests for each network component's expected output shapes and basic functionality before implementation. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/models/test_network.py`)
    *   Create `muzero_jax/models/network.py`.
    *   (Reference: `@EfficientZeroV2/ez/agents/models/base_model.py`, `@EfficientZeroV2/ez/agents/models/layer.py`)
    *   **Representation Function (h):**
        *   Input: Observation (Tensor).
        *   Output: Initial Hidden State (Tensor).
        *   Implement using Flax NNX layers (e.g., Conv, BatchNorm, Residual Blocks).
        *   Include optional `DownSample` module similar to `EfficientZeroV2` for image-based inputs.
        *   Ensure JAX-compatible handling of varying observation shapes from OpenSpiel games.
    *   **Dynamics Function (g):**
        *   Input: Hidden State (Tensor), Action (Tensor - one-hot encoded or embedded).
        *   Output: Next Hidden State (Tensor). (Reward prediction will be separate).
        *   Implement using Flax NNX layers, mirroring `EfficientZeroV2`'s structure.
    *   **Prediction Function (f):**
        *   Input: Hidden State (Tensor).
        *   Output: Policy (Logits over actions), Value (Scalar or Categorical Distribution).
        *   Implement using Flax NNX layers, similar to `EfficientZeroV2`'s `ValuePolicyNetwork`.
    *   **Reward Prediction Head/Network:**
        *   Input: Hidden State (Tensor).
        *   Output: Reward (Scalar or Categorical Distribution).
        *   Implement based on `EfficientZeroV2`'s approach (e.g., `SupportNetwork`).
    *   **(Optional) Self-Supervised Projection Heads:** (Based on `EfficientZeroV2`'s `ProjectionNetwork`)
        *   Input: Hidden State (Tensor).
        *   Output: Projected representation (Tensor).
    *   **Combined Model (`nnx.Module`):**
        *   Encapsulate `h`, `g`, `f`, reward head, and optional projection heads.
        *   Provide `init` and `apply` methods. The `apply` method should allow flexible calling of individual components.

[TODO] 3.  **Implement MCTS (JAX):**
    *   **TDD:** Write Pytest tests for node structure, UCB calculation, tree traversal, expansion, and backup logic before implementation. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_mcts.py`)
    *   Create `muzero_jax/mcts.py`.
    *   (Reference: `@EfficientZeroV2/ez/mcts/mcts.py` (and related files in that dir), with stochastic adaptations from `@Stochastic-muzero/monte_carlo_tree_search.py` or OpenSpiel AlphaZero).
    *   **Node Structure:** JAX-compatible.
    *   **UCB Calculation:** Use `EfficientZeroV2`'s UCB formula and MinMax stats normalization.
    *   **Search Loop (`run_mcts`):**
        *   Selection, Expansion & Simulation (using model components `h, g, f`, reward_head), Backup.
        *   Integrate Dirichlet noise at the root.
        *   **Stochastic Environment Handling:** (Trivial integration, primary logic from reference)
            *   If game indicates a chance node, MCTS samples an outcome (OpenSpiel AlphaZero style) or represents chance nodes explicitly.
        *   Ensure MCTS can be JIT-compiled (`@jax.jit`).

[TODO] 4.  **Game Wrapper for OpenSpiel (JAX):**
    *   **TDD:** Write Pytest tests for all wrapper methods against a known OpenSpiel game. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/envs/test_game_wrapper.py`)
    *   Create `muzero_jax/envs/game_wrapper.py`.
    *   Wrap an OpenSpiel game (`pyspiel.Game`).
    *   Provide methods: `reset()`, `step(action)`, `legal_actions()`, `current_observation()`, `num_distinct_actions()`, `is_chance_node()`, `chance_outcomes()`.
    *   Ensure observations/actions are JAX-compatible.

[TODO] 5.  **Replay Buffer (Reverb):**
    *   **TDD:** Write Pytest tests for adding trajectories and sampling batches with correct structure and data types. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_replay_buffer.py`)
    *   Create `muzero_jax/replay_buffer.py`.
    *   (Reference: `@EfficientZeroV2/ez/data/replay_buffer.py`, `@EfficientZeroV2/ez/data/trajectory.py`)
    *   Use `dm-reverb`. Define Reverb table schema based on `EfficientZeroV2`.
    *   Implement `add_trajectory` and `sample_batch`.
    *   Integrate prioritized experience replay.

[TODO] 6.  **Training Loop (JAX):**
    *   **TDD:** Write Pytest tests for loss components and the overall training step function, verifying gradient computation and parameter updates on mock data. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/training/test_trainer.py`)
    *   Create `muzero_jax/training/trainer.py`.
    *   (Reference: `@EfficientZeroV2/ez/agents/base.py` (especially `update_weights` method), `@EfficientZeroV2/ez/utils/loss.py`)
    *   **Loss Function (MuZero specific, with `EfficientZeroV2` additions):**
        *   Policy, Value, Reward losses.
        *   (Optional) Consistency/Self-Supervised Loss (from `EfficientZeroV2`).
        *   L2 regularization.
    *   **Training Step Function (`@jax.jit`):**
        *   Input: Model `nnx.State`, Optax `optimizer_state`, batch.
        *   Unroll model, calculate losses, compute gradients, update parameters.
        *   Output: New model `nnx.State`, new `optimizer_state`, metrics.
    *   **Main Training Orchestration:** (Inspired by `@EfficientZeroV2/ez/agents/base.py`'s `train` method)
        *   Initialize model, optimizer. Loop: Sample batch, train step, log (using WandB), checkpoint.
        *   Manage target network updates (EMA or periodic copy, per `EfficientZeroV2`).

[TODO] 7.  **Self-Play Loop (JAX):**
    *   **TDD:** Write Pytest tests for the actor loop, ensuring correct interaction with MCTS, game wrapper, and trajectory generation. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/self_play/test_actor.py`)
    *   Create `muzero_jax/self_play/actor.py`.
    *   (Reference: `@EfficientZeroV2/ez/worker/actor_worker.py`, `@EfficientZeroV2/ez/worker/self_play_worker.py`)
    *   **Actor Loop:** Load model, play games with MCTS, store trajectories, compute targets, add to Reverb.

[TODO] 8.  **Main Orchestration Script (`run_muzero_jax.py`):**
    *   **TDD:** Write integration tests for the local setup (actor, learner, reverb communication). (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_run_muzero_jax.py`)
    *   Initialize Reverb (local), start actor(s) (local), start training.
    *   Manage configuration using Hydra (inspired by `@EfficientZeroV2/ez/train.py`).
    *   Integrate WandB for experiment tracking.

[TODO] 9.  **Checkpointing (Orbax):**
    *   **TDD:** Write Pytest tests for saving and loading model state and optimizer state. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/utils/test_checkpointing.py`)
    *   Integrate Orbax for saving/loading Flax NNX `State` and Optax `optimizer_state`.

[TODO] 10. **Initial Testing & Debugging & Coverage Check:**
    *   Use simple OpenSpiel game (e.g., CartPole, TicTacToe).
    *   Run `coverage run -m pytest` and `coverage report` to ensure >95% coverage for Phase 1 components before proceeding. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/`)

---

### Phase 2: Distributed Training and Data Generation (adapting `EfficientZeroV2`'s distributed setup)

**Goal:** Scale using JAX's distributed capabilities, following TDD and aiming for 100% coverage. Reference `@EfficientZeroV2/ez/train.py` for overall distributed orchestration.
(Test execution command: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest path/to/your_test_file.py`)

[TODO] 11. **Distributed Replay Buffer (Reverb Server):**
    *   **TDD:** Tests for client-server interaction with Reverb. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_distributed_replay_buffer.py`)
    *   Set up Reverb server accessible by multiple processes.

[TODO] 12. **Distributed Data Generation (Actors - JAX processes):**
    *   **TDD:** Tests for actor process initialization, model loading from shared storage, and writing to distributed Reverb. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/self_play/test_distributed_actor.py`)
    *   (Reference: `@EfficientZeroV2/ez/worker/actor_worker.py` and related files for actor logic).
    *   Actors as independent JAX processes, poll S3/shared storage for Orbax checkpoints, load model, run self-play, write to Reverb.

[TODO] 13. **Distributed Training (Learner - `pjit`):**
    *   **TDD:** Tests for `pjit` sharding, distributed checkpointing, and correct gradient aggregation across devices. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/training/test_distributed_trainer.py`)
    *   (Reference: `@EfficientZeroV2/ez/agents/base.py` DDP setup, `EfficientZeroV2/ez/train.py` DDP orchestration).
    *   Modify `muzero_jax/training/trainer.py`. Use `pjit` for data/model parallelism. Define mesh, sharding for `nnx.State` and data.
    *   Distributed Orbax checkpointing to S3.

[TODO] 14. **JAX Profiler Integration:**
    *   **TDD:** (Difficult to TDD directly, but ensure profiler calls are in place and can be activated).
    *   Add hooks for `jax.profiler`.

[PARTIALLY DONE] 15. **Gradient Accumulation & Optimal Batch Sizing Script:**
    *   **TDD:** Write Pytest tests for each function in the script (`find_max_batch`, `estimate_grad_var`, `sweep_accum`) using a mock JAX/NNX model and synthetic data. Tests should cover both `float32` and `bfloat16` data types. (Test execution after `pip install -e .`: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/utils/test_batch_optimizer.py`)
    *   The script is located at `open_spiel/python/algorithms/muzero_jax/utils/batch_optimizer.py`.
    *   Ensure `init_params` initializes Flax NNX model, `forward_and_backward` uses `nnx.value_and_grad`, `flatten_grads` handles `nnx.State`.
    *   Support `float32` and `bfloat16` data types throughout the script for creating batches, performing computations, and finding optimal sizes.
    *   **Note:** `sweep_accum` currently runs its core logic in op-by-op mode to avoid JAX tracer issues; JIT for its internal step was removed. Tests for `main_batch_optimizer_workflow` are currently skipped and need implementation, including mixed precision aspects.

[TODO] 16. **Resilience and Fault Tolerance (Basic):**
    *   **TDD:** Tests for restarting actors/learner from checkpoints. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_resilience.py`)
    *   Ensure frequent checkpointing. Actors/learner can restart from latest checkpoint.

[TODO] 17. **Phase 2 Coverage Check:**
    *   Run `coverage run -m pytest` and `coverage report` to ensure >95% coverage for distributed components and batch optimizer script. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/`)

---

### Phase 3: Advanced Features & Refinements (inspired by EfficientZeroV2)

**Goal:** Incorporate advanced techniques from `EfficientZeroV2` and productionize, maintaining TDD and high test coverage.
(Test execution command: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest path/to/your_test_file.py`)

[TODO] 18. **Stochastic Environment Handling - Refined:**
    *   **TDD:** Specific tests for MCTS behavior with OpenSpiel games that have explicit chance nodes. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_stochastic_mcts.py`)
    *   Thoroughly test MCTS interaction. Ensure consistency if dynamics model learns implicit stochasticity.

[TODO] 19. **Reanalyze Implementation (EfficientZeroV2 style):**
    *   **TDD:** Tests for reanalyze worker logic, target updates in Reverb, and interaction with main training loop. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_reanalyze.py`)
    *   (Reference: `@EfficientZeroV2/ez/worker/reanalyze_worker.py`, `ez/agents/base.py` reanalyze intervals).
    *   Adapt `EfficientZeroV2`'s reanalyze mechanism.

[TODO] 20. **Self-Supervised Learning (if adopted from EfficientZeroV2):**
    *   **TDD:** Tests for the self-supervised loss component and its integration into the main loss. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/training/test_ssl_loss.py`)
    *   (Reference: `@EfficientZeroV2/ez/agents/models/base_model.py` Projection Networks, relevant loss terms in `update_weights`).
    *   If projection heads were implemented, add corresponding SSL loss to main training loss.

[TODO] 21. **Support for wider range of OpenSpiel games:**
    *   **TDD:** Add test suites for new game types as they are supported. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/games/`)
    *   Test and adapt for image-based games (e.g., Atari if `EfficientZeroV2` features are fully ported).

[TODO] 22. **Hyperparameter Tuning:**
    *   (Reference: `@EfficientZeroV2` Hydra configs in `@EfficientZeroV2/ez/config/`)
    *   Use `EfficientZeroV2`'s configurations (managed with Hydra) as starting point.

[TODO] 23. **Evaluation Pipeline:**
    *   **TDD:** Tests for the evaluation loop, metric calculation, and agent loading. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/test_evaluation.py`)
    *   (Reference: `@EfficientZeroV2/ez/eval.py`)
    *   Implement evaluation similar to `EfficientZeroV2`.

[TODO] 24. **Code Refinement and Documentation:**
    *   Ensure all new code meets TDD and coverage standards.
    *   Update README and add docstrings.

[TODO] 25. **Final Coverage Check & Polish:**
    *   Aim for 100% test coverage across the entire project. (Test execution: `source venv/bin/activate && PYTHONPATH=open_spiel/python/algorithms/ python -m pytest muzero_jax/tests/`)
    *   Final review of code quality, documentation, and examples.

---
### Later / Optional

[TODO] 26. **Advanced Network Architectures:**
    *   (Reference: `EfficientZeroV2` model variants, e.g., LSTM use).

[TODO] 27. **Support for Continuous Actions:**
    *   (Reference: `EfficientZeroV2/ez/agents/models/base_model.py` `is_continuous` flags and logic).

[TODO] 28. **More Sophisticated Distributed Setup:**
    *   (Reference: `EfficientZeroV2` Ray actor/server model for replay, storage).
