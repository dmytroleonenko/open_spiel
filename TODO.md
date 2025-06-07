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
    *   **(Optional) Self-Supervised Projection Heads:** (Based on `EfficientZeroV2`'s `ProjectionNetwork`) [DONE]
        *   Input: Hidden State (Tensor).
        *   Output: Projected representation (Tensor).
        *   **Completion Criteria:**
            *   Projection heads (`ProjectionNetwork` or similar) are implemented in `open_spiel/python/algorithms/muzero_jax/models/network.py` as Flax NNX modules, mirroring `EfficientZeroV2`'s design if applicable.
            *   The `MuZeroNetwork` (Task 2) is updated to optionally include and utilize these projection heads.
            *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/models/test_network.py` for the projection heads (and their integration into `MuZeroNetwork`) verify expected output shapes, basic functionality, and achieve 100% pass rate.
            *   100% code coverage for the projection head implementation and related `MuZeroNetwork` changes is achieved and verified (e.g., using `coverage report -m` or `coverage html`).
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

[DONE] 6.  **Training Loop (JAX):**
    *   **TDD:** Write Pytest tests for loss components and the overall training step function, verifying gradient computation and parameter updates on mock data. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_trainer.py`) [DONE]
    *   Create `open_spiel/python/algorithms/muzero_jax/training/trainer.py`. [DONE]
    *   (Reference: `@EfficientZeroV2/ez/agents/base.py` (especially `update_weights` method), `@EfficientZeroV2/ez/utils/loss.py`) [DONE]
    *   **Loss Function (MuZero specific, with `EfficientZeroV2` additions):** [DONE]
        *   Policy, Value, Reward losses. [DONE]
        *   (Optional) Consistency/Self-Supervised Loss (from `EfficientZeroV2`). [DONE]
        *   L2 regularization. [DONE]
    *   **Training Step Function (`@jax.jit`):** [DONE]
        *   Input: Model `nnx.State`, Optax `optimizer_state`, batch. [DONE]
        *   Unroll model, calculate losses, compute gradients, update parameters. [DONE]
        *   Output: New model `nnx.State`, new `optimizer_state`, metrics. [DONE]
    *   **Main Training Orchestration:** (Inspired by `@EfficientZeroV2/ez/agents/base.py`'s `train` method`) [DONE]
        *   This component is the **Learner**. Its role is to: Initialize model, optimizer. Loop: Sample batches of trajectories *from* the Replay Buffer, execute the `train_step` function to update network parameters, log metrics (using WandB), and manage checkpointing of the model and optimizer states. [DONE]
        *   Manage target network updates (EMA or periodic copy, per `EfficientZeroV2`). [DONE]
    *   **Completion Criteria:** [DONE]
        *   The `open_spiel/python/algorithms/muzero_jax/training/trainer.py` file is created and contains the complete training loop logic (Learner). [DONE]
        *   **Loss Function:** [DONE]
            *   Policy, value, and reward loss components are implemented, configurable, and correctly calculate losses based on model outputs and targets. [DONE]
            *   (Optional, if projection heads from Task 2's optional part or a later task are included) Consistency/Self-Supervised Loss is implemented and integrated. [DONE]
            *   L2 regularization is implemented and correctly applied. [DONE]
        *   **Training Step Function:** [DONE]
            *   A JIT-compiled training step function (`train_step`) is implemented that takes the model state, optimizer state, and a batch of data as input. [DONE]
            *   `train_step` correctly unrolls the MuZero model (representation, dynamics, prediction, reward) for the required number of steps. [DONE]
            *   `train_step` correctly calculates all specified loss components. [DONE]
            *   `train_step` computes gradients of the total loss with respect to model parameters. [DONE]
            *   `train_step` updates model parameters using the specified Optax optimizer. [DONE]
            *   `train_step` returns the updated model state, optimizer state, and a dictionary of relevant training metrics (e.g., individual losses, total loss, gradient norm). [DONE]
        *   **Main Training Orchestration:** [DONE]
            *   A main training function/class orchestrates the training process: initializes the model and optimizer, iteratively samples batches from the replay buffer, calls the `train_step` function, logs metrics (e.g., to WandB), and handles checkpointing. [DONE]
            *   Target network updates (e.g., EMA or periodic hard copy) are implemented and correctly managed. [DONE]
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/training/test_trainer.py` (covering loss components, `train_step`, and orchestration logic with mock data and models) pass (100%). [DONE]
        *   100% code coverage for `trainer.py` is achieved and verified. [DONE]

[DONE] 6.1.  **Connect Batch Optimizer Hooks to MuZeroNetwork:**
    *   Implement `create_muzero_model_and_params` to initialize the actual `MuZeroNetwork` (using `nnx.Rngs`).
    *   Implement `compute_muzero_loss_and_gradients` with `nnx.value_and_grad` computing the combined MuZero loss.
    *   This task depends on Task 2 (MuZeroNetwork) and Task 6 (loss function definition and training step).
    *   **Completion Criteria:**
        *   ✅ The `create_muzero_model_and_params` function in `open_spiel/python/algorithms/muzero_jax/utils/batch_optimizer.py` correctly initializes the `MuZeroNetwork` (from Task 2) and its parameters using `nnx.Rngs`, returning the initialized model instance suitable for the batch optimizer script.
        *   ✅ The `compute_muzero_loss_and_gradients` function in `batch_optimizer.py` correctly takes the `MuZeroNetwork` instance and a sample batch, performs a forward pass, computes the combined MuZero loss (as defined in Task 6), and uses `nnx.value_and_grad` to return the loss and gradients.
        *   ✅ The `batch_optimizer.py` script can successfully run its analysis (e.g., `find_max_batch_size`, `analyze_throughput`) using the actual `MuZeroNetwork` and its associated loss function.
        *   ✅ All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/utils/test_batch_optimizer.py` are updated/extended to use the real `MuZeroNetwork` (or a faithful mock) and the MuZero loss, verifying the correct functioning of these connection hooks, and all tests pass (100%).
        *   ✅ 100% code coverage for the new/modified functions in `batch_optimizer.py` and any necessary adapter code is achieved and verified.

[DONE] 7.  **Self-Play Loop (JAX):**
    *   **TDD:** Write Pytest tests for the actor loop, ensuring correct interaction with MCTS, game wrapper, and trajectory generation. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/self_play/test_actor.py`)
    *   Create `open_spiel/python/algorithms/muzero_jax/self_play/actor.py`.
    *   (Reference: `@EfficientZeroV2/ez/worker/actor_worker.py`, `@EfficientZeroV2/ez/worker/self_play_worker.py`)
    *   **Actor Loop:** This component is the **Self-Play Actor**. Its role is to: Periodically load the latest (or sufficiently recent) network parameters (from the Learner/checkpoints). Use these parameters to play games against itself (or an environment model) using MCTS. Collect game trajectories (observations, actions, rewards, MCTS policy/value targets). Add these completed trajectories *to* the Replay Buffer for the Learner to consume.
    *   **Post-Implementation Integration Fixes Applied:**
        *   ✅ **Parameter Application:** Fixed `maybe_load_latest_parameters` to actually apply loaded checkpoint parameters to `self.current_params` instead of just logging success.
        *   ✅ **MCTS Parameter Passing:** Updated MCTS calls to pass `self.current_params` instead of `params=None`, ensuring loaded network parameters are used for inference.
        *   ✅ **Observation Handling:** Fixed unused initial observation from `reset()` and corrected JAX array boolean check (`is None` instead of `not`).
        *   ✅ **Chance Node Data Storage:** Enhanced chance node handling to properly store resulting observations, actions, rewards, and policy targets in trajectories for training.
    *   **Completion Criteria:**
        *   ✅ The `open_spiel/python/algorithms/muzero_jax/self_play/actor.py` file is created and contains the self-play actor logic.
        *   ✅ The actor loop can load the latest `MuZeroNetwork` parameters (e.g., from a shared checkpoint location updated by the Learner).
        *   ✅ The actor interacts with an OpenSpiel game via the `GameWrapper` (Task 4).
        *   ✅ For each step in an episode, the actor uses the `mctx.gumbel_muzero_policy` (or its stochastic wrapper from Task 18, if ready and applicable for the game) with the current network model to select an action.
        *   ✅ The actor collects all necessary data for a trajectory (observations, actions, rewards, policy targets, value targets).
        *   ✅ Value and policy targets are computed correctly based on the game's outcome and MCTS search statistics (e.g., n-step returns, MCTS policy).
        *   ✅ Completed trajectories are added to the Flashbax replay buffer (Task 5).
        *   ✅ The actor loop can run for a specified number of games or steps.
        *   ✅ All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/self_play/test_actor.py` (covering model loading, game interaction, MCTS calls, trajectory generation, target computation, and buffer interaction on a simple game) pass (100%). **Achievement: 16/16 tests passing with 100% code coverage.**
        *   ✅ 100% code coverage for `actor.py` is achieved and verified.

[DONE] 8.  **Main Orchestration Script (`run_muzero_jax.py`):**
    *   **TDD:** Write integration tests for the local setup (actor, learner, Flashbax buffer communication). (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_run_muzero_jax.py`)
    *   Initialize Flashbax buffers (in-memory and Vault persistence), start actor(s) (local), start training.
    *   Manage configuration using Hydra (inspired by `@EfficientZeroV2/ez/train.py`).
    *   Integrate WandB for experiment tracking.
    *   **Note on Single TPU Resource Management:** If the system is deployed on a single host with a single TPU that can only be actively used by one JAX context at a time (i.e., one process effectively has exclusive use for its operations), this orchestration script will be responsible for managing TPU access between the Learner and Actor(s). This could involve:
        *   a) Sequentially scheduling distinct phases for self-play (Actors using TPU for inference) and training (Learner using TPU for updates).
        *   b) Implementing a mechanism where Actors request inference from a service (potentially managed by or alongside the Learner) that schedules TPU time.
        *   c) Actors performing inference on CPU if the model is simple enough and TPU access is a bottleneck (though typically not ideal for MuZero).
    The fundamental logical decoupling of Actor (data generation) and Learner (training) roles is maintained, but their physical execution on a single shared TPU would be coordinated by this script.
    *   **Completion Criteria:**
        *   A main script `open_spiel/python/algorithms/muzero_jax/run_muzero_jax.py` is created.
        *   The script correctly initializes the `MuZeroNetwork`, optimizer, Flashbax replay buffer (supporting in-memory and Vault for persistence), self-play actor(s) (Task 7), and the training loop (Task 6).
        *   The script manages the overall workflow: actors generate data and add to buffer, learner trains on data from buffer.
        *   Configuration is managed using Hydra, allowing for easy modification of hyperparameters, game selection, etc., inspired by `@EfficientZeroV2/ez/train.py`.
        *   Weights & Biases (WandB) integration is functional for logging metrics from training and self-play.
        *   The script can run an end-to-end MuZero training process on a simple OpenSpiel game (e.g., Tic-Tac-Toe) for a small number of iterations without errors.
        *   All Pytest integration tests in `open_spiel/python/algorithms/muzero_jax/tests/test_run_muzero_jax.py` (covering the setup and basic interaction of actor, learner, and buffer in a local environment) pass (100%).
        *   100% code coverage for `run_muzero_jax.py` (excluding Hydra boilerplate if extensive, focusing on core orchestration logic) is achieved and verified.

[DONE] 9.  **Checkpointing (Orbax):**
    *   **TDD:** Write Pytest tests for saving and loading model state and optimizer state. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/utils/test_checkpointing.py`)
    *   Integrate Orbax for saving/loading Flax NNX `State` and Optax `optimizer_state`.
    *   **Completion Criteria:**
        *   Orbax is integrated into the training loop (`trainer.py` from Task 6 and/or `run_muzero_jax.py` from Task 8).
        *   The system can periodically save:
            *   The Flax NNX `State` of the `MuZeroNetwork`.
            *   The Optax `optimizer_state`.
            *   Current training step/epoch and any other relevant metadata.
        *   The system can load the latest (or a specified) checkpoint to resume training, correctly restoring the model, optimizer, and training progress.
        *   Checkpointing is robust to interruptions.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/utils/test_checkpointing.py` (covering saving and loading of model state, optimizer state, and training metadata, and verifying state restoration) pass (100%).
        *   100% code coverage for the checkpointing utility functions and their integration points in the trainer/orchestrator is achieved and verified.

[DONE] 10. **Initial Testing & Debugging & Coverage Check:**
    *   Use simple OpenSpiel game (e.g., long_narde, tictactoe).
    *   Run `source venv/bin/activate && coverage report run -m pytestopen_spiel/python/algorithms/muzero_jax/tests/` and `coverage report report` to ensure >95% coverage for Phase 1 components before proceeding. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`)

---

### Phase 2: Distributed Training and Data Generation (adapting `EfficientZeroV2`'s distributed setup)

**Note:** Phase 2 tasks are now **deferred** until single-device training (Phase 1) is complete. We will revisit distributed training once local training is fully implemented.

**Goal (Deferred):** Scale using JAX's distributed capabilities, supporting different deployment topologies after Phase 1 progress. This includes:
    *   **Multi-process on a single machine:** Learner and Actors as separate processes, potentially coordinating access to a shared accelerator (as noted in Task 8).
    *   **Fully Distributed:** Learner and Actors running on separate machines/hosts, communicating via a distributed replay buffer and a mechanism for actors to fetch updated network parameters.

(Test execution command: `source venv/bin/activate && python -m pytest path/to/your_test_file.py`)

[DEFERRED] 11. **Distributed Replay Buffer (Flashbax Vault / Service):**
    *   **TDD:** Write tests for Flashbax Vault in distributed scenarios, or for a replay buffer service. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_distributed_replay_buffer.py`)
    *   Set up Flashbax Vault with disk-backed store (e.g., S3, GCS, NFS) accessible by multiple processes/hosts.
    *   Alternatively, design a replay buffer service (e.g., gRPC-based) that actors can send trajectories to.
    *   **Completion Criteria:**
        *   The replay buffer solution (e.g., Flashbax Vault on shared storage, or a dedicated replay buffer service) can be concurrently accessed by multiple writer (actor) processes and a reader (learner) process, potentially across different hosts.
        *   Actors (from Task 12) can successfully write trajectories to the distributed replay buffer.
        *   The Learner (from Task 13) can successfully sample batches from the distributed replay buffer.
        *   Data integrity and consistency are maintained under concurrent access.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/test_distributed_replay_buffer.py` (simulating multiple processes adding to and sampling from the buffer, checking for data consistency and race conditions) pass (100%).
        *   100% code coverage for any custom wrapper/utility code related to managing the distributed replay buffer is achieved and verified.

[DEFERRED] 12. **Distributed Data Generation (Actors - JAX processes on separate hosts):**
    *   **TDD:** Write tests for actor processes writing to the distributed replay buffer and fetching parameters. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/self_play/test_distributed_actor.py`)
    *   Actors run as independent JAX processes, potentially on different hosts than the Learner.
    *   **Parameter Updates:** Actors periodically poll a shared storage location (e.g., S3, GCS, or a shared filesystem where Learner checkpoints are saved) for the latest (or sufficiently recent) `MuZeroNetwork` parameters.
    *   **Trajectory Submission:** Actors run self-play and submit completed trajectories to the distributed replay buffer (Task 11), e.g., by writing to Flashbax Vault or sending via gRPC to a replay service.
    *   **Completion Criteria:**
        *   Self-play actors (based on Task 7) are implemented as independent JAX processes that can run on different devices/machines.
        *   Actors can periodically load the latest model parameters by polling a shared checkpoint store updated by the Learner.
        *   Each actor runs self-play episodes using the `MuZeroNetwork` and MCTS.
        *   Completed trajectories are serialized and written to the distributed replay buffer solution from Task 11.
        *   The system can scale to multiple actor processes generating data concurrently from different hosts.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/self_play/test_distributed_actor.py` (verifying actor process initialization, model parameter loading from a mock shared store, self-play execution, and trajectory writing to a mock/test distributed replay buffer) pass (100%).
        *   100% code coverage for the distributed actor logic (including parameter polling and data writing) is achieved and verified.

[DEFERRED] 13. **Distributed Training (Learner - `pjit` on dedicated host(s)):**
    *   **TDD:** Tests for `pjit` sharding, distributed checkpointing (to shared storage), and correct gradient aggregation across devices. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_distributed_trainer.py`)
    *   (Reference: `@EfficientZeroV2/ez/agents/base.py` DDP setup, `EfficientZeroV2/ez/train.py` DDP orchestration).
    *   Modify `open_spiel/python/algorithms/muzero_jax/training/trainer.py`. Use `pjit` for data/model parallelism. Define mesh, sharding for `nnx.State` and data.
    *   Learner consumes batches from the distributed replay buffer (Task 11).
    *   Learner saves Orbax checkpoints to a shared storage location (e.g., S3, GCS) for Actors to pick up (as per Task 12).
    *   **Completion Criteria:**
        *   The training loop in `open_spiel/python/algorithms/muzero_jax/training/trainer.py` (from Task 6) is modified to use `jax.pjit` for distributed training on potentially multiple devices on its dedicated host(s).
        *   A JAX device mesh is defined, and appropriate sharding strategies are applied.
        *   Gradients are correctly computed and aggregated across participating devices.
        *   Model parameters are updated synchronously.
        *   Distributed checkpointing using Orbax saves sharded model/optimizer states to a shared storage location accessible by Actors.
        *   The distributed learner successfully trains by sampling batches from the distributed replay buffer (Task 11).
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/training/test_distributed_trainer.py` (verifying `pjit` setup, sharding, gradient aggregation, parameter updates, and distributed checkpointing to a mock shared store) pass (100%).
        *   100% code coverage for the `pjit`-related modifications and distributed checkpointing logic in `trainer.py` is achieved and verified.

[DEFERRED] 14. **JAX Profiler Integration:**
    *   **TDD:** (Difficult to TDD directly, but ensure profiler calls are in place and can be activated).
    *   Add hooks for `jax.profiler`.
    *   **Completion Criteria:**
        *   `jax.profiler.start_trace()` and `jax.profiler.stop_trace()` calls (or context managers) are integrated into key sections of the code, particularly the training step (`trainer.py`) and self-play loop (`actor.py`).
        *   The profiler can be easily enabled/disabled, for example, via a configuration flag in Hydra (Task 8).
        *   The system successfully generates profiling data that can be viewed with tools like TensorBoard.
        *   A brief documented procedure exists explaining how to enable profiling and analyze the output for performance bottlenecks.
        *   While direct TDD is hard, manual verification confirms profiler output is generated and informative for both a local run and, if feasible, a small distributed setup. All Pytest tests for other modules still pass (100%) with profiler hooks present.
        *   100% code coverage for any utility functions written to manage profiling is achieved and verified.

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
    *   **Completion Criteria:**
        *   The main orchestration script (Task 8 or its distributed version) and individual components (learner Task 13, actors Task 12) are designed to handle restarts.
        *   Checkpointing (Task 9 for local, Task 13 for distributed) occurs at regular, configurable intervals.
        *   Upon restart, the learner process correctly loads the latest available valid checkpoint (model state, optimizer state, training progress).
        *   Upon restart, actor processes correctly load the latest available model parameters to continue self-play.
        *   The system can gracefully recover from a simulated crash and restart of the learner process and one or more actor processes, continuing training from the last checkpoint without data corruption.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/test_resilience.py` (simulating crashes and verifying successful restart and continuation of actors and learner from checkpoints) pass (100%).
        *   100% code coverage for any new logic specifically added for fault tolerance and restart capabilities is achieved and verified.

[DEFERRED] 17. **Phase 2 Coverage Check:**
    *   Run `coverage report run -m pytest` and `coverage report report` to ensure >95% coverage for distributed components and batch optimizer script. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`)
    *   **Completion Criteria:**
        *   After all Phase 2 `DEFERRED` tasks (11, 12, 13, 14, 16) are completed and their individual test suites pass with 100% coverage for their respective modules:
            *   A full Pytest run (`python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`) passes 100%.
            *   A combined coverage report (`coverage run -m pytest open_spiel/python/algorithms/muzero_jax/tests/` followed by `coverage report -m` or `coverage html`) shows at least 98% (striving for 100%) total code coverage for all modules developed or modified in Phase 1 and Phase 2. Any uncovered lines are justified or addressed.

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
    *   **Completion Criteria:**
        *   A wrapper around `mctx.gumbel_muzero_policy` is implemented in `open_spiel/python/algorithms/muzero_jax/mcts/mctx_wrapper.py`.
        *   This wrapper correctly:
            *   Initializes `StochasticRecurrentState` by appropriately padding `prior_logits` for chance outcomes and wrapping the root embedding.
            *   Constructs a `recurrent_fn` using `_make_stochastic_recurrent_fn` (or equivalent logic) that correctly dispatches to either a `decision_recurrent_fn` (based on the MuZero dynamics model `g` and reward model) or a `chance_recurrent_fn` (which samples from game chance outcomes and uses the representation model `h` for the next state).
            *   Calls the underlying `mctx.gumbel_muzero_policy` with the stochastic recurrent function and appropriately structured inputs.
        *   The `SelfPlay Loop` (Task 7 or 12) and `Main Orchestration Script` (Task 8) can use this stochastic MCTS wrapper when configured for a game with chance nodes.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/test_stochastic_mcts.py` (instantiating the wrapper with mock recurrent functions and a simple OpenSpiel game with chance nodes, verifying correct action selection, visit counts, embedding alternation for decision/chance nodes, deterministic behavior with RNG seeding, and handling of invalid actions) pass (100%).
        *   100% code coverage for `mctx_wrapper.py` and any helper functions for stochastic handling is achieved and verified.

[TODO] 19. **Reanalyze Implementation (EfficientZeroV2 style):**
    *   **TDD:** Tests for reanalyze worker logic, target updates in Flashbax, and interaction with main training loop. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_reanalyze.py`)
    *   (Reference: `@EfficientZeroV2/ez/worker/reanalyze_worker.py`, `ez/agents/base.py` reanalyze intervals).
    *   Adapt `EfficientZeroV2`'s reanalyze mechanism.
    *   **Completion Criteria:**
        *   A reanalyze worker/process is implemented, potentially in `open_spiel/python/algorithms/muzero_jax/self_play/reanalyze_worker.py` or similar.
        *   The reanalyze worker can sample trajectories from the Flashbax replay buffer (Task 5).
        *   For each sampled trajectory, it uses a more recent version of the `MuZeroNetwork` model to re-run MCTS for each state in the trajectory, generating new policy and value targets.
        *   The updated targets (and potentially recomputed priorities) are written back to the Flashbax buffer, replacing or updating the old targets for those trajectories.
        *   The main training loop (Task 6 or 13) can be configured to trigger reanalysis periodically or based on certain conditions.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/test_reanalyze.py` (covering reanalyze worker logic, interaction with Flashbax for reading trajectories and writing updated targets, and correct target re-computation using a mock model and trajectory data) pass (100%).
        *   100% code coverage for the reanalyze worker implementation and its integration points is achieved and verified.

[TODO] 20. **Self-Supervised Learning (if adopted from EfficientZeroV2):**
    *   **TDD:** Tests for the self-supervised loss component and its integration into the main loss. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_ssl_loss.py`)
    *   (Reference: `@EfficientZeroV2/ez/agents/models/base_model.py` Projection Networks, relevant loss terms in `update_weights`).
    *   If projection heads were implemented, add corresponding SSL loss to main training loss.
    *   **Completion Criteria:**
        *   This task assumes Self-Supervised Projection Heads (from Task 2's optional part or an earlier Phase 3 task) are implemented.
        *   A self-supervised learning (SSL) loss component (e.g., contrastive loss, BYOL-style loss) is implemented, operating on the outputs of the projection heads from the `MuZeroNetwork`.
        *   This SSL loss is integrated into the main MuZero loss function within the training loop (`trainer.py`, Task 6 or 13).
        *   The weight/contribution of the SSL loss is configurable.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/training/test_ssl_loss.py` (verifying the correct calculation of the SSL loss component given projected representations and its successful integration into the total loss calculation and gradient updates) pass (100%).
        *   100% code coverage for the SSL loss implementation and its integration into the training process is achieved and verified.

[TODO] 21. **Support for wider range of OpenSpiel games:**
    *   **TDD:** Add test suites for new game types as they are supported. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/games/`)
    *   Test and adapt for image-based games (e.g., Atari if `EfficientZeroV2` features are fully ported).
    *   **Completion Criteria:**
        *   The MuZero JAX implementation (including `GameWrapper`, network input/output processing, MCTS, and training loop) is successfully run end-to-end (self-play and training for at least a few iterations) on a diverse set of OpenSpiel games, including:
            *   At least one small, deterministic, perfect information board game (e.g., Tic-Tac-Toe, Connect Four).
            *   At least one game with chance nodes (e.g., Backgammon (if simplified) or a custom simple chance game, using stochastic MCTS from Task 18).
            *   At least one game with larger observation and/or action spaces (e.g., Go (small board), Othello).
            *   (If `EfficientZeroV2` image processing features like `DownSample` are fully ported and tested) At least one game with image-like observations (e.g., a simplified Atari-like environment or Catch).
        *   For each newly supported game type, a specific Pytest test suite is added under `open_spiel/python/algorithms/muzero_jax/tests/games/` that runs a minimal end-to-end test (e.g., one episode of self-play, one training step) to ensure compatibility. All these tests pass (100%).
        *   Any necessary adaptations in `GameWrapper` or network input/output handling for these new game types are implemented and covered by tests.
        *   100% code coverage for any game-specific adaptation code is achieved and verified. The overall project coverage remains high.

[TODO] 22. **Hyperparameter Tuning:**
    *   (Reference: `@EfficientZeroV2` Hydra configs in `@EfficientZeroV2/ez/config/`)
    *   Use `EfficientZeroV2`'s configurations (managed with Hydra) as starting point.
    *   **Completion Criteria:**
        *   A comprehensive set of default hyperparameters for the MuZero JAX agent is established and documented, drawing from `EfficientZeroV2` configurations and initial experiments. These are managed via Hydra configuration files in a structured way (e.g., `open_spiel/python/algorithms/muzero_jax/configs/`).
        *   Configuration files exist for different game types or experimental setups tested in Task 21.
        *   A documented process or script exists to launch hyperparameter sweeps (e.g., using Hydra's sweeper plugins with Optuna/Ray Tune, or a custom script if simpler).
        *   At least one hyperparameter sweep is successfully executed on a simple game (e.g., Tic-Tac-Toe or CartPole if wrapped for JAX) to demonstrate the tuning pipeline's functionality. Results (even if preliminary) are logged (e.g., to WandB).
        *   While direct TDD for "good hyperparameters" is not possible, the configuration loading, sweep execution mechanism, and logging are robust. All Pytest tests for the main orchestration script (Task 8) using various valid configurations pass (100%).
        *   100% code coverage for any custom scripts or utilities written specifically for hyperparameter management or sweep execution is achieved and verified.

[TODO] 23. **Evaluation Pipeline:**
    *   **TDD:** Tests for the evaluation loop, metric calculation, and agent loading. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/test_evaluation.py`)
    *   (Reference: `@EfficientZeroV2/ez/eval.py`)
    *   Implement evaluation similar to `EfficientZeroV2`.
    *   **Completion Criteria:**
        *   An evaluation script/module (e.g., `open_spiel/python/algorithms/muzero_jax/evaluation.py`) is implemented.
        *   The evaluation pipeline can:
            *   Load a saved/checkpointed `MuZeroNetwork` model.
            *   Play a specified number of games against a baseline opponent (e.g., random actor, MCTS with fewer simulations, or another instance of itself from a previous checkpoint) or measure performance in a single-player game.
            *   Use a deterministic version of MCTS (e.g., by taking the action with the highest visit count or value, not sampling from the policy) for evaluation.
            *   Calculate and log relevant performance metrics (e.g., win rate, average score, episode length).
            *   The evaluation can be run periodically during training or as a standalone process.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/test_evaluation.py` (covering model loading, game play for evaluation, metric calculation, and interaction with different types of baseline opponents on mock/simple games) pass (100%).
        *   100% code coverage for `evaluation.py` (or equivalent module) is achieved and verified.

[TODO] 24. **Code Refinement and Documentation:**
    *   Ensure all new code meets TDD and coverage standards.
    *   Update README and add docstrings.
    *   **Completion Criteria:**
        *   All implemented Python modules and public functions/classes have clear, concise, and accurate docstrings in a standard format (e.g., Google style or NumPy style).
        *   The main `README.md` for the `muzero_jax` algorithm is updated to reflect the current implementation, features, and usage instructions (how to run training, evaluation, use Hydra configs).
        *   Code is reviewed for clarity, efficiency, and adherence to JAX/Flax best practices. Any identified areas for minor refactoring are addressed.
        *   All previously written Pytest tests continue to pass (100%).
        *   Overall code coverage for the `muzero_jax` project remains at 98%+ (striving for 100%).
        *   A review of all TODO comments in the code is performed; they are either addressed or converted into new tasks if significant.

[TODO] 25. **Final Coverage Check & Polish:**
    *   Aim for 100% test coverage across the entire project. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/`)
    *   Final review of code quality, documentation, and examples.
    *   **Completion Criteria:**
        *   All preceding tasks in Phases 1, 2, and 3 are marked as complete according to their respective criteria.
        *   A final, full Pytest suite execution (`python -m pytest open_spiel/python/algorithms/muzero_jax/`) passes 100%.
        *   The final code coverage report (`coverage run -m pytest open_spiel/python/algorithms/muzero_jax/tests/` followed by `coverage report -m` or `coverage html`) shows 100% code coverage for all implemented `muzero_jax` modules. Any line not covered is explicitly justified and deemed acceptable (e.g., defensive assertion that is practically unreachable).
        *   The project structure, file naming, and code style are consistent and clean.
        *   The main `README.md` and all docstrings (Task 24) are complete, accurate, and provide sufficient information for users to understand and run the algorithm.
        *   At least one end-to-end example run (training + evaluation) on a well-known OpenSpiel game (e.g., Tic-Tac-Toe) is documented and works as described.

---
### Later / Optional

[TODO] 26. **Advanced Network Architectures:**
    *   (Reference: `EfficientZeroV2` model variants, e.g., LSTM use).
    *   **Completion Criteria:**
        *   At least one alternative network architecture (e.g., incorporating LSTMs for handling partial observability more explicitly, or using Transformer layers if deemed beneficial and aligned with `EfficientZeroV2` variants) is implemented as a configurable option for the `MuZeroNetwork`.
        *   The new architecture is integrated into the existing training and self-play framework.
        *   All Pytest tests are added for the new network components and their integration, verifying correct shapes and basic functionality, and all these tests pass (100%).
        *   100% code coverage for the new architectural components is achieved and verified.
        *   A comparative experiment is run against the baseline network architecture on at least one relevant game, with results logged (e.g., to WandB).

[TODO] 27. **Advanced Optimizer Support (K-FAC/ACKTR):**
    *   **TDD:** Write Pytest tests for the K-FAC optimizer integration, verifying parameter updates on a mock model and loss. (Test execution: `source venv/bin/activate && python -m pytest open_spiel/python/algorithms/muzero_jax/tests/training/test_kfac_optimizer.py`)
    *   Integrate K-FAC (Kronecker-Factored Approximate Curvature) as the primary optimizer within the training loop (Task 6 / Task 13). This is the core of ACKTR.
    *   (Reference: JAX KFAC library if available, or implementations from other JAX-based RL agents).
    *   The training setup will exclusively use K-FAC.
    *   **Completion Criteria:**
        *   K-FAC is successfully integrated as the sole optimizer in `trainer.py`.
        *   The training loop correctly uses K-FAC to update `MuZeroNetwork` parameters.
        *   K-FAC and its specific hyperparameters are configured for the training process.
        *   All Pytest tests in `open_spiel/python/algorithms/muzero_jax/tests/training/test_kfac_optimizer.py` (covering K-FAC application to a mock model, gradient computation, and parameter updates) pass (100%).
        *   100% code coverage for the K-FAC integration logic within `trainer.py` and any K-FAC utility/wrapper functions is achieved and verified.
        *   (Optional but recommended) A comparative run against a standard optimizer (e.g., Adam) on a simple environment is performed during development to ensure K-FAC behaves as expected and to understand its characteristics, though Adam will not be part of the final agent.

[TODO] 28. **Support for Continuous Actions:**
    *   (Reference: `EfficientZeroV2/ez/agents/models/base_model.py` `is_continuous` flags and logic).
    *   **Completion Criteria:**
        *   The `MuZeroNetwork`'s prediction function is adapted to output parameters for a continuous probability distribution (e.g., mean and std dev for a Gaussian) instead of logits over discrete actions, when configured for a continuous action space game.
        *   The MCTS selection and backpropagation mechanisms are adapted to handle continuous actions (this might involve sampling actions from the distribution during tree traversal and potentially different approaches for policy improvement). This should align with `EfficientZeroV2`'s approach if specified.
        *   The `GameWrapper` and OpenSpiel interaction are confirmed to correctly handle continuous action spaces.
        *   The training loop and loss functions are updated to work with continuous action distributions (e.g., using policy gradient methods appropriate for continuous actions or by discretizing the action space if that's the chosen strategy).
        *   All Pytest tests are added for continuous action support, covering network output, MCTS interaction, and training on a simple continuous action game (e.g., Pendulum if wrapped, or a custom simple environment). All these tests pass (100%).
        *   100% code coverage for all new or modified code related to continuous action support is achieved and verified.

[TODO] 29. **More Sophisticated Distributed Setup (Optional - e.g., Ray-based):**
    *   (Reference: `EfficientZeroV2` Ray actor/server model for replay, storage).
    *   **Completion Criteria:**
        *   If `EfficientZeroV2`'s Ray-based actor/server model for replay and storage is adopted:
            *   Ray actors are implemented for self-play.
            *   A Ray-based distributed replay buffer server is implemented, potentially replacing or augmenting the Flashbax Vault setup for higher throughput or more complex sampling strategies.
            *   A Ray-based parameter server or mechanism for distributing model updates to actors is implemented.
        *   The system demonstrates improved scalability or performance compared to the Phase 2 distributed setup on a benchmark task.
        *   All Pytest tests are added for the new Ray-based components (actors, replay server, parameter server), verifying their individual functionality and interactions in a distributed mock environment. All these tests pass (100%).
        *   100% code coverage for the Ray-based distributed components is achieved and verified.
        *   The setup is documented, explaining how to deploy and run it.
