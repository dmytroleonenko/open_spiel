## Action Plan

### Phase 1: Core Infrastructure and Serialization

1.  [X] Define Agent Interface Standard: Specify required methods (e.g., `__init__`, `step`, `save(path)`, `restore(path)`).
2.  [X] Implement Metadata Structure: Define fields for `metadata.yaml` (`agent_class`, `agent_hparams`, `serialization_format_version`, `openspiel_version`).
3.  [X] Implement Agent Saving (Wrapper Approach): `SerializableAgentWrapper.save(path)` created in `utils/agent_serialization.py` handles agent-specific state (pickle/TF) and metadata.
4.  [X] Implement Unified Agent Loading Function: `load_agent(path)` created in `utils/agent_serialization.py` handles metadata, dynamic import, instantiation, and state restoration (pickle/TF).
5.  [X] Update Checkpoint Saving Logic: Modified `train_agent.py` to instantiate `SerializableAgentWrapper` and call its `save(path)` method.
6.  [X] Update Agent Loading Logic: Modified `evaluate_agent.py` to use the unified `agent_serialization.load_agent(path)`. Resuming training in `train_agent.py` would also need this (currently not implemented).

### Phase 2: Configuration and Basic Enhancements

7.  [X] Add Config File Argument: Add `--config_file` flag to both scripts.
8.  [X] Implement Config File Loading: Add logic to load YAML/JSON config early in `main`.
9.  [X] Implement Flag Overriding: Add logic to apply config values to flags, respecting CLI precedence and handling unknown keys.
10. [X] Integrate TensorBoard Logging:
    -   [X] Add `--tensorboard_logdir` flag.
    -   [X] Initialize `SummaryWriter` in both scripts.
    -   [X] Log training metrics (reward, loss) in `train_agent.py`.
    -   [X] Log evaluation metrics (win/loss/tie, avg returns) in `evaluate_agent.py`.

### Phase 3: Advanced Training and Evaluation Features

11. [X] Implement Multi-Agent Training Loop (Self-Play for `long_narde`): Modify `train_agent.py` loop to handle turns for both players and allow both agents to learn from transitions.
12. [X] Implement Optional Shared Replay Buffer: (Decision: Use independent buffers) No shared buffer implemented; agents manage their own buffers (default behavior).
13. [X] Log Multi-Agent Metrics: Update CSV/TensorBoard logging in `train_agent.py` for both agents during self-play.
14. [X] Add Elo Rating Calculation: Implement Elo system in `evaluate_agent.py`, update ratings, and log Elo scores.
15. [X] Add Game-Specific Metrics (`long_narde`): Extract and log metrics like pip count difference in `evaluate_agent.py`.

### Phase 4: Performance Optimizations

16. [X] Add Device Selection Flag: Add `--device` flag ("cpu", "cuda", "mps") to `train_agent.py`.
17. [X] Pass Device to Agents: Modify DQN/PPO constructors to accept and use the `device` flag.
18. [X] Implement Vectorized Environment Option:
    -   [X] Add `--use_vector_env` and `--num_envs` flags to `train_agent.py`.
    -   [X] If enabled, instantiate `SyncVectorEnv`.
    -   [X] Modify training loop for batched `time_steps`.
    -   [X] Modify agents/wrappers for batched observations/actions.
    -   [X] Adapt learning logic for batches of transitions.
    -   [X] Adjust metric aggregation for multiple environments.
