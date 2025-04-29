# Training and Evaluating RL Agents in OpenSpiel

This document explains how to use the `train_agent.py` and `evaluate_agent.py` scripts located in `open_spiel/python/examples/`. These scripts provide a command-line interface for training various reinforcement learning agents on OpenSpiel games and evaluating their performance.

## 1. `train_agent.py` - Training an Agent

This script trains an RL agent (Tabular Q-learning, DQN, PPO supported initially) on a specified OpenSpiel game, typically in a multi-agent self-play setting. It utilizes a standardized wrapper (`agent_serialization.py`) for saving agent state and metadata.

### Usage

```bash
python open_spiel/python/examples/train_agent.py --game_name=<game> --algorithm_name=<algo> [options]
```

### Core Arguments

*   `--game_name`: (String, Required) The name of the OpenSpiel game to train on (e.g., `tic_tac_toe`, `kuhn_poker`, `long_narde`).
*   `--algorithm_name`: (String, Required) The RL algorithm to use for *all* players. Check `--list_algorithms` for available options (currently `tabular_qlearner`, `dqn`, `ppo`).
*   `--num_episodes`: (Integer, Default: 10000) The total number of training episodes to run. Note: For vectorized environments, this is the target number of *completed* episodes across all parallel environments.

### Agent Hyperparameters

*   `--agent_hparams`: (String, Can be specified multiple times) Agent-specific hyperparameters passed as key-value pairs. Values are parsed automatically into appropriate types (int, float, bool, list, dict). These hyperparameters are applied to **all** agents being trained.
    *   **Example (QLearner):** `--agent_hparams step_size=0.01 --agent_hparams epsilon_start=1.0 --agent_hparams epsilon_end=0.1` (Note: `learning_rate` might be specific to DQN/PPO)
    *   **Example (DQN):** `--agent_hparams learning_rate=0.001 --agent_hparams replay_buffer_capacity=100000 --agent_hparams hidden_layers_sizes="[128,128]"` (Note: Use quotes for lists/dicts if needed by your shell)
    *   **Example (PPO):** `--agent_hparams learning_rate=3e-4 --agent_hparams batch_size=64 --agent_hparams num_epochs=10`
    *   **Note:** Refer to the specific agent's implementation in `open_spiel/python/algorithms/` or `open_spiel/python/pytorch/` for available hyperparameter names and default values. The script passes these to the agent's `__init__` method via the `SerializableAgentWrapper`.

### Logging and Checkpointing

*   `--log_file`: (String, Optional) Path to a CSV file where training metrics will be saved. Logs episode number and reward for each player. If not provided, CSV logging is disabled.
*   `--tensorboard_logdir`: (String, Optional) Directory where TensorBoard logs will be saved. Logs per-player rewards and available agent losses. If not provided, TensorBoard logging is disabled.
*   `--checkpoint_dir`: (String, Optional) Directory where agent checkpoints will be saved. If not provided, checkpointing is disabled.
*   `--checkpoint_every`: (Integer, Default: 1000) Save a checkpoint every N episodes (for single env) or every N *completed* episodes (for vectorized env). Only active if `checkpoint_dir` is provided and this value is greater than 0.
*   **Checkpoint Structure:** Checkpoints are saved in subdirectories named `agent_p<ID>_ep<EPISODE>` within the `checkpoint_dir`. Each subdirectory contains:
    *   `metadata.yaml`: Information about the agent class, hyperparameters used at creation, OpenSpiel version, environment specs, etc.
    *   Agent-specific state files (e.g., `q_table.pkl`, `agent_state.pkl` for QLearner; TensorFlow checkpoints and `agent_state.pkl` for DQN; PyTorch state dict for PPO - TBD).

### Configuration File

*   `--config_file`: (String, Optional) Path to a YAML or JSON configuration file. Flags set in the file will override defaults but be overridden by explicit command-line flags. Keys in the config file should match the flag names (e.g., `game_name`, `num_episodes`, `agent_hparams`).

### Performance Optimization

*   `--device`: (String, Default: "cpu") Specifies the device (`cpu`, `cuda`, `mps`) for deep learning models (currently passed to PPO). DQN (TF1) placement is handled externally.
*   `--use_vector_env`: (Boolean, Default: False) If true, uses `SyncVectorEnv` to run multiple environments in parallel. This requires agent implementations that support batch processing for efficiency (Task 18d/e).
*   `--num_envs`: (Integer, Default: 4) Number of parallel environments to use if `use_vector_env` is true.

### Utility Arguments

*   `--list_algorithms`: (Boolean, Default: False) If specified, lists the algorithms supported by the script (those mapped in `ALGORITHM_CLASS_PATHS`) and exits.

### Examples

```bash
# Train Q-learner on Tic-Tac-Toe (self-play) for 50k episodes, log to CSV and TensorBoard, save checkpoints
python open_spiel/python/examples/train_agent.py \
    --game_name=tic_tac_toe \
    --algorithm_name=tabular_qlearner \
    --num_episodes=50000 \
    --log_file=/tmp/ttt_q_log.csv \
    --tensorboard_logdir=/tmp/ttt_q_tb \
    --checkpoint_dir=/tmp/ttt_q_checkpoints \
    --checkpoint_every=5000 \
    --agent_hparams step_size=0.02 \
    --agent_hparams discount_factor=0.99

# Train PPO on Long Narde using 8 vectorized environments and CUDA device
python open_spiel/python/examples/train_agent.py \
    --game_name=long_narde \
    --algorithm_name=ppo \
    --num_episodes=20000 \
    --use_vector_env=True \
    --num_envs=8 \
    --device=cuda \
    --tensorboard_logdir=/tmp/ln_ppo_tb \
    --checkpoint_dir=/tmp/ln_ppo_checkpoints \
    --checkpoint_every=1000 \
    --agent_hparams learning_rate=1e-4 \
    --agent_hparams batch_size=256 \
    --agent_hparams num_learning_epochs=4
```

## 2. `evaluate_agent.py` - Evaluating a Trained Agent

This script loads a trained agent from a checkpoint directory (using the standardized `agent_serialization.load_agent` function) and evaluates its performance against an opponent (random or another trained agent) over a specified number of episodes.

### Usage

```bash
python open_spiel/python/examples/evaluate_agent.py --game_name=<game> --agent_path=<path_to_agent_dir> [options]
```

### Core Arguments

*   `--game_name`: (String, Required) The name of the OpenSpiel game the agent was trained on (must match the game used for training and in the checkpoint metadata).
*   `--agent_path`: (String, Required) Path to the specific agent checkpoint *directory* to evaluate (e.g., `/tmp/ttt_q_checkpoints/agent_p0_ep50000`). This directory **must** contain `metadata.yaml` and the agent's state files.
*   `--num_eval_episodes`: (Integer, Default: 100) The number of episodes to run for evaluation.
*   `--output_file`: (String, Default: `evaluation_results.csv`) Path to the CSV file where evaluation metrics will be saved.

### Opponent Configuration

*   `--opponent_type`: (String, Default: `random`) Specifies the type of opponent.
    *   `random`: Pits the agent against a `random_agent.RandomAgent`.
    *   `trained`: Pits the agent against another trained agent. Requires `--opponent_path`.
*   `--opponent_path`: (String, Optional) Path to the specific checkpoint *directory* for the opponent agent. Required if `opponent_type=trained`.

### Evaluation Metrics

*   `--initial_elo`: (Float, Default: 1200.0) Initial Elo rating assigned to both agents before evaluation.
*   `--elo_k_factor`: (Float, Default: 32.0) K-factor used for updating Elo ratings after each game.
*   **Output:**
    *   **Console:** Prints summary statistics (average returns, wins/losses/ties, final Elo).
    *   **CSV (`--output_file`):** Saves average returns, total wins, total losses, total ties, and final Elo ratings per player. Also includes game-specific metrics if available (e.g., average pip difference for `long_narde`).
    *   **TensorBoard (`--tensorboard_logdir`):** Logs evaluation summary metrics (average returns, win/loss/tie rates, final Elo, average pip difference for `long_narde`) under an `eval` subdirectory.

### Logging

*   `--tensorboard_logdir`: (String, Optional) Directory where TensorBoard logs for evaluation metrics will be saved (in an `eval` subdirectory).

### Configuration File

*   `--config_file`: (String, Optional) Path to a YAML or JSON configuration file. Overrides defaults, but is overridden by command-line flags.

### Examples

```bash
# Evaluate a trained Q-learner agent against a random opponent, log to TensorBoard
python open_spiel/python/examples/evaluate_agent.py \
    --game_name=tic_tac_toe \
    --agent_path=/tmp/ttt_q_checkpoints/agent_p0_ep50000 \
    --opponent_type=random \
    --num_eval_episodes=500 \
    --output_file=eval_ttt_q_vs_random.csv \
    --tensorboard_logdir=/tmp/ttt_eval_tb

# Evaluate two trained PPO Long Narde agents against each other, adjusting Elo K-factor
python open_spiel/python/examples/evaluate_agent.py \
    --game_name=long_narde \
    --agent_path=/tmp/ln_ppo_checkpoints/agent_p0_ep10000 \
    --opponent_type=trained \
    --opponent_path=/tmp/ln_ppo_checkpoints/agent_p1_ep10000 \
    --num_eval_episodes=200 \
    --elo_k_factor=16.0 \
    --output_file=eval_ln_ppo_vs_ppo.csv \
    --tensorboard_logdir=/tmp/ln_eval_tb
```

## 3. General Notes

*   **Agent Loading/Saving:** The `SerializableAgentWrapper` and `load_agent` function in `agent_serialization.py` handle the details of saving/loading agent state and metadata. You only interact with the checkpoint directory path.
*   **Hyperparameters:** Evaluation (`evaluate_agent.py`) primarily uses the agent's learned parameters loaded from the checkpoint. Training hyperparameters are generally not needed unless the agent's loading logic specifically requires them (which is currently not the case).
*   **Dependencies:** DQN requires TensorFlow 1.x (`tensorflow-compat-v1`), while PPO requires PyTorch (`torch`). Ensure these are installed if you use those algorithms. TensorBoard logging requires `tensorboardX` or `tensorboard`. YAML support requires `PyYAML`.
*   **Vectorized Environment:** Using `--use_vector_env` in `train_agent.py` can significantly speed up training for algorithms that can process batches (like PPO), but it requires the agent implementation to correctly handle batched inputs and outputs. The current implementation has placeholders and might need further refinement in the agent wrappers or specific agent code (Tasks 18d, 18e).
*   **Checkpoint Paths:** Always ensure `agent_path` and `opponent_path` point to the specific subdirectory created during checkpointing (e.g., `.../checkpoint_dir/agent_p0_ep10000`), which contains the `metadata.yaml` and other necessary files. 