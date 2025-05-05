# Training and Evaluating RL Agents in OpenSpiel

This document explains how to use the `train_agent.py` and `evaluate_agent.py` scripts located in `open_spiel/python/examples/`. These scripts provide a command-line interface for training various reinforcement learning agents on OpenSpiel games and evaluating their performance.

## 1a. `tabular_qlearner_long_narde` - Special Q-Learner for Long Narde and Backgammon

For games like Long Narde and Backgammon, where some states have no legal actions (forced pass turns), use the custom agent `tabular_qlearner_long_narde` instead of the default `tabular_qlearner`. This agent is a necessary fork because:
1.  It is robust to forced pass turns, correctly handling states with empty legal action sets.
2.  It works correctly with the `agent_serialization.py` wrapper to ensure proper saving of agent state and metadata, resolving issues encountered with the standard QLearner and the wrapper interaction (specifically related to saving metadata including the `pyspiel.__version__`).

### Usage Example

```bash
python open_spiel/python/examples/train_agent.py --game_name=long_narde --algorithm_name=tabular_qlearner_long_narde --num_episodes=10000 --checkpoint_dir=/tmp/long_narde_q_checkpoints --checkpoint_every=1000 --device=cpu --agent_hparams step_size=0.02 --agent_hparams discount_factor=0.99
```

- Set `--algorithm_name=tabular_qlearner_long_narde` for Long Narde or Backgammon.
- All other arguments remain the same as for the default QLearner.

## 1b. Using DQN for Long Narde/Backgammon (`dqn_long_narde`)

Games like Long Narde and Backgammon feature states where a player has no legal moves and must pass (a "forced pass"). The standard DQN agent may encounter errors in these situations.

To address this, a dedicated agent fork, `dqn_long_narde`, is provided (`open_spiel/python/algorithms/dqn_long_narde.py`).

*   **How it works:**
    1.  Handles forced passes (`action=None`).
    2.  Skips adding transitions with `prev_action=None` to the replay buffer.
    3.  Includes input normalization using `tf.keras.layers.LayerNormalization` *within its neural network definition*. This step normalizes the input features (state representation) *before* they are processed by the main network layers. Input normalization is crucial for stabilizing training and improving convergence, especially when input features have significantly different scales or distributions, as can be the case in complex game states. Consider implementing similar normalization if developing new agents for games with complex or high-dimensional state representations.
*   **Requirement:** You **must** use the specific algorithm name `--algorithm_name=dqn_long_narde` when training DQN on games like Long Narde or Backgammon.
*   **Known Issues & Fixes:**
    *   **Metadata Saving:** The `SerializableAgentWrapper` used by `train_agent.py` might fail to save the necessary `metadata.yaml` file alongside the TensorFlow checkpoints for `dqn_long_narde`. If this occurs, loading the agent via `evaluate_agent.py` will fail.
        *   *Workaround:* Manually create the `metadata.yaml` file in the checkpoint directory (see previous debugging steps for structure).
        *   *Fix Status:* Fixes in `agent_serialization.load_agent` help load manually created metadata, but the root saving issue might persist.
    *   **Checkpoint Loading:** The `restore()` method within `dqn_long_narde.py` was previously incompatible with the checkpoint file structure saved by the wrapper (`agent_pX_epY_q_network.*`), causing the agent to play with uninitialized weights during evaluation.
        *   *Fix Status:* The `restore` method in `dqn_long_narde.py` has been **updated** (as of recent debugging) to correctly load checkpoints saved by the wrapper using the expected path prefix.

### Usage Example (DQN for Long Narde using the fork)

```bash
python open_spiel/python/examples/train_agent.py \
    --game_name=long_narde \
    --algorithm_name=dqn_long_narde \
    --num_episodes=15000 \
    --checkpoint_dir=/tmp/long_narde_dqn_checkpoints \
    --checkpoint_every=1000 \
    --agent_hparams hidden_layers_sizes=[128,128] \
    --agent_hparams learning_rate=0.0001
```

## 1. `train_agent.py` - Training an Agent

This script trains an RL agent on a specified OpenSpiel game, typically in a multi-agent self-play setting. It utilizes a standardized wrapper (`agent_serialization.py`) for saving agent state and metadata.

### Usage

```bash
python open_spiel/python/examples/train_agent.py --game_name=<game> --algorithm_name=<algo> [options]
```

### Core Arguments

*   `--game_name`: (String, Required) The name of the OpenSpiel game to train on (e.g., `tic_tac_toe`, `kuhn_poker`, `long_narde`).
*   `--algorithm_name`: (String, Required) The RL algorithm to use for *all* players. Check `--list_algorithms` for available options (currently `tabular_qlearner`, `tabular_qlearner_long_narde`, `dqn`, `dqn_long_narde`, `ppo`).
*   `--num_episodes`: (Integer, Default: 10000) The total number of training episodes to run. Note: For vectorized environments, this is the target number of *completed* episodes across all parallel environments.

### Agent Hyperparameters

*   `--agent_hparams`: (String, Can be specified multiple times) Agent-specific hyperparameters passed as key-value pairs. Values are parsed automatically into appropriate types (int, float, bool, list, dict). These hyperparameters are applied to **all** agents being trained.
    *   **Example (QLearner):** `--agent_hparams step_size=0.01 --agent_hparams epsilon_start=1.0 --agent_hparams epsilon_end=0.1`
    *   **Example (DQN):** `--agent_hparams learning_rate=0.001 --agent_hparams replay_buffer_capacity=100000 --agent_hparams hidden_layers_sizes="[128,128]"`
    *   **Example (PPO):** `--agent_hparams learning_rate=3e-4 --agent_hparams batch_size=64 --agent_hparams num_epochs=10`
    *   **Note:** Refer to the specific agent's implementation for available hyperparameters.

### Logging and Checkpointing

*   `--log_file`: (String, Optional) Path to a CSV file for training metrics.
*   `--tensorboard_logdir`: (String, Optional) Directory for TensorBoard logs.
*   `--checkpoint_dir`: (String, Optional) Directory where agent checkpoints will be saved.
*   `--checkpoint_every`: (Integer, Default: 1000) Save a checkpoint every N episodes.
*   **Checkpoint Structure:** Checkpoints are saved in subdirectories named `agent_p<ID>_ep<EPISODE>` within the `checkpoint_dir`. Each subdirectory *should* contain `metadata.yaml` and agent-specific state files (e.g., TF checkpoints for DQN, `.pkl` files for QLearner). **Note:** Metadata saving for `dqn_long_narde` might be unreliable (see Section 1b).

### Configuration File

*   `--config_file`: (String, Optional) Path to a YAML or JSON configuration file.

### Performance Optimization

*   `--device`: (String, Default: "cpu") Device (`cpu`, `cuda`, `mps`) for deep learning models (PPO).
*   `--use_vector_env`: (Boolean, Default: False) Use `SyncVectorEnv` for parallel environments.
*   `--num_envs`: (Integer, Default: 4) Number of parallel environments if `use_vector_env` is true.

### Utility Arguments

*   `--list_algorithms`: (Boolean, Default: False) Lists supported algorithms and exits.

### Examples

```bash
# Train Q-learner on Tic-Tac-Toe with logging and checkpoints
python open_spiel/python/examples/train_agent.py \
    --game_name=tic_tac_toe \
    --algorithm_name=tabular_qlearner \
    --num_episodes=50000 \
    --log_file=/tmp/ttt_q_log.csv \
    --tensorboard_logdir=/tmp/ttt_q_tb \
    --checkpoint_dir=/tmp/ttt_q_checkpoints \
    --checkpoint_every=5000

# Train DQN on Long Narde (using the specific fork) with vector env
python open_spiel/python/examples/train_agent.py \
    --game_name=long_narde \
    --algorithm_name=dqn_long_narde \
    --num_episodes=15000 \
    --use_vector_env=True --num_envs=4 \
    --checkpoint_dir=/tmp/ln_dqn_vec_chkpts \
    --checkpoint_every=1000 \
    --agent_hparams learning_rate=0.001 loss_str=huber optimizer_str=adam
```

## 2. `evaluate_agent.py` - Evaluating a Trained Agent

This script loads a trained agent from a checkpoint directory and evaluates its performance against an opponent.

*   **Recent Updates:**
    *   Added `--eval_random_vs_random` flag to run Random vs Random baseline.
    *   Improved agent loading logic in `agent_serialization.py` (especially for DQN session handling).
    *   Fixed `player_id` access for `RandomAgent` opponents.
    *   Added missing `calculate_new_elo_ratings` helper function.
    *   Fixed win/loss counting logic.
    *   Fixed pip count calculation for Long Narde (avoids direct `pyspiel.long_narde` access).

### Usage

```bash
# Evaluate a specific agent
python open_spiel/python/examples/evaluate_agent.py --game_name=<game> --agent_path=<path_to_agent_dir> [options]

# Evaluate Random vs Random baseline
python open_spiel/python/examples/evaluate_agent.py --game_name=<game> --eval_random_vs_random [options]
```

### Core Arguments

*   `--game_name`: (String, Required) Name of the game.
*   `--agent_path`: (String, Required unless `--eval_random_vs_random=True`) Path to the primary agent checkpoint directory.
*   `--num_eval_episodes`: (Integer, Default: 100) Number of evaluation episodes.
*   `--output_file`: (String, Default: `evaluation_results.csv`) Path for CSV output.

### Opponent Configuration

*   `--opponent_type`: (String, Default: `random`) Opponent type (`random` or `trained`). Ignored if `--eval_random_vs_random=True`.
*   `--opponent_path`: (String, Optional) Path to trained opponent checkpoint. Ignored if `--eval_random_vs_random=True`.

### Random vs Random Evaluation

*   `--eval_random_vs_random`: (Boolean, Default: False) If true, evaluates RandomAgent vs RandomAgent. Ignores `agent_path`, `opponent_path`, and `opponent_type`. Useful for establishing baseline performance and first-player advantage in games like Long Narde.

### Evaluation Metrics

*   `--initial_elo`: (Float, Default: 1200.0) Initial Elo rating.
*   `--elo_k_factor`: (Float, Default: 32.0) K-factor for Elo updates.
*   **Output:** Console summary, CSV file, optional TensorBoard logs.

### Logging

*   `--tensorboard_logdir`: (String, Optional) Directory for TensorBoard evaluation logs.

### Configuration File

*   `--config_file`: (String, Optional) Path to a YAML or JSON configuration file.

### Examples

```bash
# Evaluate trained DQN Long Narde agent vs Random (1000 episodes)
python open_spiel/python/examples/evaluate_agent.py \
    --game_name=long_narde \
    --agent_path=/tmp/ln_dqn_vec_chkpts/agent_p0_ep15000 \
    --opponent_type=random \
    --num_eval_episodes=1000 \
    --output_file=eval_dqn_vs_random.csv \
    --tensorboard_logdir=/tmp/ln_eval_tb

# Evaluate Random vs Random baseline for Long Narde (1000 episodes)
python open_spiel/python/examples/evaluate_agent.py \
    --game_name=long_narde \
    --eval_random_vs_random \
    --num_eval_episodes=1000 \
    --output_file=eval_random_vs_random.csv \
    --tensorboard_logdir=/tmp/ln_eval_tb
```

## 3. General Notes

*   **Agent Loading/Saving:** Primarily handled by `agent_serialization.py`. **Verify `dqn_long_narde` loading/saving works correctly, especially metadata and weight restoration.**
*   **Hyperparameters:** Evaluation uses learned parameters from checkpoints.
*   **Dependencies:** DQN (TF1), PPO (PyTorch), TensorBoard (`tensorboardX`/`tensorboard`), YAML (`PyYAML`).
*   **Vectorized Environment:** Use `--use_vector_env` in `train_agent.py` for potential speedup with batch-supporting agents.
*   **Checkpoint Paths:** `agent_path`/`opponent_path` point to the specific checkpoint subdirectory (e.g., `.../agent_p0_ep10000`).