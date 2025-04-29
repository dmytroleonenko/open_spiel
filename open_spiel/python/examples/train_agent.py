"""Train an agent to play an OpenSpiel game using various RL algorithms.

This script allows training different agents (Tabular Q-learning, DQN, PPO) on
any game supported by OpenSpiel. It handles command-line arguments for game
selection, algorithm choice, number of episodes, logging, checkpointing, and
agent-specific hyperparameters.

Example usage:
  python open_spiel/python/examples/train_agent.py --game_name=tic_tac_toe \
    --algorithm_name=tabular_qlearner --num_episodes=10000 \
    --log_file=tictactoe_qlearner_log.csv --checkpoint_dir=/tmp/tictactoe_q \
    --agent_hparams learning_rate=0.01 epsilon_start=1.0 epsilon_end=0.1

Based on the template from TRAIN_LONG_NARDE.md and open_spiel/python/examples/tic_tac_toe_qlearner.py
"""

from absl import app
from absl import flags

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner # Start with Q-learner
from open_spiel.python.algorithms import dqn
from open_spiel.python.pytorch import ppo
import torch # Required for PPO
import tensorflow.compat.v1 as tf # type: ignore # Required for DQN
from open_spiel.python.utils import training
from open_spiel.python.utils import agent_serialization # Added
import csv # For logging
import os # For path manipulation
import sys # For checking agent types
import ast # For parsing hyperparameters
import yaml # Added for config file loading
import json # Added for config file loading
from tensorboardX import SummaryWriter # Added for TensorBoard logging
from open_spiel.python.vector_env import SyncVectorEnv # Added for Task 18

FLAGS = flags.FLAGS

# --- Core Training Parameters ---
flags.DEFINE_string("game_name", "tic_tac_toe", "Name of the game to train on.")
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
flags.DEFINE_string("algorithm_name", "tabular_qlearner",
                    "Name of the RL algorithm to use (e.g., 'tabular_qlearner', 'dqn', 'ppo').")

# --- Agent Hyperparameters ---
# Task 31: Generic hyperparameter flag using ast.literal_eval for type parsing
flags.DEFINE_multi_string("agent_hparams", [],
                          "Agent-specific hyperparameters as key=value pairs. "
                          "Example: --agent_hparams learning_rate=0.01 --agent_hparams hidden_layers_sizes=[64,64]. "
                          "Values are parsed automatically (int, float, bool, list, dict).")

# --- Utilities ---
flags.DEFINE_boolean("list_algorithms", False, "List available algorithms and exit.")

# --- Logging and Checkpointing ---
flags.DEFINE_string("log_file", None, "Path to save training metrics (CSV). If None, logging is disabled.")
flags.DEFINE_string("checkpoint_dir", None, "Directory to save agent checkpoints. If None, checkpointing is disabled.")
flags.DEFINE_integer("checkpoint_every", 1000,
                   "Save agent checkpoint every N episodes. If 0 or checkpoint_dir is None, disables checkpointing.")

# --- Configuration File ---
flags.DEFINE_string("config_file", None, "Path to a YAML/JSON configuration file. Flags set in the file will override defaults but be overridden by command-line flags.")

# --- TensorBoard Logging ---
flags.DEFINE_string("tensorboard_logdir", None, "Optional directory to save TensorBoard logs.")

# --- Performance Optimizations --- (Phase 4)
flags.DEFINE_string("device", "cpu", "Device to use for deep learning models ('cpu', 'cuda', 'mps').")
flags.DEFINE_boolean("use_vector_env", False, "Whether to use SyncVectorEnv for training.")
flags.DEFINE_integer("num_envs", 4, "Number of parallel environments to use if use_vector_env is True.")

# --- DEPRECATED FLAGS (kept for potential backward compatibility, prefer agent_hparams) ---
# flags.DEFINE_float("q_learning_rate", 0.01, "DEPRECATED: Use --agent_hparams learning_rate=...")
# ... (other deprecated flags can be listed here if needed) ...


def _train_episode(env: rl_environment.Environment, agent: rl_environment.AbstractAgent, is_evaluation: bool = False) -> float:
  """Runs a single training or evaluation episode for a single agent.

  DEPRECATED: This function is kept for reference but the main loop now handles
  multi-agent interaction directly.

  Args:
    env: The rl_environment instance.
    agent: The agent instance to train/evaluate.
    is_evaluation: If true, the agent should act greedily (no exploration).

  Returns:
    The total reward accumulated by the agent during the episode.
  """
  total_reward = 0.
  time_step = env.reset()
  while not time_step.last():
    # Agent takes a step based on the current time_step (observation, etc.)
    agent_output = agent.step(time_step, is_evaluation=is_evaluation)

    # If the agent returned an action (it might not, e.g., if waiting)
    if agent_output is not None:
      action_list = [agent_output.action]
      # Environment steps based on the agent's action
      time_step = env.step(action_list)
      # Accumulate reward. Assumes single-agent focus or player 0 reward.
      # Might need adjustment for multi-agent training where rewards are shared/different.
      if time_step.rewards: # Rewards list might be empty
        total_reward += time_step.rewards[agent.player_id] # Index reward by agent's player_id

  # Final step for the agent after the episode ends (e.g., for learning from the final state)
  agent.step(time_step)
  return total_reward

def _parse_hparams(hparam_list: list[str]) -> dict:
    """Parses a list of 'key=value' strings into a dictionary with typed values.

    Uses `ast.literal_eval` for safe parsing of basic Python types (int, float,
    bool, list, dict). Falls back to string if parsing fails.

    Args:
      hparam_list: A list of strings, each in the format "key=value".

    Returns:
      A dictionary where keys are hyperparameter names and values are parsed types.
    """
    params = {}
    if not hparam_list:
      return params
    for item in hparam_list:
      try:
        key, value_str = item.split('=', 1)
        key = key.strip()
        try:
            # Safely evaluate the value string to its Python type
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If literal_eval fails, treat it as a raw string
            value = value_str
        params[key] = value
      except ValueError:
        print(f"Warning: Skipping invalid hyperparameter format: '{item}'. Expected 'key=value'.", file=sys.stderr)
    return params

def _load_config_from_file(config_path: str) -> dict:
    """Loads configuration from a YAML or JSON file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            if config_path.lower().endswith('.yaml') or config_path.lower().endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif config_path.lower().endswith('.json'):
                return json.load(f) or {}
            else:
                print(f"Warning: Unknown config file format for '{config_path}'. Only .yaml, .yml, or .json are supported. Ignoring.", file=sys.stderr)
                return {}
    except Exception as e:
        print(f"Warning: Error loading config file '{config_path}': {e}. Ignoring.", file=sys.stderr)
        return {}

def _apply_config_to_flags(config: dict, flags_obj):
    """Applies loaded configuration values to flags, respecting CLI precedence.

    Iterates through the loaded config dictionary. For each key that corresponds
    to a defined flag, if that flag was *not* set explicitly on the command line,
    its value is updated from the config file.

    Args:
        config: The dictionary loaded from the config file.
        flags_obj: The Abseil FLAGS object.
    """
    print("Applying config values to flags (respecting CLI precedence)...")
    flags_dict = flags_obj.flag_values_dict()
    applied_count = 0
    ignored_count = 0

    for key, value in config.items():
        if key in flags_dict:
            flag = flags_obj[key]
            # Check if the flag was set from the command line or code
            # Note: This check isn't perfect for multi_strings as they are always parsed.
            # We will override multi_strings if the config provides a list.
            is_multi_string = isinstance(flag.value, list) and flag.value != flag.default

            if not flag.present and not is_multi_string:
                try:
                    # Special handling for multi_string flags if config provides a list
                    if isinstance(flags_obj.find_flag_values_object(key).value, list) and isinstance(value, list):
                         # Clear default/existing value and set from config
                         # flags_obj.__setattr__(key, []) # This seems difficult with absl
                         # For multi_string, append might be safer, but override is intended
                         # Let's try setting directly, might require specific handling
                         print(f"  Overriding multi_string '{key}' from config: {value}")
                         flags_obj.__setattr__(key, value)
                         applied_count += 1
                    elif not isinstance(flags_obj.find_flag_values_object(key).value, list):
                         # For regular flags, set the value
                         print(f"  Setting flag '{key}' from config: {value}")
                         flags_obj.__setattr__(key, value)
                         applied_count += 1
                    else:
                         # Don't override multi_string with non-list from config
                         print(f"  Ignoring config value for multi_string '{key}' because config value is not a list.")
                         ignored_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to set flag '{key}' from config value '{value}'. Error: {e}", file=sys.stderr)
                    ignored_count += 1
            elif flag.present:
                print(f"  Flag '{key}' was set via command line. Ignoring config value '{value}'.")
                ignored_count += 1
            elif is_multi_string:
                 print(f"  Flag '{key}' is a multi_string set by default/code. Ignoring config value '{value}'.") # Decide if override needed
                 ignored_count += 1
        else:
            print(f"  Warning: Key '{key}' from config file does not match any defined flag. Ignoring.", file=sys.stderr)
            ignored_count += 1
    print(f"Config application complete. Applied: {applied_count}, Ignored/Skipped: {ignored_count}")

# Mapping algorithm names to their class paths
# TODO: Make this more robust or discoverable if possible
ALGORITHM_CLASS_PATHS = {
    "tabular_qlearner": "open_spiel.python.algorithms.tabular_qlearner.QLearner",
    "dqn": "open_spiel.python.algorithms.dqn.DQN",
    "ppo": "open_spiel.python.pytorch.ppo.PPO",
}

def main(_):
  """Main training script execution."""

  # --- Load Config File (before parsing flags) ---
  # We need to parse the config_file flag *itself* first.
  # Abseil doesn't easily allow parsing just one flag early.
  # A common workaround is to check sys.argv directly, but this is brittle.
  # For simplicity here, we'll load config *after* initial flag parsing,
  # acknowledging that the config file path itself cannot be set *in* the config file.
  config_from_file = {}
  if FLAGS.config_file:
      print(f"Loading configuration from: {FLAGS.config_file}")
      config_from_file = _load_config_from_file(FLAGS.config_file)
      print(f"  Config loaded: {config_from_file}")

  # --- Apply Config to Flags ---
  _apply_config_to_flags(config_from_file, FLAGS)

  # Now, flag values reflect the combination of defaults, config file, and CLI overrides.
  # Proceed with using FLAGS as usual.

  # --- Setup TensorBoard Writer (if logdir is provided) ---
  tb_writer = None
  if FLAGS.tensorboard_logdir:
      try:
          tb_writer = SummaryWriter(log_dir=FLAGS.tensorboard_logdir)
          print(f"TensorBoard logging enabled. Log directory: {FLAGS.tensorboard_logdir}")
      except Exception as e:
          print(f"Warning: Could not initialize TensorBoard SummaryWriter at '{FLAGS.tensorboard_logdir}'. Error: {e}", file=sys.stderr)
          tb_writer = None # Disable if initialization fails

  if FLAGS.list_algorithms:
    # Simple listing, could be made more dynamic if needed
    print("Available algorithms (supported by this script):")
    for name in ALGORITHM_CLASS_PATHS:
        print(f"- {name}")
    return # Exit cleanly after listing

  print(f"--- Training Configuration ---")
  print(f"Game: {FLAGS.game_name}")
  print(f"Algorithm: {FLAGS.algorithm_name}")
  print(f"Number of Episodes: {FLAGS.num_episodes}")
  print(f"Logging to: {FLAGS.log_file or 'Disabled'}")
  print(f"Checkpointing to: {FLAGS.checkpoint_dir or 'Disabled'} (every {FLAGS.checkpoint_every if FLAGS.checkpoint_dir and FLAGS.checkpoint_every > 0 else 'N/A'} episodes)")
  print(f"-----------------------------")

  # --- 1. Load the Game ---
  print(f"Loading game '{FLAGS.game_name}'...")
  game = pyspiel.load_game(FLAGS.game_name)
  print("Game loaded.")

  # --- 2. Create the RL Environment ---
  print("Creating RL environment...")
  env = rl_environment.Environment(game)
  if FLAGS.use_vector_env:
       if FLAGS.num_envs <= 0:
            raise ValueError("num_envs must be positive when use_vector_env is True.")
       # Create a list of environment factory functions
       env_fns = [lambda: rl_environment.Environment(FLAGS.game_name)] * FLAGS.num_envs
       env = SyncVectorEnv(env_fns)
       print(f"Using SyncVectorEnv with {FLAGS.num_envs} environments.")
  else:
       env = rl_environment.Environment(game)
       print("Using standard single RL environment.")

  # Get specs from the environment (works for both single and vector env)
  num_players = env.num_players
  num_actions = env.action_spec()["num_actions"]
  env_specs = {
      "num_actions": num_actions,
      "observation_spec": env.observation_spec(),
      "action_spec": env.action_spec(),
      # Add game name for potential use during loading/validation
      "game_name": FLAGS.game_name,
  }
  # Modify print statement slightly for clarity
  env_type = "SyncVectorEnv" if FLAGS.use_vector_env else "Single Env"
  print(f"Environment created ({env_type}): num_players={num_players}, num_actions={num_actions}")

  # --- 3. Parse Hyperparameters ---
  agent_hparams = _parse_hparams(FLAGS.agent_hparams)
  print(f"Parsed agent hyperparameters: {agent_hparams}")

  # --- 4. Create Agents using the Wrapper ---
  print(f"Creating agent wrappers for algorithm '{FLAGS.algorithm_name}'...")
  agents_wrapped = []
  try:
      agent_class_path = ALGORITHM_CLASS_PATHS.get(FLAGS.algorithm_name)
      if not agent_class_path:
          raise ValueError(f"Unsupported algorithm: {FLAGS.algorithm_name}. Use --list_algorithms to see options.")

      # Prepare base hparams (filtered from CLI) - specific checks done in wrapper
      base_hparams = agent_hparams.copy()
      # Add device flag to hparams if relevant for the agent (Task 17)
      if FLAGS.algorithm_name in ["dqn", "ppo"]: # Add other DL agents here
          base_hparams['device'] = FLAGS.device
          print(f"  Adding device='{FLAGS.device}' to base hparams for {FLAGS.algorithm_name}")

      # DQN needs TF session managed externally if restoring/sharing
      tf_session = None
      if FLAGS.algorithm_name == "dqn":
          tf.disable_eager_execution() # Ensure TF1 behavior
          tf_session = tf.Session()
          base_hparams['session'] = tf_session # Pass session to wrapper/agent
          print("  Created TF Session for DQN.")

      # PPO might need device handling here
      # ...

      # Instantiate wrappers
      for idx in range(num_players):
          print(f"  Creating wrapper for Player {idx}...")
          # The wrapper handles passing necessary env specs and hparams
          wrapper = agent_serialization.SerializableAgentWrapper(
              agent_class_path=agent_class_path,
              player_id=idx,
              env_specs=env_specs,
              agent_hparams=base_hparams
          )
          agents_wrapped.append(wrapper)
          print(f"    Wrapper created, underlying agent: {type(wrapper.agent).__name__}")

      # Initialize TF variables if a session was created for DQN
      # This is done inside the wrapper now if it creates the session.
      # if tf_session:
      #     tf_session.run(tf.global_variables_initializer())
      #     print("  Initialized TF global variables.")

      print(f"Agent wrappers created: {len(agents_wrapped)} agent(s) of type {FLAGS.algorithm_name}")

  except Exception as e:
      # Catch errors during agent wrapper creation
      print(f"Error creating agent wrappers: {e}", file=sys.stderr)
      # Clean up TF session if created
      if tf_session:
           tf_session.close()
      print("Please check the algorithm name and hyperparameters.", file=sys.stderr)
      return # Exit if agents couldn't be created


  # Use the wrapped agents for the training loop
  agents_to_use_in_loop = agents_wrapped

  # --- 5. Training Loop ---
  print(f"Starting training loop for {FLAGS.num_episodes} episodes...")
  metrics = [] # Store metrics if logging enabled
  log_writer = None
  log_file_handle = None

  # Setup logging if log_file is provided
  if FLAGS.log_file:
    try:
      log_file_handle = open(FLAGS.log_file, 'w', newline='')
      log_writer = csv.writer(log_file_handle)
      # Write header row based on the number of players
      log_header = ["episode"] + [f"player_{p}_reward" for p in range(num_players)]
      log_writer.writerow(log_header)
      print(f"Logging metrics to: {FLAGS.log_file}")
    except IOError as e:
      print(f"Warning: Could not open log file '{FLAGS.log_file}'. Logging disabled. Error: {e}", file=sys.stderr)
      log_writer = None # Disable logging if file fails to open
      log_file_handle = None

  # Select loop based on environment type
  if not FLAGS.use_vector_env:
      # --- Standard Single-Environment Training Loop ---
      print(f"Running standard loop for {FLAGS.num_episodes} episodes...")
      for ep in range(FLAGS.num_episodes):
        # --- Run one training episode (Multi-Agent Self-Play) ---
        time_step = env.reset()
        episode_rewards = [0.0] * num_players # Initialize rewards for this episode
        episode_losses = {p: [] for p in range(num_players)} # Track losses per player within episode

        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent = agents_to_use_in_loop[player_id] # Get the current player's agent

            # Agent takes a step
            agent_output = agent.step(time_step, is_evaluation=False)

            if agent_output is None:
                 raise ValueError(f"Agent {player_id} returned None output during training step.")
            action_list = [agent_output.action]
            time_step = env.step(action_list)

            if time_step.rewards:
                for p in range(num_players):
                    episode_rewards[p] += time_step.rewards[p]

            for ag_idx, ag in enumerate(agents_to_use_in_loop):
                ag.step(time_step, is_evaluation=False)
                # Capture loss after agent step if available
                agent_loss = ag.loss
                if agent_loss is not None:
                    episode_losses[ag_idx].append(agent_loss)

        # End of episode learning step
        for ag_idx, ag in enumerate(agents_to_use_in_loop):
            ag.step(time_step, is_evaluation=False)
            agent_loss = ag.loss
            if agent_loss is not None:
                episode_losses[ag_idx].append(agent_loss)

        # --- Logging (Single Env) ---
        avg_episode_losses = [sum(episode_losses[p]) / len(episode_losses[p]) if episode_losses[p] else None for p in range(num_players)]
        if log_writer:
          log_row = [ep + 1] + episode_rewards
          # Add average losses to CSV if calculated
          # log_row += [f"{l:.4f}" if l is not None else "" for l in avg_episode_losses]
          log_writer.writerow(log_row)
          metrics.append({"episode": ep + 1, "rewards": episode_rewards, "avg_losses": avg_episode_losses})

        if tb_writer:
            for p_id, reward in enumerate(episode_rewards):
                tb_writer.add_scalar(f'Reward/Player_{p_id}', reward, ep + 1)
            for p_id, avg_loss in enumerate(avg_episode_losses):
                if avg_loss is not None:
                     tb_writer.add_scalar(f'Loss/Player_{p_id}', avg_loss, ep + 1)

        if (ep + 1) % 100 == 0:
            reward_str = ", ".join([f"P{i}: {r:.2f}" for i, r in enumerate(episode_rewards)])
            loss_str = ", ".join([f"P{i}: {l:.4f}" if l else "N/A" for i, l in enumerate(avg_episode_losses)])
            print(f"  Episode {ep + 1}/{FLAGS.num_episodes}. Rewards: [{reward_str}]. Avg Losses: [{loss_str}]")
            if log_file_handle:
              log_file_handle.flush()

        # --- Checkpoint Saving (Single Env) ---
        if FLAGS.checkpoint_dir and FLAGS.checkpoint_every > 0 and (ep + 1) % FLAGS.checkpoint_every == 0:
          if not os.path.exists(FLAGS.checkpoint_dir):
               try:
                    os.makedirs(FLAGS.checkpoint_dir)
                    print(f"Created checkpoint directory: {FLAGS.checkpoint_dir}")
               except OSError as e:
                    print(f"Warning: Could not create checkpoint directory '{FLAGS.checkpoint_dir}'. Checkpointing disabled for this episode. Error: {e}", file=sys.stderr)
                    continue

          for agent_wrapper in agents_to_use_in_loop:
               try:
                    agent_path = os.path.join(FLAGS.checkpoint_dir, f"agent_p{agent_wrapper.player_id}_ep{ep+1}")
                    agent_wrapper.save(agent_path)
               except Exception as e:
                    print(f"Warning: Failed to save checkpoint for player {agent_wrapper.player_id} using wrapper. Error: {e}", file=sys.stderr)

  else: # FLAGS.use_vector_env is True
        # --- Vectorized Environment Training Loop ---
        print(f"Running vectorized loop for {FLAGS.num_episodes} episodes across {FLAGS.num_envs} environments...")
        num_envs = FLAGS.num_envs
        total_episodes_target = FLAGS.num_episodes
        episodes_completed = 0
        total_steps = 0 # Track total environment steps across all envs

        # Track rewards per environment for the current *ongoing* episode
        current_episode_rewards = [[0.0] * num_players for _ in range(num_envs)]
        # Store rewards of completed episodes for logging averages
        completed_episode_rewards_all = []
        # Track current step losses per environment? Or maybe just overall average?
        # For simplicity, let's track loss across the batch at each step

        time_steps = env.reset() # Returns list of TimeStep objects

        while episodes_completed < total_episodes_target:
            total_steps += num_envs # Increment step count

            # --- Agent Step (Requires Agent Modification - Task 18d) ---
            # Placeholder logic - needs agent batch support
            batch_actions = []
            current_players = [ts.observations["current_player"] for ts in time_steps]
            # In a real implementation, this would likely be a single call:
            # batch_agent_outputs = agents.step_batch(time_steps, is_evaluation=False)
            # For now, iterate (INEFFICIENT):
            for i, ts in enumerate(time_steps):
                 player_id = current_players[i]
                 if player_id >= 0:
                      agent = agents_to_use_in_loop[player_id]
                      # This step needs modification to handle batched input (Task 18d)
                      agent_output = agent.step(ts, is_evaluation=False)
                      if agent_output is None: raise ValueError("Agent returned None")
                      batch_actions.append(agent_output.action)
                 else:
                      # Handle cases where an environment might start terminal/chance (unlikely for SyncVectorEnv reset)
                      batch_actions.append(None) # Should match env expectation

            # --- Environment Step --- Returns list of TimeStep
            next_time_steps = env.step(batch_actions)

            # --- Agent Learning Step (Handles Batches - Task 18e) ---
            # Assumes agents can handle batch learning internally (e.g., PPO) or via a specific method.
            current_batch_losses = {p: [] for p in range(num_players)}
            for p_id, agent in enumerate(agents_to_use_in_loop):
                # Option 1: Agent handles learning internally during its step/step_batch method.
                #           In this case, this block might just be for gathering losses.
                # Option 2: Agent has an explicit batch learning method.
                if hasattr(agent, 'learn_batch'):
                    # Pass the relevant information for batch learning
                    # This might include current time_steps, actions taken, next_time_steps
                    # The exact signature depends on the agent implementation.
                    # Placeholder call:
                    loss = agent.learn_batch(time_steps, batch_actions, next_time_steps)
                    if loss is not None:
                         current_batch_losses[p_id].append(loss) # Or handle batch loss
                else:
                    # Fallback: If learn_batch not present, assume learning happened in step()
                    # (as handled by the wrapper's step fallback for Q/DQN)
                    # We just gather the loss reported by the agent property.
                    agent_loss = agent.loss
                    if agent_loss is not None:
                        # This might be a single value even after processing a batch,
                        # representing avg loss over the batch, or the most recent loss.
                        current_batch_losses[p_id].append(agent_loss)

            # --- Reward Accumulation and Episode Tracking ---
            for i in range(num_envs):
                # Accumulate rewards for the ongoing episode in environment i
                if next_time_steps[i].rewards:
                     for p in range(num_players):
                          current_episode_rewards[i][p] += next_time_steps[i].rewards[p]

                # Check if environment i finished an episode
                if next_time_steps[i].last():
                    episodes_completed += 1
                    completed_episode_rewards_all.append(current_episode_rewards[i])

                    # Log completed episode reward to CSV
                    if log_writer:
                         log_row = [episodes_completed] + current_episode_rewards[i]
                         log_writer.writerow(log_row)

                    # Reset rewards for this environment index for the new episode
                    current_episode_rewards[i] = [0.0] * num_players

                    # Print progress message occasionally
                    if episodes_completed % (num_envs * 5) == 0: # Adjust frequency as needed
                         print(f"  Completed {episodes_completed}/{total_episodes_target} episodes across all environments...")

                    # --- Checkpointing (Vectorized) --- (Based on completed episodes)
                    checkpoint_freq = FLAGS.checkpoint_every # Checkpoint every N *completed* episodes
                    if FLAGS.checkpoint_dir and checkpoint_freq > 0 and episodes_completed % checkpoint_freq == 0:
                        print(f"Checkpointing agents at {episodes_completed} completed episodes...")
                        if not os.path.exists(FLAGS.checkpoint_dir):
                           try: os.makedirs(FLAGS.checkpoint_dir); print(f"Created checkpoint directory: {FLAGS.checkpoint_dir}")
                           except OSError as e: print(f"Warning: Could not create checkpoint directory. Checkpointing skipped. Error: {e}", file=sys.stderr); continue
                        for agent_wrapper in agents_to_use_in_loop:
                           try:
                               # Suffix checkpoint with total episodes completed
                               agent_path = os.path.join(FLAGS.checkpoint_dir, f"agent_p{agent_wrapper.player_id}_ep{episodes_completed}")
                               agent_wrapper.save(agent_path)
                           except Exception as e:
                               print(f"Warning: Failed to save checkpoint for player {agent_wrapper.player_id}. Error: {e}", file=sys.stderr)

            # --- Logging (Vectorized) --- (Based on recent episodes / steps)
            log_interval_steps = 1000 * num_envs # Example: Log every 1000 steps per env average
            if tb_writer and total_steps % log_interval_steps == 0 and total_steps > 0:
                 # Calculate average reward from last N completed episodes
                 last_n = min(len(completed_episode_rewards_all), num_envs * 10) # Log avg over last 10*N eps
                 if last_n > 0:
                      recent_rewards = completed_episode_rewards_all[-last_n:]
                      for p in range(num_players):
                           avg_reward = sum(r[p] for r in recent_rewards) / last_n
                           tb_writer.add_scalar(f'Reward/AvgPlayer_{p}', avg_reward, total_steps)

                 # Calculate average loss from the last batch
                 for p_id, losses in current_batch_losses.items():
                      if losses:
                           avg_loss = sum(losses) / len(losses)
                           tb_writer.add_scalar(f'Loss/AvgPlayer_{p_id}', avg_loss, total_steps)

                 if log_file_handle: # Flush CSV periodically
                     log_file_handle.flush()

            # Prepare for next iteration
            time_steps = next_time_steps

  print("--- Training Finished ---")

  # Close the TensorBoard writer
  if tb_writer:
      tb_writer.close()

  # Close the log file if it was opened
  if log_file_handle:
    log_file_handle.close()
    print(f"Training metrics saved to {FLAGS.log_file}")

  # Note: Checkpoints are saved progressively during the loop.

if __name__ == "__main__":
  # Standard way to run an Abseil app
  app.run(main)