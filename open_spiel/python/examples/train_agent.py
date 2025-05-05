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
from open_spiel.python import rl_agent # Import rl_agent module
from open_spiel.python.algorithms import tabular_qlearner, tabular_qlearner_long_narde # Import specific modules

# Conditional imports based on algorithm
try:
    import tensorflow.compat.v1 as tf
    # Need to disable TensorFlow 2.x behavior when using V1 code.
    tf.disable_v2_behavior()
    tf_available = True
except ImportError:
    tf = None # Set tf to None if TensorFlow is not available
    tf_available = False
    print("TensorFlow not available. DQN algorithm will not work.")

try:
    import torch
    pytorch_available = True
    from open_spiel.python.algorithms.ppo import PPOAgent # Import PPOAgent if torch available
    # --- Stochastic MuZero Imports --- #
    from open_spiel.python.algorithms.stochastic_muzero import StochasticMuZero, map_dice_to_index
    from open_spiel.python.algorithms.stochastic_muzero_config import StochasticMuZeroConfig, new_long_narde_config
    from open_spiel.python.algorithms.stochastic_muzero_nets import muzero_network_factory
    # --- End Stochastic MuZero Imports --- #
except ImportError:
    torch = None # Set torch to None if PyTorch is not available
    pytorch_available = False
    print("PyTorch not available. PPO and StochasticMuZero algorithms will not work.")
    PPOAgent = None # Define as None if PyTorch unavailable
    StochasticMuZero = None
    StochasticMuZeroConfig = None
    new_long_narde_config = None
    muzero_network_factory = None
    map_dice_to_index = None

# Import DQN agents separately if TF is available
if tf_available:
    from open_spiel.python.algorithms import dqn, dqn_long_narde
else:
    dqn = None
    dqn_long_narde = None

from open_spiel.python.utils import training # Keep for potential future use
import csv # For logging
import os # For path manipulation
import sys # For checking agent types
import ast # For parsing hyperparameters
import yaml # Added for config file loading
import json # Added for config file loading
from tensorboardX import SummaryWriter # Added for TensorBoard logging
from open_spiel.python.vector_env import SyncVectorEnv # Added for Task 18
import cProfile
import pstats
import atexit
import time
import numpy as np
import collections
import datetime
import logging
from typing import List, Optional, Type, Dict, Any

FLAGS = flags.FLAGS

# --- Core Training Parameters ---
flags.DEFINE_string("game_name", "tic_tac_toe", "Name of the game to train on.")
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
flags.DEFINE_enum("algorithm_name", "tabular_qlearner",
                  # Dynamically create the list based on available imports
                  ([name for name in ["tabular_qlearner", "tabular_qlearner_long_narde"]]
                   + [name for name in ["dqn", "dqn_long_narde"] if tf_available]
                   + [name for name in ["ppo", "stochastic_muzero"] if pytorch_available]),
                  "Name of the RL algorithm to use.")

# --- Agent Hyperparameters ---
# Shortened help string
flags.DEFINE_list("agent_hparams", [], "Agent hyperparameters as key=value pairs.")

# --- Utilities ---
flags.DEFINE_boolean("list_algorithms", False, "List available algorithms and exit.")

# --- Logging and Checkpointing ---
flags.DEFINE_string("log_file", None, "Path to save training metrics (CSV). If None, logging is disabled.")
flags.DEFINE_string("checkpoint_dir", "/tmp/os_checkpoints",
                     "Directory to save agent checkpoints. If None, checkpointing is disabled.")
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

profiler = cProfile.Profile()
profiler.enable()
def print_stats():
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)
atexit.register(print_stats)

# Algorithm mapping using the actual imported classes
ALGORITHMS = {
    "tabular_qlearner": tabular_qlearner.QLearner,
    "tabular_qlearner_long_narde": tabular_qlearner_long_narde.QLearnerLongNarde,
}
if tf_available and dqn and dqn_long_narde:
    ALGORITHMS["dqn"] = dqn.DQN
    ALGORITHMS["dqn_long_narde"] = dqn_long_narde.DQNLongNarde
if pytorch_available and PPOAgent and StochasticMuZero:
    ALGORITHMS["ppo"] = PPOAgent # Use the specific agent class
    ALGORITHMS["stochastic_muzero"] = StochasticMuZero

# Removed ALGORITHM_CLASS_PATHS

def _parse_hparams(hparam_list: list[str]) -> dict:
    """Parses a list of 'key=value' strings into a dictionary with typed values."""
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
    """Applies loaded configuration values to flags, respecting CLI precedence."""
    print("Applying config values to flags (respecting CLI precedence)...")
    flags_dict = flags_obj.flag_values_dict()
    applied_count = 0
    ignored_count = 0
    for key, value in config.items():
        if key in flags_dict:
            flag = flags_obj[key]
            is_multi_string = isinstance(flag.value, list) and flag.value != flag.default
            if not flag.present and not is_multi_string:
                try:
                    if isinstance(flags_obj.find_flag_values_object(key).value, list) and isinstance(value, list):
                         print(f"  Overriding multi_string '{key}' from config: {value}")
                         flags_obj.__setattr__(key, value)
                         applied_count += 1
                    elif not isinstance(flags_obj.find_flag_values_object(key).value, list):
                         print(f"  Setting flag '{key}' from config: {value}")
                         flags_obj.__setattr__(key, value)
                         applied_count += 1
                    else:
                         print(f"  Ignoring config value for multi_string '{key}' because config value is not a list.")
                         ignored_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to set flag '{key}' from config value '{value}'. Error: {e}", file=sys.stderr)
                    ignored_count += 1
            elif flag.present:
                print(f"  Flag '{key}' was set via command line. Ignoring config value '{value}'.")
                ignored_count += 1
            elif is_multi_string:
                 print(f"  Flag '{key}' is a multi_string set by default/code. Ignoring config value '{value}'.")
                 ignored_count += 1
        else:
            print(f"  Warning: Key '{key}' from config file does not match any defined flag. Ignoring.", file=sys.stderr)
            ignored_count += 1
    print(f"Config application complete. Applied: {applied_count}, Ignored/Skipped: {ignored_count}")

def _train_agent_single_process(agent, env, num_episodes):
    """Trains a single agent instance.
    Handles storing chance outcomes for Stochastic MuZero.
    """
    total_steps = 0
    total_loss = 0.0
    num_episodes_completed = 0

    is_smz = isinstance(agent, StochasticMuZero)

    for ep in range(num_episodes):
        time_step = env.reset()
        cumulative_reward = 0.0
        steps_in_episode = 0
        agent.reset() # Assuming agent has a reset method

        while not time_step.last():
            current_player = time_step.current_player()
            agent_output = agent.step(time_step)
            action = agent_output.action

            # Store data associated *before* this action
            if not agent.is_evaluation:
                agent.store_step_data(time_step, action, agent_output)

            # Environment step
            time_step = env.step([action])
            total_steps += 1
            steps_in_episode += 1
            cumulative_reward += time_step.rewards[agent.player_id]

            # --- Store chance outcome for the *next* state --- #
            # The chance outcome c_{k+1} occurs *after* action a_k leading to state s_{k+1}
            if not agent.is_evaluation and is_smz:
                chance_outcome_index = -1 # Default/invalid
                # Check if the game state has dice info (specific to Backgammon/Narde)
                if hasattr(env.get_state(), 'dice'):
                    dice = env.get_state().dice() # Should be (d1, d2)
                    if dice and len(dice) == 2:
                        try:
                            chance_outcome_index = map_dice_to_index(dice[0], dice[1])
                        except ValueError as e:
                            print(f"[WARN] Invalid dice value in state: {dice}. Error: {e}")
                # Add this chance outcome to the last stored step data
                agent.add_chance_outcome_to_trajectory(chance_outcome_index)
            # --- End Chance Outcome --- #

            # Check if learning is possible
            if not agent.is_evaluation and agent.is_ready_to_learn():
                 loss = agent.learn()
                 if loss is not None:
                     total_loss += loss

        # End of episode
        num_episodes_completed += 1
        if not agent.is_evaluation:
            agent.finalize_trajectory(time_step)
            # Add episode stats logging here

        # TODO: Checkpointing logic

    return total_steps, total_loss / total_steps if total_steps > 0 else 0.0

def main(_):
  """Main training script execution."""
  # --- Load Config File --- #
  config_from_file = {}
  if FLAGS.config_file:
      print(f"Loading configuration from: {FLAGS.config_file}")
      config_from_file = _load_config_from_file(FLAGS.config_file)
      print(f"  Config loaded: {config_from_file}")
  _apply_config_to_flags(config_from_file, FLAGS)

  # --- Setup TensorBoard --- #
  tb_writer = None
  if FLAGS.tensorboard_logdir:
      try:
          tb_writer = SummaryWriter(log_dir=FLAGS.tensorboard_logdir)
          print(f"TensorBoard logging enabled. Log directory: {FLAGS.tensorboard_logdir}")
      except Exception as e:
          print(f"Warning: Could not initialize TensorBoard SummaryWriter at '{FLAGS.tensorboard_logdir}'. Error: {e}", file=sys.stderr)
          tb_writer = None

  if FLAGS.list_algorithms:
    print("Available algorithms (supported by this script):")
    for name in ALGORITHMS:
        print(f"- {name}")
    return

  # --- Algorithm Availability Check --- #
  if FLAGS.algorithm_name not in ALGORITHMS:
      logging.error(f"Algorithm '{FLAGS.algorithm_name}' is not available or its dependencies (TensorFlow/PyTorch) are missing.")
      logging.error(f"Available algorithms: {list(ALGORITHMS.keys())}")
      sys.exit(1)

  if FLAGS.algorithm_name == "stochastic_muzero" and FLAGS.use_vector_env:
        logging.error("Vector environment (--use_vector_env) is not yet supported for StochasticMuZero.")
        sys.exit(1)

  logging.basicConfig(level=logging.INFO)
  logging.info(f"--- Training Configuration ---")
  logging.info(f"Game: {FLAGS.game_name}")
  logging.info(f"Algorithm: {FLAGS.algorithm_name}")
  logging.info(f"Number of Episodes: {FLAGS.num_episodes}")
  logging.info(f"Device: {FLAGS.device}")
  logging.info(f"Use Vector Env: {FLAGS.use_vector_env} (Num Envs: {FLAGS.num_envs if FLAGS.use_vector_env else 'N/A'})")
  logging.info(f"Logging to: {FLAGS.log_file or 'Disabled'}")
  logging.info(f"Checkpointing to: {FLAGS.checkpoint_dir or 'Disabled'} (every {FLAGS.checkpoint_every if FLAGS.checkpoint_dir and FLAGS.checkpoint_every > 0 else 'N/A'} episodes)")
  logging.info(f"-----------------------------")

  # --- 1. Load the Game --- #
  print(f"Loading game '{FLAGS.game_name}'...")
  try:
    game = pyspiel.load_game(FLAGS.game_name)
    print("Game loaded.")
  except Exception as e:
      logging.error(f"Error loading game '{FLAGS.game_name}': {e}")
      sys.exit(1)

  num_players = game.num_players()
  temp_env_for_specs = rl_environment.Environment(game)
  action_spec = temp_env_for_specs.action_spec()
  observation_spec = temp_env_for_specs.observation_spec()
  num_actions = action_spec["num_actions"]
  del temp_env_for_specs

  # --- 2. Create the RL Environment --- #
  print("Creating RL environment...")
  if FLAGS.use_vector_env:
       if FLAGS.num_envs <= 0:
            raise ValueError("num_envs must be positive when use_vector_env is True.")
       env_instances = [rl_environment.Environment(game) for _ in range(FLAGS.num_envs)]
       try:
           env = SyncVectorEnv(env_instances)
           print(f"Using SyncVectorEnv with {FLAGS.num_envs} environments.")
       except Exception as e:
           logging.error(f"Error creating SyncVectorEnv: {e}")
           sys.exit(1)
  else:
       try:
           env = rl_environment.Environment(game)
           print("Using standard single RL environment.")
       except Exception as e:
            logging.error(f"Error creating Environment: {e}")
            sys.exit(1)

  env_specs = {
      "num_actions": num_actions,
      "observation_spec": observation_spec,
      "action_spec": action_spec,
      "game_name": FLAGS.game_name,
      "num_players": num_players
  }
  env_type = "SyncVectorEnv" if FLAGS.use_vector_env else "Single Env"
  print(f"Environment created ({env_type}): num_players={num_players}, num_actions={num_actions}")

  # --- 3. Parse Hyperparameters --- #
  agent_hparams = _parse_hparams(FLAGS.agent_hparams)
  print(f"Parsed agent hyperparameters: {agent_hparams}")

  # --- 4. Create Agents --- #
  print(f"Creating agents for algorithm '{FLAGS.algorithm_name}'...")
  agents: List[rl_agent.AbstractAgent] = []
  tf_sess = None

  try:
      agent_class: Type[rl_agent.AbstractAgent] = ALGORITHMS[FLAGS.algorithm_name]

      if FLAGS.algorithm_name in ["dqn", "dqn_long_narde"]:
          state_representation_size = env_specs["observation_spec"]["info_state"][0]
          tf_sess = tf.Session()
          for player_id in range(num_players):
              agent = agent_class(
                  session=tf_sess, player_id=player_id,
                  state_representation_size=state_representation_size,
                  num_actions=num_actions, **agent_hparams)
              agents.append(agent)
          tf_sess.run(tf.global_variables_initializer())
          print(f"Initialized DQN agents with TF session.")

      elif FLAGS.algorithm_name == "ppo":
          state_representation_size = env_specs["observation_spec"]["info_state"][0]
          for player_id in range(num_players):
              agent = agent_class(
                  player_id=player_id,
                  state_representation_size=state_representation_size,
                  num_actions=num_actions, device=FLAGS.device, **agent_hparams)
              agents.append(agent)
          print(f"Initialized PPO agents.")

      elif FLAGS.algorithm_name == "stochastic_muzero":
          if "long_narde" in FLAGS.game_name:
              config = new_long_narde_config()
          else:
              logging.warning(f"No default SMZ config for {FLAGS.game_name}. Using basic settings.")
              config = StochasticMuZeroConfig()

          config.game_name = FLAGS.game_name
          config.num_players = num_players
          config.observation_shape = env_specs["observation_spec"]["info_state"]
          config.state_representation_size = env_specs["observation_spec"]["info_state"][0]
          config.action_space_size = num_actions

          if muzero_network_factory is None:
              raise RuntimeError("Stochastic MuZero network factory is not available.")
          config.network_factory = lambda: muzero_network_factory(config)

          for key, value in agent_hparams.items():
              if hasattr(config, key):
                  if key == "known_bounds" and isinstance(value, (list, str)):
                      try:
                          if map_dice_to_index is None: raise RuntimeError("map_dice_to_index not imported.")
                          bounds_list = ast.literal_eval(value) if isinstance(value, str) else value
                          if isinstance(bounds_list, list) and len(bounds_list) == 2:
                              setattr(config, key, map_dice_to_index(bounds_list[0], bounds_list[1]))
                              logging.info(f"Overriding config.{key} = map_dice_to_index({bounds_list[0]}, {bounds_list[1]})")
                          else:
                              logging.warning(f"Could not parse known_bounds: {value}.")
                      except Exception as e:
                           logging.warning(f"Error parsing known_bounds hparam: {value}. Error: {e}")
                  else:
                      setattr(config, key, value)
                      logging.info(f"Overriding config.{key} = {value}")
              else:
                  logging.warning(f"Ignoring unknown hparam for StochasticMuZero: {key}")

          config.device = FLAGS.device
          config.checkpoint_dir = FLAGS.checkpoint_dir

          for player_id in range(num_players):
              agent = agent_class(player_id=player_id, config=config)
              agents.append(agent)
          logging.info(f"Initialized StochasticMuZero agents with Config: {config}")

      else: # Tabular QLearner variants
          # NOTE: Using direct instantiation for tabular now for consistency
          # Revert to SerializableAgentWrapper if needed for specific serialization features.
          state_representation_size = env_specs["observation_spec"]["info_state"][0]
          for player_id in range(num_players):
              agent = agent_class(
                  player_id=player_id,
                  state_representation_size=state_representation_size,
                  num_actions=num_actions,
                  **agent_hparams
              )
              agents.append(agent)
          print(f"Initialized Tabular agents directly.")

  except Exception as e:
      logging.error(f"Error initializing agent {FLAGS.algorithm_name}: {e}")
      import traceback
      traceback.print_exc()
      if tf_sess: tf_sess.close()
      sys.exit(1)

  # --- 5. Training Loop --- #
  print(f"Starting training loop for {FLAGS.num_episodes} episodes...")
  start_time = time.time()
  total_steps = 0
  episodes_completed = 0
  last_checkpoint_time = time.time()
  rolling_losses = {p: collections.deque(maxlen=1000) for p in range(num_players)}

  # Setup logging
  log_writer = None
  log_file_handle = None
  if FLAGS.log_file:
    try:
      log_file_handle = open(FLAGS.log_file, 'w', newline='')
      log_writer = csv.writer(log_file_handle)
      log_header = ["episode"] + [f"player_{p}_reward" for p in range(num_players)] + [f"player_{p}_avg_loss" for p in range(num_players)]
      log_writer.writerow(log_header)
      print(f"Logging metrics to: {FLAGS.log_file}")
    except IOError as e:
      print(f"Warning: Could not open log file '{FLAGS.log_file}'. Logging disabled. Error: {e}", file=sys.stderr)
      log_writer = None
      log_file_handle = None

  try:
      if not FLAGS.use_vector_env:
          # --- Standard Single-Environment Training Loop --- #
          print(f"Running standard loop for {FLAGS.num_episodes} episodes...")
          for ep in range(FLAGS.num_episodes):
            time_step = env.reset()
            episode_rewards = [0.0] * num_players
            current_episode_losses = {p: [] for p in range(num_players)}
            ep_step_counter = 0

            while not time_step.last():
                ep_step_counter += 1
                player_id = time_step.observations["current_player"]

                if player_id >= 0:
                    current_legal_actions = time_step.observations["legal_actions"][player_id]
                    if not current_legal_actions and not time_step.is_chance_node():
                        if "long_narde" not in FLAGS.game_name and "backgammon" not in FLAGS.game_name:
                             logging.warning(f"Episode {ep+1}, Step {ep_step_counter}: Player {player_id} has no legal actions in non-terminal/non-chance state.")
                        # Agent step should still be called to potentially handle forced pass

                    agent = agents[player_id]
                    agent_output = agent.step(time_step, is_evaluation=False)

                    action_list = [agent_output.action] if agent_output and agent_output.action is not None else []

                    agent_loss = agent.loss
                    if agent_loss is not None:
                        current_episode_losses[player_id].append(agent_loss)
                        rolling_losses[player_id].append(agent_loss)

                    time_step = env.step(action_list)
                else: # Chance node or terminal
                    time_step = env.step([])

                total_steps += 1 # Increment total steps
                if time_step.rewards:
                    for p in range(num_players):
                        episode_rewards[p] += time_step.rewards[p]

            # End of episode: Call step on the terminal time_step for all agents
            for agent in agents:
                agent.step(time_step, is_evaluation=False)
                final_loss = agent.loss
                if final_loss is not None:
                     current_episode_losses[agent.player_id].append(final_loss)
                     rolling_losses[agent.player_id].append(final_loss)

            episodes_completed += 1

            # --- Logging (Single Env) --- #
            avg_episode_losses = [sum(current_episode_losses[p]) / len(current_episode_losses[p]) if current_episode_losses[p] else None for p in range(num_players)]
            if log_writer:
              log_row = [ep + 1] + episode_rewards + [(f"{l:.6f}" if l is not None else "") for l in avg_episode_losses]
              log_writer.writerow(log_row)
            if tb_writer:
                for p_id, reward in enumerate(episode_rewards):
                    tb_writer.add_scalar(f'Reward/Player_{p_id}', reward, episodes_completed)
                for p_id, avg_loss in enumerate(avg_episode_losses):
                    if avg_loss is not None:
                         tb_writer.add_scalar(f'Loss/Player_{p_id}', avg_loss, episodes_completed)

            if episodes_completed % 100 == 0:
                reward_str = ", ".join([f"P{i}: {r:.2f}" for i, r in enumerate(episode_rewards)])
                print(f"Episodes {episodes_completed - 99}-{episodes_completed}: Last Ep Rewards: {reward_str}")
                print("  Loss Stats (rolling avg over ~last 1000 learning steps):")
                for p in range(num_players):
                    if rolling_losses[p]:
                        avg_loss = sum(rolling_losses[p]) / len(rolling_losses[p])
                        min_loss = min(rolling_losses[p])
                        max_loss = max(rolling_losses[p])
                        print(f"    Player {p}: Avg: {avg_loss:.4f}, Min: {min_loss:.4f}, Max: {max_loss:.4f} (n={len(rolling_losses[p])})")
                    else:
                        print(f"    Player {p}: No loss data recorded yet.")
                if log_file_handle:
                  log_file_handle.flush()

            # --- Checkpoint Saving (Single Env) --- #
            if FLAGS.checkpoint_dir and FLAGS.checkpoint_every > 0 and episodes_completed % FLAGS.checkpoint_every == 0:
              current_time = time.time()
              logging.info(f"Episode {episodes_completed}. Saving checkpoint... "
                           f"(Time since last chkpt: {current_time - last_checkpoint_time:.1f}s)")
              if not os.path.exists(FLAGS.checkpoint_dir):
                   try: os.makedirs(FLAGS.checkpoint_dir); print(f"Created checkpoint directory: {FLAGS.checkpoint_dir}")
                   except OSError as e: print(f"Warning: Could not create checkpoint directory. Error: {e}", file=sys.stderr); continue

              for agent in agents:
                   try:
                        checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, f"agent_p{agent.player_id}_ep{episodes_completed}")
                        # Call save method directly on the agent
                        agent.save(checkpoint_prefix)
                   except Exception as e:
                        print(f"Warning: Failed to save checkpoint for player {agent.player_id}. Error: {e}", file=sys.stderr)
              last_checkpoint_time = current_time
              logging.info("Checkpoint saved.")

      else: # FLAGS.use_vector_env is True
            # --- Vectorized Environment Loop --- #
            # (Currently disallowed for SMZ, needs agent-specific implementation)
            logging.error("Vector env loop is not implemented for this agent configuration.")
            # Previously implemented vector loop was specific to DQN/ReplayBuffer interaction
            pass

  except KeyboardInterrupt:
      logging.info("Training interrupted by user.")
  except Exception as e:
      logging.error(f"Error during training loop: {e}")
      import traceback
      traceback.print_exc()
  finally:
      if tf_sess: # Close TF session if it exists
          tf_sess.close()
          logging.info("TensorFlow session closed.")

      logging.info(f"Training finished after {episodes_completed} episodes.")
      elapsed_time = time.time() - start_time
      logging.info(f"Total steps (approx single env equivalent): {total_steps}")
      logging.info(f"Total time: {elapsed_time:.2f}s")

      # Final checkpoint save
      if FLAGS.checkpoint_dir and episodes_completed > 0:
          logging.info("Saving final checkpoint...")
          for agent in agents:
              checkpoint_prefix = os.path.join(
                  FLAGS.checkpoint_dir,
                  f"agent_p{agent.player_id}_ep{episodes_completed}_final"
              )
              try:
                  agent.save(checkpoint_prefix)
              except Exception as e:
                  logging.error(f"Error saving final agent {agent.player_id} checkpoint: {e}")
          logging.info("Final checkpoint saved.")
      else:
          logging.info("Skipping final checkpoint save (checkpoint_dir not set or no episodes completed).")

      if tb_writer:
          tb_writer.close()
      if log_file_handle:
        log_file_handle.close()
        print(f"Training metrics saved to {FLAGS.log_file}")

if __name__ == "__main__":
  app.run(main)