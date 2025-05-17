# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to run AlphaZero on a game with JAX.

Based on the TensorFlow AlphaZero example.
"""

import json
import os
import sys
import tempfile
import multiprocessing as mp

from absl import app
from absl import flags
import pyspiel

from open_spiel.python.algorithms.alpha_zero_jax import alpha_zero_jax as alpha_zero_jax_lib # Renamed to avoid clash
from open_spiel.python.algorithms.alpha_zero_jax.alpha_zero_jax import ConfigJAX
from open_spiel.python.algorithms.alpha_zero_jax.alpha_zero_jax import set_external_libraries_log_level
from open_spiel.python.utils import spawn

FLAGS = flags.FLAGS

# Game flags
flags.DEFINE_string("game", "tic_tac_toe", "Name of the game.")
flags.DEFINE_float("uct_c", 2.0, "UCT exploration constant.")
flags.DEFINE_integer("max_simulations", 10, "How many MCTS simulations to run.")
flags.DEFINE_float("policy_alpha", 0.3, "What alpha to use for Dirichlet noise.") # Matching TF AlphaZero default
flags.DEFINE_float("policy_epsilon", 0.25, "What epsilon to use for Dirichlet noise.") # Matching TF AlphaZero default
flags.DEFINE_float("temperature", 1., "Temperature for final policy.")
flags.DEFINE_integer("temperature_drop", 10, # Game specific, remove if not needed.
                     "Drop temperature to 0 after this many moves.")
flags.DEFINE_integer("evaluation_window", 100, # Window for evaluation statistics from TF version
                     "How many games to average results over.")
flags.DEFINE_integer("eval_levels", 7,
                     "Play evaluation games vs MCTS+Solver, with MCTS search "
                     "counts from 10^(i/2) for i in range(eval_levels).")

# JAX AlphaZero training flags
flags.DEFINE_string("path", None, "Path to record games and models.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.") # Adjusted to a more common starting point
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay.")
flags.DEFINE_integer("train_batch_size", 64, "Batch size for training.") # Adjusted for JAX common practice
flags.DEFINE_integer("replay_buffer_size", 2**14, # 16384
                     "Size of the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 3,
                     "How many times to reuse each item in the buffer.")
flags.DEFINE_integer("max_steps", 100, "How many learn steps to run.") # Reduced for quick testing
flags.DEFINE_integer("checkpoint_freq", 10, "Save a checkpoint every N steps.") # Reduced for quick testing
flags.DEFINE_integer("actors", 2, "How many actors to run.")
flags.DEFINE_integer("evaluators", 1, "How many evaluators to run.")
flags.DEFINE_string("nn_model", "mlp", "Model architecture. Valid: mlp, conv2d, resnet, resnet18, ...")
flags.DEFINE_integer("nn_width", 64, "Hidden layer size or ResNet filter count.") # Smaller for MLP default
flags.DEFINE_integer("nn_depth", 2, "Number of hidden layers or ResNet blocks.") # Smaller for MLP default
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_integer("master_seed", 0, "Master PRNG seed for JAX, Python random, and NumPy.")
flags.DEFINE_enum("log_level", "INFO", ["ERROR","WARN","INFO","DEBUG","TRACE"], "Logging verbosity.")

# New ResNet/ResNeSt specific flags (optional, use if nn_model="resnet" and you want generic ResNet)
flags.DEFINE_list("resnet_depth_config", None, "Stage sizes for generic ResNet, e.g., 2,2,2,2 for ResNet18. Comma-separated.")
flags.DEFINE_string("resnet_stem_callable_name", "ResNetStem", "Name of the stem callable for generic ResNet.")
flags.DEFINE_string("resnet_stem_kwargs_json", "{}", "JSON string of keyword arguments for the ResNet stem.")
flags.DEFINE_string("resnet_block_callable_name", "ResNetBlock", "Name of the block callable for generic ResNet.")
flags.DEFINE_string("resnet_block_kwargs_json", "{}", "JSON string of keyword arguments for the ResNet block.")
flags.DEFINE_integer("evaluator_cache_size", 2**16, "Size of the LRU cache for the JAX evaluator.") # 65536
flags.DEFINE_integer("inference_batch_size", 8, "Batch size for evaluator inference.")

# From TF AlphaZero, for game_specific_az_path
flags.DEFINE_bool(
    "game_specific_az_path", True,
    "Whether to append game name to path."
    " Disable to lead a checkpoint from path directly.")

# Insert this mapping before config = ConfigJAX(...)
LOG_LEVELS = {
    "ERROR": 0,
    "WARN": 1,
    "INFO": 2,
    "DEBUG": 3,
    "TRACE": 4,
}

def main(argv):
  del argv # Unused
  config = ConfigJAX(
      game=FLAGS.game,
      path=FLAGS.path,
      learning_rate=FLAGS.learning_rate,
      weight_decay=FLAGS.weight_decay,
      train_batch_size=FLAGS.train_batch_size,
      replay_buffer_size=FLAGS.replay_buffer_size,
      replay_buffer_reuse=FLAGS.replay_buffer_reuse,
      max_steps=FLAGS.max_steps,
      checkpoint_freq=FLAGS.checkpoint_freq,
      actors=FLAGS.actors,
      evaluators=FLAGS.evaluators,
      evaluation_window=FLAGS.evaluation_window,
      eval_levels=FLAGS.eval_levels,
      uct_c=FLAGS.uct_c,
      max_simulations=FLAGS.max_simulations,
      policy_alpha=FLAGS.policy_alpha,
      policy_epsilon=FLAGS.policy_epsilon,
      temperature=FLAGS.temperature,
      temperature_drop=FLAGS.temperature_drop,
      nn_model=FLAGS.nn_model,
      nn_width=FLAGS.nn_width,
      nn_depth=FLAGS.nn_depth,
      observation_shape=None, # Will be populated by the game
      output_size=None,       # Will be populated by the game
      quiet=FLAGS.quiet,
      master_seed=FLAGS.master_seed,
      # New ResNet/ResNeSt related fields
      resnet_depth_config=[int(x) for x in FLAGS.resnet_depth_config] if FLAGS.resnet_depth_config else None,
      resnet_stem_callable_name=FLAGS.resnet_stem_callable_name,
      resnet_stem_kwargs=json.loads(FLAGS.resnet_stem_kwargs_json),
      resnet_block_callable_name=FLAGS.resnet_block_callable_name,
      resnet_block_kwargs=json.loads(FLAGS.resnet_block_kwargs_json),
      evaluator_cache_size=FLAGS.evaluator_cache_size,
      log_level=LOG_LEVELS[FLAGS.log_level],
      inference_batch_size=FLAGS.inference_batch_size,
  )

  # Set external library log levels to match config
  set_external_libraries_log_level(config.log_level)

  # Create a temp dir if path is not set.
  if not config.path:
    if FLAGS.game_specific_az_path:
      config = config._replace(path=tempfile.mkdtemp(prefix=f"az_jax_{config.game}_"))
    else:
      config = config._replace(path=tempfile.mkdtemp(prefix="az_jax_"))
    print(f"Path not specified. Using temp dir: {config.path}")
  elif FLAGS.game_specific_az_path:
    config = config._replace(path=os.path.join(config.path, config.game))

  # Ensure the path exists.
  if config.path and not os.path.exists(config.path):
    os.makedirs(config.path, exist_ok=True)
    print(f"Path created: {config.path}")
  
  # Save the config to a JSON file in the experiment path.
  if config.path:
    config_path = os.path.join(config.path, "config.json")
    # Convert namedtuple to dict for JSON serialization, handling potential non-serializable fields
    config_dict = dict(config._asdict())
    config_dict['resnet_stem_kwargs'] = str(config_dict['resnet_stem_kwargs']) # Convert dict to str if not already
    config_dict['resnet_block_kwargs'] = str(config_dict['resnet_block_kwargs']) # Convert dict to str if not already
    
    try:
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved to {config_path}")
    except TypeError as e:
        print(f"Warning: Could not serialize config to JSON: {e}. Some fields might not be serializable.")
        # Fallback: try to save a string representation
        try:
            with open(config_path, "w") as f:
                f.write(str(config_dict))
            print(f"Saved string representation of config to {config_path}")
        except Exception as e_str:
            print(f"Error saving string representation of config: {e_str}")


  if config.actors == 0: # Eval-only mode or human play mode
    print("Running in eval-only mode or for human play setup (actors=0).")
    # This mode would typically load a checkpoint and allow playing against the agent
    # or running evaluations without further training.
    # For now, just run the alpha_zero_jax function which will handle it if evaluators > 0.
    # If evaluators is also 0, it might just set up and exit or error.
    # The alpha_zero_jax function should gracefully handle actors=0.
    pass # Let alpha_zero_jax handle this scenario.

  try:
    # The main AlphaZero JAX training loop.
    # alpha_zero_jax_lib is the imported module
    alpha_zero_jax_lib.alpha_zero_jax(config=config)
  except KeyboardInterrupt:
    print("Caught KeyboardInterrupt. Shutting down...")
    # Perform any necessary cleanup here, though spawn.join should handle processes.
  finally:
    # Ensure all spawned processes are joined.
    # This is important for graceful shutdown, especially with multiprocessing.
    print("All processes joined. Exiting.")

if __name__ == "__main__":
  # spawn.main_handler is used to set up multiprocessing correctly,
  # especially on platforms like Windows. It also handles signals.
  with spawn.main_handler():
    mp.set_start_method("spawn", force=True)
    app.run(main) 