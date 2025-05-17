import collections
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
import orbax.checkpoint as ocp # Orbax CheckpointManager and options
from orbax.checkpoint.path import atomicity as ocp_atomicity # For CommitFileTemporaryPath
from flax.training import orbax_utils # For save/restore args
import random
import time
import itertools # For itertools.count in learner
import numpy as np # For np.zeros, np.random in copied _play_game, TrajectoryState
import pyspiel # For game object and types
import os # Added for checkpointing directory management
import shutil # Added for rmtree and rename
import functools # For watcher decorator
import traceback # For watcher decorator
import tracemalloc
import psutil
import gc
import logging
import absl.logging
import multiprocessing as mp  # For remote inference service queue
import queue as std_queue # For robust exception handling in BatchAssemblyThread (Reviewer Task 4)
import threading # For InferenceServicer threads

from . import model_jax
from . import evaluator_jax # For AlphaZeroEvaluatorJAX (though less used by actors now)
from open_spiel.python.utils import spawn, file_logger, data_logger, stats # Activated file_logger, spawn, data_logger, stats
from open_spiel.python.algorithms import mcts # Activated mcts
from .remote_inference import RemoteEvaluator, InferenceRequest, InferenceResponse, SHUTDOWN_SENTINEL, ShutdownException

# NEW IMPORT for actor and evaluator logic
from .actor_evaluator_logic import actor, evaluator, watcher

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001

# Constants for learner statistics
VALUE_ACC_HIST_BUCKETS = 20  # Number of buckets for value accuracy histograms
VALUE_PRED_HIST_BUCKETS = 20 # Number of buckets for value prediction histograms
EVALS_STAT_WINDOW = 100      # Window for evaluation statistics

# ---- Named log-level constants for logging discipline ----
ERROR = 0
WARN = 1
INFO = 2
DEBUG = 3
TRACE = 4
# Log-level meanings:
#   ERROR: Only critical errors and experiment-ending events.
#   WARN:  Warnings about recoverable issues or unexpected states.
#   INFO:  High-level experiment progress, start/stop, per-episode summaries.
#   DEBUG: Per-step training summaries, checkpointing, detailed diagnostics.
#   TRACE: Extremely verbose, per-move or per-action logs (rarely used).
# All logging output in this file should be gated by these levels, and no print() should appear unless guarded by log_level >= DEBUG or higher.


class ConfigJAX(collections.namedtuple(
    "ConfigJAX", [
        "game",                     # pyspiel.Game: The game to play.
        "path",                     # str: Path to save and load checkpoints and logs.
        "learning_rate",            # float: Learning rate for the optimizer.
        "weight_decay",             # float: Weight decay for the optimizer.
        "train_batch_size",         # int: Batch size for training.
        "replay_buffer_size",       # int: Maximum size of the replay buffer.
        "replay_buffer_reuse",      # int: Number of times to reuse data from the replay buffer.
        "max_steps",                # int: Total number of training steps.
        "checkpoint_freq",          # int: Frequency (in steps) to save checkpoints.
        "actors",                   # int: Number of actor processes.
        "evaluators",               # int: Number of evaluator processes.
        "evaluation_window",        # int: Number of games to average for evaluation.
        "eval_levels",              # int: How many different MCTS search counts to use for evaluation.
        "uct_c",                    # float: UCT exploration constant.
        "max_simulations",          # int: Maximum number of MCTS simulations per move.
        "policy_alpha",             # float: Dirichlet noise alpha parameter for policy exploration.
        "policy_epsilon",           # float: Dirichlet noise epsilon parameter for policy exploration.
        "temperature",              # float: Initial temperature for sampling actions during self-play.
        "temperature_drop",         # int: Number of steps after which temperature becomes 0.
        "nn_model",                 # str: Name of the neural network model (e.g., "mlp", "resnet", "resnet18").
        "nn_width",                 # int: Width of the neural network (e.g., number of neurons in MLP hidden layers, or base channels in ResNet).
        "nn_depth",                 # int: Depth of the neural network (e.g., number of hidden layers in MLP, or number of blocks in ResNet).
        "observation_shape",        # tuple: Shape of the game observation tensor. Populated by the game.
        "output_size",              # int: Number of distinct actions in the game. Populated by the game.
        "quiet",                    # bool: Whether to suppress verbose logging.
        "master_seed",              # int: Master PRNG seed for JAX.
        # New fields for ResNet/ResNeSt customization
        "resnet_depth_config",      # Optional[list[int]]: Stage sizes for generic ResNet, e.g., [2, 2, 2, 2] for ResNet18.
        "resnet_stem_callable_name",# Optional[str]: Name of the stem callable for generic ResNet (e.g., "ResNetStem", "ResNetDStem").
        "resnet_stem_kwargs",       # Optional[Mapping]: Keyword arguments for the ResNet stem.
        "resnet_block_callable_name",# Optional[str]: Name of the block callable for generic ResNet (e.g., "ResNetBlock", "ResNetBottleneckBlock").
        "resnet_block_kwargs",      # Optional[Mapping]: Keyword arguments for the ResNet block.
        "evaluator_cache_size",     # int: Size of the LRU cache for the evaluator.
        "log_level",                # int: Logging verbosity: 0=outcome only, 1=minimal, 2=debug
        "remote_evaluator_timeout_ms", # int: Timeout in milliseconds for remote evaluator queue operations.
        "inference_batch_timeout_ms", # float: Timeout in milliseconds for BatchAssemblyThread to form a batch.
        "inference_batch_size",     # int: Batch size for inference requests.
    ])):                                 # Default for remote_evaluator_timeout_ms can be set at instantiation.
  """A config for the JAX AlphaZero model/experiment."""
  # To allow None defaults for Optional fields in namedtuple, provide them at instantiation.
  # Default values for new optional fields can be handled in the main script creating the ConfigJAX instance.
  pass


def alpha_zero_jax(config: ConfigJAX):
    """Main entry point for JAX AlphaZero."""
    import pyspiel  # Ensure pyspiel is imported in this scope
    main_key = jax.random.PRNGKey(config.master_seed)
    # Python and NumPy random seeds are set globally for the main process if needed,
    # but actor/evaluator subprocesses will get their own dedicated integer seeds.
    random.seed(config.master_seed) 
    np.random.seed(config.master_seed) # Seed NumPy for main process

    os.makedirs(config.path, exist_ok=True)
    main_log_directory = config.path 
    main_log_name = "main_alpha_zero_jax" 
    main_process_logger = file_logger.FileLogger(main_log_directory, main_log_name, not config.quiet)
    actual_log_file_path = os.path.join(main_log_directory, f'log-{main_log_name}.txt')
    if not config.quiet and config.log_level >= INFO:
        print(f"Main process logging to: {actual_log_file_path}")
    game = pyspiel.load_game(config.game)
    servicer_key, spawn_key = jax.random.split(main_key)
    inference_model, inference_variables = model_jax.init_flax_model_and_variables(
        servicer_key, config, game)
    process_keys = jax.random.split(spawn_key, 1 + config.actors + config.evaluators)
    learner_key = process_keys[0]
    actor_seed_keys = process_keys[1 : 1 + config.actors]
    evaluator_seed_keys = process_keys[1 + config.actors : 1 + config.actors + config.evaluators]

    actor_initial_seeds = [jax.random.randint(key, (), 0, 2**31 - 1).item() for key in actor_seed_keys]
    evaluator_initial_seeds = [jax.random.randint(key, (), 0, 2**31 - 1).item() for key in evaluator_seed_keys]

    # Create a consolidated list of response queues for remote inference
    total_remote_clients = config.actors + config.evaluators
    all_client_response_queues = [mp.Queue() for _ in range(total_remote_clients)]

    # Initialize inference_request_queue and lists for processes and their queues
    inference_request_queue = mp.Queue()
    processes = []
    actor_process_queues = []
    evaluator_process_queues = []

    if config.log_level >= INFO:
        main_process_logger.print(f"Starting {config.actors} actors...")
    for i in range(config.actors):
        actor_kwargs = {
            "game": game,
            "config": config,
            "num": i,
            "initial_seed": actor_initial_seeds[i],
            "inference_request_queue": inference_request_queue,
            "inference_response_queue": all_client_response_queues[i]
        }
        p = spawn.Process(target=actor, kwargs=actor_kwargs)
        processes.append(p)
        actor_process_queues.append(p.queue)
    if config.log_level >= INFO:
        main_process_logger.print(f"Starting {config.evaluators} evaluators...")
    for i in range(config.evaluators):
        eval_kwargs = {
            "game": game,
            "config": config,
            "num": i,
            "initial_seed": evaluator_initial_seeds[i], # CORRECTED: Was evaluator_keys[i]
            "inference_request_queue": inference_request_queue,
            "inference_response_queue": all_client_response_queues[config.actors + i]
        }
        p = spawn.Process(target=evaluator, kwargs=eval_kwargs)
        processes.append(p)
        evaluator_process_queues.append(p.queue)
    learner_kwargs = {
        "game": game,
        "config": config,
        "actor_queues": actor_process_queues,
        "evaluator_queues": evaluator_process_queues,
        "prng_key": learner_key,
        "inference_request_queue": inference_request_queue,
        # "actor_inference_response_queues": actor_inference_response_queues, # Old, now consolidated
        "all_client_response_queues": all_client_response_queues, # New consolidated list
        "initial_flax_model": inference_model,      # Pass the centrally initialized model
        "initial_variables": inference_variables   # Pass the centrally initialized variables
    }
    if config.log_level >= INFO:
        main_process_logger.print("Starting Learner in main process...")
    try:
        # Call learner directly with unpacked kwargs for clarity
        learner(
            game=learner_kwargs["game"],
            config=learner_kwargs["config"],
            actor_queues=learner_kwargs["actor_queues"],
            evaluator_queues=learner_kwargs["evaluator_queues"],
            prng_key=learner_kwargs["prng_key"],
            inference_request_queue=learner_kwargs["inference_request_queue"],
            # "actor_inference_response_queues": learner_kwargs["actor_inference_response_queues"], # Old
            all_client_response_queues=learner_kwargs["all_client_response_queues"], # New
            initial_flax_model=learner_kwargs["initial_flax_model"],
            initial_variables=learner_kwargs["initial_variables"]
        )
    except (KeyboardInterrupt, EOFError) as e:
        if config.log_level >= INFO:
            main_process_logger.print(f"Caught {type(e).__name__}, stopping AlphaZero JAX.")
    finally:
        if config.log_level >= INFO:
            main_process_logger.print("AlphaZero JAX stopping. Signaling actors and evaluators to exit.")
        for proc in processes:
            try:
                proc.join()
            except Exception as join_e:
                if config.log_level >= INFO:
                    main_process_logger.print(f"Error joining process: {join_e}")
        if config.log_level >= INFO:
            main_process_logger.print("AlphaZero JAX run completed.")


# Entry point for the script (if run directly)
# This requires absl.app and absl.flags, similar to alpha_zero.py
# For now, this function `alpha_zero_jax` can be called from another script.
# Example: 
# if __name__ == '__main__':
#   from absl import app
#   from absl import flags
#   # Define flags for ConfigJAX fields
#   FLAGS = flags.FLAGS
#   flags.DEFINE_string("game", "tic_tac_toe", "Name of the game.")
#   flags.DEFINE_string("path", "/tmp/az_jax_test", "Path for logs and checkpoints.")
#   # ... other flags ...
#   flags.DEFINE_integer("master_seed", 42, "Master PRNG seed.")

#   def main(argv):
#     del argv # Unused.
#     config = ConfigJAX(
#         game=FLAGS.game,
#         path=FLAGS.path,
#         # ... populate from other flags ...
#         master_seed=FLAGS.master_seed,
#         # Sensible defaults for other fields for testing:
#         learning_rate=0.001, weight_decay=0.0001, train_batch_size=128,
#         replay_buffer_size=10000, replay_buffer_reuse=4, max_steps=100,
#         checkpoint_freq=10, actors=1, evaluators=1, evaluation_window=100,
#         eval_levels=1, uct_c=1.414, max_simulations=50, policy_alpha=0.3,
#         policy_epsilon=0.25, temperature=1.0, temperature_drop=10,
#         nn_model="mlp", nn_width=64, nn_depth=2,
#         observation_shape=[], output_size=0, # Will be filled by game
#         quiet=False
#     )
#     alpha_zero_jax(config)

#   app.run(main)


# Ensure all necessary imports are at the top of the file.
# Missing imports that might be needed based on the code above:
# import sys
# import datetime
# import tempfile
# import json (already there from model_jax likely, but ensure it's accessible)
# file_logger was imported as `from open_spiel.python.utils import spawn, file_logger`
# spawn was imported too.


# The learner function is defined below this in the actual file.
# Make sure its signature matches what's called by alpha_zero_jax:
# learner(*, game, config, actor_queues, evaluator_queues, broadcast_fn, prng_key)
# The logger for learner is created by its own @watcher decorator.

# END OF alpha_zero_jax function


# Learner function (skeleton was present in original file, make sure signature matches)
# The original skeleton was: learner(*, game, config, actors, evaluators, broadcast_fn, logger, prng_key)
# It should be: learner(*, game, config, actor_queues, evaluator_queues, prng_key)
# The logger for learner is created by its own @watcher decorator.

@watcher
def learner(*, game: pyspiel.Game, config: ConfigJAX, logger,
            actor_queues: list[spawn._ProcessQueue], 
            evaluator_queues: list[spawn._ProcessQueue], 
            prng_key: jax.random.PRNGKey,
            inference_request_queue: mp.Queue, 
            # actor_inference_response_queues: list[mp.Queue], # Old
            all_client_response_queues: list[mp.Queue], # New
            initial_flax_model, initial_variables): 
  """A learner that consumes actor trajectories and evaluator results, and updates the model."""
  # Start Python allocation tracing and RSS monitoring
  tracemalloc.start()
  _mem_proc = psutil.Process(os.getpid())
  if logger and config.log_level >= DEBUG:
    logger.print(f"JAX Learner started with PRNG key: {prng_key}")
    logger.print(f"Learner using game: {game}, config: {config}") # Log basic info
  elif not logger:
    try:
        if config is not None and hasattr(config, 'log_level') and config.log_level >= DEBUG:
            print(f"JAX Learner started (no logger) with PRNG key: {prng_key}")
    except Exception:
        pass

  # Initialize model and optimizer
  flax_model = initial_flax_model
  variables = initial_variables
  optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
  opt_state = optimizer.init(variables['params'])

  replay_buffer = Buffer(config.replay_buffer_size)
  
  # Initialize the InferenceServicer
  # This is where the batch_timeout_ms was hardcoded.
  # The user mentioned line 316. This part is an educated guess of the surrounding code.
  # The actual InferenceServicer initialization might be slightly different.
  # We need to find where 'servicer' or 'InferenceServicer' is created.
  
  # Assuming servicer is created like this based on its __init__ signature
  # and the hardcoded value reference.
  # The `initial_flax_model` and `initial_variables` are now passed to learner
  # so the servicer will use the jitted apply function derived from these.
  
  # JIT the model application function for inference servicer
  # This jit'd function will be passed to the InferenceServicer.
  # It should take (variables, batch_observations, batch_legals_masks)
  @jax.jit
  def _batched_inference_fn_for_servicer(model_vars, obs_batch, legals_batch):
      # initial_flax_model.apply is expected to handle legals_mask internally
      # by setting logits of illegal actions to -jnp.inf.
      policy_logits, value_preds = initial_flax_model.apply(
          model_vars, 
          obs_batch, 
          legals_mask=legals_batch, # Pass legals_mask to the model
          training=False, 
          mutable=False # No batch stats updates during pure inference
      )
      policy_probs = jax.nn.softmax(policy_logits, axis=-1)
      return policy_probs, value_preds # Return probabilities and value predictions

  servicer = InferenceServicer(
      request_queue=inference_request_queue,
      all_client_response_queues=all_client_response_queues,
      model_apply_fn=_batched_inference_fn_for_servicer, # Pass the new JITted function
      initial_model_variables=initial_variables, # Pass the initial variables
      max_batch_size=config.inference_batch_size, # Use the new config field for inference batch size
      batch_timeout_ms=config.inference_batch_timeout_ms, # Use the new config field
      num_actors=config.actors, # Used for logging/config within servicer
      output_size=config.output_size, # Pass output_size
      logger=logger, # Pass the learner's logger
      log_level=config.log_level
  )
  servicer.start()
  logger.print("Learner: InferenceServicer started.")

  # ---- Orbax Checkpointing Setup ----
  # Directory for periodic, managed checkpoints
  managed_ckpt_dir = os.path.join(config.path, "checkpoints_jax_managed")

  # Directory for the single 'latest' checkpoint (atomically updated)
  latest_ckpt_target_dir = os.path.join(config.path, "checkpoints_jax_latest_atomic") 

  if logger and config.log_level >= INFO:
      logger.print(f"Managed checkpoints will be saved to: {managed_ckpt_dir}")
      logger.print(f"Latest checkpoint (atomic via temp + rename) will be at: {latest_ckpt_target_dir}")

  # For periodic checkpoints
  mngr_options = ocp.CheckpointManagerOptions(
      save_interval_steps=config.checkpoint_freq if config.checkpoint_freq > 0 else 0,
      max_to_keep=3,
      create=True,
      enable_async_checkpointing=False
  )
  checkpoint_manager = ocp.CheckpointManager(
      directory=managed_ckpt_dir,
      # checkpointers=simple_commit_handler, # REMOVED - was causing TypeError
      options=mngr_options
  )

  # Checkpointer for the single 'latest' checkpoint, using CommitFileTemporaryPath
  # This will now write to a unique temp path each time before being moved.
  latest_checkpointer = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(use_ocdbt=False),
      temporary_path_class=ocp_atomicity.AtomicRenameTemporaryPath
  )
  if logger and config.log_level >= INFO:
      logger.print(f"Using Checkpointer with AtomicRenameTemporaryPath for staging latest checkpoints.")

  # Ensure the atomic checkpoint directory exists
  os.makedirs(latest_ckpt_target_dir, exist_ok=True)

  # Attempt to restore from the CheckpointManager (latest periodic checkpoint)
  initial_step = 0
  if checkpoint_manager.latest_step() is not None:
      initial_step = checkpoint_manager.latest_step()
      try:
          # Target for restore must match what was saved. Orbax saves the raw pytree.
          # We are saving a dict: {'variables': variables, 'opt_state': opt_state}
          target_to_restore_mngr = {'variables': variables, 'opt_state': opt_state}
          restored_mngr_state = checkpoint_manager.restore(
              step=initial_step,
              args=ocp.args.Composite( # Use Composite to restore specific parts
                  variables=ocp.args.StandardRestore(variables),
                  opt_state=ocp.args.StandardRestore(opt_state)
              )
              # item=target_to_restore_mngr # Alternative if not using Composite args
          )
          if restored_mngr_state:
              variables = restored_mngr_state['variables']
              opt_state = restored_mngr_state['opt_state']
              if logger: logger.print(f"Learner restored periodic checkpoint from {managed_ckpt_dir} at step {initial_step}")
          else:
              if logger: logger.print(f"No periodic checkpoint found by manager at {managed_ckpt_dir} (step {initial_step}). Starting fresh.")
              initial_step = 0 # Reset step if restore failed
      except Exception as e:
          if logger:
              logger.print(f"Error restoring periodic checkpoint via manager from {managed_ckpt_dir} (step {initial_step}): {e}. Starting fresh.")
              logger.print(traceback.format_exc())
          initial_step = 0 # Reset step on error
  else:
      if logger: logger.print(f"No existing periodic checkpoints found by manager in {managed_ckpt_dir}. Starting fresh.")
      initial_step = 0
  # ---- End Orbax Checkpointing Setup ----

  data_log = None
  if config.path:
    data_log = data_logger.DataLoggerJsonLines(config.path, "learner")
    if logger: logger.print(f"Learner logging data to: {os.path.join(config.path, 'learner.jsonl')}")

  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
  
  # Define outcome names for the histogram
  # Ensure this list covers all expected string outcomes.
  # The order determines the bucket ID (0 for "win", 1 for "loss", etc.)
  outcome_names_for_histogram = ["win", "loss", "draw", "quit", "eval"] 
  outcomes = stats.HistogramNamed(outcome_names_for_histogram)

  value_accuracies = [stats.BasicStats() for _ in range(VALUE_ACC_HIST_BUCKETS)]
  value_predictions = [stats.BasicStats() for _ in range(VALUE_PRED_HIST_BUCKETS)]
  evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels or 1)]

  # JIT compile the training step function
  @jax.jit
  def train_step_fn(current_variables, current_opt_state, batch_observations, batch_legals_masks, batch_policy_targets, batch_value_targets):
    # Defines the loss function and computes gradients.

    def loss_and_grad_inner_fn(params):
      apply_vars = {'params': params}
      if 'batch_stats' in current_variables:
        apply_vars['batch_stats'] = current_variables['batch_stats']
      
      mutable_list = ['batch_stats'] if 'batch_stats' in apply_vars else None

      preds_and_state = flax_model.apply(
          apply_vars,
          batch_observations,
          legals_mask=batch_legals_masks,
          training=True,
          mutable=mutable_list
      )
      
      if mutable_list:
        (policy_logits, value_preds), updated_model_state = preds_and_state
      else:
        (policy_logits, value_preds) = preds_and_state
        updated_model_state = None

      policy_loss_ce = optax.safe_softmax_cross_entropy(logits=policy_logits, labels=batch_policy_targets)
      
      # Debug prints for policy loss (can be removed after verification)
      # jax.debug.print("--- train_step_fn DEBUG: policy_loss_ce (raw_values): {vals}", vals=policy_loss_ce)
      # jax.debug.print("--- train_step_fn DEBUG: policy_loss_ce stats: shape={s}, min={min_val}, max={max_val}, mean={mean_val}, NaNs={nans}, Infs={infs}",
      #                 s=policy_loss_ce.shape, min_val=jnp.min(policy_loss_ce), max_val=jnp.max(policy_loss_ce),
      #                 mean_val=jnp.mean(policy_loss_ce),
      #                 nans=jnp.sum(jnp.isnan(policy_loss_ce)), infs=jnp.sum(jnp.isinf(policy_loss_ce)))

      has_at_least_one_legal_action = jnp.any(batch_legals_masks, axis=1)
      policy_loss_final_contrib = jnp.where(has_at_least_one_legal_action, policy_loss_ce, 0.0)
      
      num_valid_policy_samples = jnp.sum(has_at_least_one_legal_action)
      policy_loss = jnp.sum(policy_loss_final_contrib) / jnp.maximum(num_valid_policy_samples, 1.0)

      value_loss = optax.squared_error(
          predictions=jnp.squeeze(value_preds, axis=-1),
          targets=jnp.squeeze(batch_value_targets, axis=-1)
      )
      value_loss = jnp.mean(value_loss)
      
      total_loss = policy_loss + value_loss
      return total_loss, (updated_model_state, policy_loss, value_loss)

    (loss_val, (new_model_state, p_loss, v_loss)), grads = jax.value_and_grad(
        loss_and_grad_inner_fn, has_aux=True)(current_variables['params'])
    
    updates, new_opt_state = optimizer.update(grads, current_opt_state, current_variables['params'])
    new_params = optax.apply_updates(current_variables['params'], updates)
    
    new_variables = current_variables.copy()
    new_variables['params'] = new_params
    if new_model_state and 'batch_stats' in new_model_state:
        new_variables['batch_stats'] = new_model_state['batch_stats']
    
    # Debug prints for variable and opt_state contents (can be removed after verification)
    # jax.debug.print("--- train_step_fn OUTPUT: new_variables PyTree structure ---")
    # ... (existing debug prints for variables and opt_state)
          
    return new_variables, new_opt_state, loss_val, p_loss, v_loss

  # ---- Main Learner Loop ----
  last_time = time.time()
  start_time = last_time  # Global start time for throughput stats
  total_trajectories = 0
  
  current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')

  def trajectory_generator():
    while True:
      found = 0
      for queue_idx, queue in enumerate(actor_queues): # Use actor_queues
        try:
          item = queue.get_nowait()
          yield item 
          found += 1
        except spawn.Empty: # Use spawn.Empty
          pass
        except Exception as e: # Catch other potential errors
          if logger: logger.print(f"Error getting trajectory from actor_queue {queue_idx}: {e}")

      if not found: # If all queues were empty, pause briefly
        time.sleep(0.001) # Small sleep to avoid busy-waiting

  # This function collects a batch of trajectories.
  # It can be made more sophisticated (e.g. to ensure diversity or recency if needed).
  def collect_trajectories(num_to_collect):
      collected = []
      for traj in trajectory_generator(): # trajectory_generator will loop until enough data is found or error
          if traj: # Ensure trajectory is not None
            collected.append(traj)
            if len(collected) >= num_to_collect:
                break
          # Add a safeguard if generator somehow misbehaves, though it should block or yield.
          # This part might need timeout logic if queues can remain empty indefinitely and block training.
      return collected
      
  for step in itertools.count(initial_step + 1): # Start step from last restored step + 1
    # Log memory usage and tracemalloc stats every 4 steps
    if logger and step % 200 == 0:
        rss_mb = _mem_proc.memory_info().rss / (1024 * 1024)
        logger.print(f"Learner step {step}: RSS memory usage: {rss_mb:.2f} MB")
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logger.print("[Top 5 memory allocations]")
            for stat in top_stats[:5]:
                logger.print(str(stat))
        except Exception as e:
            logger.print(f"Error taking tracemalloc snapshot: {e}")
    if config.max_steps > 0 and step > config.max_steps:
        if logger: logger.print(f"Max steps {config.max_steps} reached. Exiting learner.")
        break

    # Get trajectories from actors
    # How many trajectories to pull depends on how much data is needed.
    # Example: Pull enough for a batch, or a fixed number.
    # For simplicity, let's assume we pull enough trajectories that contain at least train_batch_size states.
    # This part is crucial and might need refinement based on typical trajectory length.
    
    # Simplistic approach: pull a number of trajectories, then add all their states to buffer.
    # If replay_buffer is smaller than batch_size, this will wait.
    # A better way might be to ensure buffer has enough for a batch before sampling.
    
    # For now, continuously add to replay buffer from actor queues.
    # The original TF code has a loop that tries to get one trajectory.
    
    num_states = 0
    num_trajectories = 0
    # Try to get at least one trajectory to process for stats, even if buffer is full
    # This loop will block until a trajectory is available or an error occurs
    
    try:
        # Get one trajectory to update stats and add to buffer
        # This get() might block if queues are empty.
        # Consider timeout or non-blocking with sleep if learner should do other things.
        # For now, let's assume actor_queues[0] is a valid queue to try.
        # A round-robin or random selection might be better if many actor_queues.
        # The trajectory_generator handles iterating through queues.
        
        # Using the collect_trajectories helper
        # Collect a small number of trajectories to process per learner step
        # This is a placeholder, a more robust strategy for data ingestion might be needed.
        # Drain all available trajectories from actor queues to avoid backlog
        trajectories_to_process = []
        for queue_idx, queue in enumerate(actor_queues):
            while True:
                try:
                    traj = queue.get_nowait()
                    trajectories_to_process.append(traj)
                except spawn.Empty:
                    break
                except Exception as e:
                    if logger:
                        logger.print(f"Learner: Error draining actor_queue {queue_idx}: {e}")
                    break

        for traj in trajectories_to_process:
            if not hasattr(traj, "states"): # Add this check
                if logger:
                    logger.print(f"Learner: Received object of type {type(traj)} from actor queue: {repr(traj)}. Skipping.")
                continue

            total_trajectories += 1
            num_trajectories += 1
            game_lengths.add(len(traj))
            game_lengths_hist.add(len(traj))
            num_states += len(traj)
            # outcomes.add(traj.returns) # Old incorrect way

            # Map scalar return to outcome string for HistogramNamed
            current_player_return = traj.returns # This is now a scalar
            outcome_bucket_id = None # NEW LOGIC: for integer index

            if current_player_return is not None: # Check if return value is not None
                if current_player_return > 0: # Win for this player
                    outcome_bucket_id = 0 # Index for "win"
                elif current_player_return < 0: # Loss for this player
                    outcome_bucket_id = 1 # Index for "loss"
                elif current_player_return == 0: # Draw
                    outcome_bucket_id = 2 # Index for "draw"
                # Add other mappings if quit/eval states are possible and have distinct scalar returns.
                # Example: if quit is -2 and eval is -3, map them to "quit" and "eval" strings.
                # elif current_player_return == -2: # Example for quit
                #     outcome_str = "quit"
                # elif current_player_return == -3: # Example for eval
                #     outcome_str = "eval"
                else:
                    # Log an unexpected return value but don't add to histogram or default to something.
                    # This case should ideally not be hit if actors correctly set scalar returns.
                    if logger:
                        logger.print(f"Learner: Received trajectory with unexpected (but non-None) return value: {current_player_return}. Not mapping to known outcome for histogram.")
            else: # current_player_return is None
                 if logger:
                    logger.print(f"Learner: Received trajectory with None return value. Not adding to outcomes histogram.")


            if outcome_bucket_id is not None: # NEW LOGIC
                try:
                    # Directly add the outcome string to the histogram
                    # outcomes.add(outcome_str) # OLD LOGIC
                    outcomes.add(outcome_bucket_id) # NEW LOGIC: Pass integer index
                except (ValueError, KeyError, IndexError) as e_hist: # Catch if string not in names or other issue
                    if logger:
                        logger.print(f"Learner: Error adding outcome bucket_id '{outcome_bucket_id}' (return: {current_player_return}) to histogram: {e_hist}. Histogram names: {outcomes._names if hasattr(outcomes, '_names') else 'N/A'}")


            # Add states to replay buffer
            # Each element in traj.states is a TrajectoryState
            for transition in traj.states: # CORRECTED: Iterate over traj.states
                # Create TrainInputJAX from TrajectoryState
                # Determine the actual value_target based on the player's outcome (traj.returns)
                # This assumes traj.returns is the scalar outcome for the current player's perspective
                # For AlphaZero, the value target for each state is typically the final game outcome for that player.
                
                # Ensure policy and value are appropriate. MCTS value might need to be discounted or returns used.
                # For now, assuming transition.value is the MCTS-derived value for that state,
                # and traj.returns is the final game outcome from the player's perspective.
                # The value_target for training is usually the final game outcome.
                
                # Check if transition.policy is correctly shaped/normalized if it's an MCTS policy
                # Check if transition.value is the MCTS value (usually Q-value or similar)

                train_input = model_jax.TrainInputJAX(
                    observation=transition.observation,
                    legals_mask=transition.legals_mask,
                    policy_target=transition.policy, 
                    value_target=jnp.array(traj.returns, dtype=jnp.float32) # Use game outcome as value target
                )
                replay_buffer.append(train_input)

    except spawn.Empty: # Should be handled by trajectory_generator now
        if logger and config.log_level >= DEBUG: logger.opt_print("Learner: All actor queues empty.") # opt_print for less frequent messages
        # Continue to next part of the loop (e.g. try training if buffer is full)
    except Exception as e:
        if logger: 
            logger.print(f"Learner: Error processing actor queue: {e}")
        # Potentially skip this learner step or handle error more gracefully
    
    now = time.time()
    seconds = now - last_time
    last_time = now
    
    # Compute global elapsed time since start
    global_elapsed = now - start_time
    
    # Calculate effective actors contributing to this step's data
    # This is a bit heuristic; if actors are much faster than learner, effective_actors might be high.
    # If only one trajectory was processed, effective_actors for this stat could be 1.
    effective_actors = config.actors

    log_message_timing = (
        f"Step: {step}, Global Game Speed: {total_trajectories/global_elapsed:.1f} games/s, "
        f"Global States/s: {replay_buffer.total_seen/global_elapsed:.1f}"
    )
    log_message_buffer = f"Buffer size: {len(replay_buffer)}. Total states seen by buffer: {replay_buffer.total_seen}"

    # Use initial_step for the first log, then the loop's step variable
    current_log_step = step # step starts from 1 in the loop, initial_step is 0-based from manager
    step_log_msg_prefix = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Step: {current_log_step}"
    if logger and config.log_level >= DEBUG:
      logger.print(step_log_msg_prefix)
      logger.print(log_message_timing)
      logger.print(log_message_buffer)

    # Actual JAX training step
    save_path_for_broadcast = None # Initialize, will be updated if checkpoint is saved
    if len(replay_buffer) >= config.train_batch_size and config.train_batch_size > 0:
      batch_data = replay_buffer.sample(config.train_batch_size)
      if logger and config.log_level >= DEBUG:
          logger.print(f"Training on batch of size: {len(batch_data)}")
      
      if logger and config.log_level >= DEBUG:
        logger.print(f"Learner: Processing training batch of {len(batch_data)} samples on TPU...")
      # Ensure TrainInputJAX.stack method is available and used correctly.
      # If not, stack manually here. For now, assuming model_jax.TrainInputJAX.stack exists.
      try:
        stacked_input = model_jax.TrainInputJAX.stack(batch_data)
        batch_obs_jnp = jnp.array(stacked_input.observation, dtype=jnp.float32)
        batch_legals_jnp = jnp.array(stacked_input.legals_mask, dtype=jnp.bool_) 
        batch_policy_jnp = jnp.array(stacked_input.policy_target, dtype=jnp.float32)
        batch_value_jnp = jnp.array(stacked_input.value_target, dtype=jnp.float32)
        
        variables, opt_state, total_loss_val, policy_loss_val, value_loss_val = train_step_fn(
            variables, opt_state, batch_obs_jnp, batch_legals_jnp, batch_policy_jnp, batch_value_jnp
        )
        # Update the inference servicer with the new model variables
        if servicer:
            servicer.update_model_variables(variables)
        
        current_total_loss, current_policy_loss, current_value_loss = total_loss_val, policy_loss_val, value_loss_val 
        
        loss_log_msg = f"Step: {step}, Total Loss: {current_total_loss:.4f}, Policy Loss: {current_policy_loss:.4f}, Value Loss: {current_value_loss:.4f}"
        if logger and config.log_level >= DEBUG:
          logger.print(loss_log_msg)
        
        # ---- Orbax Checkpointing: Save ----
        save_target_pytree = {'variables': variables, 'opt_state': opt_state}
        try:
            # Periodic checkpoint save if enabled
            if checkpoint_manager.should_save(step):
                checkpoint_manager.save(
                    step,
                    args=ocp.args.Composite(
                        variables=ocp.args.StandardSave(variables),
                        opt_state=ocp.args.StandardSave(opt_state),
                        metrics=ocp.args.JsonSave({
                            'step': step,
                            'policy_head_loss': float(policy_loss_val),
                            'value_head_loss': float(value_loss_val)
                        })
                    )
                )
                if logger and config.log_level >= DEBUG:
                    logger.opt_print(f"Saved checkpoint for step {step} via manager to {managed_ckpt_dir}")
            # ---- Orbax Checkpointing: Atomic Save of variables only ----
            try:
                latest_checkpointer.save(
                    latest_ckpt_target_dir,
                    args=ocp.args.PyTreeSave(item=variables),
                    force=True
                )
                if logger and config.log_level >= DEBUG:
                    logger.opt_print(f"Saved atomic latest checkpoint (variables) to {latest_ckpt_target_dir}")
                save_path_for_broadcast = latest_ckpt_target_dir
            except Exception as e_atomic:
                if logger:
                    logger.print(f"Error saving atomic latest checkpoint to {latest_ckpt_target_dir}: {e_atomic}")
        except Exception as e:
            err_msg = f"Error saving checkpoint for step {step} via manager: {e}"
            if logger:
                logger.print(err_msg)
                logger.print(traceback.format_exc())
        # ---- End Orbax Checkpointing: Save ----

      except AttributeError as e:
        error_msg = f"Error during training data preparation (possibly missing TrainInputJAX.stack): {e}"
        if logger: 
          logger.print(error_msg)
        current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan') 

    else:
      current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')
      if logger and config.log_level >= DEBUG:
        logger.opt_print(f"Step: {step}, Replay buffer not full enough for training. Size: {len(replay_buffer)}/{config.train_batch_size}")

    # Collect evaluation results
    for i, evac_queue in enumerate(evaluator_queues): # Now uses the passed evaluator_queues
        while True:
            try:
                # Assuming evaluator puts (difficulty_level_idx, outcome_for_az_player)
                # If only one evaluator, difficulty_level_idx might be 0 or not sent.
                # For now, assume simple case: outcome is for AZ player, difficulty is implicit by queue index.
                eval_outcome = evac_queue.get_nowait()
                if isinstance(eval_outcome, tuple) and len(eval_outcome) == 2:
                    difficulty_idx, outcome = eval_outcome
                    if 0 <= difficulty_idx < len(evals):
                        evals[difficulty_idx].append(outcome)
                elif isinstance(eval_outcome, (int, float)): # Simpler: evaluator sends just the outcome for its level
                    if 0 <= i < len(evals):
                         evals[i].append(eval_outcome)
            except spawn.Empty: # Make sure spawn.Empty is the correct exception from the queue
                break
            except Exception as e: # Catch other potential errors from queue processing
                if logger: 
                  logger.print(f"Error processing evaluator queue {i}: {e}")
                break # Avoid busy-looping on a consistently problematic queue

    # Log to data_logger
    if data_log:
        metrics_to_log = {
            "step": step,
            "total_states_seen_by_buffer": replay_buffer.total_seen,
            "replay_buffer_size": len(replay_buffer),
            "states_per_s": num_states / seconds if seconds > 0 else 0,
            "states_per_s_actor": num_states / (effective_actors * seconds) if seconds > 0 else 0,
            "total_trajectories": total_trajectories,
            "trajectories_per_s": num_trajectories / seconds if seconds > 0 else 0,
            "game_length": game_lengths.as_dict,
            "game_length_hist": game_lengths_hist.data, # list of counts
            "outcomes": outcomes.data, # dict with 'counts' and 'names'
            "value_accuracy": [v.as_dict for v in value_accuracies],
            "value_prediction": [v.as_dict for v in value_predictions],
            "eval": {
                # Assuming evals[0] is representative for 'count' if multiple levels
                "count": evals[0].total_seen if evals and evals[0] else 0,
                "results": [sum(e.data) / len(e.data) if len(e.data) > 0 else 0 for e in evals]
            },
            "loss": {
                "total": float(current_total_loss),
                "policy": float(current_policy_loss),
                "value": float(current_value_loss),
                # L2 loss is part of adamw, not explicitly tracked here unless added to train_step_fn
            },
            # "cache": { ... } # MCTS cache stats are not easily available here, can be added if evaluator sends them
        }
        data_log.write(metrics_to_log)

    if logger and config.log_level >= DEBUG: logger.print("") # Add a newline for readability in FileLogger

    if config.max_steps > 0 and step >= config.max_steps:
      max_steps_msg = f"Max steps {config.max_steps} reached. Exiting learner."
      if logger: 
        logger.print(max_steps_msg)
      break

    if save_path_for_broadcast: # This broadcast is now only for OTHER types of messages if any.
        broadcast_msg = f"Broadcasting checkpoint: {save_path_for_broadcast}" # This message is now potentially misleading if path is None
        if logger and config.log_level >= DEBUG:
          logger.opt_print(broadcast_msg) 
        broadcast_fn(save_path_for_broadcast) 
  
  final_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] JAX Learner finished."
  if logger and config.log_level >= INFO:
    logger.print(final_msg)

  # Stop the inference servicer before exiting
  if servicer:
    servicer.stop()

def set_external_libraries_log_level(log_level):
    """Set logging level for Orbax, JAX, Flax, and absl based on internal log_level."""
    level_map = {
        0: logging.ERROR,   # ERROR
        1: logging.WARNING, # WARN
        2: logging.INFO,    # INFO
        3: logging.DEBUG,   # DEBUG
        4: logging.NOTSET,  # TRACE (or use DEBUG)
    }
    py_level = level_map.get(log_level, logging.INFO)
    for logger_name in [
        "orbax", "orbax.checkpoint", "jax", "flax", "absl", "absl.logging"
    ]:
        logging.getLogger(logger_name).setLevel(py_level)
    # Optionally set the root logger as well
    logging.getLogger().setLevel(py_level)
    # absl logging (sometimes not fully controlled by logging module)
    if log_level == 0:
        absl.logging.set_verbosity('error')
    elif log_level == 1:
        absl.logging.set_verbosity('warning')
    elif log_level == 2:
        absl.logging.set_verbosity('info')
    else:
        absl.logging.set_verbosity('debug')


# ---- Inference Servicer Components ----
# Based on the plan in TODO.md, Section 9.

class BatchAssemblyThread(threading.Thread):
    def __init__(self, request_queue: mp.Queue,
                 ready_batch_queue: std_queue.Queue,
                 max_batch_size: int, batch_timeout_ms: float,
                 num_actors: int, logger, log_level: int):
        super().__init__(name="BatchAssemblyThread")
        self.request_queue = request_queue
        self.ready_batch_queue = ready_batch_queue
        self.max_batch_size = max_batch_size
        self.batch_timeout_sec = batch_timeout_ms / 1000.0
        self.num_actors = num_actors # For logging
        self._stop_event = threading.Event()
        self.logger = logger
        self.log_level = log_level

    def stop(self):
        self._stop_event.set()
        # Attempt to unblock the request_queue.get() by putting a sentinel
        # This helps if the thread is waiting on an empty queue during shutdown.
        try:
            self.request_queue.put_nowait(SHUTDOWN_SENTINEL)
        except std_queue.Full: # mp.Queue.put_nowait raises queue.Full
            if self.logger and self.log_level >= WARN:
                self.logger.print("BatchAssemblyThread: request_queue full while trying to put SHUTDOWN_SENTINEL during stop().")
        except Exception as e: # pylint: disable=broad-except
            if self.logger and self.log_level >= WARN:
                self.logger.print(f"BatchAssemblyThread: Error putting SHUTDOWN_SENTINEL to request_queue during stop(): {e}")


    def run(self):
        if self.logger and self.log_level >= DEBUG:
            self.logger.print("BatchAssemblyThread started.")
        current_batch_requests = [] # Stores (InferenceRequest_obj, origin_response_queue_idx)
        batch_assembly_start_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Calculate remaining timeout for the current batch assembly window
                # If batch is empty, use full timeout, otherwise use remaining time.
                if not current_batch_requests:
                    timeout_for_get = self.batch_timeout_sec
                else:
                    elapsed_in_window = time.time() - batch_assembly_start_time
                    timeout_for_get = self.batch_timeout_sec - elapsed_in_window
                
                # Ensure timeout is not negative; use a very small positive value if it is,
                # to prevent blocking indefinitely or causing errors.
                # A small positive timeout also prevents busy-waiting if timeout_for_get becomes zero.
                timeout_for_get = max(0.001, timeout_for_get)

                raw_request_tuple = self.request_queue.get(timeout=timeout_for_get)

                if raw_request_tuple == SHUTDOWN_SENTINEL:
                    if self.logger and self.log_level >= INFO:
                        self.logger.print("BatchAssemblyThread received SHUTDOWN_SENTINEL on request_queue. Signaling ready_batch_queue and exiting.")
                    # Propagate SHUTDOWN_SENTINEL to the InferenceExecutionThread
                    self.ready_batch_queue.put(SHUTDOWN_SENTINEL) 
                    break # Exit the loop

                # Deserialize the request tuple using InferenceRequest.from_tuple
                try:
                    inference_request_obj = InferenceRequest.from_tuple(raw_request_tuple)
                except ValueError as e:
                    if self.logger and self.log_level >= WARN:
                        self.logger.print(f"BatchAssemblyThread: Error deserializing request: {e}. Skipping request: {raw_request_tuple}")
                    continue # Skip this malformed request

                # The actor_id from the deserialized request is the index for all_client_response_queues
                origin_response_queue_idx = inference_request_obj.actor_id 
                current_batch_requests.append((inference_request_obj, origin_response_queue_idx))

                # If this is the first request in a new batch, reset the assembly start time
                if len(current_batch_requests) == 1:
                    batch_assembly_start_time = time.time()

            except (std_queue.Empty, mp.queues.Empty): # Catches timeout from mp.Queue.get(), which raises queue.Empty.
                                                      # mp.queues.Empty is typically an alias for queue.Empty.
                # This means self.request_queue.get() timed out.
                # Proceed to check if the current (possibly empty) batch should be sent.
                pass 

            except Exception as e: # pylint: disable=broad-except
                if self._stop_event.is_set(): # Don't log errors if we are shutting down
                    break
                if self.logger and self.log_level >= ERROR:
                    self.logger.print(f"BatchAssemblyThread: Unexpected error processing request: {e}. Traceback: {traceback.format_exc()}")
                # Avoid busy-looping on persistent errors if not timeout-related
                if not isinstance(e, std_queue.Empty):
                    time.sleep(0.01) 
                continue


            # Determine if the current batch should be sent
            # Send if:
            # 1. The batch is full OR
            # 2. The batch is non-empty AND the batch assembly timeout has been reached.
            send_batch_now = False
            if current_batch_requests: # Only consider sending if batch is non-empty
                if len(current_batch_requests) >= self.max_batch_size:
                    send_batch_now = True
                elif (time.time() - batch_assembly_start_time >= self.batch_timeout_sec):
                    send_batch_now = True
            
            if send_batch_now:
                if self.logger and self.log_level >= TRACE: # TRACE for per-batch send
                    self.logger.print(f"BatchAssemblyThread: Sending batch of size {len(current_batch_requests)}")
                try:
                    # Send a list of (InferenceRequest_obj, origin_idx) tuples
                    self.ready_batch_queue.put(list(current_batch_requests), timeout=1.0) # Use a timeout for putting
                except std_queue.Full:
                    if self.logger and self.log_level >= WARN:
                        self.logger.print("BatchAssemblyThread: ready_batch_queue is full. Batch dropped.")
                    # Batch is dropped. Consider retry or other strategies if this is critical.
                finally:
                    # Always clear the current batch and reset the timer after attempting to send
                    current_batch_requests.clear()
                    batch_assembly_start_time = time.time() 

        # Final cleanup or logging after the loop exits
        if self.logger and self.log_level >= DEBUG:
            self.logger.print("BatchAssemblyThread finished.")

class InferenceExecutionThread(threading.Thread):
    def __init__(self, ready_batch_queue: std_queue.Queue,
                 all_client_response_queues: list[mp.Queue], 
                 model_apply_fn, 
                 model_variables, 
                 output_size: int,
                 logger, log_level: int):
        super().__init__(name="InferenceExecutionThread")
        self.ready_batch_queue = ready_batch_queue
        self.all_client_response_queues = all_client_response_queues
        self.model_apply_fn = model_apply_fn # This should be a JITted function
        self.model_variables = model_variables 
        self.output_size = output_size
        self._stop_event = threading.Event()
        self.logger = logger
        self.log_level = log_level
        self._variables_lock = threading.Lock() 

    def stop(self):
        self._stop_event.set()
        # Put a sentinel on its own queue to unblock the get() call if it's waiting
        try:
            self.ready_batch_queue.put_nowait(SHUTDOWN_SENTINEL)
        except std_queue.Full:
            if self.logger and self.log_level >= WARN:
                self.logger.print("InferenceExecutionThread: ready_batch_queue full while trying to put SHUTDOWN_SENTINEL during stop().")
        except Exception as e: # pylint: disable=broad-except
            if self.logger and self.log_level >= WARN:
                self.logger.print(f"InferenceExecutionThread: Error putting SHUTDOWN_SENTINEL to ready_batch_queue during stop(): {e}")


    def update_variables(self, new_variables):
        with self._variables_lock:
            self.model_variables = new_variables
        if self.logger and self.log_level >= TRACE: # TRACE for variable updates
            self.logger.print("InferenceExecutionThread: Updated model variables.")

    def run(self):
        if self.logger and self.log_level >= DEBUG:
            self.logger.print("InferenceExecutionThread started.")
        
        while not self._stop_event.is_set():
            try:
                # Get a batch of requests (or SHUTDOWN_SENTINEL) from BatchAssemblyThread
                # Use a timeout to periodically check the _stop_event
                batch_data_from_assembler = self.ready_batch_queue.get(timeout=0.1) 
            except std_queue.Empty:
                continue # Timeout, check stop_event and loop again

            if batch_data_from_assembler == SHUTDOWN_SENTINEL:
                if self.logger and self.log_level >= INFO:
                    self.logger.print("InferenceExecutionThread received SHUTDOWN_SENTINEL. Exiting.")
                # No need to propagate SHUTDOWN_SENTINEL to client queues here,
                # as RemoteEvaluator should handle timeouts or get SHUTDOWN_SENTINEL 
                # directly from the main process orchestrating the shutdown.
                break # Exit the loop

            if not batch_data_from_assembler: # Should not happen if sentinel is handled
                continue

            # batch_data_from_assembler is a list of (InferenceRequest_obj, origin_response_queue_idx)
            batched_observations_list = []
            batched_legals_masks_list = []
            # Store (request_id, origin_response_queue_idx) for sending responses
            request_details_for_response = [] 

            for inference_request_obj, origin_idx in batch_data_from_assembler:
                batched_observations_list.append(inference_request_obj.observation)
                batched_legals_masks_list.append(inference_request_obj.legals_mask)
                request_details_for_response.append((inference_request_obj.request_id, origin_idx))
            
            if not batched_observations_list: # If batch ended up empty after processing
                continue

            # Stack observations and masks into JAX arrays
            # Observations and legals_masks are expected to be NumPy arrays from RemoteEvaluator
            try:
                obs_array_batch = jnp.asarray(np.stack(batched_observations_list))
                legals_array_batch = jnp.asarray(np.stack(batched_legals_masks_list))
            except Exception as e: # pylint: disable=broad-except
                if self.logger and self.log_level >= ERROR:
                    self.logger.print(f"InferenceExecutionThread: Error stacking batch data: {e}. Batch items: {len(batched_observations_list)}")
                # Consider sending error responses to clients for this batch.
                # For now, log and skip.
                for req_id, origin_idx in request_details_for_response:
                    try:
                        error_value = np.array(0.0, dtype=np.float32)
                        error_policy = np.zeros(self.output_size, dtype=np.float32)  # Use self.output_size
                        error_resp = InferenceResponse(request_id=req_id, value=error_value, policy_probs=error_policy) # Dummy error response
                        self.all_client_response_queues[origin_idx].put_nowait(error_resp.to_tuple())
                    except (std_queue.Full, IndexError) as err_put:
                         if self.logger and self.log_level >= WARN:
                            self.logger.print(f"InferenceExecutionThread: Failed to send error for req {req_id} to client {origin_idx}: {err_put}")
                continue

            # Perform batched inference
            # Access model_variables safely using the lock
            with self._variables_lock:
                current_vars = self.model_variables
            
            try:
                # model_apply_fn is the JITted function _batched_inference_fn_for_servicer,
                # which expects (variables, obs_batch, legals_batch)
                # and returns (policy_probs_batch, value_output_batch) where policy_probs are already softmaxed.
                policy_probs_batch, value_output_batch = self.model_apply_fn(
                    current_vars, obs_array_batch, legals_array_batch) 
                
                # Ensure results are NumPy arrays for sending via queue
                policy_arrays_np = np.asarray(policy_probs_batch)
                # value_output_batch should be (batch_size, 1), squeeze to (batch_size,)
                value_scalars_np = np.asarray(value_output_batch).squeeze(axis=-1) 
            
            except Exception as e: # pylint: disable=broad-except
                if self.logger and self.log_level >= ERROR:
                    self.logger.print(f"InferenceExecutionThread: Error during model_apply_fn: {e}. Batch obs shape: {obs_array_batch.shape}, legals shape: {legals_array_batch.shape}")
                # Send error responses
                for req_id, origin_idx in request_details_for_response:
                    try:
                        error_value = np.array(0.0, dtype=np.float32)
                        error_policy = np.zeros(self.output_size, dtype=np.float32)  # Use self.output_size
                        error_resp = InferenceResponse(request_id=req_id, value=error_value, policy_probs=error_policy) # Dummy error response
                        self.all_client_response_queues[origin_idx].put_nowait(error_resp.to_tuple())
                    except (std_queue.Full, IndexError) as err_put:
                         if self.logger and self.log_level >= WARN:
                            self.logger.print(f"InferenceExecutionThread: Failed to send error (model_apply_fn error) for req {req_id} to client {origin_idx}: {err_put}")
                continue


            # Distribute results back to the respective origin queues
            for i, (req_id, origin_idx) in enumerate(request_details_for_response):
                value_scalar_for_client = value_scalars_np[i]
                policy_array_for_client = policy_arrays_np[i]
                
                # Construct InferenceResponse object and then convert to tuple
                response_obj = InferenceResponse(request_id=req_id, 
                                                 value=value_scalar_for_client, 
                                                 policy_probs=policy_array_for_client)
                response_tuple = response_obj.to_tuple()

                try:
                    # Use origin_idx to get the correct response queue from all_client_response_queues
                    self.all_client_response_queues[origin_idx].put(response_tuple, timeout=0.1) # Short timeout
                except std_queue.Full:
                    if self.logger and self.log_level >= WARN:
                        self.logger.print(f"InferenceExecutionThread: Response queue full for client {origin_idx}. Dropping response for req {req_id}.")
                except IndexError: # Should not happen if actor_id is managed correctly
                    if self.logger and self.log_level >= ERROR:
                         self.logger.print(f"InferenceExecutionThread: Invalid origin_idx {origin_idx} for req {req_id}. Max index {len(self.all_client_response_queues)-1}. Dropping response.")
                except Exception as e_put: # pylint: disable=broad-except
                    if self.logger and self.log_level >= ERROR:
                         self.logger.print(f"InferenceExecutionThread: Error putting response for req {req_id} to client {origin_idx}: {e_put}")


        if self.logger and self.log_level >= DEBUG:
            self.logger.print("InferenceExecutionThread finished.")

# Helper class to manage the inference service threads
class InferenceServicer:
    def __init__(self,
                 request_queue: mp.Queue,
                 all_client_response_queues: list[mp.Queue], 
                 model_apply_fn, 
                 initial_model_variables,
                 max_batch_size: int,
                 batch_timeout_ms: float,
                 num_actors: int, # For logging/config
                 output_size: int, # For dummy error policies
                 logger,
                 log_level: int):
        self.request_queue = request_queue
        self.all_client_response_queues = all_client_response_queues
        self.logger = logger
        self.log_level = log_level

        self.ready_batch_queue = std_queue.Queue(maxsize=num_actors * 2) # Internal queue

        self.assembly_thread = BatchAssemblyThread(
            request_queue=self.request_queue,
            ready_batch_queue=self.ready_batch_queue,
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            num_actors=num_actors,
            logger=self.logger,
            log_level=self.log_level
        )
        self.execution_thread = InferenceExecutionThread(
            ready_batch_queue=self.ready_batch_queue,
            all_client_response_queues=self.all_client_response_queues,
            model_apply_fn=model_apply_fn,
            model_variables=initial_model_variables, # Initial variables
            output_size=output_size, # Pass output_size
            logger=self.logger,
            log_level=self.log_level
        )

    def start(self):
        if self.logger and self.log_level >= INFO:
            self.logger.print("InferenceServicer starting threads...")
        self.assembly_thread.start()
        self.execution_thread.start()
        if self.logger and self.log_level >= INFO:
            self.logger.print("InferenceServicer threads started.")

    def stop(self, timeout_sec=5.0):
        if self.logger and self.log_level >= INFO:
            self.logger.print("InferenceServicer stopping threads...")

        self.assembly_thread.stop()
        self.execution_thread.stop()

        self.assembly_thread.join(timeout=timeout_sec)
        self.execution_thread.join(timeout=timeout_sec)

        if self.assembly_thread.is_alive():
            if self.logger and self.log_level >= WARN:
                self.logger.print("BatchAssemblyThread did not stop in time.")
        if self.execution_thread.is_alive():
            if self.logger and self.log_level >= WARN:
                self.logger.print("InferenceExecutionThread did not stop in time.")
        
        # Propagate SHUTDOWN_SENTINEL to all client response queues
        for q in self.all_client_response_queues:
            try:
                q.put_nowait(SHUTDOWN_SENTINEL)
            except Exception:  # pylint: disable=broad-except
                if self.logger and self.log_level >= WARN:
                    self.logger.print(f"InferenceServicer: Error putting SHUTDOWN_SENTINEL on a client response queue.")
        
        # Drain queues (optional, good for clean shutdown)
        self._drain_queue(self.request_queue, "Request Queue")
        self._drain_queue(self.ready_batch_queue, "Ready Batch Queue")

    def _drain_queue(self, q, q_name):
        drained_count = 0
        while True:
            try:
                q.get_nowait()
                drained_count +=1
            except (mp.queues.Empty, std_queue.Empty):
                break
            except Exception: # pylint: disable=broad-except
                break 
        if drained_count > 0 and self.logger and self.log_level >= DEBUG:
            self.logger.print(f"Drained {drained_count} items from {q_name} during servicer stop.")


    def update_model_variables(self, new_variables):
        # Pass the update to the execution thread
        self.execution_thread.update_variables(new_variables)

# ---- END Inference Servicer Components ----