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
from .actor_evaluator_logic import actor, evaluator, watcher as base_watcher, Buffer # Renamed watcher to base_watcher to avoid conflict

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
#   ERROR: Critical errors and experiment-ending events.
#   WARN:  Recoverable issues or unexpected states.
#   INFO:  High-level experiment progress (start/stop, episode summaries).
#   DEBUG: Per-step training summaries, checkpointing, detailed diagnostics.
#   TRACE: Extremely verbose, per-move or per-action logs.
# All logging output in this file should be gated by these levels, and no print() should appear unless guarded by log_level >= DEBUG or higher.

# Custom watcher that catches BaseException
def watcher(fn):
    @functools.wraps(fn)
    def _watcher_wrapper(*args, **kwargs):
        logger = None
        # Attempt to get logger from kwargs, similar to original base_watcher
        if 'logger' in kwargs and kwargs['logger'] is not None:
            logger = kwargs['logger']
        elif 'config' in kwargs and hasattr(kwargs['config'], 'path') and kwargs['config'].path:
            dir_name_for_log = fn.__name__ if fn.__name__ == "learner" else "unknown_watched_function"
            log_dir = os.path.join(kwargs['config'].path, "learner")
            os.makedirs(log_dir, exist_ok=True)
            log_num_suffix = f"_{kwargs['num']}" if 'num' in kwargs else ""
            should_also_print_to_stdout = not kwargs['config'].quiet if 'config' in kwargs and hasattr(kwargs['config'], 'quiet') else True
            logger = file_logger.FileLogger(log_dir, f"{fn.__name__}{log_num_suffix}_log", also_to_stdout=should_also_print_to_stdout)
        else: # Fallback to a basic print if no logger can be configured
            class PrintLogger:
                def print(self, *pargs, **pkwargs): print(*pargs, **pkwargs)
                def opt_print(self, *pargs, **pkwargs): print(*pargs, **pkwargs) # For compatibility
            logger = PrintLogger()
            logger.print(f"{fn.__name__}_watcher: Warning: No logger or config.path provided. Using basic print for exceptions.")

        try:
            if logger and hasattr(logger, 'print'):
                 logger.print(f"{fn.__name__} (watched): Starting execution.")

            kwargs_for_fn = kwargs.copy()
            import inspect
            sig = inspect.signature(fn)
            if 'logger' in sig.parameters and 'logger' not in kwargs_for_fn:
                kwargs_for_fn['logger'] = logger

            return fn(*args, **kwargs_for_fn)
        except BaseException as e:  # Catch BaseException
            if logger and hasattr(logger, 'print'):
                logger.print(f"--- CAUGHT BY WATCHER IN {fn.__name__} ---")
                logger.print(f"{fn.__name__}_watcher: A BaseException occurred in {fn.__name__}:\n{traceback.format_exc()}")
            else: # Fallback print if logger failed or is None
                print(f"--- FALLBACK WATCHER EXCEPTION PRINT FOR {fn.__name__} ---")
                print(f"{fn.__name__}_watcher: A BaseException occurred in {fn.__name__}:\n{traceback.format_exc()}")
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise # Re-raise critical exit exceptions
        finally:
            if logger and hasattr(logger, 'print'):
                logger.print(f"{fn.__name__} (watched): Finished execution (normally or after exception).")
    return _watcher_wrapper


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
        "async_mode",               # bool: Whether to use async MCTS within each actor.
        "async_batch_size",         # int: Batch size for async MCTS leaf evaluations.
        "async_virtual_loss",       # int: Virtual loss amount for async MCTS.
        "async_timeout",            # float: Timeout (s) for async MCTS leaf evaluation futures.
        "console_summary_log_freq_steps", # int: Frequency (in training steps) to log console summary in learner.
    ])):                                 # Default for remote_evaluator_timeout_ms can be set at instantiation.
  """Configuration for the JAX AlphaZero model and experiment."""
  # Default values for optional fields are handled where ConfigJAX is instantiated.
  pass


def alpha_zero_jax(config: ConfigJAX):
    """Main entry point for JAX AlphaZero."""
    import pyspiel  # Ensure pyspiel is imported in this scope
    main_key = jax.random.PRNGKey(config.master_seed)
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
    config_updates = {}
    if config.observation_shape is None or not config.observation_shape:
        config_updates["observation_shape"] = tuple(game.observation_tensor_shape())
    if config.output_size is None or config.output_size == 0:
        config_updates["output_size"] = game.num_distinct_actions()
    if config_updates:
        config = config._replace(**config_updates)
    servicer_key, spawn_key = jax.random.split(main_key)
    inference_model, inference_variables = model_jax.init_flax_model_and_variables(
        servicer_key, config, game)
    process_keys = jax.random.split(spawn_key, 1 + config.actors + config.evaluators)
    learner_key = process_keys[0]
    actor_seed_keys = process_keys[1 : 1 + config.actors]
    evaluator_seed_keys = process_keys[1 + config.actors : 1 + config.actors + config.evaluators]
    actor_initial_seeds = [jax.random.randint(key, (), 0, 2**31 - 1).item() for key in actor_seed_keys]
    evaluator_initial_seeds = [jax.random.randint(key, (), 0, 2**31 - 1).item() for key in evaluator_seed_keys]
    total_remote_clients = config.actors + config.evaluators
    all_client_response_queues = [mp.Queue() for _ in range(total_remote_clients)]
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
            "initial_seed": evaluator_initial_seeds[i],
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
        "all_client_response_queues": all_client_response_queues,
        "initial_flax_model": inference_model,
        "initial_variables": inference_variables
    }
    if config.log_level >= INFO:
        main_process_logger.print("Starting Learner in main process...")
    try:
        learner(
            game=learner_kwargs["game"],
            config=learner_kwargs["config"],
            actor_queues=learner_kwargs["actor_queues"],
            evaluator_queues=learner_kwargs["evaluator_queues"],
            prng_key=learner_kwargs["prng_key"],
            inference_request_queue=learner_kwargs["inference_request_queue"],
            all_client_response_queues=learner_kwargs["all_client_response_queues"],
            initial_flax_model=learner_kwargs["initial_flax_model"],
            initial_variables=learner_kwargs["initial_variables"]
        )
    except (KeyboardInterrupt, EOFError) as e:
        if config.log_level >= INFO:
            main_process_logger.print(f"Caught {type(e).__name__}, stopping AlphaZero JAX.")
    except Exception as e_learner_call:
        if config.log_level >= ERROR:
             main_process_logger.print(f"Generic exception during learner execution: {type(e_learner_call)} - {e_learner_call}\n{traceback.format_exc()}")
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


@watcher
def learner(*, game: pyspiel.Game, config: ConfigJAX, logger,
            actor_queues: list[spawn._ProcessQueue], 
            evaluator_queues: list[spawn._ProcessQueue], 
            prng_key: jax.random.PRNGKey,
            inference_request_queue: mp.Queue, 
            all_client_response_queues: list[mp.Queue],
            initial_flax_model, initial_variables): 
  """A learner that consumes actor trajectories and evaluator results, and updates the model."""
  # Start Python allocation tracing and RSS monitoring
  if logger and config.log_level >= DEBUG: # DEBUG log for train_batch_size
    logger.print(f"[LEARNER_CONFIG_DEBUG] train_batch_size: {config.train_batch_size}")

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
  
  # JIT Warmup for Inference Function
  @jax.jit
  def _batched_inference_fn_for_warmup(model_vars, obs_batch, legals_batch):
      policy_logits, value_preds = initial_flax_model.apply(
          model_vars, 
          obs_batch, 
          legals_mask=legals_batch,
          training=False, 
          mutable=False
      )
      policy_probs = jax.nn.softmax(policy_logits, axis=-1)
      return policy_probs, value_preds

  # Create dummy data for warmup
  # game object is available here in learner
  dummy_observation_shape = game.observation_tensor_shape()
  dummy_output_size = game.num_distinct_actions()
  dummy_obs_batch = jnp.zeros((1,) + tuple(dummy_observation_shape), dtype=jnp.float32)
  dummy_legals_batch = jnp.ones((1, dummy_output_size), dtype=jnp.bool_) # Changed from jnp.zeros to jnp.ones

  if logger and config.log_level >= INFO:
      logger.print(f"Learner: Warming up JIT for inference function with dummy_obs_batch shape: {dummy_obs_batch.shape}, dummy_legals_batch shape: {dummy_legals_batch.shape}...")
  warmup_start_time = time.time()
  try:
    _ = _batched_inference_fn_for_warmup(variables, dummy_obs_batch, dummy_legals_batch)
    # Optionally, block until compilation is done if JAX JIT is async by default in some setups.
    # For most cases, the first call will block until compilation finishes.
    # You could use .block_until_ready() on the result if needed, e.g. _[0].block_until_ready()
    if logger and config.log_level >= INFO:
        logger.print(f"Learner: JIT warmup completed in {time.time() - warmup_start_time:.4f}s.")
  except Exception as e_warmup:
    if logger and config.log_level >= WARN:
        logger.print(f"Learner: Error during JIT warmup: {e_warmup}. Continuing without warmup...")
        logger.print(traceback.format_exc())
  # ---- End JIT Warmup ----

  # JITted inference function for the servicer
  @jax.jit
  def _batched_inference_fn_for_servicer(model_vars, obs_batch, legals_batch):
      policy_logits, value_preds = initial_flax_model.apply(
          model_vars, 
          obs_batch, 
          legals_mask=legals_batch,
          training=False, 
          mutable=False
      )
      policy_probs = jax.nn.softmax(policy_logits, axis=-1)
      return policy_probs, value_preds

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

  # Orbax Checkpointing Setup
  managed_ckpt_dir = os.path.join(config.path, "checkpoints_jax_managed")
  latest_ckpt_target_dir = os.path.join(config.path, "checkpoints_jax_latest_atomic") 

  if logger and config.log_level >= INFO:
      logger.print(f"Managed checkpoints will be saved to: {managed_ckpt_dir}")
      logger.print(f"Latest checkpoint will be at: {latest_ckpt_target_dir}")

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
          # We are saving a dict: {'variables': variables, 'opt_state': opt_state, 'replay_buffer': replay_buffer}
          # The replay_buffer is initialized before this block.
          restored_mngr_state = checkpoint_manager.restore(
              step=initial_step,
              args=ocp.args.Composite( # Use Composite to restore specific parts
                  variables=ocp.args.StandardRestore(variables),
                  opt_state=ocp.args.StandardRestore(opt_state),
                  replay_buffer=ocp.args.StandardRestore(replay_buffer)
              )
          )
          if restored_mngr_state:
              variables = restored_mngr_state['variables']
              opt_state = restored_mngr_state['opt_state']
              replay_buffer = restored_mngr_state['replay_buffer']
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
    
    return new_variables, new_opt_state, loss_val, p_loss, v_loss

  last_time = time.time()
  start_time = last_time # For throughput stats
  total_trajectories = 0
  
  training_step_count = initial_step
  loop_iteration = 0

  DEFAULT_STATS_LOG_PERIOD = 1000
  last_stats_log_iter = 0
  STATS_LOG_INTERVAL = 1000

  states_accumulated_since_last_train = 0
  trajectories_accumulated_since_last_train = 0
  time_of_last_train_step_or_start = time.time()

  accumulated_inference_queue_size = 0
  inference_queue_size_samples = 0

  last_rate_log_time = time.time()
  RATE_LOG_INTERVAL = 30.0
  states_since_last_rate_log = 0
  training_steps_since_last_rate_log = 0

  current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')

  if logger: logger.print(f"Learner starting. Initial training_step_count: {training_step_count}. Max steps: {config.max_steps}. Loop iterations will start from 1.")

  last_summary_log_time = time.time()
  SUMMARY_LOG_INTERVAL = 30.0

  last_console_summary_log_train_step = initial_step

  for current_loop_iteration_raw in itertools.count(1):
    loop_iteration = current_loop_iteration_raw

    # Accumulate inference queue size for averaging
    try:
        current_inf_q_size = inference_request_queue.qsize()
        accumulated_inference_queue_size += current_inf_q_size
        inference_queue_size_samples += 1
    except NotImplementedError: # qsize is not implemented on all platforms (e.g. macOS for mp.Queue)
        pass

    if logger and config.log_level >= TRACE:
        logger.print(f"Learner: Main loop iteration {loop_iteration} BEGIN.")

    # Collect trajectories from actor_queues
    num_states_this_iter = 0
    num_trajectories_this_iter = 0
    trajectories_to_process = []
    # Drain actor queues to avoid backlog
    if not actor_queues:
        if logger and config.log_level >= WARN:
            logger.print("Learner: No actor queues configured or list is empty. Cannot collect trajectories. Waiting briefly.")
        time.sleep(1) # Prevent busy loop if no actors

    for queue_idx, queue in enumerate(actor_queues):
        while True:
            try:
                traj = queue.get_nowait()
                if traj: # Ensure trajectory is not None
                    trajectories_to_process.append(traj)
                else: # Should not happen with get_nowait unless queue stores None
                    if logger and config.log_level >= WARN:
                        logger.print(f"Learner: Received None from actor_queue {queue_idx}. Skipping.")
            except spawn.Empty:
                break # Queue is empty for now
            except Exception as e:
                if logger:
                    logger.print(f"Learner: Error draining actor_queue {queue_idx}: {e}")
                break # Avoid busy-looping on a problematic queue

    for traj in trajectories_to_process:
        if not hasattr(traj, "states"): 
            if logger:
                logger.print(f"Learner: Received object of type {type(traj)} from actor queue: {repr(traj)[:200]}. Skipping.")
            continue

        total_trajectories += 1 # Global counter for all trajectories seen by this learner instance
        num_trajectories_this_iter += 1
        trajectories_accumulated_since_last_train += 1
        game_lengths.add(len(traj))
        game_lengths_hist.add(len(traj))
        
        current_num_states_in_traj = 0
        if hasattr(traj, 'states') and traj.states is not None:
             current_num_states_in_traj = len(traj.states)
        num_states_this_iter += current_num_states_in_traj
        states_accumulated_since_last_train += current_num_states_in_traj
        states_since_last_rate_log += current_num_states_in_traj  # Add to rate logging counter
        
        current_player_return = traj.returns 
        outcome_bucket_id = None 

        if current_player_return is not None: 
            # Determine outcome based on player 0's perspective if returns is an array
            # Assumes traj.returns is a numpy array like [P0_return, P1_return, ...]
            player0_return = current_player_return[0] if isinstance(current_player_return, (np.ndarray, list)) and len(current_player_return) > 0 else current_player_return
            
            if player0_return > 0: outcome_bucket_id = 0 
            elif player0_return < 0: outcome_bucket_id = 1 
            elif player0_return == 0: outcome_bucket_id = 2 
            else:
                if logger:
                    logger.print(f"Learner: Trajectory with unexpected player 0 return value: {player0_return} (original: {current_player_return}). Not mapping to outcome.")
        else: 
             if logger and config.log_level >= DEBUG: # Log Nones only at DEBUG
                logger.opt_print(f"Learner: Trajectory with None return value. Not adding to outcomes histogram.")

        if outcome_bucket_id is not None: 
            try:
                outcomes.add(outcome_bucket_id) 
            except (ValueError, KeyError, IndexError) as e_hist: 
                if logger:
                    logger.print(f"Learner: Error adding outcome bucket_id '{outcome_bucket_id}' (return: {current_player_return}) to histogram: {e_hist}. Names: {outcomes._names if hasattr(outcomes, '_names') else 'N/A'}")
        
        for transition in traj.states:
            train_input = model_jax.TrainInputJAX(
                observation=transition.observation,
                legals_mask=transition.legals_mask,
                policy_target=transition.policy,
                value_target=jnp.array(traj.returns[transition.current_player], dtype=jnp.float32)
            )
            replay_buffer.append(train_input)
    # --- End Data Collection ---

    # Periodic Rate Logging
    current_time = time.time()
    if current_time - last_rate_log_time >= RATE_LOG_INTERVAL:
        elapsed = current_time - last_rate_log_time
        if elapsed > 0:
            states_per_sec = states_since_last_rate_log / elapsed
            training_steps_per_sec = training_steps_since_last_rate_log / elapsed
            if logger and config.log_level >= INFO:
                logger.print(f"Rates (30s avg): {states_per_sec:.1f} states/s, {training_steps_per_sec:.1f} training steps/s")
        # Reset counters
        states_since_last_rate_log = 0
        training_steps_since_last_rate_log = 0
        last_rate_log_time = current_time

    # Periodic General Logging
    now_for_general_log = time.time()
    seconds_this_loop_iter = now_for_general_log - last_time
    last_time = now_for_general_log
    
    # Max Steps Check
    if config.max_steps > 0 and training_step_count >= config.max_steps:
        if logger and config.log_level >= INFO: # Log this at INFO
            logger.print(f"Learner: Max training steps {config.max_steps} reached (current: {training_step_count}). Exiting learner main loop.")
        break # EXIT POINT for the main learner loop

    # New Time-based Summary Logging Block
    current_time_for_summary = time.time()
    if current_time_for_summary - last_summary_log_time >= SUMMARY_LOG_INTERVAL:
        servicer_stats = servicer.inference_stats() if servicer else {}
        avg_inf_batch_size = servicer_stats.get('avg_execution_batch_size', 'N/A') # Use execution batch size
        inf_states_per_sec = servicer_stats.get('inference_per_second', 'N/A')
        avg_inf_time_ms = servicer_stats.get('avg_inference_time_ms_per_batch', 'N/A')
        avg_wait_time_ms = servicer_stats.get('avg_wait_time_ms_for_request', 'N/A')
        # Actor trajectories per second can be estimated from states_per_s_loop_iter and average game length
        # This is a rough approximation based on recent loop iteration.
        # A more accurate actor throughput would require actors to report trajectory counts over time.
        avg_game_len = game_lengths.avg if game_lengths.num > 0 else 1 # Avoid division by zero
        actor_traj_s_approx = (num_states_this_iter / seconds_this_loop_iter / avg_game_len) if seconds_this_loop_iter > 0 and avg_game_len > 0 else "N/A"

        loss_val_for_summary = current_total_loss # Use the most recent loss from training step

        if logger and config.log_level >= INFO:
            logger.print(f"Summary: LoopIter: {loop_iteration}, TrainStep: {training_step_count}, "
                         f"BufSize: {len(replay_buffer)}, Loss: {loss_val_for_summary:.3f}, "
                         f"ActorTraj/s (approx): {actor_traj_s_approx if isinstance(actor_traj_s_approx, str) else actor_traj_s_approx:.2f}, "
                         f"Inference(states/s): {inf_states_per_sec if isinstance(inf_states_per_sec, str) else inf_states_per_sec:.1f}, "
                         f"AvgInfBatch: {avg_inf_batch_size if isinstance(avg_inf_batch_size, str) else avg_inf_batch_size:.1f}, "
                         f"AvgInfTime(ms): {avg_inf_time_ms if isinstance(avg_inf_time_ms, str) else avg_inf_time_ms:.2f}, "
                         f"AvgWaitTime(ms): {avg_wait_time_ms if isinstance(avg_wait_time_ms, str) else avg_wait_time_ms:.2f}")
        last_summary_log_time = current_time_for_summary

    # --- Training Step ---
    training_performed_this_iteration = False
    # training_performed_this_iteration is reset at the start of the outer loop implicitly by not being set
    # It will be set to True if any training step in the burst below occurs.
    if len(replay_buffer) >= config.train_batch_size and config.train_batch_size > 0:
      actual_train_steps_this_burst = 0
      for _ in range(config.replay_buffer_reuse): # Use the config parameter
          if training_step_count >= config.max_steps:
              break
          if len(replay_buffer) < config.train_batch_size: # Buffer might empty during burst
              if logger and config.log_level >= DEBUG and actual_train_steps_this_burst > 0:
                   logger.print(f"Learner: Replay buffer emptied during training burst after {actual_train_steps_this_burst} steps. Burst intended for {config.replay_buffer_reuse} steps.")
              break

          batch_data = replay_buffer.sample(config.train_batch_size)
          # No DEBUG log for sampling here, it's too frequent if replay_buffer_reuse > 1
          
          try:
            stacked_input = model_jax.TrainInputJAX.stack(batch_data)
            batch_obs_jnp = jnp.array(stacked_input.observation, dtype=jnp.float32)
            batch_legals_jnp = jnp.array(stacked_input.legals_mask, dtype=jnp.bool_)
            batch_policy_jnp = jnp.array(stacked_input.policy_target, dtype=jnp.float32)
            batch_value_jnp = jnp.array(stacked_input.value_target, dtype=jnp.float32)
            
            variables, opt_state, total_loss_val, policy_loss_val, value_loss_val = train_step_fn(
                variables, opt_state, batch_obs_jnp, batch_legals_jnp, batch_policy_jnp, batch_value_jnp
            )
            
            training_step_count += 1 # Increment after successful training step
            training_steps_since_last_rate_log += 1  # Add to rate logging counter
            training_performed_this_iteration = True # Set if any step in the burst happens
            actual_train_steps_this_burst += 1
            
            if servicer: # Update servicer model after successful training step
                servicer.update_model_variables(variables)
            
            current_total_loss, current_policy_loss, current_value_loss = total_loss_val, policy_loss_val, value_loss_val
            
            loss_log_msg = f"Training Step: {training_step_count}, Total Loss: {current_total_loss:.4f}, Policy Loss: {current_policy_loss:.4f}, Value Loss: {current_value_loss:.4f}"
            if logger and config.log_level >= DEBUG: # This log might be very verbose if reuse > 1
              logger.print(loss_log_msg)
            
            # Orbax Checkpointing: Save
            save_target_pytree = {'variables': variables, 'opt_state': opt_state}
            try:
                if checkpoint_manager.should_save(training_step_count): # Use training_step_count
                    checkpoint_manager.save(
                        training_step_count, # Use training_step_count
                        args=ocp.args.Composite(
                            variables=ocp.args.StandardSave(variables),
                            opt_state=ocp.args.StandardSave(opt_state),
                            replay_buffer=ocp.args.StandardSave(replay_buffer), # ADDED for replay_buffer
                            metrics=ocp.args.JsonSave({
                                'step': training_step_count, # Use training_step_count
                                'policy_head_loss': float(policy_loss_val),
                                'value_head_loss': float(value_loss_val)
                            })
                        )
                    )
                    if logger and config.log_level >= DEBUG:
                        logger.opt_print(f"Saved checkpoint for training_step {training_step_count} via manager to {managed_ckpt_dir}")
                
                latest_checkpointer.save(
                    latest_ckpt_target_dir, # This is a directory
                    args=ocp.args.PyTreeSave(item=variables), # Saves 'variables' pytree
                    force=True # Overwrite if exists (it will be a new temp then rename)
                )
                if logger and config.log_level >= DEBUG:
                    logger.opt_print(f"Saved atomic latest checkpoint (variables) for training_step {training_step_count} to {latest_ckpt_target_dir}")
                # save_path_for_broadcast = latest_ckpt_target_dir # This path is broadcast (original comment)
                
            except Exception as e:
                err_msg = f"Error saving checkpoint for training_step {training_step_count}: {e}"
                if logger:
                    logger.print(err_msg)
                    logger.print(traceback.format_exc())
                break # Break from the inner reuse loop on error
            # ---- End Orbax Checkpointing ----

            states_accumulated_since_last_train = 0
            trajectories_accumulated_since_last_train = 0
            time_of_last_train_step_or_start = time.time()

            # Add a small sleep if this burst is purely on "old" data relative to the current outer loop iteration
            # and we want to space out these rapid reuse steps.
            if num_trajectories_this_iter == 0:
                time.sleep(0.01) # Sleep 10ms to make reuse steps less back-to-back. Consider making configurable.

          except AttributeError as e_attr: # Catch if TrainInputJAX.stack is missing or similar
            error_msg = f"Learner: Error during training data preparation (possibly missing TrainInputJAX.stack): {e_attr}"
            if logger:
              logger.print(error_msg)
              logger.print(traceback.format_exc()) # Print traceback for attribute errors
            current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')
            break # Break from the inner reuse loop on error
          except Exception as e_train: # Catch any other error during training
            error_msg = f"Learner: Error during training step {training_step_count}: {e_train}"
            if logger:
              logger.print(error_msg)
              logger.print(traceback.format_exc()) # Print traceback for training errors
            current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')
            break # Break from the inner reuse loop on error
      
      if actual_train_steps_this_burst > 0 and logger and config.log_level >= DEBUG:
          logger.print(f"Learner: Completed training burst of {actual_train_steps_this_burst} steps (intended: {config.replay_buffer_reuse}). Current training_step_count: {training_step_count}")

    else: # Not enough data in replay buffer for a training batch
      current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')
      if logger and config.log_level >= TRACE: # TRACE level for this frequent message
        logger.opt_print(f"Learner Loop Iter: {loop_iteration}, Replay buffer not full enough for training. Size: {len(replay_buffer)}/{config.train_batch_size}")
      if num_trajectories_this_iter == 0 and not training_performed_this_iteration:
          time.sleep(0.01) # Sleep 10ms to yield CPU

    # Collect evaluation results (non-blocking)
    for i, evac_queue in enumerate(evaluator_queues): 
        while True:
            try:
                eval_outcome = evac_queue.get_nowait()
                if isinstance(eval_outcome, tuple) and len(eval_outcome) == 2:
                    difficulty_idx, outcome = eval_outcome
                    if 0 <= difficulty_idx < len(evals):
                        evals[difficulty_idx].append(outcome)
                elif isinstance(eval_outcome, (int, float)): 
                    if 0 <= i < len(evals):
                         evals[i].append(eval_outcome)
                # else:
                #    if logger and config.log_level >= WARN: # Log if unexpected type from eval queue
                #        logger.print(f"Learner: Received unexpected item from eval queue {i}: {type(eval_outcome)}")
            except spawn.Empty: 
                break
            except Exception as e: 
                if logger: 
                  logger.print(f"Learner: Error processing evaluator queue {i}: {e}")
                break 

    # Log to data_logger (learner.jsonl)
    # This logging happens if training was performed OR if it's a periodic stats log iteration
    seconds_for_this_train_period = time.time() - time_of_last_train_step_or_start
    if data_log and (training_performed_this_iteration or (loop_iteration - last_stats_log_iter >= DEFAULT_STATS_LOG_PERIOD) or loop_iteration == 1) :
        metrics_to_log = {
            "loop_iteration": loop_iteration,
            "training_step": training_step_count,
            "total_states_seen_by_buffer": replay_buffer.total_seen,
            "replay_buffer_size": len(replay_buffer),
            "states_per_s_loop_iter": num_states_this_iter / seconds_this_loop_iter if seconds_this_loop_iter > 0 else 0,
            "trajectories_per_s_loop_iter": num_trajectories_this_iter / seconds_this_loop_iter if seconds_this_loop_iter > 0 else 0,
            "states_in_train_period": states_accumulated_since_last_train,
            "trajectories_in_train_period": trajectories_accumulated_since_last_train,
            "seconds_for_train_period": seconds_for_this_train_period,
            "total_trajectories_global": total_trajectories, # Global count
            "game_length": game_lengths.as_dict,
            "game_length_hist": game_lengths_hist.data,
            "outcomes": outcomes.data,
            "value_accuracy": [v.as_dict for v in value_accuracies],
            "value_prediction": [v.as_dict for v in value_predictions],
            "eval": {
                "count": evals[0].total_seen if evals and evals[0] else 0,
                "results": [sum(e.data) / len(e.data) if len(e.data) > 0 else 0 for e in evals]
            },
            "loss": {
                "total": float(current_total_loss),
                "policy": float(current_policy_loss),
                "value": float(current_value_loss),
            },
        }
        data_log.write(metrics_to_log)
        last_stats_log_iter = loop_iteration # Reset for periodic logging

        # Also print a summary to console logger at INFO level for this data_log event
        if logger and config.log_level >= INFO:
            # Condition for logging the console summary based on training steps
            ready_to_log_based_on_train_step = (
                training_step_count == initial_step or # Log at the very first relevant step
                (training_step_count - last_console_summary_log_train_step >= config.console_summary_log_freq_steps)
            )

            if training_performed_this_iteration and ready_to_log_based_on_train_step and config.console_summary_log_freq_steps > 0:
                console_summary_msg = (
                    f"LoopIter: {loop_iteration}, TrainStep: {training_step_count}, "
                    f"BufSize: {len(replay_buffer)}, Loss: {current_total_loss:.3f}, "
                    f"ActorTraj/s (iter): {num_trajectories_this_iter / seconds_this_loop_iter:.1f}"
                )
                logger.print(console_summary_msg)
                last_console_summary_log_train_step = training_step_count


    # Per-iteration DEBUG logging block
    if logger and config.log_level >= DEBUG:
      log_this_iteration_details = (
          training_performed_this_iteration or
          num_trajectories_this_iter > 0 or
          loop_iteration == 1 or
          (loop_iteration % 200 == 1) # Log every 200 iterations, and on the first one.
      )

      if log_this_iteration_details:
        log_message_timing = (
            f"Loop Iter: {loop_iteration}, Train Steps: {training_step_count}, "
            f"Iter Duration: {seconds_this_loop_iter:.3f}s, "
            f"Data states/s in iter: {num_states_this_iter / seconds_this_loop_iter:.1f} (n={num_states_this_iter}), "
            f"Traj/s in iter: {num_trajectories_this_iter / seconds_this_loop_iter:.1f} (n={num_trajectories_this_iter})"
        )
        log_message_buffer = f"Buffer size: {len(replay_buffer)}. Total states seen by buffer: {replay_buffer.total_seen}. Replay buffer reuse: {config.replay_buffer_reuse}" # Added reuse
        
        logger.print(f"--- Learner Loop Iteration {loop_iteration} (Train Step {training_step_count}) ---") # Start of iteration marker for DEBUG
        logger.print(log_message_timing)
        logger.print(log_message_buffer)
        if not training_performed_this_iteration and len(replay_buffer) < config.train_batch_size :
            logger.print(f"No training this iteration. Buffer: {len(replay_buffer)}/{config.train_batch_size}")
        if num_trajectories_this_iter == 0:
            logger.print("No new trajectories received this iteration.")

        # --- [LEARNER_INFO] Average Inference Queue Size ---
        if loop_iteration % STATS_LOG_INTERVAL == 0: # Check if it's time to log this specific average
            if inference_queue_size_samples > 0:
                avg_inf_q_size = accumulated_inference_queue_size / inference_queue_size_samples
                logger.print(f"Avg Inference Queue Size (last {STATS_LOG_INTERVAL} iters): {avg_inf_q_size:.2f} (samples: {inference_queue_size_samples})")
                accumulated_inference_queue_size = 0 # Reset for next interval
                inference_queue_size_samples = 0 # Reset for next interval
            else: # If it's time to log but no samples (e.g. qsize not impl or interval too short after reset)
                # Removed the log message as requested by the user.
                pass # Do nothing if qsize is not implemented or no samples
        
        logger.print(f"--- Learner Loop Iteration {loop_iteration} END ---") # End of iteration marker for DEBUG

    if logger and config.log_level >= TRACE: logger.print("") # Add a newline for readability in FileLogger at TRACE

    # Broadcast checkpoint path if a new one was saved (now done after training step)
    # The variable save_path_for_broadcast is set within the training block.
    # This broadcast_fn is not defined in this scope. It was part of old TF learner.
    # For JAX, actors/evaluators typically load from a known checkpoint location.
    # If broadcasting is still desired, broadcast_fn needs to be passed or handled differently.
    # For now, commenting out as its original mechanism is not present.
    # if save_path_for_broadcast: 
    #     broadcast_msg = f"Broadcasting checkpoint: {save_path_for_broadcast}" 
    #     if logger and config.log_level >= DEBUG:
    #       logger.opt_print(broadcast_msg) 
    #     # broadcast_fn(save_path_for_broadcast) # broadcast_fn is not available here
  
  if logger: # Log before the final "finished" message
      logger.print(f"Learner: Exited main loop after {loop_iteration} iterations. Final training_step_count: {training_step_count}.")

  # This is the "JAX Learner finished." message source
  if logger and config.log_level >= INFO:
    final_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] JAX Learner finished."
    logger.print(final_msg)

  # Stop the inference servicer before exiting
  if servicer:
    servicer.stop()

def set_external_libraries_log_level(log_level):
    """Set logging level for external libraries (Orbax, JAX, Flax, absl)."""
    external_lib_py_level = logging.ERROR # Enforce ERROR for external libraries to reduce verbosity.

    for logger_name in [
        "orbax", "orbax.checkpoint", "jax", "flax", "absl", "absl.logging"
    ]:
        logging.getLogger(logger_name).setLevel(external_lib_py_level)
    
    # Also set the root logger for absl to avoid it overriding the specific ones sometimes
    # absl.logging.set_verbosity only affects absl's own messages if they don't go via python logging
    if external_lib_py_level <= logging.INFO:
        absl.logging.set_verbosity('info')
    elif external_lib_py_level <= logging.WARNING:
        absl.logging.set_verbosity('warning')
    else:
        absl.logging.set_verbosity('error')

    # Note: The original mapping based on log_level is removed to enforce a fixed
    # higher level for these libraries.
    # level_map = {
    #     0: logging.ERROR,   # ERROR
    #     1: logging.WARNING, # WARN
    #     2: logging.INFO,    # INFO
    #     3: logging.DEBUG,   # DEBUG
    #     4: logging.NOTSET,  # TRACE (or use DEBUG)
    # }
    # py_level = level_map.get(log_level, logging.INFO)
    # ... (old code that used py_level)


# Inference Servicer Components

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
        # Stats
        self.stats_lock = threading.Lock()
        self.batches_assembled_count = 0
        self.requests_in_assembled_batches_count = 0
        self.total_wait_time_for_requests_sec = 0.0
        self.requests_processed_count = 0 # Total requests pulled from queue
        self.start_time = time.time()

    def get_stats(self):
        with self.stats_lock:
            uptime_sec = time.time() - self.start_time
            avg_requests_per_batch = self.requests_in_assembled_batches_count / self.batches_assembled_count if self.batches_assembled_count > 0 else 0
            # Note: avg_wait_time is an approximation based on total wait time.
            return {
                "batches_assembled": self.batches_assembled_count,
                "requests_in_batches": self.requests_in_assembled_batches_count,
                "avg_requests_per_assembled_batch": avg_requests_per_batch,
                "total_requests_pulled": self.requests_processed_count,
                "thread_uptime_sec": uptime_sec
            }

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


                wait_start_time = time.time()
                raw_request_tuple = self.request_queue.get(timeout=timeout_for_get)
                actual_wait_time = time.time() - wait_start_time
                with self.stats_lock:
                    self.total_wait_time_for_requests_sec += actual_wait_time
                    self.requests_processed_count += 1



                if raw_request_tuple == SHUTDOWN_SENTINEL:
                    if self.logger and self.log_level >= INFO:
                        self.logger.print("BatchAssemblyThread received SHUTDOWN_SENTINEL on request_queue. Signaling ready_batch_queue and exiting.")
                    self.ready_batch_queue.put(SHUTDOWN_SENTINEL) 
                    break 

                if self.logger and self.log_level >= 4: # TRACE
                    self.logger.print(f"BatchAssemblyThread: Received raw request: {raw_request_tuple}")

                try:
                    inference_request_obj = InferenceRequest.from_tuple(raw_request_tuple)
                except ValueError as e:
                    if self.logger and self.log_level >= WARN:
                        self.logger.print(f"BatchAssemblyThread: Error deserializing request: {e}. Skipping request: {raw_request_tuple}")
                    continue # Skip this malformed request

                # The actor_id from the deserialized request is the index for all_client_response_queues
                # origin_response_queue_idx = inference_request_obj.actor_id # OLD INCORRECT LINE
                req_actor_id = inference_request_obj.actor_id
                if req_actor_id >= 1000: # It's an evaluator
                    # self.num_actors is config.actors
                    evaluator_num_relative = req_actor_id - 1000
                    origin_response_queue_idx = self.num_actors + evaluator_num_relative
                else: # It's an actor
                    origin_response_queue_idx = req_actor_id
                
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
                    with self.stats_lock:
                        self.batches_assembled_count += 1
                        self.requests_in_assembled_batches_count += len(current_batch_requests)
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
        # Stats
        self.stats_lock = threading.Lock()
        self.inference_batches_executed_count = 0
        self.total_inferences_processed_count = 0 # Sum of batch sizes
        self.total_model_inference_time_sec = 0.0
        self.start_time = time.time()

    def get_stats(self):
        with self.stats_lock:
            uptime_sec = time.time() - self.start_time
            avg_inference_time_ms_per_batch = (self.total_model_inference_time_sec * 1000 / self.inference_batches_executed_count) if self.inference_batches_executed_count > 0 else 0
            avg_states_per_second = (self.total_inferences_processed_count / uptime_sec) if uptime_sec > 0 else 0
            avg_batch_size_inferred = self.total_inferences_processed_count / self.inference_batches_executed_count if self.inference_batches_executed_count > 0 else 0
            return {
                "inference_batches_executed": self.inference_batches_executed_count,
                "total_inferences_processed": self.total_inferences_processed_count,
                "avg_inference_time_ms_per_batch": avg_inference_time_ms_per_batch,
                "avg_states_inferred_per_sec": avg_states_per_second,
                "avg_actual_batch_size_inferred": avg_batch_size_inferred,
                "thread_uptime_sec": uptime_sec
            }

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
                
                inference_start_time = time.time()
                policy_probs_batch, value_output_batch = self.model_apply_fn(
                    current_vars, obs_array_batch, legals_array_batch) 
                inference_duration_sec = time.time() - inference_start_time
                with self.stats_lock:
                    self.inference_batches_executed_count += 1
                    self.total_inferences_processed_count += obs_array_batch.shape[0]
                    self.total_model_inference_time_sec += inference_duration_sec

                
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

                if self.logger and self.log_level >= 4: # TRACE
                    self.logger.print(f"InferenceExecutionThread: Sending response to client {origin_idx} for req {req_id}: {response_tuple}")

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

    def inference_stats(self):
        """Collects and computes aggregated statistics from the assembler and executor threads."""
        assembly_stats = self.assembly_thread.get_stats()
        execution_stats = self.execution_thread.get_stats()

        # Combine and derive further stats
        # Ensure to handle potential division by zero if counts are zero.
        total_requests_pulled = assembly_stats.get("total_requests_pulled", 0)
        batches_assembled = assembly_stats.get("batches_assembled", 0)
        
        total_inferences_processed = execution_stats.get("total_inferences_processed", 0)
        inference_batches_executed = execution_stats.get("inference_batches_executed", 0)
        total_model_inference_time_sec = execution_stats.get("total_model_inference_time_sec", 0.0)
        
        # Average wait time for a request before being assembled into a batch
        # This is a rough estimate based on total wait time / total requests seen by assembler.
        # More accurate per-request wait time would require timing each request individually.
        total_wait_time_for_requests_sec = assembly_stats.get("total_wait_time_for_requests_sec", 0.0)
        avg_wait_time_ms = (total_wait_time_for_requests_sec * 1000 / total_requests_pulled) if total_requests_pulled > 0 else 0.0

        # Average batch size (from assembler's perspective)
        avg_assembled_batch_size = assembly_stats.get("avg_requests_per_assembled_batch", 0.0)
        
        # Average batch size (from executor's perspective - actual number of inferences in a batch)
        avg_execution_batch_size = execution_stats.get("avg_actual_batch_size_inferred", 0.0)

        # Inference throughput (states processed per second by the model)
        # Use executor's uptime and total inferences for more accurate model throughput
        executor_uptime_sec = execution_stats.get("thread_uptime_sec", 0.0)
        inference_per_second = (total_inferences_processed / executor_uptime_sec) if executor_uptime_sec > 0 else 0.0
        
        # Average model inference time per batch (from executor)
        avg_inference_time_ms_per_batch = execution_stats.get("avg_inference_time_ms_per_batch", 0.0)


        return {
            "total_requests_pulled_by_assembler": total_requests_pulled,
            "batches_assembled": batches_assembled,
            "avg_assembled_batch_size": avg_assembled_batch_size,
            "total_inferences_processed_by_executor": total_inferences_processed,
            "inference_batches_executed": inference_batches_executed,
            "avg_execution_batch_size": avg_execution_batch_size,
            "inference_per_second": inference_per_second, # states/sec processed by model
            "avg_inference_time_ms_per_batch": avg_inference_time_ms_per_batch, # model execution time for a batch
            "avg_wait_time_ms_for_request": avg_wait_time_ms, # Approximate time request waits in queue before assembly
            "assembly_thread_stats": assembly_stats, # Raw stats from assembler
            "execution_thread_stats": execution_stats, # Raw stats from executor
        }

# ---- END Inference Servicer Components ----