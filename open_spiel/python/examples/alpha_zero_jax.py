import collections
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
import random
import time
import itertools # For itertools.count in learner
import numpy as np # For np.zeros, np.random in copied _play_game, TrajectoryState
import pyspiel # For game object and types
import os # Added for checkpointing directory management
import functools # For watcher decorator
import traceback # For watcher decorator
import json # Added for saving config to JSON
import sys # Added for sys.exit
import tempfile # Added for tempfile.mkdtemp
from absl import app # Added for executable entry point
from absl import flags # Added for executable entry point

from open_spiel.python.algorithms.alpha_zero_jax import model_jax
from open_spiel.python.algorithms.alpha_zero_jax import evaluator_jax # For AlphaZeroEvaluatorJAX
# Placeholder for future imports like spawn, file_logger, data_logger, stats
from open_spiel.python.utils import spawn, file_logger, data_logger, stats # Activated file_logger, spawn, data_logger, stats
from open_spiel.python.algorithms import mcts # Activated mcts

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


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
    ])):
  """A config for the JAX AlphaZero model/experiment."""
  # To allow None defaults for Optional fields in namedtuple, provide them at instantiation.
  # Default values for new optional fields can be handled in the main script creating the ConfigJAX instance.
  pass

# Helper classes copied from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
class TrajectoryState(object):
  """A particular point along a trajectory."""

  def __init__(self, observation, current_player, legals_mask, action, policy,
               value):
    self.observation = observation
    self.current_player = current_player
    self.legals_mask = legals_mask
    self.action = action
    self.policy = policy
    self.value = value


class Trajectory(object):
  """A sequence of observations, actions and policies, and the outcomes."""

  def __init__(self):
    self.states = []
    self.returns = None


class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    # Efficiently trim the beginning of the list if it exceeds max_size
    if len(self.data) > self.max_size:
        self.data = self.data[len(self.data) - self.max_size:]


  def sample(self, count):
    # Ensure count is not greater than the number of items in data
    count = min(count, len(self.data))
    if count == 0:
        return []
    return random.sample(self.data, count)


# Watcher decorator from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
def watcher(fn):
  """A decorator to fn/processes that gives a logger and logs exceptions."""
  @functools.wraps(fn)
  def _watcher(*, config, num=None, **kwargs):
    """Wrap the decorated function."""
    name = fn.__name__
    if num is not None:
      name += "-" + str(num)
    # Assuming config.path is available and config.quiet controls verbosity
    # The FileLogger from open_spiel.python.utils.file_logger is used here.
    # If config.path is None, FileLogger might raise an error or not log to a file.
    # Ensure config.path is appropriately set before calling functions decorated with watcher.
    if config.path:
        logger_path = config.path
    else:
        # Fallback or error handling if config.path is not set, as FileLogger needs a path.
        # For now, let's assume it's an error or a temporary path might be used.
        # This behavior should align with how file_logger.FileLogger handles a None path.
        # Alternatively, print a warning and use a dummy logger.
        print(f"Warning: config.path is not set for watcher on {name}. Logging to files might be disabled or fail.")
        logger_path = "." # Default to current directory, though FileLogger might not like this without a specific filename pattern.


    with file_logger.FileLogger(logger_path, name, config.quiet if hasattr(config, 'quiet') else True) as logger:
      print(f"{name} started")
      logger.print(f"{name} started")
      try:
        # Pass the logger to the wrapped function if it accepts it.
        # The original @watcher passes logger, so we do too.
        if 'logger' in fn.__code__.co_varnames:
            return fn(config=config, logger=logger, num=num, **kwargs)
        else: # If the wrapped function doesn't expect 'logger' or 'num' in this way.
            # We might need to adjust based on specific function signatures.
            # For actor, learner, evaluator, they do expect 'logger' and 'num' is handled by name.
            return fn(config=config, num=num, logger=logger, **kwargs) # Original watcher passes num to _watcher, not fn directly
      except Exception as e:
        logger.print("\n".join([
            "",
            f" Exception caught in {name} ".center(60, "="),
            traceback.format_exc(),
            "=" * 60,
        ]))
        print(f"Exception caught in {name}: {e}")
        raise
      finally:
        logger.print(f"{name} exiting")
        print(f"{name} exiting")
  return _watcher


# _init_bot function from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
def _init_bot(config: ConfigJAX, game: pyspiel.Game, evaluator_: mcts.Evaluator, evaluation: bool):
  """Initializes an MCTS bot with a JAX-based AlphaZero evaluator.

  This function configures an MCTS bot, specifically setting up the
  Dirichlet noise for exploration during self-play (if not in evaluation mode).
  The core of the MCTS search relies on the provided `evaluator_`,
  which for this JAX implementation is an `AlphaZeroEvaluatorJAX` instance.

  Args:
    config: The `ConfigJAX` object containing hyperparameters like UCT constant,
      max simulations, and policy noise parameters.
    game: The `pyspiel.Game` instance for which the bot is being created.
    evaluator_: An `mcts.Evaluator` instance. In the JAX AlphaZero context,
      this is expected to be an `AlphaZeroEvaluatorJAX` that uses a Flax model
      for policy and value predictions.
    evaluation: A boolean indicating whether the bot is used for evaluation.
      If True, Dirichlet noise is disabled for more deterministic play.

  Returns:
    An `mcts.MCTSBot` configured for the JAX AlphaZero algorithm.
  """
  # Dirichlet noise is added to the policy prior for exploration during training (self-play).
  # It's disabled during evaluation for a more deterministic assessment of the agent's strength.
  noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
  return mcts.MCTSBot(
      game, # The game environment.
      config.uct_c, # UCT constant for balancing exploration and exploitation.
      config.max_simulations, # Number of MCTS simulations per move.
      evaluator_, # The JAX-based evaluator for policy and value network inference.
      solve=False, # AlphaZero does not solve the game tree in the traditional sense.
      dirichlet_noise=noise, # Apply noise to root policy in MCTS if not in evaluation.
      child_selection_fn=mcts.SearchNode.puct_value, # PUCT formula for child selection.
      verbose=False, # Keep MCTS bot non-verbose by default.
      dont_return_chance_node=True) # MCTS should not return chance nodes as root.


# _play_game function from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
# Adapted to use TrajectoryState and Trajectory already defined in this file.
def _play_game(logger, game_num: int, game: pyspiel.Game, bots: list, temperature: float, temperature_drop: int):
  """Play one game, return the trajectory."""
  trajectory = Trajectory() # Uses Trajectory class defined in this file
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState() # For reproducibility if seeded, otherwise system random
  if logger:
    logger.opt_print(f" Starting game {game_num} ".center(60, "-"))
    logger.opt_print(f"Initial state:\n{state}")

  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = random_state.choice(action_list, p=prob_list)
      # action_str = state.action_to_string(state.current_player(), action) # Not used locally
      # actions.append(action_str) # Actions list only for logging final summary
      state.apply_action(action)
    else:
      player = state.current_player()
      root = bots[player].mcts_search(state)
      policy = np.zeros(game.num_distinct_actions())
      for c in root.children:
        policy[c.action] = c.explore_count
      
      # Apply temperature
      if temperature == 0: # Avoid division by zero, choose greedily
          action = root.best_child().action
      else:
          policy = policy**(1 / temperature)
          policy /= policy.sum()
          if len(actions) >= temperature_drop: # After temperature_drop moves, pick best action
            action = root.best_child().action
          else:
            action = np.random.choice(len(policy), p=policy)

      # Store state, action, policy, value
      trajectory.states.append(
          TrajectoryState(state.observation_tensor(), # Uses TrajectoryState from this file
                          state.current_player(),
                          state.legal_actions_mask(), action, policy,
                          root.total_reward / root.explore_count if root.explore_count > 0 else 0)) # Value from MCTS search
      
      action_str_log = state.action_to_string(player, action)
      actions.append(action_str_log) # For final game log
      if logger:
        logger.opt_print(f"Player {player} sampled action: {action_str_log}")
      state.apply_action(action)
  
  if logger:
    logger.opt_print(f"Game finished. Next state:\n{state}")

  trajectory.returns = state.returns()
  if logger:
    logger.print(f"Game {game_num}: Returns: {' '.join(map(str, trajectory.returns))}; Actions: {' '.join(actions)}")
  return trajectory


@watcher
def actor(*, game: pyspiel.Game, config: ConfigJAX, logger, num: int, queue: spawn.Queue, prng_key: jax.random.PRNGKey):
  """An actor process that plays games and sends trajectories to the learner."""
  logger.print(f"Actor {num} started with PRNG key: {prng_key}")

  # Actor-specific PRNG key by folding in its number
  actor_internal_key = jax.random.fold_in(prng_key, num)
  model_init_key, actor_run_key = jax.random.split(actor_internal_key) # Split key for model init and other ops if needed

  logger.print(f"Actor {num}: Initializing model")
  # Initialize JAX model
  flax_model, variables = model_jax.init_flax_model_and_variables(model_init_key, config, game)
  
  logger.print(f"Actor {num}: Initializing AlphaZeroEvaluatorJAX")
  # Initialize evaluator
  az_evaluator = evaluator_jax.AlphaZeroEvaluatorJAX(game, flax_model, variables)

  bots = [
      _init_bot(config, game, az_evaluator, evaluation=False),
      _init_bot(config, game, az_evaluator, evaluation=False),
  ]

  # Checkpoint directory - ensure it's defined based on config.path
  # The learner creates this, actor just reads from it.
  ckpt_dir = os.path.join(config.path, "checkpoints_jax") if config.path else None
  if not config.path:
      logger.print(f"Warning: Actor {num} - config.path is not set. Checkpoint loading will be disabled.")


  def update_checkpoint_fn(current_variables, current_az_evaluator):
    """Attempts to load the latest checkpoint. Returns updated variables or current if no new checkpoint."""
    new_variables = current_variables
    loaded_new_checkpoint = False
    if not ckpt_dir: # If path is not set, cannot load checkpoints
        return new_variables, loaded_new_checkpoint

    # Path to the 'latest' checkpoint. Learner saves this.
    latest_checkpoint_path = os.path.join(ckpt_dir, "latest")

    try:
      # Check if a new checkpoint is available.
      # The `checkpoints.restore_checkpoint` can take a directory and find the latest,
      # or a direct file path. Using the "latest" symlink/file approach.
      # We need to know if the checkpoint on disk is newer than what we have.
      # For simplicity, we'll try to load 'latest' and if it's different, it's an update.
      # More robust would be to check step number if available.
      # `checkpoints.latest_checkpoint(ckpt_dir)` could also be used if we save with step prefixes.
      
      # The target for restoration should match what was saved by the learner.
      # Learner saves {'variables': variables, 'opt_state': opt_state}
      # Actor only needs 'variables'.
      target_to_restore = {'variables': current_variables} # Provide current_variables as a template

      # Attempt to restore. If `latest_checkpoint_path` doesn't exist, it will return None or raise error
      # depending on flax version and exact usage. Let's assume it returns None if not found.
      if os.path.exists(latest_checkpoint_path): # Only attempt if the 'latest' file/link exists
          logger.print(f"Actor {num}: Attempting to load checkpoint from {latest_checkpoint_path}")
          restored_state = checkpoints.restore_checkpoint(
              ckpt_dir=latest_checkpoint_path, # Pass the direct path to "latest"
              target=target_to_restore
          )

          if restored_state and restored_state['variables'] is not current_variables: # Check if something was actually restored and is different
              new_variables = restored_state['variables']
              current_az_evaluator.update_variables(new_variables) # Update evaluator with new variables
              logger.print(f"Actor {num}: Loaded new checkpoint. Inference cache info: {current_az_evaluator.cache_info()}")
              current_az_evaluator.clear_cache() # Clearing cache as model updated
              logger.print(f"Actor {num}: Cache cleared. New cache info: {current_az_evaluator.cache_info()}")
              loaded_new_checkpoint = True
          else:
              logger.opt_print(f"Actor {num}: No new checkpoint found or restore failed at {latest_checkpoint_path}. Keeping current model variables.")
      else:
          logger.opt_print(f"Actor {num}: 'latest' checkpoint file not found at {latest_checkpoint_path}. No update.")

    except Exception as e:
      logger.print(f"Actor {num}: Error loading checkpoint from {latest_checkpoint_path}: {e}")
      # Continue with existing variables
    return new_variables, loaded_new_checkpoint

  # Main actor loop
  for game_num in itertools.count():
    # Try to update model from checkpoint
    variables, _ = update_checkpoint_fn(variables, az_evaluator)
    
    # Play a game
    trajectory = _play_game(
        logger=logger,
        game_num=game_num,
        game=game,
        bots=bots,
        temperature=config.temperature,
        temperature_drop=config.temperature_drop)
    
    # Send trajectory to the learner
    try:
      queue.put(trajectory)
      logger.opt_print(f"Actor {num}: Sent trajectory {game_num} to learner. Queue size: {queue.qsize()}")
    except Exception as e: # Handle potential queue errors (e.g., if queue is full or closed)
      logger.print(f"Actor {num}: Error sending trajectory to queue: {e}")
      # Decide if to break or continue based on error. For now, continue.
      pass # Or break, or re-raise

    # Short delay to prevent actor from hogging CPU if queue is slow
    # time.sleep(0.001) # Optional: 1ms sleep


@watcher
def evaluator(*, game: pyspiel.Game, config: ConfigJAX, logger, num: int, queue: spawn.Queue, prng_key: jax.random.PRNGKey):
  """A process that plays the latest checkpoint vs standard MCTS."""
  logger.print(f"Evaluator {num} started with PRNG key: {prng_key}")

  evaluator_internal_key = jax.random.fold_in(prng_key, num)
  model_init_key, evaluator_run_key = jax.random.split(evaluator_internal_key)

  logger.print(f"Evaluator {num}: Initializing model")
  flax_model, variables = model_jax.init_flax_model_and_variables(model_init_key, config, game)

  logger.print(f"Evaluator {num}: Initializing AlphaZeroEvaluatorJAX")
  az_evaluator = evaluator_jax.AlphaZeroEvaluatorJAX(game, flax_model, variables)
  
  # The MCTS bot that uses the AZ model.
  az_bot = _init_bot(config, game, az_evaluator, evaluation=True) # True for evaluation mode

  # A standard MCTS bot with a random rollout evaluator to play against.
  # It's important that this opponent is reasonably strong but not overly slow.
  # The number of simulations for the opponent can be fixed or varied.
  # Original AlphaZero used `eval_levels` to vary opponent strength.
  random_rollout_evaluator = mcts.RandomRolloutEvaluator()
  
  # `eval_levels` from config determines different difficulties for the opponent.
  # An evaluator process might be pinned to one level or cycle through them.
  # For simplicity here, let's assume this evaluator instance handles one level, 
  # or cycles if `num` (evaluator index) is used to determine difficulty.
  # The original `evaluator` function varied difficulty based on `game_num % config.eval_levels`.
  # Let's keep that structure.

  results_buffer = Buffer(config.evaluation_window) # To store recent results for averaging

  ckpt_dir = os.path.join(config.path, "checkpoints_jax") if config.path else None
  if not config.path:
    logger.print(f"Warning: Evaluator {num} - config.path is not set. Checkpoint loading will be disabled.")

  current_loaded_step = -1 # Keep track of the loaded checkpoint step to avoid redundant loads of the same file

  def update_checkpoint_eval_fn(current_vars, current_az_eval):
    nonlocal current_loaded_step
    new_vars = current_vars
    if not ckpt_dir:
        return new_vars, False # False indicates no new checkpoint loaded

    latest_checkpoint_file = os.path.join(ckpt_dir, "latest")

    try:
        if os.path.exists(latest_checkpoint_file):
            # Check modification time or a step number embedded in the target to avoid reloading the same file.
            # Flax's `checkpoints.restore_checkpoint` might just load; we need to know *if* it loaded something new.
            # A simple way is to try to read a step number if the learner saves it inside the checkpoint target.
            # For now, we can check if the target file itself has changed or simply try to load.
            # The `checkpoints.restore_checkpoint` will return the restored target. We can compare.

            # A more robust way: if "latest" is a symlink, check `os.readlink` or `os.lstat().st_mtime`.
            # If "latest" is a file containing the path or step, parse it.
            # For simplicity with current Flax checkpointing (`save_checkpoint(..., step="latest", ...)`),
            # it creates a file literally named "latest-SUFFIX" or just "latest".
            # We can check its modification time.
            
            # Let's assume the learner's "latest" checkpoint has a retrievable step or timestamp.
            # Or, more simply, the `restore_checkpoint` itself can tell us if variables changed.
            # However, `checkpoints.restore_checkpoint` loads into a *copy* of the target by default.

            # To avoid reloading the exact same file if its content hasn't changed (e.g. by step number)
            # we could try to get the step from the checkpoint file name if `prefix` is used in save_checkpoint
            # e.g. by `checkpoints.latest_checkpoint(ckpt_dir, prefix="checkpoint_")`
            # The plan saves `latest` as a specific file. Let's try to load it and see if vars change.

            target_to_restore = {'variables': current_vars} 
            restored_state = checkpoints.restore_checkpoint(ckpt_dir=latest_checkpoint_file, target=target_to_restore)

            if restored_state and restored_state['variables'] is not current_vars:
                 # A robust check would be `jax.tree_util.tree_all(jax.tree_map(np.array_equal, new_variables, current_variables))`
                 # but simple object identity check after restore might be sufficient if restore creates new objects.
                 # For now, we assume if restore_checkpoint gives back a different variables dict, it's new.
                 # This needs to be tested with how Flax's restore_checkpoint behaves.
                 # A common pattern is that restore_checkpoint mutates the passed target or returns a new one.
                 # If it mutates, we need to compare before/after. If it returns new, check identity.
                 # The current `actor` assumed `restored_state['variables'] is not current_variables`
                 # Let's refine this: check if the *content* has changed, or rely on a step number.
                 # For now, let's assume the actor's check is okay for a basic version.

                new_vars = restored_state['variables']
                current_az_eval.update_variables(new_vars)
                logger.print(f"Evaluator {num}: Loaded new checkpoint from {latest_checkpoint_file}.")
                # current_az_eval.clear_cache() is called by update_variables
                return new_vars, True # True indicates new checkpoint loaded
            else:
                logger.opt_print(f"Evaluator {num}: No new checkpoint data found at {latest_checkpoint_file}.")
        else:
            logger.opt_print(f"Evaluator {num}: 'latest' checkpoint file not found at {latest_checkpoint_file}.")

    except Exception as e:
      logger.print(f"Evaluator {num}: Error loading checkpoint from {latest_checkpoint_file}: {e}")
    return new_vars, False

  # Main evaluator loop
  for game_num in itertools.count():
    # Update model from checkpoint
    variables, loaded_new = update_checkpoint_eval_fn(variables, az_evaluator)
    if loaded_new:
        # Reset results buffer if model changed, to evaluate the new model cleanly.
        results_buffer = Buffer(config.evaluation_window)
        logger.print(f"Evaluator {num}: Model updated, results buffer reset.")

    # Determine opponent strength for this game
    # This logic is from original AlphaZero: varies difficulty based on game number and eval_levels
    difficulty = (game_num // 2) % config.eval_levels if config.eval_levels > 0 else 0
    # max_simulations_opponent = int(config.max_simulations * (10**(difficulty / 2))) # Original scaling
    # For simplicity in this JAX version, let's assume a fixed number of opponent simulations, 
    # or that `_init_bot` for the opponent needs to be created fresh if max_simulations change.
    # Let's use a fixed opponent strength for now or make it part of config.
    # The plan was "MCTS+Solver", original uses MCTSBot with random evaluator.
    opponent_simulations = config.max_simulations # Default to same as AZ bot's base, or could be a new config field.
    # To vary opponent strength, one might re-initialize this bot or have a list of them.
    # For now, a single fixed-strength MCTS opponent.
    opponent_bot = mcts.MCTSBot(
        game,
        config.uct_c, # Use same UCT as AZ for opponent, or could be different
        opponent_simulations, # Number of simulations for the opponent
        random_rollout_evaluator, # Opponent uses random rollouts
        solve=False, # Original was solve=True for MCTS+Solver. Let's match that if possible.
                     # `solve=True` requires game to have a perfect solver, might be slow.
                     # Let's use solve=False for broader compatibility like original actor's _init_bot.
        verbose=False,
        dont_return_chance_node=True
    )

    # Alternate who plays first (AZ vs Opponent)
    az_player = game_num % 2
    current_bots = [az_bot, opponent_bot] if az_player == 0 else [opponent_bot, az_bot]
    
    logger.opt_print(f"Evaluator {num}, Game {game_num}: AZ player {az_player}, Opponent difficulty {difficulty}")

    trajectory = _play_game(
        logger=logger, 
        game_num=game_num, 
        game=game, 
        bots=current_bots, 
        temperature=1,        # For evaluation, use deterministic policy (temp=1, best_child in _play_game if temp_drop=0)
        temperature_drop=0    # No randomness in action selection after initial phase for evaluation
    )

    # Log result and send to learner (or a central stats collector)
    # The `queue` for evaluator is typically for it to send results *to* the learner/main process.
    result_for_az_player = trajectory.returns[az_player]
    results_buffer.append(result_for_az_player)
    
    avg_score = sum(results_buffer.data) / len(results_buffer.data) if results_buffer.data else 0

    logger.print(f"Evaluator {num}, Game {game_num}: Result for AZ player {az_player}: {result_for_az_player:.2f}. Avg score: {avg_score:.3f} over {len(results_buffer.data)} games.")

    try:
      # Queue item: (difficulty_level, score_for_az_player)
      queue.put((difficulty, result_for_az_player))
    except Exception as e:
      logger.print(f"Evaluator {num}: Error sending result to queue: {e}")
      # Decide if to break or continue.
      pass
    
    # Optional: small delay
    # time.sleep(config.evaluator_sleep_seconds if hasattr(config, 'evaluator_sleep_seconds') else 1)


def alpha_zero_jax(config: ConfigJAX):
  """Start all the worker processes for a full JAX AlphaZero setup."""
  game = pyspiel.load_game(config.game)
  # Update config with game-specific observation and output sizes
  # This was done in the original alpha_zero, good practice to ensure consistency.
  config = config._replace(
      observation_shape=list(game.observation_tensor_shape()), # Ensure it's a list for model_jax if it expects list
      output_size=game.num_distinct_actions()
  )

  logger_fn = functools.partial(file_logger.FileLogger, config.path, config.quiet)

  if not config.quiet:
    print("Starting JAX AlphaZero game", config.game)
    print(f"Game type: {game.get_type().short_name}")
    print(f"Game instance: {game}")
    print(f"Observation tensor shape: {game.observation_tensor_shape()}")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")

  if game.num_players() != 2:
    sys.exit("AlphaZero can only handle 2-player games.") # Python's sys module needed
  game_type = game.get_type()
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    raise ValueError("Game must have terminal rewards.")
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Game must have sequential turns.")
  if game_type.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    # While AlphaZero is often for perfect info, MCTS can run on imperfect.
    # However, the standard AZ formulation assumes perfect information.
    # Log a warning if not, as state/observation handling might be subtle.
    print("Warning: Game is not perfect information. AlphaZero typically assumes perfect information.")

  path = config.path
  if not path:
    # Create a default path if not specified, similar to original AlphaZero
    # Using datetime for unique directory names.
    import datetime # Ensure datetime is imported
    path = tempfile.mkdtemp(prefix=f"az-jax-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}-{config.game}")
    config = config._replace(path=path)
  
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  if not os.path.isdir(path):
    # This check should ideally not be an exit but raise an error if path creation failed.
    # However, matching original AlphaZero structure:
    sys.exit(f"{path} isn't a directory")
  
  if not config.quiet:
    print(f"Writing logs and checkpoints to: {path}")
    print(f"Model: {config.nn_model}, Width: {config.nn_width}, Depth: {config.nn_depth}")

  # Save the config to the directory for reproducibility
  try:
    with open(os.path.join(path, "config_jax.json"), "w") as fp:
      # Convert namedtuple to dict for JSON serialization
      # Need to handle any non-serializable fields if they exist (e.g. game object itself)
      # The original config only stored basic types.
      # Game object is not in ConfigJAX, so _asdict() should be fine.
      json.dump(config._asdict(), fp, indent=2, sort_keys=True)
      fp.write("\n")
  except Exception as e:
    print(f"Warning: Could not save JAX config to JSON: {e}")

  # Initialize master JAX PRNG key
  main_key = jax.random.PRNGKey(config.master_seed)
  
  # Split keys for learner, actors, and evaluators
  num_processes = 1 + config.actors + config.evaluators # 1 for learner
  process_keys = jax.random.split(main_key, num_processes)
  
  learner_key = process_keys[0]
  actor_keys_start_index = 1
  actor_keys_end_index = actor_keys_start_index + config.actors
  actor_keys = process_keys[actor_keys_start_index:actor_keys_end_index]
  evaluator_keys_start_index = actor_keys_end_index
  evaluator_keys = process_keys[evaluator_keys_start_index:]

  # Queues for communication
  # Actors send trajectories to the learner.
  # Evaluators send evaluation results to the learner.
  # Learner broadcasts checkpoint paths/commands to actors and evaluators.
  # This implies each actor/evaluator needs a queue to receive commands from the learner.
  # And the learner needs queues to receive data from actors/evaluators.

  # The original AlphaZero used a single queue per actor/evaluator for bi-directional comms
  # where items were polymorphic. This can be complex.
  # For JAX, the actor/evaluator checkpoint update logic polls the filesystem for "latest".
  # So the queues here are primarily for actor->learner (trajectories) and evaluator->learner (results).
  # The `broadcast_fn` will be used by the learner to signal actors/evaluators, perhaps by writing a file
  # or if we decide to use command queues later.
  # For now, let's assume the queues passed to actor/evaluator are for them to *send* data.

  actor_queues = [spawn.Queue() for _ in range(config.actors)]
  evaluator_queues = [spawn.Queue() for _ in range(config.evaluators)]

  # The learner needs access to all actor_queues to get trajectories
  # and all evaluator_queues to get evaluation results.
  # The `broadcast_fn` in original AlphaZero was for learner to send checkpoint paths.
  # Since our JAX actor/evaluator polls, `broadcast_fn` might not put on these queues.
  # Or it could signal via another mechanism if needed (e.g., a simple file flag).
  # Let's define a simple broadcast_fn placeholder for now that logs, as the polling mechanism is primary.
  
  # Learner will save checkpoints. Actor/Evaluator will load the "latest".
  # The `broadcast_fn` in the original TF version sent the *path* of the new checkpoint.
  # Our JAX actor/evaluator polls for a file named "latest".
  # So, the `broadcast_fn` for JAX might not need to send a path via queue.
  # It could just be a conceptual signal or not strictly necessary if polling is frequent enough.
  # However, if learner wants to signal an "exit" command, a command queue would be useful.
  # Let's make broadcast_fn a no-op for now as polling handles checkpoints, and exit is handled by try/finally.

  def broadcast_fn(message):
    # In TF AZ, this sent checkpoint paths or "exit" to actor/evaluator queues.
    # In JAX AZ with polling, this is less critical for checkpoint paths.
    # If used for "exit", would need command queues.
    # For now, let's make it a log, or it could write a special signal file.
    if not config.quiet:
      print(f"Learner broadcast: {message}")
    # If command queues were used:
    # for q in actor_command_queues + evaluator_command_queues: q.put(message)
    pass

  # Spawn actor processes
  actors = []
  for i in range(config.actors):
    actor_kwargs = {
        "game": game,
        "config": config,
        "num": i,
        "queue": actor_queues[i], # Queue for actor to send trajectories
        "prng_key": actor_keys[i]
    }
    actors.append(spawn.Process(target=actor, kwargs=actor_kwargs))
  
  # Spawn evaluator processes
  evaluators = []
  for i in range(config.evaluators):
    eval_kwargs = {
        "game": game,
        "config": config,
        "num": i,
        "queue": evaluator_queues[i], # Queue for evaluator to send results
        "prng_key": evaluator_keys[i]
    }
    evaluators.append(spawn.Process(target=evaluator, kwargs=eval_kwargs))

  # Start the learner. It will manage the main training loop.
  # The learner function needs access to actor_queues and evaluator_queues.
  try:
    learner(
        game=game,
        config=config,
        actor_queues=actor_queues, # Pass the list of queues actors send on
        evaluator_queues=evaluator_queues, # Pass the list of queues evaluators send on
        broadcast_fn=broadcast_fn, # For learner to signal (e.g. new ckpt, though polling is used)
        prng_key=learner_key
        # Logger is created inside @watcher for learner
    )
  except (KeyboardInterrupt, EOFError) as e:
    if not config.quiet:
      print(f"Caught {type(e).__name__}, stopping AlphaZero JAX.")
  finally:
    if not config.quiet:
      print("AlphaZero JAX stopping. Signaling actors and evaluators to exit.")
    
    # Signal actors and evaluators to exit if they were using a command queue.
    # Since they poll or run indefinitely until learner stops, explicit exit signal might be needed
    # if they are waiting on queues that learner no longer services.
    # Original AZ sent "" (empty string) as exit signal on the queues.
    # If our actor/evaluator loops `itertools.count()` and `queue.put()`, they might get stuck if learner exits.
    # `spawn.Process` objects should be joined.

    # If broadcast_fn was used to send "exit" to command queues:
    # broadcast_fn("exit") 

    # For processes that `put` on queues that learner reads:
    # If learner stops reading, `put` might block or error if queue is full.
    # It's important that actor/evaluator loops can terminate gracefully.
    # The `spawn.Process` might handle termination signals, or `join` might hang if process doesn't exit.
    
    # Ensure queues are emptied to allow processes to exit if they are blocked on put() to a full queue.
    # This is tricky. A better approach is for actor/evaluator to check a flag or have a timeout on queue puts.
    # For now, rely on spawn.Process termination. Original AZ did this join loop.
    
    # Join actors
    for proc in actors:
      # Original AlphaZero had a loop to empty queue before join.
      # This was for queues *to* the actor/evaluator if they were command queues.
      # Our actor_queues are for data *from* actors. Learner should have stopped reading.
      # If actor is blocked on queue.put(), this join might hang.
      # This part needs careful handling of process lifecycle.
      try:
          proc.join(timeout=JOIN_WAIT_DELAY * 10) # Added timeout
      except Exception as join_e:
          if not config.quiet:
              print(f"Error joining actor process: {join_e}")
    
    # Join evaluators
    for proc in evaluators:
      try:
          proc.join(timeout=JOIN_WAIT_DELAY * 10)
      except Exception as join_e:
          if not config.quiet:
              print(f"Error joining evaluator process: {join_e}")

    if not config.quiet:
      print("AlphaZero JAX run completed.")


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
# It should be: learner(*, game, config, actor_queues, evaluator_queues, broadcast_fn, prng_key)
# The logger for learner is created by its own @watcher decorator.

@watcher
def learner(*, game: pyspiel.Game, config: ConfigJAX, actor_queues: list, evaluator_queues: list, broadcast_fn, prng_key: jax.random.PRNGKey, logger=None):
  """A learner that consumes actor trajectories and evaluator results, and updates the model."""
  if logger:
    logger.also_to_stdout = True # For learner, log to stdout as well
    logger.print(f"Learner started with PRNG key: {prng_key}")

  print("JAX Learner started.")
  if logger:
    logger.print("JAX Learner started.")

  replay_buffer = Buffer(config.replay_buffer_size)
  # learn_rate is the number of states to collect before learning.
  if config.replay_buffer_reuse > 0:
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
  else: 
    learn_rate = config.replay_buffer_size 
  
  if learn_rate == 0 and config.replay_buffer_size > 0 : 
      learn_rate = config.replay_buffer_size 

  # Initialize JAX model and optimizer
  learner_key_for_init, init_key = jax.random.split(prng_key) 
  print(f"Initializing JAX model with key: {init_key}")
  if logger:
    logger.print(f"Initializing JAX model with key: {init_key}")
  
  flax_model, variables = model_jax.init_flax_model_and_variables(init_key, config, game)
  optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
  opt_state = optimizer.init(variables['params'])

  # Data logging and statistics
  data_log = None
  if config.path:
    data_log = data_logger.DataLoggerJsonLines(config.path, "learner_jax", True)
    if logger: logger.print(f"Learner data will be logged to {config.path}/learner_jax.jsonl")
  else:
    if logger: logger.print("Warning: config.path is not set. Learner data logging will be disabled.")

  # Statistics objects (mirroring TF AlphaZero where applicable)
  # Game specific stats
  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1) # Ensure game.max_game_length() is valid
  outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"]) # Assuming 2 players

  # Value prediction/accuracy stats (e.g., at start, mid, end of game)
  # Using a fixed stage_count, e.g., 3 for start, mid, end
  stage_count = 3 
  value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
  value_predictions = [stats.BasicStats() for _ in range(stage_count)]

  # Evaluation stats
  # Assuming config.eval_levels is defined and > 0 if evaluators are active
  eval_levels_count = config.eval_levels if hasattr(config, 'eval_levels') and config.eval_levels > 0 else 1
  evals = [Buffer(config.evaluation_window if hasattr(config, 'evaluation_window') else 100) for _ in range(eval_levels_count)]


  total_trajectories = 0

  # Setup checkpoint directory
  if config.path: # Ensure config.path is set
    ckpt_dir = os.path.join(config.path, "checkpoints_jax")
    os.makedirs(ckpt_dir, exist_ok=True)
    if logger:
        logger.print(f"JAX checkpoints will be saved in: {ckpt_dir}")
    else:
        print(f"JAX checkpoints will be saved in: {ckpt_dir}")
  else:
    ckpt_dir = None # No checkpointing if path is not provided
    if logger:
        logger.print("Warning: config.path is not set. JAX checkpointing will be disabled.")
    else:
        print("Warning: config.path is not set. JAX checkpointing will be disabled.")

  @jax.jit
  def train_step_fn(current_variables, current_opt_state, batch_observations, batch_legals_masks, batch_policy_targets, batch_value_targets):
      """Performs a single training step, JIT-compiled."""
      
      def loss_and_grad_inner_fn(params):
          """Computes loss and gradients for the model."""
          apply_vars = {'params': params}
          has_batch_stats = 'batch_stats' in current_variables
          
          if has_batch_stats:
              apply_vars['batch_stats'] = current_variables['batch_stats']
          
          # Forward pass
          # mutable=['batch_stats'] allows Flax to update batch norm statistics
          preds_and_state_or_preds = flax_model.apply(
              apply_vars,
              batch_observations,
              training=True,  # Important for layers like BatchNorm, Dropout
              mutable=['batch_stats'] if has_batch_stats else None
          )

          if has_batch_stats:
              (policy_logits, value_preds), updated_model_state = preds_and_state_or_preds
          else:
              (policy_logits, value_preds) = preds_and_state_or_preds
              updated_model_state = None

          # Policy loss
          # batch_policy_targets is a probability distribution from MCTS (pi).
          # batch_legals_masks indicates valid actions for each sample in the batch.
          # The MCTS policy target (pi) should already have zero probability for illegal actions.
          # Thus, optax.softmax_cross_entropy (which is sum_i pi_i * log(softmax(logit_i))) can be used directly.
          policy_loss = optax.softmax_cross_entropy(
              logits=policy_logits,
              labels=batch_policy_targets # Use the policy distribution directly
          )
          # The policy loss should only be computed for samples where there are legal moves.
          # However, if a state has no legal moves (e.g. terminal state erroneously included or a game error),
          # policy_target would be ill-defined. Assuming non-terminal states with valid policies.
          # Masking can also be done by ensuring logits for illegal actions are -inf before softmax,
          # or by ensuring target policy distribution is zero for illegal actions (which MCTS does).
          # The current crude masking was: policy_loss = jnp.mean(policy_loss * batch_legals_masks.any(axis=1))
          # A more standard approach is just to average the cross-entropy loss across the batch, 
          # relying on the target distribution being zero for illegal actions.
          policy_loss = jnp.mean(policy_loss) # Mean over the batch

          # Value loss
          # Predictions and targets are expected to be of shape [batch_size]
          value_loss = optax.squared_error(
              predictions=jnp.squeeze(value_preds, axis=-1),
              targets=jnp.squeeze(batch_value_targets, axis=-1)
          )
          value_loss = jnp.mean(value_loss) # Mean over the batch
          
          total_loss = policy_loss + value_loss
          
          # Return updated_model_state (for batch norm) and losses as auxiliary data
          return total_loss, (updated_model_state, policy_loss, value_loss)

      # Compute gradients and loss value
      (loss_val, (new_model_state, p_loss, v_loss)), grads = jax.value_and_grad(
          loss_and_grad_inner_fn, has_aux=True)(current_variables['params'])
      
      # Apply optimizer updates
      updates, new_opt_state = optimizer.apply_updates(grads, current_opt_state, current_variables['params'])
      new_params = optax.apply_updates(current_variables['params'], updates)
      
      # Update variables (parameters and potentially batch_stats)
      new_variables = current_variables.copy() # Start with a copy
      new_variables['params'] = new_params
      if new_model_state and 'batch_stats' in new_model_state: # new_model_state might be None
          new_variables['batch_stats'] = new_model_state['batch_stats']
            
      return new_variables, new_opt_state, loss_val, p_loss, v_loss

  def trajectory_generator():
    """Merge all the actor queues into a single generator."""
    # actor_queues is a list of queues passed to the learner function
    # that can raise spawn.Empty (needs spawn import or alternative).
    # For now, let's assume a simplified actor communication or placeholder.
    # TODO: Replace with actual spawn.Empty or relevant queue exception handling
    while True:
      found = 0
      for queue in actor_queues: # Use actor_queues instead of actors
        try:
          yield queue.get_nowait() 
        except Exception: # Generic exception, replace with specific queue empty (e.g. queue.Empty)
          pass
        else:
          found += 1
      if found == 0:
        time.sleep(0.01)  # 10ms

  def collect_trajectories():
    """Collects the trajectories from actors into the replay buffer and updates stats."""
    num_trajectories_collected = 0
    num_states_collected = 0
    for trajectory_data in trajectory_generator(): # trajectory_data is an instance of Trajectory class
      num_trajectories_collected += 1
      current_game_length = len(trajectory_data.states)
      num_states_collected += current_game_length

      # Update game statistics
      game_lengths.add(current_game_length)
      game_lengths_hist.add(current_game_length)

      p1_outcome = trajectory_data.returns[0]
      if p1_outcome > 0:
        outcomes.add(0)  # Player1 win
      elif p1_outcome < 0:
        outcomes.add(1)  # Player2 win
      else:
        outcomes.add(2)  # Draw

      # Update value accuracy and prediction stats
      if current_game_length > 0:
          for i in range(stage_count):
              # Ensure index is within bounds
              s_idx = (current_game_length - 1) * i // (stage_count - 1) if stage_count > 1 else 0
              s_idx = min(s_idx, current_game_length - 1) # Clamp to max index
              
              state_for_val_stat = trajectory_data.states[s_idx]
              # MCTS value is in state_for_val_stat.value
              # Actual outcome for the player whose turn it was at that state.
              current_player_at_stat = state_for_val_stat.current_player
              actual_outcome_for_player = trajectory_data.returns[current_player_at_stat]
              
              # Value is accurate if its sign matches the actual outcome's sign for that player
              mcts_value = state_for_val_stat.value
              is_accurate = (mcts_value >= 0) == (actual_outcome_for_player >= 0)
              value_accuracies[i].add(1 if is_accurate else 0)
              value_predictions[i].add(abs(mcts_value))

      replay_buffer.extend(
          model_jax.TrainInputJAX(
              observation=s.observation,
              legals_mask=s.legals_mask,
              policy_target=s.policy,
              value_target=np.array([p1_outcome if s.current_player == 0 else -p1_outcome], dtype=np.float32)
          ) for s in trajectory_data.states)

      if learn_rate > 0 and num_states_collected >= learn_rate:
        break
      elif learn_rate == 0 and num_trajectories_collected > 0: 
        break
        
    return num_trajectories_collected, num_states_collected
  
  last_time = time.time() - 60 
  # Variables to store losses from train_step for data_log
  current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')

  for step in itertools.count(1):
    # Reset per-step statistics
    game_lengths.reset()
    game_lengths_hist.reset()
    outcomes.reset()
    for i in range(stage_count):
        value_accuracies[i].reset()
        value_predictions[i].reset()
    
    current_time_for_collection_msg = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    collection_msg = f"[{current_time_for_collection_msg}] Learner step {step}, collecting trajectories..."
    if logger: logger.print(collection_msg)
    else: print(collection_msg)
        
    num_trajectories, num_states = collect_trajectories()
    total_trajectories += num_trajectories
    now = time.time()
    seconds_since_last_learn = now - last_time # Can be 0 if loop is very fast
    last_time = now
    
    # Avoid division by zero if seconds_since_last_learn is 0
    effective_seconds = seconds_since_last_learn if seconds_since_last_learn > 0 else 1e-6 
    effective_actors = config.actors if config.actors > 0 else 1

    log_message_timing = (
        f"Collected {num_states:5} states from {num_trajectories:3} games, "
        f"{num_states / effective_seconds:.1f} states/s. "
        f"{num_states / (effective_actors * effective_seconds):.1f} states/(s*actor), game_length: "
        f"{num_states / num_trajectories if num_trajectories > 0 else 0:.1f}"
    )
    log_message_buffer = f"Buffer size: {len(replay_buffer)}. Total states seen by buffer: {replay_buffer.total_seen}"

    step_log_msg_prefix = f"[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Step: {step}"
    if logger:
      logger.print(step_log_msg_prefix)
      logger.print(log_message_timing)
      logger.print(log_message_buffer)
    else:
      print(step_log_msg_prefix)
      print(log_message_timing)
      print(log_message_buffer)

    # Actual JAX training step
    save_path = None # Initialize save_path, will be updated if checkpoint is saved
    if len(replay_buffer) >= config.train_batch_size and config.train_batch_size > 0:
      batch_data = replay_buffer.sample(config.train_batch_size)
      
      # Ensure TrainInputJAX.stack method is available and used correctly.
      # If not, stack manually here. For now, assuming model_jax.TrainInputJAX.stack exists.
      # TODO: Verify or implement TrainInputJAX.stack if missing from model_jax.py
      try:
        stacked_input = model_jax.TrainInputJAX.stack(batch_data)
        batch_obs_jnp = jnp.array(stacked_input.observation, dtype=jnp.float32)
        batch_legals_jnp = jnp.array(stacked_input.legals_mask, dtype=jnp.bool_) # Or appropriate dtype for masks
        batch_policy_jnp = jnp.array(stacked_input.policy_target, dtype=jnp.float32)
        batch_value_jnp = jnp.array(stacked_input.value_target, dtype=jnp.float32)

        # The learner_key_for_init was used for model init.
        # For operations within train_step that might need a key (e.g., dropout if added later),
        # a new key should be split and passed if train_step_fn expects it.
        # Current train_step_fn doesn't explicitly take a key for dropout.
        
        variables, opt_state, total_loss_val, policy_loss_val, value_loss_val = train_step_fn(
            variables, opt_state, batch_obs_jnp, batch_legals_jnp, batch_policy_jnp, batch_value_jnp
        )
        current_total_loss, current_policy_loss, current_value_loss = total_loss_val, policy_loss_val, value_loss_val # Store for data_log
        
        loss_log_msg = f"Step: {step}, Total Loss: {current_total_loss:.4f}, Policy Loss: {current_policy_loss:.4f}, Value Loss: {current_value_loss:.4f}"
        if logger: logger.print(loss_log_msg)
        else: print(loss_log_msg)
        
        # JAX Checkpointing
        if ckpt_dir: # Only save if ckpt_dir is configured
            save_target = {'variables': variables, 'opt_state': opt_state}
            step_prefix = "checkpoint_"
            
            # Save step-specific checkpoint if frequency matches
            if config.checkpoint_freq > 0 and step % config.checkpoint_freq == 0:
                # keep defines how many step-checkpoints to keep. Let's keep the latest one.
                # The plan: keep=config.checkpoint_freq if config.checkpoint_freq > 0 else float('inf')
                # This is unusual for `keep`. `keep=1` means keep the latest for this prefix.
                # Let's use a small number, e.g., 3, for step checkpoints, or 1 if only one is desired.
                # For now, interpreting the plan as: if freq > 0, it implies we are saving these periodically,
                # and `keep` might refer to how many of these periodic saves to keep. config.checkpoint_freq as keep value seems odd.
                # Let's stick to a simpler keep=1 for step checkpoints.
                # MODIFIED according to TODO: Phase 8, Item 5
                keep_value = config.checkpoint_freq # Simplified from the TODO as we are inside 'if config.checkpoint_freq > 0'
                try:
                    checkpoints.save_checkpoint(
                        ckpt_dir=ckpt_dir, 
                        target=save_target, 
                        step=step, 
                        prefix=step_prefix, 
                        overwrite=True, # Overwrite if a checkpoint for this step already exists
                        keep=keep_value
                    )
                    if logger: logger.print(f"Saved step checkpoint: {step_prefix}{step} at {ckpt_dir} (kept {keep_value})")
                    else: print(f"Saved step checkpoint: {step_prefix}{step} at {ckpt_dir} (kept {keep_value})")
                except Exception as e:
                    err_msg = f"Error saving step checkpoint {step}: {e}"
                    if logger: logger.print(err_msg)
                    else: print(err_msg)

            # Always save/update the "latest" checkpoint
            try:
                # For the "latest" checkpoint, step is a string, prefix is empty to make the filename simply "latest"
                # (or whatever flax uses by default for a string step name). Plan: step="latest", prefix=""
                # `save_checkpoint` returns the path to the saved checkpoint file/directory.
                save_path = checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir, 
                    target=save_target, 
                    step="latest", # Using "latest" as the step name for this specific checkpoint
                    prefix="", # No prefix, so it will be named based on "latest"
                    overwrite=True, # Always overwrite the latest
                    keep=1 # Keep only this one "latest" checkpoint
                )
                if logger: logger.print(f"Saved latest checkpoint to: {save_path}")
                else: print(f"Saved latest checkpoint to: {save_path}")
            except Exception as e:
                err_msg = f"Error saving latest checkpoint: {e}"
                if logger: logger.print(err_msg)
                else: print(err_msg)
                save_path = None # Ensure save_path is None if saving failed
        else:
            # Checkpointing is disabled if ckpt_dir is None
            pass

      except AttributeError as e:
        # This might happen if TrainInputJAX.stack is not defined in model_jax.py
        error_msg = f"Error during training data preparation (possibly missing TrainInputJAX.stack): {e}"
        if logger: logger.print(error_msg)
        else: print(error_msg)
        current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan') # Reset on error


    else:
      # No training step taken (e.g. buffer not full enough)
      current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')
      if logger: logger.print(f"Step: {step}, Replay buffer not full enough for training. Size: {len(replay_buffer)}/{config.train_batch_size}")
      else: print(f"Step: {step}, Replay buffer not full enough for training. Size: {len(replay_buffer)}/{config.train_batch_size}")

    # Collect evaluation results
    for i, evac_queue in enumerate(evaluator_queues):
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
                if logger: logger.print(f"Error processing evaluator queue {i}: {e}")
                break # Avoid busy-looping on a consistently problematic queue

    # Log to data_logger
    if data_log:
        metrics_to_log = {
            "step": step,
            "total_states_seen_by_buffer": replay_buffer.total_seen,
            "replay_buffer_size": len(replay_buffer),
            "states_per_s": num_states / effective_seconds if effective_seconds > 0 else 0,
            "states_per_s_actor": num_states / (effective_actors * effective_seconds) if effective_seconds > 0 else 0,
            "total_trajectories": total_trajectories,
            "trajectories_per_s": num_trajectories / effective_seconds if effective_seconds > 0 else 0,
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

    if logger: logger.print("") # Add a newline for readability in FileLogger

    if config.max_steps > 0 and step >= config.max_steps:
      max_steps_msg = f"Max steps {config.max_steps} reached. Exiting learner."
      if logger: logger.print(max_steps_msg)
      break

    if save_path and broadcast_fn: 
        broadcast_msg = f"Broadcasting checkpoint: {save_path}"
        # Actual broadcast might involve sending 'variables' directly or a path from where actors can load.
        # For JAX, sending a path to a checkpoint saved by flax.training.checkpoints is typical.
        if logger: logger.print(broadcast_msg)
        else: print(broadcast_msg)
        broadcast_fn(save_path) 
  
  final_msg = f"[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] JAX Learner finished."
  if logger: logger.print(final_msg)
  else: print(final_msg)

# TODO: Implement actor, evaluator, and alpha_zero_jax main function (Phases 4 & 5)
# TODO: Implement train_step_fn (Phase 3, Step 2) and JAX checkpointing (Phase 3, Step 3) 

# ---- Start of AlphaZero JAX main execution example ----
FLAGS = flags.FLAGS

# Game and AlphaZero parameters
flags.DEFINE_string("game", "tic_tac_toe", "Name of the game.")
flags.DEFINE_string("path", None, "Path to save data.") # If None, a temp dir is used.
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.0001, "Weight decay.")
flags.DEFINE_integer("train_batch_size", 128, "Batch size for training.")
flags.DEFINE_integer("replay_buffer_size", 2**16, "Replay buffer size.") # 65536
flags.DEFINE_integer("replay_buffer_reuse", 1, "Replay buffer reuse.")
flags.DEFINE_integer("max_steps", 0, "Number of training steps. 0 for infinite.")
flags.DEFINE_integer("checkpoint_freq", 100, "Save a checkpoint every N steps.")
flags.DEFINE_integer("actors", 2, "Number of actors.")
flags.DEFINE_integer("evaluators", 1, "Number of evaluators.")
flags.DEFINE_integer("evaluation_window", 100, "How often to run evaluations.")
flags.DEFINE_integer("eval_levels", 7, "Number of levels for MCTS eval.") # Corresponds to Bot levels
flags.DEFINE_float("uct_c", 2, "UCT constant.")
flags.DEFINE_integer("max_simulations", 100, "Max MCTS simulations per move.")
flags.DEFINE_float("policy_alpha", 0.3, "Alpha for Dirichlet noise.") # Should match game's num_actions typically
flags.DEFINE_float("policy_epsilon", 0.25, "Epsilon for Dirichlet noise.")
flags.DEFINE_float("temperature", 1.0, "Initial temperature for policy sampling.")
flags.DEFINE_integer("temperature_drop", 10, "Drop temperature to 0 after this many moves.")
flags.DEFINE_string("nn_model", "mlp", "Neural network model type (mlp, resnet, resnet18, etc.).")
flags.DEFINE_integer("nn_width", 128, "Width of the neural network.")
flags.DEFINE_integer("nn_depth", 2, "Depth of the neural network (for MLP/Conv2D) or num_blocks for generic resnet config.")
flags.DEFINE_boolean("quiet", False, "Disable all logging.")
flags.DEFINE_integer("master_seed", 42, "Master RNG seed for JAX and other random operations.")

# ResNet specific config flags (used if nn_model is 'resnet')
flags.DEFINE_list("resnet_depth_config_list", None, "List of ints for resnet stage sizes, e.g., '2,2,2,2' for ResNet18 like structure. Used if nn_model is 'resnet'.")
flags.DEFINE_string("resnet_stem_name", "ResNetStem", "Name of the stem callable (e.g., ResNetStem, ResNetDStem). Used if nn_model is 'resnet'.")
flags.DEFINE_string("resnet_stem_kwargs_json", None, "JSON string for stem_kwargs, e.g., '{\"stem_width\": 32}'. Used if nn_model is 'resnet'.")
flags.DEFINE_string("resnet_block_name", "ResNetBlock", "Name of the block callable (e.g., ResNetBlock, ResNetBottleneckBlock). Used if nn_model is 'resnet'.")
flags.DEFINE_string("resnet_block_kwargs_json", None, "JSON string for block_kwargs, e.g., '{\"expansion\": 2}'. Used if nn_model is 'resnet'.")

def main(argv):
    del argv # Unused

    game_instance = pyspiel.load_game(FLAGS.game)
    observation_shape = game_instance.observation_tensor_shape()
    output_size = game_instance.num_distinct_actions()

    # Path creation
    data_path = FLAGS.path
    if data_path is None:
        data_path = tempfile.mkdtemp(prefix=f"az_jax_{FLAGS.game}_")
        print(f"No path specified, using temporary directory: {data_path}")
    
    # Initialize ResNet specific fields to None by default
    current_resnet_depth_config = None
    current_resnet_stem_callable_name = None
    current_resnet_stem_kwargs = None
    current_resnet_block_callable_name = None
    current_resnet_block_kwargs = None

    if FLAGS.nn_model == "resnet":
        print("Configuring a generic ResNet model based on FLAGS.")
        # Example of how these fields would be populated with Python data structures:
        # current_resnet_depth_config = [2, 2, 2, 2]  # e.g., for a ResNet18-like structure
        # current_resnet_stem_callable_name = "ResNetDStem"
        # current_resnet_stem_kwargs = {"stem_width": 32, "deep_stem": False} # Example
        # current_resnet_block_callable_name = "ResNetDBlock"
        # current_resnet_block_kwargs = {"projection_shortcut": True} # Example, specific to block type

        if FLAGS.resnet_depth_config_list:
            try:
                # The list flag might come in as strings, convert to int
                current_resnet_depth_config = [int(x) for x in FLAGS.resnet_depth_config_list]
            except ValueError as e:
                print(f"Error parsing resnet_depth_config_list: {e}. It should be a list of integers.")
                sys.exit(1)
        else:
            # Default to a simple structure if nn_model is 'resnet' but no depth_config is given.
            # Using nn_depth to create a simple config, e.g., [FLAGS.nn_depth] * 2 (two stages)
            # This is just an example; init_flax_model_and_variables will raise error if not provided correctly for 'resnet'
            print(f"Warning: nn_model='resnet' but no resnet_depth_config_list provided. model_jax will likely require it.")
            # Example: current_resnet_depth_config = [FLAGS.nn_depth, FLAGS.nn_depth]
        
        current_resnet_stem_callable_name = FLAGS.resnet_stem_name
        current_resnet_block_callable_name = FLAGS.resnet_block_name

        if FLAGS.resnet_stem_kwargs_json:
            try:
                current_resnet_stem_kwargs = json.loads(FLAGS.resnet_stem_kwargs_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing resnet_stem_kwargs_json: {e}")
                sys.exit(1)
        
        if FLAGS.resnet_block_kwargs_json:
            try:
                current_resnet_block_kwargs = json.loads(FLAGS.resnet_block_kwargs_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing resnet_block_kwargs_json: {e}")
                sys.exit(1)
        
        print(f"  Using resnet_depth_config: {current_resnet_depth_config}")
        print(f"  Using resnet_stem_callable_name: {current_resnet_stem_callable_name}")
        print(f"  Using resnet_stem_kwargs: {current_resnet_stem_kwargs}")
        print(f"  Using resnet_block_callable_name: {current_resnet_block_callable_name}")
        print(f"  Using resnet_block_kwargs: {current_resnet_block_kwargs}")

    elif FLAGS.nn_model in ["mlp", "resnet18", "resnet50", "resnest50fast"]: # etc.
        print(f"Using named model: {FLAGS.nn_model}. ResNet specific config fields will be None.")
        # For named models or MLP, these specific resnet config fields are typically None,
        # as init_flax_model_and_variables handles their structure internally.
        # They *could* be set here if one wanted to use the generic override mechanism
        # for a named model, but that's an advanced use case.
        current_resnet_depth_config = None
        current_resnet_stem_callable_name = None
        current_resnet_stem_kwargs = None
        current_resnet_block_callable_name = None
        current_resnet_block_kwargs = None
        # Example: If you wanted to override resnet18's block_kwargs:
        # if FLAGS.nn_model == "resnet18" and FLAGS.resnet_block_kwargs_json:
        #     current_resnet_block_kwargs = json.loads(FLAGS.resnet_block_kwargs_json)

    config = ConfigJAX(
        game=FLAGS.game,
        path=data_path,
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
        nn_depth=FLAGS.nn_depth, # Note: nn_depth serves different purposes for MLP vs generic ResNet
        observation_shape=observation_shape,
        output_size=output_size,
        quiet=FLAGS.quiet,
        master_seed=FLAGS.master_seed,
        # Pass the (potentially None) ResNet specific fields
        resnet_depth_config=current_resnet_depth_config,
        resnet_stem_callable_name=current_resnet_stem_callable_name,
        resnet_stem_kwargs=current_resnet_stem_kwargs,
        resnet_block_callable_name=current_resnet_block_callable_name,
        resnet_block_kwargs=current_resnet_block_kwargs
    )

    # Save the config to a JSON file in the path for reproducibility
    try:
        os.makedirs(config.path, exist_ok=True)
        config_path = os.path.join(config.path, "config_jax.json")
        # Convert namedtuple to dict for JSON serialization
        # For JAX keys or other non-serializable objects, handle them appropriately if they were in config
        config_dict = config._asdict()
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        print(f"Saved config to {config_path}")
    except Exception as e:
        print(f"Error saving config to JSON: {e}")

    # Call the main AlphaZero JAX function
    # This function is expected to be defined elsewhere in this file.
    alpha_zero_jax(config)

if __name__ == "__main__":
    # It's good practice to ensure JAX is using the desired platform early.
    # For example, to force CPU:
    # jax.config.update('jax_platform_name', 'cpu')
    app.run(main)

# ---- End of AlphaZero JAX main execution example ---- 