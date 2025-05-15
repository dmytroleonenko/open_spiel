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

from . import model_jax
from . import evaluator_jax # For AlphaZeroEvaluatorJAX
from open_spiel.python.utils import spawn, file_logger, data_logger, stats # Activated file_logger, spawn, data_logger, stats
from open_spiel.python.algorithms import mcts # Activated mcts

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001

# Constants for learner statistics
VALUE_ACC_HIST_BUCKETS = 20  # Number of buckets for value accuracy histograms
VALUE_PRED_HIST_BUCKETS = 20 # Number of buckets for value prediction histograms
EVALS_STAT_WINDOW = 100      # Window for evaluation statistics


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
    ])):
  """A config for the JAX AlphaZero model/experiment."""
  # To allow None defaults for Optional fields in namedtuple, provide them at instantiation.
  # Default values for new optional fields can be handled in the main script creating the ConfigJAX instance.
  pass

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

  def __len__(self):
    return len(self.states)

  def add(self, trajectory_state: TrajectoryState):
    self.states.append(trajectory_state)


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
    if len(self.data) > self.max_size:
        self.data = self.data[len(self.data) - self.max_size:]


  def sample(self, count):
    count = min(count, len(self.data))
    if count == 0:
        return []
    return random.sample(self.data, count)


# Watcher decorator from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
def watcher(fn):
  """Decorator to print exceptions and start/end logging for processes."""

  @functools.wraps(fn)
  def _watcher(*args, **kwargs):
    name = fn.__name__
    config = kwargs.get("config")
    num = kwargs.get("num", "process")

    if not config:
        raise ValueError(f"Watcher for '{name}': 'config' object not provided in kwargs. "
                         "A config object with a 'path' attribute is required for logging.")

    if not hasattr(config, 'path') or not config.path:
        raise ValueError(f"Watcher for '{name}': 'config.path' is not set or is empty. "
                         "A valid directory path is required for file logging.")

    _file_log_path_dir = config.path
    _log_name_prefix = f"{name}_{num}"
    _also_to_stdout = not config.quiet if hasattr(config, 'quiet') else True

    _logger = file_logger.FileLogger(path=_file_log_path_dir, name=_log_name_prefix, also_to_stdout=_also_to_stdout)

    try:
      _logger.print(f"{name} started")
      kwargs_to_pass = kwargs.copy()
      if fn.__name__ == 'learner':
        kwargs_to_pass.pop('queue', None)  # Remove queue if it exists for learner
      if 'logger' in fn.__code__.co_varnames:
        kwargs_to_pass['logger'] = _logger
      return fn(*args, **kwargs_to_pass)
    except Exception as e:
      _logger.print(f"Exception caught in {name}: {type(e).__name__}: {e}")
      _logger.print(traceback.format_exc())
      raise
    finally:
      _logger.print(f"{name} exiting")
      if _logger: # Check if logger was successfully created
        _logger.close() # Use the close() method of FileLogger

  return _watcher


# NEW CLASS DEFINITION STARTS HERE
class AlphaZeroBot(mcts.MCTSBot):
  """A MCTSBot for AlphaZero with added debug logging and assertions."""
  # player_id is declared by mcts.MCTSBot, pylint: disable=no-member

  def __init__(self,
               player_id: int, # This is the player_id for this bot instance
               game: pyspiel.Game, 
               evaluator: mcts.Evaluator,
               uct_c: float,
               max_simulations: int,
               # AlphaZero specific params that MCTSBot's dirichlet_noise tuple expects
               policy_alpha: float, 
               policy_epsilon: float,
               # MCTSBot's add_dirichlet_noise flag, determined by _play_game logic
               add_dirichlet_noise_for_bot: bool, # This boolean flag indicates if noise should be applied
               temperature: float, 
               temperature_drop: int,
               # Standard MCTSBot args
               solve: bool = True, 
               verbose: bool = False,
               child_selection_fn=mcts.SearchNode.uct_value,
               dont_return_chance_node: bool = False):
    
    # mcts.MCTSBot's dirichlet_noise parameter expects a tuple (epsilon, alpha) or None.
    # Construct this tuple based on add_dirichlet_noise_for_bot.
    dirichlet_noise_tuple = (policy_epsilon, policy_alpha) if add_dirichlet_noise_for_bot else None

    super().__init__(
        game=game,
        # player_id=player_id, # REMOVED: MCTSBot.__init__ does not take player_id
        evaluator=evaluator,
        uct_c=uct_c,
        max_simulations=max_simulations,
        solve=solve,
        verbose=verbose,
        child_selection_fn=child_selection_fn,
        dirichlet_noise=dirichlet_noise_tuple, # Pass the (epsilon, alpha) tuple or None
        dont_return_chance_node=dont_return_chance_node,
    )
    # Explicitly set player_id for this bot instance.
    # pyspiel.Bot (superclass of MCTSBot) has a player_id attribute.
    # It is typically set by the system managing the bot or can be set here.
    # For OpenSpiel, player_id is usually 0 or 1 for two-player games.
    # The MCTSBot might use this internally for some logic if available.
    self.player_id = player_id # ADDED: Set player_id attribute directly

    # Store AZ-specific params on the instance if they are needed by AlphaZeroBot's methods
    # (e.g. if it were to override _dirichlet_noise, though it doesn't currently)
    self.az_policy_alpha = policy_alpha 
    self.az_policy_epsilon = policy_epsilon
    self.az_temperature = temperature
    self.az_temperature_drop = temperature_drop
    
    # self.dirichlet_noise is an attribute set by the MCTSBot superclass constructor.
    # It will be None if no noise, or the (epsilon, alpha) tuple if noise is active.
    # The actual attribute in MCTSBot is _dirichlet_noise.
    is_noise_active_on_bot_instance = self._dirichlet_noise is not None

    # Debug logging (os is imported at the top of the file)
    print(f"[DEBUG AlphaZeroBot pid={os.getpid()}] init: "
          f"player_id={self.player_id}, "
          f"policy_alpha_arg={policy_alpha}, "
          f"policy_epsilon_arg={policy_epsilon}, "
          f"add_dirichlet_noise_for_bot_arg(bool)={add_dirichlet_noise_for_bot}, "
          f"self._dirichlet_noise (tuple on MCTSBot)={self._dirichlet_noise}, "
          f"self.az_policy_alpha (stored on AZBot)={self.az_policy_alpha}")

    # Assertion: If noise is active on the bot (meaning dirichlet_noise_tuple was not None),
    # then the alpha component of that noise (self.az_policy_alpha) must be positive.
    if is_noise_active_on_bot_instance and not (self.az_policy_alpha > 0):
        raise ValueError(
            f"AlphaZeroBot Consistency Check: If Dirichlet noise is active (inferred from self._dirichlet_noise being {self._dirichlet_noise}), "
            f"then az_policy_alpha (which is '{self.az_policy_alpha}') must be > 0. "
            f"This check is for player {self.player_id}.")

# NEW CLASS DEFINITION ENDS HERE


# _init_bot function from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
def _init_bot(config: ConfigJAX, game: pyspiel.Game, evaluator_: mcts.Evaluator, evaluation: bool, player_id_for_bot: int):
  """Initializes an AlphaZeroBot (JAX specific)."""
  
  # Determine if noise should be added for this specific bot instance.
  # - `evaluation` is True: No noise (deterministic play for evaluation).
  # - `evaluation` is False (e.g. actor self-play): Noise is active if policy_alpha and policy_epsilon are > 0.
  #   This matches the logic in the original _play_game for AlphaZeroBot instantiation.
  should_add_noise_flag = False
  if not evaluation: # Only consider noise if not in evaluation mode
    if config.policy_alpha > 0 and config.policy_epsilon > 0:
      should_add_noise_flag = True
      
  return AlphaZeroBot(
      player_id=player_id_for_bot,
      game=game,
      evaluator=evaluator_,
      uct_c=config.uct_c,
      max_simulations=config.max_simulations,
      policy_alpha=config.policy_alpha, 
      policy_epsilon=config.policy_epsilon,
      add_dirichlet_noise_for_bot=should_add_noise_flag, # Pass the boolean flag
      temperature=config.temperature, # Temperature for policy sampling (though MCTSBot doesn't use it directly for search)
      temperature_drop=config.temperature_drop, # (Same as above)
      solve=False, # AlphaZero does not solve in the traditional sense
      verbose=False,
      child_selection_fn=mcts.SearchNode.puct_value, # PUCT for AlphaZero
      dont_return_chance_node=True
  )


# _play_game function from open_spiel/python/algorithms/alpha_zero/alpha_zero.py
# Adapted to use TrajectoryState and Trajectory already defined in this file.
def _play_game(logger, game_num: int, game: pyspiel.Game, bots: list, temperature: float, temperature_drop: int, numpy_seed: int):
  """Play one game, return the trajectory."""
  trajectory = Trajectory() # Uses Trajectory class defined in this file
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState(numpy_seed) # Use the passed numpy_seed
  if logger:
    logger.opt_print(f" Starting game {game_num} (seed: {numpy_seed}) ".center(60, "-"))
    logger.opt_print(f"Initial state:\\n{state}")

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
            action = random_state.choice(len(policy), p=policy) # NEW: Use seeded random_state

      # Store state, action, policy, value
      current_trajectory_state = TrajectoryState(
          state.observation_tensor(), # Uses TrajectoryState from this file
          state.current_player(),
          state.legal_actions_mask(), action, policy,
          root.total_reward / root.explore_count if root.explore_count > 0 else 0) # Value from MCTS search
      trajectory.add(current_trajectory_state)
      
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
def actor(*, game: pyspiel.Game, config: ConfigJAX, logger, num: int,
          queue: spawn._ProcessQueue, prng_key: jax.random.PRNGKey):
  """An actor process that plays games and sends trajectories to the learner."""
  logger.print(f"Actor {num} started with PRNG key: {prng_key}")

  # Actor-specific PRNG key by folding in its number
  actor_internal_key = jax.random.fold_in(prng_key, num)
  model_init_key, actor_run_key = jax.random.split(actor_internal_key)

  logger.print(f"Actor {num}: Initializing model (about to call init_flax_model_and_variables)")
  try:
    flax_model, variables = model_jax.init_flax_model_and_variables(model_init_key, config, game)
    logger.print(f"Actor {num}: Model initialized successfully. Model: {flax_model}")
  except Exception as e:
    logger.print(f"Actor {num}: Model initialization FAILED: {e}")
    import traceback as tb; logger.print(tb.format_exc())
    raise
  
  logger.print(f"Actor {num}: Initializing AlphaZeroEvaluatorJAX")
  # Initialize evaluator
  az_evaluator = evaluator_jax.AlphaZeroEvaluatorJAX(game, flax_model, variables, config.evaluator_cache_size)

  # Create a bot for each player in the game.
  # These bots will be used by _play_game.
  bots = []
  for i in range(game.num_players()): # Corrected loop range
      bots.append(_init_bot(config, game, az_evaluator, evaluation=False, player_id_for_bot=i))

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
      # Attempt to load the "latest" checkpoint. If it differs from the current, it's an update.
      # Learner saves {'variables': variables, 'opt_state': opt_state}; actor only needs 'variables'.
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
    
    # Derive a seed for this specific game from the actor's run key
    # Fold in game_num to ensure each game gets a unique PRNG sequence if actor is restarted/reused.
    game_specific_rng_key = jax.random.fold_in(actor_run_key, game_num)
    current_numpy_seed = jax.random.randint(game_specific_rng_key, shape=(), minval=0, maxval=jnp.iinfo(jnp.int32).max).item()

    # Play a game
    full_trajectory = _play_game(
        logger=logger,
        game_num=game_num,
        game=game,
        bots=bots,
        temperature=config.temperature,
        temperature_drop=config.temperature_drop,
        numpy_seed=current_numpy_seed
    )
    
    # For each player, create a player-specific trajectory and send it.
    list_of_returns_from_game = full_trajectory.returns
    if list_of_returns_from_game is None:
        logger.print(f"Actor {num}, Game {game_num}: full_trajectory.returns is None. Skipping sending.")
    elif not isinstance(list_of_returns_from_game, list) or len(list_of_returns_from_game) != game.num_players():
        logger.print(f"Actor {num}, Game {game_num}: full_trajectory.returns is not a list of correct length. Got: {list_of_returns_from_game}. Skipping.")
    else:
        for p_id in range(game.num_players()):
            player_specific_trajectory = Trajectory()
            player_specific_trajectory.states = full_trajectory.states
            player_specific_trajectory.returns = list_of_returns_from_game[p_id]

            try:
                queue.put(player_specific_trajectory)
                logger.opt_print(f"Actor {num}: Sent trajectory for player {p_id} from game {game_num} to learner. Return: {player_specific_trajectory.returns}")
            except Exception as e:
                logger.print(f"Actor {num}: Error sending player-specific trajectory (p_id {p_id}, game {game_num}) to queue: {e}")
                pass

    # Short delay to prevent actor from hogging CPU if queue is slow
    time.sleep(0.001) # Small sleep to avoid busy-waiting


@watcher
def evaluator(*, game: pyspiel.Game, config: ConfigJAX, logger, num: int,
              queue: spawn._ProcessQueue, prng_key: jax.random.PRNGKey):
  """A process that plays the latest checkpoint vs standard MCTS."""
  logger.print(f"Evaluator {num} started with PRNG key: {prng_key}")

  evaluator_internal_key = jax.random.fold_in(prng_key, num)
  model_init_key, evaluator_run_key = jax.random.split(evaluator_internal_key)

  logger.print(f"Evaluator {num}: Initializing model (about to call init_flax_model_and_variables)")
  try:
    flax_model, variables = model_jax.init_flax_model_and_variables(model_init_key, config, game)
    logger.print(f"Evaluator {num}: Model initialized successfully. Model: {flax_model}")
  except Exception as e:
    logger.print(f"Evaluator {num}: Model initialization FAILED: {e}")
    import traceback as tb; logger.print(tb.format_exc())
    raise

  logger.print(f"Evaluator {num}: Initializing AlphaZeroEvaluatorJAX")
  az_evaluator = evaluator_jax.AlphaZeroEvaluatorJAX(game, flax_model, variables, config.evaluator_cache_size)
  
  # The MCTS bot that uses the AZ model. Initialize for player 0, for example.
  # The actual player assignment happens in the game loop via current_bots.
  az_bot = _init_bot(config, game, az_evaluator, evaluation=True, player_id_for_bot=0)

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
            # Attempt to restore the "latest" checkpoint. A change in the restored variables' content
            # (or identity, depending on flax.checkpoints behavior) indicates a new checkpoint.
            # Robust checking might involve comparing variable contents or step numbers if available.
            # For now, rely on the restored object differing if an update occurred.
            target_to_restore = {'variables': current_vars} 
            restored_state = checkpoints.restore_checkpoint(ckpt_dir=latest_checkpoint_file, target=target_to_restore)

            if restored_state and restored_state['variables'] is not current_vars:
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

    # Determine opponent strength (difficulty from original AlphaZero, fixed for now).
    difficulty = (game_num // 2) % config.eval_levels if config.eval_levels > 0 else 0
    opponent_simulations = config.max_simulations 
    opponent_bot = mcts.MCTSBot(
        game,
        config.uct_c, # Use same UCT as AZ for opponent, or could be different
        opponent_simulations, # Number of simulations for the opponent
        random_rollout_evaluator, # Opponent uses random rollouts
        solve=False, # Original was solve=True for MCTS+Solver. Let's match that if possible.
        verbose=False,
        dont_return_chance_node=True
    )

    # Alternate who plays first (AZ vs Opponent)
    az_player = game_num % 2
    current_bots = [az_bot, opponent_bot] if az_player == 0 else [opponent_bot, az_bot]
    
    logger.opt_print(f"Evaluator {num}, Game {game_num}: AZ player {az_player}, Opponent difficulty {difficulty}")

    # Generate a numpy_seed for this game, as in actor
    game_specific_rng_key = jax.random.fold_in(evaluator_run_key, game_num)
    numpy_seed = jax.random.randint(game_specific_rng_key, shape=(), minval=0, maxval=jnp.iinfo(jnp.int32).max).item()

    trajectory = _play_game(
        logger=logger, 
        game_num=game_num, 
        game=game, 
        bots=current_bots, 
        temperature=1,        # For evaluation, use deterministic policy (temp=1, best_child in _play_game if temp_drop=0)
        temperature_drop=0,   # No randomness in action selection after initial phase for evaluation
        numpy_seed=numpy_seed
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


def broadcast_fn(message):
    # This is a placeholder for broadcasting messages to actors/evaluators if needed.
    # In the current implementation, it just prints/logs the message.
    # If a logger is needed, use a global or pass as argument (not required for pickling).
    print(f"Broadcasting message (placeholder): {message}")
    # If you want to use a logger, you can set a global logger variable here.
    # Or, you can make this a no-op if not needed.
    pass

def alpha_zero_jax(config: ConfigJAX):
    """Main entry point for JAX AlphaZero."""
    import pyspiel  # Ensure pyspiel is imported in this scope
    main_key = jax.random.PRNGKey(config.master_seed)
    random.seed(config.master_seed)
    np.random.seed(config.master_seed)
    os.makedirs(config.path, exist_ok=True)
    main_log_directory = config.path 
    main_log_name = "main_alpha_zero_jax" 
    main_process_logger = file_logger.FileLogger(main_log_directory, main_log_name, not config.quiet)
    actual_log_file_path = os.path.join(main_log_directory, f'log-{main_log_name}.txt')
    if not config.quiet:
        print(f"Main process logging to: {actual_log_file_path}")
    process_keys = jax.random.split(main_key, 1 + config.actors + config.evaluators)
    learner_key = process_keys[0]
    actor_keys = process_keys[1:1+config.actors]
    evaluator_keys = process_keys[1+config.actors:]
    game = pyspiel.load_game(config.game)
    processes = []
    actor_process_queues = []
    evaluator_process_queues = []
    main_process_logger.print(f"Starting {config.actors} actors...")
    for i in range(config.actors):
        actor_kwargs = {
            "game": game,
            "config": config,
            "num": i,
            "prng_key": actor_keys[i]
        }
        p = spawn.Process(target=actor, kwargs=actor_kwargs)
        processes.append(p)
        actor_process_queues.append(p.queue)
    main_process_logger.print(f"Starting {config.evaluators} evaluators...")
    for i in range(config.evaluators):
        eval_kwargs = {
            "game": game,
            "config": config,
            "num": i,
            "prng_key": evaluator_keys[i]
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
        "broadcast_fn": broadcast_fn  # Pass the top-level function
    }
    main_process_logger.print("Starting Learner...")
    p_learner = spawn.Process(target=learner, kwargs=learner_kwargs)
    processes.append(p_learner)
    try:
        learner(
            game=game,
            config=config,
            actor_queues=actor_process_queues, 
            evaluator_queues=evaluator_process_queues, 
            broadcast_fn=broadcast_fn, 
            prng_key=learner_key
        )
    except (KeyboardInterrupt, EOFError) as e: 
        main_process_logger.print(f"Caught {type(e).__name__}, stopping AlphaZero JAX.")
    finally:
        main_process_logger.print("AlphaZero JAX stopping. Signaling actors and evaluators to exit.")
        for proc in processes:
            try:
                proc.join(timeout=JOIN_WAIT_DELAY * 10) 
            except Exception as join_e: 
                main_process_logger.print(f"Error joining process: {join_e}")
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
# It should be: learner(*, game, config, actor_queues, evaluator_queues, broadcast_fn, prng_key)
# The logger for learner is created by its own @watcher decorator.

@watcher
def learner(*, game: pyspiel.Game, config: ConfigJAX, logger,
            actor_queues: list[spawn._ProcessQueue], 
            evaluator_queues: list[spawn._ProcessQueue], 
            broadcast_fn, # Added broadcast_fn here
            prng_key: jax.random.PRNGKey):
  """A learner that consumes actor trajectories and evaluator results, and updates the model."""
  # The logger variable is now reliably passed by the @watcher decorator.
  if logger: # Check if logger is provided (it should be by watcher)
    logger.print(f"JAX Learner started with PRNG key: {prng_key}")
    logger.print(f"Learner using game: {game}, config: {config}") # Log basic info
  else: # Fallback if watcher didn't provide logger (should not happen)
    print(f"JAX Learner started (no logger) with PRNG key: {prng_key}")


  # Initialize model and optimizer
  learner_key_for_init, prng_key = jax.random.split(prng_key) # Use prng_key for subsequent ops
  
  # Ensure game object is loaded if only game name was passed in config,
  # or use the passed game object.
  # The 'game' parameter to learner should be a loaded pyspiel.Game object.
  # if isinstance(game, str): # This check might be needed if only game name is passed
  #   loaded_game = pyspiel.load_game(game)
  # else:
  #   loaded_game = game

  flax_model, variables = model_jax.init_flax_model_and_variables(
      learner_key_for_init, config, game # Use the game object
  )
  optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
  opt_state = optimizer.init(variables['params'])

  replay_buffer = Buffer(config.replay_buffer_size)
  
  # Restore from checkpoint if available
  ckpt_dir = os.path.join(config.path, "checkpoints_jax") if config.path else None
  if ckpt_dir:
    os.makedirs(ckpt_dir, exist_ok=True)
    # Try to restore the latest checkpoint
    # Target for restore should match what's saved (variables and opt_state)
    restore_target = {'variables': variables, 'opt_state': opt_state}
    latest_checkpoint_path = os.path.join(ckpt_dir, "latest") # Default name by save_checkpoint(..., step="latest", prefix="")
    
    # Use checkpoints.restore_checkpoint correctly. It expects the ckpt_dir and optionally a specific step.
    # If restoring "latest", the path should be to the directory containing the "latest" file/symlink.
    try:
      restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=restore_target, step="latest", prefix="") 
      if restored_state:
        variables = restored_state['variables']
        opt_state = restored_state['opt_state']
        if logger: logger.print(f"Learner restored checkpoint from {latest_checkpoint_path}")
      else:
        if logger: logger.print(f"No 'latest' checkpoint found at {ckpt_dir} to restore. Starting fresh.")
    except FileNotFoundError: # Specific exception if the checkpoint dir or file doesn't exist
        if logger: logger.print(f"Checkpoint file/directory for 'latest' not found at {ckpt_dir}. Starting fresh.")
    except Exception as e: # Catch other potential errors during restore
        if logger: logger.print(f"Error restoring 'latest' checkpoint from {ckpt_dir}: {e}. Starting fresh.")

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
      # Ensure apply_vars includes 'params' and potentially 'batch_stats'
      apply_vars = {'params': params}
      if 'batch_stats' in current_variables: # Check if model uses batch_stats
        apply_vars['batch_stats'] = current_variables['batch_stats']
      
      # Determine if model needs mutable state for batch_stats
      mutable_list = ['batch_stats'] if 'batch_stats' in apply_vars else None

      # Model application
      # The model's __call__ should accept legals_mask and apply it internally to logits.
      preds_and_state = flax_model.apply(
          apply_vars, 
          batch_observations, 
          batch_legals_masks, # Pass legals_mask to model
          training=True, 
          mutable=mutable_list
      )
      
      if mutable_list:
        (policy_logits, value_preds), updated_model_state = preds_and_state
      else:
        (policy_logits, value_preds) = preds_and_state
        updated_model_state = None

      # Policy loss: softmax cross-entropy. Assumes policy_logits are raw logits.
      # Assumes batch_policy_targets are probability distributions.
      # The model should have handled masking illegal actions by setting their logits to -inf.
      policy_loss = optax.softmax_cross_entropy(logits=policy_logits, labels=batch_policy_targets)
      
      # Masking for samples with no legal actions (e.g. terminal states mistakenly in batch)
      # This outer mask zero_outs loss for samples where no action was possible at all.
      # batch_legals_masks.any(axis=1) is True if there's at least one legal action for that sample.
      # This is an additional safeguard. The primary masking of illegal actions should happen in the model.
      policy_loss = policy_loss * batch_legals_masks.any(axis=1)
      policy_loss = jnp.mean(policy_loss)

      # Value loss: squared error.
      # Ensure shapes are compatible for squared_error. value_preds might be (N, 1), targets (N,).
      value_loss = optax.squared_error(
          predictions=jnp.squeeze(value_preds, axis=-1),
          targets=jnp.squeeze(batch_value_targets, axis=-1)
      )
      value_loss = jnp.mean(value_loss)
      
      total_loss = policy_loss + value_loss
      return total_loss, (updated_model_state, policy_loss, value_loss)

    (loss_val, (new_model_state, p_loss, v_loss)), grads = jax.value_and_grad(
        loss_and_grad_inner_fn, has_aux=True)(current_variables['params'])
    
    updates, new_opt_state = optimizer.apply_updates(grads, current_opt_state, current_variables['params'])
    new_params = optax.apply_updates(current_variables['params'], updates)
    
    new_variables = current_variables.copy()
    new_variables['params'] = new_params
    if new_model_state and 'batch_stats' in new_model_state: # Check if batch_stats were updated
        new_variables['batch_stats'] = new_model_state['batch_stats']
          
    return new_variables, new_opt_state, loss_val, p_loss, v_loss

  # ---- Main Learner Loop ----
  last_time = time.time()
  total_trajectories = 0
  
  # Store current losses for logging, initialize to NaN
  current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')


  # This generator yields trajectories from actor_queues
  def trajectory_generator():
    while True:
      found = 0
      for queue_idx, queue in enumerate(actor_queues): # Use actor_queues
        try:
          yield queue.get_nowait() 
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
      
  for step in itertools.count(1): # Start step from 1 for 1-based indexing if preferred for logging
    if config.max_steps > 0 and step > config.max_steps: # Check before starting step
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
        trajectories_to_process = collect_trajectories(num_to_collect=1) # Process one trajectory for stats per step for now

        for traj in trajectories_to_process:
            total_trajectories += 1
            num_trajectories += 1
            game_lengths.add(len(traj))
            game_lengths_hist.add(len(traj))
            num_states += len(traj)
            # outcomes.add(traj.returns) # Old incorrect way

            # Map scalar return to outcome string for HistogramNamed
            current_player_return = traj.returns # This is now a scalar
            outcome_str = None # Initialize to None
            if current_player_return is not None: # Check if return value is not None
                if current_player_return > 0: # Win for this player
                    outcome_str = "win"
                elif current_player_return < 0: # Loss for this player
                    outcome_str = "loss"
                elif current_player_return == 0: # Draw
                    outcome_str = "draw"
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
                        logger.print(f"Learner: Received trajectory with unexpected (but non-None) return value: {current_player_return}. Not mapping to known outcome string.")
            else: # current_player_return is None
                 if logger:
                    logger.print(f"Learner: Received trajectory with None return value. Not adding to outcomes histogram.")


            if outcome_str:
                try:
                    bucket_id = outcome_names_for_histogram.index(outcome_str)
                    outcomes.add(bucket_id)
                except ValueError:
                    # This should not happen if outcome_str is one of the defined names.
                    if logger:
                        logger.print(f"Learner: Outcome string '{outcome_str}' is not in outcome_names_for_histogram. This is unexpected. Not adding to histogram.")
            
            # Add states to replay buffer
            # Each element in traj.states is a TrajectoryState
            for transition in traj.states: # CORRECTED: Iterate over traj.states
                # Create TrainInputJAX from TrajectoryState
                train_input = model_jax.TrainInputJAX(
                    observation=transition.observation,
                    legals_mask=transition.legals_mask,
                    policy_target=transition.policy, # Placeholder: MCTS policy from that state
                    value_target=transition.value    # Placeholder: MCTS value or game outcome
                )
                replay_buffer.append(train_input)

    except spawn.Empty: # Should be handled by trajectory_generator now
        if logger: logger.opt_print("Learner: All actor queues empty.") # opt_print for less frequent messages
        # Continue to next part of the loop (e.g. try training if buffer is full)
    except Exception as e:
        if logger: logger.print(f"Learner: Error processing actor queue: {e}")
        # Potentially skip this learner step or handle error more gracefully
    
    now = time.time()
    seconds = now - last_time
    last_time = now
    
    # Calculate effective actors contributing to this step's data
    # This is a bit heuristic; if actors are much faster than learner, effective_actors might be high.
    # If only one trajectory was processed, effective_actors for this stat could be 1.
    effective_actors = config.actors # Assume all actors are contributing over time.

    # Log stats even if no training step is taken, to monitor data flow
    log_message_timing = (
        f"Step: {step}, Game Speed: {num_trajectories / seconds:.1f} games/s, "
        f"{num_states / seconds:.1f} states/s. "
        f"{num_states / (effective_actors * seconds):.1f} states/(s*actor), game_length: "
        f"{num_states / num_trajectories if num_trajectories > 0 else 0:.1f}"
    )
    log_message_buffer = f"Buffer size: {len(replay_buffer)}. Total states seen by buffer: {replay_buffer.total_seen}"

    step_log_msg_prefix = f"[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Step: {step}"
    if logger:
      logger.print(step_log_msg_prefix)
      logger.print(log_message_timing)
      logger.print(log_message_buffer)

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
        if logger: 
          logger.print(loss_log_msg)
        
        # JAX Checkpointing
        if ckpt_dir: # Only save if ckpt_dir is configured
            save_target = {'variables': variables, 'opt_state': opt_state}
            step_prefix = "checkpoint_"
            
            # Save step-specific checkpoint if frequency matches
            if config.checkpoint_freq > 0 and step % config.checkpoint_freq == 0:
                # `keep=config.checkpoint_freq` aims to keep a rolling window of periodic checkpoints.
                keep_value = config.checkpoint_freq 
                try:
                    checkpoints.save_checkpoint(
                        ckpt_dir=ckpt_dir, 
                        target=save_target, 
                        step=step, 
                        prefix=step_prefix, 
                        overwrite=True, 
                        keep=keep_value
                    )
                    if logger: 
                      logger.opt_print(f"Saved step checkpoint: {step_prefix}{step} at {ckpt_dir} (kept {keep_value})")
                except Exception as e:
                    err_msg = f"Error saving step checkpoint {step}: {e}"
                    if logger: 
                      logger.print(err_msg)

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
                if logger: 
                  logger.opt_print(f"Saved latest checkpoint to: {save_path}")
            except Exception as e:
                err_msg = f"Error saving latest checkpoint: {e}"
                if logger: 
                  logger.print(err_msg)
                save_path = None # Ensure save_path is None if saving failed
        else:
            # Checkpointing is disabled if ckpt_dir is None
            pass

      except AttributeError as e:
        # This might happen if TrainInputJAX.stack is not defined in model_jax.py
        error_msg = f"Error during training data preparation (possibly missing TrainInputJAX.stack): {e}"
        if logger: 
          logger.print(error_msg)
        current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan') # Reset on error


    else:
      # No training step taken (e.g. buffer not full enough)
      current_total_loss, current_policy_loss, current_value_loss = float('nan'), float('nan'), float('nan')
      if logger: 
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

    if logger: logger.print("") # Add a newline for readability in FileLogger

    if config.max_steps > 0 and step >= config.max_steps:
      max_steps_msg = f"Max steps {config.max_steps} reached. Exiting learner."
      if logger: 
        logger.print(max_steps_msg)
      break

    if save_path and broadcast_fn: 
        broadcast_msg = f"Broadcasting checkpoint: {save_path}"
        if logger: 
          logger.opt_print(broadcast_msg) # Changed from logger.print for periodic status
        broadcast_fn(save_path) 
  
  final_msg = f"[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] JAX Learner finished."
  if logger: 
    logger.print(final_msg)

# ---- Start of AlphaZero JAX main execution example ----
# FLAGS = flags.FLAGS

# # Game and AlphaZero parameters
# flags.DEFINE_string("game", "tic_tac_toe", "Name of the game.")
# flags.DEFINE_string("path", None, "Path to save data.") # If None, a temp dir is used.
# flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
# flags.DEFINE_float("weight_decay", 0.0001, "Weight decay.")
# flags.DEFINE_integer("train_batch_size", 128, "Batch size for training.")
# flags.DEFINE_integer("replay_buffer_size", 2**16, "Replay buffer size.") # 65536
# flags.DEFINE_integer("replay_buffer_reuse", 1, "Replay buffer reuse.")
# flags.DEFINE_integer("max_steps", 0, "Number of training steps. 0 for infinite.")
# flags.DEFINE_integer("checkpoint_freq", 100, "Save a checkpoint every N steps.")
# flags.DEFINE_integer("actors", 2, "Number of actors.")
# flags.DEFINE_integer("evaluators", 1, "Number of evaluators.")
# flags.DEFINE_integer("evaluation_window", 100, "How often to run evaluations.")
# flags.DEFINE_integer("eval_levels", 7, "Number of levels for MCTS eval.") # Corresponds to Bot levels
# flags.DEFINE_float("uct_c", 2, "UCT constant.")
# flags.DEFINE_integer("max_simulations", 100, "Max MCTS simulations per move.")
# flags.DEFINE_float("policy_alpha", 0.3, "Alpha for Dirichlet noise.") # Should match game's num_actions typically
# flags.DEFINE_float("policy_epsilon", 0.25, "Epsilon for Dirichlet noise.")
# flags.DEFINE_float("temperature", 1.0, "Initial temperature for policy sampling.")
# flags.DEFINE_integer("temperature_drop", 10, "Drop temperature to 0 after this many moves.")
# flags.DEFINE_string("nn_model", "mlp", "Neural network model type (mlp, resnet, resnet18, etc.).")
# flags.DEFINE_integer("nn_width", 256, "Width of the neural network.")
# flags.DEFINE_integer("nn_depth", 20, "Depth of the neural network (e.g., number of hidden layers in MLP/Conv2D). For a generic 'resnet' model, this often corresponds to the number of residual blocks (e.g., 20 for an AlphaGo Zero-like model).")
# flags.DEFINE_boolean("quiet", False, "Disable all logging.")
# flags.DEFINE_integer("master_seed", 42, "Master RNG seed for JAX and other random operations.")
# flags.DEFINE_integer("evaluator_cache_size", 2**16, "Size of the LRU cache for the evaluator.")

# # ResNet specific config flags (used if nn_model is 'resnet')
# flags.DEFINE_list("resnet_depth_config_list", None, "List of ints for resnet stage sizes, e.g., '2,2,2,2' for ResNet18 like structure. Used if nn_model is 'resnet'.")
# flags.DEFINE_string("resnet_stem_name", "ResNetStem", "Name of the stem callable (e.g., ResNetStem, ResNetDStem). Used if nn_model is 'resnet'.")
# flags.DEFINE_string("resnet_stem_kwargs_json", None, "JSON string for stem_kwargs, e.g., '{\"stem_width\": 32}'. Used if nn_model is 'resnet'.")
# flags.DEFINE_string("resnet_block_name", "ResNetBlock", "Name of the block callable (e.g., ResNetBlock, ResNetBottleneckBlock). Used if nn_model is 'resnet'.")
# flags.DEFINE_string("resnet_block_kwargs_json", None, "JSON string for block_kwargs, e.g., '{\"expansion\": 2}'. Used if nn_model is 'resnet'.")

# def main(argv):
#     del argv # Unused

#     game_instance = pyspiel.load_game(FLAGS.game)
#     observation_shape = game_instance.observation_tensor_shape()
#     output_size = game_instance.num_distinct_actions()

#     # Path creation
#     data_path = FLAGS.path
#     if data_path is None:
#         data_path = tempfile.mkdtemp(prefix=f"az_jax_{FLAGS.game}_")
#         print(f"No path specified, using temporary directory: {data_path}")
#     
#     # Initialize ResNet specific fields to None by default
#     current_resnet_depth_config = None
#     current_resnet_stem_callable_name = None
#     current_resnet_stem_kwargs = None
#     current_resnet_block_callable_name = None
#     current_resnet_block_kwargs = None

#     if FLAGS.nn_model == "resnet":
#         print("Configuring a generic ResNet model based on FLAGS.")
#         # Example of how these fields would be populated with Python data structures:
#         # current_resnet_depth_config = [2, 2, 2, 2]  # e.g., for a ResNet18-like structure
#         # current_resnet_stem_callable_name = "ResNetDStem"
#         # current_resnet_stem_kwargs = {"stem_width": 32, "deep_stem": False} # Example
#         # current_resnet_block_callable_name = "ResNetDBlock"
#         # current_resnet_block_kwargs = {"projection_shortcut": True} # Example, specific to block type

#         if FLAGS.resnet_depth_config_list:
#             try:
#                 # The list flag might come in as strings, convert to int
#                 current_resnet_depth_config = [int(x) for x in FLAGS.resnet_depth_config_list]
#             except ValueError as e:
#                 print(f"Error parsing resnet_depth_config_list: {e}. It should be a list of integers.")
#                 sys.exit(1)
#         else:
#             # Default to a simple structure if nn_model is 'resnet' but no depth_config is given.
#             # Using nn_depth to create a simple config, e.g., [FLAGS.nn_depth] * 2 (two stages)
#             # This is just an example; init_flax_model_and_variables will raise error if not provided correctly for 'resnet'
#             print(f"Warning: nn_model='resnet' but no resnet_depth_config_list provided. model_jax will likely require it.")
#             # Example: current_resnet_depth_config = [FLAGS.nn_depth, FLAGS.nn_depth]
#         
#         current_resnet_stem_callable_name = FLAGS.resnet_stem_name
#         current_resnet_block_callable_name = FLAGS.resnet_block_name

#         if FLAGS.resnet_stem_kwargs_json:
#             try:
#                 current_resnet_stem_kwargs = json.loads(FLAGS.resnet_stem_kwargs_json)
#             except json.JSONDecodeError as e:
#                 print(f"Error parsing resnet_stem_kwargs_json: {e}")
#                 sys.exit(1)
#         
#         if FLAGS.resnet_block_kwargs_json:
#             try:
#                 current_resnet_block_kwargs = json.loads(FLAGS.resnet_block_kwargs_json)
#             except json.JSONDecodeError as e:
#                 print(f"Error parsing resnet_block_kwargs_json: {e}")
#                 sys.exit(1)
#         
#         print(f"  Using resnet_depth_config: {current_resnet_depth_config}")
#         print(f"  Using resnet_stem_callable_name: {current_resnet_stem_callable_name}")
#         print(f"  Using resnet_stem_kwargs: {current_resnet_stem_kwargs}")
#         print(f"  Using resnet_block_callable_name: {current_resnet_block_callable_name}")
#         print(f"  Using resnet_block_kwargs: {current_resnet_block_kwargs}")

#     elif FLAGS.nn_model in ["mlp", "resnet18", "resnet50", "resnest50fast"]: # etc.
#         print(f"Using named model: {FLAGS.nn_model}. ResNet specific config fields will be None.")
#         # For named models or MLP, these specific resnet config fields are typically None,
#         # as init_flax_model_and_variables handles their structure internally.
#         # They *could* be set here if one wanted to use the generic override mechanism
#         # for a named model, but that's an advanced use case.
#         current_resnet_depth_config = None
#         current_resnet_stem_callable_name = None
#         current_resnet_stem_kwargs = None
#         current_resnet_block_callable_name = None
#         current_resnet_block_kwargs = None
#         # Example: If you wanted to override resnet18's block_kwargs:
#         # if FLAGS.nn_model == "resnet18" and FLAGS.resnet_block_kwargs_json:
#         #     current_resnet_block_kwargs = json.loads(FLAGS.resnet_block_kwargs_json)

#     config = ConfigJAX(
#         game=FLAGS.game,
#         path=data_path,
#         learning_rate=FLAGS.learning_rate,
#         weight_decay=FLAGS.weight_decay,
#         train_batch_size=FLAGS.train_batch_size,
#         replay_buffer_size=FLAGS.replay_buffer_size,
#         replay_buffer_reuse=FLAGS.replay_buffer_reuse,
#         max_steps=FLAGS.max_steps,
#         checkpoint_freq=FLAGS.checkpoint_freq,
#         actors=FLAGS.actors,
#         evaluators=FLAGS.evaluators,
#         evaluation_window=FLAGS.evaluation_window,
#         eval_levels=FLAGS.eval_levels,
#         uct_c=FLAGS.uct_c,
#         max_simulations=FLAGS.max_simulations,
#         policy_alpha=FLAGS.policy_alpha,
#         policy_epsilon=FLAGS.policy_epsilon,
#         temperature=FLAGS.temperature,
#         temperature_drop=FLAGS.temperature_drop,
#         nn_model=FLAGS.nn_model,
#         nn_width=FLAGS.nn_width,
#         nn_depth=FLAGS.nn_depth, # Note: nn_depth serves different purposes for MLP vs generic ResNet
#         observation_shape=observation_shape,
#         output_size=output_size,
#         quiet=FLAGS.quiet,
#         master_seed=FLAGS.master_seed,
#         # Pass the (potentially None) ResNet specific fields
#         resnet_depth_config=current_resnet_depth_config,
#         resnet_stem_callable_name=current_resnet_stem_callable_name,
#         resnet_stem_kwargs=current_resnet_stem_kwargs,
#         resnet_block_callable_name=current_resnet_block_callable_name,
#         resnet_block_kwargs=current_resnet_block_kwargs,
#         evaluator_cache_size=FLAGS.evaluator_cache_size
#     )

#     # Save the config to a JSON file in the path for reproducibility
#     try:
#         os.makedirs(config.path, exist_ok=True)
#         config_path = os.path.join(config.path, "config_jax.json")
#         # Convert namedtuple to dict for JSON serialization
#         # For JAX keys or other non-serializable objects, handle them appropriately if they were in config
#         config_dict = config._asdict()
#         with open(config_path, "w") as f:
#             json.dump(config_dict, f, indent=2, sort_keys=True)
#         print(f"Saved config to {config_path}")
#     except Exception as e:
#         print(f"Error saving config to JSON: {e}")

#     # Call the main AlphaZero JAX function
#     # This function is expected to be defined elsewhere in this file.
#     alpha_zero_jax(config)

# if __name__ == "__main__":
#     # It's good practice to ensure JAX is using the desired platform early.
#     # For example, to force CPU:
#     # jax.config.update('jax_platform_name', 'cpu')
#     app.run(main)

# ---- End of AlphaZero JAX main execution example ---- 