# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure Python logic for AlphaZero JAX actors and evaluators."""

import collections
import functools
import itertools
import random
import time
import traceback
import os
import sys

import numpy as np
import pyspiel
import queue as std_queue # For specific exception types like Full/Empty

from open_spiel.python.algorithms import mcts
from open_spiel.python.utils import file_logger, spawn # spawn for ProcessQueue type hint
from .remote_inference import RemoteEvaluator, SHUTDOWN_SENTINEL, ShutdownException

# Define debug level constants used by actor and evaluator
# _ACTOR_DEBUG_LEVEL = 3 # No longer needed for RemoteEvaluator's debug_mode
# _EVALUATOR_DEBUG_LEVEL = 3 # No longer needed for RemoteEvaluator's debug_mode

# It's assumed that ConfigJAX is passed as an argument and actor/evaluator
# will access fields like config.uct_c, config.path, config.quiet, etc.
# Log level constants (ERROR, WARN, INFO, DEBUG, TRACE) are defined in
# the main alpha_zero_jax.py and config.log_level is passed.


# Watcher decorator (moved from alpha_zero_jax.py)
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
      # Ensure logger is passed if the wrapped function expects it and it's not already there.
      # (This check might be more complex depending on how logger is injected elsewhere)
      if 'logger' not in kwargs_to_pass and 'logger' in fn.__code__.co_varnames:
        kwargs_to_pass['logger'] = _logger
      
      # Remove queue if it exists for specific functions if not used by the core logic here
      # For example, if actor/evaluator get it but only pass to RemoteEvaluator
      # This specific pop for 'queue' was in the original learner watcher, adapt if necessary
      # if fn.__name__ == 'learner': # Learner is not in this file
      #   kwargs_to_pass.pop('queue', None)
      
      return fn(*args, **kwargs_to_pass)
    except Exception as e:
      _logger.print(f"Exception caught in {name}: {type(e).__name__}: {e}")
      _logger.print(traceback.format_exc())
      # To ensure the error propagates and the process exits correctly if fatal
      raise
    finally:
      _logger.print(f"{name} exiting")
      if _logger: # Check if logger was successfully created
        _logger.close()

  return _watcher


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


class AlphaZeroBot(mcts.MCTSBot):
  """A MCTSBot for AlphaZero with player_id handling for OpenSpiel."""
  # player_id is declared by pyspiel.Bot, pylint: disable=no-member

  def __init__(self,
               player_id: int,
               game: pyspiel.Game,
               evaluator: mcts.Evaluator,
               uct_c: float,
               max_simulations: int,
               policy_alpha: float,
               policy_epsilon: float,
               add_dirichlet_noise_for_bot: bool,
               # These are not directly used by MCTSBot but might be needed by _play_game logic
               # temperature: float, # pylint: disable=unused-argument
               # temperature_drop: int, # pylint: disable=unused-argument
               solve: bool = True,
               verbose: bool = False,
               child_selection_fn=mcts.SearchNode.uct_value,
               dont_return_chance_node: bool = False):

    dirichlet_noise_tuple = (policy_epsilon,
                             policy_alpha) if add_dirichlet_noise_for_bot else None

    super().__init__(
        game=game, # Pass game to MCTSBot constructor
        evaluator=evaluator,
        uct_c=uct_c,
        max_simulations=max_simulations,
        solve=solve,
        verbose=verbose,
        child_selection_fn=child_selection_fn,
        dirichlet_noise=dirichlet_noise_tuple,
        dont_return_chance_node=True,
    )
    # Explicitly set player_id for this bot instance.
    # pyspiel.Bot (superclass of MCTSBot) has a player_id attribute.
    self.player_id = player_id


def _init_bot(config, game: pyspiel.Game, evaluator_: mcts.Evaluator,
              evaluation: bool, player_id_for_bot: int):
  """Initializes a bot for playing or evaluation."""
  return AlphaZeroBot(
      player_id=player_id_for_bot, # Pass player_id here
      game=game, # Pass game here
      evaluator=evaluator_,
      uct_c=config.uct_c,
      max_simulations=config.max_simulations,
      policy_alpha=config.policy_alpha,
      policy_epsilon=config.policy_epsilon,
      # For evaluation, we don't want noise. For self-play, we do.
      add_dirichlet_noise_for_bot=not evaluation,
      # temperature, temperature_drop are handled by _play_game's action selection.
      # MCTSBot itself doesn't use them directly for its tree search.
      solve=False, # Typically solve=False for AlphaZero training
      verbose=config.log_level >= 4 # TRACE level for bot verbosity
      )


def _play_game(logger, game_num: int, game: pyspiel.Game, bots: list,
               temperature: float, temperature_drop: int,
               numpy_seed: int, log_level: int = 0):
  """Plays one game, returning the trajectory."""
  if logger and log_level >= 3: # DEBUG
    logger.print(f"Game {game_num}: Entered _play_game. numpy_seed: {numpy_seed}")

  np.random.seed(numpy_seed)
  trajectory = Trajectory()
  
  if logger and log_level >= 3: # DEBUG
    logger.print(f"Game {game_num}: About to call game.new_initial_state() for game '{game.get_type().short_name}'")
  state = game.new_initial_state()
  if logger and log_level >= 3: # DEBUG
    logger.print(f"Game {game_num}: game.new_initial_state() returned. State: {state}")

  random_state = np.random.RandomState(numpy_seed)

  if log_level >= 3: # DEBUG
    logger.print(f"Starting game {game_num} with numpy_seed: {numpy_seed}, "
                 f"initial temperature: {temperature}, temp_drop: {temperature_drop}")

  turn_number = 0
  while True: # Changed from `while not state.is_terminal():` to allow logging before the check
    turn_number += 1
    if logger and log_level >= 3: # DEBUG
        logger.print(f"Game {game_num} [PG_DEBUG]: Turn {turn_number} BEGIN. Current player: {state.current_player() if not state.is_terminal() else 'N/A (terminal)'}")

    if logger and log_level >= 4: # TRACE
        logger.print(f"Game {game_num} [PG_DEBUG]: Turn {turn_number} About to check state.is_terminal(). Current state: {state.history_str() if hasattr(state, 'history_str') else str(state)}")
    
    if state.is_terminal():
        if logger and log_level >= 3: # DEBUG
            logger.print(f"Game {game_num} [PG_DEBUG]: Turn {turn_number} State IS terminal. Exiting play loop.")
        break # Exit the while True loop

    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes, probs = zip(*state.chance_outcomes())
      action = random_state.choice(outcomes, p=probs)
      state.apply_action(action)
      if log_level >= 4: # TRACE
        logger.print(f"Game {game_num} Chance node action: {action}")
      continue

    current_player = state.current_player()
    bot = bots[current_player]

    # Bot play
    # The AlphaZeroBot (MCTSBot) will use its evaluator (RemoteEvaluator) here
    if logger and log_level >= 3: # DEBUG, changed from TRACE to ensure it shows up with TRACE level
        logger.print(f"Game {game_num} Player {current_player}: Calling bot.step_with_policy(state)")
    action_and_policy_or_error = bot.step_with_policy(state)
    
    # --- [PG_DEBUG] Log raw output from bot.step_with_policy ---
    if logger and log_level >= 3: # DEBUG
        logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: bot.step_with_policy returned: {action_and_policy_or_error}")

    # Check if the return is as expected (a tuple/list of two elements)
    if not (isinstance(action_and_policy_or_error, (tuple, list)) and len(action_and_policy_or_error) == 2):
        if logger and log_level >= 0: # ERROR
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG] bot.step_with_policy returned unexpected value: {action_and_policy_or_error}. Expected (action, policy_dict). Aborting game for this actor.")
        return trajectory # Return current trajectory (might be empty or partial)
        
    try:
        # --- [PG_DEBUG] Pre-unpacking policy and action ---
        if logger and log_level >= 3: # DEBUG
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: Pre-unpacking action_and_policy_or_error.")
        policy_dict, action = action_and_policy_or_error
        # --- [PG_DEBUG] Post-unpacking policy and action ---
        if logger and log_level >= 3: # DEBUG
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: Unpacked. Action: {action}, Policy_dict type: {type(policy_dict)}, Policy_dict items (first 5): {list(policy_dict)[:5] if isinstance(policy_dict, list) else str(policy_dict)[:100]}")
    except Exception as e_unpack:
        if logger and log_level >= 0: # ERROR
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG] Error unpacking (policy_dict, action): {e_unpack}. Value was: {action_and_policy_or_error}. Traceback: {traceback.format_exc()}")
        return trajectory # Return current (possibly empty or partial) trajectory

    # Convert policy from dict to a dense array based on legal actions
    # This policy is what MCTS search, after noise and temperature, recommends.
    policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    try:
        # --- [PG_DEBUG] Pre-processing policy_dict into dense policy array ---
        if logger and log_level >= 3: # DEBUG
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: Pre-processing policy_dict into dense array. Num distinct actions: {game.num_distinct_actions()}")
        for act, prob in policy_dict:
            policy[act] = prob
        # --- [PG_DEBUG] Post-processing policy_dict ---
        if logger and log_level >= 3: # DEBUG
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: Dense policy array created. Sum: {np.sum(policy):.3f}")
    except Exception as e_policy_proc:
        if logger and log_level >= 0: # ERROR
            logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG] Error processing policy_dict: {e_policy_proc}. Policy_dict was: {policy_dict}. Traceback: {traceback.format_exc()}")
        return trajectory # Return current trajectory

    # Store the state, action, policy
    # Value will be filled in later by the learner after the game is done.
    # --- [PG_DEBUG] Pre-trajectory.add ---
    if logger and log_level >= 3: # DEBUG
        logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: Adding to trajectory. Action: {action}")
    trajectory.add(
        TrajectoryState(state.observation_tensor(), current_player,
                        state.legal_actions_mask(), action, policy,
                        value=0.0)) # Placeholder value

    # --- [PG_DEBUG] Pre-apply_action ---
    if logger and log_level >= 3: # DEBUG
        logger.print(f"Game {game_num} Player {current_player} [PG_DEBUG]: Applying action {action} to state.")
    state.apply_action(action)
    if log_level >= 4: # TRACE
        logger.print(f"Game {game_num} Player {current_player} action: {action}")


    # Temperature drop
    if game_num < temperature_drop:
      # This logic was for adjusting MCTS temperature, but AlphaZeroBot uses
      # dirichlet noise. The temperature here is for action selection policy
      # which is implicitly handled by MCTSBot.step_with_policy if it implements
      # temperature-based sampling from the policy.
      # OpenSpiel's MCTSBot by default samples proportionally to visit counts.
      # For AlphaZero, the temperature is usually applied to the policy *before*
      # action selection during self-play. The MCTS search itself might use temperature
      # for its PUCT formula or visit count exponent.
      # The passed `temperature` is for this action selection sampling.
      # If `bot.step_with_policy` doesn't use this temperature, this logic is unused.
      # Let's assume for now MCTSBot implicitly handles it or we might need to adjust
      # how policy is derived or sampled if explicit temperature is needed here.
      # For now, this temperature parameter is not directly used to modify bot's behavior here.
      pass

  returns = np.array(state.returns(), dtype=np.float32)
  trajectory.returns = returns

  if log_level >= 3: # DEBUG
    logger.print(f"Game {game_num} finished. Returns: {returns}")
  
  # --- [PG_DEBUG] Pre-queue.put(trajectory) ---
  if logger and log_level >= 3: # DEBUG
    logger.print(f"Game {game_num} [PG_DEBUG]: About to put trajectory on queue. Length: {len(trajectory)}")
  return trajectory


@watcher
def actor(*, game: pyspiel.Game, config, logger, num: int, # config is ConfigJAX
          queue: spawn._ProcessQueue, initial_seed: int,
          inference_request_queue, # mp.Queue
          inference_response_queue # mp.Queue
          ):
  """An actor process that plays games and sends trajectories to the learner."""
  # Determine if debug_mode should be enabled for RemoteEvaluator
  # Based on config.log_level (DEBUG=3, TRACE=4)
  # remote_evaluator_debug_mode = config.log_level >= 3 # DEBUG or TRACE # No longer needed

  # Determine log_path for RemoteEvaluator
  # The watcher for 'actor' creates logs in config.path/actor_NUM/
  # For consistency, RemoteEvaluator could log there too, or directly in config.path
  # Current RemoteEvaluator default is CWD. Let's make it explicit.
  evaluator_log_path = os.path.join(config.path, f"actor_{num}_remote_eval")
  # os.makedirs(evaluator_log_path, exist_ok=True) # Ensure dir exists

  # Seed Python's random and NumPy for this actor process
  random.seed(initial_seed)
  np_seed = random.randint(0, 2**31 - 1) # Generate a derived seed for NumPy
  np.random.seed(np_seed)
  if logger and config.log_level >= 3: # DEBUG
    logger.print(f"Actor {num} started with initial_seed: {initial_seed}, numpy_seed: {np_seed}")


  evaluator_ = RemoteEvaluator(
      game=game,
      actor_id=num,
      inference_request_queue=inference_request_queue,
      inference_response_queue=inference_response_queue,
      max_cache_size=config.evaluator_cache_size,
      numeric_log_level=config.log_level, # CHANGED: Pass numeric_log_level from config
      # debug_mode=remote_evaluator_debug_mode, # OLD: Pass debug_mode
      log_path=evaluator_log_path # Pass the constructed log_path
  )

  # Use player_id 0 for the bot in self-play, as it's from player 0's perspective.
  # The actual current_player is handled by the game state.
  bot = _init_bot(config, game, evaluator_, evaluation=False, player_id_for_bot=0)

  for game_num in itertools.count(1): # Start game numbers from 1
    # Determine temperature for this game
    # Original logic: temperature an MCTSBot parameter.
    # Here, temperature is used for action selection *after* MCTS policy.
    # It's not a direct parameter of MCTSBot search itself in this setup.
    current_temperature = (
        config.temperature if config.temperature_drop == 0 or game_num < config.temperature_drop else 0.)
    
    if logger and config.log_level >= 4: # TRACE
        logger.print(f"Actor {num} Game {game_num}: Starting game with temperature {current_temperature}")

    # Generate a unique seed for this game play based on the initial actor seed
    # This ensures that if an actor restarts, it doesn't replay the exact same games
    # if initial_seed was the same.
    game_specific_numpy_seed = random.randint(0, 2**31 - 1)

    try:
      trajectory = _play_game(
          logger=logger, # Pass the actor's logger
          game_num=game_num,
          game=game,
          bots=[bot, bot],  # Both sides are played by the same bot logic
          temperature=current_temperature,
          temperature_drop=config.temperature_drop, # Though not directly used by _play_game's temp logic
          numpy_seed=game_specific_numpy_seed, # Pass the game-specific seed
          log_level=config.log_level # Pass the main log_level
          )
      if trajectory:
        # logger.print(f"Actor {num} Game {game_num} completed. Trajectory length: {len(trajectory.states)}")
        queue.put(trajectory)
      else: # Should not happen if _play_game returns a trajectory or raises
        if logger and config.log_level >= 1: # WARN
            logger.print(f"Actor {num} Game {game_num}: _play_game returned None or empty trajectory. Skipping.")
    except ShutdownException:
      if logger and config.log_level >= 2: # INFO
        logger.print(f"Actor {num} received ShutdownException. Exiting play game loop.")
      # Ensure the evaluator's resources are cleaned up if possible,
      # though RemoteEvaluator itself doesn't have explicit close().
      # The sentinel on its queue should handle its exit if it's blocked.
      break # Exit the game playing loop
    except Exception as e: # Catch other exceptions during game play
      error_message = f"Actor {num} Game {game_num}: Exception during _play_game: {type(e).__name__} - {e}. Traceback: {traceback.format_exc()}"
      if logger and config.log_level >= 0: # ERROR
        logger.print(error_message)
        if hasattr(logger, 'flush'): # Attempt to flush the logger
            logger.flush()
      print(error_message, file=sys.stderr) # Also print to stderr for immediate visibility
      sys.stderr.flush() # Ensure stderr is flushed
      break

  if logger and config.log_level >= 2: # INFO
    logger.print(f"Actor {num} finished {game_num -1} games.")
  # Signal to learner that this actor is done (e.g. by closing queue or sending sentinel)
  # The current setup relies on process join in the main script.
  # If queue needs explicit close or sentinel, add here.


@watcher
def evaluator(*, game: pyspiel.Game, config, logger, num: int, # config is ConfigJAX
                queue: spawn._ProcessQueue, initial_seed: int,
                inference_request_queue, # mp.Queue
                inference_response_queue # mp.Queue
                ):
  """An evaluator process that plays games against a fixed set of MCTS search counts."""
  np.random.seed(initial_seed)
  random.seed(initial_seed)

  # Determine log_path for RemoteEvaluator for evaluators
  evaluator_log_path_for_eval_process = os.path.join(config.path, f"evaluator_{num}_remote_eval")
  # os.makedirs(evaluator_log_path_for_eval_process, exist_ok=True) # Ensure dir exists

  if logger is None:
      logger = file_logger.FileLogger(config.path, f"evaluator_{num}", not config.quiet)
  
  logger.print(f"Evaluator {num} started with initial_seed: {initial_seed}")

  # Initialize RemoteEvaluator for this evaluator process
  az_evaluator = RemoteEvaluator(
      game=game,
      actor_id=num, # Use the actor/evaluator number as its ID
      inference_request_queue=inference_request_queue,
      inference_response_queue=inference_response_queue,
      max_cache_size=config.evaluator_cache_size,
      numeric_log_level=config.log_level, # CHANGED: Pass numeric_log_level from config
      # debug_mode=(getattr(config, 'evaluator_verbosity', config.log_level) >= _EVALUATOR_DEBUG_LEVEL), # OLD: Pass debug_mode
      log_path=evaluator_log_path_for_eval_process # Pass the constructed log_path
  )

  # Create bots with different MCTS budgets for evaluation
  eval_bots_configs = []
  for i in range(config.eval_levels):
    simulations = config.max_simulations // (2**(config.eval_levels - 1 - i))
    if simulations == 0: simulations = 1 # Ensure at least 1 simulation
    
    # Bot for player 0 (the agent being evaluated)
    # Evaluation bots do not use Dirichlet noise.
    bot0_eval_config = config._replace(max_simulations=simulations)
    bot0 = _init_bot(bot0_eval_config, game, az_evaluator, True, 0) # player_id 0
    
    # Bot for player 1 (opponent, also uses the agent's policy but potentially different MCTS budget)
    # Typically, for evaluation, both players use the same policy but might have symmetric MCTS settings.
    # Here, we assume a symmetric setup where the opponent is also an AZ bot with the same (potentially reduced) MCTS count.
    bot1_eval_config = config._replace(max_simulations=simulations) 
    bot1 = _init_bot(bot1_eval_config, game, az_evaluator, True, 1) # player_id 1
    
    eval_bots_configs.append({
        "simulations": simulations,
        "bots": [bot0, bot1] # Assuming a 2-player game
    })

  game_num_iterator = itertools.count(start=num, step=config.evaluators)
  
  try:
    while True:
      try:
        game_num = next(game_num_iterator)
        numpy_seed_for_game = initial_seed + game_num # Unique seed for this game

        for bot_config_info in eval_bots_configs:
            simulations = bot_config_info["simulations"]
            current_eval_bots = bot_config_info["bots"]
            
            if config.log_level >= 2: # INFO
                logger.print(f"Evaluator {num} playing game {game_num} with {simulations} simulations.")

            trajectory = _play_game(
                logger,
                game_num, # Pass actual game_num for logging
                game,
                current_eval_bots,
                temperature=0,  # Evaluation is deterministic, so temperature is 0
                temperature_drop=0, # Not relevant if temperature is 0
                numpy_seed=numpy_seed_for_game,
                log_level=config.log_level)
            
            if trajectory is None or not trajectory.states:
                logger.print(f"Evaluator {num} game {game_num} with {simulations} sims: Trajectory empty, skipping.")
                continue

            game_returns = trajectory.returns 
            
            outcome_player0 = 0
            if game_returns[0] > game_returns[1]: # P0 won
                outcome_player0 = 1
            elif game_returns[0] < game_returns[1]: # P0 lost
                outcome_player0 = -1
                
            eval_result = (len(trajectory.states), outcome_player0, simulations)
            
            if config.log_level >= 2: # INFO
                logger.print(f"Evaluator {num} game {game_num} with {simulations} sims: result {eval_result}")
            queue.put(eval_result)

      except (TimeoutError, std_queue.Full, std_queue.Empty) as e_transient: # Catch specific transient errors
        if logger:
            logger.print(f"Evaluator {num} caught transient error in main loop: {type(e_transient).__name__}: {e_transient}. Continuing.")
        time.sleep(1)  # Brief pause before continuing the loop
        continue
      # More critical errors within the loop will fall through to the outer Exception handler.

  except ShutdownException:
    logger.print(f"Evaluator {num} received shutdown signal. Exiting.")
  except Exception as e: # pylint: disable=broad-except
    logger.print(f"Evaluator {num} caught unhandled error: {e}\n{traceback.format_exc()}")
  finally:
    # Similar to actor, best-effort signal.
    try:
        # The tuple structure here was: (SHUTDOWN_SENTINEL, unique_evaluator_id, None, None, None)
        # unique_evaluator_id = config.actors + num
        # Removing to avoid deserialization errors.
        pass 
    except Exception: # pylint: disable=broad-except
        pass
    logger.print(f"Evaluator {num} finished.") 