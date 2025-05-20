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
import concurrent.futures
from typing import Any, Dict, Optional

import numpy as np
import pyspiel
import queue as std_queue # For specific exception types like Full/Empty

from open_spiel.python.algorithms import mcts
from open_spiel.python.utils import file_logger, spawn # spawn for ProcessQueue type hint
from .remote_inference import RemoteEvaluator, SHUTDOWN_SENTINEL, ShutdownException
from .alpha_zero_jax_async_mcts import AlphaZeroJaxAsyncMCTSBot

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
               child_selection_fn=mcts.SearchNode.uct_value):

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

  def step_with_policy(self, state):
    """Returns policy and action from MCTS search."""
    # Chance node handling is now done globally in _play_game
    # and MCTSBot is initialized with dont_return_chance_node=True.
    return super().step_with_policy(state)


class _AsyncRemoteEvaluatorAdapter:
  """Adapter to wrap RemoteEvaluator for async MCTS."""
  def __init__(self, remote_eval):
    self._remote = remote_eval

  def prior_and_value(self, state, mcts_search_timeout_sec: float):
    """Passes through to RemoteEvaluator's prior_and_value with timeout."""
    current_player = state.current_player()
    num_players = self._remote._game.num_players() # Access game from the remote evaluator

    raw_value, policy_probs = self._remote.prior_and_value(state, mcts_search_timeout_sec=mcts_search_timeout_sec)

    if raw_value is None or policy_probs is None:
        return [], np.zeros(num_players, dtype=np.float32) # Return empty prior and zero values for all players

    # Convert policy_probs (np.ndarray) to list of (action, prob) tuples
    if state.is_chance_node(): 
        prior_tuples = state.chance_outcomes()
        # Value should already be game returns for chance/terminal from RemoteEvaluator
        # Ensure it's a numpy array if not already.
        processed_value = np.array(raw_value, dtype=np.float32) if not isinstance(raw_value, np.ndarray) else raw_value
    elif state.is_terminal(): 
        prior_tuples = []
        processed_value = np.array(raw_value, dtype=np.float32) if not isinstance(raw_value, np.ndarray) else raw_value
    else:
        legal_actions = state.legal_actions(current_player)
        if not legal_actions: 
            prior_tuples = []
        else:
            prior_tuples = []
            for action in legal_actions:
                if action < len(policy_probs):
                    prior_tuples.append((action, policy_probs[action]))
                else:
                    # This case should ideally not happen if policy_probs covers all legal actions.
                    # Log a warning or handle as appropriate if it does.
                    prior_tuples.append((action, 0.0)) # Default to 0 prob if action index out of bounds
        
        processed_value = np.zeros(num_players, dtype=np.float32) # Default value
        current_player_val_set = False # Flag to track if a value was successfully processed

        if isinstance(raw_value, np.ndarray):
            if raw_value.size == 1:  # If it's a single-element array (could be 0-d or 1-d)
                scalar_val = raw_value.item() # This should now be safe
                processed_value[current_player] = scalar_val
                if num_players == 2:
                    processed_value[1 - current_player] = -scalar_val
                # For N>2 players and scalar input, other players' values remain 0.
                current_player_val_set = True
            elif raw_value.ndim == 1 and raw_value.size == num_players:  # Assumed per-player utilities
                if hasattr(self._remote, 'logger') and self._remote.logger:
                    logger_obj = self._remote.logger
                    msg = f"Adapter: raw_value (shape {raw_value.shape}) used as per-player utilities."
                    if hasattr(logger_obj, 'print'):
                        logger_obj.print(msg, min_level=2) # INFO for FileLogger
                    else:
                        logger_obj.info(msg) # Standard logger
                processed_value = raw_value.astype(np.float32)
                current_player_val_set = True # Values for all players are set
            else:  # Unexpected np.ndarray shape/size
                if hasattr(self._remote, 'logger') and self._remote.logger:
                    logger_obj = self._remote.logger
                    msg = f"Adapter ERROR: raw_value (np.ndarray) has unhandled shape {raw_value.shape} or size {raw_value.size}. Num_players: {num_players}. Using zeros."
                    if hasattr(logger_obj, 'print'):
                        logger_obj.print(msg, min_level=0) # ERROR for FileLogger
                    else:
                        logger_obj.error(msg) # Standard logger
                # processed_value remains zeros, current_player_val_set remains False
        elif isinstance(raw_value, (float, int)):  # Python scalar
            if hasattr(self._remote, 'logger') and self._remote.logger:
                 logger_obj = self._remote.logger
                 msg = f"Adapter WARNING: raw_value is Python scalar {raw_value}. Converting."
                 if hasattr(logger_obj, 'print'):
                     logger_obj.print(msg, min_level=1) # WARNING for FileLogger
                 else:
                     logger_obj.warning(msg) # Standard logger
            scalar_val = float(raw_value)
            processed_value[current_player] = scalar_val
            if num_players == 2:
                processed_value[1 - current_player] = -scalar_val
            current_player_val_set = True
        elif isinstance(raw_value, (list, tuple)) and len(raw_value) == num_players: # List/tuple of per-player values
            if hasattr(self._remote, 'logger') and self._remote.logger:
                 logger_obj = self._remote.logger
                 msg = f"Adapter INFO: raw_value is list/tuple, size {len(raw_value)}. Converting."
                 if hasattr(logger_obj, 'print'):
                     logger_obj.print(msg, min_level=2) # INFO for FileLogger
                 else:
                     logger_obj.info(msg) # Standard logger
            processed_value = np.array(raw_value, dtype=np.float32)
            current_player_val_set = True
        
        if not current_player_val_set: # If none of the above conditions handled it
            if hasattr(self._remote, 'logger') and self._remote.logger:
                 logger_obj = self._remote.logger
                 msg = f"Adapter ERROR: raw_value type {type(raw_value)} or format unhandled. raw_value: {str(raw_value)[:100]}. Using zeros."
                 if hasattr(logger_obj, 'print'):
                     logger_obj.print(msg, min_level=0) # ERROR for FileLogger
                 else:
                     logger_obj.error(msg) # Standard logger
            # processed_value remains zeros as initialized

    return prior_tuples, processed_value


def _init_bot(config, game: pyspiel.Game, evaluator_: mcts.Evaluator,
              evaluation: bool, player_id_for_bot: int, actor_specific_logger=None):
  """Initializes a bot for playing or evaluation."""
  # Choose async or sync MCTS based on config
  if getattr(config, "async_mode", False):
    adapter = _AsyncRemoteEvaluatorAdapter(evaluator_)
    bot = AlphaZeroJaxAsyncMCTSBot(
        game=game,
        uct_c=config.uct_c,
        max_simulations=config.max_simulations,
        evaluator=adapter,
        solve=False,
        verbose=config.log_level >= 4,
        child_selection_fn=mcts.SearchNode.uct_value,
        dirichlet_noise=(config.policy_epsilon, config.policy_alpha) if not evaluation else None,
        dont_return_chance_node=True,
        virtual_loss=getattr(config, "async_virtual_loss", 10),
        batch_size=getattr(config, "async_batch_size", 16),
        timeout=getattr(config, "async_timeout", 5.0),
        actor_logger=actor_specific_logger,
        numeric_log_level=config.log_level
    )
    bot.player_id = player_id_for_bot
    return bot

  # Fallback to synchronous AlphaZeroBot
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
  # Note: logger here is the watcher's FileLogger instance.
  logger.print(f"Actor {num} starting with PID {os.getpid()} and seed {initial_seed}")
  # Explicitly seed the actor's random number generator for its operations.
  random.seed(initial_seed)
  np.random.seed(initial_seed)

  # Initialize the RemoteEvaluator for this actor
  remote_evaluator = RemoteEvaluator(
      game=game,
      actor_id=num, 
      inference_request_queue=inference_request_queue,
      inference_response_queue=inference_response_queue,
      numeric_log_level=config.log_level,
      max_cache_size=config.evaluator_cache_size, # Using a common cache size config
      log_path=config.path
  )
  remote_evaluator.start_response_handler() # Start the handler thread

  try:
    # Initialize the bot for self-play
    # The actor_specific_logger passed to _init_bot can be the watcher's logger
    bot = _init_bot(config, game, remote_evaluator, evaluation=False, player_id_for_bot=-1, actor_specific_logger=logger)

    # Calculate temperature schedule
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

    logger.print(f"Actor {num} played {game_num -1} games, {game_num -1} trajectories.")

  except ShutdownException:
    logger.print(f"Actor {num} received ShutdownException. Exiting gracefully.")
  except Exception as e: # pylint: disable=broad-except
    logger.error(f"Actor {num} encountered an unhandled exception: {type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
    # Potentially re-raise or handle to ensure process terminates if supervisor expects it.
  finally:
    logger.print(f"Actor {num} stopping...")
    if remote_evaluator: # Ensure it was initialized
        remote_evaluator.stop_response_handler()
    # Any other cleanup specific to the actor
    logger.print(f"Actor {num} has stopped.")


@watcher
def evaluator(*, game: pyspiel.Game, config, logger, num: int, # config is ConfigJAX
                queue: spawn._ProcessQueue, initial_seed: int,
                inference_request_queue, # mp.Queue
                inference_response_queue # mp.Queue
                ):
  """An evaluator process that evaluates the model against a baseline."""
  logger.print(f"Evaluator {num} starting with PID {os.getpid()} and seed {initial_seed}")
  random.seed(initial_seed)
  np.random.seed(initial_seed)

  # Initialize the RemoteEvaluator for this evaluator process
  remote_evaluator = RemoteEvaluator(
      game=game,
      actor_id=1000 + num, # Use a different ID range for evaluators to distinguish logs
      inference_request_queue=inference_request_queue,
      inference_response_queue=inference_response_queue,
      numeric_log_level=config.log_level,
      max_cache_size=config.evaluator_cache_size, # Using a common cache size config
      log_path=config.path
  )
  remote_evaluator.start_response_handler() # Start the handler thread

  model_player_id = 0
  baseline_player_id = 1

  try:
    # Initialize the bot for the model being evaluated
    # The actor_specific_logger can be the watcher's logger
    model_bot = _init_bot(config, game, remote_evaluator, evaluation=True, player_id_for_bot=model_player_id, actor_specific_logger=logger)

    # Initialize a baseline bot (e.g., a random player or a simpler MCTS)
    # Create bots with different MCTS budgets for evaluation
    eval_bots_configs = []
    for i in range(config.eval_levels):
      simulations = config.max_simulations // (2**(config.eval_levels - 1 - i))
      if simulations == 0: simulations = 1 # Ensure at least 1 simulation
      
      # Bot for player 0 (the agent being evaluated)
      # Evaluation bots do not use Dirichlet noise.
      bot0_eval_config = config._replace(max_simulations=simulations)
      bot0 = _init_bot(bot0_eval_config, game, remote_evaluator, True, 0, actor_specific_logger=logger) # player_id 0
      
      # Bot for player 1 (opponent, also uses the agent's policy but potentially different MCTS budget)
      # Typically, for evaluation, both players use the same policy but might have symmetric MCTS settings.
      # Here, we assume a symmetric setup where the opponent is also an AZ bot with the same (potentially reduced) MCTS count.
      bot1_eval_config = config._replace(max_simulations=simulations) 
      bot1 = _init_bot(bot1_eval_config, game, remote_evaluator, True, 1, actor_specific_logger=logger) # player_id 1
      
      eval_bots_configs.append({
          "simulations": simulations,
          "bots": [bot0, bot1] # Assuming a 2-player game
      })

    game_num_iterator = itertools.count(start=num, step=config.evaluators)
    
    for game_num in game_num_iterator:
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

    logger.print(f"Evaluator {num} finished: {game_num -1} evals, {game_num -1} trajectories.")

  except ShutdownException:
    logger.print(f"Evaluator {num} received ShutdownException. Exiting gracefully.")
  except Exception as e: # pylint: disable=broad-except
    logger.error(f"Evaluator {num} encountered an unhandled exception: {type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
  finally:
    logger.print(f"Evaluator {num} stopping...")
    if remote_evaluator: # Ensure it was initialized
        remote_evaluator.stop_response_handler()
    logger.print(f"Evaluator {num} has stopped.") 