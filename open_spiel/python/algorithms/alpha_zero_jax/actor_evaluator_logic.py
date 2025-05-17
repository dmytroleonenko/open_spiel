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

import numpy as np
import pyspiel
import queue as std_queue # For specific exception types like Full/Empty

from open_spiel.python.algorithms import mcts
from open_spiel.python.utils import file_logger, spawn # spawn for ProcessQueue type hint
from .remote_inference import RemoteEvaluator, SHUTDOWN_SENTINEL, ShutdownException

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
        dont_return_chance_node=dont_return_chance_node,
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
  # Required for np.random operations to be deterministic when using spawn
  # This needs to be called in each process that uses np.random.
  np.random.seed(numpy_seed)

  trajectory = Trajectory()
  state = game.new_initial_state()
  random_state = np.random.RandomState(numpy_seed)

  if log_level >= 3: # DEBUG
    logger.print(f"Starting game {game_num} with numpy_seed: {numpy_seed}, "
                 f"initial temperature: {temperature}, temp_drop: {temperature_drop}")


  while not state.is_terminal():
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
    action, policy_dict = bot.step_with_policy(state)
    
    # Convert policy from dict to a dense array based on legal actions
    # This policy is what MCTS search, after noise and temperature, recommends.
    policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    for act, prob in policy_dict:
        policy[act] = prob
    
    # Store the state, action, policy
    # Value will be filled in later by the learner after the game is done.
    trajectory.add(
        TrajectoryState(state.observation_tensor(), current_player,
                        state.legal_actions_mask(), action, policy,
                        value=0.0)) # Placeholder value

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
  return trajectory


@watcher
def actor(*, game: pyspiel.Game, config, logger, num: int, # config is ConfigJAX
          queue: spawn.ProcessQueue, initial_seed: int,
          inference_request_queue, # mp.Queue
          inference_response_queue # mp.Queue
          ):
  """An actor process that plays games and sends trajectories to the learner."""
  # Each actor needs its own PRNG state for numpy operations.
  # JAX PRNG keys are not used here directly.
  np.random.seed(initial_seed)
  random.seed(initial_seed) # Also for Python's random if used by game or other logic

  if logger is None: # Fallback if watcher didn't inject logger (e.g. if not decorated)
      logger = file_logger.FileLogger(config.path, f"actor_{num}", not config.quiet)

  logger.print(f"Actor {num} started with initial_seed: {initial_seed}, numpy_seed: {initial_seed}")

  # Initialize the RemoteEvaluator for this actor
  # The actor_id (num) is used by the InferenceServicer to route responses.
  az_evaluator = RemoteEvaluator(
      actor_id=num, # Unique ID for this actor/client
      request_queue=inference_request_queue,
      response_queue=inference_response_queue,
      timeout_ms=config.remote_evaluator_timeout_ms,
      logger=logger,
      log_level=config.log_level
  )

  bots = [_init_bot(config, game, az_evaluator, False, p) for p in range(game.num_players())]
  
  # Seed for game num to ensure different games if multiple actors start "simultaneously"
  # And to ensure that _play_game gets a deterministic seed based on game_num + initial_seed
  game_num_iterator = itertools.count(start=num, step=config.actors)


  try:
    while True:
      try:
        game_num = next(game_num_iterator)
        
        # Construct a unique seed for this specific game play
        # to ensure np.random operations within _play_game are deterministic
        # for this game instance across potential reruns or different actor setups.
        numpy_seed_for_game = initial_seed + game_num

        trajectory = _play_game(
            logger,
            game_num,
            game,
            bots,
            config.temperature, # This temperature is for action selection policy
            config.temperature_drop,
            numpy_seed=numpy_seed_for_game,
            log_level=config.log_level)
        
        if trajectory is None or not trajectory.states:
            logger.print(f"Actor {num} game {game_num}: Trajectory was None or empty, skipping.")
            continue

        # Send the trajectory to the learner via the shared queue
        # The learner will then update the values in the trajectory.
        if config.log_level >= 3: # DEBUG
            logger.print(f"Actor {num} game {game_num}: sending trajectory of {len(trajectory.states)} states.")
        queue.put(trajectory) # This could block or raise if queue is mismanaged, but typically ProcessQueue handles it.

      except (TimeoutError, std_queue.Full, std_queue.Empty) as e_transient: # Catch specific transient errors
        if logger:
            logger.print(f"Actor {num} caught transient error in main loop: {type(e_transient).__name__}: {e_transient}. Continuing.")
        time.sleep(1)  # Brief pause before continuing the loop
        continue
      # More critical errors within the loop will fall through to the outer Exception handler.

  except ShutdownException:
    logger.print(f"Actor {num} received shutdown signal. Exiting.")
  except Exception as e: # pylint: disable=broad-except
    logger.print(f"Actor {num} caught unhandled error: {e}
{traceback.format_exc()}")
    # Optionally re-raise or signal main process
  finally:
    # Attempt to signal shutdown to the inference request queue.
    # This is mostly a best-effort and might be redundant if main servicer shutdown is robust.
    try:
        # The tuple structure here was: (SHUTDOWN_SENTINEL, actor_id, None, None, None)
        # BatchAssemblyThread expects SHUTDOWN_SENTINEL directly or an InferenceRequest.
        # Sending the simple SHUTDOWN_SENTINEL might be cleaner if this signal is desired,
        # but the main servicer.stop() should handle this.
        # Removing to avoid deserialization errors in BatchAssemblyThread.
        pass 
    except Exception: # pylint: disable=broad-except
        # Log if needed, but generally suppress errors during shutdown signaling.
        pass
    logger.print(f"Actor {num} finished.")


@watcher
def evaluator(*, game: pyspiel.Game, config, logger, num: int, # config is ConfigJAX
                queue: spawn.ProcessQueue, initial_seed: int,
                inference_request_queue, # mp.Queue
                inference_response_queue # mp.Queue
                ):
  """An evaluator process that plays games against a fixed set of MCTS search counts."""
  np.random.seed(initial_seed)
  random.seed(initial_seed)

  if logger is None:
      logger = file_logger.FileLogger(config.path, f"evaluator_{num}", not config.quiet)
  
  logger.print(f"Evaluator {num} started with initial_seed: {initial_seed}")

  # Initialize RemoteEvaluator for this evaluator process
  az_evaluator = RemoteEvaluator(
      actor_id=config.actors + num, # Unique ID, offset from actor IDs
      request_queue=inference_request_queue,
      response_queue=inference_response_queue,
      timeout_ms=config.remote_evaluator_timeout_ms,
      logger=logger,
      log_level=config.log_level
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
    logger.print(f"Evaluator {num} caught unhandled error: {e}
{traceback.format_exc()}")
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