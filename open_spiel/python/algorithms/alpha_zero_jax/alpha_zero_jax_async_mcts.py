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

"""An asynchronous Monte Carlo Tree Search algorithm for game play.

This implements asynchronous MCTS which runs evaluations asynchronously in
parallel. This version is adapted for AlphaZero JAX.
"""

from __future__ import annotations
import concurrent.futures
import math
import time
from typing import Callable, Optional

from absl import logging
import numpy as np

import pyspiel


def robust_child_with_total_reward_tiebreaker(
    root: AlphaZeroJaxAsyncSearchNode,
) -> tuple[int, AlphaZeroJaxAsyncSearchNode]:
  selection_criteria = lambda node: (node.explore_count, node.total_reward)
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def robust_child(root: AlphaZeroJaxAsyncSearchNode) -> tuple[int, AlphaZeroJaxAsyncSearchNode]:
  selection_criteria = lambda node: node.explore_count
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def max_child(root: AlphaZeroJaxAsyncSearchNode) -> tuple[int, AlphaZeroJaxAsyncSearchNode]:
  selection_criteria = (
      lambda node: node.total_reward / node.explore_count
      if node.explore_count
      else float("-inf")
  )
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def max_robust_child(
    root: AlphaZeroJaxAsyncSearchNode, find_robust: bool = False
) -> tuple[Optional[int], Optional[AlphaZeroJaxAsyncSearchNode]]:
  if find_robust:
    best_action, best_child = robust_child(root)
  else:
    _, max_child_node = max_child(root)
    _, robust_child_node = robust_child(root)
    best_action, best_child = None, None
    for child in root.children:
      if child == max_child_node and child == robust_child_node:
        best_action, best_child = child.action, child
        break
  return best_action, best_child


def secure_child(
    root: AlphaZeroJaxAsyncSearchNode, secure_c: float = 1.0
) -> tuple[int, AlphaZeroJaxAsyncSearchNode]:
  selection_criteria = (
      lambda node: node.total_reward / node.explore_count  # pylint: disable=g-long-ternary
      - secure_c / math.sqrt(node.explore_count)
      if node.explore_count
      else float("-inf")
  )
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def max_robust_secure_child(
    root: AlphaZeroJaxAsyncSearchNode, secure_c: float = 1.0, find_secure: bool = False
) -> tuple[Optional[int], Optional[AlphaZeroJaxAsyncSearchNode]]:
  if find_secure:
    best_action, best_child = secure_child(root, secure_c)
  else:
    best_action, best_child = max_robust_child(root)
  return best_action, best_child


class AlphaZeroJaxAsyncEvaluator:
  def prior_and_value(
      self, state: pyspiel.State, mcts_search_timeout_sec: float # Added for compatibility
  ) -> tuple[list[tuple[int, float]], np.ndarray]:
    raise NotImplementedError


class AlphaZeroJaxAsyncRandomRolloutEvaluator(AlphaZeroJaxAsyncEvaluator):
  def __init__(
      self,
      random_state: np.random.RandomState | None = None,
  ):
    self._random_state = random_state or np.random.RandomState()

  def prior_and_value(
      self, state: pyspiel.State, mcts_search_timeout_sec: float # Added
  ) -> tuple[list[tuple[int, float]], np.ndarray]:
    if state.is_chance_node():
      prior = state.chance_outcomes()
    else:
      legal_actions = state.legal_actions(state.current_player())
      prior = [(action, 1.0 / len(legal_actions)) for action in legal_actions]
    working_state = state.clone()
    while not working_state.is_terminal():
      if working_state.is_chance_node():
        outcomes = working_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = self._random_state.choice(action_list, p=prob_list)
      else:
        action = self._random_state.choice(working_state.legal_actions())
      working_state.apply_action(action)
    value = np.array(working_state.returns())
    return prior, value


class AlphaZeroJaxAsyncSearchNode(object):
  __slots__ = [
      "action",
      "player",
      "prior",
      "explore_count",
      "total_reward",
      "outcome",
      "children",
      "expanded",
  ]

  def __init__(self, action: int | None, player: int, prior: float):
    self.action = action
    self.player = player
    self.prior = prior
    self.explore_count = 0
    self.total_reward = 0.0
    self.outcome = None
    self.children = []
    self.expanded = False

  def uct_value(self, parent_explore_count: int, uct_c: float) -> float:
    if self.outcome is not None:
      return self.outcome[self.player]
    if self.explore_count == 0:
      return float("inf")
    return self.total_reward / self.explore_count + uct_c * math.sqrt(
        math.log(parent_explore_count) / self.explore_count
    )

  def puct_value(self, parent_explore_count: int, uct_c: float) -> float:
    if self.outcome is not None:
      return self.outcome[self.player]
    return (
        self.explore_count and self.total_reward / self.explore_count
    ) + uct_c * self.prior * math.sqrt(parent_explore_count) / (
        self.explore_count + 1
    )

  def sort_key(self):
    return (
        0 if self.outcome is None else self.outcome[self.player],
        self.explore_count,
        self.total_reward,
    )

  def best_child(self):
    return max(self.children, key=AlphaZeroJaxAsyncSearchNode.sort_key)

  def children_str(self, state=None):
    return "\n".join([
        c.to_str(state)
        for c in reversed(sorted(self.children, key=AlphaZeroJaxAsyncSearchNode.sort_key))
    ])

  def to_str(self, state=None):
    action = (
        state.action_to_string(state.current_player(), self.action)
        if state and self.action is not None
        else str(self.action)
    )
    return (
        "{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
        "outcome: {}, {:3d} children"
    ).format(
        action,
        self.player,
        self.prior,
        self.explore_count and self.total_reward / self.explore_count,
        self.explore_count,
        (
            "{:4.1f}".format(self.outcome[self.player])
            if self.outcome
            else "none"
        ),
        len(self.children),
    )

  def __str__(self):
    return self.to_str(None)


class AlphaZeroJaxAsyncMCTSBot(pyspiel.Bot):
  def __init__(
      self,
      game,
      uct_c,
      max_simulations,
      evaluator: AlphaZeroJaxAsyncEvaluator, # Use renamed Evaluator
      solve=True,
      random_state=None,
      child_selection_fn=AlphaZeroJaxAsyncSearchNode.uct_value, # Use renamed SearchNode
      best_child_fn: Callable[
          ..., tuple[Optional[int], Optional[AlphaZeroJaxAsyncSearchNode]] # Use renamed SearchNode
      ] = robust_child_with_total_reward_tiebreaker,
      dirichlet_noise=None,
      verbose=False,
      dont_return_chance_node=False,
      virtual_loss: int = 10,
      batch_size: int = 16,
      secure_c: float = 1.0,
      simulations_multiplier: float = 1.0,
      max_additional_simulation_rounds: int = 0,
      timeout: float = 5.0,
      actor_logger=None,
      numeric_log_level: int = 2,
  ):
    pyspiel.Bot.__init__(self)
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")

    self.game = game
    self.player_id = None
    self.uct_c = uct_c
    self._max_simulations = max_simulations
    self.max_simulations = max_simulations
    self.evaluator = evaluator
    self._solve = solve
    self._random_state = random_state or np.random.RandomState()
    self._child_selection_fn = child_selection_fn
    self._best_child_fn = best_child_fn
    self._dirichlet_noise = dirichlet_noise
    self.verbose = verbose
    self.dont_return_chance_node = dont_return_chance_node
    self._root = None
    self.min_utility = game.min_utility() if game else -1.0
    self.max_utility = game.max_utility() if game else 1.0

    self.virtual_loss = virtual_loss
    self.batch_size = batch_size
    self.timeout = timeout
    self.total_timeouts = 0
    self.total_num_searches = 0
    self.total_eval_errors = 0
    self.total_search_time = 0

    self._secure_c = secure_c
    self._alternative_criteria = False
    self._max_additional_simulation_rounds = max_additional_simulation_rounds
    self._simulations_multiplier = simulations_multiplier
    self.actor_logger = actor_logger
    self.numeric_log_level = numeric_log_level

  def restart_at(self, state):
    pass

  def get_root(self):
    return self._root

  def _get_selection_function_arguments(self, root):
    if self._best_child_fn is secure_child:
      arguments = (root, self._secure_c)
    elif self._best_child_fn is max_robust_child:
      arguments = (root, self._alternative_criteria)
    elif self._best_child_fn is max_robust_secure_child:
      arguments = (root, self._secure_c, self._alternative_criteria)
    else:
      arguments = (root,)
    return arguments

  def step_with_policy(self, state):
    t1 = time.time()
    simulation_round = 0
    best_action = None
    best_child = None
    root = None
    while (
        best_action is None
        and simulation_round <= self._max_additional_simulation_rounds
    ):
      simulation_round += 1
      if simulation_round == self._max_additional_simulation_rounds:
        self._alternative_criteria = True
      root = self.mcts_search(state)
      assert root is not None, "Root is None"
      self._root = root
      arguments = self._get_selection_function_arguments(root)
      best_action, best_child = self._best_child_fn(*arguments)
      if best_action is None:
        self.max_simulations = (
            int(self._simulations_multiplier * self._max_simulations)
        )
    assert best_action is not None, "Best action is None"
    assert best_child is not None, "Best child is None"
    assert self._root is not None, "Root is None"
    seconds = time.time() - t1
    self.total_search_time += seconds
    if self.verbose:
      # logging.info is used here because self.actor_logger might not be set
      # or might be a FileLogger without specific levels.
      logging.info(
          "Finished %s sims in %.3f secs, %.1f sims/s",
          root.explore_count, seconds, root.explore_count / seconds
      )
      logging.info("Root:")
      logging.info(root.to_str(state))
      logging.info("Children:")
      logging.info(root.children_str(state))
      if best_child.children:
        chosen_state = state.clone()
        chosen_state.apply_action(best_action)
        logging.info("Children of chosen:")
        logging.info(best_child.children_str(chosen_state))

    policy = [
        (action, (1.0 if action == best_action else 0.0))
        for action in state.legal_actions(state.current_player())
    ]
    self.max_simulations = self._max_simulations
    return policy, best_action

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _add_virtual_losses(self, node):
    node.total_reward += (self.virtual_loss * self.min_utility)
    node.explore_count += self.virtual_loss

  def _remove_virtual_losses(self, node):
    node.total_reward -= (self.virtual_loss * self.min_utility)
    node.explore_count -= self.virtual_loss

  def _choose_next_node(self, visit_path, working_state, current_node):
    if working_state.is_chance_node():
      outcomes = working_state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = self._random_state.choice(action_list, p=prob_list)
      chosen_child = next(
          c for c in current_node.children if c.action == action
      )
    else:
      chosen_child = max(
          current_node.children,
          key=lambda c: self._child_selection_fn(
              c, current_node.explore_count, self.uct_c
          ),
      )
    working_state.apply_action(chosen_child.action)
    current_node = chosen_child
    self._add_virtual_losses(current_node)
    visit_path.append(current_node)
    return current_node

  def _apply_tree_policy(self, root, state):
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    self._add_virtual_losses(root)
    unexplored_explore_count = self.virtual_loss
    while (
        not working_state.is_terminal()
        and current_node.explore_count > unexplored_explore_count
    ) or (working_state.is_chance_node() and self.dont_return_chance_node):
      if not current_node.children:
        # This node is unexpanded.
        # If it's a chance node that we are supposed to skip (dont_return_chance_node is True),
        # then we must not return it for evaluation via prior_and_value.
        # Instead, we need to "pass through" it by sampling an outcome and continuing the traversal.
        if working_state.is_chance_node() and self.dont_return_chance_node:
          # Populate children for this chance node (current_node is parent of outcomes)
          # using the chance outcomes themselves as priors. The 'player' for these
          # outcome nodes should be pyspiel.PlayerId.CHANCE.
          outcomes_with_probs = working_state.chance_outcomes()
          current_node.children = [
              AlphaZeroJaxAsyncSearchNode(action, pyspiel.PlayerId.CHANCE, prob)
              for action, prob in outcomes_with_probs
          ]
          # Note: current_node.expanded remains False here, as "true" expansion
          # involves priors from the neural network evaluator. This temporary
          # child population is for traversal purposes only.
          
          # Now that children representing chance outcomes are created,
          # call _choose_next_node. It will sample an outcome, apply it to working_state,
          # update current_node to be the chosen outcome node, and add virtual losses.
          current_node = self._choose_next_node(visit_path, working_state, current_node)
          
          # After _choose_next_node, working_state has advanced past the chance node,
          # and current_node is the outcome. We must continue the loop to evaluate
          # conditions for this new current_node (e.g., is it terminal? is it unexplored?).
          continue 
        else:
          # It's an unexpanded decision_node, or a chance node that we are NOT skipping
          # (i.e., dont_return_chance_node is False). This is a true leaf for evaluation.
          return visit_path, working_state, current_node
      else: # Node has children (already expanded, or chance outcomes populated by a previous iteration of this logic)
        current_node = self._choose_next_node(
            visit_path, working_state, current_node
        )
    return visit_path, working_state, current_node

  def backpropagate(self, visit_path, returns):
    while visit_path:
      decision_node_idx = -1
      while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
        decision_node_idx -= 1
      target_return = returns[visit_path[decision_node_idx].player]
      node = visit_path.pop()
      node.total_reward += target_return
      node.explore_count += 1
      self._remove_virtual_losses(node)
      assert node.explore_count >= 1

  def backpropagate_timeout(self, visit_path):
    while visit_path:
      node = visit_path.pop()
      self._remove_virtual_losses(node)

  def expand(self, root, working_state, current_node, prior, value):
    if current_node is root and self._dirichlet_noise:
      epsilon, alpha = self._dirichlet_noise
      noise = self._random_state.dirichlet([alpha] * len(prior))
      prior = [
          (a, (1 - epsilon) * p + epsilon * n)
          for (a, p), n in zip(prior, noise)
      ]
    self._random_state.shuffle(prior)
    player = working_state.current_player()
    current_node.children = [
        AlphaZeroJaxAsyncSearchNode(action, player, prob) for action, prob in prior # Use renamed SearchNode
    ]
    current_node.expanded = True

  def evaluate(
      self, working_state
  ) -> tuple[list[tuple[int, float]], np.ndarray]:
    if working_state.is_terminal():
      prior = []
      values = working_state.returns()
    else:
      prior, values = self.evaluator.prior_and_value(working_state, mcts_search_timeout_sec=self.timeout)
    return prior, values

  def handle_leaf(self, prior, value, arguments, timeout=False):
    visit_path, working_state, node, root = arguments
    if timeout:
      self.backpropagate_timeout(visit_path)
      return
    assert node is not None
    if not node.expanded:
      self.expand(root, working_state, node, prior, value)
    if working_state.is_terminal():
      visit_path[-1].outcome = working_state.returns()
    self.backpropagate(visit_path, value)

  def async_mcts_search(self, state):
    root = AlphaZeroJaxAsyncSearchNode(None, state.current_player(), 1) # Use renamed SearchNode
    self._add_virtual_losses(root)
    working_state_initial = state.clone()
    # Initial evaluation before starting the parallel batch processing
    prior_initial, value_initial = self.evaluate(working_state_initial)
    self.handle_leaf(
        prior_initial, value_initial, ([root], working_state_initial, root, root), timeout=False
    )
    
    # total_simulations should start from 1 as one simulation is done above
    simulations_done_this_search = 1
    search_timeouts = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self.batch_size 
    ) as executor:
      while simulations_done_this_search < self.max_simulations:
        
        remaining_simulations_for_loop = self.max_simulations - simulations_done_this_search
        num_to_queue_this_iteration = min(self.batch_size, remaining_simulations_for_loop)
        
        if num_to_queue_this_iteration <= 0: # Should not happen if loop condition is correct
            break

        futures = set()
        batch_for_evaluation_args = [] # Stores arguments for handle_leaf for this batch

        def evaluate_batch_local(batch_to_evaluate_tuples):
            # batch_to_evaluate_tuples is a list of (leaf_state, arguments_for_leaf)
            results_for_batch = []
            for leaf_state_local, arguments_for_leaf_local in batch_to_evaluate_tuples:
                if self.actor_logger and self.numeric_log_level >= 3: # DEBUG
                    self.actor_logger.print(f"DEBUG: Evaluating leaf state: {leaf_state_local}")
                
                # The evaluator might return (None, None) or (prior, np.array_of_zeros) on timeout/error
                prior_local, value_local = self.evaluator.prior_and_value(leaf_state_local, mcts_search_timeout_sec=self.timeout)
                results_for_batch.append(((prior_local, value_local), arguments_for_leaf_local))
            return results_for_batch

        # Prepare batch by applying tree policy
        for _ in range(num_to_queue_this_iteration):
            visit_path, working_state_leaf, current_node_leaf = self._apply_tree_policy(root, state)
            if current_node_leaf is None: 
                if self.actor_logger and self.numeric_log_level >= 1: # WARNING
                    self.actor_logger.print(f"WARNING: Tree policy returned None node, skipping one simulation path.")
                # This path won't be added to batch, effectively skipping one simulation count for this iteration
                continue 
            batch_for_evaluation_args.append((working_state_leaf, (visit_path, working_state_leaf, current_node_leaf, root)))
        
        if not batch_for_evaluation_args: # If all tree policy calls failed for this iteration
            if num_to_queue_this_iteration > 0 : # If we intended to queue but couldn't
                 simulations_done_this_search += num_to_queue_this_iteration # Assume these attempts count
            continue # Try next iteration

        # Submit the prepared batch to one worker
        future = executor.submit(evaluate_batch_local, batch_for_evaluation_args)
        futures.add(future)

        # Process the single future for the batch
        # The timeout for wait/result should be based on self.timeout per simulation,
        # but here it's for the whole batch. A simple approach:
        batch_wait_timeout = self.timeout * len(batch_for_evaluation_args) 
        completed_futures, _ = concurrent.futures.wait(
            futures,
            timeout=batch_wait_timeout, 
            return_when=concurrent.futures.ALL_COMPLETED,
        )

        if future in completed_futures:
            try:
                # Get the result of the batch execution
                # Use a very short timeout here as 'wait' should have handled it
                batch_evaluation_results = future.result(timeout=0.1) 
                
                for (prior_res, value_res), args_res in batch_evaluation_results:
                    self.handle_leaf(prior_res, value_res, args_res, timeout=False)
                    simulations_done_this_search += 1 
            except concurrent.futures.TimeoutError:
                if self.actor_logger and self.numeric_log_level >= 1: # WARNING
                    self.actor_logger.print(f"WARNING: Timeout obtaining result for batch future. Marking {len(batch_for_evaluation_args)} simulations as timeout.")
                for _, args_for_timeout in batch_for_evaluation_args:
                    self.handle_leaf(None, None, args_for_timeout, timeout=True)
                    search_timeouts +=1
                simulations_done_this_search += len(batch_for_evaluation_args)
            except Exception as e:
                if self.actor_logger and self.numeric_log_level >= 0: # ERROR
                    self.actor_logger.print(f"ERROR: Exception in batch future result: {e}. Marking as timeout.")
                for _, args_for_error in batch_for_evaluation_args:
                    self.handle_leaf(None, None, args_for_error, timeout=True)
                    search_timeouts +=1
                simulations_done_this_search += len(batch_for_evaluation_args)
        else: # Future did not complete (timed out in wait or cancelled)
            if self.actor_logger and self.numeric_log_level >= 1: # WARNING
                self.actor_logger.print(f"WARNING: Batch future did not complete. Marking {len(batch_for_evaluation_args)} simulations as timeout.")
            for _, args_for_incomplete in batch_for_evaluation_args:
                self.handle_leaf(None, None, args_for_incomplete, timeout=True)
                search_timeouts += 1
            simulations_done_this_search += len(batch_for_evaluation_args)

    if self.verbose:
      # Use logging.info if actor_logger is not guaranteed or for general MCTSBot verbosity
      logging.info("Timeouts for this search: %d", search_timeouts)
    
    self.total_timeouts += search_timeouts
    self.total_num_searches += 1
    
    if self.verbose:
      logging.info(
          "Total MCTS searches: %d, Avg timeouts per search: %.2f",
          self.total_num_searches,
          self.total_timeouts / self.total_num_searches if self.total_num_searches else 0
      )
    return root

  def mcts_search(self, state):
    return self.async_mcts_search(state) 