# Copyright 2019 DeepMind Technologies Limited
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

"""Tabular Q-learning agent."""

import collections
import numpy as np
import os
import pickle
import yaml
import pyspiel

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools


def valuedict():
  # The default factory is called without arguments to produce a new value when
  # a key is not present, in __getitem__ only. This value is added to the dict,
  # so modifying it will modify the dict.
  return collections.defaultdict(float)


class QLearner(rl_agent.AbstractAgent):
  """Tabular Q-Learning agent.

  See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
  """

  def __init__(self,
               player_id,
               num_actions,
               step_size=0.1,
               epsilon_start=0.2,
               epsilon_end=0.2,
               epsilon_decay_duration=1,
               discount_factor=1.0,
               q_init=0.0,
               centralized=False,
               **kwargs):
    """Initialize the Q-Learning agent."""
    self._player_id = player_id
    self._num_actions = num_actions
    self._step_size = step_size
    if 'epsilon_schedule' in kwargs:
        self._epsilon_schedule = kwargs['epsilon_schedule']
    elif epsilon_decay_duration > 1 and epsilon_start != epsilon_end:
        self._epsilon_schedule = rl_tools.EpsilonSchedule(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_duration=epsilon_decay_duration)
    else:
         self._epsilon_schedule = rl_tools.ConstantSchedule(epsilon_start)

    self._epsilon = self._epsilon_schedule.value
    self._discount_factor = discount_factor
    self._centralized = centralized
    self._q_values = collections.defaultdict(lambda: collections.defaultdict(lambda: q_init))
    self._prev_info_state = None
    self._last_loss_value = None
    self._step_counter = 0

    self._hparams = {
        'player_id': player_id,
        'num_actions': num_actions,
        'step_size': step_size,
        'discount_factor': discount_factor,
        'centralized': centralized,
        'q_init': q_init,
        'epsilon_start': self._epsilon_schedule.epsilon_start if hasattr(self._epsilon_schedule, 'epsilon_start') else self._epsilon_schedule.value,
        'epsilon_end': self._epsilon_schedule.epsilon_end if hasattr(self._epsilon_schedule, 'epsilon_end') else self._epsilon_schedule.value,
        'epsilon_decay_duration': self._epsilon_schedule.epsilon_decay_duration if hasattr(self._epsilon_schedule, 'epsilon_decay_duration') else 1,
    }
    self._hparams.update(kwargs)

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    If the agent has not been to `info_state`, a valid random action is chosen.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of actions at `info_state`.
      epsilon: float, prob of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
    greedy_actions = [
        a for a in legal_actions if self._q_values[info_state][a] == greedy_q
    ]
    probs[legal_actions] = epsilon / len(legal_actions)
    probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
    action = np.random.choice(range(self._num_actions), p=probs)
    return action, probs

  def _get_action_probs(self, info_state, legal_actions, epsilon):
    """Returns a selected action and the probabilities of legal actions.

    To be overwritten by subclasses that implement other action selection
    methods.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of actions at `info_state`.
      epsilon: float: current value of the epsilon schedule or 0 in case
        evaluation. QLearner uses it as the exploration parameter in
        epsilon-greedy, but subclasses are free to interpret in different ways
        (e.g. as temperature in softmax).
    """
    return self._epsilon_greedy(info_state, legal_actions, epsilon)

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the Q-values if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    if self._centralized:
      info_state = str(time_step.observations["info_state"])
    else:
      info_state = str(time_step.observations["info_state"][self._player_id])
    legal_actions = time_step.observations["legal_actions"][self._player_id]

    # Prevent undefined errors if this agent never plays until terminal step
    action, probs = None, None

    # Act step: don't act at terminal states.
    if not time_step.last():
      epsilon = 0.0 if is_evaluation else self._epsilon
      action, probs = self._get_action_probs(info_state, legal_actions, epsilon)

    # Learn step: don't learn during evaluation or at first agent steps.
    if self._prev_info_state and not is_evaluation:
      target = time_step.rewards[self._player_id]
      if not time_step.last():  # Q values are zero for terminal.
        target += self._discount_factor * max(
            [self._q_values[info_state][a] for a in legal_actions])

      prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
      self._last_loss_value = target - prev_q_value
      self._q_values[self._prev_info_state][self._prev_action] += (
          self._step_size * self._last_loss_value)

      # Decay epsilon, if necessary.
      self._epsilon = self._epsilon_schedule.step()
      self._step_counter += 1

      if time_step.last():  # prepare for the next episode.
        self._prev_info_state = None
        return

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._prev_info_state = info_state
      self._prev_action = action
    return rl_agent.StepOutput(action=action, probs=probs)

  @property
  def loss(self):
    return self._last_loss_value

  def save(self, path):
    """Saves the Q-learner state and metadata.

    Creates the directory if it doesn't exist. Saves the Q-table,
    the current step counter for the epsilon schedule, and a metadata file.

    Args:
      path: Directory path to save the agent data.
    """
    os.makedirs(path, exist_ok=True)

    # 1. Save Q-table (convert to standard dict for better compatibility)
    q_values_dict = {k: dict(v) for k, v in self._q_values.items()}
    q_table_path = os.path.join(path, "q_table.pkl")
    try:
        with open(q_table_path, 'wb') as f:
            pickle.dump(q_values_dict, f)
    except Exception as e:
        raise IOError(f"Could not save Q-table to {q_table_path}: {e}")

    # 2. Save internal state (step counter for epsilon schedule)
    agent_state_path = os.path.join(path, "agent_state.pkl")
    agent_state = {"step_counter": self._step_counter}
    try:
        with open(agent_state_path, 'wb') as f:
            pickle.dump(agent_state, f)
    except Exception as e:
        raise IOError(f"Could not save agent state to {agent_state_path}: {e}")

    # 3. Save metadata
    metadata = {
        "agent_class": self.__class__.__name__,
        "agent_hparams": self._hparams,
        "serialization_format_version": "1.0",
        "openspiel_version": pyspiel.__version__,
    }
    metadata_path = os.path.join(path, "metadata.yaml")
    try:
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    except Exception as e:
        raise IOError(f"Could not save metadata to {metadata_path}: {e}")

    print(f"QLearner saved successfully to {path}")

  # TODO: Implement restore(path) method
