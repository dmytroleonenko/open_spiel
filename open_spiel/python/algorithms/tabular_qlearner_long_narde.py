import collections
import numpy as np
import os
import pickle
import yaml
import pyspiel

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools

def valuedict():
    return collections.defaultdict(float)

class QLearnerLongNarde(rl_agent.AbstractAgent):
    """Tabular Q-Learning agent for Long Narde and similar games (handles forced pass turns)."""

    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.1,
                 epsilon_start=1.0,  # Default to high exploration
                 discount_factor=1.0,
                 q_init=0.0,
                 centralized=False,
                 **kwargs):
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size = step_size
        # Always use ConstantSchedule for now
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
            'epsilon_start': epsilon_start, # Store the constant epsilon used
        }
        self._hparams.update(kwargs) # Allow passing other hyperparameters

    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        if not legal_actions:
            # Forced pass: no legal actions available
            return None, np.zeros(self._num_actions)
        probs = np.zeros(self._num_actions)
        greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
        greedy_actions = [
            a for a in legal_actions if self._q_values[info_state][a] == greedy_q
        ]
        # Ensure probabilities sum to 1, handle potential division by zero if legal_actions is empty (already guarded)
        num_legal = len(legal_actions)
        num_greedy = len(greedy_actions)
        if num_legal > 0:
            non_greedy_prob = epsilon / num_legal
            greedy_prob_boost = (1 - epsilon) / num_greedy if num_greedy > 0 else 0
            for action_id in legal_actions:
                probs[action_id] = non_greedy_prob
                if action_id in greedy_actions:
                    probs[action_id] += greedy_prob_boost

            # Normalize probabilities due to potential floating point inaccuracies
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                 probs /= prob_sum
            else: # Fallback if all probabilities somehow became zero (e.g., epsilon=1 and no greedy actions?)
                 probs[legal_actions] = 1.0 / num_legal

        # Check for NaN values and handle them - this shouldn't happen with the logic above, but as a safeguard
        if np.isnan(probs).any():
            # Fallback for safety, should not happen with current logic
            probs = np.zeros(self._num_actions)
            if num_legal > 0:
                probs[legal_actions] = 1.0 / num_legal
            else:
                return None, np.zeros(self._num_actions)

        # Sample action using the calculated probabilities
        try:
             chosen_action = np.random.choice(range(self._num_actions), p=probs)
        except ValueError as e:
             # Fallback: choose a random legal action if possible
             if legal_actions:
                 chosen_action = np.random.choice(legal_actions)
             else:
                 return None, np.zeros(self._num_actions)

        return chosen_action, probs


    def _get_action_probs(self, info_state, legal_actions, epsilon):
        # Initialize Q-values for the state if not seen before
        if info_state not in self._q_values:
            for action in range(self._num_actions):
                self._q_values[info_state][action] = self._hparams.get('q_init', 0.0)

        return self._epsilon_greedy(info_state, legal_actions, epsilon)

    def step(self, time_step, is_evaluation=False):
        if self._centralized:
            info_state = str(time_step.observations["info_state"])
        else:
            info_state = str(time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        action, probs = None, None

        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            action, probs = self._get_action_probs(info_state, legal_actions, epsilon)

        if self._prev_info_state and not is_evaluation:
            reward = time_step.rewards[self._player_id] if time_step.rewards is not None else 0.0
            target = reward
            if not time_step.last():
                # Ensure current info_state Q-values are initialized before calculating max
                if info_state not in self._q_values:
                     self._get_action_probs(info_state, legal_actions, 0) # Initialize Q-values for the new state

                if legal_actions: # Only add future Q value if legal actions exist
                    q_values_for_legal_actions = [self._q_values[info_state][a] for a in legal_actions]
                    if q_values_for_legal_actions: # Check if list is not empty
                         max_q = max(q_values_for_legal_actions)
                         target += self._discount_factor * max_q

                else:
                     # No legal actions for next state, target is just reward
                     target = reward

            # Ensure prev_info_state exists before accessing Q-value
            if self._prev_info_state in self._q_values and self._prev_action is not None:
                prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
                self._last_loss_value = target - prev_q_value
                self._q_values[self._prev_info_state][self._prev_action] += (
                    self._step_size * self._last_loss_value)
            else:
                 self._last_loss_value = None # Indicate no loss calculated

            self._step_counter += 1

            if time_step.last():
                self._prev_info_state = None
                # Return last calculated action/probs for consistency
                return rl_agent.StepOutput(action=action, probs=probs)

        if not is_evaluation:
             # Store current state/action for next step's update
             self._prev_info_state = info_state
             self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        return self._last_loss_value

    def save(self, path):
        """Returns the agent's state components for saving.

        Args:
            path: The directory path (unused, for compatibility).

        Returns:
            A tuple containing:
              - q_table_dict: The Q-table as a serializable dictionary.
              - agent_state_dict: A dictionary with other state like step_counter.
        """
        q_table_dict = {k: dict(v) for k, v in self._q_values.items()}
        agent_state_dict = {
            "step_counter": self._step_counter,
            # Add other relevant state variables here if needed
        }
        return q_table_dict, agent_state_dict

    # TODO: Implement restore(path) method to match this new save logic

    # TODO: Implement restore(path) method 