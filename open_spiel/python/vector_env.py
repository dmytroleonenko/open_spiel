# Copyright 2022 DeepMind Technologies Limited
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
"""A vectorized RL Environment."""


class SyncVectorEnv(object):
  """A vectorized RL Environment.

  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

  def __init__(self, envs):
    if not isinstance(envs, list):
      raise ValueError(
          "Need to call this with a list of rl_environment.Environment objects")
    self.envs = envs
    self.num_envs = len(envs)

  def __len__(self):
    return len(self.envs)

  def observation_spec(self):
    return self.envs[0].observation_spec()

  @property
  def num_players(self):
    return self.envs[0].num_players

  def step(self, step_outputs):
    """Steps the underlying environments.

    Args:
      step_outputs: A list containing the outputs for each environment. This
        should be a list of actions, one per environment.

    Returns:
      A list of TimeStep objects, one from each environment.
    """
    # Verify input is a list of actions matching the number of envs
    assert len(step_outputs) == self.num_envs

    time_steps = []
    for i in range(self.num_envs):
      action_for_env_i = step_outputs[i]
      # If the action is None (e.g., env is terminal or not this player's turn),
      # don't step the environment, just get the current TimeStep.
      if action_for_env_i is None:
          time_steps.append(self.envs[i].get_time_step())
      else:
          # Otherwise, step the environment with the provided action.
          # Each internal env expects a list of actions.
          action_list_for_env_i = [action_for_env_i]
          time_steps.append(self.envs[i].step(action_list_for_env_i))
    return time_steps

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(len(self.envs))]

    time_steps = [
        self.envs[i].reset()
        if envs_to_reset[i] else self.envs[i].get_time_step()
        for i in range(len(self.envs))
    ]
    return time_steps

  def close(self):
    # Implementation of close method
    pass
