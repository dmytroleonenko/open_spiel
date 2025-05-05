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

"""Deep Q Network (DQN) agent implemented in TensorFlow.

This is a version modified to handle games like Long Narde where forced pass
turns (empty legal action sets) can occur.
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class DQNLongNarde(rl_agent.AbstractAgent):
  """Deep Q-Network agent modified for Long Narde.

  Handles states with no legal actions (forced pass turns).
  """

  def __init__(self,
               session,
               player_id,
               state_representation_size,
               num_actions,
               hidden_layers_sizes=128,
               replay_buffer_capacity=10000,
               batch_size=128,
               replay_buffer_class=ReplayBuffer,
               learning_rate=0.01,
               update_target_network_every=1000,
               learn_every=10,
               discount_factor=1.0,
               min_buffer_size_to_learn=1000,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_duration=int(1e6),
               optimizer_str="sgd",
               loss_str="mse"):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()

    self.player_id = player_id
    self._session = session
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor

    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration

    # TODO(author6) Allow for optional replay buffer config.
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create required TensorFlow placeholders to perform the Q-network updates.
    self._info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="info_state_ph")
    self._action_ph = tf.placeholder(
        shape=[None], dtype=tf.int32, name="action_ph")
    self._reward_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="reward_ph")
    self._is_final_step_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="is_final_step_ph")
    self._next_info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="next_info_state_ph")
    self._legal_actions_mask_ph = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="legal_actions_mask_ph")

    self._q_network = simple_nets.MLP(state_representation_size,
                                      self._layer_sizes, num_actions)
    self._q_values = self._q_network(self._info_state_ph)

    self._target_q_network = simple_nets.MLP(state_representation_size,
                                             self._layer_sizes, num_actions)
    self._target_q_values = self._target_q_network(self._next_info_state_ph)

    # Stop gradient to prevent updates to the target network while learning
    self._target_q_values = tf.stop_gradient(self._target_q_values)

    self._update_target_network = self._create_target_network_update_op(
        self._q_network, self._target_q_network)

    # Create the loss operations.
    # Sum a large negative constant to illegal action logits before taking the
    # max. This prevents illegal action values from being considered as target.
    illegal_actions = 1 - self._legal_actions_mask_ph
    illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY
    max_next_q = tf.reduce_max(
        tf.math.add(tf.stop_gradient(self._target_q_values), illegal_logits),
        axis=-1)
    target = (
        self._reward_ph +
        (1 - self._is_final_step_ph) * self._discount_factor * max_next_q)
    # Clip the target TD value to prevent large swings
    target = tf.clip_by_value(target, -1.0, 1.0)

    action_indices = tf.stack(
        [tf.range(tf.shape(self._q_values)[0]), self._action_ph], axis=-1)
    predictions = tf.gather_nd(self._q_values, action_indices)

    self._savers = [("q_network", tf.train.Saver(self._q_network.variables)),
                    ("target_q_network",
                     tf.train.Saver(self._target_q_network.variables))]

    if loss_str == "mse":
      loss_class = tf.losses.mean_squared_error
    elif loss_str == "huber":
      loss_class = tf.losses.huber_loss
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    self._loss = tf.reduce_mean(
        loss_class(labels=target, predictions=predictions))

    # --- Optimizer and Gradient Clipping --- #
    if optimizer_str == "adam":
      self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
      self._optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    # Compute gradients
    grads_and_vars = self._optimizer.compute_gradients(self._loss)

    # Clip gradients by value
    clipped_grads_and_vars = []
    for grad, var in grads_and_vars:
        if grad is not None:
            clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
            clipped_grads_and_vars.append((clipped_grad, var))
        else:
            clipped_grads_and_vars.append((grad, var)) # Keep None gradients

    # Apply clipped gradients
    self._learn_step = self._optimizer.apply_gradients(clipped_grads_and_vars)

    self._initialize()

  def get_step_counter(self):
    return self._step_counter

  def step(self, time_step, is_evaluation=False, add_transition_record=True):
    """Returns the action to be taken and updates the Q-network if needed.

    Handles both single TimeStep and list of TimeSteps (for vectorized envs).

    Args:
      time_step: an instance of rl_environment.TimeStep or a list of them.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action, or
      a list of `rl_agent.StepOutput` if the input was a list.
    """

    # --- Handle Batched Input (from Vectorized Env) ---
    if isinstance(time_step, list):
        # Input is a batch of time_steps from multiple environments
        batch_size = len(time_step)
        agent_outputs = []
        info_states_batch = []
        legal_actions_batch = []
        active_envs_indices = [] # Track indices requiring action selection

        # First pass: Prepare batch for network inference
        for i, ts in enumerate(time_step):
            if (not ts.last()) and (ts.is_simultaneous_move() or self.player_id == ts.current_player()):
                info_states_batch.append(ts.observations["info_state"][self.player_id])
                legal_actions_batch.append(ts.observations["legal_actions"][self.player_id])
                active_envs_indices.append(i)
            else:
                 # For terminal states or non-active players, prepare None action later
                 pass

        # Perform batched Q-value computation if any envs are active
        actions_selected = {} # Store selected action for active envs
        probs_selected = {} # Store probs for active envs
        if info_states_batch:
            epsilon = self._get_epsilon(is_evaluation)

            # --- Normalize Batch States --- #
            info_states_batch_np = np.array(info_states_batch)
            # Board (0-47): Divide by 15.0
            info_states_batch_np[:, 0:48] /= 15.0
            # Scores (48-49): Divide by 15.0
            info_states_batch_np[:, 48:50] /= 15.0
            # Turn indicators (50-51): Already 0/1
            # Dice (52-53): Divide by 6.0
            info_states_batch_np[:, 52:54] /= 6.0
            # --- End Normalize Batch States --- #

            # Run network for the batch of active states
            q_values_batch = self._session.run(
                self._q_values, feed_dict={self._info_state_ph: info_states_batch_np}) # Use normalized batch

            # Epsilon-greedy for each active environment
            for i, active_idx in enumerate(active_envs_indices):
                legal_actions = legal_actions_batch[i]
                current_q_values = q_values_batch[i]
                # Need a version of _epsilon_greedy that takes computed q_values
                action, probs = self._epsilon_greedy_from_q_values(
                    legal_actions, current_q_values, epsilon)
                actions_selected[active_idx] = action
                probs_selected[active_idx] = probs

        # Second pass: Construct output list
        for i in range(batch_size):
            if i in actions_selected:
                agent_outputs.append(rl_agent.StepOutput(action=actions_selected[i], probs=probs_selected[i]))
            else:
                # Env was terminal or it wasn't this agent's turn
                agent_outputs.append(rl_agent.StepOutput(action=None, probs=[]))

        # --- Learning and Transition Adding (Handled Outside Batch Loop) ---
        # IMPORTANT: The original step logic handles learning updates based on step_counter.
        # For vectorized envs, the train_agent loop *itself* might handle calling
        # learn() or add_transition() based on the *batch* results. Avoid duplicating
        # learning calls here. We only focus on batch action selection.
        # The add_transition record should likely be handled by the training loop
        # iterating through the time_step and next_time_step lists.

        # Simplified: Just increment step counter once per batch step call if not evaluating
        if not is_evaluation:
            self._step_counter += 1 # Increment once per call, even for batch
            # Learning and target network updates might still be based on this counter
            if self._step_counter % self._learn_every == 0:
                self._last_loss_value = self.learn()
            if self._step_counter % self._update_target_network_every == 0:
                self._session.run(self._update_target_network)

        return agent_outputs # Return list of StepOutput

    # --- Handle Single TimeStep Input (Original Logic) ---
    else:
        # Act step: don't act at terminal info states or if its not our turn.
        if (not time_step.last()) and (
            time_step.is_simultaneous_move() or
            self.player_id == time_step.current_player()):
          info_state = time_step.observations["info_state"][self.player_id]
          legal_actions = time_step.observations["legal_actions"][self.player_id]
          epsilon = self._get_epsilon(is_evaluation)

          # --- Normalize Single State --- #
          info_state_copy = np.array(info_state, dtype=np.float32) # Create float copy
          # Board (0-47): Divide by 15.0
          info_state_copy[0:48] /= 15.0
          # Scores (48-49): Divide by 15.0
          info_state_copy[48:50] /= 15.0
          # Turn indicators (50-51): Already 0/1
          # Dice (52-53): Divide by 6.0
          info_state_copy[52:54] /= 6.0
          # --- End Normalize Single State --- #

          # Compute Q-values for the single state (needed for _epsilon_greedy)
          # Reshape needed for the network
          info_state_vector = np.reshape(info_state_copy, [1, -1]) # Use normalized copy
          self._current_q_values_single = self._session.run(
               self._q_values, feed_dict={self._info_state_ph: info_state_vector})[0]
          # Pass computed Q-values to helper
          action, probs = self._epsilon_greedy_from_q_values(
              legal_actions, self._current_q_values_single, epsilon)
        else:
          action = None
          probs = []

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
          self._step_counter += 1

          if self._step_counter % self._learn_every == 0:
            self._last_loss_value = self.learn()

          if self._step_counter % self._update_target_network_every == 0:
            self._session.run(self._update_target_network)

          # --- Check if transition should be added --- #
          if self._prev_timestep and add_transition_record:
            # Only add transitions if the current step is not FIRST (rewards is not None)
            # and if the previous action was not None (handle forced pass cases)
            if time_step.rewards is not None and self._prev_action is not None:
              self.add_transition(self._prev_timestep, self._prev_action, time_step)
          # --- End check --- #

          if time_step.last():  # prepare for the next episode.
            self._prev_timestep = None
            self._prev_action = None
            # Still return the last computed action/probs
            return rl_agent.StepOutput(action=action, probs=probs)
          else:
            self._prev_timestep = time_step
            self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`. This must not be None.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    # assert prev_action is not None, "add_transition called with prev_action=None" # <-- Comment out this line
    legal_actions = (time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            prev_time_step.observations["info_state"][self.player_id][:]),
        action=prev_action,
        reward=time_step.rewards[self.player_id],
        next_info_state=time_step.observations["info_state"][self.player_id][:],
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask)
    self._replay_buffer.add(transition)

  def _create_target_network_update_op(self, q_network, target_q_network):
    """Create TF ops copying the params of the Q-network to the target network.

    Args:
      q_network: A q-network object that implements provides the `variables`
                 property representing the TF variable list.
      target_q_network: A target q-net object that provides the `variables`
                        property representing the TF variable list.

    Returns:
      A `tf.Operation` that updates the variables of the target.
    """
    self._variables = q_network.variables[:]
    self._target_variables = target_q_network.variables[:]
    assert self._variables
    assert len(self._variables) == len(self._target_variables)
    return tf.group([
        tf.assign(target_v, v)
        for (target_v, v) in zip(self._target_variables, self._variables)
    ])

  # Epsilon greedy function now takes pre-computed q_values
  def _epsilon_greedy_from_q_values(self, legal_actions, q_values, epsilon):
    """Returns epsilon-greedy action from pre-computed q_values.

    Args:
      legal_actions: list of legal actions.
      q_values: numpy array of q_values for all actions.
      epsilon: float, probability of taking a random action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)

    # --- Debug Start ---
    # print(f"[DEBUG Agent] _epsilon_greedy_from_q_values received legal_actions: {legal_actions}")
    # --- Debug End ---

    # Handle forced pass turn first
    if not legal_actions:
      # print("[DEBUG Agent] _epsilon_greedy_from_q_values: No legal actions, returning None.") # Debug
      return None, probs # Return None action, and zero probs

    # Epsilon-greedy choice
    if np.random.rand() < epsilon:
      # Choose random legal action
      action = np.random.choice(legal_actions)
      # Assign uniform probability for exploration step
      probs[legal_actions] = 1.0 / len(legal_actions)
      # print(f"[DEBUG Agent] _epsilon_greedy_from_q_values: Chose random action {action} from {legal_actions}") # Debug
    else:
      # Choose greedy action
      # Use the already computed q_values
      legal_q_values = q_values[legal_actions]
      action = legal_actions[np.argmax(legal_q_values)]
      # Assign full probability to the greedy action
      probs[action] = 1.0
      # print(f"[DEBUG Agent] _epsilon_greedy_from_q_values: Chose greedy action {action} (Q-values for legal: {legal_q_values})") # Debug

    # --- Debug Start ---
    # print(f"[DEBUG Agent] _epsilon_greedy_from_q_values returning action: {action}, probs: {probs}")
    # --- Debug End ---
    return action, probs

  def _get_epsilon(self, is_evaluation, power=1.0):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**power)
    return decayed_epsilon

  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._replay_buffer.sample(self._batch_size)

    # --- Debug Check: Ensure actions in sampled transitions are not None --- #
    for t in transitions:
        if t.action is None:
            # This indicates a potential problem from the step/epsilon_greedy method
            # during forced pass turns or other edge cases.
            # We might skip this transition or handle it specifically.
            # For now, let's raise an error to pinpoint the issue if it happens here.
            raise ValueError(f"Sampled transition contains action=None. Transition: {t}")
    # --- End Debug Check --- #

    info_states = np.array([t.info_state for t in transitions])
    next_info_states = np.array([t.next_info_state for t in transitions])

    # --- Normalize States --- #
    # Board (0-47): Divide by 15.0
    info_states[:, 0:48] /= 15.0
    next_info_states[:, 0:48] /= 15.0
    # Scores (48-49): Divide by 15.0
    info_states[:, 48:50] /= 15.0
    next_info_states[:, 48:50] /= 15.0
    # Turn indicators (50-51): Already 0/1
    # Dice (52-53): Divide by 6.0
    info_states[:, 52:54] /= 6.0
    next_info_states[:, 52:54] /= 6.0
    # --- End Normalize States --- #

    # --- Debug Print Input State Stats --- #
    # print(f"[DEBUG LEARN] info_states shape: {info_states.shape}, min: {np.min(info_states)}, max: {np.max(info_states)}, mean: {np.mean(info_states)}")
    # --- End Debug Print --- #
    actions = np.array([t.action for t in transitions])
    rewards = np.array([t.reward for t in transitions])
    # Reward Clipping is currently disabled
    are_final_steps = np.array([t.is_final_step for t in transitions])
    legal_actions_mask = np.array([t.legal_actions_mask for t in transitions])

    loss, _ = self._session.run(
        [self._loss, self._learn_step],
        feed_dict={
            self._info_state_ph: info_states,
            self._action_ph: actions,
            self._reward_ph: rewards,
            self._is_final_step_ph: are_final_steps,
            self._next_info_state_ph: next_info_states,
            self._legal_actions_mask_ph: legal_actions_mask,
        })
    return loss

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, path_prefix):
    """Saves the q network and the target q-network using a path prefix.

    Args:
      path_prefix: The prefix for the checkpoint files (e.g., '/tmp/chkpt_p0_ep1000').
                   Network names ('q_network', 'target_q_network') will be appended.
    """
    # We assume the directory structure exists (handled by the caller/wrapper)
    try:
      for name, saver in self._savers:
        # Construct the full path for this specific network's checkpoint
        # Note: saver.save() uses the provided path as the prefix
        #       and appends internal checkpoint metadata.
        #       We don't need _full_checkpoint_name or _latest_checkpoint_filename here
        #       if the wrapper manages the overall naming scheme.
        save_path = f"{path_prefix}_{name}"
        returned_path = saver.save(self._session, save_path)
        logging.info(f"Saved {name} for player {self.player_id} to path prefix: {returned_path}")
    except Exception as e:
      logging.error(f"Error saving agent for player {self.player_id} with prefix {path_prefix}: {e}")
      raise # Re-raise the exception to notify the caller

  def has_checkpoint(self, checkpoint_dir): # Keep this potentially, or adapt?
    # This might need adaptation if the wrapper handles checking existence
    # based on the prefix convention.
    for name, _ in self._savers:
      # Check based on the old directory structure? Or adapt to prefix?
      # Let's keep the old logic for now, assuming it might still be used elsewhere,
      # but acknowledge it might mismatch the new save logic if only the wrapper calls save/restore.
      if tf.train.latest_checkpoint(
          self._full_checkpoint_name(checkpoint_dir, name),
          os.path.join(checkpoint_dir,
                       self._latest_checkpoint_filename(name))) is None:
        return False
    return True

  def restore(self, path_prefix):
    """Restores the q network and the target q-network from a path prefix.

    Args:
      path_prefix: The base path prefix used when saving the agent (e.g.,
                   '/tmp/checkpoints/agent_p0_ep1000'). The savers are expected
                   to restore their respective variables from checkpoints matching
                   this prefix, potentially with suffixes like '_q_network'.
    """
    logging.info(f"Attempting to restore agent {self.player_id} from prefix: {path_prefix}")
    # Expand user path just in case '~' was used
    expanded_prefix = os.path.expanduser(path_prefix)
    logging.info(f"Expanded path prefix: {expanded_prefix}")

    restored_q = False
    restored_target = False
    try:
        for name, saver in self._savers:
            # Construct the specific checkpoint prefix TF saver.save() uses
            specific_checkpoint_prefix = f"{expanded_prefix}_{name}"
            logging.info(f"  Attempting restore for '{name}' using checkpoint prefix: {specific_checkpoint_prefix}")

            try:
                # Try restoring directly using the specific prefix
                saver.restore(self._session, specific_checkpoint_prefix)
                logging.info(f"    Successfully restored {name} from {specific_checkpoint_prefix}")
                if name == "q_network": restored_q = True
                if name == "target_q_network": restored_target = True
            except Exception as e:
                 # Log the specific error during restore attempt
                 logging.error(f"    Failed to restore {name} using prefix {specific_checkpoint_prefix}: {e}")
                 # Note: Could add fallback logic using tf.train.latest_checkpoint here if needed

    except Exception as e:
        logging.error(f"Unexpected error during restore loop for agent {self.player_id}: {e}")

    if not restored_q or not restored_target:
        logging.error(f"Restore incomplete for agent {self.player_id}. Q Network restored: {restored_q}, Target Network restored: {restored_target}. Agent may have uninitialized weights.")

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def info_state_ph(self):
    return self._info_state_ph

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._step_counter

  def _initialize(self):
    initialization_weights = tf.group(
        *[var.initializer for var in self._variables])
    initialization_target_weights = tf.group(
        *[var.initializer for var in self._target_variables])
    initialization_opt = tf.group(
        *[var.initializer for var in self._optimizer.variables()])

    self._session.run(
        tf.group(*[
            initialization_weights, initialization_target_weights,
            initialization_opt,
        ]))

  def get_weights(self):
    variables = [self._session.run(self._q_network.variables)]
    variables.append(self._session.run(self._target_q_network.variables))
    return variables

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates it with noise.

    Args:
      sigma: gaussian dropout variance term : Multiplicative noise following
        (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
        model weight. sigma=0 means no perturbation.
      copy_weights: Boolean determining whether to copy model weights (True) or
        just model hyperparameters.

    Returns:
      Perturbated copy of the model.
    """
    _ = self._kwargs.pop("self", None)
    copied_object = DQNLongNarde(**self._kwargs)

    q_network = getattr(copied_object, "_q_network")
    target_q_network = getattr(copied_object, "_target_q_network")

    if copy_weights:
      copy_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(q_network.variables, self._q_network.variables)
      ])
      self._session.run(copy_weights)

      copy_target_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(target_q_network.variables,
                            self._target_q_network.variables)
      ])
      self._session.run(copy_target_weights)
    return copied_object

  # Add step_batch method for compatibility with wrappers/vectorized envs
  def step_batch(self, time_step_list, is_evaluation=False):
    """Processes a batch of time steps by calling the main step method.

    Args:
      time_step_list: A list of TimeStep objects.
      is_evaluation: bool, whether this is training or evaluation.

    Returns:
      A list of StepOutput objects.
    """
    # The main step method already handles list input
    return self.step(time_step_list, is_evaluation=is_evaluation)

  # Add sync_target_network method
  def sync_target_network(self):
    """Explicitly runs the TF operation to update the target network."""
    self._session.run(self._update_target_network)
