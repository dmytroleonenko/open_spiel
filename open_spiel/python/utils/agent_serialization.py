"""Utilities for saving and loading OpenSpiel RL agents consistently."""

import os
import pickle
import yaml
import importlib
import logging
import sys
import tensorflow.compat.v1 as tf # type: ignore # Needed for DQN restore check

from open_spiel.python import rl_agent
import pyspiel

# Define standard filenames
METADATA_FILENAME = "metadata.yaml"
QLEARNER_QTABLE_FILENAME = "q_table.pkl"
QLEARNER_STATE_FILENAME = "agent_state.pkl" # For step_counter, etc.
DQN_CHECKPOINT_NAME_Q = "q_network" # From DQN internal naming
DQN_CHECKPOINT_NAME_TARGET = "target_q_network" # From DQN internal naming
DQN_STATE_FILENAME = "agent_state.pkl" # For step_counter, etc.
# Add other agent-specific filenames as needed

class SerializableAgentWrapper(rl_agent.AbstractAgent):
    """A wrapper around an OpenSpiel agent to handle saving and metadata."""

    def __init__(self, agent_class_path: str, player_id: int, env_specs: dict, agent_hparams: dict):
        """Initializes the wrapper and the underlying agent.

        Args:
            agent_class_path: Full import path to the agent class (e.g., 'open_spiel.python.algorithms.tabular_qlearner.QLearner').
            player_id: The ID of the player this agent controls.
            env_specs: A dictionary containing environment specifications like
                       'num_actions', 'observation_spec', 'action_spec'. Needed for agent init.
            agent_hparams: A dictionary of hyperparameters for the underlying agent.
        """
        self._agent_class_path = agent_class_path
        self._player_id = player_id
        self._env_specs = env_specs # Store env specs if needed later
        self._agent_hparams = agent_hparams.copy() # Store hparams

        # Dynamically import and instantiate the underlying agent
        try:
            module_path, class_name = agent_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Could not import agent class '{agent_class_path}': {e}")

        # Prepare arguments for the underlying agent's __init__
        # Common args: player_id, num_actions, possibly session for DQN
        init_args = {'player_id': player_id}
        if 'num_actions' in env_specs:
            init_args['num_actions'] = env_specs['num_actions']

        # Special handling for DQN's TF session and state size
        if class_name == 'DQN':
            # DQN requires a TF session and state_representation_size
            if 'session' not in agent_hparams:
                 # Create a default session if none provided (may need configuration)
                 logging.warning("DQN requires a tf.Session. Creating a default one.")
                 # Ensure TF1 behavior if needed
                 tf.disable_eager_execution()
                 init_args['session'] = tf.Session()
            else:
                 init_args['session'] = agent_hparams['session'] # Use provided session

            # Note: DQN agent itself doesn't have a 'device' param in its __init__.
            # TF placement is typically handled via context managers (e.g., with tf.device(...)).
            # The wrapper doesn't manage this; it should be handled where the session/graph is run.
            if 'device' in agent_hparams:
                logging.info(f"Device '{agent_hparams['device']}' specified, but DQN TF placement is handled externally.")
                # init_args['device'] = agent_hparams['device'] # Not passed to DQN init

            if 'state_representation_size' not in agent_hparams:
                 # Attempt to infer from env_specs
                 obs_spec = env_specs.get('observation_spec', {})
                 info_state_shape = obs_spec.get('info_state', [])
                 if len(info_state_shape) == 1:
                      init_args['state_representation_size'] = info_state_shape[0]
                 else:
                      raise ValueError("Cannot infer 'state_representation_size' for DQN from env_specs.")
            else:
                 init_args['state_representation_size'] = agent_hparams['state_representation_size']

        # Special handling for PPO's input shape and device
        elif class_name == 'PPO':
             # PPO (pytorch version) expects input_shape, num_actions, num_players, device
             # Infer input_shape from env_specs
             if 'input_shape' not in agent_hparams:
                  obs_spec = env_specs.get('observation_spec', {})
                  info_state_shape = obs_spec.get('info_state', [])
                  if info_state_shape:
                       init_args['input_shape'] = info_state_shape
                       logging.info(f"Inferred PPO input_shape: {info_state_shape}")
                  else:
                       raise ValueError("Cannot infer 'input_shape' for PPO from env_specs.")
             else:
                  init_args['input_shape'] = agent_hparams['input_shape']

             # Ensure num_players is present (it should be in env_specs but double-check)
             if 'num_players' not in agent_hparams:
                  if 'num_players' in env_specs:
                       init_args['num_players'] = env_specs['num_players']
                  else:
                       # Try to infer from wrapper's env_specs (though it might not be directly available)
                       # This part might need refinement based on how num_players is passed around
                       logging.warning("Cannot infer 'num_players' for PPO. Defaulting might cause issues.")
                       # Or raise ValueError("Cannot infer 'num_players' for PPO from env_specs.")
             else:
                   init_args['num_players'] = agent_hparams['num_players']

             # Add device if specified (Task 17)
             if 'device' in agent_hparams:
                  init_args['device'] = agent_hparams['device']
                  logging.info(f"Passing device='{agent_hparams['device']}' to PPO.")
             else:
                  logging.warning("No device specified for PPO, using default.")
             # Remove device from hparams passed directly if it's handled by init_args
             # agent_hparams.pop('device', None)
             # pass # Placeholder removed


        # Combine inferred/required args with provided hparams
        # Provided hparams take precedence
        init_args.update(agent_hparams)


        try:
            self.agent = agent_class(**init_args)
            # Initialize TF variables if it's a DQN agent with a new session
            if class_name == 'DQN' and 'session' not in agent_hparams:
                 logging.info("Initializing TF variables for newly created DQN session.")
                 self.agent._session.run(tf.global_variables_initializer())

        except Exception as e:
            raise ValueError(f"Could not instantiate agent {class_name} with args {init_args}: {e}")

        # Store cleaned hparams (without session object for serialization)
        self._serializable_hparams = {k: v for k, v in init_args.items() if not isinstance(v, tf.compat.v1.Session)}
        # Remove env_specs if it was temporarily added
        self._serializable_hparams.pop('env_specs', None)


        # Explicitly set player_id on the wrapper as well
        super().__init__(player_id=player_id, num_actions=env_specs.get('num_actions', 0))


    def step(self, time_step, is_evaluation=False):
        """Delegates the step call to the underlying agent.

        Handles both single TimeStep and list/batch of TimeSteps.
        """
        # Check if input is a list (indicating batch from vector env)
        is_batch = isinstance(time_step, list)

        if not is_batch:
            # Standard single TimeStep handling
            return self.agent.step(time_step, is_evaluation=is_evaluation)
        else:
            # Batch handling (Task 18d)
            agent_class_name = self.agent.__class__.__name__
            if hasattr(self.agent, 'step_batch'):
                # Use explicit batch stepping if available (e.g., for PPO)
                return self.agent.step_batch(time_step, is_evaluation=is_evaluation)
            elif agent_class_name in ['QLearner', 'DQN']:
                # Fallback for QLearner/DQN: Iterate, call single step, return list of outputs.
                # Warning: Learning logic (buffer add/update) happens per-step here,
                # which might be inefficient or incorrect for some vectorized RL patterns.
                # This primarily aims to get actions for env.step().
                batch_outputs = []
                for ts in time_step:
                    agent_output = self.agent.step(ts, is_evaluation=is_evaluation)
                    batch_outputs.append(agent_output)
                return batch_outputs
            else:
                # No batch support or fallback for this agent type
                raise NotImplementedError(
                    f"Agent type {agent_class_name} does not support batch stepping "
                    f"via step_batch(), and no fallback is implemented."
                )

    # Delegate other necessary AbstractAgent methods/properties
    @property
    def loss(self):
        if hasattr(self.agent, 'loss'):
            return self.agent.loss
        return None # Or raise AttributeError?

    # ... potentially delegate other methods like add_transition if needed ...

    def save(self, path):
        """Saves the underlying agent's state and metadata."""
        os.makedirs(path, exist_ok=True)
        agent_class_name = self.agent.__class__.__name__
        print(f"Saving agent of type {agent_class_name} using wrapper...")

        # --- Agent-Specific State Saving ---
        try:
            if agent_class_name == 'QLearner':
                # 1. Save Q-table
                q_values_dict = {k: dict(v) for k, v in self.agent._q_values.items()}
                q_table_path = os.path.join(path, QLEARNER_QTABLE_FILENAME)
                with open(q_table_path, 'wb') as f:
                    pickle.dump(q_values_dict, f)

                # 2. Save internal state (step counter)
                agent_state = {"step_counter": self.agent._step_counter}
                state_path = os.path.join(path, QLEARNER_STATE_FILENAME)
                with open(state_path, 'wb') as f:
                    pickle.dump(agent_state, f)

            elif agent_class_name == 'DQN':
                 # 1. Use DQN's built-in TF saver
                 self.agent.save(path) # Saves q_network and target_q_network

                 # 2. Save additional state (step counter)
                 agent_state = {"step_counter": self.agent._step_counter}
                 state_path = os.path.join(path, DQN_STATE_FILENAME)
                 with open(state_path, 'wb') as f:
                     pickle.dump(agent_state, f)

            # elif agent_class_name == 'PPO':
            #     # TODO: Implement PPO saving logic (likely using torch.save)
            #     print(f"Warning: Saving not fully implemented for {agent_class_name}")
            #     pass

            else:
                # Fallback/Warning for unsupported agents
                print(f"Warning: No specific save logic implemented for agent type {agent_class_name}. Only metadata will be saved.", file=sys.stderr)

        except Exception as e:
            raise IOError(f"Error saving agent-specific state for {agent_class_name} to {path}: {e}")

        # --- Metadata Saving (Common to all) ---
        metadata = {
            "agent_class_path": self._agent_class_path,
            "agent_hparams": self._serializable_hparams, # Use cleaned hparams
            "serialization_format_version": "1.0",
            "openspiel_version": pyspiel.__version__,
            "env_specs": self._env_specs # Include env specs used at creation
        }
        metadata_path = os.path.join(path, METADATA_FILENAME)
        try:
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Could not save metadata to {metadata_path}: {e}")

        print(f"Agent saved successfully to {path}")

    # Restore is handled by the load_agent function, not the wrapper itself


def load_agent(path: str, env=None):
    """Loads an agent from a specified path using metadata.

    Args:
        path: The directory containing the saved agent data (metadata.yaml, etc.).
        env: Optional RL environment. If provided, used for validation and potentially
             for agents that need env specs during init/restore (currently unused here).

    Returns:
        The loaded and restored agent instance.
    """
    metadata_path = os.path.join(path, METADATA_FILENAME)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 1. Load Metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        raise IOError(f"Could not load metadata from {metadata_path}: {e}")

    agent_class_path = metadata.get("agent_class_path")
    agent_hparams = metadata.get("agent_hparams", {})
    # env_specs_from_meta = metadata.get("env_specs", {}) # Load if needed

    if not agent_class_path:
        raise ValueError("Metadata missing 'agent_class_path'.")

    print(f"Loading agent from path: {path}")
    print(f"  Agent class path: {agent_class_path}")
    print(f"  Agent hparams: {agent_hparams}")

    # 2. Dynamically import and instantiate the agent class
    try:
        module_path, class_name = agent_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f"Could not import agent class '{agent_class_path}': {e}")

    # Prepare init arguments - might need env_specs if not in hparams
    init_args = agent_hparams.copy()
    # Ensure essential args like player_id are present
    if 'player_id' not in init_args:
         raise ValueError("Metadata's agent_hparams missing 'player_id'.")


    # Special handling for DQN session
    if class_name == 'DQN':
        # DQN needs a session, create one if not somehow passed via hparams (unlikely)
        # The session state is implicitly restored by saver.restore()
        if 'session' not in init_args:
             tf.disable_eager_execution()
             init_args['session'] = tf.Session()
             # Need to initialize variables *before* restoring
             # This is tricky, maybe restore should happen *after* init?
             # Let's try initializing first.
             # tf.global_variables_initializer().run(session=init_args['session'])


    # Instantiate the agent
    try:
        agent = agent_class(**init_args)
        # For DQN, run initializer *after* graph construction in __init__
        if class_name == 'DQN':
             logging.info("Initializing TF variables before restoring DQN checkpoint.")
             agent._session.run(tf.global_variables_initializer())

    except Exception as e:
        raise ValueError(f"Could not instantiate agent {class_name} with hparams {init_args}: {e}")

    print(f"Agent {class_name} instantiated.")

    # 3. Restore Agent State
    print("Restoring agent state...")
    try:
        if class_name == 'QLearner':
            # Load Q-table
            q_table_path = os.path.join(path, QLEARNER_QTABLE_FILENAME)
            if os.path.exists(q_table_path):
                with open(q_table_path, 'rb') as f:
                    q_values_dict = pickle.load(f)
                # Convert back to defaultdict
                agent._q_values.clear()
                for k, v_dict in q_values_dict.items():
                    for k2, v2 in v_dict.items():
                        agent._q_values[k][k2] = v2
                print("  Restored Q-table.")
            else:
                print(f"  Warning: Q-table file not found: {q_table_path}")

            # Load internal state
            state_path = os.path.join(path, QLEARNER_STATE_FILENAME)
            if os.path.exists(state_path):
                 with open(state_path, 'rb') as f:
                      agent_state = pickle.load(f)
                 agent._step_counter = agent_state.get("step_counter", 0)
                 # Restore epsilon schedule state if needed (requires schedule knowledge)
                 # agent._epsilon = agent._epsilon_schedule.value_from_step(agent._step_counter)
                 print(f"  Restored step_counter: {agent._step_counter}")
            else:
                 print(f"  Warning: Agent state file not found: {state_path}")


        elif class_name == 'DQN':
            # Use DQN's built-in restore method for TF variables
            agent.restore(path) # This handles q_network and target_q_network

            # Load additional pickled state
            state_path = os.path.join(path, DQN_STATE_FILENAME)
            if os.path.exists(state_path):
                 with open(state_path, 'rb') as f:
                      agent_state = pickle.load(f)
                 agent._step_counter = agent_state.get("step_counter", 0)
                 print(f"  Restored step_counter: {agent._step_counter}")
            else:
                 print(f"  Warning: Agent state file not found: {state_path}")


        # elif class_name == 'PPO':
        #     # TODO: Implement PPO loading (likely using torch.load)
        #     print(f"Warning: Loading not fully implemented for {agent_class_name}")
        #     pass

        else:
            print(f"Warning: No specific restore logic implemented for agent type {class_name}.", file=sys.stderr)

    except Exception as e:
        raise IOError(f"Error restoring agent-specific state for {class_name} from {path}: {e}")

    print(f"Agent loaded and restored successfully from {path}")
    return agent


# Example Usage section removed due to linting issues during initial file creation.
# The core load_agent function needs context parameters (player_id, num_actions) added
# before it can be fully implemented and tested. 