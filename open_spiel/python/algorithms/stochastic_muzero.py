import collections
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from open_spiel.python import rl_agent
from open_spiel.python.algorithms.stochastic_muzero_config import StochasticMuZeroConfig, KnownBounds, Action, Player, Outcome
from open_spiel.python.algorithms.stochastic_muzero_nets import StochasticMuZeroNetwork
from open_spiel.python.utils.replay_buffer import ReplayBuffer

MAXIMUM_FLOAT_VALUE = float('inf')

# Helper function to map dice roll (1-6, 1-6) to index (0-35)
def map_dice_to_index(d1: int, d2: int) -> int:
    """Maps a dice roll (d1, d2) from 1-6 to a unique index 0-35."""
    if not (1 <= d1 <= 6 and 1 <= d2 <= 6):
        raise ValueError(f"Invalid dice roll: ({d1}, {d2})")
    # Ensure canonical order (smaller die first) if needed, or keep distinct
    # Keep distinct order for now: (1, 2) is different from (2, 1)
    return (d1 - 1) * 6 + (d2 - 1)

# --- Rotation Helpers (for Long Narde / Symmetric Board Games) ---

def rotate_observation_long_narde(observation: np.ndarray, player_id: int) -> np.ndarray:
    """Rotates the observation vector to a canonical (Player 0/White) perspective.

    Assumes observation structure:
    - obs[0:24]: White checkers on points 0-23
    - obs[24:48]: Black checkers on points 0-23
    - obs[48]: White borne-off count
    - obs[49]: Black borne-off count
    - obs[50]: Player to play (0 or 1)
    - obs[51:]: Other info (dice, etc.) - copied directly.

    NOTE: This structure MUST be verified against the actual game observation spec.
    """
    if player_id == 0:
        # Already in canonical perspective
        # Ensure player_to_play is 0 if it's part of the agent's input
        # If obs[50] is part of the NN input, we might still want to ensure it's 0.
        # However, let's assume the NN handles the player input separately or it's not used directly.
        # For now, just return the original if player is 0.
        return observation

    if player_id == 1:
        rotated_obs = np.zeros_like(observation)
        obs_len = len(observation)

        # Rotate points (assuming indices 0-23 White, 24-47 Black)
        for p in range(24):
            rotated_p = (p + 12) % 24
            # Canonical White (player 0) checkers = Original Black checkers at rotated pos
            rotated_obs[p] = observation[24 + rotated_p]
            # Canonical Black (player 1) checkers = Original White checkers at rotated pos
            rotated_obs[24 + p] = observation[rotated_p]

        # Rotate borne-off counts (assuming indices 48 White, 49 Black)
        rotated_obs[48] = observation[49] # Canonical White borne-off = Original Black borne-off
        rotated_obs[49] = observation[48] # Canonical Black borne-off = Original White borne-off

        # Set player to canonical player 0
        rotated_obs[50] = 0

        # Copy remaining info (dice, etc.) - Adjust index 51 if spec differs
        if obs_len > 51:
            rotated_obs[51:] = observation[51:]

        return rotated_obs
    else:
        raise ValueError(f"Invalid player_id for rotation: {player_id}")

# Placeholder for action rotation - requires specific action encoding knowledge
def rotate_action_long_narde(action: Action, player_id: int) -> Action:
    """Rotates an action to/from the canonical perspective.
    Placeholder - Implementation depends heavily on action representation.
    """
    # TODO: Implement based on how actions (e.g., integer indices)
    # map to (from_pos, to_pos) tuples.
    # Example logic:
    # if player_id == 1:
    #     from_pos, to_pos = map_action_index_to_tuple(action)
    #     rotated_from = (from_pos + 12) % 24 # Adjust for head/special cases
    #     rotated_to = (to_pos + 12) % 24
    #     return map_tuple_to_action_index(rotated_from, rotated_to)
    # else: # player_id == 0, rotating back
    #     from_canonical, to_canonical = map_action_index_to_tuple(action)
    #     original_from = (from_canonical + 12) % 24 # Adjust for head
    #     original_to = (to_canonical + 12) % 24
    #     return map_tuple_to_action_index(original_from, original_to)
    return action # No-op for now

# Define Transition structure (similar to DQN)
Transition = collections.namedtuple(
    "Transition",
    "observation action reward next_observation is_final_step player_id "
    "legal_actions_mask policy value") # Add policy/value from MCTS

# --- MCTS Helper Classes (from pseudocode) ---

class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

# An object that holds an action or a chance outcome.
ActionOrOutcome = Any # Union[Action, Outcome]

class ActionOutcomeHistory:
    """Simple history container used inside the search.

    Only used to keep track of the actions and chance outcomes executed.
    """
    def __init__(self, initial_player: Player, history: Optional[List[ActionOrOutcome]] = None):
        self.initial_player = initial_player
        self.history = list(history or [])
        # TODO: Implement proper player tracking based on game rules
        self._player_map = {0: 1, 1: 0} # Simple alternating players

    def clone(self):
        return ActionOutcomeHistory(self.initial_player, self.history)

    def add_action_or_outcome(self, action_or_outcome: ActionOrOutcome):
        self.history.append(action_or_outcome)

    def last_action_or_outcome(self) -> ActionOrOutcome:
        return self.history[-1]

    def to_play(self) -> Player:
        """Returns the next player to play based on the history."""
        # This needs to be adapted based on the specific game's turn structure
        # For Long Narde/Backgammon, turns alternate unless doubles are rolled.
        # For now, assume simple alternation after an action.
        # Chance outcomes don't change the player.
        # Needs refinement!
        current_player = self.initial_player
        action_count = sum(1 for item in self.history if not isinstance(item, Outcome))
        if action_count % 2 == 1:
            return self._player_map[self.initial_player]
        return self.initial_player

class Node:
    """A Node in the MCTS search tree."""
    def __init__(self, prior: float, is_chance: bool = False):
        self.visit_count = 0
        self.to_play = -1 # Player who makes the decision *leading* to this state
        self.prior = prior # Policy prior for actions (if decision node), or chance prob (if chance node)
        self.value_sum = 0.0
        self.children: Dict[ActionOrOutcome, Node] = {}
        self.latent_state: Optional[torch.Tensor] = None # Latent state 's'
        self.afterstate: Optional[torch.Tensor] = None # Afterstate 'as'
        self.is_chance = is_chance # True if this node represents a state *after* a chance event
        self.reward = 0.0 # Immediate reward received upon reaching this state

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

# --- Stochastic MuZero Agent --- #

class StochasticMuZero(rl_agent.AbstractAgent):
    """Stochastic MuZero Agent implementation in PyTorch."""

    def __init__(self, player_id: int, config: StochasticMuZeroConfig):
        """Initialize the Stochastic MuZero agent."""
        super().__init__(player_id=player_id)
        self.config = config
        self._device = torch.device(config.device)

        # Instantiate the network
        if config.network_factory is None:
            raise ValueError("Network factory must be provided in the config.")
        self.network = config.network_factory().to(self._device)
        self.network.train() # Start in training mode

        # Instantiate the optimizer
        if config.optimizer_str == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer_str == "sgd":
             self.optimizer = torch.optim.SGD(
                self.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer_str}")

        # Replay buffer
        # TODO: Adapt ReplayBuffer or implement a new one for trajectories/prioritization
        # Ensure the buffer stores trajectories (lists of step_data dicts)
        self._replay_buffer = config.replay_buffer_class(config.replay_buffer_capacity)
        self._min_buffer_size_to_learn = config.min_buffer_size_to_learn

        # Step counters and tracking
        self._step_counter = 0
        self._learn_step_counter = 0
        self._last_loss_value = None
        self._current_mcts_root: Optional[Node] = None # Store root for stats/next step
        self._current_game_trajectory = [] # Store sequence of step data dicts
        self.is_evaluation = False # Added flag

    def _observation_to_tensor(self, observation_list: List[float]) -> torch.Tensor:
        """Converts observation list to a PyTorch tensor on the correct device."""
        obs_np = np.array(observation_list, dtype=np.float32)
        # Reshape for batch dimension if needed (e.g., [1, obs_size])
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self._device)
        return obs_tensor

    # --- Trajectory Management --- #
    def store_step_data(self, time_step, action, agent_output):
        """Stores the data for a single step *before* the environment steps.
        Rotates observation and action if player is 1.
        """
        if self.is_evaluation:
            return

        original_observation = time_step.observations["info_state"][self.player_id]
        original_player = self.player_id
        original_legal_mask = time_step.observations["legal_actions_mask"][original_player]

        # Rotate observation and action for canonical perspective if player 1
        if original_player == 1:
            canonical_observation_np = rotate_observation_long_narde(np.array(original_observation), 1)
            canonical_observation = canonical_observation_np.tolist()
            # TODO: Rotate action and legal_mask (requires action mapping)
            canonical_action = rotate_action_long_narde(action, 1) # Placeholder
            canonical_legal_mask = original_legal_mask # Placeholder - needs rotation
        else:
            canonical_observation = original_observation
            canonical_action = action
            canonical_legal_mask = original_legal_mask

        mcts_policy = {}
        mcts_value = 0.0
        if self._current_mcts_root and not self._current_mcts_root.is_chance:
            # Policy/Value from MCTS are already from the canonical perspective
            mcts_policy = {a: node.visit_count for a, node in self._current_mcts_root.children.items() if not node.is_chance}
            mcts_value = self._current_mcts_root.value()

        step_data = {
            "observation": canonical_observation, # Store canonical observation
            "action": canonical_action,         # Store canonical action taken
            "reward": 0.0, # Placeholder, will be filled by next step's actual reward
            "policy": mcts_policy, # MCTS policy over canonical actions
            "value": mcts_value, # MCTS value from canonical state
            "legal_mask": canonical_legal_mask, # Canonical legal mask (placeholder)
            "player": 0, # Always store as player 0 (canonical)
            "is_final": False, # Placeholder
            "discount": self.config.discount, # Placeholder
            "chance_outcome_index": -1 # Placeholder for c_{k+1}
        }
        self._current_game_trajectory.append(step_data)

    def add_chance_outcome_to_trajectory(self, chance_outcome_index: int):
        """Adds the chance outcome that occurred *after* the last action.

        NOTE: For games like Long Narde with half-moves, this index represents the
        single dice roll governing the sequence of half-moves taken *after* the
        state recorded in this step data. It's stored here for association during learning,
        although the direct chance prediction loss isn't used.
        """
        if self.is_evaluation or not self._current_game_trajectory:
            return
        # Add the chance outcome index to the *most recent* step data added
        # The chance outcome itself is not rotated.
        self._current_game_trajectory[-1]["chance_outcome_index"] = chance_outcome_index

    def finalize_trajectory(self, final_time_step):
        """Updates final step info and adds trajectory to replay buffer."""
        if self.is_evaluation or not self._current_game_trajectory:
            self._current_game_trajectory = [] # Clear incomplete trajectory
            return

        # Update the last step's reward, is_final, and discount
        # IMPORTANT: Use the *original* player's reward
        final_reward = final_time_step.rewards[self.player_id] if final_time_step.rewards else 0.0
        self._current_game_trajectory[-1]["reward"] = final_reward # Reward received entering final state
        self._current_game_trajectory[-1]["is_final"] = True
        self._current_game_trajectory[-1]["discount"] = 0.0

        # We need to backfill rewards: reward r_{k+1} belongs to step_data_k
        # Shift rewards back by one step
        num_steps = len(self._current_game_trajectory)
        if num_steps > 1:
            for i in range(num_steps - 1):
                self._current_game_trajectory[i]["reward"] = self._current_game_trajectory[i+1]["reward"]
                self._current_game_trajectory[i]["is_final"] = False # Only last step is final
                self._current_game_trajectory[i]["discount"] = self.config.discount
            # First reward is unknown, set to 0? Or handle in target calculation?
            # Let's set the first step's reward to 0 as it's received before first action
            self._current_game_trajectory[0]["reward"] = 0.0

        # Add the completed trajectory to the buffer
        # Ensure buffer expects List[Dict[str, Any]]]
        if self._current_game_trajectory:
            self._replay_buffer.add(self._current_game_trajectory)

        self._current_game_trajectory = [] # Clear for next episode

    def is_ready_to_learn(self):
        """Checks if the replay buffer is ready for learning."""
        return (len(self._replay_buffer) >= self.config.batch_size and
                len(self._replay_buffer) >= self._min_buffer_size_to_learn)
    # --- End Trajectory Management --- #

    def step(self, time_step, is_evaluation=False): # Removed add_transition_record
        """Returns the action to be taken and updates the agent.
           Manages internal state but relies on external loop for trajectory storage.
        """
        self.is_evaluation = is_evaluation # Set mode
        if is_evaluation:
            self.network.eval()
        else:
            self.network.train()

        if time_step.last(): # Don't act at terminal states
            # Finalize trajectory is handled by the external loop calling finalize_trajectory
            self._current_mcts_root = None # Reset MCTS root
            return rl_agent.StepOutput(action=None, probs=[])
        elif time_step.current_player() != self.player_id: # Or if it's not our turn.
             # Still need to observe if it's opponent's turn, but don't act or run MCTS
             self._current_mcts_root = None
             return rl_agent.StepOutput(action=None, probs=[]) # Or maybe a special PASS action?

        # --- MCTS --- #
        original_observation = time_step.observations["info_state"][self.player_id]
        original_legal_actions = time_step.observations["legal_actions"][self.player_id]

        # Rotate observation for canonical perspective if player 1
        if self.player_id == 1:
            canonical_observation_np = rotate_observation_long_narde(np.array(original_observation), 1)
            # TODO: Rotate legal actions (requires action mapping)
            canonical_legal_actions = [rotate_action_long_narde(a, 1) for a in original_legal_actions] # Placeholder
        else:
            canonical_observation_np = np.array(original_observation)
            canonical_legal_actions = original_legal_actions

        # Convert canonical observation to tensor
        obs_tensor = self._observation_to_tensor(canonical_observation_np.tolist())

        # Create root node and run MCTS (operates on canonical state/actions)
        with torch.no_grad(): # MCTS planning doesn't require gradients
            root = self._run_mcts(obs_tensor, canonical_legal_actions)

        self._current_mcts_root = root # Save root for stats/next step data storage

        # Select action based on visit counts (action is canonical)
        canonical_action, policy_probs = self._select_action(root)

        # --- Learning Trigger (handled by external loop) --- #
        if not is_evaluation:
            self._step_counter += 1
            # Learning step is called by external loop after env.step()

        # --- Rotate action back for environment step --- #
        if self.player_id == 1:
            # Use player_id 0 in rotate_action to rotate *back* from canonical
            action_to_take = rotate_action_long_narde(canonical_action, 0) # Placeholder
        else:
            action_to_take = canonical_action

        # Return original perspective action and canonical policy
        return rl_agent.StepOutput(action=action_to_take, probs=policy_probs)

    def _select_action(self, root: Node) -> Tuple[Action, List[float]]:
        """Selects an action based on MCTS visit counts and temperature."""
        if not root.children:
             # Should not happen if MCTS ran, maybe handle edge case (no legal actions?)
             # For Long Narde, this might mean a forced pass. MCTS should handle this.
             # If root has no children AFTER MCTS, it implies no legal actions were found/expanded.
             print("[WARN] MCTS root has no children after search. Returning None action.")
             return None, np.zeros(self.config.action_space_size) # Or handle forced pass

        visit_counts = []
        actions = []
        child_items = list(root.children.items()) # Ensure consistent order
        for action, node in child_items:
            if not node.is_chance: # Only consider action children
                actions.append(action)
                visit_counts.append(node.visit_count)

        if not actions:
            print("[WARN] MCTS root has no ACTION children after search. Returning None action.")
            return None, np.zeros(self.config.action_space_size)

        # Apply temperature softmax
        temperature = self.config.visit_softmax_temperature_fn(self._learn_step_counter)
        if temperature == 0:
            # Greedy selection
            action_idx = np.argmax(visit_counts)
            selected_action = actions[action_idx]
            probs = np.zeros(len(actions))
            probs[action_idx] = 1.0
        else:
            # Softmax sampling
            visit_counts_temp = np.array(visit_counts)**(1. / temperature)
            probs = visit_counts_temp / np.sum(visit_counts_temp)
            selected_action = np.random.choice(actions, p=probs)

        # Create full probability vector
        full_probs = np.zeros(self.config.action_space_size)
        action_to_idx_map = {a: i for i, a in enumerate(actions)}
        for i, a in enumerate(actions):
             full_probs[a] = probs[i] # Assuming actions are integer indices

        return selected_action, full_probs.tolist()

    def _run_mcts(self, observation_tensor: torch.Tensor, legal_actions: List[int]) -> Node:
        """Runs the MCTS search for the given observation."""
        # Initialize root node
        root = Node(prior=0.0, is_chance=False)
        min_max_stats = MinMaxStats(self.config.known_bounds)

        # Initial inference
        network_output = self.network.initial_inference(observation_tensor)
        root.latent_state = network_output["latent_state"]

        # Expand root node with legal actions
        # Mask policy logits for illegal actions
        policy_logits = network_output["policy_logits"].squeeze(0) # Remove batch dim
        policy_probs = F.softmax(policy_logits, dim=0)
        legal_mask = torch.zeros_like(policy_probs)
        if legal_actions:
            legal_mask[legal_actions] = 1.0
            masked_probs = policy_probs * legal_mask
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum() # Renormalize
            else:
                 # All legal actions had zero probability? Assign uniform.
                 print("[WARN] All legal actions had zero initial probability. Using uniform.")
                 masked_probs[legal_actions] = 1.0 / len(legal_actions)

            for action in legal_actions:
                root.children[action] = Node(prior=masked_probs[action].item(), is_chance=False)
        else:
            # Handle case with no legal actions (forced pass)
            # MuZero paper doesn't explicitly cover this? How should MCTS handle it?
            # Option 1: Return the root immediately, action selection handles None.
            # Option 2: Add a special PASS node?
            # Let's go with Option 1 for now.
            print("[DEBUG] MCTS Root: No legal actions provided.")
            # No children to expand.

        # Add exploration noise (if root has children)
        if root.children:
             self._add_exploration_noise(root)

        # Initial backpropagation from root inference
        self._backpropagate([root], network_output["value"].item(), self.player_id, min_max_stats)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            history = ActionOutcomeHistory(self.player_id) # TODO: Get correct initial player

            # --- Selection --- #
            while node.expanded():
                action_or_outcome, node = self._select_child(node, min_max_stats)
                history.add_action_or_outcome(action_or_outcome)
                search_path.append(node)

            # --- Expansion & Simulation --- #
            parent = search_path[-2]
            # Get the action or outcome that led to the leaf node
            action_or_outcome_taken = history.last_action_or_outcome()

            # Network inference to expand the leaf node
            # Requires parent state and action/outcome taken
            if parent.is_chance:
                # Parent was chance -> Dynamics (afterstate, outcome) -> state, reward
                # Then Prediction(state) -> policy, value
                if not isinstance(action_or_outcome_taken, Outcome):
                    raise ValueError("Expected Outcome from chance node selection")
                # TODO: Represent outcome correctly (e.g., index for one-hot)
                outcome_tensor = torch.tensor([action_or_outcome_taken], dtype=torch.long, device=self._device)
                parent_afterstate = parent.afterstate # Should have been stored
                if parent_afterstate is None: raise ValueError("Parent chance node missing afterstate")

                dynamics_output = self.network.dynamics_net(parent_afterstate, outcome_tensor)
                leaf_latent_state = dynamics_output[0]
                leaf_reward = dynamics_output[1].item()
                pred_output = self.network.prediction_net(leaf_latent_state)
                leaf_policy_logits = pred_output[0].squeeze(0)
                leaf_value = pred_output[1].item()

                # TODO: Get legal actions for the new state (tricky without env)
                # MuZero assumes model predicts valid policy over *all* actions
                # For now, expand with all actions
                policy_probs = F.softmax(leaf_policy_logits, dim=0)
                self._expand_node(node, history.to_play(), leaf_reward, policy_probs, leaf_latent_state, is_chance=False)

            else: # Parent was decision -> AfterstateDynamics(state, action) -> afterstate
                  # Then AfterstatePrediction(afterstate) -> chance_logits, Q(s,a)
                if not isinstance(action_or_outcome_taken, Action):
                    raise ValueError("Expected Action from decision node selection")
                # TODO: Represent action correctly (e.g., index for one-hot)
                action_tensor = torch.tensor([action_or_outcome_taken], dtype=torch.long, device=self._device)
                parent_state = parent.latent_state # Should have been stored
                if parent_state is None: raise ValueError("Parent decision node missing latent_state")

                leaf_afterstate = self.network.afterstate_dynamics_net(parent_state, action_tensor)
                after_pred_output = self.network.afterstate_prediction_net(leaf_afterstate)
                leaf_chance_logits = after_pred_output[0].squeeze(0)
                leaf_value = after_pred_output[1].item() # This is Q(s,a) - use for backprop

                # Expand with chance outcomes
                chance_probs = F.softmax(leaf_chance_logits, dim=0)
                # TODO: Define what outcomes correspond to the codebook indices
                # For now, assume indices 0 to codebook_size-1 are the outcomes
                outcomes = list(range(self.config.codebook_size))
                self._expand_node(node, history.to_play(), 0.0, # Reward is 0 for afterstate transition
                                 chance_probs, leaf_afterstate, is_chance=True, outcomes=outcomes)

            # --- Backpropagation --- #
            self._backpropagate(search_path, leaf_value, history.to_play(), min_max_stats)

        return root

    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[ActionOrOutcome, Node]:
        """Selects the child node based on UCB or sampling."""
        if node.is_chance:
            # Sample outcome from chance node based on predicted probabilities
            outcomes, child_nodes = zip(*node.children.items())
            probs = [n.prior for n in child_nodes]
            # Normalize just in case (should already be normalized by softmax)
            prob_sum = sum(probs)
            if prob_sum <= 0: # Handle case where all priors are zero
                probs = [1.0 / len(child_nodes)] * len(child_nodes)
            else:
                probs = [p / prob_sum for p in probs]

            sampled_outcome = np.random.choice(outcomes, p=probs)
            return sampled_outcome, node.children[sampled_outcome]
        else:
            # Select action child using PUCT formula
            best_score = -MAXIMUM_FLOAT_VALUE
            best_action = -1
            best_child = None
            for action, child in node.children.items():
                score = self._ucb_score(node, child, min_max_stats)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child
            if best_child is None:
                 # This should not happen if node has children
                 raise RuntimeError("Could not select child from non-empty decision node")
            return best_action, best_child

    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """Calculates the PUCT score."""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                        self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Use Q-value (value_sum / visit_count)
            # Normalize the value estimate Q(s,a)
            q_value = child.value()
            # Apply discount? MuZero paper uses GAE/TD-lambda for targets,
            # but uses V(s') in the UCB. Let's use child.value() directly.
            # Note: parent.to_play logic might be needed if values are from opponent perspective
            value_score = min_max_stats.normalize(q_value)
        else:
            value_score = 0.0 # Default value for unvisited nodes (or use parent value?)

        return prior_score + value_score

    def _expand_node(self, node: Node, to_play: Player, reward: float,
                     policy_or_chance_probs: torch.Tensor, state_or_afterstate: torch.Tensor,
                     is_chance: bool, outcomes: Optional[List[Outcome]] = None):
        """Expands a leaf node, adding children with priors."""
        node.to_play = to_play
        node.reward = reward
        node.is_chance = is_chance
        if is_chance:
            node.afterstate = state_or_afterstate
            if outcomes is None:
                 raise ValueError("Outcomes must be provided for chance node expansion")
            if len(outcomes) != len(policy_or_chance_probs):
                 raise ValueError(f"Number of outcomes ({len(outcomes)}) doesn't match probs ({len(policy_or_chance_probs)})")
            for outcome, prob in zip(outcomes, policy_or_chance_probs):
                node.children[outcome] = Node(prior=prob.item(), is_chance=True)
        else:
            node.latent_state = state_or_afterstate
            # Assumes policy_or_chance_probs corresponds to actions 0..N-1
            # TODO: How to handle complex/non-integer actions?
            # Assume for now actions are integers 0 to action_space_size-1
            for action_idx, prob in enumerate(policy_or_chance_probs):
                 # TODO: Only add children for legal actions? MuZero doesn't require this
                 # if action_idx in legal_actions: # Need legal_actions here?
                 node.children[action_idx] = Node(prior=prob.item(), is_chance=False)


    def _backpropagate(self, search_path: List[Node], value: float, leaf_player: Player, min_max_stats: MinMaxStats):
        """Propagates the final value estimate back up the search path.
        Handles negating values for the opponent in zero-sum games.
        Args:
            search_path: List of nodes from root to leaf.
            value: The value estimate from the leaf node (network prediction).
            leaf_player: The player whose perspective the initial `value` is from.
            min_max_stats: MinMaxStats object to update.
        """
        # The value starts from the perspective of the leaf_player.
        current_perspective_value = value

        for node in reversed(search_path):
            # Determine the value relative to the player who *made the decision* at this node.
            # If the node's player is different from the perspective of the value coming up,
            # negate the value before adding it to the node's sum.
            if node.to_play != -1 and node.to_play != leaf_player:
                node_update_value = -current_perspective_value
            else: 
                # Value is from the same perspective as the node's decision-maker (or node.to_play is -1 for root)
                node_update_value = current_perspective_value
                
            node.value_sum += node_update_value
            node.visit_count += 1
            min_max_stats.update(node.value()) # Update stats with the node's average value (always from node's perspective)

            # The value propagated to the parent needs to be discounted and include the reward *received transitioning into this node*.
            # Crucially, the value perspective must remain consistent with the initial leaf_player.
            # We discount the original `current_perspective_value`, not the potentially negated `node_update_value`.
            current_perspective_value = node.reward + self.config.discount * current_perspective_value

    def _add_exploration_noise(self, node: Node):
        """Adds Dirichlet noise to the priors of action children."""
        actions = [a for a, child in node.children.items() if not child.is_chance]
        if not actions:
            return

        alpha = self.config.root_dirichlet_alpha
        # TODO: Implement adaptive dirichlet noise if config.root_dirichlet_adaptive is True
        # if self.config.root_dirichlet_adaptive:
        #     alpha = 1.0 / np.sqrt(len(actions))

        noise = np.random.dirichlet([alpha] * len(actions))
        frac = self.config.root_dirichlet_fraction
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = child.prior * (1 - frac) + noise[i] * frac

    # --- Target Calculation Helper --- #
    def _calculate_n_step_target(self, segment_rewards: List[float], segment_discounts: List[float], bootstrap_value: float, n: int) -> float:
        """Calculates the n-step bootstrapped return.

        Args:
            segment_rewards: List of rewards [r_1, r_2, ..., r_n]. Length n.
            segment_discounts: List of discounts [d_1, d_2, ..., d_n]. Length n.
                           (d_t is discount for state t, applied to reward r_{t+1})
            bootstrap_value: Value estimate V(s_n) to bootstrap from.
            n: The number of steps (config.td_steps).

        Returns:
            The n-step target value G_0:n.
        """
        if not segment_rewards or len(segment_rewards) != n or len(segment_discounts) != n:
            # Handle cases where the segment is too short (e.g., end of episode)
            # Simple approach: Calculate return with available steps and 0 bootstrap if short.
            # More robust: The sampling in _prepare_batch should handle this.
            # For now, assume correct length is provided. If not, it's an error.
            raise ValueError(f"Segment length mismatch: Expected {n}, got rewards={len(segment_rewards)}, discounts={len(segment_discounts)}")

        target = bootstrap_value
        # Iterate backwards from n-1 down to 0
        for i in range(n - 1, -1, -1):
            target = segment_rewards[i] + segment_discounts[i] * target
        return target

    # --- Batch Preparation Helper --- #
    def _prepare_batch(self, sampled_trajectories: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Prepares tensors for observations, actions, rewards, policies, values, etc.
           from a batch of sampled trajectory segments. Includes n-step target calculation.
        """
        batch = collections.defaultdict(list)
        # Need K steps for unrolling, N steps for targets. Total length K + N + 1
        K = self.config.num_unroll_steps
        N = self.config.td_steps
        segment_len = K + N + 1

        for trajectory in sampled_trajectories:
            if len(trajectory) < segment_len:
                # print(f"[WARN] Sampled trajectory shorter ({len(trajectory)}) than required ({segment_len}). Skipping.")
                continue # Skip trajectories that are too short

            # Sample a starting point ensuring we have segment_len steps
            start_idx = random.randint(0, len(trajectory) - segment_len)
            segment = trajectory[start_idx : start_idx + segment_len]

            observations = [self._observation_to_tensor(s["observation"]).squeeze(0) for s in segment]
            actions = [s["action"] for s in segment[:-1]] # Actions for steps 0 to K+N-1
            rewards = [s["reward"] for s in segment[1:]]   # Rewards r_1 to r_{K+N}
            discounts = [s["discount"] for s in segment[1:]] # Discounts d_1 to d_{K+N}
            mcts_policies = []
            for s in segment:
                p_target = torch.zeros(self.config.action_space_size, dtype=torch.float32)
                if s["policy"]:
                    visits = sum(s["policy"].values())
                    if visits > 0:
                        for action, count in s["policy"].items():
                            if action < self.config.action_space_size: # Safety check
                                p_target[action] = count / visits
                mcts_policies.append(p_target)
            mcts_values = [s["value"] for s in segment]
            legal_masks = [s["legal_mask"] for s in segment]

            # Calculate n-step targets for the first K+1 states (0 to K)
            target_vals = []
            for k in range(K + 1):
                # Rewards r_{k+1} to r_{k+N}
                target_rewards = rewards[k : k + N]
                # Discounts d_{k+1} to d_{k+N}
                target_discounts = discounts[k : k + N]
                # Bootstrap value V(s_{k+N})
                bootstrap_value = mcts_values[k + N]
                # Calculate target G_{k:k+N}
                n_step_target = self._calculate_n_step_target(target_rewards, target_discounts, bootstrap_value, N)
                target_vals.append(n_step_target)

            # Collate data for this batch entry (only need K+1 steps for network input/loss)
            batch["observations"].append(torch.stack(observations[:K+1]))
            batch["actions"].append(torch.tensor(actions[:K], dtype=torch.long)) # Actions a_0 to a_{K-1}
            batch["rewards"].append(torch.tensor(rewards[:K], dtype=torch.float32))   # Rewards r_1 to r_K
            batch["discounts"].append(torch.tensor(discounts[:K], dtype=torch.float32)) # Discounts d_1 to d_K
            batch["target_policies"].append(torch.stack(mcts_policies[:K+1])) # MCTS policies p_0 to p_K
            batch["target_values"].append(torch.tensor(target_vals, dtype=torch.float32)) # N-step targets G_0 to G_K
            batch["target_legal_masks"].append(torch.tensor(legal_masks[:K+1], dtype=torch.float32))
            # TODO: Add target_chance_outcomes if needed
            # NOTE: target_chance_outcomes was removed because the chance prediction loss
            # is not applicable when training on observed data in Long Narde's half-move structure.
            # The chance outcome (dice roll) precedes the sequence of actions.

        # Stack collected data into batch tensors
        try:
            batch_tensors = {
                # Shape: [B, K+1, *obs_shape]
                "observations": torch.stack(batch["observations"], dim=0).to(self._device),
                # Shape: [B, K]
                "actions": torch.stack(batch["actions"], dim=0).to(self._device),
                # Shape: [B, K, 1]
                "rewards": torch.stack(batch["rewards"], dim=0).to(self._device).unsqueeze(-1),
                 # Shape: [B, K, 1]
                "discounts": torch.stack(batch["discounts"], dim=0).to(self._device).unsqueeze(-1),
                # Shape: [B, K+1, action_space_size]
                "target_policies": torch.stack(batch["target_policies"], dim=0).to(self._device),
                # Shape: [B, K+1, 1]
                "target_values": torch.stack(batch["target_values"], dim=0).to(self._device).unsqueeze(-1),
                # Shape: [B, K+1, action_space_size]
                "target_legal_masks": torch.stack(batch["target_legal_masks"], dim=0).to(self._device),
            }
        except RuntimeError as e:
            print(f"[ERROR] Failed to stack batch tensors: {e}")
            print(f"Shapes: obs={[t.shape for t in batch['observations']]}, actions={[t.shape for t in batch['actions']]}")
            # Add more shape debugging if needed
            return None # Indicate failure

        return batch_tensors

    def learn(self) -> Optional[float]:
        """Samples trajectories and performs a training step."""
        if (len(self._replay_buffer) < self.config.batch_size or
            len(self._replay_buffer) < self._min_buffer_size_to_learn):
            # print(f"[DEBUG] Skipping learn: Buffer size {len(self._replay_buffer)} < min {self._min_buffer_size_to_learn} or batch {self.config.batch_size}")
            return None # Not enough data yet

        # Sample a batch of trajectories
        # Assume sample() returns List[List[Dict]] (list of trajectories)
        sampled_trajectories = self._replay_buffer.sample(self.config.batch_size)
        if not sampled_trajectories or len(sampled_trajectories) < self.config.batch_size:
             print(f"[WARN] Replay buffer returned insufficient samples ({len(sampled_trajectories)}). Skipping learn.")
             return None

        # --- Prepare Batch Tensors --- #
        batch = self._prepare_batch(sampled_trajectories)
        if batch is None:
            print("[ERROR] Failed to prepare batch. Skipping learn step.")
            return None

        observations = batch["observations"] # Shape: [B, K+1, *obs_shape]
        actions = batch["actions"]           # Shape: [B, K]
        target_rewards = batch["rewards"]      # Shape: [B, K, 1]
        target_policies = batch["target_policies"] # Shape: [B, K+1, A]
        target_values = batch["target_values"]   # Shape: [B, K+1, 1]
        target_masks = batch["target_legal_masks"] # Shape: [B, K+1, A]
        # discounts = batch["discounts"]          # Shape: [B, K, 1] # Not directly used in loss calc, only target calc
        # Note: K = num_unroll_steps

        # Initialize losses
        total_loss = torch.tensor(0.0, device=self._device, requires_grad=True)
        policy_loss_total = torch.tensor(0.0, device=self._device)
        value_loss_total = torch.tensor(0.0, device=self._device)
        reward_loss_total = torch.tensor(0.0, device=self._device)
        afterstate_value_loss_total = torch.tensor(0.0, device=self._device)

        self.optimizer.zero_grad()

        # --- Initial Step Loss (Representation + Prediction) --- #
        initial_observation = observations[:, 0] # Get first observation: [B, *obs_shape]
        initial_inference_output = self.network.initial_inference(initial_observation)
        latent_state = initial_inference_output["latent_state"]
        policy_logits = initial_inference_output["policy_logits"]
        value = initial_inference_output["value"]

        # Value Loss (t=0)
        # Use n-step target G_0
        value_loss = F.mse_loss(value, target_values[:, 0])
        value_loss_total += value_loss

        # Policy Loss (t=0)
        # Cross-entropy between predicted logits and MCTS policy target p_0
        policy_loss = F.cross_entropy(policy_logits, target_policies[:, 0], reduction='none')
        policy_loss_total += policy_loss.mean()

        # --- Unrolled Steps Loss (Dynamics Models) --- #
        for k in range(self.config.num_unroll_steps):
            action_k = actions[:, k] # Action a_k

            # --- Afterstate Dynamics + Prediction --- # (phi + psi)
            afterstate = self.network.afterstate_dynamics(latent_state, action_k)
            afterstate_pred = self.network.afterstate_prediction(afterstate)
            chance_logits_k = afterstate_pred[0] # Predicted chance outcome logits sigma(c|as_k)
            afterstate_value_k = afterstate_pred[1] # Predicted Q(s_k, a_k)

            # Afterstate Value Loss Q(s_k, a_k)
            # Target is n-step value G_k (same as value target for state s_k)
            # This aligns with pseudocode: value_or_reward_loss(afterstate_predictions.value, value_target)
            target_G_k = target_values[:, k]
            afterstate_value_loss = F.mse_loss(afterstate_value_k, target_G_k)
            afterstate_value_loss_total += afterstate_value_loss

            # Chance loss removed - not applicable in this half-move structure
            # EXPLANATION: In Long Narde (half-move API), the dice roll (chance outcome c_k)
            # occurs *before* a sequence of half-move actions (a_k, a_{k+1}, ...).
            # The standard SMZ assumes s_k --a_k--> as_k --c_{k+1}--> s_{k+1}.
            # Therefore, training the chance prediction sigma(c | as_k) from the afterstate
            # of a half-move against the *already known* dice roll that *preceded* that half-move
            # is not meaningful. The prediction sigma(c | as) IS still used during MCTS planning
            # for hypothetical future chance nodes.
            # Still need to sample for the dynamics step, even if target is invalid
            _, sampled_chance_outcome_k = torch.max(chance_logits_k, dim=1) # Greedy sample c_{k+1}

            # --- Dynamics + Prediction --- # (g + f)
            dynamics_pred = self.network.dynamics(afterstate, sampled_chance_outcome_k)
            next_latent_state = dynamics_pred[0] # Predicted s_{k+1}
            reward_k_plus_1 = dynamics_pred[1] # Predicted reward r_{k+1}

            prediction = self.network.prediction(next_latent_state)
            policy_logits_k_plus_1 = prediction[0] # Predicted policy p_{k+1}
            value_k_plus_1 = prediction[1] # Predicted value V(s_{k+1})

            # Reward Loss (r_{k+1})
            # Target is actual reward r_{k+1} from replay buffer
            reward_loss = F.mse_loss(reward_k_plus_1, target_rewards[:, k])
            reward_loss_total += reward_loss

            # Value Loss (V(s_{k+1}))
            # Target is n-step value G_{k+1}
            value_loss_step = F.mse_loss(value_k_plus_1, target_values[:, k+1])
            value_loss_total += value_loss_step

            # Policy Loss (p_{k+1})
            # Target is MCTS policy p_{k+1} from replay buffer
            policy_loss_step = F.cross_entropy(policy_logits_k_plus_1, target_policies[:, k+1], reduction='none')
            policy_loss_total += policy_loss_step.mean()

            # Prepare latent state for next unroll step
            # TODO: Gradient scaling? (scale_gradient(next_latent_state, 0.5))
            latent_state = next_latent_state

        # --- Combine Losses --- #
        # TODO: Add loss weights (config.policy_weight, config.value_weight, etc.)
        # Placeholder: Simple averaging, assuming K = num_unroll_steps
        # Loss terms correspond to predictions at steps 0..K (K+1 terms)
        policy_loss_avg = policy_loss_total / (K + 1)
        value_loss_avg = value_loss_total / (K + 1)
        # Loss terms correspond to predictions at steps 1..K (K terms)
        reward_loss_avg = reward_loss_total / K if K > 0 else 0.0
        afterstate_value_loss_avg = afterstate_value_loss_total / K if K > 0 else 0.0

        total_loss = (policy_loss_avg + value_loss_avg + reward_loss_avg + afterstate_value_loss_avg
                      )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[WARN] NaN or Inf loss detected: {total_loss.item()}. Skipping backward pass.")
            # TODO: Add logging for individual loss components to debug NaN/Inf
            print(f"Loss breakdown: Pol={policy_loss_avg.item()}, Val={value_loss_avg.item()}, Rew={reward_loss_avg.item()}, ASVal={afterstate_value_loss_avg.item()}")
            return None

        # --- Backward Pass and Optimization --- #
        if total_loss.requires_grad:
            total_loss.backward()
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            return total_loss.item()
        else:
             print("[WARN] Calculated loss does not require grad.")
             # Add dummy loss if needed for testing graph connection
             # dummy_loss = 0.00001 * sum(p.sum() for p in self.network.parameters() if p.requires_grad)
             # dummy_loss.backward()
             # self.optimizer.step()
             return 0.0 # Or some indicator of failure

    def add_transition(self, prev_time_step, prev_action, time_step):
        """Adds transition to replay buffer. (Called by step method)"""
        # This is handled within the step method by appending to _current_game_trajectory
        # and adding the full trajectory at the end of the episode.
        pass

    def save(self, path_prefix: str):
        """Saves the agent's network and optimizer state."""
        # Saves network state_dict and optimizer state_dict
        network_path = f"{path_prefix}_network.pth"
        optimizer_path = f"{path_prefix}_optimizer.pth"
        try:
            torch.save(self.network.state_dict(), network_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            print(f"Saved agent {self.player_id} network to {network_path}")
            print(f"Saved agent {self.player_id} optimizer to {optimizer_path}")
        except Exception as e:
            print(f"Error saving agent {self.player_id} to prefix {path_prefix}: {e}")
            raise

    def restore(self, path_prefix: str):
        """Restores the agent's network and optimizer state."""
        network_path = f"{path_prefix}_network.pth"
        optimizer_path = f"{path_prefix}_optimizer.pth"
        restored_net = False
        restored_opt = False
        try:
            if os.path.exists(network_path):
                self.network.load_state_dict(torch.load(network_path, map_location=self._device))
                self.network.to(self._device)
                print(f"Restored agent {self.player_id} network from {network_path}")
                restored_net = True
            else:
                print(f"Network checkpoint not found at: {network_path}")

            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self._device))
                # Ensure optimizer state is also moved to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self._device)
                print(f"Restored agent {self.player_id} optimizer from {optimizer_path}")
                restored_opt = True
            else:
                print(f"Optimizer checkpoint not found at: {optimizer_path}")

        except Exception as e:
            print(f"Error restoring agent {self.player_id} from prefix {path_prefix}: {e}")
            # Don't raise, allow initialization from scratch if restore fails

        if not restored_net:
            print(f"Agent {self.player_id} network NOT restored. Using initialized weights.")
        if not restored_opt:
             print(f"Agent {self.player_id} optimizer NOT restored. Using fresh optimizer state.")

    @property
    def loss(self):
        return self._last_loss_value

    @property
    def step_counter(self):
        return self._step_counter
