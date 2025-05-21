import pyspiel

class GameWrapper:
    """
    Thin wrapper around OpenSpiel's Game and State to provide a uniform interface.

    Methods:
        reset() -> list: Reset the game and return the initial observation.
        step(action: int) -> tuple[list, list, bool]: Apply an action, return observation, rewards, done flag.
        legal_actions() -> list[int]: Return the list of legal actions for the current state.
        current_observation() -> list: Return the current observation.
        num_distinct_actions() -> int: Return number of distinct actions in the game.
        is_chance_node() -> bool: Return True if current state is a chance node.
        chance_outcomes() -> list[tuple[int, float]]: Return list of (action, probability) pairs for chance outcomes.
        is_terminal() -> bool: Return True if the current state is terminal.
    """
    def __init__(self, game_name: str):
        self._game = pyspiel.load_game(game_name)
        self.reset()

    def reset(self) -> list:
        """Reset the game to initial state and return the initial observation."""
        self._state = self._game.new_initial_state()
        # For chance nodes, observation_tensor is not available
        if self._state.is_chance_node():
            return []
        return list(self._state.observation_tensor())

    def step(self, action: int) -> tuple[list, list, bool]:
        """Apply action to the environment and return (observation, rewards, done)."""
        try:
            self._state.apply_action(action)
        except Exception as e:
            raise RuntimeError(f"Error applying action {action}: {e}")

        done = self._state.is_terminal()

        if done:
            rewards = list(self._state.rewards()) # Rewards are valid at terminal state
            return [], rewards, True
        
        if self._state.is_chance_node():
            # For chance nodes, rewards are typically not meaningful until a player state.
            # Return empty rewards. Observation is also empty by convention here.
            return [], [], False # done is False because it's a chance node, not terminal.
        
        # Non-terminal, non-chance player node
        observation = list(self._state.observation_tensor())
        rewards = list(self._state.rewards()) # Rewards are valid at player states
        return observation, rewards, False

    def legal_actions(self) -> list:
        """Return list of legal actions at the current state."""
        return list(self._state.legal_actions())

    def current_observation(self) -> list:
        """Return the current observation without resetting."""
        if self._state.is_chance_node():
            return []
        return list(self._state.observation_tensor())

    def num_distinct_actions(self) -> int:
        """Return the total number of distinct actions in the underlying game."""
        return self._game.num_distinct_actions()

    def is_chance_node(self) -> bool:
        """Return True if the current state is a chance node."""
        return self._state.is_chance_node()

    def chance_outcomes(self) -> list:
        """Return a list of (action, probability) pairs for chance node outcomes."""
        if not self._state.is_chance_node():
            return []
        return list(self._state.chance_outcomes())

    def is_terminal(self) -> bool:
        """Return True if the current state is terminal."""
        return self._state.is_terminal() 