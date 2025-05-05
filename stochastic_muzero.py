# --- Rotation Helpers (for Long Narde / Symmetric Board Games) ---

def rotate_observation_long_narde(observation: np.ndarray, player_id: int) -> np.ndarray:
    """Rotates the observation vector to a canonical (Player 0/White) perspective.

    Assumes observation structure based on common OpenSpiel layouts like Backgammon:
    - obs[0]: Player 0 (White) borne-off count
    - obs[1]: Player 1 (Black) borne-off count
    - obs[2:26]: Point states (0-23). Encoding: 0=empty, +N=N White, -N=N Black.
    - obs[26]: Current player (0 or 1).
    - obs[27+]: Other info (dice, etc.) - copied directly.

    NOTE: This structure MUST be verified against the actual game observation spec.
    The previous assumption (0-23 White, 24-47 Black) was likely incorrect.
    """
    if player_id == 0:
        return observation # Already canonical

    if player_id == 1:
        rotated_obs = np.zeros_like(observation)
        obs_len = len(observation)

        # Rotate borne-off counts
        rotated_obs[0] = observation[1] # Canonical White borne-off = Original Black borne-off
        rotated_obs[1] = observation[0] # Canonical Black borne-off = Original White borne-off

        # Rotate points (indices 2 to 25)
        for p in range(24):
            original_point_idx = 2 + (p + 12) % 24
            canonical_point_idx = 2 + p
            # Swap player perspective by negating the value
            rotated_obs[canonical_point_idx] = -observation[original_point_idx]

        # Set player to canonical player 0
        rotated_obs[26] = 0

        # Copy remaining info (dice, etc.) - Adjust index 27 if spec differs
        if obs_len > 27:
            rotated_obs[27:] = observation[27:]

        return rotated_obs
    else:
        raise ValueError(f"Invalid player_id for rotation: {player_id}") 