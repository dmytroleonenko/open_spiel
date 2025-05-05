import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from open_spiel.python.utils.replay_buffer import ReplayBuffer

# An action to apply to the environment.
Action = Any
Player = int
Outcome = Any # Represents a chance outcome (e.g., dice roll)

class KnownBounds(NamedTuple):
    min: float
    max: float

# Takes the training step and returns the temperature of the softmax policy.
VisitSoftmaxTemperatureFn = Callable[[int], float]

# Define a type hint for the Network Factory which returns a PyTorch module
NetworkFactory = Callable[[], nn.Module]


@dataclasses.dataclass
class StochasticMuZeroConfig:
    """Configuration settings for the Stochastic MuZero agent (PyTorch)."""

    # --- Game Specific ---
    # Filled by the training script based on the game
    game_name: str = "unknown"
    num_players: int = 0
    observation_shape: Optional[Tuple[int, ...]] = None
    action_representation: Optional[List[int]] = None # For complex actions like Backgammon/Narde
    action_space_size: int = 0
    state_representation_size: int = 0 # Size of the input observation vector

    # --- Network ---
    network_factory: Optional[NetworkFactory] = None # Function to create the MuZero PyTorch network
    encoding_size: int = 128 # Size of the latent state 's' and afterstate 'as'

    # --- MCTS ---
    num_simulations: int = 50 # Number of MCTS simulations per move
    discount: float = 1.0 # Discount factor (1.0 for terminal rewards only games like Backgammon)

    # Temperature function for action selection during self-play
    visit_softmax_temperature_fn: Optional[VisitSoftmaxTemperatureFn] = None

    # Root prior exploration noise (Dirichlet)
    root_dirichlet_alpha: float = 0.25
    root_dirichlet_fraction: float = 0.25
    root_dirichlet_adaptive: bool = False # For games like Backgammon

    # UCB formula parameters (from pseudocode)
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    # Optional bounds for value normalization (e.g., for Backgammon)
    known_bounds: Optional[KnownBounds] = None

    # --- Replay Buffer ---
    replay_buffer_capacity: int = int(1e5) # Number of trajectories/games
    replay_buffer_class = ReplayBuffer # Class to use for replay buffer
    min_buffer_size_to_learn: int = 1000 # Minimum games/transitions before learning starts

    # --- Training ---
    batch_size: int = 128 # Batch size for training steps
    num_unroll_steps: int = 5 # Number of game steps to unroll for modeling losses
    td_steps: int = 10 # Number of steps for TD value targets (can be large for MC returns)
    # td_lambda: float = 1.0 # Optional: for TD(lambda) targets (omit for simplicity first)

    # Prioritization (set alpha=0, beta=0 for uniform sampling initially)
    priority_alpha: float = 0.0
    priority_beta: float = 0.0

    # Optimizer parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_str: str = "adamw" # 'adamw' or 'adam' or 'sgd'

    # Gradient clipping
    clip_grad_norm: float = 1.0

    # --- Stochastic MuZero Specific ---
    # Number of discrete codes for representing chance outcomes (e.g., dice rolls)
    # For Backgammon/Narde, needs to represent all possible non-double (15) and double (6) rolls = 21?
    # Or maybe just represent the individual dice outcomes (6)? Let's start with 6.
    # The original paper used 32 for 2048. Let's use a reasonable number.
    codebook_size: int = 36 # Represents 6x6 dice outcomes

    # Commitment loss weight for VQ-VAE variant if used for chance outcomes
    vq_commitment_beta: float = 0.25 # Hyperparameter 'beta' from paper

    # --- Infrastructure ---
    device: str = "cpu" # "cpu", "cuda", "mps"
    checkpoint_dir: str = "/tmp/smz_checkpoints" # Directory for saving checkpoints


# --- Helper Functions ---

def new_long_narde_config() -> StochasticMuZeroConfig:
    """Provides a default configuration for Long Narde."""

    def visit_softmax_temperature(train_steps: int) -> float:
        """Standard temperature schedule."""
        # Simplified: Constant temperature for now
        # Could implement annealing later based on training steps
        return 1.0

    # Define the network factory function here or import it
    # Placeholder: Assumes network_factory is defined elsewhere
    # from .stochastic_muzero_nets import MuZeroNetwork
    # def network_factory():
    #    return MuZeroNetwork(...) # Need to pass game specific shapes etc.

    return StochasticMuZeroConfig(
        # Game specific will be filled later
        # Network
        network_factory=None, # Must be provided later
        encoding_size=256,
        # MCTS
        num_simulations=100, # Start lower than paper's 1600
        discount=1.0,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        root_dirichlet_alpha=0.3, # Backgammon paper used adaptive, but start fixed
        root_dirichlet_fraction=0.25,
        root_dirichlet_adaptive=False, # Set True later if needed
        pb_c_base=19652,
        pb_c_init=1.25,
        known_bounds=KnownBounds(min=-1.0, max=1.0), # Assuming rewards are +/- 1
        # Replay Buffer
        replay_buffer_capacity=int(5e4), # Smaller buffer to start
        min_buffer_size_to_learn=1000,
        # Training
        batch_size=128, # Smaller batch size for initial tests
        num_unroll_steps=5,
        td_steps=100, # Effectively Monte Carlo returns for discount=1.0
        priority_alpha=0.0, # Uniform sampling
        priority_beta=0.0,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer_str="adamw",
        clip_grad_norm=1.0,
        # Stochastic MuZero Specific
        codebook_size=36, # 6x6 dice outcomes
        vq_commitment_beta=0.25,
        # Infra
        device="cpu",
        checkpoint_dir="/tmp/smz_long_narde_checkpoints",
    )

# Add similar factory functions for other games if needed
# def new_2048_config(): ... 