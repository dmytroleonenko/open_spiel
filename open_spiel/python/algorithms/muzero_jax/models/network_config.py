import dataclasses
from typing import Tuple

@dataclasses.dataclass(frozen=True)
class MuZeroNetworkConfig:
    observation_shape: Tuple[int, ...] = (0,) # Needs to be set based on game
    num_actions: int = 0 # Needs to be set based on game
    num_channels: int = 64 # Common default, from EfficientZero
    # For image observations
    use_image_observation: bool = False # Derived from observation_shape typically
    spatial_extents: Tuple[int, int] = (0,0) # H, W for conv features, if used
    downsample_channels: int = 32
    downsample_blocks: int = 1
    # For MLP/Residual Blocks
    num_residual_blocks: int = 2 # representation + dynamics
    num_fc_residual_blocks: int = 1 # prediction
    num_hidden_units_fc: int = 128 # For MLPs in prediction/reward
    # Supports
    value_support_size: int = 0 # 0 for scalar, >0 for categorical
    reward_support_size: int = 0 # 0 for scalar, >0 for categorical
    # Projection head (for SSL)
    use_projection: bool = False
    projection_hidden_dim: int = 128
    projection_head_output_dim: int = 64
    # Batch norm
    use_batch_norm: bool = True
    # Added for DummyDynamicsNetwork
    action_embedding_dim: int = 32 