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
    prediction_num_blocks: int = 2 # For resblocks before value/policy heads
    fc_prediction_layers: list[int] = dataclasses.field(default_factory=lambda: [64]) # FC layers in prediction head
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
    # Loss types - configures model head output format to match loss function expectations
    value_loss_type: str = "mse" # "mse", "symlog", or "categorical"
    reward_loss_type: str = "mse" # "mse", "symlog", "kl", or "categorical"
    symlog_base: float = 2.71828182845904523536 # Base for symlog transformation (natural log base e)
    
    def get_value_output_dim(self) -> int:
        """Determines the correct output dimension for the value head based on loss type and support size."""
        if self.value_loss_type == "categorical":
            return self.value_support_size if self.value_support_size > 0 else 601
        else:  # "mse" or "symlog"
            return 1
    
    def get_reward_output_dim(self) -> int:
        """Determines the correct output dimension for the reward head based on loss type and support size."""
        if self.reward_loss_type in ["categorical", "kl"]:
            return self.reward_support_size if self.reward_support_size > 0 else 601
        else:  # "mse" or "symlog"
            return 1 