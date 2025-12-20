"""Value conversion utilities for MuZero JAX."""

import math
from open_spiel.python.algorithms.muzero_jax.training.losses import support_to_scalar, symexp

def convert_value(value, config):
    """
    Convert value from network output to scalar value for MCTS.

    Args:
        value: The value output from the network (logits, symlog, or scalar)
        config: The configuration object. Can be MuZeroConfig or MuZeroNetworkConfig.
                Handles both support_min/max (MuZeroConfig) and min/max_value (MuZeroNetworkConfig).

    Returns:
        Scalar value in the real value range
    """
    value_loss_type = getattr(config, "value_loss_type", "mse")

    if value_loss_type == "categorical":
        # Handle different field names for support range
        # MuZeroConfig uses support_min/support_max
        # MuZeroNetworkConfig uses min_value/max_value (added recently)
        min_val = getattr(config, "min_value", getattr(config, "support_min", -300.0))
        max_val = getattr(config, "max_value", getattr(config, "support_max", 300.0))

        # Default support size if not present
        support_size = getattr(config, "value_support_size", 0)
        if support_size <= 0:
            support_size = 601

        return support_to_scalar(value,
                                 support_min=min_val,
                                 support_max=max_val,
                                 num_atoms=support_size)
    elif value_loss_type == "symlog":
        # Use configured symlog_base or default to e (EfficientZeroV2 parity)
        symlog_base = getattr(config, "symlog_base", math.e)
        return symexp(value, base=symlog_base)
    else: # mse
        # Ensure it's squeezed if it has shape (B, 1)
        if value.ndim == 2 and value.shape[-1] == 1:
            return value.squeeze(axis=-1)
        return value
