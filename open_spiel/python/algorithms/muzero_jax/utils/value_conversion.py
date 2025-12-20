"""Value conversion utilities for MuZero JAX."""

from open_spiel.python.algorithms.muzero_jax.training.losses import support_to_scalar, symexp

def convert_value(value, config):
    """
    Convert value from network output to scalar value for MCTS.

    Args:
        value: The value output from the network (logits, symlog, or scalar)
        config: The configuration object

    Returns:
        Scalar value in the real value range
    """
    if config.value_loss_type == "categorical":
        return support_to_scalar(value,
                                 support_min=config.min_value,
                                 support_max=config.max_value,
                                 num_atoms=config.value_support_size)
    elif config.value_loss_type == "symlog":
        return symexp(value)
    else: # mse
        # Ensure it's squeezed if it has shape (B, 1)
        if value.ndim == 2 and value.shape[-1] == 1:
            return value.squeeze(axis=-1)
        return value
