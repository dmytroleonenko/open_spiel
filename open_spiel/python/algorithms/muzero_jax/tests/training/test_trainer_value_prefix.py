import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses

# Import from the common utils module
from trainer_utils import (
    NUM_UNROLL_STEPS,
    NUM_ACTIONS,
    BATCH_SIZE,
    VALUE_SUPPORT_SCALAR,
    REWARD_SUPPORT_SCALAR,
    VALUE_SUPPORT_CATEGORICAL,
    REWARD_SUPPORT_CATEGORICAL,
    key as common_key, 
    cfg_flat as common_cfg_flat,
    cfg_img as common_cfg_img,
    make_model, 
    make_cfg,
    make_batch,
    MockNetCfg
)

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    apply_value_prefix_reward_accumulation as _apply_value_prefix_reward_accumulation,
    Learner, 
    MuZeroConfig, 
    Batch,
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_utils import MockRep, MockDyn, MockPred, MockRew

# Helper function to handle tuple return value from apply_value_prefix_reward_accumulation
def apply_value_prefix_reward_accumulation(rewards, config, mask=None):
    """
    Wrapper for apply_value_prefix_reward_accumulation that handles tuple return values.
    
    The function now returns (rewards, hidden_state) but tests expect just rewards.
    This wrapper extracts and returns only the rewards part for backward compatibility.
    """
    if mask is not None:
        result = _apply_value_prefix_reward_accumulation(rewards, config, mask)
    else:
        result = _apply_value_prefix_reward_accumulation(rewards, config)
    
    # Unpack result if it's a tuple (rewards, hidden_state)
    if isinstance(result, tuple):
        result_rewards, _ = result  # Extract rewards, ignore hidden_state
        return result_rewards
    else:
        return result

def test_apply_value_prefix_reward_accumulation_categorical_basic(common_key, common_cfg_flat):
    """Test basic categorical reward accumulation with value prefix enabled."""
    batch_size, sequence_length, support_size = 2, 4, 3

    # Create test config
    config = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=2)

    # Create test categorical rewards (one-hot distributions)
    # Batch 0: [1,0,0] -> [0,1,0] -> [0,0,1] -> [1,0,0]
    # Batch 1: [0,0,1] -> [0,0,1] -> [1,0,0] -> [0,1,0]
    rewards = jnp.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )

    # Expected accumulation with horizon=2 (reset at steps 0, 2):
    # Batch 0:
    # Step 0: [0,0,0] + [1,0,0] = [1,0,0]
    # Step 1: [1,0,0] + [0,1,0] = [1,1,0]
    # Step 2: [0,0,0] + [0,0,1] = [0,0,1]  # Reset at step 2
    # Step 3: [0,0,1] + [1,0,0] = [1,0,1]
    expected = jnp.array(
        [
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )

    # Apply function
    result = apply_value_prefix_reward_accumulation(rewards, config)

    # Verify accumulation
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

def test_apply_value_prefix_reward_accumulation_with_mask(common_key, common_cfg_flat):
    """Test reward accumulation with game history mask to handle invalid steps."""
    batch_size, sequence_length = 2, 5

    # Create test config
    config = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=3)

    # Create test rewards
    rewards = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 1.0, 1.0]])

    # Create mask with some invalid steps
    mask = jnp.array(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]]  # Step 2 is invalid
    )  # Steps 1,3 are invalid

    # Expected accumulation:
    # Batch 0: Step 0: 0+1=1, Step 1: 1+2=3, Step 2: 0 (masked),
    #          Step 3: 0+4=4 (reset), Step 4: 4+5=9
    # Batch 1: Step 0: 0+1=1, Step 1: 0 (masked), Step 2: 1+1=2,
    #          Step 3: 0 (reset+masked), Step 4: 0+1=1
    expected = jnp.array([[1.0, 3.0, 0.0, 4.0, 9.0], [1.0, 0.0, 2.0, 0.0, 1.0]])

    # Apply function with mask
    result = apply_value_prefix_reward_accumulation(rewards, config, mask)

    # Verify accumulation
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

def test_apply_value_prefix_reward_accumulation_horizon_length_one(common_key, common_cfg_flat):
    """Test reward accumulation with LSTM horizon length of 1 (reset every step)."""
    batch_size, sequence_length = 1, 4

    # Create test config with horizon=1
    config = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=1)

    # Create test rewards
    rewards = jnp.array([[5.0, 3.0, 8.0, 2.0]])

    # With horizon=1, accumulator resets every step, so result should equal input
    expected = rewards

    # Apply function
    result = apply_value_prefix_reward_accumulation(rewards, config)

    # Verify accumulation
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

def test_apply_value_prefix_reward_accumulation_edge_cases(common_key, common_cfg_flat):
    """Test edge cases for value prefix reward accumulation."""
    # Test with very long horizon (longer than sequence)
    config_long = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=100)
    rewards_long = jnp.array([[1.0, 2.0, 3.0]])

    # With horizon > sequence length, no resets should occur
    # Expected: [1, 3, 6] (cumulative sum)
    expected_long = jnp.array([[1.0, 3.0, 6.0]])
    result_long = apply_value_prefix_reward_accumulation(rewards_long, config_long)
    assert jnp.allclose(result_long, expected_long)

    # Test with empty sequence
    config_empty = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=2)
    rewards_empty = jnp.array([]).reshape(0, 0)
    result_empty = apply_value_prefix_reward_accumulation(rewards_empty, config_empty)
    assert result_empty.shape == (0, 0)

    # Test with zero rewards
    config_zero = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=2)
    rewards_zero = jnp.zeros((2, 4))
    expected_zero = jnp.zeros((2, 4))
    result_zero = apply_value_prefix_reward_accumulation(rewards_zero, config_zero)
    assert jnp.allclose(result_zero, expected_zero)

def test_apply_value_prefix_integration_with_trainer(common_key, common_cfg_flat):
    """Test value prefix integration with the trainer's loss computation."""
    mk, bk = jax.random.split(common_key, 2)

    # Create model and config with value prefix enabled
    model = make_model(mk, common_cfg_flat)
    config = make_cfg(
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
        NUM_UNROLL_STEPS,  # Use optimized constant instead of 3
        False,  # use_projection
        "value_prefix_integration",
    )
    config = dataclasses.replace(
        config, 
        use_value_prefix=True, 
        lstm_horizon_length=2,
        batch_size=BATCH_SIZE  # Use optimized constant
    )

    # Create batch with known reward pattern
    batch = make_batch(
        bk,
        BATCH_SIZE,  # Use optimized constant
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        NUM_UNROLL_STEPS,  # Use optimized constant
        common_cfg_flat.value_support_size,
        common_cfg_flat.reward_support_size,
    )

    # Replace with predictable rewards for testing (adjusted for optimized sizes)
    original_rewards = jnp.array([[1.0, 2.0]])  # Shape: (BATCH_SIZE=1, NUM_UNROLL_STEPS+1=2)
    batch["target_reward"] = original_rewards

    # Compute loss (this should apply value prefix internally)
    loss, metrics = Learner._compute_total_loss_static(
        model, config, batch, common_key, training=True
    )

    # Verify that loss computation completed successfully
    assert jnp.isfinite(loss)
    assert "reward_loss" in metrics
    assert jnp.isfinite(metrics["reward_loss"])

    # Compare with disabled value prefix to ensure different behavior
    config_disabled = dataclasses.replace(config, use_value_prefix=False)
    loss_disabled, metrics_disabled = Learner._compute_total_loss_static(
        model, config_disabled, batch, common_key, training=True
    )

    # Losses should be different due to reward accumulation
    assert not jnp.allclose(
        loss, loss_disabled, atol=1e-6
    ), "Expected different losses with/without value prefix"

def test_apply_value_prefix_mathematical_properties(common_key, common_cfg_flat):
    """Test mathematical properties of value prefix accumulation."""
    config = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=3)

    # Test linearity: accumulation of (a*x + b*y) should equal a*acc(x) + b*acc(y)
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    y = jnp.array([[2.0, 1.0, 4.0, 3.0, 6.0, 5.0]])
    a, b = 2.0, 3.0

    combined = a * x + b * y
    acc_combined = apply_value_prefix_reward_accumulation(combined, config)

    acc_x = apply_value_prefix_reward_accumulation(x, config)
    acc_y = apply_value_prefix_reward_accumulation(y, config)
    acc_linear = a * acc_x + b * acc_y

    assert jnp.allclose(
        acc_combined, acc_linear, rtol=1e-6
    ), "Value prefix accumulation should be linear"

    # Test that accumulation preserves total reward within each horizon segment
    rewards = jnp.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    )  # Two segments: [1,2,3] and [4,5,6]
    accumulated = apply_value_prefix_reward_accumulation(rewards, config)

    # First segment sum should match last value of first segment: 1+2+3 = 6
    assert jnp.allclose(accumulated[0, 2], 6.0)
    # Second segment sum should match last value of second segment: 4+5+6 = 15
    assert jnp.allclose(accumulated[0, 5], 15.0)

def test_apply_value_prefix_comprehensive_coverage(common_key, common_cfg_flat):
    """Comprehensive test for complete coverage of value prefix functionality."""
    # Test all combinations of scalar/categorical with different horizon lengths
    horizon_lengths = [1, 2, 5]
    reward_types = ["scalar", "categorical"]

    for horizon_len in horizon_lengths:
        for reward_type in reward_types:
            config = MuZeroConfig(
                use_value_prefix=True, lstm_horizon_length=horizon_len
            )

            if reward_type == "scalar":
                rewards = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
                expected_shape = (1, 5)
            else:
                rewards = jnp.array(
                    [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
                )
                expected_shape = (1, 5, 2)

            result = apply_value_prefix_reward_accumulation(rewards, config)
            
            # Verify shape preservation
            assert (
                result.shape == expected_shape
            ), f"Shape mismatch for {reward_type} with horizon {horizon_len}"

            # Verify numerical stability
            assert jnp.all(
                jnp.isfinite(result)
            ), f"Non-finite values for {reward_type} with horizon {horizon_len}"

            # Verify that first step equals original first step
            if reward_type == "scalar":
                assert jnp.allclose(result[0, 0], rewards[0, 0])
            else:
                assert jnp.allclose(result[0, 0], rewards[0, 0])

def test_value_prefix_config_parameter_verification(common_key, common_cfg_flat):
    """Verify that value prefix configuration parameters are properly handled."""
    # Test default configuration
    default_config = MuZeroConfig()
    assert default_config.use_value_prefix == False
    assert default_config.lstm_horizon_length == 5
    assert default_config.lstm_hidden_size == 512

    # Test configuration override
    custom_config = MuZeroConfig(
        use_value_prefix=True, lstm_horizon_length=10, lstm_hidden_size=256
    )
    assert custom_config.use_value_prefix == True
    assert custom_config.lstm_horizon_length == 10
    assert custom_config.lstm_hidden_size == 256

    # Test that config affects accumulation behavior
    config_h2 = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=2)
    config_h3 = MuZeroConfig(use_value_prefix=True, lstm_horizon_length=3)

    rewards = jnp.array([[1.0, 1.0, 1.0, 1.0]])

    result_h2 = apply_value_prefix_reward_accumulation(rewards, config_h2)
    result_h3 = apply_value_prefix_reward_accumulation(rewards, config_h3)

    # Results should be different due to different horizon lengths
    assert not jnp.allclose(
        result_h2, result_h3
    ), "Different horizon lengths should produce different results" 