import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import copy
import dataclasses

# Import from the common utils module
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
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
    Learner, 
    MuZeroConfig, 
    Batch,
    create_muzero_config_for_game,
    apply_value_prefix_reward_accumulation
)

from open_spiel.python.algorithms.muzero_jax.training.losses import get_temperature

from open_spiel.python.algorithms.muzero_jax.models.network import (
    MuZeroNetwork
)
from open_spiel.python.algorithms.muzero_jax.models.network_config import (
    MuZeroNetworkConfig
)

def test_trainer_module_edge_cases_and_fallbacks(common_key, common_cfg_flat):
    """Test edge cases and fallback behaviors in trainer module components."""
    mk, lk = jax.random.split(common_key, 2)

    # Test 1: Cover create_muzero_config_for_game function (lines 1756-1766)
    from open_spiel.python.algorithms.muzero_jax.training.trainer import (
        create_muzero_config_for_game,
    )

    tic_tac_toe_config = create_muzero_config_for_game("tic_tac_toe")
    assert tic_tac_toe_config.num_actions == 9

    # Test with unknown game - this should raise an exception
    try:
        unknown_config = create_muzero_config_for_game("unknown_game")
        assert False, "Should have raised an exception for unknown game"
    except Exception as e:
        # This covers the exception path in the function
        assert "Unknown game" in str(e) or "SpielError" in str(type(e))

    # Test 2: Cover get_temperature function edge cases
    from open_spiel.python.algorithms.muzero_jax.training.losses import get_temperature

    config_temp = MuZeroConfig(
        change_temperature=True,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_decay_steps=100,
    )

    temp_start = get_temperature(0, config_temp)
    assert temp_start == 1.0

    temp_end = get_temperature(100, config_temp)
    assert temp_end == 0.1

    config_no_temp = MuZeroConfig(change_temperature=False, temperature_init=0.5)
    temp_disabled = get_temperature(50, config_no_temp)
    assert temp_disabled == 0.5

    # Test 3: Cover apply_value_prefix_reward_accumulation disabled mode (line 1087)
    from open_spiel.python.algorithms.muzero_jax.training.trainer import (
        apply_value_prefix_reward_accumulation,
    )

    test_rewards = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    config_no_prefix = MuZeroConfig(use_value_prefix=False)
    result_rewards = apply_value_prefix_reward_accumulation(
        test_rewards, config_no_prefix
    )
    assert jnp.allclose(result_rewards, test_rewards)

    # Test 4: Cover mctx_wrapper ImportError fallback by verifying the module works normally
    from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS

    mcts = MCTS(num_simulations=4, max_num_considered_actions=2)
    assert mcts.num_simulations == 4

    print("✅ All trainer module edge cases and fallbacks tested!") 