
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, MagicMock
from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator
from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig

def test_convert_batch_sarsa_computation():
    # Setup
    config = MuZeroConfig(
        discount_factor=0.9,
        td_steps=2,
        num_unroll_steps=2,
        value_loss_type='mse', # Test scalar first
        trajectory_size=200
    )

    orchestrator = MagicMock(spec=MuZeroOrchestrator)
    orchestrator.muzero_config = config
    orchestrator.game_wrapper = Mock()
    orchestrator.game_wrapper.num_distinct_actions.return_value = 5
    # Mock replay buffer
    orchestrator.replay_buffer = Mock()
    orchestrator.replay_buffer._next_traj_id = 100
    orchestrator.replay_buffer._total_transitions = 20000
    orchestrator.network = Mock()

    # Mock network inference
    # input: [B, ...]
    # output: (hidden, reward, value, policy, proj, reward_hidden)
    # value should be [B] or [B, 1]
    def mock_inference(obs, training=False):
        B = obs.shape[0]
        # Return constant value 0.5 for bootstrap
        return (None, None, jnp.full((B,), 0.5), None, None, None)

    orchestrator.network.initial_inference.side_effect = mock_inference

    # Test trajectories
    # effective_length = num_unroll_steps + 1 = 3
    # rewards: [1.0, 1.0, 1.0, 1.0, 1.0]
    # t=0: r0 + g*r1 + g^2*v(t+2) = 1.0 + 0.9*1.0 + 0.81*0.5 = 2.305
    trajectory = {
        'observations': [np.zeros(1)] * 10,
        'actions': [0] * 10,
        'rewards': [1.0] * 10,
        'value_targets': [0.0] * 10,
        'policy_targets': [np.zeros(5)] * 10,
        'target_search_value': [0.0] * 10, # Not used for bootstrap if network used
        'start_transition_index': 0
    }

    # Bind the method from class to instance
    convert_method = MuZeroOrchestrator._convert_trajectories_to_batch.__get__(orchestrator, MuZeroOrchestrator)

    batch = convert_method([trajectory])

    sarsa = batch['target_sarsa_value'][0] # [K+1] = 3 steps

    # t=0
    assert np.isclose(sarsa[0], 2.305)

    # t=1: r1 + g*r2 + g^2*v(t+3) = 1 + 0.9 + 0.81*0.5 = 2.305
    assert np.isclose(sarsa[1], 2.305)

def test_convert_batch_symlog():
    # Setup
    config = MuZeroConfig(
        discount_factor=0.9,
        td_steps=1,
        num_unroll_steps=1,
        value_loss_type='symlog',
        symlog_base=np.e,
        trajectory_size=200
    )

    orchestrator = MagicMock(spec=MuZeroOrchestrator)
    orchestrator.muzero_config = config
    orchestrator.game_wrapper = Mock()
    orchestrator.game_wrapper.num_distinct_actions.return_value = 5
    orchestrator.replay_buffer = Mock()
    orchestrator.replay_buffer._next_traj_id = 100
    orchestrator.replay_buffer._total_transitions = 20000
    orchestrator.network = Mock()

    # Mock network inference returning symlog value
    # We want bootstrap value 2.0.
    # symlog(2.0) = sign(2)*ln(|2|+1) = ln(3) = 1.0986
    expected_bootstrap = 2.0
    symlog_val = np.log(3.0)

    def mock_inference(obs, training=False):
        B = obs.shape[0]
        return (None, None, jnp.full((B,), symlog_val), None, None, None)

    orchestrator.network.initial_inference.side_effect = mock_inference

    trajectory = {
        'observations': [np.zeros(1)] * 5,
        'actions': [0] * 5,
        'rewards': [1.0] * 5,
        'value_targets': [0.0] * 5,
        'policy_targets': [np.zeros(5)] * 5,
        'start_transition_index': 0
    }

    convert_method = MuZeroOrchestrator._convert_trajectories_to_batch.__get__(orchestrator, MuZeroOrchestrator)
    batch = convert_method([trajectory])

    sarsa = batch['target_sarsa_value'][0]

    # t=0: r0 + g * v(1) = 1.0 + 0.9 * 2.0 = 2.8
    assert np.isclose(sarsa[0], 2.8)

if __name__ == "__main__":
    test_convert_batch_sarsa_computation()
    test_convert_batch_symlog()
    print("Tests passed!")
