import jax
import jax.numpy as jnp
import pytest
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS


def test_mcts_run_calls_gumbel_policy(monkeypatch):
    called = {}
    def fake_policy(**kwargs):
        called.update(kwargs)
        return 'fake_result'
    
    # Mock the mctx.gumbel_muzero_policy function directly since that's what MCTS.run calls internally
    monkeypatch.setattr('mctx.gumbel_muzero_policy', fake_policy)
    
    num_simulations = 5
    max_actions = 10
    gumbel_scale = 0.7
    mcts = MCTS(num_simulations=num_simulations, max_num_considered_actions=max_actions, gumbel_scale=gumbel_scale)
    params = {'param': 'value'}
    rng_key = jax.random.PRNGKey(0)
    root = 'root_node'
    # Define a dummy recurrent function
    def recurrent_fn(hidden_state, action, rng_key, training=False):
        return 'out'
    result = mcts.run(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        invalid_actions='inv',
        max_depth=3,
        loop_fn='loop',
        qtransform='qt'
    )
    assert result == 'fake_result'
    assert called['params'] == params
    assert jnp.array_equal(called['rng_key'], rng_key)
    assert called['root'] == root
    assert called['recurrent_fn'] == recurrent_fn
    assert called['num_simulations'] == num_simulations
    assert called['max_num_considered_actions'] == max_actions
    assert called['gumbel_scale'] == gumbel_scale
    assert called['invalid_actions'] == 'inv'
    assert called['max_depth'] == 3
    assert called['loop_fn'] == 'loop'
    assert called['qtransform'] == 'qt'