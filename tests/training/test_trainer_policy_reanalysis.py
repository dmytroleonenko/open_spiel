from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner, 
    MuZeroConfig, 
    Batch,
    compute_policy_reanalysis_targets,
    create_network_config_from_muzero_config
)

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew, MockProj


def create_test_muzero_network(config):
    """Create a simple MuZero network for testing."""
    mk = jax.random.PRNGKey(42)
    
    class TestRep(nnx.Module):
        def __init__(self, *, rngs):
            self.dense = nnx.Linear(config.observation_shape[-1], config.hidden_size, rngs=rngs)
            
        def __call__(self, x, training):
            if x.ndim > 2:
                x = x.reshape((x.shape[0], -1))
            return self.dense(x)
    
    class TestDyn(nnx.Module):
        def __init__(self, *, rngs):
            self.dense = nnx.Linear(config.hidden_size + config.num_actions, config.hidden_size, rngs=rngs)
            
        def __call__(self, h, a, training):
            a_one_hot = jax.nn.one_hot(a, config.num_actions)
            x = jnp.concatenate([h, a_one_hot], axis=-1)
            return self.dense(x)
    
    class TestPred(nnx.Module):
        def __init__(self, *, rngs):
            self.policy_head = nnx.Linear(config.hidden_size, config.num_actions, rngs=rngs)
            self.value_head = nnx.Linear(config.hidden_size, 1, rngs=rngs)
            
        def __call__(self, h, training):
            policy = self.policy_head(h)
            value = self.value_head(h)
            return policy, value
    
    class TestRew(nnx.Module):
        def __init__(self, *, rngs):
            self.reward_head = nnx.Linear(config.hidden_size, 1, rngs=rngs)
            
        def __call__(self, h, training):
            return self.reward_head(h)
    
    return MuZeroNetwork(
        representation_network_def=lambda cfg, *, rngs: TestRep(rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: TestDyn(rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: TestPred(rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: TestRew(rngs=rngs),
        projection_network_def=None,
        config=config,
        rngs=nnx.Rngs(params=mk)
    )

