# --- trainer_utils.py ---
import pytest
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.nnx.graph as nnx_graph # Keep if used by any utils or tests directly
import optax # Keep if used by any utils or tests directly

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.training.trainer import MuZeroConfig # For make_cfg
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig

# Constants
OBS_SHAPE_FLAT = (4,)
OBS_SHAPE_IMAGE = (3, 16, 16)
NUM_ACTIONS = 3
BATCH_SIZE = 1
NUM_UNROLL_STEPS = 1
VALUE_SUPPORT_SCALAR = 0
REWARD_SUPPORT_SCALAR = 0
VALUE_SUPPORT_CATEGORICAL = 5
REWARD_SUPPORT_CATEGORICAL = 7

# Mock network components - ultra-lightweight for fastest testing
class MockRep(nnx.Module):
    def __init__(self, obs_shape, hidden, *, rngs):
        # Direct linear mapping for minimal computation
        self.dense = nnx.Linear(jnp.prod(jnp.array(obs_shape)), hidden, rngs=rngs)
        
    def __call__(self, x, training):
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        return self.dense(x)  # No activation for speed

class MockDyn(nnx.Module):
    def __init__(self, hidden, nact, *, rngs):
        # Minimal embedding - just 2 dimensions
        self.embed = nnx.Embed(nact, 2, rngs=rngs)
        self.fc = nnx.Linear(hidden + 2, hidden, rngs=rngs)

    def __call__(self, h, a, training):
        e = self.embed(a)
        if e.ndim == 1:
            e = jnp.broadcast_to(e, (h.shape[0], e.shape[-1]))
        return self.fc(jnp.concatenate([h, e], -1))  # No activation for speed

class MockPred(nnx.Module):
    def __init__(self, hidden, nact, vsup, *, rngs):
        self.ph = nnx.Linear(hidden, nact, rngs=rngs)
        self.vh = nnx.Linear(hidden, vsup if vsup > 0 else 1, rngs=rngs)

    def __call__(self, h, training):
        policy_logits = self.ph(h)
        value = self.vh(h)
        
        # Apply squeezing for scalar values (like real PredictionNetwork)
        # For MSE loss (vsup == 0), ensure scalar output
        if value.ndim == 2 and value.shape[-1] == 1:
            value = jnp.squeeze(value, axis=-1)
            
        return policy_logits, value

class MockRew(nnx.Module):
    def __init__(self, hidden, rsup, *, rngs):
        self.rh = nnx.Linear(hidden, rsup if rsup > 0 else 1, rngs=rngs)

    def __call__(self, h, training):
        reward = self.rh(h)
        
        # Apply squeezing for scalar rewards (like real RewardNetwork)
        # For MSE loss (rsup == 0), ensure scalar output
        if reward.ndim == 2 and reward.shape[-1] == 1:
            reward = jnp.squeeze(reward, axis=-1)
            
        return reward

class MockProj(nnx.Module):
    def __init__(self, hidden, psize, *, rngs):
        self.ph = nnx.Linear(hidden, psize, rngs=rngs)

    def __call__(self, h, training):
        return self.ph(h)

@dataclasses.dataclass(frozen=True)
class MockNetCfg: # Simplified config for defining mock network parameters
    observation_shape: tuple = OBS_SHAPE_FLAT
    num_actions: int = NUM_ACTIONS
    hidden_size: int = 4  # Reduced from 8 to 4 for even faster testing
    value_support_size: int = VALUE_SUPPORT_SCALAR
    reward_support_size: int = REWARD_SUPPORT_SCALAR
    projection_output_size: int = 2  # Reduced from 4 to 2
    use_projection: bool = False
    batch_size: int = 1  # Already optimized to 1
    noisy_net: bool = False

class MockMuZeroNetwork(MuZeroNetwork):
    """Mock MuZeroNetwork that uses the above mock components."""
    def __init__(self, config: MuZeroNetworkConfig, *, rngs):
        hidden_size = config.num_channels
        rep_def = lambda model_cfg, *, rngs_rep: MockRep(
            config.observation_shape, hidden_size, rngs=rngs_rep
        )
        dyn_def = lambda model_cfg, *, rngs_dyn: MockDyn(
            hidden_size, config.num_actions, rngs=rngs_dyn
        )
        pred_def = lambda model_cfg, *, rngs_pred: MockPred(
            hidden_size, config.num_actions, config.value_support_size, rngs=rngs_pred
        )
        rew_def = lambda model_cfg, *, rngs_rew: MockRew(
            hidden_size, config.reward_support_size, rngs=rngs_rew
        )
        proj_def_lambda = (
            (
                lambda model_cfg, *, rngs_proj: MockProj(
                    hidden_size, config.projection_output_size, rngs=rngs_proj
                )
            )
            if config.use_projection
            else None
        )
        super().__init__(rep_def, dyn_def, pred_def, rew_def, proj_def_lambda, config, rngs=rngs)

# Common Fixtures
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

@pytest.fixture
def cfg_flat() -> MockNetCfg: # Type hint for clarity
    return MockNetCfg()

@pytest.fixture
def cfg_img() -> MockNetCfg:
    return dataclasses.replace(MockNetCfg(observation_shape=OBS_SHAPE_IMAGE))

# Helper Functions
def _mock_net_cfg_to_muzero_network_config(mock_cfg: MockNetCfg) -> MuZeroNetworkConfig:
    """Converts MockNetCfg to the MuZeroNetworkConfig needed by MuZeroNetwork."""
    return MuZeroNetworkConfig(
        observation_shape=mock_cfg.observation_shape,
        num_actions=mock_cfg.num_actions,
        num_channels=mock_cfg.hidden_size,
        value_support_size=mock_cfg.value_support_size,
        reward_support_size=mock_cfg.reward_support_size,
        use_projection=mock_cfg.use_projection,
        projection_head_output_dim=mock_cfg.projection_output_size,
        num_residual_blocks=1, # Default, not in MockNetCfg
        use_batch_norm=True, # Default
        noisy_net=mock_cfg.noisy_net
    )

def make_model(key_rng: jax.random.PRNGKey, simple_cfg: MockNetCfg) -> MuZeroNetwork:
    """Creates a MuZeroNetwork with mock components based on MockNetCfg."""
    network_config_for_muzero = _mock_net_cfg_to_muzero_network_config(simple_cfg)

    rep = lambda model_config, *, rngs: MockRep(
        simple_cfg.observation_shape, simple_cfg.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        simple_cfg.hidden_size, simple_cfg.num_actions, rngs=rngs
    )
    pred = lambda model_config, *, rngs: MockPred(
        simple_cfg.hidden_size,
        simple_cfg.num_actions,
        simple_cfg.value_support_size,
        rngs=rngs,
    )
    rew = lambda model_config, *, rngs: MockRew(
        simple_cfg.hidden_size, simple_cfg.reward_support_size, rngs=rngs
    )
    proj_def_lambda = (
        (
            lambda model_config, *, rngs: MockProj(
                simple_cfg.hidden_size, simple_cfg.projection_output_size, rngs=rngs
            )
        )
        if simple_cfg.use_projection
        else None
    )
    return MuZeroNetwork(
        rep, dyn, pred, rew, proj_def_lambda, network_config_for_muzero, rngs=nnx.Rngs(params=key_rng)
    )

def make_model_from_muzero_config(key_rng: jax.random.PRNGKey, muzero_cfg: MuZeroConfig, observation_shape: tuple = OBS_SHAPE_FLAT) -> MuZeroNetwork:
    """Creates a MuZeroNetwork with mock components based on MuZeroConfig.
    
    This is a compatibility function for tests that have MuZeroConfig instead of MockNetCfg.
    """
    # Convert MuZeroConfig to MockNetCfg for reuse
    mock_cfg = MockNetCfg(
        observation_shape=observation_shape,
        num_actions=muzero_cfg.num_actions,
        hidden_size=8,  # Default hidden size for testing
        value_support_size=muzero_cfg.value_support_size,
        reward_support_size=muzero_cfg.reward_support_size,
        projection_output_size=4,  # Default projection size
        use_projection=muzero_cfg.use_projection,
        batch_size=muzero_cfg.batch_size,
        noisy_net=getattr(muzero_cfg, 'noisy_net', False)
    )
    return make_model(key_rng, mock_cfg)

def maybe_val(x):
    return x.value if isinstance(x, nnx.Variable) else x

def make_cfg(
    vsup: int,
    rsup: int,
    steps: int,
    proj: bool,
    suffix: str, # Suffix might be less relevant if unique temp dirs are used by pytest
    use_ema: bool = False,
    ssl_weight: float = 0.0,
    l2_weight: float = 1e-4,
    checkpoint_dir: str = None,
    num_actions: int = NUM_ACTIONS,
    batch_size: int = BATCH_SIZE,
    **kwargs # To allow other MuZeroConfig overrides
) -> MuZeroConfig:
    value_loss_type = "categorical" if vsup > 0 else "mse"
    reward_loss_type = "categorical" if rsup > 0 else "mse"

    # Base config
    config_args = {
        "value_support_size": vsup,
        "reward_support_size": rsup,
        "value_loss_type": value_loss_type,
        "reward_loss_type": reward_loss_type,
        "discount_factor": 0.99,
        "num_unroll_steps": steps,
        "td_steps": steps + 1, # Common default
        "value_loss_weight": 0.25,
        "reward_loss_weight": 1.0,
        "policy_loss_weight": 1.0,
        "l2_weight": l2_weight,
        "use_projection": proj,
        "consistency_loss_coeff": ssl_weight,
        "learning_rate": 1e-3,
        "adam_b1": 0.9,
        "adam_b2": 0.999,
        "clip_grad_norm": 5.0,
        "batch_size": batch_size,
        "use_target_network_ema": use_ema,
        "ema_decay": 0.99,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_frequency": 2,
        "max_checkpoints_to_keep": 1,
        "resume_from_checkpoint": False,
        "use_iql": True,
        "iql_weight": 1.0,
        "num_actions": num_actions,
    }
    # Update with any additional kwargs passed
    config_args.update(kwargs)
    return MuZeroConfig(**config_args)


def make_batch(
    key_rng: jax.random.PRNGKey,
    bs: int,
    obs_shape: tuple,
    nact: int,
    steps: int, # This is num_unroll_steps
    vsup: int,
    rsup: int,
    proj_dim: int = None,
    use_proj: bool = False
):
    # steps here refers to num_unroll_steps (K). Batch data needs K+1 time steps.
    num_time_steps = steps + 1
    k1, k2, k3, k4, k5, k6 = jax.random.split(key_rng, 6)
    obs = jax.random.uniform(k1, (bs, num_time_steps, *obs_shape))
    acts = jax.random.randint(k2, (bs, steps), 0, nact) # Actions are for K steps
    val = (
        jax.random.normal(k3, (bs, num_time_steps))
        if vsup == 0
        else jax.random.uniform(k3, (bs, num_time_steps, vsup))
    )
    rew = (
        jax.random.normal(k4, (bs, num_time_steps))
        if rsup == 0
        else jax.random.uniform(k4, (bs, num_time_steps, rsup))
    )
    pol = jax.random.uniform(k5, (bs, num_time_steps, nact))
    pol = pol / jnp.sum(pol, axis=-1, keepdims=True)
    mask = jnp.ones((bs, num_time_steps))
    batch_data = {
        "observation": obs,
        "action": acts,
        "target_reward": rew,
        "target_value": val,
        "target_policy": pol,
        "game_history_mask": mask,
    }
    return batch_data

def create_network_config_from_muzero_config(
    muzero_config: MuZeroConfig,
    observation_shape: tuple,
    num_actions: int,
    use_image_observation: bool = False,
) -> MuZeroNetworkConfig:
    """Creates a MuZeroNetworkConfig from a MuZeroConfig and game info."""
    return MuZeroNetworkConfig(
        observation_shape=observation_shape,
        num_actions=num_actions,
        num_channels=128,  # Example, adjust or pass from muzero_config if it had this
        value_support_size=muzero_config.value_support_size,
        reward_support_size=muzero_config.reward_support_size,
        use_projection=muzero_config.use_projection,
        projection_head_output_dim=128, # Fixed parameter name
        use_image_observation=use_image_observation,
        noisy_net=getattr(muzero_config, 'noisy_net', False) # Add noisy_net from MuZeroConfig
    )

def create_test_muzero_network(config: MuZeroNetworkConfig) -> MuZeroNetwork:
    """Create a simple MuZero network for testing, expects MuZeroNetworkConfig."""
    from open_spiel.python.algorithms.muzero_jax.models.layers import MLP

    class SimpleRepresentation(nnx.Module):
        def __init__(self, cfg: MuZeroNetworkConfig, *, rngs):
            self.flatten = lambda x: x.reshape(x.shape[0], -1)
            input_size = int(jnp.prod(jnp.array(cfg.observation_shape)))
            self.mlp = MLP(input_size, [64], cfg.num_channels, rngs=rngs) # Output to num_channels

        def __call__(self, x, training=False):
            x = self.flatten(x)
            return self.mlp(x, training)

    class SimplePrediction(nnx.Module):
        def __init__(self, cfg: MuZeroNetworkConfig, *, rngs):
            self.value_head = nnx.Linear(
                cfg.num_channels,
                cfg.value_support_size if cfg.value_support_size > 0 else 1,
                rngs=rngs,
            )
            self.policy_head = nnx.Linear(cfg.num_channels, cfg.num_actions, rngs=rngs)

        def __call__(self, x, training=False):
            value = self.value_head(x)
            policy = self.policy_head(x)
            return policy, value

    class SimpleDynamics(nnx.Module):
        def __init__(self, cfg: MuZeroNetworkConfig, *, rngs):
             # Input: hidden_state + action embedding. Action embedding size could be num_channels // 2
            action_embed_size = cfg.num_channels // 2
            self.action_embed = nnx.Embed(cfg.num_actions, action_embed_size, rngs=rngs)
            self.mlp = MLP(cfg.num_channels + action_embed_size, [64], cfg.num_channels, rngs=rngs)


        def __call__(self, hidden_state, action, training=False):
            action_embedding = self.action_embed(action)
            if action_embedding.ndim == 1 : # If batch_size is 1
                 action_embedding = jnp.expand_dims(action_embedding, axis=0)
            # Ensure hidden_state and action_embedding can be concatenated
            if hidden_state.shape[0] != action_embedding.shape[0] and action_embedding.shape[0] == 1:
                 action_embedding = jnp.broadcast_to(action_embedding, (hidden_state.shape[0], action_embedding.shape[1]))

            combined_input = jnp.concatenate([hidden_state, action_embedding], axis=-1)
            return self.mlp(combined_input, training)


    class SimpleReward(nnx.Module):
        def __init__(self, cfg: MuZeroNetworkConfig, *, rngs):
            self.head = nnx.Linear(
                cfg.num_channels,
                cfg.reward_support_size if cfg.reward_support_size > 0 else 1,
                rngs=rngs,
            )

        def __call__(self, x, training=False):
            return self.head(x)

    return MuZeroNetwork(
        representation_network_def=lambda net_cfg, *, rngs: SimpleRepresentation(
            net_cfg, rngs=rngs
        ),
        prediction_network_def=lambda net_cfg, *, rngs: SimplePrediction(
            net_cfg, rngs=rngs
        ),
        dynamics_network_def=lambda net_cfg, *, rngs: SimpleDynamics(net_cfg, rngs=rngs),
        reward_network_def=lambda net_cfg, *, rngs: SimpleReward(net_cfg, rngs=rngs),
        projection_network_def=None, # Assuming no projection for this simple test network
        config=config, # This should be MuZeroNetworkConfig
        rngs=nnx.Rngs(params=jax.random.PRNGKey(42)), # Use a fixed key for reproducibility
    )
# --- END OF trainer_utils.py --- 