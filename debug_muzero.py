import jax
import jax.numpy as jnp
import flax.nnx as nnx
from open_spiel.python.algorithms.muzero_jax.models.network import (
    MuZeroNetwork, RepresentationNetwork, DynamicsNetwork, 
    PredictionNetwork, RewardNetwork
)
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig

# Create config
config = MuZeroNetworkConfig(
    observation_shape=(8, 8, 3),
    num_actions=4,
    num_channels=64,
    use_image_observation=True,
    spatial_extents=(8, 8),
    num_residual_blocks=2,
    value_support_size=0,  # Scalar values
    reward_support_size=0,  # Scalar rewards
    use_value_prefix=True,
    lstm_hidden_size=128,
    reduced_channels_reward=16,
    lstm_horizon_length=5,
    hidden_state_size=64,
    spatial_size=64,  # 8*8
)

print(f'Config values:')
print(f'  hidden_state_size: {config.hidden_state_size}')
print(f'  reduced_channels_reward: {config.reduced_channels_reward}')
print(f'  spatial_size: {config.spatial_size}')
print(f'  Expected LSTM input: {config.reduced_channels_reward * config.spatial_size}')

# Create MuZero network
rng_key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng_key)

def repr_net_def(cfg, *, rngs):
    return RepresentationNetwork(cfg, rngs=rngs)

def dyn_net_def(cfg, *, rngs):
    return DynamicsNetwork(cfg, rngs=rngs)

def pred_net_def(cfg, *, rngs):
    return PredictionNetwork(cfg, rngs=rngs)

def reward_net_def(cfg, *, rngs):
    return RewardNetwork(cfg, rngs=rngs)

model = MuZeroNetwork(
    representation_network_def=repr_net_def,
    dynamics_network_def=dyn_net_def,
    prediction_network_def=pred_net_def,
    reward_network_def=reward_net_def,
    projection_network_def=None,
    config=config,
    rngs=rngs
)

print(f'LSTM reward network exists: {model.lstm_reward_network is not None}')
if model.lstm_reward_network:
    print(f'LSTM cell in_features: {model.lstm_reward_network.lstm_cell.in_features}')
    print(f'LSTM cell hidden_features: {model.lstm_reward_network.lstm_cell.hidden_features}')

# Test representation network output
batch_size = 2
observation = jax.random.normal(rng_key, (batch_size, 8, 8, 3))
print(f'Observation shape: {observation.shape}')

hidden_state = model.representation_network(observation, training=False)
print(f'Hidden state from representation network: {hidden_state.shape}')

# Test LSTM with this hidden state
if model.lstm_reward_network:
    reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size)
    print(f'Reward hidden shapes: c={reward_hidden[0].shape}, h={reward_hidden[1].shape}')
    
    try:
        reward_pred, new_reward_hidden = model.lstm_reward_network(hidden_state, reward_hidden, training=False)
        print(f'Success! Reward pred shape: {reward_pred.shape}')
    except Exception as e:
        print(f'Error in LSTM forward pass: {e}')
        import traceback
        traceback.print_exc()

# Test full initial inference
try:
    reward_hidden = model.lstm_reward_network.init_hidden_state(batch_size) if model.lstm_reward_network else None
    output = model.initial_inference(observation, training=False, reward_hidden=reward_hidden)
    print(f'Full initial inference success!')
    print(f'Output lengths: {len(output)}')
except Exception as e:
    print(f'Error in full initial inference: {e}')
    import traceback
    traceback.print_exc() 