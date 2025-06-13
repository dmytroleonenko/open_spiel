import jax
import jax.numpy as jnp
import flax.nnx as nnx
from open_spiel.python.algorithms.muzero_jax.models.network import SupportLSTMRewardNetwork
from open_spiel.python.algorithms.muzero_jax.models.network_config import MuZeroNetworkConfig

# Create config
config = MuZeroNetworkConfig(
    observation_shape=(8, 8, 3),
    num_actions=4,
    hidden_state_size=64,
    reduced_channels_reward=16,
    spatial_size=64,
    lstm_hidden_size=128,
    use_value_prefix=True,
    reward_support_size=0
)

print(f'Config values:')
print(f'  hidden_state_size: {config.hidden_state_size}')
print(f'  reduced_channels_reward: {config.reduced_channels_reward}')
print(f'  spatial_size: {config.spatial_size}')
print(f'  Expected LSTM input: {config.reduced_channels_reward * config.spatial_size}')

# Create LSTM network
rng_key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng_key)
lstm_network = SupportLSTMRewardNetwork(config, rngs=rngs)

print(f'LSTM cell in_features: {lstm_network.lstm_cell.in_features}')
print(f'LSTM cell hidden_features: {lstm_network.lstm_cell.hidden_features}')

# Test input
batch_size = 2
hidden_state_input = jax.random.normal(rng_key, (batch_size, 8, 8, config.hidden_state_size))
reward_hidden = lstm_network.init_hidden_state(batch_size)

print(f'Hidden state input shape: {hidden_state_input.shape}')
print(f'Reward hidden shapes: c={reward_hidden[0].shape}, h={reward_hidden[1].shape}')

try:
    reward_pred, new_reward_hidden = lstm_network(hidden_state_input, reward_hidden, training=False)
    print(f'Success! Reward pred shape: {reward_pred.shape}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc() 