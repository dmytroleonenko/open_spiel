import jax
import jax.numpy as jnp
from open_spiel.python.algorithms.muzero_jax.training.trainer import create_muzero_config_for_game, create_network_config_from_muzero_config
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork, RepresentationNetwork, DynamicsNetwork, PredictionNetwork, RewardNetwork, ProjectionNetwork
import flax.nnx as nnx

config = create_muzero_config_for_game('tic_tac_toe')
observation_shape = (27,)
network_config = create_network_config_from_muzero_config(config, observation_shape, config.num_actions, use_image_observation=False)

print('MuZero config num_actions:', config.num_actions)
print('Network config num_actions:', network_config.num_actions)
print('Network config num_channels:', network_config.num_channels)
print('Network config use_image_observation:', network_config.use_image_observation)
print('Network config fc_prediction_layers:', network_config.fc_prediction_layers)

rng = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng)
network = MuZeroNetwork(
    representation_network_def=RepresentationNetwork,
    dynamics_network_def=DynamicsNetwork,
    prediction_network_def=PredictionNetwork,
    reward_network_def=RewardNetwork,
    projection_network_def=ProjectionNetwork if config.use_projection else None,
    config=network_config,
    rngs=rngs
)

# Test with dummy observation
dummy_obs = jnp.ones((1, 27))
hidden_state, policy_logits, value, reward, _ = network.initial_inference(dummy_obs, training=False)
print('Hidden state shape:', hidden_state.shape)
print('Policy logits shape:', policy_logits.shape)
print('Value shape:', value.shape) 
print('Reward shape:', reward.shape)

# Test the prediction network directly
print('\n--- Testing prediction network directly ---')
pred_policy, pred_value = network.prediction_network(hidden_state, training=False)
print('Direct prediction policy shape:', pred_policy.shape)
print('Direct prediction value shape:', pred_value.shape) 