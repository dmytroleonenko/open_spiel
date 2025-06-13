import jax
import jax.numpy as jnp
from open_spiel.python.algorithms.muzero_jax.training.trainer import compute_policy_reanalysis_targets, MuZeroConfig, create_network_config_from_muzero_config
from open_spiel.python.algorithms.muzero_jax.tests.training.test_trainer_policy_reanalysis import create_test_muzero_network

config = MuZeroConfig(
    num_actions=4,
    num_unroll_steps=2,
    reanalyze_ratio=0.6,
    num_simulations=4,
    c_init=1.25,
    c_base=19652,
    c_scale=0.1,
    explore_frac=0.25,
    dirichlet_alpha=0.3,
    discount_factor=0.997,
    temperature_init=1.0,
    temperature_final=0.1,
    temperature_decay_steps=50000,
    change_temperature=True,
)

network_config = create_network_config_from_muzero_config(
    config, observation_shape=(2, 2), num_actions=4, use_image_observation=False
)
model = create_test_muzero_network(network_config)

batch_size = 4
observations = jnp.ones((batch_size, config.num_unroll_steps + 1, 2, 2))
rng_key = jax.random.PRNGKey(161718)

policies = compute_policy_reanalysis_targets(
    model=model,
    observations=observations,
    config=config,
    training=True,
    rng_key=rng_key,
)

uniform_policies = jnp.ones_like(policies) / config.num_actions
policy_differences = jnp.abs(policies - uniform_policies)

print(f'Policies shape: {policies.shape}')
print(f'Policies[0,0]: {policies[0,0]}')
print(f'Uniform[0,0]: {uniform_policies[0,0]}')
print(f'Differences[0,0]: {policy_differences[0,0]}')
print(f'Max difference: {jnp.max(policy_differences)}')
print(f'Any diff > 1e-3: {jnp.any(policy_differences > 1e-3)}')
print(f'reanalyze_ratio: {config.reanalyze_ratio}')
print(f'reanalyze_batch_size: {int(batch_size * config.reanalyze_ratio)}') 