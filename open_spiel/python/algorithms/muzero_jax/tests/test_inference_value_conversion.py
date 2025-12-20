
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock
import dataclasses

from open_spiel.python.algorithms.muzero_jax.services.inference_client import LocalInferenceClient
from open_spiel.python.algorithms.muzero_jax.training.losses import support_to_scalar, symlog

class MockConfig:
    def __init__(self, value_loss_type, reward_loss_type, value_support_size=0, reward_support_size=0):
        self.value_loss_type = value_loss_type
        self.reward_loss_type = reward_loss_type
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
        self.symlog_base = np.e
        # dummy values
        self.observation_shape = (10,)
        self.num_actions = 5
        self.num_channels = 8
        self.use_image_observation = False
        self.use_projection = False
        # Support range defaults
        self.support_min = -300.0
        self.support_max = 300.0

class MockNetwork:
    def __init__(self, config):
        self.config = config

    def initial_inference(self, observation, training=False):
        batch_size = observation.shape[0]

        # Determine output shapes based on config
        if self.config.value_loss_type in ["categorical", "kl"]:
            # Return logits
            value = jax.random.normal(jax.random.PRNGKey(0), (batch_size, self.config.value_support_size))
        elif self.config.value_loss_type == "symlog":
            # Return scalar but symlogged. e.g. real value 10 -> symlog(10)
            real_val = jnp.array([10.0] * batch_size)
            value = symlog(real_val, base=self.config.symlog_base)
        else:
            value = jnp.zeros((batch_size,))

        if self.config.reward_loss_type in ["categorical", "kl"]:
            reward = jax.random.normal(jax.random.PRNGKey(1), (batch_size, self.config.reward_support_size))
        elif self.config.reward_loss_type == "symlog":
             real_reward = jnp.array([5.0] * batch_size)
             reward = symlog(real_reward, base=self.config.symlog_base)
        else:
            reward = jnp.zeros((batch_size,))

        hidden = jnp.zeros((batch_size, self.config.num_channels))
        policy = jnp.zeros((batch_size, self.config.num_actions))

        return hidden, reward, value, policy, None, None

    def recurrent_inference(self, hidden_state, action, training=False):
        # Same logic as initial
        batch_size = hidden_state.shape[0]
        return self.initial_inference(jnp.zeros((batch_size, 10)), training)

def test_inference_value_conversion_categorical():
    """Test that InferenceClient converts categorical logits to scalar."""
    config = MockConfig("categorical", "categorical", value_support_size=601, reward_support_size=601)
    network = MockNetwork(config)
    client = LocalInferenceClient(network)

    batch_size = 2
    obs = jnp.zeros((batch_size, 10))

    # Run inference
    _, reward, value, _, _, _ = client.initial_inference(obs)

    # EXPECTATION: The client should return SCALARS.

    print(f"Value shape: {value.shape}")
    print(f"Reward shape: {reward.shape}")

    assert value.ndim == 1, f"Value output is not scalar! Shape: {value.shape}"
    assert reward.ndim == 1, f"Reward output is not scalar! Shape: {reward.shape}"

def test_inference_value_conversion_symlog():
    """Test that InferenceClient converts symlog values to scalar."""
    config = MockConfig("symlog", "symlog")
    network = MockNetwork(config)
    client = LocalInferenceClient(network)

    batch_size = 2
    obs = jnp.zeros((batch_size, 10))

    _, reward, value, _, _, _ = client.initial_inference(obs)

    # In MockNetwork, we set value to symlog(10).
    # If conversion happens, we should get ~10 back.

    expected_value = 10.0
    expected_reward = 5.0

    print(f"Value: {value}")

    assert value.ndim == 1, f"Value output is not scalar! Shape: {value.shape}"
    assert reward.ndim == 1, f"Reward output is not scalar! Shape: {reward.shape}"

    # Check if value matches expected real value
    assert np.allclose(value, expected_value, atol=0.1), f"Value {value} does not match expected real value {expected_value}. It might still be in symlog space."
    assert np.allclose(reward, expected_reward, atol=0.1), f"Reward {reward} does not match expected real value {expected_reward}."

def test_inference_value_conversion_categorical_custom_support():
    """Test that InferenceClient converts categorical logits to scalar using custom support."""
    # Config with custom support range
    config = MockConfig("categorical", "categorical", value_support_size=21, reward_support_size=21)
    config.support_min = -10.0
    config.support_max = 10.0

    class DeterministicMockNetwork(MockNetwork):
        def initial_inference(self, observation, training=False):
            batch_size = observation.shape[0]
            # Max value logits: index 20 (for size 21) has high logit
            value = jnp.zeros((batch_size, self.config.value_support_size))
            value = value.at[:, -1].set(100.0) # High probability for max value

            # Min reward logits: index 0 has high logit
            reward = jnp.zeros((batch_size, self.config.reward_support_size))
            reward = reward.at[:, 0].set(100.0) # High probability for min value

            hidden = jnp.zeros((batch_size, self.config.num_channels))
            policy = jnp.zeros((batch_size, self.config.num_actions))
            return hidden, reward, value, policy, None, None

    network = DeterministicMockNetwork(config)
    client = LocalInferenceClient(network)

    batch_size = 2
    obs = jnp.zeros((batch_size, 10))

    _, reward, value, _, _, _ = client.initial_inference(obs)

    print(f"Value (expected ~10.0): {value}")
    print(f"Reward (expected ~-10.0): {reward}")

    assert np.allclose(value, 10.0, atol=0.1), f"Value {value} should be close to support_max 10.0"
    assert np.allclose(reward, -10.0, atol=0.1), f"Reward {reward} should be close to support_min -10.0"

def test_inference_value_conversion_kl_reward():
    """Test that InferenceClient converts KL reward logits to scalar."""
    config = MockConfig("categorical", "kl", value_support_size=601, reward_support_size=601)
    network = MockNetwork(config)
    client = LocalInferenceClient(network)

    batch_size = 2
    obs = jnp.zeros((batch_size, 10))

    _, reward, value, _, _, _ = client.initial_inference(obs)

    print(f"KL Reward shape: {reward.shape}")

    assert reward.ndim == 1, f"KL Reward output is not scalar! Shape: {reward.shape}"

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_inference_value_conversion_categorical()
        print("Categorical test PASSED")
    except Exception as e:
        print(f"Categorical test FAILED: {e}")

    try:
        test_inference_value_conversion_symlog()
        print("Symlog test PASSED")
    except Exception as e:
        print(f"Symlog test FAILED: {e}")

    try:
        test_inference_value_conversion_categorical_custom_support()
        print("Categorical custom support test PASSED")
    except Exception as e:
        print(f"Categorical custom support test FAILED: {e}")

    try:
        test_inference_value_conversion_kl_reward()
        print("KL Reward test PASSED")
    except Exception as e:
        print(f"KL Reward test FAILED: {e}")
