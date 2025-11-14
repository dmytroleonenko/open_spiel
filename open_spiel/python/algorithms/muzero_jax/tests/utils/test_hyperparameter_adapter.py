# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for hyperparameter_adapter."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
import dataclasses

from open_spiel.python.algorithms.muzero_jax.utils.hyperparameter_adapter import (
    HyperparameterAdapter,
    HyperparameterAdapterConfig,
)


class HyperparameterAdapterTest(parameterized.TestCase):

    def test_adaptive_td_lambda_decay(self):
        """Verify td_lambda decays linearly with sample age."""
        cfg = HyperparameterAdapterConfig(
            td_lambda=0.95, auto_td_steps=100_000, use_adaptive_td_steps=True, value_target="search"
        )
        adapter = HyperparameterAdapter(cfg, collected_transitions=200_000)

        # New sample (age=0) -> lambda should be 0.95
        new_sample_idx = jnp.asarray(200_000)
        self.assertAlmostEqual(adapter.compute_td_lambda(new_sample_idx), 0.95, places=5)

        # Mid-age sample (age=50k) -> lambda should be 0.95 * (1 - 0.5 * 0.5) = 0.7125
        mid_age_idx = jnp.asarray(150_000)
        self.assertAlmostEqual(
            adapter.compute_td_lambda(mid_age_idx), 0.7125, places=5
        )

        # Old sample (age=100k) -> lambda should be 0.95 * 0.5 = 0.475
        old_idx = jnp.asarray(100_000)
        self.assertAlmostEqual(adapter.compute_td_lambda(old_idx), 0.475, places=5)

        # Very old sample (age > 100k) -> lambda should also be 0.475
        very_old_idx = jnp.asarray(50_000)
        self.assertAlmostEqual(adapter.compute_td_lambda(very_old_idx), 0.475, places=5)

    def test_adaptive_td_steps_decay(self):
        """Verify td_steps decays in discrete steps."""
        cfg = HyperparameterAdapterConfig(
            td_steps=10, auto_td_steps=30_000, use_adaptive_td_steps=True, value_target="search"
        )
        adapter = HyperparameterAdapter(cfg, collected_transitions=100_000)

        # New sample (age < 30k) -> td_steps = 10
        new_idx = jnp.asarray(80_000)
        self.assertEqual(adapter.compute_td_steps(new_idx), 10)

        # Age ~30k -> td_steps = 9
        mid_age_idx = jnp.asarray(70_000)
        self.assertEqual(adapter.compute_td_steps(mid_age_idx), 9)

        # Age ~60k -> td_steps = 8
        old_idx = jnp.asarray(40_000)
        self.assertEqual(adapter.compute_td_steps(old_idx), 8)

        # Very old sample (age = 270k) -> td_steps = 1
        # 100k - (-170k) = 270k. 270k // 30k = 9. 10 - 9 = 1.
        very_old_idx = jnp.asarray(-170_000)
        self.assertEqual(adapter.compute_td_steps(very_old_idx), 1)

    def test_td_steps_no_decay_if_disabled(self):
        """Verify td_steps remains fixed if use_adaptive_td_steps is False."""
        cfg = HyperparameterAdapterConfig(
            td_steps=10, auto_td_steps=30_000, use_adaptive_td_steps=False
        )
        adapter = HyperparameterAdapter(cfg, collected_transitions=100_000)

        self.assertEqual(adapter.compute_td_steps(jnp.asarray(80_000)), 10)
        self.assertEqual(adapter.compute_td_steps(jnp.asarray(10_000)), 10)

        cfg_max = dataclasses.replace(cfg, value_target="max")
        adapter_max = HyperparameterAdapter(cfg_max, collected_transitions=100_000)
        self.assertEqual(adapter_max.compute_td_steps(jnp.asarray(10_000)), 10)

    def test_td_steps_no_decay_for_mixed_max_value_target(self):
        """Verify td_steps decay is disabled for 'mixed' or 'max' value targets."""
        cfg = HyperparameterAdapterConfig(
            td_steps=10,
            auto_td_steps=30_000,
            use_adaptive_td_steps=True,
            value_target="mixed",
        )
        adapter = HyperparameterAdapter(cfg, collected_transitions=100_000)

        # Even with old samples, td_steps should not decay
        self.assertEqual(adapter.compute_td_steps(jnp.asarray(10_000)), 10)

        cfg_max = dataclasses.replace(cfg, value_target="max")
        adapter_max = HyperparameterAdapter(cfg_max, collected_transitions=100_000)
        self.assertEqual(adapter_max.compute_td_steps(jnp.asarray(10_000)), 10)

    def test_vectorized_computation(self):
        """Verify vectorized calls match sequential calls."""
        cfg = HyperparameterAdapterConfig(
            td_lambda=0.9,
            td_steps=10,
            auto_td_steps=30_000,
            use_adaptive_td_steps=True,
            value_target="search",
        )
        adapter = HyperparameterAdapter(cfg, collected_transitions=100_000)
        indices = jnp.array([90_000, 70_000, 40_000, -170_000])

        lambdas_v, steps_v = adapter.vectorized(indices)

        lambdas_s = jnp.array([adapter.compute_td_lambda(i) for i in indices])
        steps_s = jnp.array([adapter.compute_td_steps(i) for i in indices])

        self.assertTrue(jnp.allclose(lambdas_v, lambdas_s))
        self.assertTrue(jnp.array_equal(steps_v, steps_s))

    def test_model_update_interval_interpolation(self):
        """Adaptive interval decays from base to the configured minimum."""
        cfg = HyperparameterAdapterConfig(auto_td_steps=100)
        adapter = HyperparameterAdapter(cfg, collected_transitions=0)
        base_interval = 40
        min_interval = 10

        early = adapter.compute_model_update_interval(0, base_interval, min_interval)
        mid = adapter.compute_model_update_interval(50, base_interval, min_interval)
        late = adapter.compute_model_update_interval(200, base_interval, min_interval)

        self.assertEqual(early, base_interval)
        self.assertLess(mid, base_interval)
        self.assertEqual(late, min_interval)

    def test_model_update_interval_handles_zero_base(self):
        """If the base interval is zero we should return zero immediately."""
        cfg = HyperparameterAdapterConfig(auto_td_steps=1)
        adapter = HyperparameterAdapter(cfg, collected_transitions=0)
        self.assertEqual(adapter.compute_model_update_interval(10, 0, 0), 0)


if __name__ == "__main__":
    absltest.main() 
