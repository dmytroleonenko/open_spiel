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

"""Tests for dynamic model updates and multi-model orchestration."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from absl.testing import absltest, parameterized

from open_spiel.python.algorithms.muzero_jax.models.network import (
    MuZeroNetwork,
)
from open_spiel.python.algorithms.muzero_jax.models.network_config import (
    MuZeroNetworkConfig,
)
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    Learner,
    MuZeroConfig,
    create_network_config_from_muzero_config,
)


class DummyNet(nnx.Module):

  def __init__(self, *, rngs: nnx.Rngs):
    self.torso = nnx.Linear(in_features=1, out_features=1, rngs=rngs)

  def __call__(self, x, training=False):
    return self.torso(x)


class DummyDynamicsNet(nnx.Module):

  def __init__(self, *, rngs: nnx.Rngs):
    self.torso = nnx.Linear(in_features=2, out_features=1, rngs=rngs)

  def __call__(self, state, action, training=False):
    # action is a scalar, state has shape (1,).
    if action.ndim == 0:
        action = jnp.expand_dims(action, axis=0)
    if action.ndim == 1:
        action = jnp.expand_dims(action, axis=-1)

    net_input = jnp.concatenate([state, action], axis=-1)
    return self.torso(net_input)


class DummyPredictionNet(nnx.Module):

  def __init__(self, num_actions: int, *, rngs: nnx.Rngs):
    self.policy = nnx.Linear(in_features=1, out_features=num_actions, rngs=rngs)
    self.value = DummyNet(rngs=rngs)

  def __call__(self, x, training=False):
    return self.policy(x), self.value(x, training)

  def reset_noise(self, rng_key: jax.Array):
    pass


class DynamicModelUpdatesTest(parameterized.TestCase):

    def _get_config_and_learner(
        self, rng_key, **kwargs
    ) -> tuple[MuZeroConfig, Learner]:
        learner_cfg = MuZeroConfig(
            num_unroll_steps=1,
            td_steps=1,
            batch_size=1,
            learning_rate=0.01,
            **kwargs,
        )
        net_cfg = create_network_config_from_muzero_config(
            muzero_config=learner_cfg,
            observation_shape=(1,),
            num_actions=learner_cfg.num_actions,
            use_image_observation=False
        )
        model = MuZeroNetwork(
            representation_network_def=lambda cfg, *, rngs: DummyNet(rngs=rngs),
            prediction_network_def=lambda cfg, *, rngs: DummyPredictionNet(
                num_actions=cfg.num_actions, rngs=rngs
            ),
            dynamics_network_def=lambda cfg, *, rngs: DummyDynamicsNet(
                rngs=rngs
            ),
            reward_network_def=lambda cfg, *, rngs: DummyNet(rngs=rngs),
            projection_network_def=None,
            config=net_cfg,
            rngs=nnx.Rngs(params=rng_key),
        )
        optimizer = optax.adam(learner_cfg.learning_rate)
        learner = Learner(model, optimizer, learner_cfg, rng_key)
        return learner_cfg, learner

    def _get_dummy_batch(self, batch_size, num_unroll_steps, num_actions):
        return {
            "observation": jnp.ones([batch_size, num_unroll_steps + 1, 1]),
            "action": jnp.zeros([batch_size, num_unroll_steps]),
            "target_reward": jnp.ones([batch_size, num_unroll_steps + 1]),
            "target_value": jnp.ones([batch_size, num_unroll_steps + 1]),
            "target_policy": jnp.ones(
                [batch_size, num_unroll_steps + 1, num_actions]
            ),
            "game_history_mask": jnp.ones([batch_size, num_unroll_steps + 1]),
        }

    def test_multi_model_sync_frequency(self):
        """Verify reanalysis and self-play models sync at correct intervals."""
        rng_key = jax.random.PRNGKey(0)
        cfg, learner = self._get_config_and_learner(
            rng_key, reanalyze_update_interval=2, self_play_update_interval=3
        )
        batch = self._get_dummy_batch(
            cfg.batch_size, cfg.num_unroll_steps, cfg.num_actions
        )

        # Get initial online params
        online_params_0 = nnx.state(learner.model, nnx.Param)
        reanalysis_params_0 = nnx.state(learner.reanalysis_model, nnx.Param)
        selfplay_params_0 = nnx.state(learner.self_play_model, nnx.Param)
        self.assertTrue(
            all(
                jnp.allclose(p1, p2)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(online_params_0),
                    jax.tree_util.tree_leaves(reanalysis_params_0),
                )
            )
        )

        # Step 1
        learner.train_step(batch)
        online_params_1 = nnx.state(learner.model, nnx.Param)
        reanalysis_params_1 = nnx.state(learner.reanalysis_model, nnx.Param)
        selfplay_params_1 = nnx.state(learner.self_play_model, nnx.Param)
        # Online model updated
        self.assertFalse(
            all(
                jnp.allclose(p1, p2)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(online_params_0),
                    jax.tree_util.tree_leaves(online_params_1),
                )
            )
        )
        # Other models not updated
        self.assertTrue(
            all(
                jnp.allclose(p1, p2)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(reanalysis_params_0),
                    jax.tree_util.tree_leaves(reanalysis_params_1),
                )
            )
        )
        self.assertTrue(
            all(
                jnp.allclose(p1, p2)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(selfplay_params_0),
                    jax.tree_util.tree_leaves(selfplay_params_1),
                )
            )
        )

        # Step 2: Reanalysis model should sync
        learner.train_step(batch)
        online_params_2 = nnx.state(learner.model, nnx.Param)
        reanalysis_params_2 = nnx.state(learner.reanalysis_model, nnx.Param)
        self.assertTrue(
            all(
                jnp.allclose(p1, p2)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(online_params_2),
                    jax.tree_util.tree_leaves(reanalysis_params_2),
                )
            )
        )

        # Step 3: Self-play model should sync
        learner.train_step(batch)
        online_params_3 = nnx.state(learner.model, nnx.Param)
        selfplay_params_3 = nnx.state(learner.self_play_model, nnx.Param)
        self.assertTrue(
            all(
                jnp.allclose(p1, p2)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(online_params_3),
                    jax.tree_util.tree_leaves(selfplay_params_3),
                )
            )
        )

    def test_momentum_blend_numerical_equivalence(self):
        """Verify JIT momentum blend matches analytic formula."""
        rng_key = jax.random.PRNGKey(42)
        cfg, learner = self._get_config_and_learner(
            rng_key,
            training_steps=20000,
            ema_m_init=0.9,
            ema_m_peak=0.99,
            ema_m_final=0.999,
            ema_m_warmup_steps=100,
        )
        batch = self._get_dummy_batch(
            cfg.batch_size, cfg.num_unroll_steps, cfg.num_actions
        )

        # Test warmup phase - account for step increment during train_step
        learner.num_training_steps = 49  # Will be 50 during the actual training step
        target_before = nnx.state(learner.target_model, nnx.Param)
        learner.train_step(batch)
        online_after = nnx.state(learner.model, nnx.Param)  # Get online params AFTER optimizer update
        target_after = nnx.state(learner.target_model, nnx.Param)

        m = cfg.ema_m_init + (cfg.ema_m_peak - cfg.ema_m_init) * (50 / 100)
        expected = jax.tree_util.tree_map(
            lambda t, o: m * t + (1 - m) * o, target_before, online_after  # Use online_after
        )
        
        self.assertTrue(
            all(
                jnp.allclose(p1, p2, atol=1e-5, rtol=1e-5)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(expected),
                    jax.tree_util.tree_leaves(target_after),
                )
            )
        )

        # Test decay phase - account for step increment during train_step
        learner.num_training_steps = 10049  # Will be 10050 during the actual training step
        target_before = nnx.state(learner.target_model, nnx.Param)
        learner.train_step(batch)
        online_after = nnx.state(learner.model, nnx.Param)  # Get online params AFTER optimizer update
        target_after = nnx.state(learner.target_model, nnx.Param)

        progress = (10050 - 100) / (20000 - 100)
        cosine = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        m = cfg.ema_m_final + (cfg.ema_m_peak - cfg.ema_m_final) * cosine
        expected = jax.tree_util.tree_map(
            lambda t, o: m * t + (1 - m) * o, target_before, online_after  # Use online_after
        )
        self.assertTrue(
            all(
                jnp.allclose(p1, p2, atol=1e-5, rtol=1e-5)
                for p1, p2 in zip(
                    jax.tree_util.tree_leaves(expected),
                    jax.tree_util.tree_leaves(target_after),
                )
            )
        )

    def test_adaptive_intervals_shrink_to_minimums(self):
        """Adaptive scheduling should move towards the configured minimums."""
        rng_key = jax.random.PRNGKey(123)
        cfg, learner = self._get_config_and_learner(
            rng_key,
            auto_td_steps=4,
            reanalyze_update_interval=8,
            self_play_update_interval=6,
            reanalyze_update_interval_min=2,
            self_play_update_interval_min=1,
        )
        # Initial next syncs should match the base intervals since step=0
        self.assertEqual(learner._next_reanalyze_sync, cfg.reanalyze_update_interval)
        self.assertEqual(learner._next_self_play_sync, cfg.self_play_update_interval)

        # Advance training far enough to hit the minimum interval
        learner.num_training_steps = cfg.auto_td_steps
        learner._reschedule_model_updates()

        reanalyze_interval = learner._next_reanalyze_sync - learner.num_training_steps
        self_play_interval = learner._next_self_play_sync - learner.num_training_steps

        self.assertEqual(reanalyze_interval, learner._reanalyze_min_interval)
        self.assertEqual(self_play_interval, learner._self_play_min_interval)

    def test_compute_next_sync_step_handles_disabled_paths(self):
        """Helper should return 0 if either disabled or interval collapses."""
        rng_key = jax.random.PRNGKey(7)
        cfg, learner = self._get_config_and_learner(
            rng_key,
            reanalyze_update_interval=0,
            self_play_update_interval=5,
            self_play_update_interval_min=2,
        )
        self.assertEqual(
            learner._compute_next_sync_step(0, base_interval=0, min_interval=1, enabled=True), 0
        )
        self.assertEqual(
            learner._compute_next_sync_step(0, base_interval=10, min_interval=2, enabled=False), 0
        )

    def test_resolve_min_interval_non_positive_base(self):
        """_resolve_min_interval should short-circuit when base interval <= 0."""
        rng_key = jax.random.PRNGKey(11)
        cfg, learner = self._get_config_and_learner(rng_key)
        self.assertEqual(learner._resolve_min_interval(base_interval=0, override=5), 0)
        self.assertEqual(learner._resolve_min_interval(base_interval=-3, override=0), 0)

    def test_compute_next_sync_step_collapsed_interval(self):
        """If the adapter collapses the interval we should return zero."""
        rng_key = jax.random.PRNGKey(13)
        cfg, learner = self._get_config_and_learner(
            rng_key,
            reanalyze_update_interval=5,
            self_play_update_interval=5,
        )
        learner._model_update_adapter.compute_model_update_interval = lambda *args, **kwargs: 0
        self.assertEqual(
            learner._compute_next_sync_step(
                current_step=10,
                base_interval=5,
                min_interval=1,
                enabled=True,
            ),
            0,
        )


if __name__ == "__main__":
    absltest.main() 
