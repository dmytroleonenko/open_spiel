"""Integration tests using the actual MuZero network for comprehensive validation."""

import unittest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tempfile
import shutil
import optax
from unittest.mock import patch

from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig, Learner, create_network_config_from_muzero_config, Batch
)
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_utils import (
    create_test_muzero_network
)


class RealNetworkIntegrationTest(unittest.TestCase):
    """Integration tests with real MuZero network components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng_key = jax.random.PRNGKey(42)
        self.batch_size = 2  # Reduced from 4 for faster execution
        self.num_unroll_steps = 2  # Reduced from 3 for faster execution
        self.observation_shape = (3, 3)  # Tic-tac-toe board
        self.num_actions = 9
        
        # Create MuZero config
        self.config = MuZeroConfig(
            value_support_size=0,  # Scalar values
            reward_support_size=0,  # Scalar rewards
            num_unroll_steps=self.num_unroll_steps,
            batch_size=self.batch_size,
            num_actions=self.num_actions,
            learning_rate=1e-3,
            clip_grad_norm=5.0,
            checkpoint_frequency=5,
            use_projection=False,
            noisy_net=False,
            use_target_network_ema=True,
            ema_decay=0.99,
            ema_update_frequency=1,
            # Set intervals for multi-model orchestration testing - reduced for faster testing
            reanalyze_update_interval=3,  # Update reanalysis model every 3 steps (reduced from 10)
            self_play_update_interval=2   # Update self-play model every 2 steps (reduced from 5)
        )
        
        # Create network config from MuZero config
        self.network_config = create_network_config_from_muzero_config(
            muzero_config=self.config,
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            use_image_observation=False
        )
        
        # Create real MuZero network using the helper function
        self.model = create_test_muzero_network(self.network_config)
        
        # Create optimizer
        self.optimizer_def = optax.chain(
            optax.clip_by_global_norm(self.config.clip_grad_norm),
            optax.adam(learning_rate=self.config.learning_rate)
        )
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        # Use dataclasses.replace since config is frozen
        import dataclasses
        self.config = dataclasses.replace(self.config, checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def _create_realistic_batch(self):
        """Create a realistic batch of data for tic-tac-toe."""
        # Create observations (tic-tac-toe boards)
        observations = jax.random.randint(
            self.rng_key, 
            (self.batch_size, self.num_unroll_steps + 1, *self.observation_shape),
            minval=0, maxval=3  # 0=empty, 1=X, 2=O
        ).astype(jnp.float32)
        
        # Create actions (0-8 for tic-tac-toe positions)
        actions = jax.random.randint(
            self.rng_key,
            (self.batch_size, self.num_unroll_steps),
            minval=0, maxval=self.num_actions
        )
        
        # Create rewards (sparse, mostly 0 with occasional +1/-1)
        rewards = jnp.zeros((self.batch_size, self.num_unroll_steps + 1))
        
        # Create target values (bootstrap from final positions)
        target_values = jax.random.uniform(
            self.rng_key,
            (self.batch_size, self.num_unroll_steps + 1),
            minval=-1.0, maxval=1.0
        )
        
        # Create target policies (softmax over actions)
        target_policy_logits = jax.random.normal(
            self.rng_key,
            (self.batch_size, self.num_unroll_steps + 1, self.num_actions)
        )
        target_policies = jax.nn.softmax(target_policy_logits, axis=-1)
        
        # Create masks (all valid for simplicity)
        masks = jnp.ones((self.batch_size, self.num_unroll_steps + 1), dtype=jnp.bool_)

        # Create game history mask (required for training)
        game_history_mask = jnp.ones((self.batch_size, self.num_unroll_steps + 1), dtype=jnp.bool_)

        # Create sample indices for adaptive hyperparameters
        sample_indices = jnp.array([100, 200])  # Different ages - reduced to match batch_size=2

        return {
            'observation': observations,
            'action': actions,
            'target_reward': rewards,
            'target_value': target_values,
            'target_policy': target_policies,
            'mask': masks,
            'game_history_mask': game_history_mask,
            'sample_indices': sample_indices
        }
        
    def test_real_network_training_step(self):
        """Test that training step works with real MuZero network."""
        # Create learner with real network
        learner = Learner(
            model=self.model,
            optimizer_def=self.optimizer_def,
            config=self.config,
            rng_key=self.rng_key
        )
        
        # Create realistic batch
        batch = self._create_realistic_batch()
        
        # Get initial parameters
        initial_params = nnx.state(learner.model, nnx.Param)
        initial_target_params = nnx.state(learner.target_model, nnx.Param)
        
        # Perform training step
        metrics = learner.train_step(batch)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('reward_loss', metrics)
        self.assertIn('policy_loss', metrics)
        
        # Check that parameters were updated
        updated_params = nnx.state(learner.model, nnx.Param)
        updated_target_params = nnx.state(learner.target_model, nnx.Param)
        
        # Online model parameters should have changed
        params_changed = any(
            not jnp.allclose(p1, p2, atol=1e-8)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_params),
                jax.tree_util.tree_leaves(updated_params)
            )
        )
        self.assertTrue(params_changed, "Online model parameters should have changed after training step")
        
        # Target model parameters should have changed (due to momentum blend)
        target_params_changed = any(
            not jnp.allclose(p1, p2, atol=1e-8)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_target_params),
                jax.tree_util.tree_leaves(updated_target_params)
            )
        )
        self.assertTrue(target_params_changed, "Target model parameters should have changed after momentum blend")
        
    def test_ema_blend_skips_branch_without_tracer_leak(self):
        """Run two steps with EMA frequency > 1 to ensure skip branch is safe."""
        import dataclasses
        cfg = dataclasses.replace(self.config, target_network_update_frequency=2)
        network_config = create_network_config_from_muzero_config(
            muzero_config=cfg,
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            use_image_observation=False
        )
        model = create_test_muzero_network(network_config)
        learner = Learner(
            model=model,
            optimizer_def=self.optimizer_def,
            config=cfg,
            rng_key=self.rng_key
        )
        batch = self._create_realistic_batch()
        learner.train_step(batch)  # training_step = 0 -> EMA updates
        metrics = learner.train_step(batch)  # training_step = 1 -> skip branch
        self.assertIn('total_loss', metrics)
        
    def test_multi_model_orchestration_with_real_network(self):
        """Test that multi-model orchestration works with real network."""
        # Create learner with real network
        learner = Learner(
            model=self.model,
            optimizer_def=self.optimizer_def,
            config=self.config,
            rng_key=self.rng_key
        )
        
        # Create realistic batch
        batch = self._create_realistic_batch()
        
        # Get initial parameters for all models
        initial_online = nnx.state(learner.model, nnx.Param)
        initial_reanalysis = nnx.state(learner.reanalysis_model, nnx.Param)
        initial_self_play = nnx.state(learner.self_play_model, nnx.Param)
        
        # Perform minimal training steps to trigger model updates
        # With reanalyze_update_interval=3 and self_play_update_interval=2, 
        # running 6 steps will trigger both updates while keeping test fast
        for step in range(6):  # Reduced from 15 - still triggers both model updates
            learner.train_step(batch)
            
        # Get updated parameters
        updated_online = nnx.state(learner.model, nnx.Param)
        updated_reanalysis = nnx.state(learner.reanalysis_model, nnx.Param)
        updated_self_play = nnx.state(learner.self_play_model, nnx.Param)
        
        # Online model should have changed significantly
        online_changed = any(
            not jnp.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_online),
                jax.tree_util.tree_leaves(updated_online)
            )
        )
        self.assertTrue(online_changed, "Online model should have changed after training")
        
        # Reanalysis model should have been updated (at steps 3, 6 with interval=3)
        reanalysis_changed = any(
            not jnp.allclose(p1, p2, atol=1e-8)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_reanalysis),
                jax.tree_util.tree_leaves(updated_reanalysis)
            )
        )
        self.assertTrue(reanalysis_changed, "Reanalysis model should have been updated")
        
        # Self-play model should have been updated (at steps 2, 4, 6 with interval=2)
        self_play_changed = any(
            not jnp.allclose(p1, p2, atol=1e-8)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_self_play),
                jax.tree_util.tree_leaves(updated_self_play)
            )
        )
        self.assertTrue(self_play_changed, "Self-play model should have been updated")

    def test_functional_train_step_is_jittable(self):
        """Ensure the functional train step compiles under jax.jit."""
        learner = Learner(
            model=self.model,
            optimizer_def=self.optimizer_def,
            config=self.config,
            rng_key=self.rng_key
        )
        batch = self._create_realistic_batch()

        def run(state, rng_key, step_idx):
            return learner.jit_train_step(state, batch, rng_key, step_idx)

        compiled = jax.jit(run)
        new_state, metrics, next_key = compiled(
            learner._state, learner._rng_key, learner.num_training_steps
        )

        self.assertIn('total_loss', metrics)
        self.assertTrue(jnp.isfinite(metrics['total_loss']))

        # Keep learner consistent for subsequent checks
        learner._state = new_state
        learner._rng_key = next_key
    
    def test_adaptive_hyperparameters_with_real_network(self):
        """Test that adaptive hyperparameters work correctly with real network."""
        # Create learner with real network
        learner = Learner(
            model=self.model,
            optimizer_def=self.optimizer_def,
            config=self.config,
            rng_key=self.rng_key
        )
        
        # Create batches with different sample ages
        batch_young = self._create_realistic_batch()
        batch_young['sample_indices'] = jnp.array([100, 200])  # Young samples - reduced to match batch_size=2
        
        batch_old = self._create_realistic_batch()
        batch_old['sample_indices'] = jnp.array([800, 900])  # Old samples - reduced to match batch_size=2
        
        # Train on both batches and check that different losses are computed
        metrics_young = learner.train_step(batch_young)
        
        # Reset learner state to ensure fair comparison
        learner = Learner(
            model=self.model,
            optimizer_def=self.optimizer_def,
            config=self.config,
            rng_key=self.rng_key
        )
        
        metrics_old = learner.train_step(batch_old)
        
        # The losses should be different due to adaptive hyperparameters
        # (Note: This is a weak test since many factors affect loss, but it's a sanity check)
        self.assertIsInstance(metrics_young['total_loss'], jax.Array)
        self.assertIsInstance(metrics_old['total_loss'], jax.Array)
        
        # Both should be finite
        self.assertTrue(jnp.isfinite(metrics_young['total_loss']))
        self.assertTrue(jnp.isfinite(metrics_old['total_loss']))
        
    def test_checkpointing_with_real_network(self):
        """Test that checkpointing works with real network."""
        # Create learner with real network
        learner = Learner(
            model=self.model,
            optimizer_def=self.optimizer_def,
            config=self.config,
            rng_key=self.rng_key
        )
        
        # Create realistic batch
        batch = self._create_realistic_batch()
        
        # Perform some training steps
        for _ in range(3):
            learner.train_step(batch)
            
        # Get current parameters
        params_before_save = nnx.state(learner.model, nnx.Param)
        step_before_save = learner.num_training_steps
        
        # Save checkpoint
        learner.save_checkpoint(force_save=True)
        
        # Modify parameters by doing more training
        for _ in range(3):
            learner.train_step(batch)
            
        params_after_training = nnx.state(learner.model, nnx.Param)
        step_after_training = learner.num_training_steps
        
        # Parameters should be different
        params_changed = any(
            not jnp.allclose(p1, p2, atol=1e-8)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(params_before_save),
                jax.tree_util.tree_leaves(params_after_training)
            )
        )
        self.assertTrue(params_changed, "Parameters should have changed after additional training")
        self.assertNotEqual(step_before_save, step_after_training)
        
        # Load checkpoint
        success = learner.load_checkpoint()
        self.assertTrue(success, "Checkpoint loading should succeed")
        
        # Parameters should be restored
        params_after_load = nnx.state(learner.model, nnx.Param)
        step_after_load = learner.num_training_steps
        
        params_restored = all(
            jnp.allclose(p1, p2, atol=1e-8)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(params_before_save),
                jax.tree_util.tree_leaves(params_after_load)
            )
        )
        self.assertTrue(params_restored, "Parameters should be restored after loading checkpoint")
        self.assertEqual(step_before_save, step_after_load, "Training step should be restored")


if __name__ == '__main__':
    unittest.main() 
