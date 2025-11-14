"""Offline reanalysis worker that refreshes replay buffer targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    PrioritizedTrajectoryBuffer,
    TrajectoryBuffer,
)
from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig,
    compute_policy_reanalysis_targets,
)

ReplayBufferLike = Union[TrajectoryBuffer, PrioritizedTrajectoryBuffer]
PolicyReanalysisFn = Callable[
    [MuZeroNetwork, jax.Array, MuZeroConfig, bool, Optional[jax.random.PRNGKey]],
    jax.Array,
]


@dataclass
class ReanalyzeWorkerResult:
    """Summary of a single reanalysis iteration."""

    sampled: int
    updated: int
    buffer_size: int


class ReanalyzeWorker:
    """Background worker that refreshes stored targets using the latest model."""

    def __init__(
        self,
        *,
        config: MuZeroConfig,
        model: MuZeroNetwork,
        replay_buffer: ReplayBufferLike,
        sample_batch_size: Optional[int] = None,
        rng_key: Optional[jax.random.PRNGKey] = None,
        policy_reanalysis_fn: Optional[PolicyReanalysisFn] = None,
    ):
        self.config = config
        self.model = model
        self.replay_buffer = replay_buffer
        self.sample_batch_size = sample_batch_size or config.batch_size
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        self._policy_reanalysis_fn = policy_reanalysis_fn or compute_policy_reanalysis_targets

    def _next_rng(self) -> jax.random.PRNGKey:
        self._rng_key, new_key = jax.random.split(self._rng_key)
        return new_key

    def _prepare_observation_batch(self, trajectories: Sequence[Dict[str, Any]]) -> jax.Array:
        required_steps = self.config.num_unroll_steps + 1
        processed = []
        for trajectory in trajectories:
            obs = np.array(trajectory['observations'])
            if obs.shape[0] >= required_steps:
                trimmed = obs[:required_steps]
            else:
                pad = np.repeat(obs[-1:], required_steps - obs.shape[0], axis=0)
                trimmed = np.concatenate([obs, pad], axis=0)
            processed.append(trimmed.astype(np.float32))
        return jnp.array(processed)

    def _compute_value_targets(self, observations: jax.Array) -> jax.Array:
        batch_size, num_steps = observations.shape[:2]
        flat_obs = observations.reshape(batch_size * num_steps, *observations.shape[2:])
        hidden, rewards, values, policy_logits = self.model.initial_inference(flat_obs, training=False)
        del hidden, rewards, policy_logits  # Not needed for value targets

        if values.ndim > 1 and values.shape[-1] > 1:
            values_scalar = losses_lib.support_to_scalar(
                values,
                support_min=self.config.support_min,
                support_max=self.config.support_max,
                num_atoms=values.shape[-1],
            )
        elif values.ndim > 1:
            values_scalar = jnp.squeeze(values, axis=-1)
        else:
            values_scalar = values
        return values_scalar.reshape(batch_size, num_steps)

    def _compute_priorities(
        self,
        old_policy_targets: np.ndarray,
        new_policy_targets: np.ndarray,
    ) -> float:
        delta = float(np.mean(np.abs(new_policy_targets - old_policy_targets)))
        return max(delta, getattr(self.config, "min_priority", 1e-6))

    def run_once(self) -> ReanalyzeWorkerResult:
        buffer_size = len(self.replay_buffer)
        if buffer_size == 0:
            return ReanalyzeWorkerResult(sampled=0, updated=0, buffer_size=0)

        batch_size = min(self.sample_batch_size, buffer_size)
        sample_output = self.replay_buffer.sample_batch_with_ids(batch_size, rng_key=self._next_rng())

        if len(sample_output) == 4:
            trajectories, trajectory_ids, _, _ = sample_output
        else:
            trajectories, trajectory_ids = sample_output

        observation_batch = self._prepare_observation_batch(trajectories)

        policies = self._policy_reanalysis_fn(
            self.model, observation_batch, self.config, training=False, rng_key=self._next_rng()
        )
        values = self._compute_value_targets(observation_batch)

        policies_np = np.array(policies)
        values_np = np.array(values)

        new_priorities = []
        for idx, trajectory_id in enumerate(trajectory_ids):
            old_traj = self.replay_buffer.get_trajectory(int(trajectory_id))
            self.replay_buffer.update_trajectory_targets(
                int(trajectory_id),
                policy_targets=policies_np[idx],
                value_targets=values_np[idx],
            )
            if isinstance(self.replay_buffer, PrioritizedTrajectoryBuffer):
                old_policy = np.stack(old_traj['policy_targets'][: policies_np.shape[1]], axis=0)
                new_priority = self._compute_priorities(old_policy, policies_np[idx])
                new_priorities.append(new_priority)

        if isinstance(self.replay_buffer, PrioritizedTrajectoryBuffer) and new_priorities:
            self.replay_buffer.update_priorities_by_ids(trajectory_ids, new_priorities)

        return ReanalyzeWorkerResult(
            sampled=batch_size,
            updated=batch_size,
            buffer_size=len(self.replay_buffer),
        )
