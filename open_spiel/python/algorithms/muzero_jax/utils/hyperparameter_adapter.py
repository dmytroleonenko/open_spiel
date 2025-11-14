import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

# Note: We keep this file dependency-free (except JAX) so it can be imported
# inside jit-compiled functions without triggering extra tracing overhead.

@dataclass(frozen=True)
class HyperparameterAdapterConfig:
    """Minimal subset of MuZeroConfig needed for the adapter.
    We do not import MuZeroConfig here to avoid circular imports.
    """

    td_lambda: float = 0.95
    td_steps: int = 10
    auto_td_steps: int = 30_000  # Frequency at which td_steps decays
    use_adaptive_td_steps: bool = True
    value_target: str = "mixed"  # Needed to follow EfficientZeroV2 quirk


class HyperparameterAdapter:
    """Utility that adaptively schedules td_lambda and td_steps.

    Behaviour matches the logic in EfficientZeroV2 (batch_worker.py):
    >>> delta_td = (collected_transitions - idx) // auto_td_steps
    >>> td_steps = max(1, config.td_steps - delta_td)

    And the age-based td_lambda decay:
    >>> sample_age = collected_transitions - idx
    >>> adaptive_lambda = td_lambda * (1 - 0.5 * min(sample_age / auto_td_steps, 1))
    """

    def __init__(self, cfg: HyperparameterAdapterConfig, collected_transitions: int):
        self.cfg = cfg
        # We store as scalar ints; inside jitted functions they will be lifted
        self.collected_transitions = int(collected_transitions)

    # ---------------------------------------------------------------------
    # Public JIT-friendly helpers
    # ---------------------------------------------------------------------
    def compute_td_lambda(self, sample_idx: jax.Array) -> jax.Array:
        """Return adaptive lambda for a *single* buffer index (scalar array)."""
        age = jnp.asarray(self.collected_transitions) - sample_idx
        ratio = jnp.clip(age / self.cfg.auto_td_steps, 0.0, 1.0)
        return self.cfg.td_lambda * (1.0 - 0.5 * ratio)

    def compute_td_steps(self, sample_idx: jax.Array) -> jax.Array:
        """Return adaptive td_steps (int32) for a buffer index."""
        if not self.cfg.use_adaptive_td_steps:
            return jnp.asarray(self.cfg.td_steps, dtype=jnp.int32)

        delta_td = (self.collected_transitions - sample_idx) // self.cfg.auto_td_steps
        # EfficientZeroV2 skip decay for mixed / max value targets
        delta_td = jax.lax.cond(
            jnp.logical_or(self.cfg.value_target == "mixed", self.cfg.value_target == "max"),
            lambda: jnp.asarray(0),
            lambda: delta_td,
        )
        td = self.cfg.td_steps - delta_td
        return jnp.clip(td, 1, self.cfg.td_steps).astype(jnp.int32)

    # ------------------------------------------------------------------
    # Vectorised convenience wrappers
    # ------------------------------------------------------------------
    def vectorized(self, sample_indices: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Return (td_lambdas, td_steps) for a 1-D array of indices."""
        td_lambda_vmap = jax.vmap(self.compute_td_lambda)
        td_steps_vmap = jax.vmap(self.compute_td_steps)
        return td_lambda_vmap(sample_indices), td_steps_vmap(sample_indices)

    # ------------------------------------------------------------------
    # Scheduling helpers for host-side orchestration
    # ------------------------------------------------------------------
    def compute_model_update_interval(
        self,
        step_count: int,
        base_interval: int,
        min_interval: int,
    ) -> int:
        """Interpolate between base_interval and min_interval over training."""
        if base_interval <= 0:
            return 0
        min_interval = max(1, min(min_interval, base_interval))
        if base_interval == min_interval:
            return base_interval
        decay_steps = max(1, self.cfg.auto_td_steps)
        progress = jnp.clip(step_count / decay_steps, 0.0, 1.0)
        interval = base_interval - (base_interval - min_interval) * progress
        interval = jnp.maximum(min_interval, jnp.round(interval))
        return int(interval)
