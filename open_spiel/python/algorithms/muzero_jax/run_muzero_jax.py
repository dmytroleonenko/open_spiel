#!/usr/bin/env python3
"""
Main orchestration script for MuZero JAX implementation.

This script coordinates the training process between actors (self-play) and learners,
manages configuration through Hydra, and provides logging through Weights & Biases.

Usage:
    python run_muzero_jax.py                     # Use default config
    python run_muzero_jax.py game=chess         # Override game
    python run_muzero_jax.py training.learning_rate=0.01  # Override specific params
"""

import os
import sys
import time
import queue
import logging
import threading
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import argparse

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional during tests
    tqdm = None

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# MuZero JAX imports
from open_spiel.python.algorithms.muzero_jax.training.trainer import (
    MuZeroConfig, 
    Learner, 
    create_muzero_config_for_game,
    create_network_config_from_muzero_config
)
from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor
from open_spiel.python.algorithms.muzero_jax.self_play import bootstrap_actor as bootstrap_module

if TYPE_CHECKING:  # pragma: no cover - static typing helpers only
    from open_spiel.python.algorithms.muzero_jax.self_play.bootstrap_actor import (
        BootstrapActor as BootstrapActorType,
        BootstrapConfig as BootstrapConfigType,
    )
else:
    BootstrapActorType = Any  # pragma: no cover - typing helper
    BootstrapConfigType = Any  # pragma: no cover - typing helper
from open_spiel.python.algorithms.muzero_jax.replay_buffer.replay_buffer import (
    TrajectoryBuffer, 
    PrioritizedTrajectoryBuffer
)
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.utils.checkpointing import (
    create_checkpoint_manager,
    get_latest_checkpoint
)
from open_spiel.python.algorithms.muzero_jax.mcts.mctx_wrapper import MCTS
from open_spiel.python.algorithms.muzero_jax.services.inference_client import (
    build_inference_client,
    LocalInferenceClient,
)
from open_spiel.python.algorithms.muzero_jax.services.parameter_client_inference_adapter import (
    ParameterRefreshingInferenceClient,
)
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayClient,
    RemoteReplayBufferAdapter,
)
from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
    GrpcParameterPublisherClient,
    GrpcParameterPublisherServer,
    LocalParameterPublisher,
    GrpcParameterPublisherServer,
)
from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
    GrpcReplayClient,
    RemoteReplayBufferAdapter,
)

def _parse_cli_verbosity(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    parser.add_argument('-q', '--quiet', action='store_true', dest='quiet')
    args, remaining = parser.parse_known_args(argv[1:])
    argv[:] = [argv[0]] + remaining
    if args.quiet:  # pragma: no cover - trivial CLI parsing
        return -1
    return args.verbosity  # pragma: no cover - trivial CLI parsing


CLI_VERBOSITY = _parse_cli_verbosity(sys.argv)


def get_cli_verbosity() -> int:
    return CLI_VERBOSITY


def build_replay_buffer(
    config: DictConfig,
    observation_shape,
    num_actions: int,
    game_obj,
    register_replay_client=None,
):
    """Construct a local or remote replay buffer based on Hydra config."""
    base_length = getattr(config.replay_buffer, "max_trajectory_length", 0) or 0
    game_required_length = 0
    if game_obj is not None and hasattr(game_obj, "max_game_length"):
        try:
            game_required_length = int(game_obj.max_game_length()) + 1
        except Exception:  # pragma: no cover - defensive path
            game_required_length = 0

    if base_length <= 0 and game_required_length <= 0:
        max_trajectory_length = 200
    elif base_length <= 0:
        max_trajectory_length = game_required_length
    elif game_required_length <= 0:
        max_trajectory_length = base_length
    else:
        if base_length < game_required_length:
            logger.info(
                "Replay buffer max_trajectory_length=%s is below %s max_game_length=%s; expanding to %s steps.",
                base_length,
                getattr(config.game, "name", "unknown"),
                game_required_length - 1,
                game_required_length,
            )
        max_trajectory_length = max(base_length, game_required_length)

    if getattr(config.replay_buffer, "remote_enabled", False):
        endpoint = getattr(config.replay_buffer, "rpc_endpoint", "") or ""
        if not endpoint:
            raise ValueError("replay_buffer.remote_enabled=true but no rpc_endpoint provided")
        prioritized = getattr(config.replay_buffer, "priority_alpha", 0) > 0
        client = GrpcReplayClient(
            endpoint,
            timeout_s=float(getattr(config.replay_buffer, "timeout_s", 10.0)),
            max_message_mb=int(getattr(config.replay_buffer, "grpc_max_message_mb", 64)),
        )
        if register_replay_client:
            register_replay_client(client)
        logger.info("Using remote replay buffer at %s (prioritized=%s)", endpoint, prioritized)
        return RemoteReplayBufferAdapter(client, prioritized=prioritized)

    if getattr(config.replay_buffer, "priority_alpha", 0) > 0:
        return PrioritizedTrajectoryBuffer(
            capacity=config.replay_buffer.capacity,
            observation_shape=observation_shape,
            num_actions=num_actions,
            alpha=config.replay_buffer.priority_alpha,
            max_trajectory_length=max_trajectory_length,
        )

    logger.info(
        "Replay buffer initialized: capacity=%s, obs_shape=%s, num_actions=%s, max_traj_length=%s",
        config.replay_buffer.capacity,
        observation_shape,
        num_actions,
        max_trajectory_length,
    )
    return TrajectoryBuffer(
        capacity=config.replay_buffer.capacity,
        observation_shape=observation_shape,
        num_actions=num_actions,
        max_trajectory_length=max_trajectory_length,
    )


def build_parameter_publisher_client(config: DictConfig):
    """
    Build parameter publisher client (and optional server) based on config.

    Returns (client, server_or_none). When remote_enabled=false, both are None.
    """
    pub_cfg = getattr(config, "publisher", None)
    if pub_cfg is None or not getattr(pub_cfg, "remote_enabled", False):
        return None, None
    endpoint = getattr(pub_cfg, "rpc_endpoint", "") or ""
    if not endpoint:
        raise ValueError("publisher.remote_enabled=true but no publisher.rpc_endpoint provided")
    client = GrpcParameterPublisherClient(endpoint)
    return client, None


logging_level = logging.WARNING if CLI_VERBOSITY <= 0 else logging.INFO
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)
actor_logger = logging.getLogger('open_spiel.python.algorithms.muzero_jax.self_play.actor')
actor_logger.setLevel(logging.INFO if CLI_VERBOSITY >= 3 else logging.WARNING)



@dataclass
class OrchestrationConfig:
    """Configuration for orchestration-specific settings."""
    sequential_training: bool = True
    concurrent: bool = False
    training_phase_steps: int = 10
    selfplay_phase_episodes: int = 5
    max_episodes_without_training: int = 100
    max_training_steps_without_episodes: int = 50
    actor_queue_capacity: int = 128
    learner_idle_sleep_ms: int = 20


@dataclass
class TrajectoryPacket:
    """Container for trajectories emitted by actor workers."""
    trajectory: Dict[str, Any]
    worker_id: int
    actor_type: str
    timestamp: float


class ActorWorker(threading.Thread):
    """Background worker that continuously generates trajectories."""

    def __init__(
        self,
        worker_id: int,
        trajectory_queue: "queue.Queue[TrajectoryPacket]",
        stop_event: threading.Event,
        bootstrap_event: threading.Event,
        muzero_actor: Actor,
        bootstrap_actor: Optional[BootstrapActorType],
        checkpoint_dir: str,
        checkpoint_sync_interval: int,
        rng_seed: int,
        error_queue: "queue.Queue[BaseException]",
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.trajectory_queue = trajectory_queue
        self.stop_event = stop_event
        self.bootstrap_event = bootstrap_event
        self.muzero_actor = muzero_actor
        self.bootstrap_actor = bootstrap_actor
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_sync_interval = checkpoint_sync_interval
        self.error_queue = error_queue
        self._rng_key = jax.random.PRNGKey(rng_seed + worker_id)
        self._episodes_since_sync = 0
        self._thread_logger = logging.getLogger(f"{__name__}.ActorWorker{worker_id}")

    def run(self):
        try:
            while not self.stop_event.is_set():
                actor_type, actor = self._select_actor()
                if actor is None:  # pragma: no cover - unreachable in normal flow
                    time.sleep(0.01)
                    continue

                if actor_type == "muzero" and hasattr(actor, "refresh_params"):
                    actor.refresh_params()

                self._rng_key, episode_key = jax.random.split(self._rng_key)
                try:
                    trajectory = actor.play_episode(episode_key)
                except Exception as exc:  # pragma: no cover - defensive
                    self._thread_logger.error(
                        "Actor worker %s failed to generate trajectory: %s",
                        self.worker_id,
                        exc,
                    )
                    self.error_queue.put(exc)
                    continue

                packet = TrajectoryPacket(
                    trajectory=trajectory,
                    worker_id=self.worker_id,
                    actor_type=actor_type,
                    timestamp=time.time(),
                )

                self._put_packet(packet)
                self._maybe_refresh_checkpoint(actor_type)
        except Exception as exc:  # pragma: no cover
            self.error_queue.put(exc)

    def _select_actor(self):
        if self.bootstrap_event.is_set() and self.bootstrap_actor is not None:
            return "bootstrap", self.bootstrap_actor
        return "muzero", self.muzero_actor

    def _put_packet(self, packet: TrajectoryPacket):
        while not self.stop_event.is_set():
            try:
                self.trajectory_queue.put(packet, timeout=0.5)
                return
            except queue.Full:
                continue

    def _maybe_refresh_checkpoint(self, actor_type: str):
        if actor_type != "muzero":
            return
        if self.checkpoint_sync_interval <= 0:
            return
        self._episodes_since_sync += 1
        if self._episodes_since_sync % self.checkpoint_sync_interval == 0:
            try:
                self.muzero_actor.maybe_load_latest_parameters(self.checkpoint_dir)
            except Exception as exc:  # pragma: no cover
                self._thread_logger.warning(
                    "Actor worker %s failed to refresh checkpoint: %s",
                    self.worker_id,
                    exc,
                )


class LearnerWorker(threading.Thread):
    """Background worker that continuously trains the learner."""

    def __init__(
        self,
        orchestrator: "MuZeroOrchestrator",
        stop_event: threading.Event,
        idle_sleep_s: float,
        error_queue: "queue.Queue[BaseException]",
    ):
        super().__init__(daemon=True)
        self.orchestrator = orchestrator
        self.stop_event = stop_event
        self.idle_sleep_s = max(idle_sleep_s, 0.0)
        self.error_queue = error_queue

    def run(self):
        try:
            while not self.stop_event.is_set():
                if not self.orchestrator.should_continue_training():
                    break  # pragma: no cover - defensive

                metrics = self.orchestrator._perform_training_step()
                if metrics is None:
                    time.sleep(self.idle_sleep_s)
                    continue

                self.orchestrator._maybe_log_training_metrics(metrics)
        except Exception as exc:  # pragma: no cover
            self.error_queue.put(exc)


class MuZeroOrchestrator:
    """
    Main orchestrator for MuZero JAX training.
    
    Coordinates between actors (self-play) and learners, manages TPU/GPU resource
    allocation, and handles checkpointing and logging.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the orchestrator with Hydra configuration."""
        self.config = config
        concurrent_flag = bool(getattr(config.resource_management, 'concurrent', False))
        sequential_flag = getattr(config.resource_management, 'sequential_training', True)
        sequential_mode = sequential_flag and not concurrent_flag
        self.orchestration_config = OrchestrationConfig(
            sequential_training=sequential_mode,
            concurrent=concurrent_flag,
            training_phase_steps=config.resource_management.training_phase_steps,
            selfplay_phase_episodes=config.resource_management.selfplay_phase_episodes,
            actor_queue_capacity=getattr(
                config.resource_management, 'actor_queue_capacity', 128
            ),
            learner_idle_sleep_ms=getattr(
                config.resource_management, 'learner_idle_sleep_ms', 20
            ),
        )
        
        # Set up directories
        self.setup_directories()
        
        # Set random seed
        if hasattr(config.exp_config, 'seed'):
            self.set_random_seed(config.exp_config.seed)
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize training state (before setup_components so checkpoint loading can override)
        self.training_step = 0
        self.total_episodes = 0
        self.start_time = time.time()
        self._last_eval_step = 0
        self._buffer_lock = threading.Lock()
        self._inference_clients: List[Any] = []
        self._replay_clients: List[Any] = []
        self._remote_replay = False
        self._parameter_publisher = None
        self._parameter_client = None
        self._replay_clients: List[Any] = []
        self._launched_replay_server = None
        self._launched_publisher_server = None
        self._launched_inference_server = None
        
        # Initialize components (checkpoint loading may update training_step)
        self.setup_components()
    
    def _register_inference_client(self, client) -> None:
        """Track inference clients for cleanup."""
        if client is None:
            return
        self._inference_clients.append(client)

    def _register_replay_client(self, client) -> None:
        """Track replay clients for cleanup."""
        if client is None:  # pragma: no cover - defensive
            return
        self._replay_clients.append(client)

    def _buffer_size(self) -> int:
        """Return current buffer size with locking for thread safety."""
        with self._buffer_lock:
            return len(self.replay_buffer)

    def _add_trajectory_to_buffer(self, trajectory: Dict[str, Any]) -> None:
        """Add a trajectory to the replay buffer under lock."""
        with self._buffer_lock:
            self.replay_buffer.add_trajectory(trajectory)

    def _maybe_log_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics when verbosity or interval thresholds are met."""
        log_interval = getattr(self.config.output, 'log_interval', 0) or 1
        if get_cli_verbosity() >= 1 or self.training_step % log_interval == 0:
            self.log_training_metrics(metrics)

    def _perform_training_step(self) -> Optional[Dict[str, Any]]:
        """Run a single training step. Returns metrics or None if skipped."""
        min_required = self._min_buffer_before_training()
        with self._buffer_lock:
            current_buffer = len(self.replay_buffer)
        if current_buffer < min_required:
            return None

        self.rng_key, sample_key = jax.random.split(self.rng_key)

        sampled_indices = None
        importance_weights = None
        is_prioritized = isinstance(self.replay_buffer, PrioritizedTrajectoryBuffer) or hasattr(
            self.replay_buffer, "update_priorities"
        ) or hasattr(self.replay_buffer, "update_priorities_by_ids")
        try:
            if hasattr(self.replay_buffer, "sample_batch_with_ids"):
                with self._buffer_lock:
                    sample_out = self.replay_buffer.sample_batch_with_ids(
                        self.config.training.batch_size,
                        rng_key=sample_key,
                    )

                # Robust unpacking for 2, 3, or 4 return values
                traj_ids = None
                sampled_indices = None
                importance_weights = None

                if len(sample_out) == 4:
                    # (trajectories, traj_ids, buffer_indices, weights)
                    trajectory_list, traj_ids, sampled_indices, importance_weights = sample_out
                elif len(sample_out) == 3:
                    # (trajectories, ids, weights) - IDs are used as indices and traj_ids (e.g. Remote)
                    trajectory_list, sampled_indices, importance_weights = sample_out
                    traj_ids = sampled_indices
                elif len(sample_out) == 2:
                    # (trajectories, ids)
                    trajectory_list, sampled_indices = sample_out
                    traj_ids = sampled_indices
                else:
                    raise ValueError(f"Unexpected sample_batch_with_ids return length: {len(sample_out)}")
            elif is_prioritized:
                with self._buffer_lock:
                    sample_out = self.replay_buffer.sample_batch(
                        self.config.training.batch_size,
                        rng_key=sample_key,
                    )
                # Normalize outputs: adapter may return 2 or 3 elements.
                if isinstance(sample_out, tuple):
                    if len(sample_out) == 3:
                        trajectory_list, sampled_indices, importance_weights = sample_out
                    elif len(sample_out) == 2:
                        trajectory_list, sampled_indices = sample_out
                        importance_weights = None
                    elif len(sample_out) == 1:  # pragma: no cover - defensive adapter normalization
                        trajectory_list = sample_out[0]
                        sampled_indices = None
                        importance_weights = None
                    else:
                        raise ValueError(f"Unexpected sample_batch output length: {len(sample_out)}")  # pragma: no cover
                else:
                    trajectory_list = sample_out
            else:
                with self._buffer_lock:
                    trajectory_list = self.replay_buffer.sample_batch(
                        self.config.training.batch_size,
                        rng_key=sample_key,
                    )
        except ValueError:
            # Buffer may be too small; defer training step
            return None  # pragma: no cover - rare

        if not trajectory_list:
            return None

        batch = self._convert_trajectories_to_batch(
            trajectory_list,
            sampled_indices,
            importance_weights,
            traj_ids,
        )
        metrics = self.learner.train_step(batch)
        self.training_step += 1
        self._maybe_publish_params()
        self._update_priorities(sampled_indices, metrics)
        return metrics

    def _update_priorities(self, sampled_indices, metrics: Optional[Dict[str, Any]]) -> None:
        """Update replay priorities when prioritized replay is enabled."""
        prioritized = isinstance(self.replay_buffer, PrioritizedTrajectoryBuffer) or hasattr(
            self.replay_buffer, "update_priorities"
        ) or hasattr(self.replay_buffer, "update_priorities_by_ids")
        if not prioritized:
            return
        if sampled_indices is None or metrics is None:
            return
        if 'priorities' not in metrics:
            return
        try:
            new_priorities = jnp.array(metrics['priorities'])
            new_priorities = jnp.maximum(new_priorities, self.muzero_config.min_priority)
            with self._buffer_lock:
                if hasattr(self.replay_buffer, "update_priorities"):
                    self.replay_buffer.update_priorities(sampled_indices, new_priorities)
                elif hasattr(self.replay_buffer, "update_priorities_by_ids"):  # pragma: no cover - adapter path
                    self.replay_buffer.update_priorities_by_ids(sampled_indices, new_priorities)
            logger.debug(
                "Updated %s priorities, mean priority: %.6f",
                len(sampled_indices),
                float(jnp.mean(new_priorities)),
            )
        except Exception as exc:
            logger.warning(f"Failed to update priorities: {exc}")

    def _maybe_publish_params(self, force: bool = False):
        """Publish parameters to inference clients (local publisher)."""
        if self._parameter_client is None:
            return
        publish_interval = getattr(self.config.publisher, "publish_interval", 0) or 0
        if not force and publish_interval > 0 and self.training_step % publish_interval != 0:
            return
        try:
            if not hasattr(self.network, "get_variables"):
                return
            params = self.network.get_variables()
            self._parameter_client.publish(params, step=self.training_step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to publish parameters: %s", exc)
    
    def _evaluation_enabled(self) -> bool:
        """Safely determine if evaluation is enabled."""
        eval_cfg = getattr(self.config, "evaluation", None)
        if eval_cfg is None:
            return False
        enabled_value = getattr(eval_cfg, "enabled", None)
        if enabled_value is None:
            return True
        return bool(enabled_value)

    def _maybe_run_periodic_evaluation(self):
        """Run evaluation when the configured interval elapses."""
        if not self._evaluation_enabled():
            return
        if self.training_step <= 0:
            return
        interval = getattr(self.config.evaluation, 'interval', 0) or 0
        if interval <= 0:
            return
        if self.training_step % interval != 0:
            return
        if self._last_eval_step == self.training_step:
            return
        self._last_eval_step = self.training_step
        self.run_evaluation()
    
    def _should_log_episode_metrics(self) -> bool:
        """Decide whether to log episode metrics for self-play."""
        interval = getattr(self.config.output, "log_interval", 0) or 0
        if get_cli_verbosity() >= 2:
            return True
        if interval <= 0:
            return False
        return (self.total_episodes % interval) == 0
        
    def setup_directories(self):
        """Set up output directories."""
        self.save_path = Path(self.config.output.save_path)
        self.checkpoint_dir = self.save_path / "checkpoints"
        self.logs_dir = self.save_path / "logs"
        
        # Create directories
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Save path: {self.save_path}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        
    def set_random_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        # JAX uses a different random system, we'll set keys as needed
        self.rng_key = jax.random.PRNGKey(seed)
        logger.info(f"Set random seed to {seed}")
        
    def setup_logging(self):
        """Set up Weights & Biases logging."""
        if self.config.wandb.enabled:
            # Create tags
            tags = list(self.config.wandb.tags) if self.config.wandb.tags else []
            tags.append(self.config.game.name)
            tags.append(self.config.exp_config.tag)
            
            # Initialize wandb
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=f"{self.config.game.name}-{self.config.exp_config.tag}",
                tags=tags,
                notes=self.config.wandb.notes,
                config=OmegaConf.to_container(self.config, resolve=True),
                dir=str(self.logs_dir)
            )
            logger.info("Initialized Weights & Biases logging")
        else:
            logger.info("Weights & Biases logging disabled")
            
    def setup_components(self):
        """Initialize all MuZero components."""
        logger.info("Setting up MuZero components...")
        
        # Create MuZero configuration from Hydra config
        self.muzero_config = self.create_muzero_config_from_hydra_config()
        
        # Initialize game wrapper to get observation shape
        self.game_wrapper = GameWrapper(self.config.game.name)
        observation_shape = self.game_wrapper.observation_shape
        num_actions = self.game_wrapper.num_distinct_actions()
        
        logger.info(f"Game: {self.config.game.name}")
        logger.info(f"Observation shape: {observation_shape}")
        logger.info(f"Number of actions: {num_actions}")
        
        # Initialize network
        self.setup_network(observation_shape, num_actions)

        # Auto-launch local services if remote is enabled but endpoint not provided.
        self._maybe_launch_local_services()
        
        # Initialize replay buffer
        self.setup_replay_buffer(observation_shape, num_actions)

        # Parameter publisher client (remote only; local path is no-op)
        self._parameter_client, _ = build_parameter_publisher_client(self.config)
        
        # Initialize learner
        self.setup_learner()
        
        # Initialize actors
        self.setup_actors()
        
        logger.info("All components initialized successfully")

    def _write_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Persist metrics alongside checkpoints for eval scripts."""
        try:
            ckpt_dir = Path(self.config.output.save_path) / "checkpoints" / str(step) / "metrics"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with (ckpt_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write metrics: %s", exc)

    def _is_tpu(self) -> bool:
        try:
            devices = jax.devices()
            return bool(devices) and devices[0].platform == "tpu"
        except Exception:  # pragma: no cover - defensive
            return False

    def _maybe_launch_local_services(self):
        """Start local gRPC services when remote flags are on but endpoints are empty."""
        # Replay service
        rb_cfg = getattr(self.config, "replay_buffer", None)
        if rb_cfg and getattr(rb_cfg, "remote_enabled", False) and not getattr(rb_cfg, "rpc_endpoint", ""):
            from open_spiel.python.algorithms.muzero_jax.services.replay_service import (
                InMemoryReplayService,
                GrpcReplayServer,
            )

            backend = InMemoryReplayService(
                capacity=int(getattr(rb_cfg, "capacity", 2000)),
                alpha=float(getattr(rb_cfg, "priority_alpha", 0.0)),
            )
            server = GrpcReplayServer(backend)
            server.start()
            rb_cfg.rpc_endpoint = server.endpoint
            self._launched_replay_server = server
            logger.info("Auto-launched local replay server at %s", server.endpoint)

        # Parameter publisher
        pub_cfg = getattr(self.config, "publisher", None)
        if pub_cfg and getattr(pub_cfg, "remote_enabled", False) and not getattr(pub_cfg, "rpc_endpoint", ""):
            from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import (
                GrpcParameterPublisherServer,
                LocalParameterPublisher,
            )

            backend = LocalParameterPublisher()
            server = GrpcParameterPublisherServer(backend)
            server.start()
            pub_cfg.rpc_endpoint = server.endpoint
            self._launched_publisher_server = server
            logger.info("Auto-launched local parameter publisher at %s", server.endpoint)

        # Inference remains local-only; no remote endpoints handled here.
        
    def _min_buffer_before_training(self) -> int:
        """Return the minimum number of transitions required before training."""
        # For remote buffers, size() is cached in adapter; rely on start_transitions/batch_size thresholds.
        return max(
            int(self.muzero_config.start_transitions),
            int(self.config.training.batch_size),
            int(getattr(self.config.replay_buffer, "min_size_to_sample", 0)),
        )
        
    def create_muzero_config_from_hydra_config(self) -> MuZeroConfig:
        """Convert Hydra config to MuZeroConfig."""
        # Start with game-specific defaults
        muzero_config = create_muzero_config_for_game(self.config.game.name)
        
        # Override with Hydra config values
        config_overrides = {
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'training_steps': self.config.training.training_steps,
            'start_transitions': self.config.training.start_transitions,  # Add start_transitions
            'discount_factor': self.config.training.discount,  # Note: field name is discount_factor
            'num_unroll_steps': self.config.training.num_unroll_steps,
            'td_steps': self.config.training.td_steps,
            'value_loss_weight': self.config.training.value_loss_weight,
            'policy_loss_weight': self.config.training.policy_loss_weight,
            'reward_loss_weight': self.config.training.reward_loss_weight,
            'l2_weight': self.config.training.l2_regularization,  # Note: field name is l2_weight
            'num_simulations': self.config.mcts.num_simulations,  # Note: field name is num_simulations
            'temperature_init': self.config.mcts.temperature_init,
            'temperature_final': self.config.mcts.temperature_final,
            'temperature_decay_steps': self.config.mcts.temperature_decay_steps,
            'dirichlet_alpha': self.config.mcts.dirichlet_alpha,
            'explore_frac': self.config.mcts.exploration_fraction,  # Note: field name is explore_frac
        }
        
        # Filter overrides to only include fields that exist in MuZeroConfig
        valid_overrides = {}
        for key, value in config_overrides.items():
            if hasattr(muzero_config, key):
                valid_overrides[key] = value
                
        # Use dataclasses.replace to create new config with overrides
        return dataclasses.replace(muzero_config, **valid_overrides)
        
    def setup_network(self, observation_shape, num_actions):
        """Initialize the MuZero network."""
        # Create network configuration
        # OpenSpiel games are always state-based, never image-based (following EfficientZeroV2 approach)
        self.network_config = create_network_config_from_muzero_config(
            self.muzero_config, 
            observation_shape, 
            num_actions,
            use_image_observation=False,  # OpenSpiel games are state-based, not image-based
        )
        
        # Import the network definitions
        from open_spiel.python.algorithms.muzero_jax.models.network import (
            RepresentationNetwork, DynamicsNetwork, PredictionNetwork, 
            RewardNetwork, ProjectionNetwork
        )
        
        # Initialize network with random parameters
        self.rng_key, network_key = jax.random.split(self.rng_key)
        rngs = nnx.Rngs(params=network_key)
        
        self.network = MuZeroNetwork(
            representation_network_def=RepresentationNetwork,
            dynamics_network_def=DynamicsNetwork,
            prediction_network_def=PredictionNetwork,
            reward_network_def=RewardNetwork,
            projection_network_def=ProjectionNetwork if self.muzero_config.use_projection else None,
            config=self.network_config,
            rngs=rngs
        )
        
        logger.info(f"Network initialized with config: {self.network_config}")
        
    def setup_replay_buffer(self, observation_shape, num_actions):
        """Initialize the replay buffer."""
        self.replay_buffer = build_replay_buffer(
            self.config,
            observation_shape,
            num_actions,
            self.game_wrapper._game if hasattr(self.game_wrapper, "_game") else None,
            self._register_replay_client,
        )
        self._remote_replay = isinstance(self.replay_buffer, RemoteReplayBufferAdapter)
            
    def setup_learner(self):
        """Initialize the learner."""
        # Create learner with correct constructor arguments
        self.rng_key, learner_key = jax.random.split(self.rng_key)
        
        # Set checkpoint_dir in config for learner (MuZeroConfig is frozen, so use replace)
        learner_config = dataclasses.replace(
            self.muzero_config,
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        self.learner = Learner(
            model=self.network,
            optimizer_def=None,  # Will use default from config
            config=learner_config,
            rng_key=learner_key
        )
        
        # Try to load existing checkpoint via the learner's manager first
        checkpoint_loaded = False
        if self.learner.checkpoint_manager is not None:
            latest_step = self.learner.checkpoint_manager.latest_step()
            if latest_step is not None:
                logger.info(f"Checkpoint manager reports latest step {latest_step}, attempting restore.")
                checkpoint_loaded = self.learner.load_checkpoint()
        
        # Fallback to filesystem scan (legacy checkpoints saved without manager metadata)
        if not checkpoint_loaded:
            latest_checkpoint = get_latest_checkpoint(str(self.checkpoint_dir))
            if latest_checkpoint:
                logger.info(f"Loading checkpoint from path: {latest_checkpoint}")
                checkpoint_loaded = self.learner.load_checkpoint(latest_checkpoint)
        
        if checkpoint_loaded:
            self.training_step = self.learner.num_training_steps
            logger.info(f"Resumed from training step: {self.training_step}")
        else:
            logger.info("No existing checkpoint found, starting from scratch")
            
    def setup_actors(self):
        """Initialize the actors for self-play."""
        self.actors = []
        num_actors = self.config.actors.num_actors
        inference_cfg = getattr(self.config, "inference", None)
        
        bootstrap_config = bootstrap_module.BootstrapConfig(
            num_simulations=self.muzero_config.num_simulations,
            c_puct=getattr(self.config.bootstrap, 'c_puct', 1.25),
            n_step_return=self.muzero_config.td_steps,
            discount_factor=self.muzero_config.discount_factor,
        )
        self.bootstrap_config = bootstrap_config
        self.bootstrap_actor = self._make_bootstrap_actor()
        
        # Create MuZero actors for later use (when network is trained)
        for i in range(num_actors):
            inference_client = build_inference_client(self.network, inference_cfg)
            self._register_inference_client(inference_client)
            if self._parameter_client is not None:
                inference_client = ParameterRefreshingInferenceClient(inference_client, self._parameter_client)
            actor = Actor(
                network=self.network,
                game_wrapper=GameWrapper(self.config.game.name),
                replay_buffer=self.replay_buffer,
                config=self.muzero_config,
                num_simulations=self.muzero_config.num_simulations,
                max_num_considered_actions=self.game_wrapper.num_distinct_actions(),
                gumbel_scale=1.0,
                n_step_return=self.muzero_config.td_steps,
                discount_factor=self.muzero_config.discount_factor,
                inference_client=inference_client,
            )
            self.actors.append(actor)
            
        # Track whether we're in bootstrap phase
        self.use_bootstrap = getattr(self.config.bootstrap, 'enabled', True)
        self.bootstrap_episodes_generated = 0
        self.min_bootstrap_episodes = getattr(self.config.bootstrap, 'min_episodes', 50)
        
        logger.info(f"Initialized bootstrap actor and {num_actors} MuZero actors")
        logger.info(f"Will use bootstrap actor for first {self.min_bootstrap_episodes} episodes")

    def _make_bootstrap_actor(self) -> BootstrapActorType:
        """Create a new bootstrap actor instance."""
        bootstrap_cls = getattr(bootstrap_module, "BootstrapActor")
        return bootstrap_cls(
            game_wrapper=GameWrapper(self.config.game.name),
            replay_buffer=self.replay_buffer,
            config=self.bootstrap_config,
        )
        
    def run_training_phase(self) -> Dict[str, Any]:
        """Run a training phase and return metrics."""
        metrics_list = []
        min_required = self._min_buffer_before_training()
        skipped_due_to_buffer = False

        for _ in range(self.orchestration_config.training_phase_steps):
            metrics = self._perform_training_step()
            if metrics is None:
                if not skipped_due_to_buffer:
                    current_buffer = self._buffer_size()
                    logger.info(
                        "Buffer size (%s) below minimum for training (%s), skipping training",
                        current_buffer,
                        min_required,
                    )
                    skipped_due_to_buffer = True
                break

            metrics_list.append(metrics)
            self._maybe_log_training_metrics(metrics)
            
            # Save checkpoint
            if self.training_step % self.config.output.checkpoint_interval == 0:
                checkpoint_path = self.learner.save_checkpoint(force_save=True)
                if checkpoint_path:
                    logger.info(f"Saved checkpoint: {checkpoint_path}")  # pragma: no cover
                
            # Check if training is complete
            if self.training_step >= self.muzero_config.training_steps:
                break
                
        # Aggregate metrics
        if metrics_list:
            aggregated_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    aggregated_metrics[key] = np.mean(values)
            aggregated_metrics['steps_trained'] = len(metrics_list)
            # Persist metrics alongside checkpoints for downstream eval tools.
            self._write_metrics(self.training_step, aggregated_metrics)
            return aggregated_metrics
        else:
            return {'steps_trained': 0}
            
    def _convert_trajectories_to_batch(self, trajectories: List[Dict], indices: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None, traj_ids: Optional[np.ndarray] = None) -> Dict:
        """Convert list of trajectories to batched format expected by trainer.
        
        Args:
            trajectories: List of trajectory dictionaries
            indices: Buffer indices for priority replay (optional)
            weights: Importance sampling weights for priority replay (optional)
            
        Returns:
            Batch dictionary with all necessary fields for training
        """
        if not trajectories:
            raise ValueError("Cannot create batch from empty trajectory list")
        
        # Get dimensions
        batch_size = len(trajectories)
        expected_length = self.muzero_config.num_unroll_steps + 1
        observation_shape = trajectories[0]['observations'][0].shape
        num_actions = self.game_wrapper.num_distinct_actions()
        max_length = expected_length
        
        # Initialize batch arrays
        batch_observations = np.zeros((batch_size, max_length, *observation_shape))
        batch_actions = np.zeros((batch_size, max_length), dtype=np.int32)
        batch_target_rewards = np.zeros((batch_size, max_length))  
        batch_target_values = np.zeros((batch_size, max_length))
        batch_target_policies = np.zeros((batch_size, max_length, num_actions))
        batch_masks = np.zeros((batch_size, max_length), dtype=np.float32)
        batch_target_search_values = np.zeros((batch_size, max_length))
        batch_target_sarsa_values = np.zeros((batch_size, max_length))
        
        # Prepare for bootstrapping SARSA targets
        bootstrap_requests = [] # list of (batch_idx, time_idx, obs)
        gamma = self.muzero_config.discount_factor
        td_steps = self.muzero_config.td_steps

        # First pass: fill observations and collect bootstrap requests
        for i, trajectory in enumerate(trajectories):
            traj_length = len(trajectory['actions'])
            effective_length = min(traj_length, max_length)
            rewards = trajectory['rewards']
            
            # Fill observations
            for j in range(max_length):
                if j < len(trajectory['observations']) and j < max_length:
                    batch_observations[i, j] = trajectory['observations'][j]
                elif len(trajectory['observations']) > 0:
                    batch_observations[i, j] = trajectory['observations'][min(j, len(trajectory['observations']) - 1)]
            
            # Fill basic targets
            batch_actions[i, :effective_length] = trajectory['actions'][:effective_length]
            batch_target_rewards[i, :effective_length] = rewards[:effective_length]
            batch_target_values[i, :effective_length] = trajectory['value_targets'][:effective_length]
            
            # Search values
            if 'target_search_value' in trajectory:
                batch_target_search_values[i, :effective_length] = trajectory['target_search_value'][:effective_length]
            else:
                batch_target_search_values[i, :effective_length] = trajectory['value_targets'][:effective_length]

            # Policies
            for j in range(effective_length):
                batch_target_policies[i, j] = trajectory['policy_targets'][j]

            batch_masks[i, :effective_length] = 1.0

            # Collect bootstrap requests for SARSA
            for t in range(effective_length):
                bootstrap_t = t + td_steps
                if bootstrap_t < len(rewards): # Within trajectory
                    obs = trajectory['observations'][bootstrap_t]
                    bootstrap_requests.append((i, t, obs))

        # Run inference for bootstrapping if needed
        bootstrap_values_map = {}
        if bootstrap_requests:
            obs_batch = np.stack([req[2] for req in bootstrap_requests])
            obs_batch_jax = jnp.array(obs_batch)

            # Run inference
            _, _, values, _, _, _ = self.network.initial_inference(obs_batch_jax, training=False)

            # Convert to scalar
            if values.ndim > 1 and values.shape[-1] > 1:
                 values = losses_lib.support_to_scalar(values, self.muzero_config.support_min, self.muzero_config.support_max, values.shape[-1])
            elif self.muzero_config.value_loss_type == "symlog":
                 values = losses_lib.symexp(values, self.muzero_config.symlog_base)
                 if values.ndim > 1:
                     values = jnp.squeeze(values, axis=-1)
            elif values.ndim > 1:
                 values = jnp.squeeze(values, axis=-1)

            values_np = np.array(values)
            for idx, (i, t, _) in enumerate(bootstrap_requests):
                bootstrap_values_map[(i, t)] = values_np[idx]

        # Second pass: compute SARSA targets
        for i, trajectory in enumerate(trajectories):
            effective_length = min(len(trajectory['actions']), max_length)
            rewards = trajectory['rewards']

            for t in range(effective_length):
                g_val = 0.0
                for k in range(td_steps):
                    if t + k < len(rewards):
                        g_val += (gamma ** k) * rewards[t + k]
                    else:
                        break

                # Bootstrap
                if (i, t) in bootstrap_values_map:
                    g_val += (gamma ** td_steps) * bootstrap_values_map[(i, t)]

                batch_target_sarsa_values[i, t] = g_val

        # For value_target="max" or default fallback, populate target_value with max(search, sarsa)
        # This prevents training on zeros if the config falls through to target_value.
        batch_target_values = np.maximum(batch_target_search_values, batch_target_sarsa_values)

        # Create base batch dictionary
        batch = {
            'observation': jnp.array(batch_observations),
            'action': jnp.array(batch_actions),
            'target_reward': jnp.array(batch_target_rewards),
            'target_value': jnp.array(batch_target_values),
            'target_policy': jnp.array(batch_target_policies),
            'game_history_mask': jnp.array(batch_masks),
            'target_search_value': jnp.array(batch_target_search_values),
            'target_sarsa_value': jnp.array(batch_target_sarsa_values),
            # Use raw trajectory counts (unscaled) for collected_transitions
            'collected_transitions': jnp.array(
                getattr(self.replay_buffer, "_total_transitions", self._buffer_size() * self.muzero_config.trajectory_size)
            ),
        }
        
        # Add priority replay fields
        if indices is not None:
            batch['indices'] = jnp.array(indices)

        # Use start_transition_index for accurate age, fallback to scaled indices
        batch_sample_indices = np.zeros(batch_size, dtype=np.int64)
        for i, trajectory in enumerate(trajectories):
            if 'start_transition_index' in trajectory:
                batch_sample_indices[i] = trajectory['start_transition_index']
            elif traj_ids is not None:
                # Fallback to trajectory ID scaling if start index missing (e.g. old data or remote)
                batch_sample_indices[i] = int(traj_ids[i]) * self.muzero_config.trajectory_size
            else:
                batch_sample_indices[i] = 0

        batch['sample_indices'] = jnp.array(batch_sample_indices)

        if weights is not None:
            batch['weights'] = jnp.array(weights)
            
        return batch

    def run_selfplay_phase(self) -> Dict[str, Any]:
        """Run a self-play phase and return metrics."""
        episodes_played = 0
        total_episode_length = 0
        
        # Determine which actor to use
        if self.use_bootstrap:
            # Check if we should transition: need both min episodes AND enough transitions for training
            min_required = self._min_buffer_before_training()
            if (self.bootstrap_episodes_generated >= self.min_bootstrap_episodes and 
                self._buffer_size() >= min_required):
                logger.info(f"Transitioning from bootstrap to MuZero actors after "
                           f"{self.bootstrap_episodes_generated} bootstrap episodes and "
                           f"{self._buffer_size()} transitions in buffer")
                self.use_bootstrap = False
                
        if self.use_bootstrap:
            # Use bootstrap actor for initial trajectory generation
            logger.info(f"Using bootstrap actor (plain MCTS) for episode generation "
                       f"({self.bootstrap_episodes_generated}/{self.min_bootstrap_episodes})")
            
            for _ in range(self.orchestration_config.selfplay_phase_episodes):
                # Continue bootstrap until we have enough transitions for training
                # (The transition check above will handle the switch to MuZero actors)
                    
                # Play episode with bootstrap actor
                self.rng_key, episode_key = jax.random.split(self.rng_key)
                episode_data = self.bootstrap_actor.play_episode(episode_key)
                
                # Add episode data to replay buffer
                self._add_trajectory_to_buffer(episode_data)
                
                # Extract episode length from episode data
                episode_length = len(episode_data.get('observations', []))
                episodes_played += 1
                total_episode_length += episode_length
                self.total_episodes += 1
                self.bootstrap_episodes_generated += 1
                
                # Log episode metrics
                if self._should_log_episode_metrics():
                    self.log_episode_metrics(episode_length)
                    
        else:
            # Use MuZero actors with trained network
            logger.info("Using MuZero actors with neural network guidance")
            
            # Update actor parameters from latest checkpoint
            for actor in self.actors:
                actor.maybe_load_latest_parameters(str(self.checkpoint_dir))
                
            for _ in range(self.orchestration_config.selfplay_phase_episodes):
                # Round-robin through actors
                actor = self.actors[episodes_played % len(self.actors)]
                
                # Play episode
                self.rng_key, episode_key = jax.random.split(self.rng_key)
                episode_data = actor.play_episode(episode_key)
                
                # Add episode data to replay buffer
                self._add_trajectory_to_buffer(episode_data)
                
                # Extract episode length from episode data
                episode_length = len(episode_data.get('observations', []))
                episodes_played += 1
                total_episode_length += episode_length
                self.total_episodes += 1
                
                # Log episode metrics
                if self._should_log_episode_metrics():
                    self.log_episode_metrics(episode_length)
                
        return {
            'episodes_played': episodes_played,
            'avg_episode_length': total_episode_length / max(episodes_played, 1),
            'buffer_size': self._buffer_size(),
            'using_bootstrap': self.use_bootstrap,
            'bootstrap_episodes_generated': self.bootstrap_episodes_generated
        }
        
    def log_training_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics."""
        log_data = {
            'training_step': self.training_step,
            'total_episodes': self.total_episodes,
            'buffer_size': self._buffer_size(),
            'runtime_hours': (time.time() - self.start_time) / 3600,
            **metrics
        }
        
        if self.config.wandb.enabled:
            wandb.log(log_data, step=self.training_step)
            
        logger.info(f"Step {self.training_step}: {log_data}")
        
    def log_episode_metrics(self, episode_length: int):
        """Log episode metrics."""
        log_data = {
            'episode': self.total_episodes,
            'episode_length': episode_length,
            'buffer_size': self._buffer_size(),
            'training_step': self.training_step,
        }
        
        if self.config.wandb.enabled:
            wandb.log(log_data, step=self.training_step)
            
        logger.info(f"Episode {self.total_episodes}: length={episode_length}, buffer_size={self._buffer_size()}")
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation episodes."""
        if not self.config.evaluation.enabled:
            return {}
            
        logger.info("Running evaluation...")
        
        # Create evaluation actor with deterministic policy
        eval_game_wrapper = GameWrapper(self.config.game.name)
        eval_mcts = MCTS(
            num_simulations=self.muzero_config.num_simulations,
            max_num_considered_actions=eval_game_wrapper.num_distinct_actions(),
            gumbel_scale=1.0
        )
        eval_inference_client = LocalInferenceClient(self.network)
        eval_actor = Actor(
            network=self.network,
            mcts=eval_mcts,
            game_wrapper=eval_game_wrapper,
            replay_buffer=None,  # Don't add to replay buffer
            config=self.muzero_config,
            n_step_return=self.muzero_config.td_steps,
            discount_factor=self.muzero_config.discount_factor,
            inference_client=eval_inference_client,
        )
        
        # Load latest parameters
        eval_actor.maybe_load_latest_parameters(str(self.checkpoint_dir))
        
        # Run evaluation episodes
        episode_lengths = []
        episode_rewards = []
        
        for _ in range(self.config.evaluation.num_episodes):
            self.rng_key, eval_key = jax.random.split(self.rng_key)
            episode_data = eval_actor.play_episode(eval_key)
            episode_length = len(episode_data.get('observations', []))
            episode_lengths.append(episode_length)
            # Note: episode rewards would need to be tracked separately
            
        # Compute evaluation metrics
        eval_metrics = {
            'eval_avg_episode_length': np.mean(episode_lengths),
            'eval_std_episode_length': np.std(episode_lengths),
            'eval_min_episode_length': np.min(episode_lengths),
            'eval_max_episode_length': np.max(episode_lengths),
        }
        
        logger.info(f"Evaluation results: {eval_metrics}")  # pragma: no cover
        
        if self.config.wandb.enabled:
            wandb.log(eval_metrics, step=self.training_step)  # pragma: no cover
        eval_inference_client.close()
        return eval_metrics
        
    def should_continue_training(self) -> bool:
        """Check if training should continue."""
        return self.training_step < self.muzero_config.training_steps

    def _start_actor_workers(
        self,
        trajectory_queue: "queue.Queue[TrajectoryPacket]",
        stop_event: threading.Event,
        bootstrap_event: threading.Event,
        error_queue: "queue.Queue[BaseException]",
    ) -> List[ActorWorker]:
        """Launch actor workers for concurrent orchestration."""
        workers = []
        if not self.actors:
            raise ValueError("At least one MuZero actor is required for concurrent mode")
        num_workers = max(1, int(self.config.actors.num_actors))
        checkpoint_sync = getattr(self.config.actors, 'checkpoint_sync_interval', 0)
        base_seed = getattr(self.config.exp_config, 'seed', 0)

        for worker_id in range(num_workers):
            muzero_actor = self.actors[worker_id % len(self.actors)]
            bootstrap_actor = self._make_bootstrap_actor() if self.use_bootstrap else None
            worker = ActorWorker(
                worker_id=worker_id,
                trajectory_queue=trajectory_queue,
                stop_event=stop_event,
                bootstrap_event=bootstrap_event,
                muzero_actor=muzero_actor,
                bootstrap_actor=bootstrap_actor,
                checkpoint_dir=str(self.checkpoint_dir),
                checkpoint_sync_interval=checkpoint_sync,
                rng_seed=base_seed,
                error_queue=error_queue,
            )
            worker.start()
            workers.append(worker)

        return workers

    def _raise_worker_errors(self, error_queue: "queue.Queue[BaseException]"):
        """Raise any worker errors captured asynchronously."""
        try:
            exc = error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("Asynchronous MuZero worker failed") from exc

    def _update_progress(self, progress):
        if progress is None:
            return
        progress.n = min(self.training_step, self.muzero_config.training_steps)
        progress.set_postfix(buffer=self._buffer_size())
        progress.refresh()

    def _run_sequential(self, progress) -> None:
        """Original sequential orchestration loop."""
        while self.should_continue_training():
            selfplay_metrics = self.run_selfplay_phase()
            if get_cli_verbosity() >= 1:
                logger.info(f"Self-play phase completed: {selfplay_metrics}")

            if self._buffer_size() >= self.muzero_config.start_transitions:
                training_metrics = self.run_training_phase()
                if get_cli_verbosity() >= 1:
                    logger.info(f"Training phase completed: {training_metrics}")
            else:
                if get_cli_verbosity() >= 1:
                    logger.info(f"Skipping training phase, buffer size: {self._buffer_size()}")

            self._maybe_run_periodic_evaluation()
            self._update_progress(progress)

    def _run_concurrent(self, progress) -> None:
        """Concurrent actor/learner orchestration loop."""
        queue_capacity = max(1, int(self.orchestration_config.actor_queue_capacity))
        trajectory_queue: "queue.Queue[TrajectoryPacket]" = queue.Queue(maxsize=queue_capacity)
        stop_event = threading.Event()
        bootstrap_event = threading.Event()
        if self.use_bootstrap:
            bootstrap_event.set()
        error_queue: "queue.Queue[BaseException]" = queue.Queue()

        actor_workers = self._start_actor_workers(
            trajectory_queue=trajectory_queue,
            stop_event=stop_event,
            bootstrap_event=bootstrap_event,
            error_queue=error_queue,
        )

        learner_worker = LearnerWorker(
            orchestrator=self,
            stop_event=stop_event,
            idle_sleep_s=self.orchestration_config.learner_idle_sleep_ms / 1000.0,
            error_queue=error_queue,
        )
        learner_worker.start()

        try:
            while self.should_continue_training():
                self._raise_worker_errors(error_queue)
                try:
                    packet = trajectory_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self._add_trajectory_to_buffer(packet.trajectory)
                self.total_episodes += 1

                if packet.actor_type == "bootstrap":
                    self.bootstrap_episodes_generated += 1
                    min_required = self._min_buffer_before_training()
                    if (
                        bootstrap_event.is_set()
                        and self.bootstrap_episodes_generated >= self.min_bootstrap_episodes
                        and self._buffer_size() >= min_required
                    ):
                        bootstrap_event.clear()
                        self.use_bootstrap = False
                        logger.info(
                            "Transitioning from bootstrap to MuZero actors after %s bootstrap episodes "
                            "and %s transitions in buffer",
                            self.bootstrap_episodes_generated,
                            self._buffer_size(),
                        )

                episode_length = len(packet.trajectory.get('observations', []))
                if self._should_log_episode_metrics():
                    self.log_episode_metrics(episode_length)

                trajectory_queue.task_done()
                self._maybe_run_periodic_evaluation()
                self._update_progress(progress)

        finally:
            stop_event.set()
            learner_worker.join(timeout=5.0)
            for worker in actor_workers:
                worker.join(timeout=5.0)
            self._raise_worker_errors(error_queue)

    def run(self):
        """Main training loop."""
        if get_cli_verbosity() >= 1:
            logger.info("Starting MuZero JAX training...")
            logger.info(f"Configuration: {OmegaConf.to_yaml(self.config)}")

        progress = None
        if get_cli_verbosity() <= 0 and tqdm is not None:
            progress = tqdm(total=self.muzero_config.training_steps, desc="Training", unit="step")

        if (
            not self.orchestration_config.sequential_training
            and not self.orchestration_config.concurrent
        ):
            logger.warning(
                "Concurrent mode requested but concurrency workers are disabled; falling back to sequential execution."
            )
            self.orchestration_config.sequential_training = True

        try:
            if self.orchestration_config.concurrent:
                self._run_concurrent(progress)
            else:
                self._run_sequential(progress)
            self._maybe_publish_params(force=True)
        except KeyboardInterrupt:
            if get_cli_verbosity() >= 1:
                logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.cleanup()
            if progress is not None:
                progress.close()

        if get_cli_verbosity() >= 1:
            logger.info("Training completed!")
        
    def cleanup(self):
        """Clean up resources."""
        # Save final checkpoint
        if hasattr(self, 'learner'):
            final_checkpoint = self.learner.save_checkpoint(force_save=True)
            if final_checkpoint:
                logger.info(f"Saved final checkpoint: {final_checkpoint}")
            self.learner.wait_for_pending_checkpoints()
            
        # Close wandb
        if self.config.wandb.enabled:
            wandb.finish()

        for client in getattr(self, "_inference_clients", []):
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to close inference client: %s", exc)
        self._inference_clients.clear()
        for client in getattr(self, "_replay_clients", []):
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to close replay client: %s", exc)
        if hasattr(self, "_replay_clients"):
            self._replay_clients.clear()

        # Stop auto-launched servers
        for srv in [
            getattr(self, "_launched_inference_server", None),
            getattr(self, "_launched_replay_server", None),
            getattr(self, "_launched_publisher_server", None),
        ]:
            if srv is not None:
                try:
                    srv.stop()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to stop server: %s", exc)
            
        logger.info("Cleanup completed")


def setup_jax_environment(config: DictConfig):
    """Set up JAX environment and device configuration."""
    # Configure JAX based on config
    if config.resource_management.device == "cpu":
        jax.config.update('jax_platform_name', 'cpu')
        logger.info("Configured JAX to use CPU")
    elif config.resource_management.device == "gpu":
        # JAX will automatically use GPU if available
        logger.info("JAX will use GPU if available")
    elif config.resource_management.device == "tpu":
        # JAX will automatically use TPU if available
        logger.info("JAX will use TPU if available")
    else:  # auto
        logger.info("JAX will automatically select best available device")
        
    # Log available devices
    devices = jax.devices()
    logger.info(f"Available JAX devices: {devices}")
    logger.info(f"JAX default backend: {jax.default_backend()}")


@hydra.main(version_base="1.1", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point."""
    logger.info("Starting MuZero JAX orchestration")  # pragma: no cover
    logger.info(f"Working directory: {os.getcwd()}")  # pragma: no cover
    
    # Set up JAX environment
    setup_jax_environment(config)
    
    # Create and run orchestrator
    orchestrator = MuZeroOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":  # pragma: no cover
    main() 
