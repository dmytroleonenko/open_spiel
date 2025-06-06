"""
JAX/Flax NNX Batch Optimizer for MuZero (REFACTORED VERSION)

This script provides tools to optimize batch sizes for JAX/Flax NNX models,
with specific support for MuZero networks.

ARCHITECTURAL IMPROVEMENTS:
--------------------------
✅ Binary search for maximum batch size (efficient, no more hangs)
✅ JAX compilation pre-warming (eliminates repeated compilation overhead)
✅ Robust hardware interrogation (pynvml + psutil fallbacks)
✅ Modular, optional analyses (fast by default, detailed on request)
✅ Corrected gradient accumulation (proper gradient summing)
✅ Single-threaded synchronous workflow (no more ThreadPoolExecutor complexity)

PURPOSE:
--------
The batch optimizer helps you find the most hardware-efficient configuration for training
your MuZero model by:

1. **Memory Optimization**: Uses binary search to efficiently find the maximum local 
   batch size (B_max) that fits in device memory.

2. **Optional Throughput Analysis**: Identifies the batch size that maximizes throughput
   (samples/second) across different batch sizes.

3. **Optional Gradient Accumulation Analysis**: Tests gradient accumulation strategies
   with proper gradient summing to simulate larger effective batch sizes.

TYPICAL WORKFLOW:
----------------
1. Run `find_max_batch_size()` to discover memory limits efficiently
2. Optionally run `analyze_throughput()` to find the sweet spot for throughput
3. Optionally run `analyze_gradient_accumulation()` to test accumulation strategies

INTEGRATION WITH MUZERO:
-----------------------
The script provides configurable hooks for MuZero integration:
- `create_muzero_model_and_params()`: Initialize your MuZeroNetwork with config
- `compute_muzero_loss_and_gradients()`: Compute loss and gradients using MuZero's loss function
"""

import time
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple, Any, Optional, Dict, List, NamedTuple
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum
import gc

# Import MuZero specific types for the hooks
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork, RepresentationNetwork, DynamicsNetwork, PredictionNetwork, RewardNetwork
from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig, create_network_config_from_muzero_config

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ModelType(Enum):
    """Supported model types for batch optimization."""
    MUZERO = "muzero"
    GENERIC = "generic"

@dataclass
class OptimizationConfig:
    """Configuration for batch optimization."""
    # Model configuration
    model_type: ModelType = ModelType.MUZERO
    num_actions: int = 9
    observation_shape: Tuple[int, ...] = (27,)
    num_unroll_steps: int = 5
    td_steps: int = 5
    discount_factor: float = 0.997
    
    # Search parameters
    binary_search_low: int = 1024  # More reasonable starting point
    binary_search_high: int = 131072
    timeout_seconds: float = 620.0
    
    # Data type
    dtype: jnp.dtype = jnp.bfloat16
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.binary_search_low <= 0:
            raise ValueError("binary_search_low must be positive")
        if self.binary_search_high <= self.binary_search_low:
            raise ValueError("binary_search_high must be greater than binary_search_low")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if not self.observation_shape or any(dim <= 0 for dim in self.observation_shape):
            raise ValueError("observation_shape must have positive dimensions")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

# Type aliases
ModelParams = nnx.State
Grads = nnx.State
ModelInstance = nnx.Module
MuZeroBatchData = Dict[str, jax.Array]
GenericBatchData = jax.Array
BatchData = Any

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class BatchOptimizerError(Exception):
    """Base exception for batch optimizer errors."""
    pass

class ModelInitializationError(BatchOptimizerError):
    """Error during model initialization."""
    pass

class OutOfMemoryError(BatchOptimizerError):
    """Out of memory during batch processing."""
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST HARDWARE INTERROGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _get_device_memory_usage() -> Tuple[Optional[int], Optional[int]]:
    """Get JAX device memory usage with robust detection.
    
    Returns:
        Tuple of (used_bytes, total_bytes) or (None, None) if unavailable
    """
    try:
        devices = jax.devices()
        if not devices:
            return None, None
            
        device = devices[0]
        device_kind = device.device_kind.lower()

        # For GPU devices, try multiple methods
        if device_kind in ('gpu', 'cuda'):
            # Method 1: JAX device memory stats
            try:
                if hasattr(device, 'memory_stats') and callable(device.memory_stats):
                    stats = device.memory_stats()
                    if stats and isinstance(stats, dict):
                        used = stats.get('bytes_in_use', 0)
                        limit = stats.get('bytes_limit', 0)
                        if used >= 0 and limit > 0:
                            return used, limit
            except Exception:
                pass

            # Method 2: Try nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used, info.total
            except (ImportError, Exception):
                pass

        elif device_kind == 'cpu':
            # For CPU devices, try to get system memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                jax_estimated = int(mem.used * 0.1)  # Conservative estimate
                return jax_estimated, mem.total
            except ImportError:
                pass

        # If all methods fail, return None
        return None, None
        
    except Exception:
        return None, None

def _is_oom_error(error: Exception) -> bool:
    """Robustly detect OOM errors from various sources.
    
    Args:
        error: Exception to check

    Returns:
        True if this appears to be an out-of-memory error
    """
    # Direct memory error types
    if isinstance(error, MemoryError):
        return True

    # Check for XLA/JAX specific OOM errors
    error_str = str(error).lower()
    oom_indicators = [
        'out of memory',
        'oom',
        'cuda out of memory',
        'device_out_of_memory',
        'resource_exhausted',
        'memory allocation failed',
        'insufficient memory',
        'cudnn_status_alloc_failed',
        'xla::resourceexhausted',
        'failed to allocate',
        'memory pool exhausted',
        'allocator',
        'memory',
        'killed'
    ]

    return any(indicator in error_str for indicator in oom_indicators)

# ═══════════════════════════════════════════════════════════════════════════════
# MUZERO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_muzero_model_and_params(config: OptimizationConfig, rng_key: jax.random.PRNGKey) -> ModelInstance:
    """Initialize a MuZero model with parameters for batch optimization.
    
    Args:
        config: Optimization configuration containing MuZero parameters
        rng_key: JAX random key for initialization
        
    Returns:
        Initialized MuZero model (nnx.Module)
        
    Raises:
        ModelInitializationError: If model creation fails
    """
    try:
        # Create MuZero configuration from optimization config
        muzero_config = MuZeroConfig(
            num_actions=config.num_actions,
            num_unroll_steps=config.num_unroll_steps,
            td_steps=config.td_steps,
            discount_factor=config.discount_factor,
            dirichlet_alpha=0.03,
            num_simulations=10,
            batch_size=32,  # This will be overridden during optimization
            learning_rate=0.05,
            # Disable complex features that might cause crashes during testing
            reanalyze_ratio=0.0,
            use_projection=False,
            use_value_prefix=False,
            value_target="search",
            value_target_type="bootstrapped",
            use_iql=False,
            noisy_net=False,
        )
        
        # Create network configuration
        network_config = create_network_config_from_muzero_config(
            muzero_config=muzero_config,
            observation_shape=config.observation_shape,
            num_actions=config.num_actions,
            use_image_observation=False,
            spatial_extents=(1, 1),
        )
        
        # Create network definition functions
        def representation_network_def(net_config, *, rngs):
            return RepresentationNetwork(net_config, rngs=rngs)
        
        def dynamics_network_def(net_config, *, rngs):
            return DynamicsNetwork(net_config, rngs=rngs)
        
        def prediction_network_def(net_config, *, rngs):
            return PredictionNetwork(net_config, rngs=rngs)
        
        def reward_network_def(net_config, *, rngs):
            return RewardNetwork(net_config, rngs=rngs)
        
        # Initialize the MuZero network
        model = MuZeroNetwork(
            representation_network_def=representation_network_def,
            dynamics_network_def=dynamics_network_def,
            prediction_network_def=prediction_network_def,
            reward_network_def=reward_network_def,
            projection_network_def=None,
            config=network_config,
            rngs=nnx.Rngs(rng_key)
        )
        
        # Store the training config on the model for the loss function
        model.config = muzero_config
        
        return model
        
    except Exception as e:
        raise ModelInitializationError(
            f"Failed to initialize MuZero model: {e}") from e

def compute_muzero_loss_and_gradients(
    model_instance: ModelInstance, 
    batch_dict: Dict[str, jax.Array],
    rng_key: jax.random.PRNGKey
) -> Tuple[jax.Array, Grads]:
    """Compute MuZero loss and gradients for a batch.
    
    Args:
        model_instance: MuZeroNetwork instance with attached .config attribute
        batch_dict: Dictionary containing MuZero batch data
        rng_key: Random key for loss computation
    
    Returns:
        Tuple of (loss_value, gradients_state)
        
    Raises:
        BatchOptimizerError: If loss computation fails
    """
    def loss_fn(model_for_loss_fn: ModelInstance):
        if not hasattr(model_for_loss_fn, 'config') or \
           not isinstance(getattr(model_for_loss_fn, 'config', None), MuZeroConfig):
            raise ValueError(
                "MuZeroNetwork instance must have a 'config' attribute of type MuZeroConfig")
        
        effective_config: MuZeroConfig = getattr(model_for_loss_fn, 'config')

        total_loss, metrics = Learner._compute_total_loss_static(
            model=model_for_loss_fn, 
            config=effective_config,
            batch=batch_dict, 
            rng_key=rng_key, 
            training=True             
        )
        return total_loss, metrics 

    try:
        (loss_value, metrics_val), grads_state = nnx.value_and_grad(
            loss_fn, has_aux=True)(model_instance)
        return loss_value, grads_state
        
    except ValueError:
        # Re-raise ValueError as-is (for test compatibility)
        raise
    except Exception as e:
        raise BatchOptimizerError(
            f"Failed to compute MuZero loss and gradients: {e}") from e

# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _create_generic_model(config: OptimizationConfig, rng_key: jax.random.PRNGKey) -> ModelInstance:
    """Create a generic neural network model for batch optimization testing.

    Args:
        config: Optimization configuration
        rng_key: JAX random key for initialization

    Returns:
        Initialized generic model (nnx.Module)

    Raises:
        ModelInitializationError: If model creation fails
    """
    try:
        class GenericMLP(nnx.Module):
            """Simple MLP for generic testing."""

            def __init__(self, config: OptimizationConfig, *, rngs: nnx.Rngs):
                # Create a reasonably complex network for realistic testing
                input_size = int(np.prod(config.observation_shape))
                # Ensure decent complexity
                hidden_size = max(256, input_size * 2)

                self.layers = [
                    nnx.Linear(input_size, hidden_size, rngs=rngs),
                    nnx.Linear(hidden_size, hidden_size, rngs=rngs),
                    nnx.Linear(hidden_size, hidden_size // 2, rngs=rngs),
                    nnx.Linear(hidden_size // 2, config.num_actions, rngs=rngs),
                ]

                # Add some memory-intensive components to make optimization meaningful
                self.value_head = nnx.Linear(hidden_size // 2, 1, rngs=rngs)

            def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
                # Flatten input
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)

                # Forward pass through layers
                for i, layer in enumerate(self.layers[:-1]):
                    x = layer(x)
                    x = jax.nn.relu(x)

                # Output heads
                policy_logits = self.layers[-1](x)
                value = self.value_head(x)

                return policy_logits, value.squeeze(-1)

        # Initialize the model
        model = GenericMLP(config, rngs=nnx.Rngs(rng_key))
        return model

    except Exception as e:
        raise ModelInitializationError(
            f"Failed to initialize generic model: {e}") from e

def _generic_loss_fn(
    model_instance: ModelInstance,
    batch_data: jax.Array,
    rng_key: jax.random.PRNGKey
) -> Tuple[jax.Array, Grads]:
    """Compute generic loss and gradients for a batch.

    Args:
        model_instance: Generic model instance
        batch_data: Input batch data (observations)
        rng_key: Random key (for synthetic targets)

    Returns:
        Tuple of (loss_value, gradients_state)

    Raises:
        BatchOptimizerError: If loss computation fails
    """
    def loss_fn(model_for_loss_fn: ModelInstance):
        policy_logits, values = model_for_loss_fn(batch_data)

        # Create synthetic targets for testing
        batch_size = batch_data.shape[0]
        target_policy = jax.nn.softmax(
            jax.random.normal(rng_key, policy_logits.shape))
        target_values = jax.random.normal(
            jax.random.fold_in(rng_key, 1), values.shape)

        # Compute losses
        policy_loss = -jnp.mean(jnp.sum(target_policy *
                     jax.nn.log_softmax(policy_logits), axis=-1))
        value_loss = jnp.mean((values - target_values) ** 2)

        total_loss = policy_loss + 0.5 * value_loss
        return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss}

    try:
        (loss_value, metrics), grads_state = nnx.value_and_grad(
            loss_fn, has_aux=True)(model_instance)
        return loss_value, grads_state

    except Exception as e:
        raise BatchOptimizerError(
            f"Failed to compute generic loss and gradients: {e}") from e

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _create_random_batch(
    rng_key: jax.random.PRNGKey, 
    batch_size: int, 
    config: OptimizationConfig
) -> BatchData:
    """Create a random batch for testing based on model type.
    
    Args:
        rng_key: JAX random key
        batch_size: Number of samples in batch
        config: Optimization configuration
    
    Returns:
        Batch data appropriate for the model type
    """
    if config.model_type == ModelType.MUZERO:
        batch = {
            'observation': jax.random.normal(
                rng_key, 
                (batch_size, config.num_unroll_steps + 1, *config.observation_shape), 
                dtype=config.dtype
            ),
            'action': jax.random.randint(
                jax.random.fold_in(rng_key, 1), 
                (batch_size, config.num_unroll_steps), 
                0, config.num_actions
            ),
            'target_reward': jax.random.normal(
                jax.random.fold_in(rng_key, 2), 
                (batch_size, config.num_unroll_steps + 1), 
                dtype=config.dtype
            ),
            'target_value': jax.random.normal(
                jax.random.fold_in(rng_key, 3), 
                (batch_size, config.num_unroll_steps + 1), 
                dtype=config.dtype
            ),
            'target_policy': jax.nn.softmax(
                jax.random.normal(
                    jax.random.fold_in(rng_key, 4), 
                    (batch_size, config.num_unroll_steps + 1, config.num_actions), 
                    dtype=config.dtype
                ), 
                axis=-1
            ),
            'game_history_mask': jnp.ones(
                (batch_size, config.num_unroll_steps + 1), 
                dtype=config.dtype
            ),
            'weights': jnp.ones((batch_size,), dtype=config.dtype)
        }
        return batch
    else:
        # Generic model mode
        return jax.random.normal(rng_key, (batch_size, *config.observation_shape), dtype=config.dtype)

def _flatten_gradients(grads_pytree: Grads) -> jax.Array:
    """Flatten gradient PyTree into a single JAX array.
    
    Args:
        grads_pytree: Gradient PyTree from nnx.value_and_grad
        
    Returns:
        Flattened gradient array
    """
    leaves, _ = jax.tree_util.tree_flatten(grads_pytree)
    if not leaves:
        return jnp.array([], dtype=jnp.float32) 
        
    flat_leaves = [jnp.reshape(x, (-1,)) for x in leaves]
    return jnp.concatenate(flat_leaves, axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH OPTIMIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BatchOptimizer:
    """Efficient batch optimizer with binary search and JAX pre-warming."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize model creation and loss functions based on type
        if config.model_type == ModelType.MUZERO:
            self.model_init_fn = create_muzero_model_and_params
            self.loss_fn = compute_muzero_loss_and_gradients
        elif config.model_type == ModelType.GENERIC:
            self.model_init_fn = _create_generic_model
            self.loss_fn = _generic_loss_fn
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        # State
        self.model_instance: Optional[ModelInstance] = None
        self.max_batch_size: Optional[int] = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the optimizer."""
        logger = logging.getLogger(f"BatchOptimizer_{id(self)}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _initialize_model(self, rng_key: jax.random.PRNGKey):
        """Initialize the model instance."""
        if self.model_instance is not None:
            return  # Already initialized
            
        self.logger.info(f"Initializing {self.config.model_type.value} model...")
        self.model_instance = self.model_init_fn(self.config, rng_key)
        self.logger.info("Model initialized successfully")
    
    def _cleanup_model(self):
        """Clean up model instance."""
        if self.model_instance is not None:
            del self.model_instance
            self.model_instance = None
            gc.collect()
    
    def _test_batch_size(self, batch_size: int, rng_key: jax.random.PRNGKey) -> bool:
        """Test if a specific batch size works without OOM.
        
        Args:
            batch_size: Batch size to test
            rng_key: Random key for batch generation
            
        Returns:
            True if successful, False if OOM
        """
        try:
            # Create batch data
            batch_data = _create_random_batch(rng_key, batch_size, self.config)
            
            # Run forward-backward pass
            _, grads = self.loss_fn(self.model_instance, batch_data, rng_key)
            
            # Ensure computation completes
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
            
            # Clean up
            del batch_data, grads
            return True
            
        except Exception as e:
            if _is_oom_error(e):
                return False
            else:
                # Re-raise non-OOM errors
                raise e
    
    def _pre_warm_compilation(self, rng_key: jax.random.PRNGKey):
        """Pre-warm JAX compilation to avoid repeated compilation overhead.
        
        This is critical for JAX-Metal backend where each new batch size
        triggers expensive XLA compilation. By doing one small forward-backward
        pass upfront, we compile all necessary operations.
        
        Args:
            rng_key: Random key for batch generation
        """
        self.logger.info("Pre-warming JAX compilation...")
        start_time = time.perf_counter()
        
        # Use a very small batch for compilation
        warmup_batch_size = 1
        try:
            batch_data = _create_random_batch(rng_key, warmup_batch_size, self.config)
            _, grads = self.loss_fn(self.model_instance, batch_data, rng_key)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
            del batch_data, grads
        except Exception as e:
            self.logger.warning(f"Compilation pre-warming failed: {e}")

        compilation_time = time.perf_counter() - start_time
        self.logger.info(f"JAX compilation pre-warming completed in {compilation_time:.2f}s")
    
    def find_max_batch_size(self, rng_key: jax.random.PRNGKey) -> int:
        """Find maximum batch size using smart adaptive search with performance monitoring.

        This implementation is specifically optimized for Apple Silicon M3 Max systems where
        OOM conditions don't cause crashes but result in heavy swapping and performance degradation.

        Args:
            rng_key: Random key for testing

        Returns:
            Maximum practical batch size that avoids performance degradation
        """
        self.logger.info("Finding maximum batch size using adaptive search with performance monitoring...")
        
        # Initialize model if not already done
        rng_init, rng_search = jax.random.split(rng_key)
        self._initialize_model(rng_init)
        
        # Pre-warm compilation
        self._pre_warm_compilation(rng_search)
        
        # Phase 1: Quick exponential search to find approximate upper bound
        self.logger.info("Phase 1: Exponential search for approximate upper bound...")
        current_batch_size = self.config.binary_search_low  # Start from config
        max_time_per_sample = None  # Baseline for performance degradation detection
        last_successful = current_batch_size
        exponential_factor = 2
        
        while current_batch_size <= self.config.binary_search_high:
            self.logger.info(f"Testing batch size {current_batch_size} (exponential phase)")
            
            rng_test = jax.random.fold_in(rng_search, current_batch_size)
            
            try:
                # Time the operation for performance monitoring
                start_time = time.perf_counter()
                if self._test_batch_size(current_batch_size, rng_test):
                    end_time = time.perf_counter()
                    test_duration = end_time - start_time
                    time_per_sample = test_duration / current_batch_size
                    
                    # Check for performance degradation (swapping indicator)
                    if max_time_per_sample is None:
                        max_time_per_sample = time_per_sample * 3.0  # Allow 3x degradation
                        self.logger.info(f"✅ Baseline: {current_batch_size} in {test_duration:.1f}s ({time_per_sample*1000:.2f}ms/sample)")
                    elif time_per_sample > max_time_per_sample:
                        self.logger.info(f"⚠️  Performance degradation detected at {current_batch_size} ({time_per_sample*1000:.2f}ms/sample > {max_time_per_sample*1000:.2f}ms/sample)")
                        break
                    else:
                        self.logger.info(f"✅ {current_batch_size} in {test_duration:.1f}s ({time_per_sample*1000:.2f}ms/sample)")
                    
                    last_successful = current_batch_size
                    current_batch_size *= exponential_factor
                else:
                    self.logger.info(f"❌ Batch size {current_batch_size} failed (OOM)")
                    break
                    
            except Exception as e:
                if _is_oom_error(e):
                    self.logger.info(f"❌ Batch size {current_batch_size} failed (OOM exception)")
                    break
                else:
                    self.logger.error(f"❌ Batch size {current_batch_size} failed with error: {e}")
                    break
        
        # If we found a practical limit due to performance degradation, use it
        practical_upper_bound = min(last_successful * exponential_factor, self.config.binary_search_high)
        
        # Phase 2: Binary search refinement in a narrow range
        self.logger.info(f"Phase 2: Binary search refinement between {last_successful} and {practical_upper_bound}")
        
        low = last_successful
        high = practical_upper_bound
        
        iteration = 0
        while low < high - 64:  # Stop when range is small enough (within 64)
            iteration += 1
            mid = (low + high) // 2
            
            # Round to nearest multiple of 64 for cleaner batch sizes
            mid = ((mid + 31) // 64) * 64
            
            if mid <= low:
                break
                
            self.logger.info(f"Refinement iteration {iteration}: Testing batch size {mid}")
            
            rng_test = jax.random.fold_in(rng_search, mid + 1000)  # Different seed for refinement
            
            try:
                start_time = time.perf_counter()
                if self._test_batch_size(mid, rng_test):
                    end_time = time.perf_counter()
                    test_duration = end_time - start_time
                    time_per_sample = test_duration / mid
                    
                    # Check if performance is still acceptable
                    if max_time_per_sample is not None and time_per_sample > max_time_per_sample:
                        self.logger.info(f"⚠️  Performance degradation at {mid}, reducing upper bound")
                        high = mid - 1
                    else:
                        self.logger.info(f"✅ Batch size {mid} succeeded ({time_per_sample*1000:.2f}ms/sample)")
                        last_successful = mid
                        low = mid
                else:
                    self.logger.info(f"❌ Batch size {mid} failed (OOM)")
                    high = mid - 1
                    
            except Exception as e:
                if _is_oom_error(e):
                    self.logger.info(f"❌ Batch size {mid} failed (OOM exception)")
                    high = mid - 1
                else:
                    self.logger.error(f"❌ Batch size {mid} failed with error: {e}")
                    high = mid - 1
        
        self.max_batch_size = last_successful
        self.logger.info(f"🎯 Maximum practical batch size found: {last_successful}")
        
        # Log performance characteristics
        if max_time_per_sample is not None:
            self.logger.info(f"📊 Performance limit: {max_time_per_sample*1000:.2f}ms/sample")
        
        return last_successful
    
    def analyze_throughput(self, max_batch_size: Optional[int] = None, num_test_points: int = 10) -> Dict[str, Any]:
        """Analyze throughput across different batch sizes using adaptive sampling.
        
        Args:
            max_batch_size: Maximum batch size to test (uses found max if None)
            num_test_points: Number of test points to sample (default: 10)
            
        Returns:
            Dictionary with throughput analysis results
        """
        if max_batch_size is None:
            if self.max_batch_size is None:
                raise ValueError("Must run find_max_batch_size() first or provide max_batch_size")
            max_batch_size = self.max_batch_size
        
        self.logger.info(f"Analyzing throughput up to batch size {max_batch_size} using {num_test_points} test points")
        
        # Ensure model is initialized
        if self.model_instance is None:
            rng_key = jax.random.PRNGKey(42)
            self._initialize_model(rng_key)
        
        results = {}
        best_throughput = 0.0
        best_batch_size = 0
        
        # Create adaptive batch size sampling - logarithmic spacing for better coverage
        min_batch_size = max(1024, max_batch_size // 32)  # Start from reasonable minimum
        
        # Use logarithmic spacing to cover the range efficiently
        if max_batch_size <= min_batch_size:
            batch_sizes = [max_batch_size]
        else:
            # Generate logarithmically spaced points
            log_min = np.log(min_batch_size)
            log_max = np.log(max_batch_size)
            log_points = np.linspace(log_min, log_max, num_test_points)
            batch_sizes = [int(np.round(np.exp(log_point))) for log_point in log_points]
            
            # Remove duplicates and sort
            batch_sizes = sorted(list(set(batch_sizes)))
            
            # Always include the maximum batch size
            if max_batch_size not in batch_sizes:
                batch_sizes.append(max_batch_size)
        
        self.logger.info(f"Testing batch sizes: {batch_sizes}")
        
        for batch_size in batch_sizes:
            self.logger.info(f"Testing throughput for batch size {batch_size}")
            
            # Time multiple runs
            times = []
            for i in range(3):  # 3 runs for averaging
                rng_key = jax.random.PRNGKey(batch_size * 100 + i)
                batch_data = _create_random_batch(rng_key, batch_size, self.config)
                
                start_time = time.perf_counter()
                _, grads = self.loss_fn(self.model_instance, batch_data, rng_key)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                del batch_data, grads
            
            # Calculate throughput
            mean_time = np.mean(times)
            throughput = batch_size / mean_time
            
            results[batch_size] = {
                'mean_time': mean_time,
                'throughput': throughput,
                'samples_per_second': throughput
            }
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            self.logger.info(f"Batch {batch_size}: {throughput:.1f} samples/sec")
        
        self.logger.info(f"🚀 Best throughput: {best_throughput:.1f} samples/sec at batch size {best_batch_size}")
        
        return {
            'results': results,
            'best_batch_size': best_batch_size,
            'best_throughput': best_throughput
        }
    
    def analyze_gradient_accumulation(self, batch_size: int, max_accumulation_steps: int = 8) -> Dict[str, Any]:
        """Analyze gradient accumulation with proper gradient summing.
        
        Args:
            batch_size: Base batch size for accumulation
            max_accumulation_steps: Maximum number of accumulation steps to test
            
        Returns:
            Dictionary with gradient accumulation analysis results
        """
        self.logger.info(f"Analyzing gradient accumulation for batch size {batch_size}")
        
        # Ensure model is initialized
        if self.model_instance is None:
            rng_key = jax.random.PRNGKey(42)
            self._initialize_model(rng_key)
        
        results = {}
        
        # Compute reference gradient with large batch
        self.logger.info(f"Computing reference gradient with batch size {batch_size}")
        rng_ref = jax.random.PRNGKey(999)
        batch_data_ref = _create_random_batch(rng_ref, batch_size, self.config)
        _, ref_grads = self.loss_fn(self.model_instance, batch_data_ref, rng_ref)
        ref_grad_norm = float(jnp.linalg.norm(_flatten_gradients(ref_grads)))
        del batch_data_ref
        
        # Test different accumulation strategies
        for k in range(1, max_accumulation_steps + 1):
            micro_batch_size = batch_size // k
            if micro_batch_size == 0:
                continue
                
            self.logger.info(f"Testing {k} accumulation steps with micro-batch size {micro_batch_size}")
            
            # Accumulate gradients properly
            accumulated_grads = None
            
            for step in range(k):
                rng_step = jax.random.PRNGKey(k * 1000 + step)
                batch_data = _create_random_batch(rng_step, micro_batch_size, self.config)
                _, step_grads = self.loss_fn(self.model_instance, batch_data, rng_step)
                
                if accumulated_grads is None:
                    accumulated_grads = step_grads
                else:
                    # Properly sum gradients
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda acc, step: acc + step,
                        accumulated_grads,
                        step_grads
                    )
                
                del batch_data, step_grads
            
            # Average the accumulated gradients
            final_grads = jax.tree_util.tree_map(
                lambda acc: acc / k,
                accumulated_grads
            )
            
            # Compare to reference
            final_grad_norm = float(jnp.linalg.norm(_flatten_gradients(final_grads)))
            relative_error = abs(final_grad_norm - ref_grad_norm) / ref_grad_norm
            
            results[k] = {
                'micro_batch_size': micro_batch_size,
                'effective_batch_size': micro_batch_size * k,
                'gradient_norm': final_grad_norm,
                'reference_norm': ref_grad_norm,
                'relative_error': relative_error
            }
            
            self.logger.info(f"K={k}: grad_norm={final_grad_norm:.3e}, relative_error={relative_error:.1%}")
            
            del accumulated_grads, final_grads
        
        del ref_grads
        
        return results
    
    def run(self, 
            analyze_throughput: bool = False, 
            analyze_accumulation: bool = False,
            rng_seed: int = 42) -> Dict[str, Any]:
        """Run the complete batch optimization workflow.
        
        Args:
            analyze_throughput: Whether to run throughput analysis
            analyze_accumulation: Whether to run gradient accumulation analysis  
            rng_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("🚀 Starting batch optimization workflow")
        
        # Log hardware information
        used_mem, total_mem = _get_device_memory_usage()
        if used_mem is not None and total_mem is not None:
            self.logger.info(f"Device memory: {used_mem/(1024**3):.1f}GB / {total_mem/(1024**3):.1f}GB")
        else:
            self.logger.info("Device memory information not available")
        
        results = {}
        rng_key = jax.random.PRNGKey(rng_seed)
        
        try:
            # Phase 1: Find maximum batch size (always run)
            rng_max, rng_key = jax.random.split(rng_key)
            max_batch_size = self.find_max_batch_size(rng_max)
            results['max_batch_size'] = max_batch_size
            
            # Phase 2: Optional throughput analysis
            if analyze_throughput:
                self.logger.info("Running throughput analysis...")
                throughput_results = self.analyze_throughput(max_batch_size)
                results['throughput_analysis'] = throughput_results
            
            # Phase 3: Optional gradient accumulation analysis
            if analyze_accumulation:
                self.logger.info("Running gradient accumulation analysis...")
                accumulation_results = self.analyze_gradient_accumulation(max_batch_size)
                results['accumulation_analysis'] = accumulation_results
            
            # Final report
            self.logger.info("🎉 Optimization completed successfully!")
            self.logger.info(f"📊 Maximum batch size: {max_batch_size}")
            
            if analyze_throughput and 'throughput_analysis' in results:
                best_batch = results['throughput_analysis']['best_batch_size']
                best_throughput = results['throughput_analysis']['best_throughput']
                self.logger.info(f"🚀 Best throughput: {best_throughput:.1f} samples/sec at batch size {best_batch}")
            
            return results
            
        finally:
            # Always cleanup
            self._cleanup_model()

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO/EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("BATCH OPTIMIZER DEMO - REFACTORED VERSION")
    print("="*80)
    
    # Test configurations
    test_configs = [
        ("MuZero Model", OptimizationConfig(
            model_type=ModelType.MUZERO,
            num_actions=9,
            observation_shape=(27,),
            binary_search_low=1024,  # Use sensible defaults
            binary_search_high=131072,
            dtype=jnp.bfloat16,
            timeout_seconds=120.0,
        )),
        ("Generic Model", OptimizationConfig(
            model_type=ModelType.GENERIC,
            num_actions=4,
            observation_shape=(64,),
            binary_search_low=1024,  # Use sensible defaults
            binary_search_high=131072,
            dtype=jnp.bfloat16,
            timeout_seconds=120.0,
        ))
    ]

    for model_name, config in test_configs:
        print(f"\n{'='*60}")
        print(f"TESTING: {model_name}")
        print(f"{'='*60}")
    
        try:
            # Create optimizer
            optimizer = BatchOptimizer(config)
            
            # Run fast analysis (just find max batch size)
            print(f"\n--- FAST ANALYSIS ---")
            results = optimizer.run(analyze_throughput=False, analyze_accumulation=False)
            print(f"Maximum batch size: {results['max_batch_size']}")
            
            # Run detailed analysis (with throughput and accumulation)
            print(f"\n--- DETAILED ANALYSIS ---")
            optimizer = BatchOptimizer(config)  # Fresh instance
            results = optimizer.run(analyze_throughput=True, analyze_accumulation=True)
            
            # Print detailed results
            print(f"\nDETAILED RESULTS - {model_name}")
            print("-" * 40)
            print(f"Model type: {config.model_type.value}")
            print(f"Observation shape: {config.observation_shape}")
            print(f"Maximum batch size: {results['max_batch_size']}")

            if 'throughput_analysis' in results:
                ta = results['throughput_analysis']
                print(f"Best throughput batch size: {ta['best_batch_size']}")
                print(f"Best throughput: {ta['best_throughput']:.1f} samples/sec")
            
            if 'accumulation_analysis' in results:
                aa = results['accumulation_analysis']
                print(f"Gradient accumulation strategies tested: {len(aa)} configurations")
                # Show best accumulation strategy (lowest relative error)
                best_k = min(aa.keys(), key=lambda k: aa[k]['relative_error'])
                best_result = aa[best_k]
                print(f"Best accumulation: K={best_k}, micro_batch={best_result['micro_batch_size']}, "
                      f"relative_error={best_result['relative_error']:.1%}")

        except Exception as e:
            print(f"Optimization failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("DEMO COMPLETED")
    print(f"{'='*80}")
 