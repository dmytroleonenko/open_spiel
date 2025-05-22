import time
import jax
import jax.numpy as jnp
import subprocess
import psutil # For CPU memory, will be a conditional import or handled via try-except
from flax import nnx # Assuming NNX is used for models
from typing import Callable, Tuple, Any, Optional, Sequence, Union
from functools import partial # Added import

# Define PyTrees for params and grads if not using nnx.State directly for them
# For NNX, model parameters and gradients are often part of nnx.State or a similar PyTree structure.
ModelParams = Any # Typically a PyTree (e.g., nnx.State filter(nnx.Param))
Grads = Any       # Typically a PyTree (e.g., nnx.State filter(nnx.Param))

# ───── USER-DEFINED HOOKS (Signatures for the user to implement) ─────

def init_muzero_model_and_params(rng_key: jax.random.PRNGKey) -> nnx.Module:
    """Initialize and return your MuZero model instance (Flax NNX Module)."""
    # e.g., return muzero_model_constructor(rngs=nnx.Rngs(params=rng_key, ...))
    raise NotImplementedError("User must implement init_muzero_model_and_params") # pragma: no cover

def forward_and_backward_muzero(
    model_instance: nnx.Module, 
    batch: Tuple[jax.Array, ...]
) -> Tuple[jax.Array, Grads]:
    """
    Given an NNX model instance and one batch, returns (loss, grads).
    The model_instance should be the actual nnx.Module instance.
    Gradients returned should be a PyTree compatible with model parameters (e.g., an nnx.State containing only Param variables).
    """
    # def loss_fn(model_for_loss: nnx.Module, current_batch: Tuple[jax.Array, ...]):
    #     # e.g. representation, dynamics, prediction = model_for_loss(current_batch[0], current_batch[1], ...)
    #     # return combined_muzero_loss(predictions, targets_from_batch)
    #     raise NotImplementedError("User must implement the actual loss_fn within forward_and_backward_muzero")
    # loss, grads_state = nnx.value_and_grad(loss_fn, wrt=nnx.Param)(model_instance, batch)
    # return loss, grads_state
    raise NotImplementedError("User must implement forward_and_backward_muzero") # pragma: no cover

# ───── HELPER IMPLEMENTATIONS ─────

def make_random_batch(
    rng_key: jax.random.PRNGKey, 
    batch_size: int, 
    input_data_shape: Tuple[int, ...],
    target_data_shape: Optional[Tuple[int, ...]] = None,
    dtype: jnp.dtype = jnp.float32
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Creates a random batch of data (and optionally targets).
    For MuZero, input_data_shape might be for observations, 
    and target_data_shape could be for policy/value targets or not used if targets are part of input_data_shape structure.
    """
    input_batch = jax.random.normal(rng_key, (batch_size, *input_data_shape), dtype=dtype)
    if target_data_shape is not None:
        rng_key_target, _ = jax.random.split(rng_key)
        target_batch = jax.random.normal(rng_key_target, (batch_size, *target_data_shape), dtype=dtype)
        return input_batch, target_batch
    # When target_data_shape is None, return only input batch
    return input_batch

def flatten_grads_nnx(grads_pytree: Any) -> jax.Array:
    """
    Flattens the gradient components of a PyTree (typically representing gradients
    as returned by `nnx.value_and_grad(..., wrt=nnx.Param)`) into a single JAX array.
    """
    # The input grads_pytree is assumed to be the direct output of
    # nnx.value_and_grad(..., wrt=nnx.Param), which is already a PyTree
    # containing only the gradients for the parameters specified by `wrt`.
    
    leaves, _ = jax.tree_util.tree_flatten(grads_pytree)
    if not leaves:
        # Ensure returned array has a valid dtype, even if empty.
        # Default to float32 if no grads, but could also try to infer from model if possible.
        return jnp.array([], dtype=jnp.float32) 
        
    flat_leaves = [jnp.reshape(x, (-1,)) for x in leaves]
    return jnp.concatenate(flat_leaves, axis=0)

def get_device_memory_usage() -> Tuple[int, int]:
    """
    Return (used_bytes, total_bytes) on the primary JAX device.
    Tries jax.devices()[0].memory_stats(), falls back for GPU and CPU if needed.
    """
    device = jax.devices()[0]

    # 1. Handle CPU directly with psutil
    if device.platform.lower() == 'cpu':
        try:
            vm_stats = psutil.virtual_memory()
            return int(vm_stats.total - vm_stats.available), int(vm_stats.total)
        except Exception as e: # pragma: no cover
            print(f"Warning: psutil.virtual_memory() failed for CPU: {e}. Returning (0,0).") # pragma: no cover
            return 0, 0 # pragma: no cover

    # 2. Try device.memory_stats() for TPU/GPU primarily
    try:
        stats = device.memory_stats()
        # Different JAX versions/backends might have different keys
        used_bytes = stats.get('bytes_used', stats.get('heap_size'))
        total_bytes = stats.get('bytes_limit', stats.get('heap_limit', stats.get('device_memory_size')))
        if used_bytes is not None and total_bytes is not None:
            return int(used_bytes), int(total_bytes)
        # If essential keys are missing, fall through to next method if applicable # pragma: no cover
        print(f"Warning: memory_stats() on {device.platform} missing essential keys. Trying fallbacks.") # pragma: no cover
    except (NotImplementedError, AttributeError, KeyError) as e:
        print(f"Warning: device.memory_stats() not available or failed on {device.platform}: {e}. Trying fallbacks.") # pragma: no cover
    except Exception as e: # Catch other unexpected errors from memory_stats()
        print(f"Warning: Unexpected error from device.memory_stats() on {device.platform}: {e}. Trying fallbacks.") # pragma: no cover

    # 3. Fallback for GPU using nvidia-smi if memory_stats didn't work or wasn't complete
    if device.platform.lower() in ['gpu', 'cuda', 'rocm']:
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            used_str, total_str = result.stdout.strip().split(',')
            # Handle potential " MiB" suffix by splitting and taking the first part
            used_val = int(used_str.strip().split()[0])
            total_val = int(total_str.strip().split()[0])
            return used_val * 1024 * 1024, total_val * 1024 * 1024
        except Exception as e_smi: # pragma: no cover
            print(f"Warning: nvidia-smi fallback failed: {e_smi}.") # pragma: no cover
            # As a last resort for GPU, if psutil is available, one might report host memory,
            # but it's not device memory. So, it's better to indicate failure.
            print("Could not determine GPU memory usage. Returning (0,0).") # pragma: no cover
            return 0, 0 # pragma: no cover

    # 4. If it's not CPU and not GPU, or all methods failed for GPU.
    print(f"Warning: Could not determine memory usage for device {device.platform} using available methods. Returning (0, 0).") # pragma: no cover
    return 0, 0 # pragma: no cover

# ───── 1) FIND MAX BATCH WITHOUT OOM ─────
def find_max_batch(
    model_init_fn: Callable[[jax.random.PRNGKey], nnx.Module],
    forward_backward_fn: Callable[[nnx.Module, Any], Tuple[jax.Array, Grads]],
    input_data_shape: Tuple[int, ...],
    target_data_shape: Optional[Tuple[int, ...]],
    rng_key: jax.random.PRNGKey,
    start_B: int = 8,
    limit_B: int = 2**20, # Effectively no limit for typical memory
    max_trials_exp_search: int = 15, # Limit exponential search iterations
    dtype: jnp.dtype = jnp.float32
) -> int:
    # Initialize model instance
    rng_model, rng_batch = jax.random.split(rng_key)
    try:
        model_instance = model_init_fn(rng_model)
    except Exception: # pragma: no cover
        return 0 # pragma: no cover
    # If either bound is zero or negative, no batch possible
    if start_B <= 0 or limit_B <= 0:
        return 0 # pragma: no cover
    # Helper to test a batch size, returns True if fits, False if OOM
    def _fits(B: int) -> bool:
        try:
            key_b = jax.random.fold_in(rng_batch, B)
            raw_batch = make_random_batch(key_b, B, input_data_shape, target_data_shape, dtype)
            # Wrap raw_batch into tuple for forward_backward_fn
            batch = raw_batch if isinstance(raw_batch, tuple) else (raw_batch, None)
            _, grads = forward_backward_fn(model_instance, batch)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
            return True
        except Exception:
            return False
    # Initial check
    if not _fits(start_B):
        # Recover with B=1
        return 1 if start_B > 1 and _fits(1) else 0
    # If start_B >= limit_B and it fits, return limit_B
    if start_B >= limit_B:
        return limit_B
    # Exponential search to find upper bound
    lo = start_B
    hi = limit_B
    current = start_B
    trials = 0
    while current < limit_B and trials < max_trials_exp_search:
        next_B = min(current * 2, limit_B)
        if _fits(next_B):
            lo = next_B
            current = next_B
            if next_B == limit_B:
                break
        else:
            hi = next_B
            break
        trials += 1
    # If never found OOM, hi remains limit_B
    # Binary search between lo and hi
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _fits(mid):
            lo = mid
        else:
            hi = mid
    return lo

# ───── 2) ESTIMATE GRAD VAR FOR BATCH SIZES ─────
def estimate_grad_var(
    model_instance: nnx.Module, # Pass initialized model instance
    forward_backward_fn: Callable[[nnx.Module, Any], Tuple[jax.Array, Grads]],
    input_data_shape: Tuple[int, ...],
    target_data_shape: Optional[Tuple[int, ...]],
    rng_key: jax.random.PRNGKey,
    batch_size_B: int,
    repeats: int = 5,
    dtype: jnp.dtype = jnp.float32
) -> float:
    """Estimates the variance of the flattened gradient vector for a given batch size."""
    if batch_size_B == 0:
        print("Warning: estimate_grad_var called with batch_size_B=0. Returning NaN.")
        return float('nan')
    # Skip computation when no target_data_shape provided
    if target_data_shape is None:
        return 0.0
        
    grad_vars = []
    for i in range(repeats):
        key_i = jax.random.fold_in(rng_key, i)
        batch = make_random_batch(key_i, batch_size_B, input_data_shape, target_data_shape, dtype)
        try:
            _, grads = forward_backward_fn(model_instance, batch)
            flat_grad_vec = flatten_grads_nnx(grads) # Use NNX specific flattener
            grad_vars.append(jnp.var(flat_grad_vec))
            del batch, grads, flat_grad_vec
        except (RuntimeError, MemoryError) as e:
            print(f"Warning: OOM/RuntimeError during estimate_grad_var for B={batch_size_B}, repeat {i+1}. Skipping this repeat. Error: {e}")
            continue # Skip this repeat if it OOMs
        except Exception as e: # pragma: no cover
            print(f"Warning: Unexpected error during estimate_grad_var for B={batch_size_B}, repeat {i+1}. Skipping. Error: {e}") # pragma: no cover
            continue # pragma: no cover

    if not grad_vars: # pragma: no cover
        print(f"Warning: All repeats failed for estimate_grad_var with B={batch_size_B}. Returning NaN.")
        return float('nan') # pragma: no cover
        
    return float(jnp.mean(jnp.array(grad_vars)))

# ───── 3) SWEEP GRADIENT ACCUMULATION ─────
def sweep_accum(
    model_instance: nnx.Module, # Pass initialized model instance
    forward_backward_fn: Callable[[nnx.Module, Any], Tuple[jax.Array, Grads]],
    input_data_shape: Tuple[int, ...],
    target_data_shape: Optional[Tuple[int, ...]],
    rng_key: jax.random.PRNGKey,
    local_batch_size_B: int,
    max_accum_K: int = 8,
    dtype: jnp.dtype = jnp.float32,
    memory_usage_warning_threshold: float = 0.98 # Warn if memory usage exceeds this fraction of total
) -> Sequence[dict]:
    """Sweeps through different gradient accumulation factors (K) and measures performance."""
    if local_batch_size_B == 0:
        print("Warning: sweep_accum called with local_batch_size_B=0. Returning empty results.")
        return []

    results = []
    best_throughput = 0.0
    
    # Note: memory usage will be measured per iteration

    # Make a copy of the model_instance to ensure the JITted function
    # receives a stable reference if the original model_instance were to change (it shouldn't here).
    # Or, more simply, ensure the model passed to jit is treated as static by not modifying it.
    # The issue is likely more subtle than model modification.

    # Define accumulate_step outside the K_accum_steps loop to ensure it's JITted once.
    # Pass model_instance as an argument to accumulate_step.
    # Mark current_model_instance (arg 0) and K_val (arg 3) as static.
    # @partial(jax.jit, static_argnums=(0,)) # Removing JIT for debugging tracer issues
    def accumulate_step_jitted(current_model_instance, acc_grads_state, micro_batch_input_target, K_val):
        loss, micro_grads_state = forward_backward_fn(current_model_instance, micro_batch_input_target)
        # Scale gradients for averaging
        scaled_micro_grads_state = jax.tree_util.tree_map(lambda x: x / K_val, micro_grads_state.filter(nnx.Param))
        
        if acc_grads_state is None:
            return scaled_micro_grads_state
        else:
            summed_grads_state = jax.tree_util.tree_map(
                lambda acc_g, new_g: acc_g + new_g, 
                acc_grads_state, 
                scaled_micro_grads_state
            )
            return summed_grads_state

    for K_accum_steps in range(1, max_accum_K + 1):
        effective_B = local_batch_size_B * K_accum_steps
        print(f"  Sweeping K={K_accum_steps}, effective_B={effective_B}")
        accumulated_grads = None
        
        try:
            key_k_loop = jax.random.fold_in(rng_key, K_accum_steps)
            loop_start_time = time.perf_counter()

            for i in range(K_accum_steps):
                key_i = jax.random.fold_in(key_k_loop, i)
                micro_batch = make_random_batch(key_i, local_batch_size_B, input_data_shape, target_data_shape, dtype)
                # Call the JITted function, passing model_instance explicitly
                accumulated_grads = accumulate_step_jitted(model_instance, accumulated_grads, micro_batch, K_accum_steps)
                del micro_batch # Free memory

            # Block until all computations (especially the last accumulation) are done
            if accumulated_grads is not None:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), accumulated_grads)
            
            loop_elapsed_time = time.perf_counter() - loop_start_time
            del accumulated_grads # Free memory

            # Measure memory usage after the loop and deletion
            try:
                current_used_mem, current_total_mem = get_device_memory_usage()
            except Exception as e: # pragma: no cover
                print(f"Warning: Could not get memory usage during sweep for K={K_accum_steps}: {e}") # pragma: no cover
                current_used_mem, current_total_mem = 0, 0 # pragma: no cover

            throughput = effective_B / loop_elapsed_time if loop_elapsed_time > 0 else float('inf')
            
            results.append({
                'K': K_accum_steps,
                'effective_B': effective_B,
                'time_s': loop_elapsed_time,
                'throughput': throughput,
                'mem_used_bytes': current_used_mem
            })
            print(f"    K={K_accum_steps}: time={loop_elapsed_time:.3f}s, throughput={throughput:.1f} samples/s, mem_used={current_used_mem / (1024**3):.2f}GB")

            if current_total_mem > 0 and current_used_mem >= current_total_mem * memory_usage_warning_threshold:
                print(f"    Memory usage ({current_used_mem / (1024**3):.2f}GB) exceeded threshold ({memory_usage_warning_threshold*100}% of {current_total_mem / (1024**3):.2f}GB). Stopping sweep.")
                break
            
            # Heuristic to stop if throughput drops significantly (e.g., by more than 5% of best)
            if K_accum_steps > 1 and throughput < best_throughput * 0.95:
                 print(f"    Throughput ({throughput:.1f}) dropped significantly from best ({best_throughput:.1f}). Stopping sweep.")
                 # break # Optional: uncomment to stop early if throughput degrades

            best_throughput = max(best_throughput, throughput)

        except (RuntimeError, MemoryError) as e: # pragma: no cover
            print(f"  OOM or RuntimeError during sweep_accum at K={K_accum_steps}. Stopping sweep. Error: {e}") # pragma: no cover
            break # Stop sweep if OOM occurs
        except Exception as e: # pragma: no cover
            print(f"  Unexpected error during sweep_accum at K={K_accum_steps}. Stopping sweep. Error: {e}") # pragma: no cover
            break # pragma: no cover
            
    return results

# ───── MAIN WORKFLOW (Example) ─────
def main_batch_optimizer_workflow(
    model_init_fn: Callable[[jax.random.PRNGKey], nnx.Module],
    forward_backward_fn: Callable[[nnx.Module, Any], Tuple[jax.Array, Grads]],
    input_data_shape: Tuple[int, ...],
    target_data_shape: Optional[Tuple[int, ...]] = None,
    rng_seed: int = 0,
    start_batch_size_B_max: int = 8,
    limit_batch_size_B_max: int = 2**12, # e.g., 4096
    grad_var_test_batch_size_B: Optional[int] = None, # Batch size for initial gradient variance test, e.g. min(32, B_max)
    grad_var_repeats: int = 5,
    max_accum_steps_K: int = 16,
    dtype: jnp.dtype = jnp.float32
):
    """Demonstrates the full workflow for finding optimal batch size and accumulation."""
    rng_key_main = jax.random.PRNGKey(rng_seed)
    rng_key_init, rng_key_bmax, rng_key_gvar, rng_key_sweep = jax.random.split(rng_key_main, 4)

    # Initialize model once (closed over by forward_backward_fn if it's a partial or lambda)
    # Or, if forward_backward_fn expects model_instance, it's passed in.
    try:
        print("Initializing model for batch optimization workflow...")
        model_instance = model_init_fn(rng_key_init) # This instance will be used throughout
        print("Model initialized.")
    except Exception as e: # pragma: no cover
        print(f"Failed to initialize model: {e}. Aborting batch optimization.") # pragma: no cover
        return # pragma: no cover

    print("\n1) Finding max batch size (B_max) without OOM…")
    B_max = find_max_batch(
        model_init_fn, # Pass the init_fn, find_max_batch will call it if it needs to re-init (though current impl doesn't)
        forward_backward_fn, 
        input_data_shape, 
        target_data_shape, 
        rng_key_bmax, 
        start_B=start_batch_size_B_max, 
        limit_B=limit_batch_size_B_max,
        dtype=dtype
    )
    if B_max == 0:
        print("→ Max local batch size (B_max) is 0. Cannot proceed with further optimization.")
        return
    print(f"→ Max local batch size (B_max) = {B_max}")

    # Determine batch size for initial gradient variance test
    if grad_var_test_batch_size_B is None: # pragma: no cover
        initial_grad_var_B = min(32, B_max)
    else:
        initial_grad_var_B = min(grad_var_test_batch_size_B, B_max)
    
    if initial_grad_var_B == 0: # pragma: no cover
        print("Cannot run gradient variance estimation with B=0. Skipping.") # pragma: no cover
        sweet_spot_B = B_max # pragma: no cover
    else:
        print(f"\n2) Estimating gradient variance (starting with B={initial_grad_var_B})…")
        base_var = estimate_grad_var(
            model_instance, forward_backward_fn, 
            input_data_shape, target_data_shape, 
            rng_key_gvar, initial_grad_var_B, grad_var_repeats, dtype
        )
        print(f"  var(B={initial_grad_var_B}) = {base_var:.3e}")
        
        sweet_spot_B = initial_grad_var_B
        # Test candidate batch sizes up to B_max for sweet spot
        # Candidates could be, e.g., B_max/4, B_max/2, B_max (if different from initial_grad_var_B)
        candidate_Bs = sorted(list(set([B_max // 4, B_max // 2, B_max])))
        for cand_B in candidate_Bs:
            if cand_B <= initial_grad_var_B or cand_B == 0: continue # Already tested or invalid
            cand_B = min(cand_B, B_max) # Ensure it doesn't exceed B_max
            print(f"  Estimating grad var for candidate B={cand_B}")
            v = estimate_grad_var(
                model_instance, forward_backward_fn, 
                input_data_shape, target_data_shape, 
                jax.random.fold_in(rng_key_gvar, cand_B), # Vary RNG key per candidate
                cand_B, grad_var_repeats, dtype
            )
            print(f"  var(B={cand_B}) = {v:.3e}")
            # Heuristic: if variance doesn't drop too much (e.g., less than half of base_var reduction)
            # and it's a larger batch size, it might be a better sweet spot.
            # This heuristic might need refinement based on actual variance behavior.
            if not jnp.isnan(v) and v >= base_var * 0.5: # If variance is still reasonably high # pragma: no cover
                sweet_spot_B = cand_B # Prefer larger B if variance reduction is not drastic
        print(f"→ Sweet-spot batch size (sweet_spot_B) = {sweet_spot_B}")

    if sweet_spot_B == 0: # pragma: no cover
        print("Sweet-spot batch size is 0. Cannot proceed with accumulation sweep.") # pragma: no cover
        return # pragma: no cover

    print(f"\n3) Timing & memory at sweet-spot batch size (B={sweet_spot_B})…")
    try: # pragma: no cover
        # Single step timing
        key_sweet_spot_time = jax.random.fold_in(rng_key_sweep, sweet_spot_B)
        batch_sweet_spot = make_random_batch(key_sweet_spot_time, sweet_spot_B, input_data_shape, target_data_shape, dtype)
        t0_step = time.perf_counter()
        _, grads_sweet_spot = forward_backward_fn(model_instance, batch_sweet_spot)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads_sweet_spot)
        t_step = time.perf_counter() - t0_step
        del batch_sweet_spot, grads_sweet_spot
        
        used_mem_sweet_spot, total_mem_sweet_spot = get_device_memory_usage()
        print(f"  Time per step = {t_step:.3f}s")
        if total_mem_sweet_spot > 0:
            print(f"  Memory used   = {used_mem_sweet_spot / (1024**3):.2f} GB / {total_mem_sweet_spot / (1024**3):.2f} GB")
        else:
            print(f"  Memory used   = {used_mem_sweet_spot / (1024**3):.2f} GB (Total memory unknown)")
    except Exception as e: # pragma: no cover
        print(f"Error during sweet-spot timing: {e}") # pragma: no cover
        t_step = float('nan') # pragma: no cover
        used_mem_sweet_spot, total_mem_sweet_spot = 0,0 # pragma: no cover

    print(f"\n4) Sweeping gradient accumulation factors (K) with local_B={sweet_spot_B}…")
    accum_stats = sweep_accum(
        model_instance, forward_backward_fn, 
        input_data_shape, target_data_shape, 
        rng_key_sweep, 
        sweet_spot_B, 
        max_accum_steps_K, 
        dtype
    )

    print("\n=== SUMMARY ===")
    print(f"Max local batch size (B_max)        : {B_max}")
    print(f"Sweet-spot batch size (sweet_spot_B): {sweet_spot_B}")
    if not jnp.isnan(t_step):
      print(f"Time per step @ sweet_spot_B      : {t_step:.3f}s")
    if total_mem_sweet_spot > 0:
      print(f"Memory used @ sweet_spot_B        : {used_mem_sweet_spot / (1024**3):.2f}/{total_mem_sweet_spot / (1024**3):.2f} GB")
    else:
      print(f"Memory used @ sweet_spot_B        : {used_mem_sweet_spot / (1024**3):.2f} GB")

    if accum_stats:
        print("\nGradient Accumulation Sweep Results (K_accum | effective_B | time_s | throughput | mem_used_GB):")
        for r in accum_stats:
            print(f"{r['K']:>7} | {r['effective_B']:>11} | {r['time_s']:>8.3f}s | {r['throughput']:>10.1f} | {r['mem_used_bytes']/(1024**3):>11.2f}")
        
        best_result_by_throughput = max(accum_stats, key=lambda x: x['throughput'])
        print(f"\nRecommended based on max throughput:")
        print(f"  Local batch size (for one grad step) : {sweet_spot_B}")
        print(f"  Gradient accumulation steps (K)    : {best_result_by_throughput['K']}")
        print(f"  Effective batch size               : {best_result_by_throughput['effective_B']}")
        print(f"  Achieved throughput                : {best_result_by_throughput['throughput']:.1f} samples/s")
    else: # pragma: no cover
        print("No accumulation sweep results generated.") # pragma: no cover

# Example usage (requires user to define model_init_fn and forward_backward_fn):
if __name__ == "__main__": # pragma: no cover
    # This is a placeholder example. User needs to provide actual implementations.
    # Define a simple NNX model for demonstration
    class SimpleNNXModel(nnx.Module):
        def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
            self.dense = nnx.Linear(din, dout, rngs=rngs)
        def __call__(self, x: jax.Array) -> jax.Array:
            return self.dense(x)

    def my_model_init_fn(rng_key: jax.random.PRNGKey) -> nnx.Module:
        print("Inside my_model_init_fn")
        return SimpleNNXModel(din=10, dout=2, rngs=nnx.Rngs(params=rng_key))

    def my_forward_backward_fn(model_instance: nnx.Module, batch_data: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, Grads]:
        print(f"Inside my_forward_backward_fn with model: {type(model_instance)}, batch shapes: ({batch_data[0].shape}, {batch_data[1].shape})")
        
        def loss_fn_for_grad(mdl, b_input, b_target):
            # Ensure mdl is treated as the model instance for apply, not a PyTree of params
            pred = mdl(b_input)
            return jnp.mean((pred - b_target)**2)
        
        loss, grads = nnx.value_and_grad(loss_fn_for_grad)(model_instance, batch_data[0], batch_data[1])
        return loss, grads

    print("Running batch optimizer workflow example with bfloat16...") # pragma: no cover
    main_batch_optimizer_workflow(
        model_init_fn=my_model_init_fn,
        forward_backward_fn=my_forward_backward_fn,
        input_data_shape=(10,),
        target_data_shape=(2,),
        rng_seed=42,
        start_batch_size_B_max=2,
        limit_batch_size_B_max=1024,
        grad_var_test_batch_size_B=4,
        max_accum_steps_K=4,
        dtype=jnp.bfloat16 # Example: run with bfloat16
    )
    
    print("\nRunning batch optimizer workflow example with float32...") # pragma: no cover
    main_batch_optimizer_workflow(
        model_init_fn=my_model_init_fn,
        forward_backward_fn=my_forward_backward_fn,
        input_data_shape=(10,),
        target_data_shape=(2,),
        rng_seed=43, # Different seed for variety
        start_batch_size_B_max=2,
        limit_batch_size_B_max=1024,
        grad_var_test_batch_size_B=4,
        max_accum_steps_K=4,
        dtype=jnp.float32 # Example: run with float32
    ) 