"""
Verification tests for Task 6.4: Loss Computation Strategy Optimization
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
import functools
import inspect

from open_spiel.python.algorithms.muzero_jax.training import losses as losses_lib
from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig


class TestTask64Verification:
    """Test suite verifying Task 6.4 completion claims."""

    def test_current_trainer_implementation_analysis(self):
        """CRITICAL: Analyze the actual trainer implementation for compliance."""
        
        # Check the actual loss computation function
        func = Learner._compute_total_loss_static
        
        # If properly implemented, it should be JIT compiled with static_argnums
        has_static_args = hasattr(func, '__wrapped__') or hasattr(func, 'static_argnums')
        print(f"JIT with static args: {has_static_args}")
        
        # Check function signature for static loss function parameters
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        has_static_loss_params = 'value_loss_fn' in param_names and 'reward_loss_fn' in param_names
        print(f"Static loss params: {has_static_loss_params}")
        print(f"Current parameters: {param_names}")
        
        # Check if runtime branching exists in the function
        source = inspect.getsource(func)
        
        # Count runtime conditionals
        runtime_value_conditionals = source.count('if config.value_loss_type')
        runtime_reward_conditionals = source.count('if config.reward_loss_type')
        has_per_step_loop = 'for k_idx in range(config.num_unroll_steps' in source
        
        print(f"Runtime value loss conditionals: {runtime_value_conditionals}")
        print(f"Runtime reward loss conditionals: {runtime_reward_conditionals}")
        print(f"Has per-step loop: {has_per_step_loop}")
        
        # Check claims vs reality
        total_runtime_branching = runtime_value_conditionals + runtime_reward_conditionals
        
        print(f"\n=== TASK 6.4 CLAIM VERIFICATION ===")
        print(f"CLAIM: 'Static function selection eliminates runtime branching'")
        print(f"REALITY: {total_runtime_branching} runtime conditionals found")
        print(f"STATUS: {'✅ VERIFIED' if total_runtime_branching == 0 else '❌ FALSE CLAIM'}")
        
        print(f"\nCLAIM: 'Vectorized loss computation eliminates per-step loops'")
        print(f"REALITY: Per-step loop found: {has_per_step_loop}")
        print(f"STATUS: {'❌ FALSE CLAIM' if has_per_step_loop else '✅ VERIFIED'}")
        
        return {
            "has_static_args": has_static_args,
            "has_static_loss_params": has_static_loss_params,
            "signature": param_names,
            "runtime_value_conditionals": runtime_value_conditionals,
            "runtime_reward_conditionals": runtime_reward_conditionals,
            "has_per_step_loop": has_per_step_loop,
            "total_runtime_branching": total_runtime_branching
        }


if __name__ == "__main__":
    test_suite = TestTask64Verification()
    
    print("=" * 60)
    print("TASK 6.4 VERIFICATION REPORT")
    print("=" * 60)
    
    try:
        impl_analysis = test_suite.test_current_trainer_implementation_analysis()
        print(f"ANALYSIS: {impl_analysis}")
    except Exception as e:
        print(f"FAILED: Implementation analysis: {e}")
    
    print("=" * 60)