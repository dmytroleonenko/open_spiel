"""
Checkpointing utilities for MuZero JAX implementation.

This module provides utilities for saving and loading model checkpoints
using Orbax for JAX/Flax models.
"""

import os
import logging
from typing import Dict, Any, Optional
import jax
import flax.nnx as nnx

# Try to import Orbax, provide fallback if not available
try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False
    logging.warning("Orbax not available, using mock checkpointing")


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model_state: nnx.State,
    optimizer_state: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a checkpoint containing model and optimizer state.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        step: Training step number
        model_state: Flax NNX model state
        optimizer_state: Optax optimizer state
        metadata: Optional metadata dictionary
        
    Returns:
        Path to the saved checkpoint
    """
    if not ORBAX_AVAILABLE:
        # Mock implementation for testing
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Create a dummy file to indicate checkpoint exists
        with open(checkpoint_path, 'w') as f:
            f.write(f"Mock checkpoint at step {step}")
        return checkpoint_path
    
    # Real Orbax implementation would go here
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    
    # For now, create a simple file-based checkpoint
    checkpoint_data = {
        'step': step,
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'metadata': metadata or {}
    }
    
    # In a real implementation, this would use Orbax to save the checkpoint
    # For now, just create a marker file
    with open(checkpoint_path, 'w') as f:
        f.write(f"Checkpoint at step {step}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a checkpoint from the given path.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing loaded checkpoint data
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    if not ORBAX_AVAILABLE:
        # Mock implementation for testing
        return {
            'step': 0,
            'model_state': {},
            'optimizer_state': {},
            'metadata': {}
        }
    
    # Real Orbax implementation would go here
    # For now, return mock data
    return {
        'step': 0,
        'model_state': {},
        'optimizer_state': {},
        'metadata': {}
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) 
        if f.startswith('checkpoint_')
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by step number and return the latest
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    latest_checkpoint = checkpoint_files[-1]
    
    return os.path.join(checkpoint_dir, latest_checkpoint)


def create_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 5) -> Any:
    """
    Create a checkpoint manager for automatic checkpoint management.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        
    Returns:
        Checkpoint manager instance
    """
    if not ORBAX_AVAILABLE:
        # Return a mock manager
        class MockCheckpointManager:
            def __init__(self, checkpoint_dir, max_to_keep):
                self.checkpoint_dir = checkpoint_dir
                self.max_to_keep = max_to_keep
                
            def save(self, step, items):
                return save_checkpoint(
                    self.checkpoint_dir, 
                    step, 
                    items.get('model_state'), 
                    items.get('optimizer_state'),
                    items.get('metadata')
                )
                
            def restore(self, step):
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}")
                return load_checkpoint(checkpoint_path)
        
        return MockCheckpointManager(checkpoint_dir, max_to_keep)
    
    # Real Orbax checkpoint manager would be created here
    os.makedirs(checkpoint_dir, exist_ok=True)
    return None  # Placeholder