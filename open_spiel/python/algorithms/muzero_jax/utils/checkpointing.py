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
import orbax.checkpoint as ocp


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
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint data structure
    checkpoint_data = {
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'step': step,
    }
    
    if metadata:
        checkpoint_data['metadata'] = metadata
    
    # Use Orbax checkpointer to save
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    
    checkpointer.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint_data))
    
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
    
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    
    # Load without specifying target structure (unsafe but works for testing)
    checkpoint_data = checkpointer.restore(checkpoint_path)
    
    return checkpoint_data


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
    
    # Look for checkpoint files/directories
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) 
        if f.startswith('checkpoint_')
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by step number and return the latest
    def extract_step(checkpoint_name):
        try:
            return int(checkpoint_name.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoint_files.sort(key=extract_step)
    latest_checkpoint = checkpoint_files[-1]
    
    return os.path.join(checkpoint_dir, latest_checkpoint)


def create_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 5) -> ocp.CheckpointManager:
    """
    Create a real Orbax checkpoint manager for automatic checkpoint management.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        
    Returns:
        Orbax CheckpointManager instance
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,  # Save every step by default
        enable_async_checkpointing=True,
        cleanup_tmp_directories=True,
    )
    
    manager = ocp.CheckpointManager(checkpoint_dir, options=options)
    
    return manager


def save_composite_checkpoint(
    checkpoint_dir: str,
    step: int,
    items: Dict[str, Any]
) -> str:
    """
    Save a composite checkpoint with multiple items.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        step: Training step number
        items: Dictionary of items to save (model_state, optimizer_state, metadata, etc.)
        
    Returns:
        Path to the saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create composite save args
    save_args = {}
    for key, value in items.items():
        if key == 'metadata':
            save_args[key] = ocp.args.JsonSave(value)
        else:
            save_args[key] = ocp.args.StandardSave(value)
    
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    
    checkpointer.save(checkpoint_path, args=ocp.args.Composite(**save_args))
    
    return checkpoint_path


def load_composite_checkpoint(
    checkpoint_path: str,
    item_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Load a composite checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        item_names: Optional list of item names to restore
        
    Returns:
        Dictionary containing loaded checkpoint items
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    
    # Restore without specifying structure (unsafe but works for testing)
    checkpoint_data = checkpointer.restore(checkpoint_path)
    
    return checkpoint_data