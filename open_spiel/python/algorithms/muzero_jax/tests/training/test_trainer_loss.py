"""Module docstring."""

import pytest
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.nnx.graph as nnx_graph
import optax

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from open_spiel.python.algorithms.muzero_jax.tests.training.trainer_test_utils import (
    key,
    cfg_flat,
    cfg_img,
    make_model,
    make_cfg,
    OBS_SHAPE_FLAT,
    OBS_SHAPE_IMAGE,
    NUM_ACTIONS,
    NUM_UNROLL_STEPS,
    VALUE_SUPPORT_SCALAR,
    VALUE_SUPPORT_CATEGORICAL,
    REWARD_SUPPORT_SCALAR,
    REWARD_SUPPORT_CATEGORICAL,
    MockNetCfg,
)

# Placeholder for loss tests (e.g., test_loss_static and related functions)

