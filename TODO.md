# TODO: Long Narde Implementation Plan

## Overview
Implementation of Long Narde rules, based on a copy of "games/backgammon".

**Key Rules:**
1.  **Setup:** White: 15 checkers on point 24. Black: 15 checkers on point 12.
2.  **Movement:** Both move counter-clockwise into home (White: 1–6, Black: 13–18), then bear off.
3.  **Starting:** Each rolls 1 die; higher goes first (White). No doubles on first roll.
4.  **Turns:** Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves possible, skip turn. If only one die usable, use the higher value. Doubles grant 4 moves.
5.  **Head Rule:** Only 1 checker may leave the head per turn, except on the first turn if a double 6, 4, or 3 is rolled, allowing 2 checkers from the head.
6.  **Bearing Off:** Allowed once all checkers reach home. Use exact or higher rolls.
7.  **Ending/Scoring:** Mars (2 points) if loser bore off none; Oin (1 point) otherwise. `allow_last_roll_tie_` parameter enables optional tie rule.
8.  **Block (Bridge) Rule:** Forming a contiguous block of 6 checkers is illegal unless at least 1 opponent checker is ahead of the block.

## Current Status & Architecture

*   **Core Implementation:** All the rules are implemented.
*   **Code Structure:** Successfully modularized from monolithic `long_narde.cc` into `_state.cc`, `_moves.cc`, `_encoding.cc`, `_validation.cc`, `_legal_actions.cc`, `_api.cc`, `_utils.cc`, `_game.cc`.

## Plan

## Plan: AlphaZero JAX Implementation

**Phase 1: Project Setup and Basic JAX Model Definition**

[DONE] 1.  **Create Directory Structure**:
    *   Create file: `open_spiel/python/examples/alpha_zero_jax.py`
    *   Create directory: `open_spiel/python/algorithms/alpha_zero_jax/`
    *   Create file: `open_spiel/python/algorithms/alpha_zero_jax/__init__.py` (empty)
    *   Create file: `open_spiel/python/algorithms/alpha_zero_jax/model_jax.py`
    *   Create file: `open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py`

[DONE] 2.  **Define `ConfigJAX` in `alpha_zero_jax.py`**:
    *   Copy `Config` namedtuple from `open_spiel/python/algorithms/alpha_zero/alpha_zero.py` to `alpha_zero_jax.py`.
    *   Rename to `ConfigJAX`.
    *   Add `master_seed: int` field for JAX PRNG.
    *   Verify other fields are suitable (e.g., `learning_rate`, `weight_decay`, `train_batch_size`, `replay_buffer_size`, `nn_model`, `nn_width`, `nn_depth`).

[DONE] 3.  **Define `TrainInputJAX` in `model_jax.py`**:
    *   `import collections`
    *   Create `TrainInputJAX = collections.namedtuple("TrainInputJAX", "observation legals_mask policy_target value_target")`.
    *   Add a static method `stack(train_inputs)` to `TrainInputJAX` (similar to `model_lib.TrainInput.stack`) to batch a list of `TrainInputJAX` objects into NumPy arrays.

[DONE] 4.  **Implement Basic MLP Model in `model_jax.py` (`MLP_JAX`)**:
    *   Imports: `import jax`, `import jax.numpy as jnp`, `import flax.linen as nn`.
    *   Define class `MLP_JAX(nn.Module)`:
        *   Fields: `nn_width: int`, `nn_depth: int`, `output_size: int`.
        *   `@nn.compact def __call__(self, x: jax.Array, training: bool)`:
            *   Torso: Loop `self.nn_depth` times: `x = nn.Dense(features=self.nn_width)(x)`, `x = nn.relu(x)`.
            *   Policy head: `policy_logits = nn.Dense(features=self.output_size, name="policy_head")(x)`.
            *   Value head: `value_hidden = nn.Dense(features=self.nn_width, name="value_hidden")(x)`, `value_hidden = nn.relu(value_hidden)`, `value_output = nn.Dense(features=1, name="value_output")(value_hidden)`, `value_output = nn.tanh(value_output)`.
            *   Return `(policy_logits, value_output)`.

[DONE] 5.  **Implement Model Initialization Helper in `model_jax.py` (`init_flax_model_and_variables`)**:
    *   Function `init_flax_model_and_variables(key: jax.random.PRNGKey, config: ConfigJAX, game: pyspiel.Game)`:
        *   `observation_shape = game.observation_tensor_shape()`.
        *   `output_size = game.num_distinct_actions()`.
        *   If `config.nn_model == "mlp"`: `model = MLP_JAX(nn_width=config.nn_width, nn_depth=config.nn_depth, output_size=output_size)`.
        *   Else (add more models later): `raise ValueError(f"Unsupported model type: {config.nn_model}")`.
        *   `dummy_input = jnp.zeros((1, *observation_shape), dtype=jnp.float32)`.
        *   `variables = model.init(key, dummy_input, training=False)`.
        *   Return `model` (Flax module instance) and `variables` (dictionary, e.g., `{'params': ...}`).

**Phase 2: MCTS Evaluator for JAX Model**

[DONE] 1.  **Implement `AlphaZeroEvaluatorJAX` in `evaluator_jax.py`**:
    *   Imports: `import numpy as np`, `import jax`, `from open_spiel.python.algorithms import mcts`, `import pyspiel`.
    *   Class `AlphaZeroEvaluatorJAX(mcts.Evaluator)`:
        *   `__init__(self, game: pyspiel.Game, model: nn.Module, variables: dict)`: Store `game`, `model`, `variables`.
        *   `_inference(self, state: pyspiel.State)`:
            *   `obs_tensor = np.expand_dims(state.observation_tensor(), 0)`.
            *   `legal_actions_mask = np.expand_dims(state.legal_actions_mask(), 0)`.
            *   `policy_logits_batch, value_output_batch = self.model.apply(self.variables, obs_tensor, training=False)`.
            *   `value_scalar = np.array(value_output_batch[0, 0])`.
            *   `policy_logits = np.array(policy_logits_batch[0])`.
            *   `policy_probs = jax.nn.softmax(policy_logits)`.
            *   `masked_policy = policy_probs * legal_actions_mask[0]`.
            *   `masked_policy /= np.sum(masked_policy)`.
            *   Return `(value_scalar, masked_policy)`.
        *   `evaluate(self, state: pyspiel.State)`:
            *   `value, _ = self._inference(state)`.
            *   Return `np.array([value, -value])` (assuming two-player zero-sum).
        *   `prior(self, state: pyspiel.State)`:
            *   If `state.is_chance_node()`: return `state.chance_outcomes()`.
            *   `_, policy = self._inference(state)`.
            *   Return `[(action, policy[action]) for action in state.legal_actions()]`.

**Phase 3: Learner Implementation with JAX**

[DONE] 1.  **Adapt `learner` Function in `alpha_zero_jax.py`**:
    *   Initial imports: `import jax`, `import jax.numpy as jnp`, `import optax`, `from flax.training import checkpoints`.
    *   Copy structure of `learner` from `open_spiel/python/algorithms/alpha_zero/alpha_zero.py`.
    *   Helper classes `TrajectoryState`, `Trajectory`, `Buffer` can be copied from `alpha_zero.py` to `alpha_zero_jax.py` or a shared util.
    *   `TrainInputJAX` definition from `model_jax.py` will be used.
    *   Initialize JAX model and optimizer:
        *   `learner_key, init_key = jax.random.split(main_key_for_learner) # main_key_for_learner passed to learner`
        *   `flax_model, variables = model_jax.init_flax_model_and_variables(init_key, config, game)`.
        *   `optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)`.
        *   `opt_state = optimizer.init(variables['params'])`.
    *   Replay buffer (`Buffer` class) stores `TrainInputJAX` instances.

[DONE] 2.  **Implement `train_step` in `learner` (as a JITted function)**:
    *   `@jax.jit def train_step_fn(current_variables, current_opt_state, batch_observations, batch_legals_masks, batch_policy_targets, batch_value_targets)`:
        *   Define `loss_and_grad_inner_fn(params)`:
            *   `apply_vars = {'params': params}`
            *   If 'batch_stats' in `current_variables`: `apply_vars['batch_stats'] = current_variables['batch_stats']`.
            *   `preds_and_state = flax_model.apply(apply_vars, batch_observations, training=True, mutable=['batch_stats'] if 'batch_stats' in apply_vars else None)`.
            *   If model has batch_stats: `(policy_logits, value_preds), updated_model_state = preds_and_state`. Else: `(policy_logits, value_preds) = preds_and_state`, `updated_model_state = None`.
            *   `policy_loss = optax.softmax_cross_entropy(logits=policy_logits, labels=batch_policy_targets)`.
            *   `policy_loss = jnp.mean(policy_loss * batch_legals_masks.any(axis=1))`.
            *   `value_loss = optax.squared_error(predictions=jnp.squeeze(value_preds, axis=-1), targets=jnp.squeeze(batch_value_targets, axis=-1))`
            *   `value_loss = jnp.mean(value_loss)`.
            *   `total_loss = policy_loss + value_loss`.
            *   Return `total_loss`, `(updated_model_state, policy_loss, value_loss)`.
        *   `(loss_val, (new_model_state, p_loss, v_loss)), grads = jax.value_and_grad(loss_and_grad_inner_fn, has_aux=True)(current_variables['params'])`.
        *   `updates, new_opt_state = optimizer.apply_updates(grads, current_opt_state, current_variables['params'])`.
        *   `new_params = optax.apply_updates(current_variables['params'], updates)`.
        *   `new_variables = current_variables.copy()`
        *   `new_variables['params'] = new_params`.
        *   If `new_model_state` and 'batch_stats' in `new_model_state`: `new_variables['batch_stats'] = new_model_state['batch_stats']`.
        *   Return `new_variables`, `new_opt_state`, `loss_val`, `p_loss`, `v_loss`.
    *   In the `learn` loop of the `learner` function:
        *   Sample data using `replay_buffer.sample()`.
        *   Use `TrainInputJAX.stack(batch_data)` to get batched numpy arrays. Convert to `jnp.array`.
        *   `variables, opt_state, total_loss, policy_loss, value_loss = train_step_fn(variables, opt_state, batch_obs_jnp, batch_legals_jnp, batch_policy_jnp, batch_value_jnp)`.
        *   Log losses.

[DONE] 3.  **Implement JAX Checkpointing in `learner`**:
    *   Checkpoint directory: `ckpt_dir = os.path.join(config.path, "checkpoints_jax")`. `os.makedirs(ckpt_dir, exist_ok=True)`.
    *   Saving:
        *   `save_target = {'variables': variables, 'opt_state': opt_state}`.
        *   `step_prefix = "checkpoint_"`
        *   `checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=save_target, step=step, prefix=step_prefix, overwrite=True, keep=config.checkpoint_freq if config.checkpoint_freq > 0 else float('inf'))`.
        *   To broadcast 'latest', save another copy: `checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=save_target, step="latest", prefix="", overwrite=True)`.
        *   The path broadcast would be `os.path.join(ckpt_dir, "latest")` or `os.path.join(ckpt_dir, f"{step_prefix}{step}")`.

**Phase 4: Actor and Evaluator Processes with JAX**

[DONE] 1.  **Adapt `actor` Function in `alpha_zero_jax.py`**:
    *   Copy structure from `alpha_zero.py`. `watcher` decorator can be copied too.
    *   Derive `actor_key` from a main PRNG key passed to `actor` (e.g., `actor_key = jax.random.fold_in(main_actor_key, num_for_actor)`).
    *   Initialize JAX model: `flax_model, variables = model_jax.init_flax_model_and_variables(actor_key, config, game)`.
    *   Initialize evaluator: `az_evaluator = evaluator_jax.AlphaZeroEvaluatorJAX(game, flax_model, variables)`.
    *   `update_checkpoint` function for JAX:
        *   `target_to_restore = {'variables': variables}` (opt_state not needed by actor).
        *   `restored_state = checkpoints.restore_checkpoint(ckpt_dir=path_to_checkpoint_file_or_dir, target=target_to_restore)`. `path_to_checkpoint_file_or_dir` should be the directory if using `step` argument, or full path to specific file. If using "latest" symlink, path is `os.path.join(ckpt_dir, "latest")`.
        *   `variables = restored_state['variables']`.
        *   Update `az_evaluator.variables = variables`.
    *   `_play_game` (copied from `alpha_zero.py`) should work with the new `az_evaluator`. `TrajectoryState` and `Trajectory` are also copied.

[DONE] 2.  **Adapt `evaluator` Function in `alpha_zero_jax.py`**:
    *   Similar JAX initialization and checkpoint loading as `actor`.

**Phase 5: Top-Level Script and Integration**

[DONE] 1.  **Adapt `alpha_zero_jax` (main orchestrating function) in `alpha_zero_jax.py`**:
    *   Copy structure from `alpha_zero` function in `alpha_zero.py`.
    *   Use `ConfigJAX`.
    *   Initialize master JAX PRNG key: `main_key = jax.random.PRNGKey(config.master_seed)`.
    *   Split keys: `process_keys = jax.random.split(main_key, config.actors + config.evaluators + 1)`. `learner_key = process_keys[0]`, `actor_keys = process_keys[1:1+config.actors]`, `evaluator_keys = process_keys[1+config.actors:]`.
    *   Pass appropriate keys to spawned processes (e.g., `kwargs={"game": game, "config": config, "num": i, "prng_key": actor_keys[i]}`).
    *   `spawn.Process` calls point to JAX-adapted `actor`, `learner`, `evaluator`.

[DONE] 2.  **Add JAX Imports to `alpha_zero_jax.py`**: `import jax`, `import jax.numpy as jnp`, `import os`, `from .algorithms.alpha_zero_jax import model_jax`, `from .algorithms.alpha_zero_jax import evaluator_jax`, `from open_spiel.python.utils import spawn`, etc. (adjust paths based on where it's run from).
[DONE] 3.  **Testing**: Start with "tic_tac_toe". Test model init, inference, one train step, actor game generation, learner training, checkpoint save/load.

**Phase 6: Implement ResNet and Conv2D Models (`model_jax.py`)**

[DONE] 1.  **Integrate `jax-resnet` for `ResNet_JAX`**:
    *   [DONE] Utilize `ConvBlock`, `ResNetStem`, `ResNetBlock`, and `ResNetBottleneckBlock` by importing them into `model_jax.py`. The source for these components is the `n2cholas/jax-resnet` implementation, which has been cloned into the workspace at `open_spiel/python/algorithms/jax-resnet/jax_resnet/`.
    *   [DONE] **Add clear attribution to `n2cholas/jax-resnet` in comments.**
    *   Define base `ResNet_JAX(nn.Module)`:
        *   [DONE] Fields: `block_cls: nn.Module`, `stage_sizes: list[int]`, `hidden_sizes: list[int]`, `output_size: int`, `stem_cls: nn.Module = ResNetStem`, `conv_block_cls: nn.Module = ConvBlock` (from `jax-resnet`), `norm_cls: nn.Module = partial(nn.BatchNorm, momentum=0.9)`.
        *   [DONE] `@nn.compact def __call__(self, x: jax.Array, training: bool)`:
            *   Instantiate `stem_cls` and `block_cls` with appropriate partials (e.g., passing `conv_block_cls` which itself is partialled with `norm_cls`).
            *   Build ResNet torso using stem and looping through `stage_sizes` with `block_cls`. Pass `training` to `BatchNorm` via `ConvBlock` and other modules as needed.
            *   Global average pooling: `x = jnp.mean(x, axis=(1, 2))`.
            *   Policy head: `policy_logits = nn.Dense(features=self.output_size, name="policy_head")(x)`.
            *   Value head: `value_hidden = nn.Dense(features=self.hidden_sizes[-1], name="value_hidden")(x)`, `value_hidden = nn.relu(value_hidden)`, `value_output = nn.Dense(features=1, name="value_output")(value_hidden)`, `value_output = nn.tanh(value_output)`.
            *   Return `(policy_logits, value_output)`.
    *   Update `init_flax_model_and_variables` for different ResNet variants:
        *   [DONE] For a generic "resnet" type in `ConfigJAX.nn_model`, map `config.nn_width` to a base `hidden_size` (e.g., 64) and `config.nn_depth` to a simple `stage_sizes` configuration (e.g., `[config.nn_depth // N] * N` for some N, or use a default like ResNet18/34 structure if depth isn't directly specified).
        *   Add specific model types to `ConfigJAX.nn_model` enum (or allow string matching) and `init_flax_model_and_variables` to instantiate various ResNet architectures from `jax-resnet`:
            *   [DONE] ResNet18_JAX: Use `ResNetBlock`, `STAGE_SIZES[18]`
            *   [DONE] ResNet34_JAX: Use `ResNetBlock`, `STAGE_SIZES[34]`
            *   [DONE] ResNet50_JAX: Use `ResNetBottleneckBlock`, `STAGE_SIZES[50]`
            *   [DONE] ResNet101_JAX: Use `ResNetBottleneckBlock`, `STAGE_SIZES[101]`
            *   [DONE] ResNet152_JAX: Use `ResNetBottleneckBlock`, `STAGE_SIZES[152]`
            *   [DONE] ResNet200_JAX: Use `ResNetBottleneckBlock`, `STAGE_SIZES[200]`
            *   [DONE] WideResNet50_JAX: Modify `hidden_sizes` and `expansion` for `ResNetBottleneckBlock`.
            *   [DONE] WideResNet101_JAX: Modify `hidden_sizes` and `expansion` for `ResNetBottleneckBlock`.
            *   [DONE] ResNeXt50_JAX: Use `ResNetBottleneckBlock` with `groups` and `base_width`.
            *   [DONE] ResNeXt101_JAX: Use `ResNetBottleneckBlock` with `groups` and `base_width`.
            *   [DONE] ResNetD variants (ResNetD18_JAX, D34_JAX, D50_JAX, D101_JAX, D152_JAX, D200_JAX): Use `ResNetDStem`, `ResNetDBlock` / `ResNetDBottleneckBlock`.
                *   [DONE] Refactor `ResNet_JAX` to accept `stem_constructor`, `block_constructor`, and `block_kwargs`.
                *   [DONE] Implement ResNetD18_JAX.
                *   [DONE] Implement ResNetD34_JAX.
                *   [DONE] Implement ResNetD50_JAX.
                *   [DONE] Implement ResNetD101_JAX.
                *   [DONE] Implement ResNetD152_JAX.
                *   [DONE] Implement ResNetD200_JAX.
            *   [DONE] ResNeSt variants (ResNeSt50Fast_JAX, ResNeSt50_JAX, ResNeSt101_JAX, etc.): Use `ResNetDStem`, `ResNeStBottleneckBlock` (potentially importing `SplAtConv2d` from `jax_resnet.splat`). This might involve more complex integration of `splat.py`.
                *   [DONE] ResNeSt50Fast_JAX: Use `ResNetDStem`, `ResNeStBottleneckBlock` with `avg_pool_first=True`.
                *   [DONE] ResNeSt50_JAX: Use `ResNetDStem`, `ResNeStBottleneckBlock`.
                *   [DONE] ResNeSt101_JAX: Use `partial(ResNetDStem, stem_width=64)`, `ResNeStBottleneckBlock`.
            *   [DONE] Update `ConfigJAX` in `alpha_zero_jax.py` (example script) to include fields for these new ResNet variants as they are added (e.g. `resnet_d_stem_width`, `resnest_radix`, `resnest_avg_pool_first`, etc. or more generic `block_kwargs` if preferred for user flexibility, though explicit is clearer).
                *   [DONE] Added `resnet_depth_config`, `resnet_stem_callable_name`, `resnet_stem_kwargs`, `resnet_block_callable_name`, `resnet_block_kwargs` to `ConfigJAX` namedtuple.
                *   [DONE] Updated `init_flax_model_and_variables` to use these for `nn_model="resnet"`.
                *   [DONE] Demonstrated pattern for specific models (e.g., "resnet18", "resnest50fast") to allow overrides from `config.resnet_stem_kwargs` and `config.resnet_block_kwargs`.
                *   [DONE] Propagate override pattern to all other named ResNet model types in `init_flax_model_and_variables`.
                *   [DONE] Add example usage in `alpha_zero_jax.py` `main` showing how to populate these new `ConfigJAX` fields.
        *   [DONE] Ensure `variables` will correctly include 'params' and 'batch_stats' for all ResNet variants.
    *   [DONE] Ensure `train_step_fn` correctly handles `batch_stats` (gets them from `apply_vars`, passes them to `apply`, and gets them back from `updated_model_state`).

[DONE] 2.  **Implement `Conv2D_JAX` Model**:
    *   [DONE] `Conv2D_JAX(nn.Module)`.
    *   [DONE] Fields: `nn_width: int`, `nn_depth: int`, `output_size: int`.
    *   [DONE] `@nn.compact def __call__(self, x: jax.Array, training: bool)`:
        *   Torso: `nn_depth` layers of `ConvBlock` (copied/adapted from `jax-resnet/jax_resnet/common.py`, ensuring `norm_cls` gets `training` status via `use_running_average=not training`).
        *   `nn.Flatten()` layer after conv blocks.
        *   Policy and Value heads.
    *   [DONE] Update `init_flax_model_and_variables`. Also uses `BatchNorm`.

**Phase 7: Refinement and Finalization**

[DONE] 1.  **Logging**: Adapt `FileLogger` from `alpha_zero.py` or use Python's `logging`.
[DONE] 2.  **Configuration Documentation**: Document `ConfigJAX` fields clearly.
[DONE] 3.  **Comments and Docstrings**: Add/update for JAX-specifics.
[DONE] 4.  **Policy Target in `train_step`**: Revisit `optax.softmax_cross_entropy_with_integer_labels` if policy targets are not one-hot or if masking needs refinement. If policy target is a probability distribution, use `optax.softmax_cross_entropy`.
[DONE] 5.  **Dependencies**: Add `jax`, `flax`, `optax` to `requirements.txt` or setup.
[DONE] 6.  **Entry Point**: Make `alpha_zero_jax.py` executable with `absl.app` and `absl.flags` similar to `alpha_zero.py`.

**Phase 8: Post-Review Fixes and Refinements**

[DONE] 1. **Correct Policy Loss Calculation in Learner (`alpha_zero_jax.py`)**  
    * Changed from `softmax_cross_entropy_with_integer_labels` + `argmax` to  
      `optax.softmax_cross_entropy(logits=policy_logits, labels=batch_policy_targets)`.

[DONE] 2. **Improve Policy Loss Masking in Learner (`alpha_zero_jax.py`)**  
    * **Original Critique**: The implementation does not fully match the description. The current code applies masking at the sample level using `batch_legals_masks.any(axis=1)`, which zeros out policy loss for samples with no legal actions, rather than applying element-wise masking per action. Additionally, there is no normalization by the number of legal actions as claimed. (Citation: `open_spiel/python/examples/alpha_zero_jax.py`, lines 846-851)
    * **Resolution (Verified)**:
        *   Models (`MLP_JAX`, `ResNet_JAX`, `Conv2D_JAX` in `model_jax.py`) updated to accept `legals_mask` in their `__call__` method and apply it to `policy_logits` by setting illegal action logits to `-jnp.inf`.
        *   The `learner` in `alpha_zero_jax.py` now correctly passes `legals_mask` (as `batch_legals_masks`) to the model's `apply` method using a keyword argument.
        *   With logits for illegal actions set to `-jnp.inf` by the model, `optax.softmax_cross_entropy` will correctly handle them (their contribution to loss will be effectively zero if their target probability is zero).
        *   The existing sample-level masking in the learner (`policy_loss = policy_loss * batch_legals_masks.any(axis=1)`) correctly zeros out the policy loss for any sample that has no legal actions at all. This is a distinct and valid step.
        *   The "normalization by the number of legal actions per sample" mentioned in the original TODO item description is not implemented as it's generally not required when logits are properly masked and standard cross-entropy loss is used, followed by a mean over the batch. The current approach is standard.

[DONE] 3. **Implement "latest" Checkpoint Saving in Learner (`alpha_zero_jax.py`)**  
    * After saving each step checkpoint, invoke  
      ```python
      checkpoints.save_checkpoint(ckpt_dir, target, step="latest", prefix="", overwrite=True)
      ```  
      so actors/evaluators can always load `latest`.

[DONE] 5. **Align Learner Checkpoint `keep` Logic (`alpha_zero_jax.py`)**  
    * **Original Critique**: The implementation does not fully match the description. The `keep` parameter for step-specific checkpoints uses `config.checkpoint_freq` without the conditional fallback to `float('inf')` if `config.checkpoint_freq <= 0`. (Citation: `open_spiel/python/examples/alpha_zero_jax.py`, lines 964-969)
    * **Resolution (Verified)**:
        *   The previous description of the fix ("Now uses `keep=config.checkpoint_freq if config.checkpoint_freq > 0 else float('inf')`") was potentially misleading regarding the overall behavior.
        *   The current code in `alpha_zero_jax.py` (lines 980-995) implements the following behavior for step-specific checkpoints:
            *   If `config.checkpoint_freq > 0`: A checkpoint is saved every `config.checkpoint_freq` steps, and `keep=config.checkpoint_freq` such checkpoints are retained.
            *   If `config.checkpoint_freq <= 0`: No periodic step-specific checkpoints are saved (this block of code is skipped). Only the "latest" checkpoint is saved (handled separately).
        *   This implementation is a standard and reasonable approach to managing checkpoint frequency and retention. The critique was valid in that the code did not match the literal prior description, but the code's behavior itself is sound for this interpretation.

[DONE] [CHECKED] 6. **Add caching layer to `AlphaZeroEvaluatorJAX` (`evaluator_jax.py`).**
    * Integrate an LRU cache (e.g., `open_spiel.python.utils.lru_cache.LRUCache`) around `_inference`.
    * Assessment: Missing caching. Unlike the TF `AlphaZeroEvaluator`, `AlphaZeroEvaluatorJAX` does not initialize or use an `LRUCache`, so repeated `_inference` calls cannot benefit from caching. We should add a cache (e.g., `self._cache = lru_cache.LRUCache(cache_size)`) and wrap `_inference` with it.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py lines 16–20; open_spiel/python/algorithms/alpha_zero/evaluator.py lines 20–23

[DONE] [CHECKED] 7. **Enforce game-type validations in `AlphaZeroEvaluatorJAX` constructor (`evaluator_jax.py`).**
    * Check `num_players()==2`, `reward_model==TERMINAL`, `dynamics==SEQUENTIAL`.
    * Assessment: Recommended refinement. The JAX evaluator currently lacks validations for game type, which is a critical operational constraint. Adding these validations to the evaluator's constructor improves robustness and makes its operational constraints explicit, aligning with the TF reference.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py lines 1–15

[DONE] [CHECKED] 8. **Modify `Conv2D_JAX` and `ResNet_JAX` to match TF head structures (`model_jax.py`).**
    * Use 1×1 conv + BN + ReLU before flatten+Dense for both policy and value heads.
    * Assessment: Recommended refinement. The JAX models currently have different head structures compared to the TF reference. TF uses a 1x1Conv->BN->ReLU->Flatten sequence before final dense layers in both policy and value heads for ResNet/Conv2D. JAX ResNet also uses GlobalAvgPool instead of Flatten. Aligning these head structures with the TF reference is recommended for architectural fidelity.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 26–34, 38–47

[DONE] [CHECKED] 9. **Ensure explicit flattening or reshaping in `MLP_JAX` for non-flat observations (`model_jax.py`).**
    * Add `x = x.reshape((x.shape[0], -1))` at the start of `__call__`.
    * Assessment: Recommended refinement. The JAX `MLP_JAX` does not explicitly flatten its input. If observations are multi-dimensional, `nn.Dense` will operate on the last dimension only, which is not typical MLP behavior for game states and differs from the TF reference. Explicit flattening is required.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 15–23

[DONE] [CHECKED] 10. **Unify RNG handling for determinism (`alpha_zero_jax.py`).**
    * Wire `config.master_seed` into NumPy, `random`, and `jax.random`; remove mixed RNG use.
    * Assessment: Recommended refinement. JAX PRNG keys are used for model initialization, but Python's `random` (for replay buffer) and `numpy.random` (for action selection, chance nodes in `_play_game`) are not explicitly seeded from `config.master_seed`. This needs to be done for determinism. Also, `_play_game` mixes `np.random.RandomState()` with global `np.random.choice`.
    * Citation: open_spiel/python/examples/alpha_zero_jax.py lines 136–142, 278–281

[DONE] [CHECKED] 11. **Restore `quiet` logging discipline (`alpha_zero_jax.py`, actors/evaluators).**
    * Replace unconditional `print()` with `logger.opt_print` or suppress when `quiet=True`.
    * Assessment: Recommended refinement. Numerous unconditional `print()` statements exist in `alpha_zero_jax.py` (main script, learner, actor, evaluator, watcher) that bypass `config.quiet`. These should be replaced with `logger.print()` or `logger.opt_print()` to ensure console output respects the quiet flag. The learner's `logger.also_to_stdout=True` is fine, but direct `print()` calls should still be converted.
    * Citation: open_spiel/python/examples/alpha_zero_jax.py lines 801–805

[DONE] [CHECKED] 12. **Harden `watcher` fallback log-path logic (`alpha_zero_jax.py`).**
    * Assessment: Recommended refinement. The main `alpha_zero_jax` script already ensures `config.path` is set (to user input or a temp dir) before worker processes (and thus their watchers) are initialized. The watcher's current fallback to `logger_path = "."` if `config.path` is missing is weak but should ideally not be hit in the standard workflow. Hardening could involve the watcher erroring if `config.path` is not provided, as it should be by the caller.
    * Citation: open_spiel/python/examples/alpha_zero_jax.py lines 117–127

[DONE] [CHECKED] 13. **Provide legacy-compatible defaults for new ResNet fields (`alpha_zero_jax.py`).**
    * Fall back to TF-reference 256-filter, 20-block ResNet when optional `resnet_*` fields are absent.
    * Assessment: Recommended refinement. Currently, if `config.nn_model == "resnet"`, the JAX `init_flax_model_and_variables` expects `config.resnet_depth_config` & other specific ResNet fields. It does not fall back to using `config.nn_width` and `config.nn_depth` to construct a default TF-like ResNet (e.g., 256 filters, 20 blocks). This fallback should be implemented for ease of use and compatibility with expectations from the TF version.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 148–153

[DONE] [CHECKED] 14. **Align default ResNet capacity with TF reference (`model_jax.py`).**
    * Use 256-filter "AlphaGo Zero" layout as default unless a lighter variant is explicitly requested.
    * Assessment: Recommended refinement. If Item 13 implements a fallback for `nn_model="resnet"` using `config.nn_width` and `config.nn_depth`, then this item requires that the default values for `config.nn_width` and `config.nn_depth` (e.g., from script flags) are set to 256 and 20 respectively for this fallback scenario. Lighter variants (e.g., "resnet18") are already available as explicit choices and should not be altered to 256 filters. Action: Ensure default config values align if generic "resnet" is chosen.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 148–153

[DONE] [CHECKED] 15. **Expose evaluator cache size and reinstate `lru_cache` utility (`evaluator_jax.py`).**
    * Add `cache_size` parameter (default `2**16`) and wire through `cache_info()`/`clear_cache()`.
    * Assessment: Recommended refinement. This is a direct follow-up to Item 6 (add caching). The JAX evaluator needs to implement an LRU cache, its constructor should accept `cache_size` (default `2**16`), and `cache_info()`/`clear_cache()` methods must be made functional. This aligns with the TF reference and is crucial for performance. `ConfigJAX` should also be updated with an `evaluator_cache_size` field.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py lines 1–15

[DONE] [CHECKED] 16. **Ensure policy-masking strategy is consistent across train & inference.**
    * Decide whether masking lives in learner or evaluator and consolidate logic.
    * Assessment: Recommended refinement. Policy masking is inconsistent. Inference masks probabilities post-softmax. Training (current) relies on target distribution and doesn't mask logits pre-softmax, allowing illegal logits to affect normalization. TF reference masks logits pre-softmax within the model definition. Recommendation: Modify JAX models to accept `legals_mask` and mask `policy_logits` (to -large_negative_val for illegal actions) internally before they are returned/used by loss/inference. This centralizes logic and aligns training/inference behavior more closely.
    * Citation: open_spiel/python/examples/alpha_zero_jax.py lines 916–925; open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py lines 23–30

[DONE] [CHECKED] 17. **Add Batch Normalization to `MLP_JAX` torso (`model_jax.py`).**
    * Insert `nn.BatchNorm(use_running_average=not training)` after each Dense layer.
    * Assessment: Potential enhancement, not TF MLP parity fix. The TF reference MLP model does not use BatchNorm in its torso, nor does the current JAX `MLP_JAX`. Adding BatchNorm would be an architectural modification, possibly beneficial for deeper MLPs, but not required for matching the TF MLP. If implemented, `batch_stats` handling for `MLP_JAX` would be needed.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 18–23

[DONE] [CHECKED] 18. **Implement JIT compilation for inference in `AlphaZeroEvaluatorJAX` (`evaluator_jax.py`).**
    * Annotate `_inference` or model apply with `@jax.jit` for faster repeated calls.
    * Assessment: Recommended refinement. JIT compilation is not currently applied to the inference path in `AlphaZeroEvaluatorJAX`. This is a critical performance optimization in JAX and should be implemented (e.g., by JIT-compiling a helper function that calls `model.apply` with `training=False`).
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py lines 23–30

[DONE] [CHECKED] 19. **Confirm `init_flax_model_and_variables` wiring for all `nn_model` options (`model_jax.py`).**
    * Ensure each `config.nn_model` value (mlp, conv2d, resnet variants) is correctly instantiated and tested.
    * Assessment: Mostly confirmed from code structure. `init_flax_model_and_variables` has paths for "mlp", "conv2d", many named ResNet variants (using `jax_resnet` structures), and a generic "resnet" (requiring `config.resnet_*` fields). Wiring seems plausible. Key missing piece is Item 13 (fallback for generic "resnet"). Full confirmation requires runtime testing of all model options.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 1–80

[DONE] [CHECKED] 20. **Ensure explicit `jnp.asarray` conversion of inputs in `AlphaZeroEvaluatorJAX` (`evaluator_jax.py`).**
    * Convert `obs_tensor` and `legal_actions_mask` to JAX arrays before model application.
    * Assessment: Recommended refinement. Currently, NumPy arrays are passed to `model.apply`, and Flax handles conversion. Explicit `jnp.asarray` conversion is good JAX practice for clarity and can be beneficial when JIT-compiling the inference step (related to TODO 18). Not strictly a bug, but a good enhancement.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/evaluator_jax.py lines 12–20

[DONE] [CHECKED] 21. **Document and Review ResNet Torso Output Discrepancy (`model_jax.py`).**
    * The JAX `ResNet_JAX` uses global average pooling (`jnp.mean`) after the convolutional torso.
    * The TensorFlow reference `Model`


[DONE] 22. **Add Missing `add` Method to `Trajectory` Class in JAX Implementation.**
    * Implement the `add` method in `Trajectory` class to match TensorFlow's convenience method for adding states to trajectories.
    * Assessment: Recommended for code readability and consistency. This omission could lead to less readable code in the actor function where trajectories are built.
    * Citation: open_spiel/python/examples/alpha_zero_jax.py lines 94-99; open_spiel/python/algorithms/alpha_zero/alpha_zero.py lines 79-81

[DONE] 23. **Simplify ResNet Variants to Essential Set or Document Extensively.**
    * Reduce the number of ResNet variants in `model_jax.py` to a core set (e.g., ResNet18, ResNet50) that aligns with typical AlphaZero needs, or provide extensive documentation for each variant's use case and performance impact.
    * Assessment: Recommended for maintainability. The extensive array of ResNet variants adds complexity that may be challenging for maintenance, especially for junior developers. Simplification or detailed documentation would improve usability.
    * Citation: open_spiel/python/algorithms/alpha_zero_jax/model_jax.py lines 1-200

[DONE] 24. **Harden `config.path` Requirement in `watcher` Decorator.**
    * Update the `watcher` decorator to raise a `ValueError` if `config.path` is not set, ensuring file logging is always possible or explicitly handled.
    * Assessment: Recommended for robustness. The current JAX implementation attempts to handle missing `config.path` with error messages but doesn't enforce a solution, which could lead to logging failures.
    * Citation: open_spiel/python/examples/alpha_zero_jax.py lines 122-167

[DONE] 25. **Relocate JAX Main Script to Align with TensorFlow Structure.**
    * Move `alpha_zero_jax.py` from `open_spiel/python/examples/` to `open_spiel/python/algorithms/alpha_zero_jax/` to match the TensorFlow implementation's placement under `algorithms/`.
    * Assessment: Recommended for consistency. This organizational deviation affects discoverability and perceived status of the JAX implementation within the OpenSpiel project structure. Since the file is in a git repo we need to move it in a compatible way using terminal commands
    * Citation: open_spiel/python/examples/alpha_zero_jax.py; open_spiel/python/algorithms/alpha_zero/alpha_zero.py

[DONE] 26. **Refactor JAX AlphaZero for Separate Example Entrypoint.**
    *   Separate the main executable logic (ABSL flags, `main` function) from `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py` into a new example script, e.g., `open_spiel/python/examples/alpha_zero_jax.py`.
    *   The `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py` file should then primarily contain the core algorithm functions (`alpha_zero_jax`, `learner`, `actor`, `evaluator`, helper classes/functions like `ConfigJAX`, `Trajectory`, etc.), making it a library module.
    *   The new example script will import and use the functions from the algorithm module, similar to how `open_spiel/python/examples/alpha_zero.py` uses `open_spiel/python/algorithms/alpha_zero/alpha_zero.py`.
    *   Assessment: Recommended for structural consistency with the TensorFlow AlphaZero implementation and to promote better separation of concerns (library vs. example).
    *   Citation: Current structure of `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py` vs. `open_spiel/python/algorithms/alpha_zero/alpha_zero.py` and `open_spiel/python/examples/alpha_zero.py`

[DONE] 28. **Ensure JAX ConvNets Reshape Flat Inference Input (`model_jax.py`)**
    *   **Why**: JAX convolutional models (`Conv2D_JAX`, `ResNet_JAX`) are initialized using a dummy input with the correct 3D spatial shape (e.g., `(Batch, H, W, C)`). However, during inference (e.g., in `_play_game`), the observation tensor provided by `state.observation_tensor()` is often flat (e.g., `(Batch, Features)`). Passing this flat tensor directly to the convolutional layers (which expect 3D spatial input) causes a `flax.errors.ScopeParamShapeError` because the existing initialized parameters are for a 3D input, but Flax re-evaluates based on the flat input, leading to a shape mismatch.
    *   **What**: The `__call__` methods of `Conv2D_JAX` and `ResNet_JAX` must check if their input tensor `x` is flat (e.g., `x.ndim == 2`). If so, `x` must be reshaped to its expected 3D spatial format (e.g., `(Batch, H, W, C)`) before being processed by any convolutional layers.
    *   **Where**:
        *   The `Conv2D_JAX` and `ResNet_JAX` classes in `open_spiel/python/algorithms/alpha_zero_jax/model_jax.py`.
        *   The `init_flax_model_and_variables` function in the same file needs to pass the `processed_observation_shape` to the model constructors.
    *   **How**:
        1.  Modify `Conv2D_JAX` and `ResNet_JAX` constructors to accept an `expected_input_shape` parameter (e.g., `(H, W, C)` for the game's observations) and store it as an instance attribute (e.g., `self.expected_input_shape`).
        2.  In `init_flax_model_and_variables`, when instantiating these models, pass the `observation_shape` (which is the processed HxWxC shape, e.g., `(8,8,5)` for checkers) as this `expected_input_shape` argument.
        3.  At the beginning of the `__call__` method in `Conv2D_JAX` and `ResNet_JAX`:
            *   Check `if x.ndim == 2:`.
            *   If true, reshape `x`: `x = x.reshape((x.shape[0],) + self.expected_input_shape)`.
            *   This reshaped `x` is then used by the subsequent convolutional layers.
    *   **Citation**: Debug logs from checkers run showing `model.init()` with `(1,8,8,5)` and `model.apply()` (via `_play_game`) receiving `(1,320)` leading to `ScopeParamShapeError`. Specifically, `[DEBUG Conv2D_JAX __call__] Initial x.shape: (1, 8, 8, 5)` during init vs. `[DEBUG Conv2D_JAX __call__] Initial x.shape: (1, 320)` during inference.

[DONE] 29. **Investigate and Fix `NaN` Losses in Learner (`alpha_zero_jax.py`)**
    *   **`NaN` Losses**:
        *   **Symptom**: `Total Loss` and `Policy Loss` are reported as `NaN` during training, while `Value Loss` might be a number.
        *   **Potential Causes**:
            *   Logits becoming `inf` or `-inf` passed to `optax.softmax_cross_entropy`.
            *   Invalid policy targets (e.g., not summing to 1, containing `NaN`s).
            *   All actions for a state being illegal, leading to all logits being `-inf`, potentially causing `NaN` in softmax cross-entropy.
            *   Division by zero if `batch_legals_masks.any(axis=1)` is `False` and the corresponding unmasked policy loss was `inf` or `NaN` (leading to `0 * inf` or `0 * nan`).
        *   **Debugging Steps**:
            *   Verify `batch_policy_targets` are valid probability distributions.
            *   Log statistics (min, max, mean, presence of `NaN`/`inf`) of `policy_logits` and `batch_policy_targets` just before `optax.softmax_cross_entropy`.
            *   Check if `batch_legals_masks.any(axis=1)` is `False` for any samples and how it interacts with the calculated policy loss for those samples.
            *   Inspect the model's internal masking of illegal actions (setting logits to `-jnp.inf`).
    *   **Citation**: Learner logs showing `NaN` for policy/total loss.

[DONE] 30. **Investigate and Fix Checkpointing Errors in Learner (`alpha_zero_jax.py`)**
    *   **Checkpointing Errors**:
        *   **Symptom**: Errors like "No such file or directory" (e.g., `b'opt_state.0.count/'`) or "add() argument after ** must be a mapping, not NoneType" during `checkpoints.save_checkpoint`.
        *   **Potential Causes**:
            *   Race conditions or issues with atomic saving/renaming of the "latest" checkpoint, especially with Orbax backend.
            *   `NaN` values in model parameters or optimizer state, causing serialization failures.
            *   Incorrect state structure being passed to `checkpoints.save_checkpoint` (e.g., `None` where a dictionary is expected).
        *   **Debugging Steps**:
            *   Examine the structure and content of `variables` and `opt_state` just before saving.
            *   Simplify the checkpoint saving logic temporarily (e.g., only save step-numbered checkpoints, disable "latest" to isolate).
            *   Review Orbax documentation for best practices in atomic saving.
    *   **Citation**: Learner logs showing errors during checkpoint saving.

[DONE] 31. **Define and Clean Up Logging Levels in `alpha_zero_jax.py`)**
    *   Remove all pre-existing debug print statements introduced for shape/type inspection (e.g., "Learner Python Loop: Types...", "Stacked batch shapes", "Shapes before train_step_fn call").
    *   Categorize log output by `log_level`:
        - **log_level=3**: Per-step training summaries (games/s, states/s, buffer size, losses).
        - **log_level=2**: Warnings and errors (checkpoint failures, NaNs), and per-episode or checkpoint summaries.
        - **log_level=1**: Final experiment summary only (aggregate losses, final checkpoint path).
    *   Suppress redundant library and Orbax INFO logs at all levels by configuring absl and the logger appropriately.
    *   Ensure `quiet=True` fully suppresses per-move actor prints.

32. **Implement named log-level constants and suppress third-party verbose logs**  
    *   Add in-code suppression of Orbax/Abseil INFO logs via `os.environ['GLOG_minloglevel']` and `absl.logging.set_verbosity`.  
    *   Define `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE` constants at the top of `alpha_zero_jax.py`.  
    *   Change example script's `--log_level` flag from `DEFINE_integer` to `DEFINE_enum("log_level", "INFO", ["ERROR","WARN","INFO","DEBUG","TRACE"], "Logging verbosity level")`.  
    *   Replace all numeric `config.log_level >= N` checks with the new named constants.  
    *   Citation: Chat discussion on 2025-05-16 about logging suppression and log-level constants.

## 9. Remote Inference Service Tasks

**High-Level Overview**

We centralize all model initialization, inference, and TPU interactions in the main process, ensuring exactly one TPU context and avoiding cross-process conflicts. Actor and evaluator processes remain pure-Python and CPU-bound, running MCTS self-play and generating inference requests. These requests are batched by the main process to maximize TPU throughput, with results returned to the origin processes. This architecture provides:

- Full CPU parallelism for game simulation and MCTS actors.
- Efficient, batched TPU inference in a single process to avoid repeated initialization.
- Clear separation: actors/evaluators handle data generation; the learner/inference service handles model computation and training.
- Robustness in environments (e.g., Colab) where TPU resources cannot be shared across processes.

**Why Refactor?**

- Multi-process JAX/TPU initialization leads to resource conflicts and `XlaRuntimeError` (e.g., "TPU is already in use") on platforms like Colab or single-TPU servers.
- In the current design, each actor/evaluator subprocess attempts to import JAX and initialize the TPU, causing repeated context acquisitions and failures.
- Centralizing all TPU-based inference and model initialization in one main process eliminates contention and runtime errors.
- Batching inference requests maximizes TPU utilization, reducing per-request overhead and improving overall throughput.

- [DONE] [VERIFIED] Define message protocol:
  - Request: ("inference_req", request_id, actor_id, observation, legals_mask)
  - Response: ("inference_resp", request_id, value, policy_probs)

- [DONE] [VERIFIED] Create a shared inference request queue in the main process.

- [DONE] [VERIFIED] Implement `RemoteEvaluator` stub in actor and evaluator processes that:
  1. Generates a unique `request_id` (e.g., via `uuid4().hex`).
  2. Sends `("inference_req", request_id, actor_id, observation, legals_mask)` on the shared request queue.
  3. Blocks (with optional timeout) on `queue.get()` for the matching `("inference_resp", request_id, value, policy)` message.
  4. Returns `(value, policy)` to the MCTS logic.

- [DONE] [VERIFIED] In the main script, initialize the JAX/Flax model and its variables exactly once on the TPU before spawning any processes.

- [DONE] [VERIFIED] Build an `InferenceServicer` in the main process comprising:
  - **BatchAssemblyThread**:
    - Polls all actor & evaluator queues for "inference_req" messages.
    - Accumulates requests into batches (up to batch size N or after timeout T).
    - Enqueues prepared batches on a `ReadyForInferenceBatchQueue`.
  - **InferenceExecutionThread**:
    - Dequeues a batch from `ReadyForInferenceBatchQueue`.
    - Calls `model.apply(variables, batched_obs, training=False, legals_mask=batched_legals)` on the TPU.
    - Sends back `("inference_resp", request_id, value, policy)` to each origin queue.

- [DONE] [VERIFIED] Pass the shared request queue and per-process queues into spawn.Process kwargs for actor and evaluator.

- [DONE] [VERIFIED] Remove all JAX/Flax imports and `init_flax_model_and_variables` calls from actor and evaluator subprocesses.

- [DONE] [VERIFIED] Ensure actor and evaluator processes run only pure-Python MCTS logic and use `RemoteEvaluator` for inference.

- [DONE] [VERIFIED] Update the learner function to reuse the single TPU-initialized model and variables for both training and inference servicing.

- [DONE] [VERIFIED] Implement graceful shutdown: 
  - Main process signals `InferenceServicer` to stop.
  - `InferenceServicer` threads use `SHUTDOWN_SENTINEL` to stop and signal downstream.
  - `RemoteEvaluator` raises `ShutdownException` on receiving sentinel or queue errors.
  - Actor and Evaluator processes catch `ShutdownException` to break their main loops and exit gracefully.
  - Main process joins all spawned processes.

- [DONE] [VERIFIED] Refactor `alpha_zero_jax.py` to prevent actor/evaluator subprocesses from importing JAX/Flax directly (e.g., move to separate pure-Python file or use import guards).
- [DONE] [VERIFIED] Modify `InferenceServicer` in `alpha_zero_jax.py` to use `InferenceRequest.from_tuple()` in `BatchAssemblyThread` and `InferenceResponse(...).to_tuple()` in `InferenceExecutionThread`.
- [DONE] [VERIFIED] Refactor `InferenceExecutionThread` in `alpha_zero_jax.py` to use a JIT-compiled `batched_inference_fn` that includes `jax.nn.softmax` internally.
- [DONE] [VERIFIED] Update `BatchAssemblyThread.run()` in `alpha_zero_jax.py` to catch both `mp.queues.Empty` and `std_queue.Empty` for robust timeout handling.
- [DONE] [VERIFIED] Initialize `processes = [], actor_process_queues = [], and evaluator_process_queues = []` lists in the `alpha_zero_jax` function within `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py` before these lists are appended to.
- [DONE] [VERIFIED] Instantiate `inference_request_queue = mp.Queue()` in the `alpha_zero_jax` function within `open_spiel/python/algorithms/alpha_zero_jax/alpha_zero_jax.py` before it is passed as an argument to actors, evaluators, or the learner.
- [DONE] **Decouple `RemoteEvaluator` Logging for Dedicated Log Files**
    -   **Why**: To provide a separate, dedicated log file for each `RemoteEvaluator` instance (e.g., `log-remote_evaluator_actor_{actor_id}.txt`), making it easier to debug remote inference communication without interference from general actor logs. This also simplifies `RemoteEvaluator`'s interface and standardizes its internal logging.
    -   **What & Where**:
        1.  **Modify `RemoteEvaluator` in `open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py`**:
            *   Remove `logger`, `log_level`, and `log_file` parameters from `RemoteEvaluator.__init__` (around lines 71-86).
            *   In `__init__`, unconditionally create a standard Python `logging.Logger` (e.g., `logging.getLogger(f"RemoteEvaluator_Actor_{actor_id}")`) and configure it with a `FileHandler` to write to `log-remote_evaluator_actor_{actor_id}.txt`. Set a default logging level (e.g., `logging.DEBUG`).
            *   Replace all conditional `if self.logger: if self.log_level >= _XXX_LEVEL: self.logger.print(...)` calls throughout `RemoteEvaluator` (e.g., lines 132-134, 140-142, 159-161, 170-172, 175-177, 183-185, 189-192, 196-198, 205-207, 213-216, 222-224) with standard `self.logger.debug()`, `self.logger.info()`, `self.logger.warning()`, or `self.logger.error()` calls, as appropriate for the message severity.
            *   Remove the `_ERROR_LEVEL`, `_WARN_LEVEL`, `_INFO_LEVEL`, `_DEBUG_LEVEL`, `_TRACE_LEVEL` constants (around lines 17-22).
        2.  **Update `RemoteEvaluator` Instantiation in `open_spiel/python/algorithms/alpha_zero_jax/actor_evaluator_logic.py`**:
            *   In the `actor` function, when `RemoteEvaluator` is created (around line 100, but this line number will change as the dummy actor is reverted), remove the `logger=logger` and `log_level=config.actor_verbosity` arguments from the constructor call.
    -   **Citation**: Current `RemoteEvaluator` implementation in [remote_inference.py](mdc:open_spiel/python/algorithms/alpha_zero_jax/remote_inference.py) and its usage in `actor_evaluator_logic.py`. This task addresses the user's request for separate log files for remote inference components.
- [DONE] Conduct end-to-end testing on TPU to verify:
  - Only one TPU initialization occurs (no XlaRuntimeError).
  - Batched inference requests are processed correctly and efficiently.
  - Actor processes generate trajectories in parallel with expected throughput.
  - Learner training and checkpointing operate as intended.

## Phase 10: MCTS Enhancements for Chance Nodes and Async Search

[DONE] [VERIFIED] 1. Fix chance-node evaluation crash:
    * Set `dont_return_chance_node=True` in `AlphaZeroBot` (in actor_evaluator_logic.py) and `MCTSBot` constructors.
    * Add a unit test to verify that no evaluator is ever called on a chance node.

[DONE] [VERIFIED] 2. Integrate asynchronous MCTS:
    * Import `