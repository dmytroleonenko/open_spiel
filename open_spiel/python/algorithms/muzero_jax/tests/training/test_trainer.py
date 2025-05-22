import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import os
import shutil
from unittest.mock import patch
import dataclasses
import flax.nnx.graph as nnx_graph
import orbax.checkpoint as ocp
from jax.tree_util import tree_structure

from open_spiel.python.algorithms.muzero_jax.training.trainer import Learner, MuZeroConfig, Batch
from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork

# Constants
OBS_SHAPE_FLAT = (10,)
OBS_SHAPE_IMAGE = (3, 32, 32)
NUM_ACTIONS = 5
BATCH_SIZE = 2
NUM_UNROLL_STEPS = 3
VALUE_SUPPORT_SCALAR = 0
REWARD_SUPPORT_SCALAR = 0
VALUE_SUPPORT_CATEGORICAL = 11
REWARD_SUPPORT_CATEGORICAL = 21

# Mock network components
class MockRep(nnx.Module):
    def __init__(self, obs_shape, hidden, *, rngs):
        self.dense = nnx.Linear(jnp.prod(jnp.array(obs_shape)), hidden, rngs=rngs)
    def __call__(self, x, training):
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))
        return self.dense(x)

class MockDyn(nnx.Module):
    def __init__(self, hidden, nact, *, rngs):
        self.embed = nnx.Embed(nact, hidden // 2, rngs=rngs)
        self.fc = nnx.Linear(hidden + hidden//2, hidden, rngs=rngs)
    def __call__(self, h, a, training):
        e = self.embed(a)
        if e.ndim == 1:
            e = jnp.broadcast_to(e, (h.shape[0], e.shape[-1]))
        return nnx.relu(self.fc(jnp.concatenate([h, e], -1)))

class MockPred(nnx.Module):
    def __init__(self, hidden, nact, vsup, *, rngs):
        self.ph = nnx.Linear(hidden, nact, rngs=rngs)
        self.vh = nnx.Linear(hidden, vsup if vsup>0 else 1, rngs=rngs)
    def __call__(self, h, training):
        return self.ph(h), self.vh(h)

class MockRew(nnx.Module):
    def __init__(self, hidden, rsup, *, rngs):
        self.rh = nnx.Linear(hidden, rsup if rsup>0 else 1, rngs=rngs)
    def __call__(self, h, training):
        return self.rh(h)

class MockProj(nnx.Module):
    def __init__(self, hidden, psize, *, rngs):
        self.ph = nnx.Linear(hidden, psize, rngs=rngs)
    def __call__(self, h, training):
        return self.ph(h)

@dataclasses.dataclass(frozen=True)
class MockNetCfg:
    observation_shape: tuple = OBS_SHAPE_FLAT
    num_actions: int = NUM_ACTIONS
    hidden_size: int = 16
    value_support_size: int = VALUE_SUPPORT_SCALAR
    reward_support_size: int = REWARD_SUPPORT_SCALAR
    projection_output_size: int = 8
    use_projection: bool = False
    batch_size: int = BATCH_SIZE

# Fixtures
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

@pytest.fixture
def cfg_flat():
    return MockNetCfg()

@pytest.fixture
def cfg_img():
    return dataclasses.replace(MockNetCfg(observation_shape=OBS_SHAPE_IMAGE))

# Helpers
def make_model(key, cfg):
    rep = lambda model_config, *, rngs: MockRep(model_config.observation_shape, model_config.hidden_size, rngs=rngs)
    dyn = lambda model_config, *, rngs: MockDyn(model_config.hidden_size, model_config.num_actions, rngs=rngs)
    pred = lambda model_config, *, rngs: MockPred(model_config.hidden_size, model_config.num_actions, model_config.value_support_size, rngs=rngs)
    rew = lambda model_config, *, rngs: MockRew(model_config.hidden_size, model_config.reward_support_size, rngs=rngs)
    proj_def_lambda = (lambda model_config, *, rngs_lambda: MockProj(model_config.hidden_size, model_config.projection_output_size, rngs=rngs_lambda)) if cfg.use_projection else None
    return MuZeroNetwork(rep, dyn, pred, rew, proj_def_lambda, cfg, rngs=nnx.Rngs(params=key))

def make_cfg(vsup, rsup, steps, proj, suffix, use_ema=False, ssl_weight=0.0):
    base = f"/tmp/mz_test_{suffix}" # Added _test_ to avoid potential conflict with main example
    if os.path.exists(base):
        shutil.rmtree(base)
    return MuZeroConfig(
        value_support_size=vsup,
        reward_support_size=rsup,
        discount_factor=0.99,
        num_unroll_steps=steps,
        td_steps=steps+1,
        value_loss_weight=0.25,
        reward_loss_weight=1.0,
        policy_loss_weight=1.0,
        l2_weight=1e-4,
        use_projection=proj,
        ssl_consistency_loss_weight=ssl_weight,
        learning_rate=1e-3,
        adam_b1=0.9,
        adam_b2=0.999,
        clip_grad_norm=5.0,
        batch_size=BATCH_SIZE,
        use_target_network_ema=use_ema,
        ema_decay=0.99,
        checkpoint_dir=base,
        checkpoint_frequency=5,
        max_checkpoints_to_keep=1,
        resume_from_checkpoint=False
    )

def make_batch(key, bs, obs_shape, nact, steps, vsup, rsup, proj_dim=None, use_proj=False):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    obs = jax.random.uniform(k1, (bs, steps+1, *obs_shape))
    acts = jax.random.randint(k2, (bs, steps), 0, nact)
    val = jax.random.normal(k3, (bs, steps+1)) if vsup==0 else jax.random.uniform(k3, (bs, steps+1, vsup))
    rew = jax.random.normal(k4, (bs, steps+1)) if rsup==0 else jax.random.uniform(k4, (bs, steps+1, rsup))
    pol = jax.random.uniform(k5, (bs, steps+1, nact))
    pol = pol / jnp.sum(pol, axis=-1, keepdims=True)
    mask = jnp.ones((bs, steps+1))
    batch_data = {
        'observation': obs, 'action': acts, 
        'target_reward': rew, 'target_value': val,
        'target_policy': pol, 'game_history_mask': mask
    }
    # No need to add projected_hidden_state to batch, it's a model internal
    return batch_data

# Tests
def test_init(key, cfg_flat):
    mk = jax.random.fold_in(key, 1)
    model = make_model(mk, cfg_flat)
    cfg = make_cfg(cfg_flat.value_support_size, cfg_flat.reward_support_size, NUM_UNROLL_STEPS, False, 'init')
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, mk)
    assert learner.num_training_steps == 0
    assert learner.opt_state is not None

@pytest.mark.parametrize("img,val_cat,proj,use_ema", [
    (False, False, False, False), 
    (True, True, True, True),
    (False, False, True, False), # Test projection without EMA
    (False, False, False, True)  # Test EMA without projection
])
def test_loss_static(key, img, val_cat, proj, use_ema, cfg_flat, cfg_img):
    bk, mk, lk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_img if img else cfg_flat)
    cfgn = dataclasses.replace(cfgn, 
                               use_projection=proj,
                               value_support_size=VALUE_SUPPORT_CATEGORICAL if val_cat else VALUE_SUPPORT_SCALAR,
                               reward_support_size=REWARD_SUPPORT_CATEGORICAL if val_cat else REWARD_SUPPORT_SCALAR
                              )
    model = make_model(mk, cfgn)
    cfg = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, NUM_UNROLL_STEPS, proj, 'loss_static_test', use_ema=use_ema, ssl_weight=0.1 if proj else 0.0)
    batch = make_batch(bk, cfg.batch_size, cfgn.observation_shape, cfgn.num_actions, cfg.num_unroll_steps, cfgn.value_support_size, cfgn.reward_support_size, cfgn.projection_output_size, proj)
    loss, met = Learner._compute_total_loss_static(model, cfg, batch, lk, training=True)
    assert isinstance(loss, jax.Array) and loss.shape == ()
    for m in ['total_loss', 'policy_loss', 'value_loss', 'reward_loss', 'l2_loss']:
        assert m in met
    if proj and cfg.ssl_consistency_loss_weight > 0:
        assert 'ssl_loss' in met
        assert met['ssl_loss'] != 0 # Should contribute if weight > 0

@pytest.mark.parametrize("img,val_cat,proj,use_ema", [
    (False, False, False, False), 
    (True, True, True, True),
    (False, False, True, False), # Test projection without EMA
    (False, False, False, True)  # Test EMA without projection
])
def test_step(key, img, val_cat, proj, use_ema, cfg_flat, cfg_img):
    bk, mk, lk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_img if img else cfg_flat)
    cfgn = dataclasses.replace(cfgn, 
                               use_projection=proj,
                               value_support_size=VALUE_SUPPORT_CATEGORICAL if val_cat else VALUE_SUPPORT_SCALAR,
                               reward_support_size=REWARD_SUPPORT_CATEGORICAL if val_cat else REWARD_SUPPORT_SCALAR
                              )
    model = make_model(mk, cfgn)
    cfg = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, NUM_UNROLL_STEPS, proj, f'step_{img}_{val_cat}_{proj}_{use_ema}', use_ema=use_ema, ssl_weight=0.1 if proj else 0.0)
    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)
    batch = make_batch(bk, cfg.batch_size, cfgn.observation_shape, cfgn.num_actions, cfg.num_unroll_steps, cfgn.value_support_size, cfgn.reward_support_size, cfgn.projection_output_size, proj)
    
    # Get params before step
    _, initial_online_params_state, _, _, _, _ = nnx.split(learner.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)

    updated_model, _, metrics = learner.train_step(batch)
    
    # Get params after step
    _, online_params_state_after, _, _, _, _ = nnx.split(updated_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)

    before_leaves = jax.tree_util.tree_leaves(initial_online_params_state) 
    after_leaves = jax.tree_util.tree_leaves(online_params_state_after) 
    assert any(not jnp.allclose(a, b) for a, b in zip(before_leaves, after_leaves))
    if proj and cfg.ssl_consistency_loss_weight > 0:
        assert 'ssl_loss' in metrics
    if use_ema:
        assert learner.target_model is not None
        assert learner.ema_params_state is not None
        # Check target model params are different from online IF ema_decay < 1
        _, target_params_state, _, _, _, _ = nnx.split(learner.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
        
        if cfg.ema_decay < 1.0:
            # After one step, EMA parameters should differ from the *initial* online parameters.
            # And also from the *current* online parameters if updates happened.
            initial_online_leaves = jax.tree_util.tree_leaves(initial_online_params_state)
            target_leaves = jax.tree_util.tree_leaves(target_params_state)
            current_online_leaves = jax.tree_util.tree_leaves(online_params_state_after) # online_params_state_after was defined above

            assert any(not jnp.allclose(init_online, target) for init_online, target in zip(initial_online_leaves, target_leaves)), \
                "Target params should not be identical to initial online params after EMA update with decay < 1"
            assert any(not jnp.allclose(curr_online, target) for curr_online, target in zip(current_online_leaves, target_leaves)), \
                "Target params should not be identical to current online params after EMA update with decay < 1"

@pytest.mark.parametrize("use_ema, resume", [(False, False), (True, False), (True, True)])
def test_train_loop_and_ckpt(key, cfg_flat, use_ema, resume):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_flat, use_projection=use_ema) # Enable projection if EMA is used for more coverage
    model = make_model(mk, cfgn)
    cfg_suffix = f'loop_ema_{use_ema}_resume_{resume}'
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, 
                     proj=cfgn.use_projection, 
                     suffix=cfg_suffix, 
                     use_ema=use_ema, 
                     ssl_weight=0.1 if cfgn.use_projection else 0.0)
    cfg = dataclasses.replace(cfg, resume_from_checkpoint=False) # Start fresh for first run

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)

    num_total_steps = 4 
    batches = [make_batch(jax.random.fold_in(bk, i),
                          cfg.batch_size,
                          cfgn.observation_shape,
                          cfgn.num_actions,
                          cfg.num_unroll_steps,
                          cfgn.value_support_size,
                          cfgn.reward_support_size,
                          cfgn.projection_output_size, cfgn.use_projection)
               for i in range(num_total_steps)]

    def get_batch_generator_fn():
        # This function now returns a new generator each time it's called
        def gen():
            for item in batches:
                yield item
        return gen()

    # with patch.object(learner, 'save_checkpoint') as mock_save: # Removed patch to test actual saving
    learner.train(get_batch_generator_fn, num_epochs=1, steps_per_epoch=num_total_steps)
    
    assert learner.num_training_steps == num_total_steps
    # Check that a checkpoint was actually saved (latest_step should not be None)
    if cfg.checkpoint_dir and learner.checkpoint_manager:
        latest_saved_step = learner.checkpoint_manager.latest_step()
        assert latest_saved_step == num_total_steps, f"Expected checkpoint at step {num_total_steps}, found {latest_saved_step}"

    if resume:
        # Create new model and learner to simulate restart for loading
        mk_resume, lk_resume = jax.random.split(jax.random.fold_in(key, 100), 2)
        model_resume = make_model(mk_resume, cfgn)
        cfg_resume = dataclasses.replace(cfg, resume_from_checkpoint=True)
        opt_resume = optax.adam(cfg_resume.learning_rate)
        learner_resume = Learner(model_resume, opt_resume, cfg_resume, lk_resume)
        assert learner_resume.num_training_steps == num_total_steps
        if use_ema:
            assert learner_resume.target_model is not None
            assert learner_resume.ema_params_state is not None

def test_train_loop_exhausted_buffer(key, cfg_flat):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = cfg_flat
    model = make_model(mk, cfgn)
    cfg_suffix = 'exhausted_buffer'
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, 
                     proj=False, 
                     suffix=cfg_suffix, 
                     use_ema=False)
    cfg = dataclasses.replace(cfg, checkpoint_frequency=100) # Avoid ckpt logic

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)

    # Empty batch list
    batches = []

    def get_empty_batch_generator_fn():
        def gen():
            for item in batches: # Will not yield anything
                yield item
            # Simulate exhaustion even after re-init by not yielding again
            # Or, more realistically, ensure it stops after one attempt to re-init
        return gen()
    
    # To test the re-initialization and immediate exhaustion, 
    # we can make the generator yield once, then be empty upon re-initialization.
    # However, the current test structure is simpler by just providing an always-empty generator.
    # The code under test will try to re-init, then StopIteration again.

    with patch('builtins.print') as mock_print:
        learner.train(get_empty_batch_generator_fn, num_epochs=1, steps_per_epoch=1)
    
    assert learner.num_training_steps == 0
    
    # Check if the specific print messages were called
    assert any("Replay buffer iterator exhausted." in call_args[0][0] for call_args in mock_print.call_args_list)
    assert any("Replay buffer truly exhausted. Stopping training." in call_args[0][0] for call_args in mock_print.call_args_list)

def test_checkpointing_no_manager(key, cfg_flat):
    mk, lk = jax.random.split(key, 2)
    model = make_model(mk, cfg_flat)
    # Create a config with checkpoint_dir=None
    cfg_no_ckpt = MuZeroConfig(
        value_support_size=cfg_flat.value_support_size,
        reward_support_size=cfg_flat.reward_support_size,
        checkpoint_dir=None # Explicitly None
    )
    opt = optax.adam(cfg_no_ckpt.learning_rate)
    learner = Learner(model, opt, cfg_no_ckpt, lk)

    assert learner.checkpoint_manager is None

    # Test save_checkpoint
    with patch('builtins.print') as mock_print_save:
        learner.save_checkpoint()
    mock_print_save.assert_any_call("Checkpoint manager not configured. Skipping save.")

    # Test load_checkpoint
    with patch('builtins.print') as mock_print_load:
        loaded = learner.load_checkpoint()
    assert not loaded
    mock_print_load.assert_any_call("Checkpoint manager not configured. Skipping load.")

def test_load_checkpoint_no_checkpoint_exists(key, cfg_flat):
    mk, lk = jax.random.split(key, 2)
    model = make_model(mk, cfg_flat)
    cfg_suffix = "no_ckpt_exists"
    # Ensure a unique directory that will be empty
    cfg_ckpt = make_cfg(cfg_flat.value_support_size, 
                        cfg_flat.reward_support_size, 
                        1, 
                        proj=False, 
                        suffix=cfg_suffix)
    # Clean up if it somehow exists from a previous failed run
    if os.path.exists(cfg_ckpt.checkpoint_dir):
        shutil.rmtree(cfg_ckpt.checkpoint_dir)
    os.makedirs(cfg_ckpt.checkpoint_dir) # Create empty dir

    opt = optax.adam(cfg_ckpt.learning_rate)
    learner = Learner(model, opt, cfg_ckpt, lk)

    assert learner.checkpoint_manager is not None

    with patch('builtins.print') as mock_print:
        loaded = learner.load_checkpoint()
    assert not loaded
    mock_print.assert_any_call("No checkpoint found to resume from.")
    # Clean up the created directory
    if os.path.exists(cfg_ckpt.checkpoint_dir):
        shutil.rmtree(cfg_ckpt.checkpoint_dir)

def test_save_checkpoint_conditions(key, cfg_flat):
    mk, lk, bk = jax.random.split(key, 3)
    cfgn = cfg_flat
    model = make_model(mk, cfgn)
    cfg_suffix = "save_conditions"
    cfg = make_cfg(cfgn.value_support_size, 
                     cfgn.reward_support_size, 
                     1, 
                     proj=False, 
                     suffix=cfg_suffix)
    # Set checkpoint frequency high to test skipping, then force save
    cfg = dataclasses.replace(cfg, checkpoint_frequency=100, max_checkpoints_to_keep=1)

    opt = optax.adam(cfg.learning_rate)
    learner = Learner(model, opt, cfg, lk)
    assert learner.checkpoint_manager is not None

    # --- Test skipping save due to frequency --- 
    learner.num_training_steps = 50 # Less than checkpoint_frequency
    with patch.object(learner.checkpoint_manager, 'save') as mock_manager_save, \
         patch('logging.info') as mock_logging_info_skip:
        learner.save_checkpoint(force_save=False)
    mock_manager_save.assert_not_called()
    # Check for the specific log message indicating skip
    assert any(
        f"SAVE_CHECKPOINT: Condition NOT met. force_save=False, num_training_steps={learner.num_training_steps}, freq={cfg.checkpoint_frequency}" in call_args[0][0]
        for call_args in mock_logging_info_skip.call_args_list
    ), "Log message for skipping save due to frequency not found."


    # --- Test force_save=True at end of hypothetical training (within train loop logic) ---
    # This part simulates the condition within the train() method
    num_epochs = 1
    steps_per_epoch = 3
    learner.num_training_steps = 0 # Reset
    cfg_train_end = dataclasses.replace(cfg, checkpoint_frequency=2) # Save every 2 steps
    learner.config = cfg_train_end # Update learner's config
    
    batches = [make_batch(jax.random.fold_in(bk, i),
                          cfg_train_end.batch_size,
                          cfgn.observation_shape,
                          cfgn.num_actions,
                          cfg_train_end.num_unroll_steps,
                          cfgn.value_support_size,
                          cfgn.reward_support_size)
               for i in range(steps_per_epoch)]

    def get_batch_gen_fn():
        def gen(): yield from batches
        return gen()

    with patch.object(learner.checkpoint_manager, 'save') as mock_manager_save_train, \
         patch('logging.info') as mock_logging_info_train:
        learner.train(get_batch_gen_fn, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch)
    
    # Expected saves: 
    # Step 2 (regular)
    # Step 3 (end of training, force_save=True via internal logic of train() calling save_checkpoint(force_save=True))
    assert mock_manager_save_train.call_count == 2
    # Check the call for step 2 (regular)
    args_step2, _ = mock_manager_save_train.call_args_list[0]
    assert args_step2[0] == 2 # step number
    # Check the call for step 3 (end of training)
    args_step3, _ = mock_manager_save_train.call_args_list[1]
    assert args_step3[0] == 3 # step number
    
    # Verify the logging for force_save=True for the last step
    # The save_checkpoint method is called with force_save=True by the train method internally.
    # We need to check the logging call that reflects this forced save.
    found_force_save_log = False
    for call_args in mock_logging_info_train.call_args_list:
        log_message = call_args[0][0]
        if f"End of training checkpoint: step {steps_per_epoch}" in log_message:
            # This is logged just before calling save_checkpoint(force_save=True)
            # Now find the corresponding SAVE_CHECKPOINT log for this step
            for subsequent_call_args in mock_logging_info_train.call_args_list:
                if f"SAVE_CHECKPOINT: Condition met. force_save=True, num_training_steps={steps_per_epoch}, freq={cfg_train_end.checkpoint_frequency}" in subsequent_call_args[0][0]:
                    found_force_save_log = True
                    break
            if found_force_save_log: break
    assert found_force_save_log, "Log message for force_save=True at end of training not found."

    # --- Test regular save due to frequency ---
    # learner.config.checkpoint_frequency is 2 at this point from the previous section of the test.
    learner.num_training_steps = learner.config.checkpoint_frequency # This will be 2
    with patch.object(learner.checkpoint_manager, 'save') as mock_manager_save_freq, \
         patch('logging.info') as mock_logging_info_freq:
        learner.save_checkpoint(force_save=False)
    mock_manager_save_freq.assert_called_once()
    # Check for the specific log message indicating save due to frequency
    assert any(
        f"SAVE_CHECKPOINT: Condition met. force_save=False, num_training_steps={learner.num_training_steps}, freq={learner.config.checkpoint_frequency}" in call_args[0][0]
        for call_args in mock_logging_info_freq.call_args_list
    ), f"Log message for saving due to frequency not found. Log calls: {mock_logging_info_freq.call_args_list}"

    # Clean up
    if os.path.exists(cfg.checkpoint_dir):
        shutil.rmtree(cfg.checkpoint_dir)

def test_load_checkpoint_missing_ema_details(key, cfg_flat):
    mk, lk_save, lk_load, bk = jax.random.split(key, 4)
    cfgn = cfg_flat # Using flat config for simplicity
    model_save = make_model(mk, cfgn)
    cfg_suffix = "missing_ema"

    # 1. Configure and run a learner WITHOUT EMA to save a checkpoint
    cfg_save = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, 1, False, 
                        suffix=f"{cfg_suffix}_save", use_ema=False)
    opt_save = optax.adam(cfg_save.learning_rate)
    learner_save = Learner(model_save, opt_save, cfg_save, lk_save)
    learner_save.num_training_steps = 1 # Make it save something
    learner_save.save_checkpoint(force_save=True) # target_model and ema_params_state will be None
    if learner_save.checkpoint_manager: # Ensure manager exists before closing
        learner_save.checkpoint_manager.close()

    # 2. Configure a new learner WITH EMA and try to load the checkpoint
    #    This learner will expect target_model_params etc. in the checkpoint.
    model_load = make_model(jax.random.fold_in(mk,1), cfgn) # Fresh model instance
    cfg_load = dataclasses.replace(cfg_save, use_target_network_ema=True, resume_from_checkpoint=True)
    # Ensure the checkpoint_dir is the same
    opt_load = optax.adam(cfg_load.learning_rate)
    
    with patch('builtins.print') as mock_print, \
         patch.object(ocp.CheckpointManager, 'latest_step', return_value=1) as mock_latest_step, \
         patch.object(ocp.CheckpointManager, 'restore') as mock_restore:

        # Simulate that restore will return a dictionary as if loaded from checkpoint,
        # but without the target_model components and ema_params_state initially.
        # The actual checkpoint was saved by learner_save, which had these as None.
        # learner_save.opt_state and learner_save._rng_key are optax/jax types.
        # model_params_save would be an nnx.State object.
        _, model_params_save, model_bs_save, model_rngs_save, model_static_save, model_ellipsis_save = nnx.split(
            learner_save.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)

        restored_dict_simulation = {
            'model_graphdef': nnx.graphdef(learner_save.model),
            'model_params': model_params_save,
            'model_batch_stats': model_bs_save,
            'model_rngs': model_rngs_save,
            'model_static': model_static_save,
            'model_ellipsis': model_ellipsis_save,
            'opt_state': learner_save.opt_state,
            'num_training_steps': learner_save.num_training_steps, # Should be 1
            'rng_key': learner_save._rng_key,
            # Simulate these being absent from the loaded checkpoint dict initially
            'target_model_graphdef': None,
            'target_model_params': None,
            'target_model_batch_stats': None,
            'target_model_rngs': None,
            'target_model_static': None,
            'target_model_ellipsis': None,
            'ema_params_state': None 
        }
        mock_restore.return_value = restored_dict_simulation

        learner_load = Learner(model_load, opt_load, cfg_load, lk_load)
        # Learner init calls load_checkpoint if resume_from_checkpoint is True

    mock_latest_step.assert_called()

    # To get the correct abstract items for the assertion:
    # Create a temporary learner with the same setup as learner_load but *without* resume,
    # to capture its initial state for opt_state and ema_params_state, which are used
    # by the actual load_checkpoint method to build its abstract_items_to_restore.
    cfg_temp_for_abstract = dataclasses.replace(cfg_load, resume_from_checkpoint=False)
    # Use a different key for this temp learner if its _rng_key is part of abstract items and needs to match lk_load exactly.
    # However, lk_load is what learner_load receives, so its _rng_key will be lk_load.
    temp_learner_for_abstract = Learner(model_load, opt_load, cfg_temp_for_abstract, lk_load) 

    model_graphdef_abs, model_params_abs, model_bs_abs, model_rngs_abs, model_static_abs, model_ellipsis_abs = nnx.split(
        temp_learner_for_abstract.model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...
    )
    # The RngKey used for abstract item should be the one that the learner_load instance has when it calls load_checkpoint.
    # This is lk_load, which is temp_learner_for_abstract._rng_key if lk_load was passed to it.

    expected_abstract_items = {
        'model_graphdef': model_graphdef_abs,
        'model_params': model_params_abs,
        'model_batch_stats': model_bs_abs,
        'model_rngs': model_rngs_abs,
        'model_static': model_static_abs,
        'model_ellipsis': model_ellipsis_abs,
        'opt_state': temp_learner_for_abstract.opt_state,
        'num_training_steps': 0, # Abstract, value doesn't strictly matter here, type does
        'rng_key': temp_learner_for_abstract._rng_key, # This is lk_load
        'ema_params_state': temp_learner_for_abstract.ema_params_state
    }

    if temp_learner_for_abstract.target_model is not None:
        target_gdef_abs, target_params_abs, target_bs_abs, target_rngs_abs, target_static_abs, target_ellipsis_abs = nnx.split(
            temp_learner_for_abstract.target_model, nnx.Param, nnx.BatchStat, nnx.Rngs, nnx_graph.Static, ...)
        expected_abstract_items['target_model_graphdef'] = target_gdef_abs
        expected_abstract_items['target_model_params'] = target_params_abs
        expected_abstract_items['target_model_batch_stats'] = target_bs_abs
        expected_abstract_items['target_model_rngs'] = target_rngs_abs
        expected_abstract_items['target_model_static'] = target_static_abs
        expected_abstract_items['target_model_ellipsis'] = target_ellipsis_abs
    else:
        expected_abstract_items['target_model_graphdef'] = None
        expected_abstract_items['target_model_params'] = None
        expected_abstract_items['target_model_batch_stats'] = None
        expected_abstract_items['target_model_rngs'] = None
        expected_abstract_items['target_model_static'] = None
        expected_abstract_items['target_model_ellipsis'] = None
    
    # mock_restore.assert_called_once_with(1, args=ocp.args.StandardRestore(expected_abstract_items))
    # Instead, verify the call and compare the abstract_tree directly for clarity
    mock_restore.assert_called_once() # Check it was called
    
    call_args_tuple = mock_restore.call_args
    assert call_args_tuple is not None, "mock_restore was not called despite assert_called_once passing?"

    # call_args can be a tuple (args, kwargs) or just args if no kwargs were used.
    # For CheckpointManager.restore(step, items, bundle, args), args is passed as a kwarg.
    if isinstance(call_args_tuple, tuple):
        called_pos_args, called_kwargs = call_args_tuple
    else: # Should not happen with how restore is called with kwargs
        # For safety, handle if it's just positional (though unlikely for .restore with args=)
        called_pos_args = call_args_tuple
        called_kwargs = {}

    assert called_pos_args[0] == 1 # Step number
    assert 'args' in called_kwargs
    standard_restore_arg_received = called_kwargs['args']
    assert isinstance(standard_restore_arg_received, ocp.args.StandardRestore)
    
    # Compare the PyTrees. Pytest should give a good diff if they are not equal.
    assert tree_structure(standard_restore_arg_received.item) == tree_structure(expected_abstract_items)

    assert learner_load.num_training_steps == 1 # Should load steps
    assert learner_load.target_model is not None # Should be re-initialized
    assert learner_load.ema_params_state is not None # Should be re-initialized
    
    # Check for the specific warning prints
    found_warn_target_missing = any(
        "Warning: EMA enabled, target model components not fully in ckpt. Re-syncing with online model." in call_args[0][0]
        for call_args in mock_print.call_args_list
    )
    found_warn_ema_state_missing = any(
        "Warning: EMA enabled, ema_params_state not in ckpt. Reinitializing EMA state from online model." in call_args[0][0]
        for call_args in mock_print.call_args_list
    )
    assert found_warn_target_missing or found_warn_ema_state_missing, \
        f"Expected EMA recovery warnings not found. Logs: {mock_print.call_args_list}"

    # Clean up
    if os.path.exists(cfg_save.checkpoint_dir):
        shutil.rmtree(cfg_save.checkpoint_dir)

def test_loss_static_missing_projection_in_model_output(key, cfg_flat):
    """Test _compute_total_loss_static when use_projection=True but model.initial_inference is misbehaving."""
    bk, mk, lk = jax.random.split(key, 3)
    cfgn = dataclasses.replace(cfg_flat, use_projection=True) # Enable projection in model config
    
    # Scenario 1: initial_inference returns too few elements
    class MockModelShortInitial(MuZeroNetwork):
        def initial_inference(self, x, training):
            # Returns hidden_state, reward, value, policy_logits (4 elements)
            # Actual mock model parts need to be set up if they are accessed by base class
            hidden_state = jnp.zeros((x.shape[0], self.config.hidden_size))
            reward = jnp.zeros((x.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((x.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((x.shape[0], self.config.num_actions))
            return hidden_state, reward, value, policy_logits 
        # recurrent_inference also needs to be properly mocked if reached
        def recurrent_inference(self, h, a, training):
            reward = jnp.zeros((h.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((h.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((h.shape[0], self.config.num_actions))
            # No projection returned here either for simplicity, though not directly testing this part for coverage here
            return h, reward, value, policy_logits

    model_short = MockModelShortInitial(
        representation_network_def=lambda cfg, *, rngs: MockRep(cfg.observation_shape, cfg.hidden_size, rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: MockDyn(cfg.hidden_size, cfg.num_actions, rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: MockPred(cfg.hidden_size, cfg.num_actions, cfg.value_support_size, rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: MockRew(cfg.hidden_size, cfg.reward_support_size, rngs=rngs),
        projection_network_def=None, # This won't be called due to override
        config=cfgn, 
        rngs=nnx.Rngs(params=mk)
    )

    # Learner config with projection enabled and SSL loss active
    cfg_learner_proj = make_cfg(cfgn.value_support_size, cfgn.reward_support_size, NUM_UNROLL_STEPS, proj=True, 
                                suffix='loss_missing_proj1', ssl_weight=0.1)
    batch = make_batch(bk, cfg_learner_proj.batch_size, cfgn.observation_shape, cfgn.num_actions, 
                       cfg_learner_proj.num_unroll_steps, cfgn.value_support_size, cfgn.reward_support_size,
                       use_proj=True) # Batch is made as if proj is expected

    # Test with model_short
    loss1, met1 = Learner._compute_total_loss_static(model_short, cfg_learner_proj, batch, lk, training=True)
    assert 'ssl_loss' in met1 # SSL loss should still be in metrics (even if 0)
    assert met1['ssl_loss'] == 0.0 # SSL loss should be zero as no projections were processed

    # Scenario 2: initial_inference returns projection as None
    class MockModelNoneInitialProjection(MuZeroNetwork):
        def initial_inference(self, x, training):
            hidden_state = jnp.zeros((x.shape[0], self.config.hidden_size))
            reward = jnp.zeros((x.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((x.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((x.shape[0], self.config.num_actions))
            return hidden_state, reward, value, policy_logits, None # 5th element is None
        def recurrent_inference(self, h, a, training):
            reward = jnp.zeros((h.shape[0], 1 if self.config.reward_support_size == 0 else self.config.reward_support_size))
            value = jnp.zeros((h.shape[0], 1 if self.config.value_support_size == 0 else self.config.value_support_size))
            policy_logits = jnp.zeros((h.shape[0], self.config.num_actions))
            return h, reward, value, policy_logits, None # Return None projection here too

    model_none_proj = MockModelNoneInitialProjection(
        representation_network_def=lambda cfg, *, rngs: MockRep(cfg.observation_shape, cfg.hidden_size, rngs=rngs),
        dynamics_network_def=lambda cfg, *, rngs: MockDyn(cfg.hidden_size, cfg.num_actions, rngs=rngs),
        prediction_network_def=lambda cfg, *, rngs: MockPred(cfg.hidden_size, cfg.num_actions, cfg.value_support_size, rngs=rngs),
        reward_network_def=lambda cfg, *, rngs: MockRew(cfg.hidden_size, cfg.reward_support_size, rngs=rngs),
        projection_network_def=None, 
        config=cfgn, 
        rngs=nnx.Rngs(params=jax.random.fold_in(mk,1))
    )
    # Test with model_none_proj
    loss2, met2 = Learner._compute_total_loss_static(model_none_proj, cfg_learner_proj, batch, jax.random.fold_in(lk,1), training=True)
    assert 'ssl_loss' in met2
    assert met2['ssl_loss'] == 0.0

def test_load_checkpoint_load_exception(key, cfg_flat):
    mk, lk = jax.random.split(key, 2)
    model = make_model(mk, cfg_flat)
    cfg_suffix = "load_exception"
    cfg = make_cfg(cfg_flat.value_support_size, cfg_flat.reward_support_size, 1, False, cfg_suffix)
    cfg = dataclasses.replace(cfg, resume_from_checkpoint=True)

    opt = optax.adam(cfg.learning_rate)
    # Create a dummy checkpoint manager that will raise an exception on restore
    class FailingCheckpointManager:
        def __init__(self, *args, **kwargs):
            self.latest_step_val = 1
        def latest_step(self):
            return self.latest_step_val # Pretend a checkpoint exists
        def restore(self, step, args=None):
            raise ValueError("Simulated restore error")
        def wait_until_finished(self):
            pass
        def save(self, step, args=None): # Add save to allow Learner init to proceed far enough
            pass 

    with patch('orbax.checkpoint.CheckpointManager', FailingCheckpointManager), \
         patch('builtins.print') as mock_print:
        # Learner init calls load_checkpoint
        learner = Learner(model, opt, cfg, lk) 
    
    assert learner.num_training_steps == 0 # Should not have loaded steps
    found_error_log = any(
        "Error loading checkpoint: Simulated restore error" in call_args[0][0]
        for call_args in mock_print.call_args_list
    )
    assert found_error_log, f"Expected error log not found. Logs: {mock_print.call_args_list}"
    # No directory to clean up as we mocked CheckpointManager heavily

def teardown_module(module):
    tmp_dir = "/tmp"
    for item in os.listdir(tmp_dir):
        if item.startswith("mz_test_"):
            path = os.path.join(tmp_dir, item)
            if os.path.isdir(path): shutil.rmtree(path) # pragma: no cover

# More tests to come for train, save/load checkpoint 