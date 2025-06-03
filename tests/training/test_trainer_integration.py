from open_spiel.python.algorithms.muzero_jax.models.network import MuZeroNetwork
from trainer_test_utils import MockRep, MockDyn, MockPred, MockRew, MockProj


def test_wandb_logging_disabled(common_key, common_cfg_flat):
    """Test training with wandb logging disabled to cover missing lines."""
    import wandb

    # Ensure wandb is not initialized
    if wandb.run is not None:
        wandb.finish()

    cfg = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_no_wandb", use_ema=False
    )
    model = make_model(common_key, cfg)
    optimizer_def = optax.adam(learning_rate=1e-4)
    learner = Learner(model, optimizer_def, cfg, common_key)

    # Create a simple batch generator
    def batch_generator():
        while True:
            batch = make_batch(
                common_key,
                cfg.batch_size,
                common_cfg_flat.observation_shape,
                common_cfg_flat.num_actions,
                cfg.num_unroll_steps,
                vsup=0,
                rsup=0,
                use_proj=False,
            )
            yield batch

    # Train for 1 epoch, 2 steps (should not crash)
    try:
        learner.train(batch_generator, num_epochs=1, steps_per_epoch=2)
        print("✅ Training without wandb logging successful!")
    except Exception as e:
        pytest.fail(f"Training failed: {e}")


def test_remaining_squeeze_operations_comprehensive(common_key, common_cfg_flat):
    """Test the remaining squeeze operations for KL loss and other edge cases."""

    # Test KL loss with squeeze operations (lines 487, 498)
    class KLSqueezeTestRew(nnx.Module):
        def __init__(self, *, rngs):
            pass

        def __call__(self, h, training):
            return jnp.ones((h.shape[0], 1))  # B, 1 - triggers squeeze on line 487

    cfg_kl = make_cfg(
        vsup=0, rsup=601, steps=1, proj=False, suffix="_kl_squeeze", use_ema=False
    )
    cfg_kl = dataclasses.replace(cfg_kl, reward_loss_type="kl")

    rep = lambda model_config, *, rngs: MockRep(
        common_cfg_flat.observation_shape, common_cfg_flat.hidden_size, rngs=rngs
    )
    dyn = lambda model_config, *, rngs: MockDyn(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, rngs=rngs
    )
    pred = lambda model_config, *, rngs: MockPred(
        common_cfg_flat.hidden_size, common_cfg_flat.num_actions, 0, rngs=rngs
    )
    rew_kl = lambda model_config, *, rngs: KLSqueezeTestRew(rngs=rngs)

    model_kl = MuZeroNetwork(
        rep, dyn, pred, rew_kl, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    # Create batch with scalar targets that will need conversion to distributions
    batch_kl = make_batch(
        common_key,
        cfg_kl.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg_kl.num_unroll_steps,
        vsup=0,
        rsup=0,
        use_proj=False,
    )

    # Add targets with shape (B, K+1, 1) to trigger squeeze on line 498
    target_rewards_reshaped = batch_kl["target_reward"][..., None]  # Add dimension
    batch_kl_modified = {**batch_kl, "target_reward": target_rewards_reshaped}

    loss_kl, metrics_kl = Learner._compute_total_loss_static(
        model_kl, cfg_kl, batch_kl_modified, common_key, training=True
    )
    assert loss_kl.shape == ()
    assert "reward_loss" in metrics_kl

    # Test MSE reward loss with distribution predictions (lines 514, 525)
    class MSERewardDistRew(nnx.Module):
        def __init__(self, *, rngs):
            pass

        def __call__(self, h, training):
            # Return distribution to trigger support_to_scalar on line 514
            return jnp.ones((h.shape[0], 601))  # B, 601

    cfg_mse_rew = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_mse_rew_dist", use_ema=False
    )
    cfg_mse_rew = dataclasses.replace(cfg_mse_rew, reward_loss_type="mse")

    rew_mse_dist = lambda model_config, *, rngs: MSERewardDistRew(rngs=rngs)
    model_mse_rew = MuZeroNetwork(
        rep, dyn, pred, rew_mse_dist, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    # Create batch with distribution targets to trigger squeeze on line 525
    batch_mse_rew = make_batch(
        common_key,
        cfg_mse_rew.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg_mse_rew.num_unroll_steps,
        vsup=0,
        rsup=601,
        use_proj=False,
    )  # rsup=601 creates distributions

    loss_mse_rew, metrics_mse_rew = Learner._compute_total_loss_static(
        model_mse_rew, cfg_mse_rew, batch_mse_rew, common_key, training=True
    )
    assert loss_mse_rew.shape == ()
    assert "reward_loss" in metrics_mse_rew

    # Test symlog reward loss with squeeze (lines 539, 550, 557)
    class SymlogRewardSqueezeRew(nnx.Module):
        def __init__(self, *, rngs):
            pass

        def __call__(self, h, training):
            return jnp.ones((h.shape[0], 1))  # B, 1 - triggers squeeze on line 539

    cfg_symlog_rew = make_cfg(
        vsup=0, rsup=0, steps=1, proj=False, suffix="_symlog_rew", use_ema=False
    )
    cfg_symlog_rew = dataclasses.replace(cfg_symlog_rew, reward_loss_type="symlog")

    rew_symlog = lambda model_config, *, rngs: SymlogRewardSqueezeRew(rngs=rngs)
    model_symlog_rew = MuZeroNetwork(
        rep, dyn, pred, rew_symlog, None, common_cfg_flat, rngs=nnx.Rngs(params=common_key)
    )

    # Create batch with distribution targets that will trigger squeeze on line 550
    batch_symlog_rew = make_batch(
        common_key,
        cfg_symlog_rew.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg_symlog_rew.num_unroll_steps,
        vsup=0,
        rsup=601,
        use_proj=False,
    )  # Distribution targets

    loss_symlog_rew, metrics_symlog_rew = Learner._compute_total_loss_static(
        model_symlog_rew, cfg_symlog_rew, batch_symlog_rew, common_key, training=True
    )
    assert loss_symlog_rew.shape == ()
    assert "reward_loss" in metrics_symlog_rew

    # Also test with scalar targets having shape (B, K+1, 1) to hit line 557
    # Create a batch with scalar targets first, then reshape
    batch_scalar_rew = make_batch(
        common_key,
        cfg_symlog_rew.batch_size,
        common_cfg_flat.observation_shape,
        common_cfg_flat.num_actions,
        cfg_symlog_rew.num_unroll_steps,
        vsup=0,
        rsup=0,
        use_proj=False,
    )  # Scalar targets
    target_rewards_1d = batch_scalar_rew["target_reward"][
        ..., None
    ]  # Add dimension: (B, K+1, 1)
    batch_symlog_rew_1d = {**batch_scalar_rew, "target_reward": target_rewards_1d}

    loss_symlog_rew_1d, metrics_symlog_rew_1d = Learner._compute_total_loss_static(
        model_symlog_rew, cfg_symlog_rew, batch_symlog_rew_1d, common_key, training=True
    )
    assert loss_symlog_rew_1d.shape == ()
    assert "reward_loss" in metrics_symlog_rew_1d 