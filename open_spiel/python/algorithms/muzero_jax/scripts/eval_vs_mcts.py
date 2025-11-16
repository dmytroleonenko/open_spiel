#!/usr/bin/env python3
"""
Evaluate a trained MuZero JAX network against a pure MCTS baseline.

Usage example (after running training into /tmp/muzero_ttt_big64):

    python -m open_spiel.python.algorithms.muzero_jax.scripts.eval_vs_mcts \
        --save-path /tmp/muzero_ttt_big64 \
        --games 100 \
        --mcts-simulations 400 \
        --override hydra.job_logging.root.level=WARNING \
        --override +hydra.job_logging.handlers.console.level=WARNING

The script:
  * Loads the MuZero configuration via Hydra (with optional overrides)
  * Restores the latest checkpoint from --save-path (or the step provided)
  * Runs alternating games between MuZero (with the trained network) and a
    baseline pyspiel.MCTSBot configured with --mcts-simulations
  * Prints aggregate win/draw stats
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pyspiel
from hydra import initialize_config_dir, compose

from open_spiel.python.algorithms.muzero_jax.envs.game_wrapper import GameWrapper
from open_spiel.python.algorithms.muzero_jax.run_muzero_jax import MuZeroOrchestrator
from open_spiel.python.algorithms.muzero_jax.self_play.actor import Actor


@dataclass
class EvalResult:
    games: int
    muzero_wins: int
    baseline_wins: int
    draws: int


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config_dir = (
        Path(__file__).resolve().parents[1] / "configs"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Directory containing Hydra configs (default: %(default)s)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name to load (default: %(default)s)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra-style override (repeatable, e.g. --override training.batch_size=64)",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Directory containing checkpoints from training (output.save_path)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific checkpoint step to restore (default: latest available)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of evaluation games to play (default: %(default)s)",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=200,
        help="Number of simulations for the pure MCTS baseline (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for RNGs controlling MuZero tie-breakers and baseline randomness",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="MuZero action temperature during evaluation (default: deterministic argmax)",
    )
    parser.add_argument(
        "--muzero-player",
        type=str,
        choices=("auto", "0", "1"),
        default="auto",
        help="Which player MuZero controls: alternate automatically or fix to player 0/1",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-game outcomes",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def evaluate_against_mcts(argv: Optional[Iterable[str]] = None) -> None:
    """Convenience wrapper for tests; mirrors main without sys.exit."""
    args = parse_args(argv)
    evaluate(
        cfg_dir=args.config_dir,
        cfg_name=args.config_name,
        overrides=args.override,
        save_path=args.save_path,
        checkpoint_step=args.checkpoint_step,
        games=args.games,
        baseline_simulations=args.mcts_simulations,
        seed=args.seed,
        mu_player_mode=args.muzero_player,
        verbose=args.verbose,
    )


def _select_muzero_action(
    actor: Actor,
    rng_key: jax.Array,
    temperature_step: int,
) -> Tuple[int, jax.Array]:
    """Mirror Actor.play_episode decision logic for evaluation."""
    current_obs = actor.game_wrapper.current_observation()
    if not current_obs:  # pragma: no cover - defensive
        raise RuntimeError("MuZero observation unavailable at player node.")

    obs_array = jnp.array([current_obs])
    hidden_state, reward, value, policy_logits, _, reward_hidden = actor.network.initial_inference(
        obs_array, training=False
    )

    from mctx._src.base import RootFnOutput

    root = RootFnOutput(
        prior_logits=policy_logits,
        value=value,
        embedding=hidden_state,
    )
    legal_actions = actor.game_wrapper.legal_actions()
    num_actions = actor.config.num_actions
    invalid_actions = jnp.ones((1, num_actions), dtype=bool)
    invalid_actions = invalid_actions.at[0, jnp.array(legal_actions)].set(False)

    rng_key, subkey = jax.random.split(rng_key)
    if actor._is_stochastic_mcts:  # pragma: no cover - deterministic path used in tests
        policy_output = actor.mcts.run_stochastic(
            rng_key=subkey,
            root=root,
            network=actor.network,
            invalid_actions=invalid_actions,
        )
    else:
        recurrent_fn = actor._create_recurrent_fn()
        policy_output = actor.mcts.run(
            params=None,
            rng_key=subkey,
            root=root,
            recurrent_fn=recurrent_fn,
            invalid_actions=invalid_actions,
        )

    # Force deterministic behavior by overriding temperature schedule when needed.
    actor.temperature = 0.0  # force greedy evaluation
    action, _ = actor._select_action(policy_output, temperature_step)
    return int(action), rng_key


def run_game(
    actor: Actor,
    baseline_bot: pyspiel.MCTSBot,
    mu_player: int,
    rng_key: jax.Array,
    rng_np: np.random.Generator,
) -> Tuple[int, jax.Array]:
    """Play a single game, returning MuZero payoff (+1 win, -1 loss, 0 draw)."""
    actor.game_wrapper.reset()
    state = actor.game_wrapper._state

    step_idx = 0
    while not state.is_terminal():
        if state.is_chance_node():  # pragma: no cover - chance games not in fast tests
            outcomes = state.chance_outcomes()
            if not outcomes:
                break
            vals, probs = zip(*outcomes)
            rng_key, subkey = jax.random.split(rng_key)
            choice = int(rng_np.choice(vals, p=np.array(probs, dtype=np.float32)))
            actor.game_wrapper.step(choice)
            state = actor.game_wrapper._state
            continue

        current_player = state.current_player()
        if current_player == mu_player:
            action, rng_key = _select_muzero_action(actor, rng_key, step_idx)
            step_idx += 1
        else:
            action = baseline_bot.step(state)

        actor.game_wrapper.step(action)
        state = actor.game_wrapper._state

    returns = state.returns()
    payoff = returns[mu_player]
    return payoff, rng_key


def evaluate(
    cfg_dir: Path,
    cfg_name: str,
    overrides: List[str],
    save_path: Path,
    checkpoint_step: Optional[int],
    games: int,
    baseline_simulations: int,
    seed: int,
    mu_player_mode: str,
    verbose: bool,
) -> EvalResult:
    overrides = list(overrides)

    def _has_override(key: str) -> bool:
        for item in overrides:
            stripped = item.lstrip("+")
            if stripped.split("=")[0] == key:
                return True  # pragma: no cover - trivial lookup
        return False
    overrides.append(f"output.save_path={save_path}")
    overrides.extend(
        [
            "resource_management.concurrent=false",
            "resource_management.sequential_training=true",
            "actors.num_actors=1",
            "evaluation.enabled=false",
        ]
    )
    if not _has_override("training.start_transitions"):
        overrides.append("+training.start_transitions=1")

    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        cfg = compose(config_name=cfg_name, overrides=overrides)

    orchestrator = MuZeroOrchestrator(cfg)
    checkpoint_path = None
    if checkpoint_step is not None:
        checkpoint_path = str(save_path / "checkpoints" / str(checkpoint_step))
    restore_ok = orchestrator.learner.load_checkpoint(checkpoint_path)
    if not restore_ok:
        # Fallback: proceed with in-memory params (helpful for tiny smoke tests)
        print("No checkpoint found; evaluating current parameters.")

    # Configure evaluation actor
    eval_actor = Actor(
        network=orchestrator.network,
        game_wrapper=GameWrapper(cfg.game.name),
        replay_buffer=None,
        config=orchestrator.muzero_config,
        num_simulations=orchestrator.muzero_config.num_simulations,
        max_num_considered_actions=orchestrator.game_wrapper.num_distinct_actions(),
        gumbel_scale=1.0,
        n_step_return=orchestrator.muzero_config.td_steps,
        discount_factor=orchestrator.muzero_config.discount_factor,
        temperature=0.0,
        temperature_threshold=0,
    )
    eval_actor.maybe_load_latest_parameters(str(orchestrator.checkpoint_dir))

    rng_key = jax.random.PRNGKey(seed)
    rng_np = np.random.default_rng(seed)

    muzero_wins = baseline_wins = draws = 0
    for game_idx in range(games):
        mu_player = (
            game_idx % 2 if mu_player_mode == "auto" else int(mu_player_mode)
        )
        baseline_bot = pyspiel.MCTSBot(
            game=eval_actor.game_wrapper._game,
            evaluator=pyspiel.RandomRolloutEvaluator(1, seed + game_idx),
            uct_c=1.25,
            max_simulations=baseline_simulations,
            max_memory_mb=1024,
            solve=False,
            seed=seed + game_idx,
            verbose=False,
        )

        payoff, rng_key = run_game(
            actor=eval_actor,
            baseline_bot=baseline_bot,
            mu_player=mu_player,
            rng_key=rng_key,
            rng_np=rng_np,
        )
        if payoff > 0:
            muzero_wins += 1
            outcome = "MuZero win"
        elif payoff < 0:
            baseline_wins += 1
            outcome = "MCTS win"
        else:
            draws += 1
            outcome = "Draw"
        if verbose:  # pragma: no cover - printing only
            print(
                f"Game {game_idx+1:03d}: {outcome} "
                f"(MuZero as player {mu_player})"
            )

    return EvalResult(
        games=games,
        muzero_wins=muzero_wins,
        baseline_wins=baseline_wins,
        draws=draws,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover - CLI entry
    args = parse_args(argv)
    result = evaluate(
        cfg_dir=args.config_dir,
        cfg_name=args.config_name,
        overrides=args.override,
        save_path=args.save_path,
        checkpoint_step=args.checkpoint_step,
        games=args.games,
        baseline_simulations=args.mcts_simulations,
        seed=args.seed,
        mu_player_mode=args.muzero_player,
        verbose=args.verbose,
    )
    print(
        f"MuZero vs pure MCTS ({args.mcts_simulations} sims) over {result.games} games:\n"
        f"  MuZero wins : {result.muzero_wins}\n"
        f"  MCTS wins   : {result.baseline_wins}\n"
        f"  Draws       : {result.draws}"
    )  # pragma: no cover - CLI output


if __name__ == "__main__":  # pragma: no cover - manual utility
    main(sys.argv[1:])
