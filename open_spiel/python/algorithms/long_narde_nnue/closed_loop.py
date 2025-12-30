"""Closed-loop self-play training for Long Narde NNUE."""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from open_spiel.python.algorithms.long_narde_nnue.lnue_reader import (
    LnueShardReader,
)
from open_spiel.python.algorithms.long_narde_nnue.train import (
    NnueNet,
    _build_base_cmd,
    _bootstrap_targets,
    _iter_batches,
    _resolve_device,
    _save_nnue,
    _train_batch,
    add_training_args,
    run_selfplay,
)


class ProgressBar:
    """Simple ASCII progress bar with rate and ETA."""
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        total: int,
        prefix: str,
        width: int = 30,
        unit: str = "items",
        min_interval: float = 0.5,
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        self.total = max(total, 1)
        self.prefix = prefix
        self.width = width
        self.unit = unit
        self.min_interval = min_interval
        self.last_percent = -1
        self.start_time = time.perf_counter()
        self.last_time = self.start_time

    def update(self, current: int) -> None:
        """Updates the progress bar with the latest count."""
        now = time.perf_counter()
        percent = int(100 * current / self.total)
        percent = min(100, max(0, percent))
        if (
            percent == self.last_percent
            and (now - self.last_time) < self.min_interval
        ):
            return
        self.last_percent = percent
        self.last_time = now
        filled = int(self.width * percent / 100)
        bar_str = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start_time
        rate = current / elapsed if elapsed > 0.0 else 0.0
        remaining = max(self.total - current, 0)
        eta = remaining / rate if rate > 0.0 else 0.0
        print(
            f"\r{self.prefix} [{bar_str}] {percent}% ({current}/{self.total}) "
            f"{rate:,.2f} {self.unit}/s ETA {eta:,.0f}s",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        """Finishes the progress bar and moves to a new line."""
        self.update(self.total)
        print()


def _repo_root() -> Path:
    """Returns the repository root path."""
    return Path(__file__).resolve().parents[4]


def _resolve_bin(explicit: str, repo_root: Path, name: str) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        repo_root / "build-release" / "games" / name,
        repo_root / "build-relwithdebinfo" / "games" / name,
        repo_root / "build" / "games" / name,
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[-1]
    return max(existing, key=lambda path: path.stat().st_mtime)


def _count_samples(paths: Iterable[Path]) -> int:
    """Counts samples across LNUE shards."""
    total = 0
    for path in paths:
        reader = LnueShardReader(str(path))
        for chunk in reader.iter_chunks():
            total += int(chunk.outcome.shape[0])
    return total


def _parse_iter_index(path: Path) -> int | None:
    name = path.name
    if not name.startswith("iter_"):
        return None
    suffix = name.split("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _collect_replay_shards(
    output_dir: Path,
    current_iter: int,
    replay_iters: int,
    replay_max_shards: int,
) -> List[Path]:
    if replay_iters <= 0:
        return []
    candidates = []
    for path in output_dir.glob("iter_*"):
        idx = _parse_iter_index(path)
        if idx is None or idx >= current_iter:
            continue
        candidates.append((idx, path))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    candidates = candidates[:replay_iters]
    shards: List[Path] = []
    for _, iter_dir in candidates:
        data_dir = iter_dir / "data"
        shards.extend(sorted(data_dir.glob("*.lnue")))
        if 0 < replay_max_shards <= len(shards):
            return shards[:replay_max_shards]
    return shards


def _collect_nats_shards(
    stream_dir: Path,
    iter_data_dir: Path,
    target_shards: int,
    poll_seconds: float,
    progress: bool,
) -> List[Path]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-locals
    iter_data_dir.mkdir(parents=True, exist_ok=True)
    stream_dir.mkdir(parents=True, exist_ok=True)
    collected = len(list(iter_data_dir.glob("*.lnue")))
    if target_shards <= 0:
        raise ValueError("target_shards must be positive.")
    if progress:
        progress_bar = ProgressBar(target_shards, "nats collect", unit="shards")
        progress_bar.update(collected)
    last_report = -1
    while collected < target_shards:
        candidates = sorted(stream_dir.glob("*.lnue"))
        moved_any = False
        for path in candidates:
            dest = iter_data_dir / path.name
            if dest.exists():
                dest = iter_data_dir / f"{path.stem}_dup{collected:04d}.lnue"
            shutil.move(str(path), str(dest))
            collected += 1
            moved_any = True
            if progress:
                progress_bar.update(collected)
            if collected >= target_shards:
                break
        if collected >= target_shards:
            break
        if not moved_any:
            if not progress and collected != last_report:
                print(
                    f"waiting for shards: {collected}/{target_shards}",
                    flush=True,
                )
                last_report = collected
            time.sleep(max(poll_seconds, 0.1))
    if progress:
        progress_bar.finish()
    return sorted(iter_data_dir.glob("*.lnue"))


def _train_epoch(
    shard_paths: List[Path],
    model: NnueNet,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    ev_weight: float,
    quant_loss: bool,
    quant_loss_weight: float,
    bootstrap_steps: int,
    bootstrap_alpha: float,
    seed: int,
    epoch_idx: int,
) -> None:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    np_rng = np.random.default_rng(seed + epoch_idx)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    total_samples = _count_samples(shard_paths)
    progress = ProgressBar(
        total_samples,
        f"train epoch {epoch_idx + 1}",
        unit="samples",
    )
    processed = 0

    shard_list = list(shard_paths)
    np_rng.shuffle(shard_list)
    for shard_path in shard_list:
        reader = LnueShardReader(str(shard_path))
        for chunk in reader.iter_chunks():
            target_override = None
            if bootstrap_steps > 0:
                target_override = _bootstrap_targets(
                    chunk, bootstrap_steps, bootstrap_alpha
                )
            for indices, offsets, outcome, target in _iter_batches(
                chunk, batch_size, np_rng, target_override
            ):
                _train_batch(
                    model,
                    optimizer,
                    criterion,
                    indices,
                    offsets,
                    outcome,
                    target,
                    device,
                    ev_weight,
                    quant_loss,
                    quant_loss_weight,
                )

                processed += int(outcome.shape[0])
                progress.update(processed)

    progress.finish()


def _run_eval(  # pylint: disable=too-many-locals
    bin_path: Path,
    nnue_a: Path | None,
    nnue_b: Path | None,
    games: int,
    depth: int,
    seed: int,
    eval_workers: int,
    progress: bool,
    report_every: int,
    root_full_depth_top_k: int,
    root_reduced_depth: int,
    chance_samples: int,
    chance_sample_depth: int,
    chance_seed: int,
) -> str:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cmd = _build_base_cmd(bin_path, games, depth, seed)
    cmd.extend(
        [
            "--progress",
            "1" if progress else "0",
            "--report_every",
            str(report_every),
            "--workers",
            str(eval_workers),
        ]
    )
    if root_full_depth_top_k > 0:
        cmd.extend(["--root_full_depth_top_k", str(root_full_depth_top_k)])
    if root_reduced_depth >= 0:
        cmd.extend(["--root_reduced_depth", str(root_reduced_depth)])
    if chance_samples > 0:
        cmd.extend(["--chance_samples", str(chance_samples)])
        cmd.extend(["--chance_sample_depth", str(chance_sample_depth)])
        cmd.extend(["--chance_seed", str(chance_seed)])
    if nnue_a is not None:
        cmd.extend(["--nnue_a", str(nnue_a)])
    if nnue_b is not None:
        cmd.extend(["--nnue_b", str(nnue_b)])
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Eval produced no output.")
    return lines[-1].strip()


def _publish_nnue(
    publish_bin: Path,
    nats_url: str,
    run_id: str,
    nnue_path: Path,
    version: int,
) -> None:
    cmd = [
        str(publish_bin),
        "--nats",
        nats_url,
        "--run_id",
        run_id,
        "--version",
        str(version),
        "--nnue",
        str(nnue_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Runs closed-loop self-play training and evaluation."""
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="results/nnue_closed_loop")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games_per_iter", type=int, default=10000)
    parser.add_argument("--games_per_shard", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature_end", type=float, default=-1.0)
    parser.add_argument("--temperature_decay_plies", type=int, default=0)
    parser.add_argument("--root_full_depth_top_k", type=int, default=0)
    parser.add_argument("--root_reduced_depth", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--selfplay_progress", type=int, default=1)
    parser.add_argument("--selfplay_report_every", type=int, default=100)
    parser.add_argument("--selfplay_stream", type=int, default=1)
    parser.add_argument("--selfplay_resume", type=int, default=1)
    parser.add_argument("--selfplay_flush_every_games", type=int, default=1)
    parser.add_argument("--selfplay_lock", default="")
    parser.add_argument("--selfplay_force_lock", type=int, default=0)
    parser.add_argument("--skip_selfplay", action="store_true")
    parser.add_argument("--selfplay_dir", default="")
    parser.add_argument("--eval_games", type=int, default=1000)
    parser.add_argument("--eval_depth", type=int, default=-1)
    parser.add_argument("--eval_workers", type=int, default=0)
    parser.add_argument("--eval_progress", type=int, default=1)
    parser.add_argument("--eval_report_every", type=int, default=100)
    parser.add_argument("--replay_iters", type=int, default=0)
    parser.add_argument("--replay_frac", type=float, default=0.0)
    parser.add_argument("--replay_max_shards", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--chance_samples", type=int, default=0)
    parser.add_argument("--chance_sample_depth", type=int, default=1)
    parser.add_argument("--chance_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)
    add_training_args(parser)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init_nnue", default="")
    parser.add_argument("--selfplay_bin", default="")
    parser.add_argument("--eval_bin", default="")
    parser.add_argument("--nats", default="")
    parser.add_argument("--nats_run_id", default="")
    parser.add_argument("--nats_shards_per_iter", type=int, default=0)
    parser.add_argument("--nats_poll_secs", type=float, default=2.0)
    parser.add_argument("--nats_publish", type=int, default=1)
    parser.add_argument("--nats_publish_bin", default="")
    parser.add_argument("--nats_version_start", type=int, default=0)
    args = parser.parse_args()
    device = _resolve_device(args.device)
    bootstrap_alpha = args.bootstrap_alpha
    if bootstrap_alpha < 0.0:
        bootstrap_alpha = args.alpha
    temperature_end = args.temperature_end
    if temperature_end < 0.0:
        temperature_end = None
    temperature_decay_plies = args.temperature_decay_plies
    if temperature_decay_plies <= 0:
        temperature_decay_plies = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"

    repo_root = _repo_root()
    eval_bin = _resolve_bin(args.eval_bin, repo_root, "long_narde_eval")
    publish_bin = None
    nats_url = args.nats.strip()
    nats_run_id = args.nats_run_id
    selfplay_bin = None

    if not eval_bin.exists():
        raise FileNotFoundError(f"eval binary not found: {eval_bin}")
    print(f"using eval_bin: {eval_bin}", flush=True)
    if not nats_url and not args.skip_selfplay:
        selfplay_bin = _resolve_bin(
            args.selfplay_bin, repo_root, "long_narde_selfplay"
        )
        if not selfplay_bin.exists():
            raise FileNotFoundError(
                f"selfplay binary not found: {selfplay_bin}"
            )
        print(f"using selfplay_bin: {selfplay_bin}", flush=True)
    if nats_url:
        publish_bin = _resolve_bin(
            args.nats_publish_bin, repo_root, "long_narde_nats_publish"
        )
        if not publish_bin.exists():
            raise FileNotFoundError(
                f"nats publish binary not found: {publish_bin}"
            )
        if not nats_run_id:
            if args.selfplay_dir:
                nats_run_id = Path(args.selfplay_dir).name
            else:
                nats_run_id = "default"
        print(f"using nats_publish_bin: {publish_bin}", flush=True)
        print(f"using nats_run_id: {nats_run_id}", flush=True)

    prev_nnue = Path(args.init_nnue) if args.init_nnue else None

    for iteration in range(args.iterations):
        iter_dir = output_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        data_dir = iter_dir / "data"
        use_nats = bool(nats_url)
        if use_nats:
            stream_dir = (
                Path(args.selfplay_dir)
                if args.selfplay_dir
                else Path("results/nnue_stream") / nats_run_id
            )
            total_games = max(1, args.games_per_iter)
            games_per_shard = max(1, args.games_per_shard)
            if args.nats_shards_per_iter > 0:
                target_shards = args.nats_shards_per_iter
            else:
                target_shards = int(
                    math.ceil(total_games / games_per_shard)
                )
            print(
                f"iter {iteration}: collecting {target_shards} shard(s) "
                f"from {stream_dir}",
                flush=True,
            )
            shard_paths = _collect_nats_shards(
                stream_dir,
                data_dir,
                target_shards,
                args.nats_poll_secs,
                args.selfplay_progress != 0,
            )
        elif args.skip_selfplay:
            if args.iterations > 1:
                raise ValueError("skip_selfplay only supports iterations=1.")
            if args.selfplay_dir:
                data_dir = Path(args.selfplay_dir)
            if not data_dir.exists():
                raise FileNotFoundError(
                    f"selfplay_dir not found: {data_dir}"
                )
            print(
                f"iter {iteration}: using existing shards from {data_dir}",
                flush=True,
            )
            shard_paths = sorted(data_dir.glob("*.lnue"))
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            total_games = max(1, args.games_per_iter)
            games_per_shard = max(1, args.games_per_shard)
            num_shards = int(math.ceil(total_games / games_per_shard))
            print(
                f"iter {iteration}: selfplay {total_games} games "
                f"(depth={args.depth}, shards={num_shards})",
                flush=True,
            )

            overall_progress = ProgressBar(
                total_games, "selfplay overall", unit="games"
            )
            games_done = 0
            for shard_idx in range(num_shards):
                remaining = total_games - shard_idx * games_per_shard
                shard_games = min(games_per_shard, remaining)
                shard_seed = args.seed + iteration * 100000 + shard_idx * 97
                shard_path = data_dir / f"shard_{shard_idx:04d}.lnue"
                print(
                    f"selfplay shard {shard_idx + 1}/{num_shards} start "
                    f"({shard_games} games)",
                    flush=True,
                )
                run_selfplay(
                    bin_path=selfplay_bin,
                    out_path=shard_path,
                    games=shard_games,
                    depth=args.depth,
                    seed=shard_seed,
                    chunk=args.chunk,
                    temperature=args.temperature,
                    temperature_end=temperature_end,
                    temperature_decay_plies=temperature_decay_plies,
                    alpha=args.alpha,
                    workers=args.workers,
                    shard_id=shard_idx,
                    progress=args.selfplay_progress != 0,
                    report_every=args.selfplay_report_every,
                    nnue_path=prev_nnue,
                    root_full_depth_top_k=args.root_full_depth_top_k,
                    root_reduced_depth=args.root_reduced_depth,
                    stream=args.selfplay_stream != 0,
                    resume=args.selfplay_resume != 0,
                    flush_every_games=args.selfplay_flush_every_games,
                    lock_path=(
                        (
                            Path(
                                f"{args.selfplay_lock}.{shard_idx:04d}"
                            )
                            if args.selfplay_lock and num_shards > 1
                            else Path(args.selfplay_lock)
                        )
                        if args.selfplay_lock
                        else None
                    ),
                    force_lock=args.selfplay_force_lock != 0,
                    chance_samples=args.chance_samples,
                    chance_sample_depth=args.chance_sample_depth,
                    chance_seed=args.chance_seed,
                )
                games_done += shard_games
                overall_progress.update(games_done)
                print()
                print(
                    f"selfplay shard {shard_idx + 1}/{num_shards} complete "
                    f"({shard_games} games)",
                    flush=True,
                )
            overall_progress.finish()
            shard_paths = sorted(data_dir.glob("*.lnue"))
        if not shard_paths:
            raise ValueError("No LNUE shards generated.")
        replay_candidates = _collect_replay_shards(
            output_dir,
            iteration,
            args.replay_iters,
            args.replay_max_shards,
        )

        header = LnueShardReader(str(shard_paths[0])).header
        model = NnueNet(feature_dim=header.feature_dim).to(device)
        torch.manual_seed(args.seed + iteration)

        for epoch in range(args.epochs):
            train_shards = shard_paths
            replay_subset: List[Path] = []
            if replay_candidates and args.replay_frac > 0.0:
                desired = int(round(args.replay_frac * len(shard_paths)))
                if desired > 0:
                    np_rng = np.random.default_rng(
                        args.seed + iteration * 1000 + epoch
                    )
                    if desired >= len(replay_candidates):
                        replay_subset = replay_candidates
                    else:
                        indices = np_rng.choice(
                            len(replay_candidates),
                            size=desired,
                            replace=False,
                        )
                        replay_subset = [
                            replay_candidates[i] for i in indices
                        ]
                    train_shards = shard_paths + replay_subset
                    print(
                        f"iter {iteration} epoch {epoch + 1}: "
                        f"replay {len(replay_subset)} shard(s) "
                        f"(pool {len(replay_candidates)})",
                        flush=True,
                    )
            _train_epoch(
                train_shards,
                model,
                device,
                args.batch_size,
                args.lr,
                args.weight_decay,
                args.ev_weight,
                args.quant_loss,
                args.quant_loss_weight,
                args.bootstrap_steps,
                bootstrap_alpha,
                args.seed + iteration * 1000,
                epoch,
            )

        nnue_path = iter_dir / f"nnue_iter_{iteration:02d}.nnue"
        _save_nnue(
            str(nnue_path),
            model,
            name=f"iter_{iteration:02d}",
            author="closed_loop",
            run_block_threshold=header.run_block_threshold,
        )
        if nats_url and args.nats_publish != 0:
            version = args.nats_version_start + iteration
            _publish_nnue(
                publish_bin,
                nats_url,
                nats_run_id,
                nnue_path,
                version,
            )

        eval_seed = args.seed + iteration * 1000000 + 777
        eval_depth = args.eval_depth if args.eval_depth > 0 else args.depth
        summary_random = _run_eval(
            eval_bin,
            nnue_path,
            None,
            args.eval_games,
            eval_depth,
            eval_seed,
            args.eval_workers,
            args.eval_progress != 0,
            args.eval_report_every,
            args.root_full_depth_top_k,
            args.root_reduced_depth,
            args.chance_samples,
            args.chance_sample_depth,
            args.chance_seed,
        )
        summary_prev = _run_eval(
            eval_bin,
            nnue_path,
            prev_nnue,
            args.eval_games,
            eval_depth,
            eval_seed + 1,
            args.eval_workers,
            args.eval_progress != 0,
            args.eval_report_every,
            args.root_full_depth_top_k,
            args.root_reduced_depth,
            args.chance_samples,
            args.chance_sample_depth,
            args.chance_seed,
        )

        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"iter={iteration} match=vs_random {summary_random}\n"
            )
            prev_label = prev_nnue.name if prev_nnue else "random"
            handle.write(
                f"iter={iteration} match=vs_prev({prev_label}) {summary_prev}\n"
            )

        print(summary_random)
        print(summary_prev)

        prev_nnue = nnue_path


if __name__ == "__main__":
    main()
