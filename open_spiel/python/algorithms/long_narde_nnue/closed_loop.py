"""Closed-loop self-play training for Long Narde NNUE."""

from __future__ import annotations

import argparse
import math
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
    _iter_batches,
    _save_nnue,
    _targets_from_outcome,
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
        if percent == self.last_percent and (now - self.last_time) < self.min_interval:
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


def _count_samples(paths: Iterable[Path]) -> int:
    """Counts samples across LNUE shards."""
    total = 0
    for path in paths:
        reader = LnueShardReader(str(path))
        for chunk in reader.iter_chunks():
            total += int(chunk.outcome.shape[0])
    return total


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
            for indices, offsets, outcome, target in _iter_batches(
                chunk, batch_size, np_rng
            ):
                indices_t = torch.from_numpy(indices).to(device)
                offsets_t = torch.from_numpy(offsets).to(device)
                outcome_t = torch.from_numpy(outcome).to(device)
                target_t = torch.from_numpy(target).to(device)

                optimizer.zero_grad()
                logits = model(indices_t, offsets_t)
                win_t, mars_t = _targets_from_outcome(outcome_t)
                loss = criterion(logits[:, 0], win_t) + criterion(
                    logits[:, 1], mars_t
                )
                ev = torch.sigmoid(logits[:, 0]) + torch.sigmoid(logits[:, 1])
                loss = loss + ev_weight * torch.mean((ev - target_t) ** 2)
                if quant_loss:
                    loss = loss + quant_loss_weight * model.quantization_error()
                loss.backward()
                optimizer.step()
                model.clip_parameters()

                processed += int(outcome.shape[0])
                progress.update(processed)

    progress.finish()


def _run_selfplay(
    bin_path: Path,
    out_path: Path,
    games: int,
    depth: int,
    seed: int,
    shard_id: int,
    workers: int,
    chunk: int,
    temperature: float,
    alpha: float,
    nnue_path: Path | None,
    progress: bool,
    report_every: int,
) -> None:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cmd = [
        str(bin_path),
        "--out",
        str(out_path),
        "--games",
        str(games),
        "--depth",
        str(depth),
        "--seed",
        str(seed),
        "--shard_id",
        str(shard_id),
        "--workers",
        str(workers),
        "--chunk",
        str(chunk),
        "--temperature",
        str(temperature),
        "--alpha",
        str(alpha),
        "--progress",
        "1" if progress else "0",
        "--report_every",
        str(report_every),
    ]
    if nnue_path is not None:
        cmd.extend(["--nnue", str(nnue_path)])
    subprocess.run(cmd, check=True)


def _run_eval(
    bin_path: Path,
    nnue_a: Path | None,
    nnue_b: Path | None,
    games: int,
    depth: int,
    seed: int,
) -> str:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cmd = [
        str(bin_path),
        "--games",
        str(games),
        "--depth",
        str(depth),
        "--seed",
        str(seed),
        "--progress",
        "0",
    ]
    if nnue_a is not None:
        cmd.extend(["--nnue_a", str(nnue_a)])
    if nnue_b is not None:
        cmd.extend(["--nnue_b", str(nnue_b)])
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Eval produced no output.")
    return lines[-1].strip()


def main() -> None:
    """Runs closed-loop self-play training and evaluation."""
    # pylint: disable=too-many-locals,too-many-statements
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="results/nnue_closed_loop")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games_per_iter", type=int, default=10000)
    parser.add_argument("--games_per_shard", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--selfplay_progress", type=int, default=1)
    parser.add_argument("--selfplay_report_every", type=int, default=100)
    parser.add_argument("--eval_games", type=int, default=1000)
    parser.add_argument("--eval_depth", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ev_weight", type=float, default=0.5)
    parser.add_argument("--quant_loss", action="store_true")
    parser.add_argument("--quant_loss_weight", type=float, default=0.0001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init_nnue", default="")
    parser.add_argument("--selfplay_bin", default="")
    parser.add_argument("--eval_bin", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"

    repo_root = _repo_root()
    selfplay_bin = (
        Path(args.selfplay_bin)
        if args.selfplay_bin
        else repo_root / "build" / "games" / "long_narde_selfplay"
    )
    eval_bin = (
        Path(args.eval_bin)
        if args.eval_bin
        else repo_root / "build" / "games" / "long_narde_eval"
    )

    if not selfplay_bin.exists():
        raise FileNotFoundError(f"selfplay binary not found: {selfplay_bin}")
    if not eval_bin.exists():
        raise FileNotFoundError(f"eval binary not found: {eval_bin}")

    prev_nnue = Path(args.init_nnue) if args.init_nnue else None

    for iteration in range(args.iterations):
        iter_dir = output_dir / f"iter_{iteration:02d}"
        data_dir = iter_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        total_games = max(1, args.games_per_iter)
        games_per_shard = max(1, args.games_per_shard)
        num_shards = int(math.ceil(total_games / games_per_shard))
        print(
            f"iter {iteration}: selfplay {total_games} games "
            f"(depth={args.depth}, shards={num_shards})",
            flush=True,
        )

        overall_progress = ProgressBar(total_games, "selfplay overall", unit="games")
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
            _run_selfplay(
                selfplay_bin,
                shard_path,
                shard_games,
                args.depth,
                shard_seed,
                shard_idx,
                args.workers,
                args.chunk,
                args.temperature,
                args.alpha,
                prev_nnue,
                args.selfplay_progress != 0,
                args.selfplay_report_every,
            )
            games_done += shard_games
            overall_progress.update(games_done)
            print(
                f"selfplay shard {shard_idx + 1}/{num_shards} complete "
                f"({shard_games} games)",
                flush=True,
            )
        overall_progress.finish()

        shard_paths = sorted(data_dir.glob("*.lnue"))
        if not shard_paths:
            raise ValueError("No LNUE shards generated.")

        header = LnueShardReader(str(shard_paths[0])).header
        model = NnueNet(feature_dim=header.feature_dim).to(args.device)
        torch.manual_seed(args.seed + iteration)

        for epoch in range(args.epochs):
            _train_epoch(
                shard_paths,
                model,
                args.device,
                args.batch_size,
                args.lr,
                args.weight_decay,
                args.ev_weight,
                args.quant_loss,
                args.quant_loss_weight,
                args.seed + iteration * 1000,
                epoch,
            )

        nnue_path = iter_dir / f"nnue_iter_{iteration:02d}.nnue"
        _save_nnue(
            str(nnue_path),
            model,
            name=f"iter_{iteration:02d}",
            author="closed_loop",
            run_features_enabled=1,
            run_block_threshold=header.run_block_threshold,
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
        )
        summary_prev = _run_eval(
            eval_bin,
            nnue_path,
            prev_nnue,
            args.eval_games,
            eval_depth,
            eval_seed + 1,
        )

        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"iter={iteration} match=vs_random {summary_random}\n"
            )
            prev_label = prev_nnue.name if prev_nnue else "random"
            handle.write(
                f"iter={iteration} match=vs_prev({prev_label}) {summary_prev}\n"
            )

        print(f"iter {iteration} vs_random: {summary_random}")
        print(f"iter {iteration} vs_prev({prev_label}): {summary_prev}")

        prev_nnue = nnue_path


if __name__ == "__main__":
    main()
