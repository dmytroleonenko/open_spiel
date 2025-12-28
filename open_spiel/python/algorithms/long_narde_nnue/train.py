"""Train a Long Narde NNUE from LNUE shards."""

from __future__ import annotations

import argparse
import glob
import os
import struct
import subprocess
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch

from open_spiel.python.algorithms.long_narde_nnue.lnue_reader import (
    LnueShardReader,
)

_Q = 127.0 / 64.0
_SCALE = 64.0
_NNUE_MAGIC = b"LNNU"
_NNUE_ENDIAN = 0x01020304
_NNUE_VERSION = 2
_NNUE_TYPE_INT8 = 1
_NNUE_TYPE_INT16 = 2


class NnueNet(torch.nn.Module):
    """Sparse NNUE network for Long Narde."""

    def __init__(self, feature_dim: int, l1: int = 256, l2: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.l1 = l1
        self.l2 = l2
        self.embed = torch.nn.EmbeddingBag(
            feature_dim, l1, mode="sum", include_last_offset=True
        )
        self.b0 = torch.nn.Parameter(torch.zeros(l1))
        self.fc1 = torch.nn.Linear(l1, l2)
        self.fc2 = torch.nn.Linear(l2, 3)

    def forward(
        self, indices: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Runs a forward pass on sparse feature indices."""
        x = self.embed(indices, offsets) + self.b0
        x = torch.clamp(x, min=0.0, max=_Q)
        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=_Q)
        return self.fc2(x)

    def quantization_error(self) -> torch.Tensor:
        """Returns quantization penalty for 1/64 grid."""
        total = torch.tensor(0.0, device=self.embed.weight.device)
        count = 0
        for param in self.parameters():
            if param.requires_grad:
                q = torch.clamp(torch.round(param * _SCALE) / _SCALE, -_Q, _Q)
                total = total + torch.sum((param - q) ** 2)
                count += param.numel()
        if count == 0:
            return total
        return total / float(count)

    def clip_parameters(self) -> None:
        """Clamps weights and biases into quantization range."""
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad:
                    param.clamp_(-_Q, _Q)


def _iter_batches(
    chunk, batch_size: int, rng: np.random.Generator
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yields index bags and targets for mini-batches within a chunk."""
    # pylint: disable=too-many-locals
    num_samples = chunk.outcome.shape[0]
    order = np.arange(num_samples)
    rng.shuffle(order)

    offsets = chunk.offsets
    indices = chunk.indices

    for start in range(0, num_samples, batch_size):
        batch_ids = order[start : start + batch_size]
        new_offsets = np.zeros(len(batch_ids) + 1, dtype=np.int64)
        segments = []
        total = 0
        for i, idx in enumerate(batch_ids):
            begin = offsets[idx]
            end = offsets[idx + 1]
            segments.append(indices[begin:end])
            total += int(end - begin)
            new_offsets[i + 1] = total
        if segments:
            batch_indices = np.concatenate(segments).astype(np.int64)
        else:
            batch_indices = np.zeros(0, dtype=np.int64)
        yield (
            batch_indices,
            new_offsets,
            chunk.outcome[batch_ids].astype(np.float32),
            chunk.target[batch_ids].astype(np.float32),
        )


def _targets_from_outcome(
    outcome: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives win/mars/opponent-mars targets from signed outcomes."""
    win = (outcome > 0.0).to(torch.float32)
    mars = (outcome == 2.0).to(torch.float32)
    opp_mars = (outcome == -2.0).to(torch.float32)
    return win, mars, opp_mars


def _ev_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Computes signed EV from NNUE logits."""
    p_win = torch.sigmoid(logits[:, 0])
    p_mars = torch.sigmoid(logits[:, 1])
    p_opp_mars = torch.sigmoid(logits[:, 2])
    return p_win + p_mars - (1.0 - p_win) - p_opp_mars


def _loss_from_logits(
    logits: torch.Tensor,
    outcome_t: torch.Tensor,
    target_t: torch.Tensor,
    criterion: torch.nn.Module,
    ev_weight: float,
) -> torch.Tensor:
    """Computes the combined BCE + EV regression loss."""
    win_t, mars_t, opp_mars_t = _targets_from_outcome(outcome_t)
    loss = (
        criterion(logits[:, 0], win_t)
        + criterion(logits[:, 1], mars_t)
        + criterion(logits[:, 2], opp_mars_t)
    )
    if ev_weight:
        ev = _ev_from_logits(logits)
        loss = loss + ev_weight * torch.mean((ev - target_t) ** 2)
    return loss


def _train_batch(
    model: NnueNet,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    indices: np.ndarray,
    offsets: np.ndarray,
    outcome: np.ndarray,
    target: np.ndarray,
    device: str,
    ev_weight: float,
    quant_loss: bool,
    quant_loss_weight: float,
) -> None:
    """Runs a single optimization step on one batch."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    indices_t = torch.from_numpy(indices).to(device)
    offsets_t = torch.from_numpy(offsets).to(device)
    outcome_t = torch.from_numpy(outcome).to(device)
    target_t = torch.from_numpy(target).to(device)

    optimizer.zero_grad()
    logits = model(indices_t, offsets_t)
    loss = _loss_from_logits(logits, outcome_t, target_t, criterion, ev_weight)
    if quant_loss:
        loss = loss + quant_loss_weight * model.quantization_error()
    loss.backward()
    optimizer.step()
    model.clip_parameters()


def _build_selfplay_cmd(
    bin_path: Path,
    out_path: Path,
    games: int,
    depth: int,
    seed: int,
    chunk: int,
    temperature: float,
    alpha: float,
    workers: int | None = None,
    shard_id: int | None = None,
    progress: bool | None = None,
    report_every: int | None = None,
    nnue_path: Path | None = None,
    chance_samples: int | None = None,
    chance_sample_depth: int | None = None,
    chance_seed: int | None = None,
) -> list[str]:
    """Builds a long_narde_selfplay command with optional flags."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cmd = _build_base_cmd(bin_path, games, depth, seed)
    cmd.extend(["--out", str(out_path)])
    cmd.extend(
        [
            "--chunk",
            str(chunk),
            "--temperature",
            str(temperature),
            "--alpha",
            str(alpha),
        ]
    )
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if shard_id is not None:
        cmd.extend(["--shard_id", str(shard_id)])
    if progress is not None:
        cmd.extend(["--progress", "1" if progress else "0"])
    if report_every is not None:
        cmd.extend(["--report_every", str(report_every)])
    if nnue_path is not None:
        cmd.extend(["--nnue", str(nnue_path)])
    if chance_samples is not None and chance_samples > 0:
        cmd.extend(["--chance_samples", str(chance_samples)])
        if chance_sample_depth is not None:
            cmd.extend(["--chance_sample_depth", str(chance_sample_depth)])
        if chance_seed is not None:
            cmd.extend(["--chance_seed", str(chance_seed)])
    return cmd


def _build_base_cmd(
    bin_path: Path, games: int, depth: int, seed: int
) -> list[str]:
    """Builds the shared portion of selfplay/eval command lines."""
    return [
        str(bin_path),
        "--games",
        str(games),
        "--depth",
        str(depth),
        "--seed",
        str(seed),
    ]


def run_selfplay(
    bin_path: Path,
    out_path: Path,
    games: int,
    depth: int,
    seed: int,
    chunk: int,
    temperature: float,
    alpha: float,
    workers: int | None = None,
    shard_id: int | None = None,
    progress: bool | None = None,
    report_every: int | None = None,
    nnue_path: Path | None = None,
    chance_samples: int | None = None,
    chance_sample_depth: int | None = None,
    chance_seed: int | None = None,
) -> None:
    """Runs long_narde_selfplay with optional flags."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cmd = _build_selfplay_cmd(
        bin_path=bin_path,
        out_path=out_path,
        games=games,
        depth=depth,
        seed=seed,
        chunk=chunk,
        temperature=temperature,
        alpha=alpha,
        workers=workers,
        shard_id=shard_id,
        progress=progress,
        report_every=report_every,
        nnue_path=nnue_path,
        chance_samples=chance_samples,
        chance_sample_depth=chance_sample_depth,
        chance_seed=chance_seed,
    )
    subprocess.run(cmd, check=True)


def _save_nnue(
    path: str,
    model: NnueNet,
    name: str,
    author: str,
    run_features_enabled: int = 1,
    run_block_threshold: int = 1,
) -> None:
    """Exports a quantized NNUE file for C++ inference."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    model.eval()
    with torch.no_grad():
        w0 = model.embed.weight.cpu().numpy()
        b0 = model.b0.cpu().numpy()
        w1 = model.fc1.weight.cpu().numpy()
        b1 = model.fc1.bias.cpu().numpy()
        w2 = model.fc2.weight.cpu().numpy()
        b2 = model.fc2.bias.cpu().numpy()

    w0_q = np.clip(np.round(w0 * _SCALE), -32768, 32767).astype(np.int16)
    b0_q = np.clip(np.round(b0 * _SCALE), -32768, 32767).astype(np.int16)
    w1_q = np.clip(np.round(w1 * _SCALE), -127, 127).astype(np.int8)
    b1_q = np.clip(np.round(b1 * _SCALE), -127, 127).astype(np.int8)
    w2_q = np.clip(np.round(w2 * _SCALE), -127, 127).astype(np.int8)
    b2_q = np.clip(np.round(b2 * _SCALE), -127, 127).astype(np.int8)

    name_bytes = name.encode("ascii", errors="ignore")[:64].ljust(64, b"\0")
    author_bytes = author.encode("ascii", errors="ignore")[:64].ljust(64, b"\0")

    payload_size = (
        len(name_bytes)
        + len(author_bytes)
        + w0_q.nbytes
        + b0_q.nbytes
        + w1_q.nbytes
        + b1_q.nbytes
        + w2_q.nbytes
        + b2_q.nbytes
    )

    l3 = int(model.fc2.out_features)
    header = struct.pack(
        "<4s16I",
        _NNUE_MAGIC,
        _NNUE_VERSION,
        _NNUE_ENDIAN,
        model.feature_dim,
        model.l1,
        model.l2,
        l3,
        run_features_enabled,
        run_block_threshold,
        _NNUE_TYPE_INT16,
        _NNUE_TYPE_INT8,
        _NNUE_TYPE_INT8,
        _NNUE_TYPE_INT16,
        _NNUE_TYPE_INT8,
        _NNUE_TYPE_INT8,
        payload_size,
        0,
    )

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(name_bytes)
        handle.write(author_bytes)
        handle.write(w0_q.tobytes(order="C"))
        handle.write(b0_q.tobytes(order="C"))
        handle.write(w1_q.tobytes(order="C"))
        handle.write(b1_q.tobytes(order="C"))
        handle.write(w2_q.tobytes(order="C"))
        handle.write(b2_q.tobytes(order="C"))


def train(args) -> None:
    """Training loop for LNUE shard datasets."""
    # pylint: disable=too-many-locals
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.lnue")))
    if not shard_paths:
        raise ValueError("No LNUE shards found.")

    first_header = LnueShardReader(shard_paths[0]).header
    model = NnueNet(feature_dim=first_header.feature_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(args.epochs):
        np_rng.shuffle(shard_paths)
        for shard_path in shard_paths:
            reader = LnueShardReader(shard_path)
            for chunk in reader.iter_chunks():
                for indices, offsets, outcome, target in _iter_batches(
                    chunk, args.batch_size, np_rng
                ):
                    batch_kwargs = {
                        "model": model,
                        "optimizer": optimizer,
                        "criterion": criterion,
                        "indices": indices,
                        "offsets": offsets,
                        "outcome": outcome,
                        "target": target,
                        "device": device,
                        "ev_weight": args.ev_weight,
                        "quant_loss": args.quant_loss,
                        "quant_loss_weight": args.quant_loss_weight,
                    }
                    _train_batch(**batch_kwargs)

        if args.output_path:
            out_path = args.output_path
            if args.epochs > 1:
                base, ext = os.path.splitext(out_path)
                out_path = f"{base}_epoch{epoch + 1}{ext}"
            _save_nnue(out_path, model, args.name, args.author)

def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Adds common optimizer/training flags to an argument parser."""
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ev_weight", type=float, default=0.5)
    parser.add_argument("--quant_loss", action="store_true")
    parser.add_argument("--quant_loss_weight", type=float, default=0.0001)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", required=True, help="Directory with LNUE shards."
    )
    parser.add_argument("--output_path", default="", help="Output NNUE path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    add_training_args(parser)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", default="LongNarde NNUE")
    parser.add_argument("--author", default="OpenSpiel")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
