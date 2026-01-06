"""Train a Long Narde NNUE from LNUE shards."""
# pylint: disable=duplicate-code

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
_NNUE_VERSION = 3
_NNUE_TYPE_INT8 = 1
_NNUE_TYPE_INT16 = 2
_NNUE_FEATURE_FLAGS = (1 << 0) | (1 << 1) | (1 << 2)
_NNUE_HEADER_FMT = "<4s16I"
_NNUE_HEADER_SIZE = struct.calcsize(_NNUE_HEADER_FMT)
_BOOTSTRAP_PLY_SHIFT = 16


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
        self.fc2 = torch.nn.Linear(l2, 4)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
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


def _mps_embedding_bag_supported() -> bool:
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        return False
    try:
        weight = torch.zeros((2, 2), device="mps")
        indices = torch.tensor([0], dtype=torch.int64, device="mps")
        offsets = torch.tensor([0], dtype=torch.int64, device="mps")
        torch.nn.functional.embedding_bag(indices, weight, offsets)
        return True
    except (NotImplementedError, RuntimeError):
        return False


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if _mps_embedding_bag_supported():
        return "mps"
    if torch.backends.mps.is_available():
        print("MPS embedding_bag unsupported; falling back to CPU.", flush=True)
    return "cpu"


def _configure_torch_threads(
    torch_threads: int,
    torch_interop_threads: int,
) -> tuple[int, int]:
    """Configures PyTorch CPU thread pools.

    Args:
      torch_threads: Intra-op threads. <=0 means "use all cores".
      torch_interop_threads: Inter-op threads. <=0 means "leave default".

    Returns:
      (intra_threads, interop_threads) after configuration.
    """
    cpu_count = os.cpu_count() or 1
    threads = int(torch_threads)
    if threads <= 0:
        threads = cpu_count
    threads = max(1, threads)
    torch.set_num_threads(threads)
    interop = int(torch_interop_threads)
    if interop > 0:
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            pass
    return torch.get_num_threads(), torch.get_num_interop_threads()


def _parse_lr_milestones(spec: str) -> list[tuple[int, float]]:
    """Parses a comma-separated list of step:lr pairs."""
    milestones: list[tuple[int, float]] = []
    spec = spec.strip()
    if not spec:
        return milestones
    for entry in spec.split(","):
        item = entry.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid lr milestone {item!r} (expected step:lr).")
        step_str, lr_str = item.split(":", 1)
        step = int(step_str.strip())
        lr = float(lr_str.strip())
        if step < 0:
            raise ValueError(f"lr milestone step must be >=0, got {step}.")
        if lr <= 0.0:
            raise ValueError(f"lr milestone lr must be >0, got {lr}.")
        milestones.append((step, lr))
    milestones.sort(key=lambda pair: pair[0])
    return milestones


def _lr_for_step(base_lr: float, milestones: list[tuple[int, float]], step: int) -> float:
    """Returns the learning rate for a given step."""
    lr = float(base_lr)
    for milestone_step, milestone_lr in milestones:
        if step < milestone_step:
            break
        lr = float(milestone_lr)
    return lr


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _iter_batches(
    chunk,
    batch_size: int,
    rng: np.random.Generator,
    target: np.ndarray | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yields index bags and targets for mini-batches within a chunk."""
    # pylint: disable=too-many-locals
    num_samples = chunk.outcome.shape[0]
    order = np.arange(num_samples)
    rng.shuffle(order)

    offsets = chunk.offsets
    indices = chunk.indices
    target_arr = target if target is not None else chunk.target

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
            target_arr[batch_ids].astype(np.float32),
            chunk.mobility[batch_ids].astype(np.float32),
        )


def _bootstrap_targets(
    chunk,
    steps: int,
    alpha: float,
    lookup: dict[int, float] | None = None,
) -> np.ndarray:
    if steps <= 0:
        return chunk.target
    alpha = float(np.clip(alpha, 0.0, 1.0))
    outcome = chunk.outcome.astype(np.float32)
    target = np.empty_like(chunk.target, dtype=np.float32)
    if lookup is None:
        game_map: dict[int, dict[int, int]] = {}
        for idx, (game_id, ply) in enumerate(zip(chunk.game_id, chunk.ply)):
            game_map.setdefault(int(game_id), {})[int(ply)] = idx
        for idx, (game_id, ply) in enumerate(zip(chunk.game_id, chunk.ply)):
            ply_lookup = game_map[int(game_id)]
            next_idx = ply_lookup.get(int(ply) + steps)
            if next_idx is None:
                bootstrap = float(chunk.v_search[idx])
            else:
                bootstrap = float(chunk.v_search[next_idx])
            target[idx] = alpha * bootstrap + (1.0 - alpha) * outcome[idx]
        return target
    for idx, (game_id, ply) in enumerate(zip(chunk.game_id, chunk.ply)):
        next_key = (int(game_id) << _BOOTSTRAP_PLY_SHIFT) | (
            int(ply) + steps
        )
        bootstrap = lookup.get(next_key)
        if bootstrap is None:
            bootstrap = float(chunk.v_search[idx])
        target[idx] = alpha * bootstrap + (1.0 - alpha) * outcome[idx]
    return target


def _build_bootstrap_lookup(shard_path: str) -> dict[int, float]:
    lookup: dict[int, float] = {}
    reader = LnueShardReader(shard_path)
    for chunk in reader.iter_chunks():
        for game_id, ply, v_search in zip(
            chunk.game_id, chunk.ply, chunk.v_search
        ):
            key = (int(game_id) << _BOOTSTRAP_PLY_SHIFT) | int(ply)
            lookup[key] = float(v_search)
    return lookup


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


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _loss_from_logits(
    logits: torch.Tensor,
    outcome_t: torch.Tensor,
    target_t: torch.Tensor,
    mobility_t: torch.Tensor,
    criterion: torch.nn.Module,
    ev_weight: float,
    mobility_weight: float,
) -> torch.Tensor:
    """Computes the combined BCE + EV regression loss."""
    win_t, mars_t, opp_mars_t = _targets_from_outcome(outcome_t)
    loss = (
        criterion(logits[:, 0], win_t)
        + criterion(logits[:, 1], mars_t)
        + criterion(logits[:, 2], opp_mars_t)
    )
    if mobility_weight:
        p_mobility = torch.sigmoid(logits[:, 3])
        loss = loss + mobility_weight * torch.mean((p_mobility - mobility_t) ** 2)
    if ev_weight:
        ev = _ev_from_logits(logits)
        loss = loss + ev_weight * torch.mean((ev - target_t) ** 2)
    return loss


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _train_batch(
    model: NnueNet,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    indices: np.ndarray,
    offsets: np.ndarray,
    outcome: np.ndarray,
    target: np.ndarray,
    mobility: np.ndarray,
    device: str,
    ev_weight: float,
    mobility_weight: float,
    quant_loss: bool,
    quant_loss_weight: float,
    grad_clip_norm: float,
) -> None:
    """Runs a single optimization step on one batch."""
    # pylint: disable=too-many-locals,too-many-branches
    indices_t = torch.from_numpy(indices).to(device)
    offsets_t = torch.from_numpy(offsets).to(device)
    outcome_t = torch.from_numpy(outcome).to(device)
    target_t = torch.from_numpy(target).to(device)
    mobility_t = torch.from_numpy(mobility).to(device)

    optimizer.zero_grad()
    logits = model(indices_t, offsets_t)
    loss = _loss_from_logits(
        logits, outcome_t, target_t, mobility_t, criterion, ev_weight, mobility_weight
    )
    if quant_loss:
        loss = loss + quant_loss_weight * model.quantization_error()
    loss.backward()
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
    temperature_end: float | None,
    temperature_decay_plies: int | None,
    alpha: float,
    workers: int | None = None,
    shard_id: int | None = None,
    progress: bool | None = None,
    report_every: int | None = None,
    nnue_path: Path | None = None,
    root_full_depth_top_k: int | None = None,
    root_reduced_depth: int | None = None,
    stream: bool | None = None,
    resume: bool | None = None,
    flush_every_games: int | None = None,
    lock_path: Path | None = None,
    force_lock: bool | None = None,
    chance_samples: int | None = None,
    chance_sample_depth: int | None = None,
    chance_seed: int | None = None,
) -> list[str]:
    """Builds a long_narde_selfplay command with optional flags."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    cmd = _build_base_cmd(bin_path, games, depth, seed)
    cmd.extend(["--out", str(out_path)])
    cmd.extend(["--chunk", str(chunk), "--temperature", str(temperature)])
    if temperature_end is not None:
        cmd.extend(["--temperature_end", str(temperature_end)])
    if temperature_decay_plies is not None:
        cmd.extend(["--temperature_decay_plies", str(temperature_decay_plies)])
    cmd.extend(["--alpha", str(alpha)])
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
    if root_full_depth_top_k is not None:
        cmd.extend(["--root_full_depth_top_k", str(root_full_depth_top_k)])
    if root_reduced_depth is not None:
        cmd.extend(["--root_reduced_depth", str(root_reduced_depth)])
    if stream is not None:
        cmd.extend(["--stream", "1" if stream else "0"])
    if resume is not None:
        cmd.extend(["--resume", "1" if resume else "0"])
    if flush_every_games is not None:
        cmd.extend(["--flush_every_games", str(flush_every_games)])
    if lock_path is not None:
        cmd.extend(["--lock", str(lock_path)])
    if force_lock is not None:
        cmd.extend(["--force_lock", "1" if force_lock else "0"])
    if chance_samples is not None and chance_samples > 0:
        cmd.extend(["--chance_samples", str(chance_samples)])
        if chance_sample_depth is not None:
            cmd.extend(["--chance_sample_depth", str(chance_sample_depth)])
        if chance_seed is not None:
            cmd.extend(["--chance_seed", str(chance_seed)])
    return cmd


def _build_base_cmd(bin_path: Path, games: int, depth: int, seed: int) -> list[str]:
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
    temperature_end: float | None = None,
    temperature_decay_plies: int | None = None,
    workers: int | None = None,
    shard_id: int | None = None,
    progress: bool | None = None,
    report_every: int | None = None,
    nnue_path: Path | None = None,
    root_full_depth_top_k: int | None = None,
    root_reduced_depth: int | None = None,
    stream: bool | None = None,
    resume: bool | None = None,
    flush_every_games: int | None = None,
    lock_path: Path | None = None,
    force_lock: bool | None = None,
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
        temperature_end=temperature_end,
        temperature_decay_plies=temperature_decay_plies,
        alpha=alpha,
        workers=workers,
        shard_id=shard_id,
        progress=progress,
        report_every=report_every,
        nnue_path=nnue_path,
        root_full_depth_top_k=root_full_depth_top_k,
        root_reduced_depth=root_reduced_depth,
        stream=stream,
        resume=resume,
        flush_every_games=flush_every_games,
        lock_path=lock_path,
        force_lock=force_lock,
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
    run_features_enabled: int = _NNUE_FEATURE_FLAGS,
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


def _load_nnue(path: str, model: NnueNet, device: str) -> tuple[str, str]:
    """Loads a quantized NNUE file into a PyTorch model.

    Args:
      path: NNUE file path.
      model: Model to populate (must match dimensions).
      device: Torch device string to load onto.

    Returns:
      (name, author) strings from the NNUE header.
    """
    # pylint: disable=too-many-locals
    with open(path, "rb") as handle:
        header_bytes = handle.read(_NNUE_HEADER_SIZE)
        if len(header_bytes) != _NNUE_HEADER_SIZE:
            raise ValueError("Incomplete NNUE header.")
        (
            magic,
            version,
            endian,
            feature_dim,
            l1,
            l2,
            l3,
            _run_features_enabled,
            _run_block_threshold,
            w0_type,
            w1_type,
            w2_type,
            b0_type,
            b1_type,
            b2_type,
            payload_size,
            _,
        ) = struct.unpack(_NNUE_HEADER_FMT, header_bytes)
        if magic != _NNUE_MAGIC or endian != _NNUE_ENDIAN or version != _NNUE_VERSION:
            raise ValueError(
                "Unsupported NNUE header: "
                f"magic={magic!r} endian={endian} version={version}"
            )
        if payload_size <= 0:
            raise ValueError("Invalid NNUE payload size.")
        if w0_type != _NNUE_TYPE_INT16 or b0_type != _NNUE_TYPE_INT16:
            raise ValueError("Unsupported NNUE embedding types.")
        if (
            w1_type != _NNUE_TYPE_INT8
            or b1_type != _NNUE_TYPE_INT8
            or w2_type != _NNUE_TYPE_INT8
            or b2_type != _NNUE_TYPE_INT8
        ):
            raise ValueError("Unsupported NNUE linear types.")
        if feature_dim != model.feature_dim:
            raise ValueError(
                f"NNUE feature_dim mismatch: {feature_dim} != {model.feature_dim}"
            )
        if l1 != model.l1 or l2 != model.l2:
            raise ValueError("NNUE hidden size mismatch.")
        if l3 != int(model.fc2.out_features):
            raise ValueError("NNUE output size mismatch.")

        name = handle.read(64).split(b"\0", 1)[0].decode("ascii", errors="ignore")
        author = (
            handle.read(64).split(b"\0", 1)[0].decode("ascii", errors="ignore")
        )

        w0_bytes = feature_dim * l1 * 2
        b0_bytes = l1 * 2
        w1_bytes = l1 * l2
        b1_bytes = l2
        w2_bytes = l2 * l3
        b2_bytes = l3

        w0_q = np.frombuffer(handle.read(w0_bytes), dtype=np.int16).copy()
        b0_q = np.frombuffer(handle.read(b0_bytes), dtype=np.int16).copy()
        w1_q = np.frombuffer(handle.read(w1_bytes), dtype=np.int8).copy()
        b1_q = np.frombuffer(handle.read(b1_bytes), dtype=np.int8).copy()
        w2_q = np.frombuffer(handle.read(w2_bytes), dtype=np.int8).copy()
        b2_q = np.frombuffer(handle.read(b2_bytes), dtype=np.int8).copy()

    w0 = w0_q.reshape((feature_dim, l1)).astype(np.float32) / _SCALE
    b0 = b0_q.astype(np.float32) / _SCALE
    w1 = w1_q.reshape((l2, l1)).astype(np.float32) / _SCALE
    b1 = b1_q.astype(np.float32) / _SCALE
    w2 = w2_q.reshape((l3, l2)).astype(np.float32) / _SCALE
    b2 = b2_q.astype(np.float32) / _SCALE

    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(w0).to(device))
        model.b0.copy_(torch.from_numpy(b0).to(device))
        model.fc1.weight.copy_(torch.from_numpy(w1).to(device))
        model.fc1.bias.copy_(torch.from_numpy(b1).to(device))
        model.fc2.weight.copy_(torch.from_numpy(w2).to(device))
        model.fc2.bias.copy_(torch.from_numpy(b2).to(device))
    model.clip_parameters()
    return name, author


def train(args) -> None:
    """Training loop for LNUE shard datasets."""
    # pylint: disable=too-many-locals
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    device = _resolve_device(args.device)
    intra, interop = _configure_torch_threads(
        getattr(args, "torch_threads", 0),
        getattr(args, "torch_interop_threads", 0),
    )
    print(f"torch threads: intra={intra} interop={interop}", flush=True)
    lr_milestones = _parse_lr_milestones(getattr(args, "lr_milestones", ""))
    grad_clip_norm = float(getattr(args, "grad_clip_norm", 0.0))
    bootstrap_steps = max(0, int(args.bootstrap_steps))
    bootstrap_alpha = args.bootstrap_alpha
    if bootstrap_alpha < 0.0:
        bootstrap_alpha = 0.5

    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.lnue")))
    if not shard_paths:
        raise ValueError("No LNUE shards found.")

    first_header = LnueShardReader(shard_paths[0]).header
    model = NnueNet(feature_dim=first_header.feature_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    _set_optimizer_lr(optimizer, _lr_for_step(args.lr, lr_milestones, 0))
    bootstrap_cache: dict[str, dict[int, float]] = {}

    for epoch in range(args.epochs):
        _set_optimizer_lr(optimizer, _lr_for_step(args.lr, lr_milestones, epoch))
        np_rng.shuffle(shard_paths)
        for shard_path in shard_paths:
            lookup = None
            if bootstrap_steps > 0:
                lookup = bootstrap_cache.get(shard_path)
                if lookup is None:
                    lookup = _build_bootstrap_lookup(shard_path)
                    bootstrap_cache[shard_path] = lookup
            reader = LnueShardReader(shard_path)
            for chunk in reader.iter_chunks():
                target_override = None
                if bootstrap_steps > 0:
                    target_override = _bootstrap_targets(
                        chunk, bootstrap_steps, bootstrap_alpha, lookup
                    )
                for indices, offsets, outcome, target, mobility in _iter_batches(
                    chunk, args.batch_size, np_rng, target_override
                ):
                    batch_kwargs = {
                        "model": model,
                        "optimizer": optimizer,
                        "criterion": criterion,
                        "indices": indices,
                        "offsets": offsets,
                        "outcome": outcome,
                        "target": target,
                        "mobility": mobility,
                        "device": device,
                        "ev_weight": args.ev_weight,
                        "mobility_weight": args.mobility_weight,
                        "quant_loss": args.quant_loss,
                        "quant_loss_weight": args.quant_loss_weight,
                        "grad_clip_norm": grad_clip_norm,
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
    parser.add_argument("--mobility_weight", type=float, default=0.1)
    parser.add_argument(
        "--lr_milestones",
        default="",
        help=(
            "Comma-separated step:lr pairs to override --lr "
            "(step=epoch for train.py, step=iteration for closed_loop)."
        ),
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.0,
        help="Global gradient clipping (0 disables).",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="PyTorch intra-op CPU threads (<=0 uses all cores).",
    )
    parser.add_argument(
        "--torch_interop_threads",
        type=int,
        default=0,
        help="PyTorch inter-op threads (<=0 keeps default).",
    )
    parser.add_argument("--quant_loss", action="store_true")
    parser.add_argument("--quant_loss_weight", type=float, default=0.0001)
    parser.add_argument("--bootstrap_steps", type=int, default=0)
    parser.add_argument("--bootstrap_alpha", type=float, default=-1.0)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory with LNUE shards.")
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
