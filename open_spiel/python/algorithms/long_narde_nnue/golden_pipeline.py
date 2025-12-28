"""Golden pipeline harness for Long Narde NNUE."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Iterable, Tuple

import numpy as np
import torch

from open_spiel.python.algorithms.long_narde_nnue.lnue_reader import (
    LnueShardReader,
)
from open_spiel.python.algorithms.long_narde_nnue.train import (
    NnueNet,
    _loss_from_logits,
    _save_nnue,
    run_selfplay,
)

_LNNU_HEADER_FMT = "<4s16I"
_LNNU_MAGIC = b"LNNU"

_NUM_POINTS = 24
_BUCKETS = 16
_RUN_MIN = 2
_RUN_MAX = 6
_BASE_PER_SIDE = (_NUM_POINTS + 1) * _BUCKETS
_BASE_FEATURES = _BASE_PER_SIDE * 2
_RUN_FEATURES_PER_SIDE = sum(
    _NUM_POINTS - length + 1 for length in range(_RUN_MIN, _RUN_MAX + 1)
)
_MAX_ACTIVE = _BASE_FEATURES // _BUCKETS + _RUN_FEATURES_PER_SIDE * 2
_MIN_ACTIVE = _BASE_FEATURES // _BUCKETS


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_lnue(reader: LnueShardReader) -> None:
    header = reader.header
    if header.feature_dim <= 0:
        raise ValueError("LNUE header feature_dim is invalid.")
    if header.run_block_threshold != 1:
        raise ValueError("Unexpected run block threshold in LNUE header.")
    for chunk in reader.iter_chunks():
        offsets = chunk.offsets
        if offsets[0] != 0:
            raise ValueError("LNUE offsets must start at 0.")
        if not np.all(offsets[:-1] <= offsets[1:]):
            raise ValueError("LNUE offsets must be non-decreasing.")
        if np.any(chunk.indices >= header.feature_dim):
            raise ValueError("LNUE indices exceed feature_dim.")
        counts = offsets[1:] - offsets[:-1]
        if np.any(counts < _MIN_ACTIVE) or np.any(counts > _MAX_ACTIVE):
            raise ValueError("Unexpected active feature count.")
        if np.any(chunk.v_search < -2.0) or np.any(chunk.v_search > 2.0):
            raise ValueError("v_search out of range.")
        if np.any(chunk.target < -2.0) or np.any(chunk.target > 2.0):
            raise ValueError("target out of range.")
        if not np.all(np.isin(chunk.outcome, [-2, -1, 1, 2])):
            raise ValueError("outcome values out of range.")
        break


def _gather_batch(
    chunk, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # pylint: disable=duplicate-code
    num_samples = min(batch_size, chunk.outcome.shape[0])
    batch_ids = np.arange(num_samples)
    offsets = chunk.offsets
    indices = chunk.indices
    new_offsets = np.zeros(len(batch_ids) + 1, dtype=np.int64)
    segments = []
    total = 0
    for i, idx in enumerate(batch_ids):
        start = offsets[idx]
        end = offsets[idx + 1]
        segments.append(indices[start:end])
        total += int(end - start)
        new_offsets[i + 1] = total
    if segments:
        batch_indices = np.concatenate(segments).astype(np.int64)
    else:
        batch_indices = np.zeros(0, dtype=np.int64)
    return (
        batch_indices,
        new_offsets,
        chunk.outcome[batch_ids].astype(np.float32),
        chunk.target[batch_ids].astype(np.float32),
    )


def _train_one_step(reader: LnueShardReader, batch_size: int) -> NnueNet:
    # pylint: disable=too-many-locals
    chunk = next(reader.iter_chunks())
    indices, offsets, outcome, target = _gather_batch(chunk, batch_size)

    model = NnueNet(feature_dim=reader.header.feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    indices_t = torch.from_numpy(indices)
    offsets_t = torch.from_numpy(offsets)
    outcome_t = torch.from_numpy(outcome)
    target_t = torch.from_numpy(target)

    logits = model(indices_t, offsets_t)
    loss = _loss_from_logits(logits, outcome_t, target_t, criterion, 0.5)
    loss.backward()
    optimizer.step()
    model.clip_parameters()
    return model


def _read_nnue(path: Path) -> dict:
    # pylint: disable=too-many-locals
    with open(path, "rb") as handle:
        header_bytes = handle.read(struct.calcsize(_LNNU_HEADER_FMT))
        if len(header_bytes) != struct.calcsize(_LNNU_HEADER_FMT):
            raise ValueError("Incomplete LNNU header.")
        (
            magic,
            version,
            endian,
            feature_dim,
            l1,
            l2,
            l3,
            run_features_enabled,
            run_block_threshold,
            w0_type,
            w1_type,
            w2_type,
            b0_type,
            b1_type,
            b2_type,
            payload_size,
            _,
        ) = struct.unpack(_LNNU_HEADER_FMT, header_bytes)
        if magic != _LNNU_MAGIC:
            raise ValueError("Invalid LNNU magic.")
        if payload_size <= 0:
            raise ValueError("Invalid LNNU payload size.")
        name = handle.read(64)
        author = handle.read(64)
        w0 = np.frombuffer(handle.read(feature_dim * l1 * 2), dtype=np.int16).copy()
        b0 = np.frombuffer(handle.read(l1 * 2), dtype=np.int16).copy()
        w1 = np.frombuffer(handle.read(l1 * l2), dtype=np.int8).copy()
        b1 = np.frombuffer(handle.read(l2), dtype=np.int8).copy()
        w2 = np.frombuffer(handle.read(l2 * l3), dtype=np.int8).copy()
        b2 = np.frombuffer(handle.read(l3), dtype=np.int8).copy()
    return {
        "version": version,
        "endian": endian,
        "feature_dim": feature_dim,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "run_features_enabled": run_features_enabled,
        "run_block_threshold": run_block_threshold,
        "w0_type": w0_type,
        "w1_type": w1_type,
        "w2_type": w2_type,
        "b0_type": b0_type,
        "b1_type": b1_type,
        "b2_type": b2_type,
        "name": name,
        "author": author,
        "w0": w0.reshape((feature_dim, l1)),
        "b0": b0,
        "w1": w1.reshape((l2, l1)),
        "b1": b1,
        "w2": w2.reshape((l3, l2)),
        "b2": b2,
    }


def _trunc_div(value: int, denom: int) -> int:
    if value >= 0:
        return value // denom
    return -((-value) // denom)


def _infer_raw_logits(nnue: dict, features: Iterable[int]) -> Tuple[int, int, int]:
    # pylint: disable=too-many-locals
    w0 = nnue["w0"]
    b0 = nnue["b0"]
    w1 = nnue["w1"]
    b1 = nnue["b1"]
    w2 = nnue["w2"]
    b2 = nnue["b2"]

    acc = b0.copy()
    for feat in features:
        acc = (acc + w0[feat]).astype(np.int16)
    l1 = np.clip(acc, 0, 127).astype(np.int8)

    l2 = np.empty(nnue["l2"], dtype=np.int8)
    for o in range(nnue["l2"]):
        dot = int(np.dot(l1.astype(np.int32), w1[o].astype(np.int32)))
        total = int(b1[o]) * 64 + dot
        scaled = _trunc_div(total, 64)
        l2[o] = np.clip(scaled, 0, 127)

    dot_win = int(np.dot(l2.astype(np.int32), w2[0].astype(np.int32)))
    out_win = _trunc_div(int(b2[0]) * 64 + dot_win, 64)
    dot_mars = int(np.dot(l2.astype(np.int32), w2[1].astype(np.int32)))
    out_mars = _trunc_div(int(b2[1]) * 64 + dot_mars, 64)
    dot_opp_mars = int(np.dot(l2.astype(np.int32), w2[2].astype(np.int32)))
    out_opp_mars = _trunc_div(int(b2[2]) * 64 + dot_opp_mars, 64)
    return out_win, out_mars, out_opp_mars


def _iter_feature_slices(
    reader: LnueShardReader, max_samples: int
) -> Iterable[np.ndarray]:
    yielded = 0
    for chunk in reader.iter_chunks():
        offsets = chunk.offsets
        indices = chunk.indices
        for i in range(chunk.outcome.shape[0]):
            if max_samples > 0:
                if yielded >= max_samples:
                    return
            start = offsets[i]
            end = offsets[i + 1]
            yield indices[start:end]
            yielded += 1


def _read_probe_output(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        raw = handle.read(4)
        if len(raw) != 4:
            raise ValueError("Probe output missing count.")
        count = struct.unpack("<I", raw)[0]
        data = np.frombuffer(handle.read(), dtype=np.int32)
    if data.size != count * 3:
        raise ValueError("Probe output size mismatch.")
    return data.reshape((count, 3))


def main() -> None:
    """Runs the golden pipeline harness."""
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfplay_bin", default="", help="Path to selfplay binary")
    parser.add_argument("--probe_bin", default="", help="Path to LNUE probe binary")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=64)
    args = parser.parse_args()

    repo_root = _repo_root()
    selfplay_bin = (
        Path(args.selfplay_bin)
        if args.selfplay_bin
        else repo_root / "build" / "games" / "long_narde_selfplay"
    )
    probe_bin = (
        Path(args.probe_bin)
        if args.probe_bin
        else repo_root / "build" / "games" / "long_narde_lnue_probe"
    )

    if not selfplay_bin.exists():
        raise FileNotFoundError(f"selfplay binary not found: {selfplay_bin}")
    if not probe_bin.exists():
        raise FileNotFoundError(f"probe binary not found: {probe_bin}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        shard_a = tmp_path / "selfplay_a.lnue"
        shard_b = tmp_path / "selfplay_b.lnue"

        run_selfplay(
            bin_path=selfplay_bin,
            out_path=shard_a,
            games=args.games,
            depth=args.depth,
            seed=args.seed,
            chunk=args.chunk,
            temperature=args.temperature,
            alpha=args.alpha,
            workers=0,
            nnue_path=None,
        )
        run_selfplay(
            bin_path=selfplay_bin,
            out_path=shard_b,
            games=args.games,
            depth=args.depth,
            seed=args.seed,
            chunk=args.chunk,
            temperature=args.temperature,
            alpha=args.alpha,
            workers=0,
            nnue_path=None,
        )

        if _hash_file(shard_a) != _hash_file(shard_b):
            raise ValueError("Self-play shards differ for identical seeds.")

        reader = LnueShardReader(str(shard_a))
        _validate_lnue(reader)

        model = _train_one_step(reader, args.batch_size)
        nnue_path = tmp_path / "golden.nnue"
        _save_nnue(
            str(nnue_path),
            model,
            name="golden",
            author="golden",
            run_features_enabled=1,
            run_block_threshold=reader.header.run_block_threshold,
        )

        probe_out = tmp_path / "probe_logits.bin"
        subprocess.run(
            [
                str(probe_bin),
                "--shard",
                str(shard_a),
                "--nnue",
                str(nnue_path),
                "--out",
                str(probe_out),
                "--max_samples",
                str(args.max_samples),
            ],
            check=True,
        )

        nnue = _read_nnue(nnue_path)
        py_logits = []
        for feats in _iter_feature_slices(reader, args.max_samples):
            py_logits.append(_infer_raw_logits(nnue, feats))
        py_logits = np.array(py_logits, dtype=np.int32)
        cxx_logits = _read_probe_output(probe_out)

        if py_logits.shape != cxx_logits.shape:
            raise ValueError("Logit count mismatch between Python and C++.")
        if not np.array_equal(py_logits, cxx_logits):
            diff = np.abs(py_logits - cxx_logits)
            max_diff = int(diff.max()) if diff.size else 0
            raise ValueError(f"Logit mismatch (max diff {max_diff}).")

    print("Golden pipeline checks passed.")


if __name__ == "__main__":
    main()
