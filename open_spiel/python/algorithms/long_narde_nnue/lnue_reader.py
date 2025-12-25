"""LNUE shard reader for Long Narde NNUE training."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterator

import numpy as np

_LNUE_MAGIC = b"LNUE"
_CHUNK_MAGIC = b"CHNK"
_HEADER_FMT = "<4s6IQ2I5I"
_CHUNK_FMT = "<4s6I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_CHUNK_SIZE = struct.calcsize(_CHUNK_FMT)
_ENDIAN_MARKER = 0x01020304


@dataclass(frozen=True)
class LnueHeader:
    """LNUE file header."""
    # pylint: disable=too-many-instance-attributes

    version: int
    schema_id: int
    feature_dim: int
    flags: int
    feature_index_bytes: int
    seed: int
    shard_id: int
    run_block_threshold: int


@dataclass(frozen=True)
class LnueChunk:
    """Chunk payload with columnar arrays."""

    offsets: np.ndarray
    indices: np.ndarray
    v_search: np.ndarray
    outcome: np.ndarray
    target: np.ndarray
    game_id: np.ndarray
    ply: np.ndarray


class LnueShardReader:
    """Reader for LNUE shard files."""

    def __init__(self, path: str):
        self._path = path
        self._header = None

    @property
    def header(self) -> LnueHeader:
        """Returns the LNUE header (cached)."""
        if self._header is None:
            with open(self._path, "rb") as handle:
                self._header = self._read_header(handle)
        return self._header

    def iter_chunks(self) -> Iterator[LnueChunk]:
        """Yields chunks in order from the LNUE shard."""
        # pylint: disable=too-many-locals
        with open(self._path, "rb") as handle:
            header = self._read_header(handle)
            while True:
                raw = handle.read(_CHUNK_SIZE)
                if not raw:
                    break
                if len(raw) != _CHUNK_SIZE:
                    raise ValueError("Incomplete LNUE chunk header.")
                magic, unc_bytes, comp_bytes, num_samples, _, _, _ = struct.unpack(
                    _CHUNK_FMT, raw
                )
                if magic != _CHUNK_MAGIC:
                    raise ValueError("Invalid LNUE chunk magic.")
                if comp_bytes != 0:
                    raise ValueError("Compressed chunks are not supported.")

                offsets = _read_array(handle, np.uint32, num_samples + 1)
                indices_len = int(offsets[-1])
                if header.feature_index_bytes == 2:
                    indices = _read_array(handle, np.uint16, indices_len)
                elif header.feature_index_bytes == 4:
                    indices = _read_array(handle, np.uint32, indices_len)
                else:
                    raise ValueError("Unsupported feature index width.")

                v_search = _read_array(handle, np.float32, num_samples)
                outcome = _read_array(handle, np.int8, num_samples)
                target = _read_array(handle, np.float32, num_samples)
                game_id = _read_array(handle, np.uint64, num_samples)
                ply = _read_array(handle, np.uint16, num_samples)

                read_bytes = (
                    offsets.nbytes
                    + indices.nbytes
                    + v_search.nbytes
                    + outcome.nbytes
                    + target.nbytes
                    + game_id.nbytes
                    + ply.nbytes
                )
                if read_bytes != unc_bytes:
                    raise ValueError("LNUE chunk size mismatch.")

                yield LnueChunk(
                    offsets=offsets,
                    indices=indices,
                    v_search=v_search,
                    outcome=outcome,
                    target=target,
                    game_id=game_id,
                    ply=ply,
                )

    def _read_header(self, handle) -> LnueHeader:
        """Reads and validates the LNUE header."""
        raw = handle.read(_HEADER_SIZE)
        if len(raw) != _HEADER_SIZE:
            raise ValueError("Incomplete LNUE header.")
        (
            magic,
            version,
            endian,
            schema_id,
            feature_dim,
            flags,
            index_bytes,
            seed,
            shard_id,
            run_block_threshold,
            *_,
        ) = struct.unpack(_HEADER_FMT, raw)
        if magic != _LNUE_MAGIC:
            raise ValueError("Invalid LNUE header magic.")
        if version not in (1, 2):
            raise ValueError("Unsupported LNUE version.")
        if endian != _ENDIAN_MARKER:
            raise ValueError("Unsupported endianness.")
        return LnueHeader(
            version=version,
            schema_id=schema_id,
            feature_dim=feature_dim,
            flags=flags,
            feature_index_bytes=index_bytes,
            seed=seed,
            shard_id=shard_id,
            run_block_threshold=run_block_threshold,
        )


def _read_array(handle, dtype: np.dtype, count: int) -> np.ndarray:
    """Reads a typed array from the current file position."""
    byte_count = np.dtype(dtype).itemsize * count
    data = handle.read(byte_count)
    if len(data) != byte_count:
        raise ValueError("Unexpected EOF while reading LNUE payload.")
    return np.frombuffer(data, dtype=dtype).copy()
