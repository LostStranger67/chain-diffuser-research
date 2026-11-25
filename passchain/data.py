"""Deterministic memory-mapped shard loader for block diffusion.

Windows are non-overlapping 512-token slices.  When a run needs slightly more
windows than the 150M-token pool contains (1145 * 256 = 293,120 vs 292,968),
the deterministic permutation wraps around, reusing the first 152 windows.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch


@dataclass
class ShardMeta:
    train_files: List[str]
    val_files: Dict[str, str]
    train_tokens: int
    vocab_size: int
    eos_id: int
    tokenizer: str


def load_meta(data_dir: str) -> ShardMeta:
    path = os.path.join(data_dir, "meta.json")
    with open(path) as f:
        m = json.load(f)
    return ShardMeta(
        train_files=m["train_files"],
        val_files=m["val_files"],
        train_tokens=m["train_tokens"],
        vocab_size=m["vocab_size"],
        eos_id=m["eos_id"],
        tokenizer=m["tokenizer"],
    )


class ShardDataset:
    """Train windows, in a fixed permutation of the global window index."""

    def __init__(self, data_dir: str, seq_len: int = 512, seed: int = 42):
        self.meta = load_meta(data_dir)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.maps: List[np.memmap] = []
        shard_of: List[np.ndarray] = []
        off_of: List[np.ndarray] = []
        for si, name in enumerate(self.meta.train_files):
            path = os.path.join(data_dir, name)
            n = os.path.getsize(path) // 2
            self.maps.append(np.memmap(path, dtype=np.uint16, mode="r"))
            n_win = n // seq_len
            shard_of.append(np.full(n_win, si, dtype=np.int16))
            off_of.append(np.arange(n_win, dtype=np.int64) * seq_len)
        self.win_shard = np.concatenate(shard_of) if shard_of else np.empty(0, np.int16)
        self.win_off = np.concatenate(off_of) if off_of else np.empty(0, np.int64)
        self.total_windows = int(self.win_shard.size)
        self.perm = np.random.default_rng(seed).permutation(self.total_windows)

    def __len__(self) -> int:
        return self.total_windows

    def get_batch(self, start: int, batch_size: int, wrap: bool = True) -> np.ndarray:
        idx = np.arange(start, start + batch_size, dtype=np.int64)
        if wrap:
            idx %= self.total_windows
        else:
            if int(idx[-1]) >= self.total_windows:
                raise IndexError("window index past end")
        g = self.perm[idx]
        out = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for r, gi in enumerate(g.tolist()):
            shard = int(self.win_shard[gi])
            off = int(self.win_off[gi])
            out[r] = self.maps[shard][off : off + self.seq_len]
        return out

    def fingerprint(self, batch_size: int = 4, n: int = 3) -> str:
        h = hashlib.sha256()
        for s in range(0, batch_size * n, batch_size):
            h.update(self.get_batch(s, batch_size, wrap=False).tobytes())
        return h.hexdigest()


class DomainEvalDataset:
    """Held-out windows per domain, non-overlapping and deterministic."""

    def __init__(self, data_dir: str, seq_len: int = 512):
        self.meta = load_meta(data_dir)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.domains = list(self.meta.val_files.keys())
        self.arrays = {
            d: np.memmap(os.path.join(data_dir, self.meta.val_files[d]), dtype=np.uint16, mode="r")
            for d in self.domains
        }
        self.windows = {d: int(len(a) // seq_len) for d, a in self.arrays.items()}

    def get_batch(self, domain: str, start: int, batch_size: int) -> np.ndarray:
        arr = self.arrays[domain]
        w = self.windows[domain]
        if start + batch_size > w:
            raise IndexError(f"{domain}: {start}+{batch_size} > {w}")
        off = start * self.seq_len
        chunk = np.asarray(arr[off : off + batch_size * self.seq_len], dtype=np.int64)
        return chunk.reshape(batch_size, self.seq_len).copy()


class SyntheticTokenDataset:
    """Deterministic random-token dataset for debug/smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 42, length: int = 4096):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed
        self.length = length

    def __len__(self) -> int:
        return self.length

    def get_batch(self, start: int, batch_size: int, wrap: bool = True) -> np.ndarray:
        idx = np.arange(start, start + batch_size, dtype=np.int64)
        if wrap:
            idx %= self.length
        # A tiny repeatable token grammar makes the 20-step local smoke test a
        # meaningful "loss goes down" check (real runs use PackedTokenDataset).
        period = 64
        base = (np.arange(self.seq_len, dtype=np.int64) * 7 + 3) % period + 1
        out = np.broadcast_to(base[None, :], (batch_size, self.seq_len)).copy()
        # Offset each window by a stable row phase so the model sees variation.
        out = (out + (idx[:, None] % 7)) % min(self.vocab_size, 512)
        return out.astype(np.int64)


class PackedTokenDataset:
    """Sequence-packed uint16 shards with no padding.

    Shards are concatenated in metadata order and read in fixed ``seq_len``
    windows.  The tokenizer/preparation pipeline is expected to place the
    ``<|endoftext|>`` document-boundary token between documents; packing then
    consumes those boundaries without adding pad tokens.  Windows never pad:
    the dataset length is ``total_tokens // seq_len``.
    """

    def __init__(self, data_dir: str, seq_len: int = 2048, seed: int = 42):
        self.meta = load_meta(data_dir)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.maps: List[np.memmap] = []
        self.shard_tokens: List[int] = []
        for name in self.meta.train_files:
            arr = np.memmap(os.path.join(data_dir, name), dtype=np.uint16, mode="r")
            self.maps.append(arr)
            self.shard_tokens.append(int(len(arr)))
        self.starts = np.concatenate([[0], np.cumsum(self.shard_tokens)]).astype(np.int64)
        self.total_tokens = int(self.starts[-1])
        self.n_windows = self.total_tokens // seq_len
        self.perm = np.random.default_rng(seed).permutation(self.n_windows)

    def __len__(self) -> int:
        return self.n_windows

    def _window(self, global_start: int) -> np.ndarray:
        out = np.empty(self.seq_len, dtype=np.int64)
        filled = 0
        pos = global_start
        while filled < self.seq_len:
            si = int(np.searchsorted(self.starts, pos, side="right") - 1)
            off = pos - int(self.starts[si])
            take = min(self.seq_len - filled, self.shard_tokens[si] - off)
            out[filled : filled + take] = self.maps[si][off : off + take]
            filled += take
            pos += take
        return out

    def get_batch(self, start: int, batch_size: int, wrap: bool = True) -> np.ndarray:
        idx = np.arange(start, start + batch_size, dtype=np.int64)
        if wrap:
            idx %= self.n_windows
        elif int(idx[-1]) >= self.n_windows:
            raise IndexError("window index past end")
        windows = self.perm[idx]
        out = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for r, w in enumerate(windows.tolist()):
            out[r] = self._window(int(w) * self.seq_len)
        return out
