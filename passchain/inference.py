"""Block-diffusion inference + deterministic 2-stream FIFO PCIe benchmark.

Generation for a 64-token block:
  4 denoising forwards (phase 0 -> 3), unmasking 16 most-confident tokens per
  phase, then one phase-3 forward that finalises the block and appends its
  K/V to the clean-block KV cache.

Streaming expert store:
  the canonical key is the *pool row* (pool_id * pool_size + layer*stride +
  local_expert).  A 32-slot GPU ring buffer holds the 16 experts of the layer
  currently computing plus the next layer's new experts.  A dedicated
  non-blocking stream prefetches exactly the stride new rows; every transfer
  is padded to 3.1 MB from pinned host RAM.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import DiffConfig
from .data import DomainEvalDataset
from .model import DiffusionTransformer, apply_rope, _dispatch_padded


# ---------------------------------------------------------------------------
# model / prompt loading
# ---------------------------------------------------------------------------
def load_model(path: str, device: str = "cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = DiffConfig.from_dict(ckpt["model_config"])
    model = DiffusionTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt


def build_prompts(data_dir: str, prompt_len: int = 256, per_domain: int = 5) -> List[Dict]:
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    ds = DomainEvalDataset(data_dir, seq_len=prompt_len)
    out: List[Dict] = []
    for dom in ds.domains:
        arr = ds.arrays[dom]
        n = len(arr)
        step = max(prompt_len, (n - prompt_len) // max(per_domain, 1))
        for i in range(per_domain):
            off = min(i * step, n - prompt_len - 1)
            ids = torch.from_numpy(np.asarray(arr[off : off + prompt_len], dtype=np.int64))[None]
            out.append({"domain": dom, "ids": ids})
    return out


# ---------------------------------------------------------------------------
# weight providers
# ---------------------------------------------------------------------------
class ResidentProvider:
    """All experts for all phases/layers resident in VRAM (upper bound).

    For the old phase-pool layouts this is the same gather layout as the FIFO
    provider.  For the independent-layer A/B it also exposes ``weights_for_chunk``
    so the Unrestricted-40 sync-LRU path uses exactly the same chunked compute.
    """

    def __init__(self, model: DiffusionTransformer, cfg: DiffConfig):
        self.cfg = cfg
        self.model = model
        self.device = model.pool_wg.device
        self.wg_elems = cfg.d_model * 2 * cfg.d_ff_expert
        self.wd_elems = cfg.d_ff_expert * cfg.d_model
        self.used_elems = self.wg_elems + self.wd_elems
        n_rows = cfg.n_pools * cfg.pool_size
        self.n_rows = n_rows
        self.gpu = torch.zeros((n_rows, self.used_elems), dtype=torch.bfloat16, device=self.device)
        for p in range(cfg.n_pools):
            for off in range(cfg.pool_size):
                row = p * cfg.pool_size + off
                self.gpu[row, : self.wg_elems] = model.pool_wg[p, off].detach().to(torch.bfloat16).reshape(-1)
                self.gpu[row, self.wg_elems : self.used_elems] = model.pool_wd[p, off].detach().to(torch.bfloat16).reshape(-1)
        self.slot_idx = torch.empty(16, dtype=torch.long, device=self.device)
        self.host_bytes = 0
        self.gpu_bytes = self.gpu.numel() * self.gpu.element_size()

    def rows_for(self, mi: int, phase: int) -> List[int]:
        if self.cfg.spec.kind == "independent_layers":
            base = mi * self.cfg.experts_per_layer
            if self.cfg.router_mode == "full40":
                return list(range(base, base + self.cfg.experts_per_layer))
            start = self.cfg.window_starts[phase]
            return list(range(base + start, base + start + self.cfg.experts_per_phase))
        p = self.cfg.phase_to_pool[phase]
        base = p * self.cfg.pool_size + mi * self.cfg.stride
        return list(range(base, base + 16))

    def rows_for_full(self, mi: int) -> List[int]:
        base = mi * self.cfg.experts_per_layer
        return list(range(base, base + self.cfg.experts_per_layer))

    def ensure(self, mi: int, phase: int):
        return None

    def ensure_chunk(self, mi: int, experts: Sequence[int]):
        return None

    def prefetch(self, mi: int, phase: int, protect=frozenset()):
        return None

    def wait_layer(self, mi: int, phase: int):
        return None

    def after_layer(self, mi: int, phase: int, next_phase: int):
        return None

    def prepare_phase(self, phase: int):
        return None

    def weights_for(self, mi: int, phase: int):
        rows = self.rows_for(mi, phase)
        if self.cfg.spec.kind == "independent_layers":
            idx = torch.tensor(rows, dtype=torch.long, device=self.device)
            n = len(rows)
        else:
            self.slot_idx.copy_(torch.tensor(rows, dtype=torch.long))
            idx = self.slot_idx
            n = 16
        wg = self.gpu[idx, : self.wg_elems].view(n, self.cfg.d_model, 2 * self.cfg.d_ff_expert)
        wd = self.gpu[idx, self.wg_elems : self.used_elems].view(n, self.cfg.d_ff_expert, self.cfg.d_model)
        return wg, wd

    def weights_for_chunk(self, mi: int, experts: Sequence[int]):
        rows = [mi * self.cfg.experts_per_layer + int(e) for e in experts]
        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        n = len(rows)
        wg = self.gpu[idx, : self.wg_elems].view(n, self.cfg.d_model, 2 * self.cfg.d_ff_expert)
        wd = self.gpu[idx, self.wg_elems : self.used_elems].view(n, self.cfg.d_ff_expert, self.cfg.d_model)
        return wg, wd

    def sync(self):
        return None

    def reset_stats(self):
        return None

    def snapshot(self) -> Dict[str, float]:
        return {"hits": 0, "misses": 0, "bytes_moved": 0, "stall_s": 0.0,
                "fetch_calls": 0, "prefetch_calls": 0}

    def stats(self, tokens: int) -> Dict[str, float]:
        return {
            "hits": 0, "misses": 0, "hit_rate": 1.0,
            "stall_ms_per_token": 0.0, "bytes_moved_MB_per_token": 0.0,
            "fetch_calls": 0, "prefetch_calls": 0,
            "host_pool_GB": 0.0, "gpu_slots_GB": self.gpu_bytes / 1e9,
        }


class StreamingProvider:
    """Deterministic FIFO ring buffer with async next-layer prefetch."""

    def __init__(
        self,
        model: DiffusionTransformer,
        cfg: DiffConfig,
        resident: Optional[ResidentProvider] = None,
        pad_bytes: int = 3_100_000,
        capacity: int = 32,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device)
        self.pad_bytes = pad_bytes
        self.pad_elems = pad_bytes // 2
        self.wg_elems = cfg.d_model * 2 * cfg.d_ff_expert
        self.wd_elems = cfg.d_ff_expert * cfg.d_model
        self.used_elems = self.wg_elems + self.wd_elems
        assert self.used_elems <= self.pad_elems
        self.capacity = capacity

        n_rows = cfg.n_pools * cfg.pool_size
        self.n_rows = n_rows
        self.host = torch.zeros((n_rows, self.pad_elems), dtype=torch.bfloat16, pin_memory=True)
        for p in range(cfg.n_pools):
            for off in range(cfg.pool_size):
                row = p * cfg.pool_size + off
                self.host[row, : self.wg_elems] = (
                    model.pool_wg[p, off].detach().to(torch.bfloat16).reshape(-1)
                )
                self.host[row, self.wg_elems : self.used_elems] = (
                    model.pool_wd[p, off].detach().to(torch.bfloat16).reshape(-1)
                )
        self.host_bytes = self.host.numel() * self.host.element_size()

        self.gpu = torch.zeros((capacity, self.pad_elems), dtype=torch.bfloat16, device=self.device)
        self.slot_idx = torch.empty(16, dtype=torch.long, device=self.device)
        self.slot_key: List[int] = [-1] * capacity
        self.key_slot: Dict[int, int] = {}
        self.slot_events: Dict[int, torch.cuda.Event] = {}
        self.alloc_order: List[int] = []
        self.prefetch_stream = torch.cuda.Stream(priority=-1)

        self.hits = 0
        self.misses = 0
        self.bytes_moved = 0
        self.stall_s = 0.0
        self.fetch_calls = 0
        self.prefetch_calls = 0

    # ------------------------------------------------------------------
    def rows_for(self, mi: int, phase: int) -> List[int]:
        p = self.cfg.phase_to_pool[phase]
        base = p * self.cfg.pool_size + mi * self.cfg.stride
        return [base + i for i in range(16)]

    def _pending_keys(self) -> set:
        out = set()
        for slot, key in enumerate(self.slot_key):
            ev = self.slot_events.get(slot)
            if key >= 0 and ev is not None and not ev.query():
                out.add(key)
        return out

    def _alloc_slot(self, protect: set) -> int:
        for slot in range(self.capacity):
            if self.slot_key[slot] < 0:
                if slot not in self.alloc_order:
                    self.alloc_order.append(slot)
                return slot
        pending = self._pending_keys()
        for slot in list(self.alloc_order):
            key = self.slot_key[slot]
            if key in protect or key in pending:
                continue
            self.key_slot.pop(key, None)
            self.slot_events.pop(slot, None)
            self.slot_key[slot] = -1
            self.alloc_order.remove(slot)
            self.alloc_order.append(slot)
            return slot
        raise RuntimeError("ring buffer has no evictable slot")

    def _set_slot(self, slot: int, row: int, event: Optional[torch.cuda.Event]) -> None:
        if slot not in self.alloc_order:
            self.alloc_order.append(slot)
        self.slot_key[slot] = row
        self.key_slot[row] = slot
        if event is None:
            self.slot_events.pop(slot, None)
        else:
            self.slot_events[slot] = event

    def _wait_slot(self, slot: int) -> None:
        ev = self.slot_events.get(slot)
        if ev is not None and not ev.query():
            t0 = time.perf_counter()
            ev.synchronize()
            self.stall_s += time.perf_counter() - t0

    def ensure(self, mi: int, phase: int) -> List[int]:
        slots: List[int] = []
        rows = self.rows_for(mi, phase)
        protect = set(rows) | self._pending_keys()
        for row in rows:
            slot = self.key_slot.get(row)
            if slot is not None:
                self._wait_slot(slot)
                self.hits += 1
            else:
                self.misses += 1
                slot = self._alloc_slot(protect)
                t0 = time.perf_counter()
                self.gpu[slot].copy_(self.host[row], non_blocking=False)
                self.stall_s += time.perf_counter() - t0
                self.bytes_moved += self.pad_bytes
                self.fetch_calls += 1
                self._set_slot(slot, row, None)
            slots.append(slot)
            protect.add(row)
        return slots

    def prefetch(self, mi: int, phase: int, protect=frozenset()) -> None:
        rows = self.rows_for(mi, phase)
        protect = set(protect)
        pending = self._pending_keys()
        for row in rows:
            if row in self.key_slot:
                continue
            slot = self._alloc_slot(protect | pending)
            with torch.cuda.stream(self.prefetch_stream):
                self.gpu[slot].copy_(self.host[row], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.prefetch_stream)
            self.bytes_moved += self.pad_bytes
            self.prefetch_calls += 1
            self._set_slot(slot, row, ev)
            protect.add(row)
            pending.add(row)

    def weights_for(self, mi: int, phase: int):
        rows = self.rows_for(mi, phase)
        # fill a pre-allocated index tensor: no per-call GPU allocation
        self.slot_idx.copy_(torch.tensor([self.key_slot[r] for r in rows], dtype=torch.long))
        wg = self.gpu[self.slot_idx, : self.wg_elems].view(16, self.cfg.d_model, 2 * self.cfg.d_ff_expert)
        wd = self.gpu[self.slot_idx, self.wg_elems : self.used_elems].view(16, self.cfg.d_ff_expert, self.cfg.d_model)
        return wg, wd

    def sync(self):
        self.prefetch_stream.synchronize()

    def reset_stats(self):
        self.hits = self.misses = self.bytes_moved = 0
        self.stall_s = 0.0
        self.fetch_calls = self.prefetch_calls = 0

    def snapshot(self) -> Dict[str, float]:
        return {
            "hits": self.hits, "misses": self.misses,
            "bytes_moved": self.bytes_moved, "stall_s": self.stall_s,
            "fetch_calls": self.fetch_calls, "prefetch_calls": self.prefetch_calls,
        }

    def stats(self, tokens: int) -> Dict[str, float]:
        access = self.hits + self.misses
        return {
            "tokens": tokens,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(access, 1),
            "stall_ms_per_token": 1000.0 * self.stall_s / max(tokens, 1),
            "bytes_moved_MB_per_token": self.bytes_moved / 1e6 / max(tokens, 1),
            "fetch_calls": self.fetch_calls,
            "prefetch_calls": self.prefetch_calls,
            "host_pool_GB": self.host_bytes / 1e9,
            "gpu_slots_GB": self.gpu.numel() * self.gpu.element_size() / 1e9,
        }


# ---------------------------------------------------------------------------
# block-diffusion decode engine
# ---------------------------------------------------------------------------
class SyncLRUProvider:
    """Unrestricted-40 streaming: 16-slot synchronous LRU cache per layer.

    At each layer the engine asks for the union of experts selected by the
    64-token block.  If the union is larger than 16 it is processed in
    deterministic sorted chunks of at most 16; each chunk is fetched
    synchronously into the layer's 16 slots before its FFN math.
    """

    def __init__(self, model: DiffusionTransformer, cfg: DiffConfig,
                 pad_bytes: int = 3_100_000, capacity: int = 16, device: str = "cuda"):
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device)
        self.pad_bytes = pad_bytes
        self.pad_elems = pad_bytes // 2
        self.wg_elems = cfg.d_model * 2 * cfg.d_ff_expert
        self.wd_elems = cfg.d_ff_expert * cfg.d_model
        self.used_elems = self.wg_elems + self.wd_elems
        assert self.used_elems <= self.pad_elems
        self.capacity = capacity
        self.n_moe = cfg.n_moe_layers
        self.n_experts = cfg.experts_per_layer
        n_rows = self.n_moe * self.n_experts
        self.n_rows = n_rows
        self.host = torch.zeros((n_rows, self.pad_elems), dtype=torch.bfloat16, pin_memory=True)
        for mi in range(self.n_moe):
            for off in range(self.n_experts):
                row = mi * self.n_experts + off
                self.host[row, : self.wg_elems] = (
                    model.pool_wg[mi, off].detach().to(torch.bfloat16).reshape(-1)
                )
                self.host[row, self.wg_elems : self.used_elems] = (
                    model.pool_wd[mi, off].detach().to(torch.bfloat16).reshape(-1)
                )
        self.host_bytes = self.host.numel() * self.host.element_size()
        self.gpu = torch.zeros((self.n_moe, capacity, self.pad_elems), dtype=torch.bfloat16, device=self.device)
        self.gpu_bytes = self.gpu.numel() * self.gpu.element_size()
        self.slot_key: List[List[int]] = [[-1] * capacity for _ in range(self.n_moe)]
        self.key_slot: List[Dict[int, int]] = [dict() for _ in range(self.n_moe)]
        self.order: List[List[int]] = [list(range(capacity)) for _ in range(self.n_moe)]
        self.hits = 0
        self.misses = 0
        self.bytes_moved = 0
        self.stall_s = 0.0
        self.fetch_calls = 0
        self.prefetch_calls = 0

    def _row(self, mi: int, e: int) -> int:
        return mi * self.n_experts + e

    def ensure(self, mi: int, phase: int):
        return None

    def ensure_chunk(self, mi: int, experts: Sequence[int]) -> List[int]:
        experts = [int(e) for e in experts]
        assert len(experts) <= self.capacity
        slots: List[int] = []
        for e in experts:
            slot = self.key_slot[mi].get(e)
            if slot is not None:
                self.hits += 1
                self.order[mi].remove(slot)
                self.order[mi].append(slot)
            else:
                self.misses += 1
                free = next((sl for sl in self.order[mi] if self.slot_key[mi][sl] < 0), None)
                if free is not None:
                    slot = free
                    self.order[mi].remove(slot)
                else:
                    slot = self.order[mi].pop(0)
                    old = self.slot_key[mi][slot]
                    if old >= 0:
                        self.key_slot[mi].pop(old, None)
                t0 = time.perf_counter()
                self.gpu[mi, slot].copy_(self.host[self._row(mi, e)], non_blocking=False)
                self.stall_s += time.perf_counter() - t0
                self.bytes_moved += self.pad_bytes
                self.fetch_calls += 1
                self.slot_key[mi][slot] = e
                self.key_slot[mi][e] = slot
                self.order[mi].append(slot)
            slots.append(slot)
        return slots

    def weights_for_chunk(self, mi: int, experts: Sequence[int]):
        slots = self.ensure_chunk(mi, experts)
        idx = torch.tensor(slots, dtype=torch.long, device=self.device)
        n = len(slots)
        wg = self.gpu[mi, idx, : self.wg_elems].view(n, self.cfg.d_model, 2 * self.cfg.d_ff_expert)
        wd = self.gpu[mi, idx, self.wg_elems : self.used_elems].view(n, self.cfg.d_ff_expert, self.cfg.d_model)
        return wg, wd

    def prefetch(self, mi: int, phase: int, protect=frozenset()):
        return None

    def wait_layer(self, mi: int, phase: int):
        return None

    def after_layer(self, mi: int, phase: int, next_phase: int):
        return None

    def prepare_phase(self, phase: int):
        return None

    def sync(self):
        return None

    def reset_stats(self):
        self.hits = self.misses = self.bytes_moved = 0
        self.stall_s = 0.0
        self.fetch_calls = self.prefetch_calls = 0

    def snapshot(self) -> Dict[str, float]:
        return {
            "hits": self.hits, "misses": self.misses,
            "bytes_moved": self.bytes_moved, "stall_s": self.stall_s,
            "fetch_calls": self.fetch_calls, "prefetch_calls": self.prefetch_calls,
        }

    def stats(self, tokens: int) -> Dict[str, float]:
        access = self.hits + self.misses
        return {
            "tokens": tokens,
            "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / max(access, 1),
            "stall_ms_per_token": 1000.0 * self.stall_s / max(tokens, 1),
            "bytes_moved_MB_per_token": self.bytes_moved / 1e6 / max(tokens, 1),
            "fetch_calls": self.fetch_calls, "prefetch_calls": self.prefetch_calls,
            "host_pool_GB": self.host_bytes / 1e9,
            "gpu_slots_GB": self.gpu_bytes / 1e9,
        }


class VerticalSliceProvider:
    """PassChain-40 streaming: deterministic 16-slot vertical-slice swapper.

    A single non-blocking copy stream overwrites layer L's retiring experts
    with its next-pass experts as soon as layer L finishes the current pass.
    The compute stream waits on layer L's copy-done event only when it reaches
    that layer in the next pass.
    """

    def __init__(self, model: DiffusionTransformer, cfg: DiffConfig,
                 pad_bytes: int = 3_100_000, capacity: int = 16, device: str = "cuda"):
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device)
        self.pad_bytes = pad_bytes
        self.pad_elems = pad_bytes // 2
        self.wg_elems = cfg.d_model * 2 * cfg.d_ff_expert
        self.wd_elems = cfg.d_ff_expert * cfg.d_model
        self.used_elems = self.wg_elems + self.wd_elems
        assert self.used_elems <= self.pad_elems
        self.capacity = capacity
        self.n_moe = cfg.n_moe_layers
        self.n_experts = cfg.experts_per_layer
        n_rows = self.n_moe * self.n_experts
        self.n_rows = n_rows
        self.host = torch.zeros((n_rows, self.pad_elems), dtype=torch.bfloat16, pin_memory=True)
        for mi in range(self.n_moe):
            for off in range(self.n_experts):
                row = mi * self.n_experts + off
                self.host[row, : self.wg_elems] = (
                    model.pool_wg[mi, off].detach().to(torch.bfloat16).reshape(-1)
                )
                self.host[row, self.wg_elems : self.used_elems] = (
                    model.pool_wd[mi, off].detach().to(torch.bfloat16).reshape(-1)
                )
        self.host_bytes = self.host.numel() * self.host.element_size()
        self.gpu = torch.zeros((self.n_moe, capacity, self.pad_elems), dtype=torch.bfloat16, device=self.device)
        self.gpu_bytes = self.gpu.numel() * self.gpu.element_size()
        self.slot_key: List[List[int]] = [[-1] * capacity for _ in range(self.n_moe)]
        self.key_slot: List[Dict[int, int]] = [dict() for _ in range(self.n_moe)]
        self.events: List[Optional[torch.cuda.Event]] = [None] * self.n_moe
        self.copy_stream = torch.cuda.Stream(priority=-1)
        self.hits = 0
        self.misses = 0
        self.bytes_moved = 0
        self.stall_s = 0.0
        self.fetch_calls = 0
        self.prefetch_calls = 0
        self.prepare_phase(0)

    def _row(self, mi: int, e: int) -> int:
        return mi * self.n_experts + e

    def prepare_phase(self, phase: int) -> None:
        """Synchronously install the 16-expert window for ``phase`` (prefill/start)."""
        self.copy_stream.synchronize()
        start = self.cfg.window_starts[phase]
        for mi in range(self.n_moe):
            self.key_slot[mi].clear()
            for slot in range(self.capacity):
                e = start + slot
                self.gpu[mi, slot].copy_(self.host[self._row(mi, e)], non_blocking=False)
                self.slot_key[mi][slot] = e
                self.key_slot[mi][e] = slot
            self.events[mi] = None

    def wait_layer(self, mi: int, phase: int) -> None:
        ev = self.events[mi]
        if ev is not None and not ev.query():
            t0 = time.perf_counter()
            ev.synchronize()
            self.stall_s += time.perf_counter() - t0
        self.events[mi] = None

    def after_layer(self, mi: int, phase: int, next_phase: int) -> None:
        if next_phase == phase:
            return
        if phase == 0 and next_phase == 1:
            slots, new = range(0, 8), range(16, 24)
        elif phase == 1 and next_phase == 2:
            slots, new = range(8, 16), range(24, 32)
        elif phase == 2 and next_phase == 3:
            slots, new = range(0, 8), range(32, 40)
        elif phase == 3 and next_phase == 0:
            slots, new = range(0, 16), range(0, 16)
        else:
            return
        for slot, e in zip(slots, new):
            old = self.slot_key[mi][slot]
            if old >= 0:
                self.key_slot[mi].pop(old, None)
            self.slot_key[mi][slot] = e
            self.key_slot[mi][e] = slot
            with torch.cuda.stream(self.copy_stream):
                self.gpu[mi, slot].copy_(self.host[self._row(mi, e)], non_blocking=True)
            self.bytes_moved += self.pad_bytes
            self.prefetch_calls += 1
        ev = torch.cuda.Event()
        ev.record(self.copy_stream)
        self.events[mi] = ev

    def weights_for(self, mi: int, phase: int):
        start = self.cfg.window_starts[phase]
        slots = [self.key_slot[mi][start + i] for i in range(self.capacity)]
        if any(s < 0 for s in slots):
            raise RuntimeError(f"PassChain window not resident: layer={mi} phase={phase}")
        idx = torch.tensor(slots, dtype=torch.long, device=self.device)
        wg = self.gpu[mi, idx, : self.wg_elems].view(
            self.capacity, self.cfg.d_model, 2 * self.cfg.d_ff_expert
        )
        wd = self.gpu[mi, idx, self.wg_elems : self.used_elems].view(
            self.capacity, self.cfg.d_ff_expert, self.cfg.d_model
        )
        return wg, wd

    def ensure(self, mi: int, phase: int):
        return None

    def ensure_chunk(self, mi: int, experts: Sequence[int]):
        return None

    def prefetch(self, mi: int, phase: int, protect=frozenset()):
        return None

    def sync(self):
        self.copy_stream.synchronize()

    def reset_stats(self):
        self.hits = self.misses = self.bytes_moved = 0
        self.stall_s = 0.0
        self.fetch_calls = self.prefetch_calls = 0

    def snapshot(self) -> Dict[str, float]:
        return {
            "hits": self.hits, "misses": self.misses,
            "bytes_moved": self.bytes_moved, "stall_s": self.stall_s,
            "fetch_calls": self.fetch_calls, "prefetch_calls": self.prefetch_calls,
        }

    def stats(self, tokens: int) -> Dict[str, float]:
        access = self.hits + self.misses
        return {
            "tokens": tokens,
            "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / max(access, 1),
            "stall_ms_per_token": 1000.0 * self.stall_s / max(tokens, 1),
            "bytes_moved_MB_per_token": self.bytes_moved / 1e6 / max(tokens, 1),
            "fetch_calls": self.fetch_calls, "prefetch_calls": self.prefetch_calls,
            "host_pool_GB": self.host_bytes / 1e9,
            "gpu_slots_GB": self.gpu_bytes / 1e9,
        }


class BlockDiffusionEngine:
    def __init__(self, model: DiffusionTransformer, cfg: DiffConfig, provider, device="cuda"):
        self.model = model
        self.cfg = cfg
        self.provider = provider
        self.device = torch.device(device)
        self.n_layers = cfg.n_layers
        self.n_moe = cfg.n_moe_layers
        self.bs = cfg.block_size
        self.cache_k: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self.cache_v: List[Optional[torch.Tensor]] = [None] * self.n_layers

    def reset_cache(self):
        self.cache_k = [None] * self.n_layers
        self.cache_v = [None] * self.n_layers

    # ------------------------------------------------------------------
    def _attn(self, layer, x, pos_start, return_kv):
        B, L, _ = x.shape
        attn = layer.attn
        h = layer.norm1(x)
        q = attn.q_proj(h).view(B, L, attn.n_heads, attn.head_dim).transpose(1, 2)
        k, v = attn.kv_proj(h).view(B, L, 2, attn.n_kv, attn.head_dim).permute(2, 0, 3, 1, 4)
        if getattr(attn, "q_norm", None) is not None:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        pos = torch.arange(pos_start, pos_start + L, device=self.device)
        cos, sin = self.model.rope_positions(pos)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        li = self._layer_idx
        ck, cv = self.cache_k[li], self.cache_v[li]
        if ck is not None:
            k_all = torch.cat([ck, k], dim=2)
            v_all = torch.cat([cv, v], dim=2)
        else:
            k_all, v_all = k, v
        o = F.scaled_dot_product_attention(
            q, k_all, v_all, is_causal=False, enable_gqa=(attn.n_heads != attn.n_kv)
        )
        o = attn.o_proj(o.transpose(1, 2).reshape(B, L, attn.n_heads * attn.head_dim))
        return o, (k, v) if return_kv else None

    def _moe(self, layer, h, mi, phase):
        wg, wd = self.provider.weights_for(mi, phase)
        xf = h.reshape(-1, self.cfg.d_model).to(torch.bfloat16)
        z = xf.float() @ layer.ffn.router_w[phase].t()
        probs = torch.softmax(z, dim=-1)
        sel = z + layer.ffn.bias[phase]
        top_idx = torch.topk(sel, self.cfg.top_k, dim=-1).indices
        top_w = torch.gather(probs, -1, top_idx)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
        routed, _, _ = _dispatch_padded(xf, top_idx, top_w, wg, wd, cap_factor=4.0)
        routed = routed.view(1, -1, self.cfg.d_model).to(h.dtype)
        return routed + layer.ffn.shared(h)

    def _moe_unrestricted(self, layer, h, mi, phase):
        """Unrestricted-40: one full router, any pass can use any of 40 experts.

        All-resident and sync-LRU both use the same deterministic chunked
        dispatch; the only difference is whether ``weights_for_chunk`` triggers
        a synchronous PCIe copy.
        """
        xf = h.reshape(-1, self.cfg.d_model).to(torch.bfloat16)
        z = xf.float() @ layer.ffn.router_w_full
        probs = torch.softmax(z, dim=-1)
        sel = z + layer.ffn.bias_full
        top_idx = torch.topk(sel, self.cfg.top_k, dim=-1).indices
        top_w = torch.gather(probs, -1, top_idx)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
        E = self.cfg.experts_per_layer
        unique = torch.unique(top_idx.reshape(-1)).tolist()
        routed = torch.zeros_like(xf)
        for i in range(0, len(unique), 16):
            chunk = unique[i : i + 16]
            wg, wd = self.provider.weights_for_chunk(mi, chunk)
            chunk_t = torch.tensor(chunk, dtype=torch.long, device=xf.device)
            local = torch.zeros(E, dtype=torch.long, device=xf.device)
            local[chunk_t] = torch.arange(len(chunk), dtype=torch.long, device=xf.device)
            mask = torch.zeros(E, dtype=torch.bool, device=xf.device)
            mask[chunk_t] = True
            in_chunk = mask[top_idx]
            top_idx_local = local[top_idx]
            top_w_local = top_w * in_chunk.to(top_w.dtype)
            y, _, _ = _dispatch_padded(
                xf, top_idx_local, top_w_local, wg, wd, cap_factor=4.0
            )
            routed = routed + y
        routed = routed.view(1, -1, self.cfg.d_model).to(h.dtype)
        return routed + layer.ffn.shared(h)

    def _forward_block(self, ids, phase: int, pos_start: int, return_kv: bool, next_spec):
        """ids: (1, bs). next_spec = (mi, phase) of the next forward's first MoE."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = self.model.embed(ids)
            new_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
            independent = self.cfg.spec.kind == "independent_layers"
            full = self.cfg.router_mode == "full40"
            window = self.cfg.router_mode == "window40"
            if not independent:
                # start the first MoE layer's DMA while the dense layer 0 computes
                self.provider.prefetch(0, phase, protect=frozenset())
            mi = 0
            for li, layer in enumerate(self.model.layers):
                self._layer_idx = li
                if layer.is_moe and not independent:
                    self.provider.ensure(mi, phase)
                    if li + 1 < self.n_layers:
                        nxt = (mi + 1, phase)
                    else:
                        nxt = next_spec
                    if nxt is not None:
                        protect = set(self.provider.rows_for(mi, phase)) if hasattr(self.provider, "rows_for") else frozenset()
                        self.provider.prefetch(nxt[0], nxt[1], protect=protect)
                attn_out, kv = self._attn(layer, x, pos_start, return_kv)
                x = x + attn_out
                h = layer.norm2(x)
                if layer.is_moe:
                    if full:
                        x = x + self._moe_unrestricted(layer, h, mi, phase)
                    else:
                        if window:
                            self.provider.wait_layer(mi, phase)
                        x = x + self._moe(layer, h, mi, phase)
                    if window:
                        next_phase = next_spec[1] if next_spec is not None else phase
                        self.provider.after_layer(mi, phase, next_phase)
                    mi += 1
                else:
                    x = x + layer.ffn(h)
                if return_kv:
                    new_kv.append(kv)
            hidden = self.model.norm_f(x)
            logits = self.model.lm_head(hidden)
            return logits, new_kv

    def _append_cache(self, new_kv):
        for li, (k, v) in enumerate(new_kv):
            if self.cache_k[li] is None:
                self.cache_k[li] = k
                self.cache_v[li] = v
            else:
                self.cache_k[li] = torch.cat([self.cache_k[li], k], dim=2)
                self.cache_v[li] = torch.cat([self.cache_v[li], v], dim=2)

    def prefill(self, prompt_ids: torch.Tensor):
        assert prompt_ids.shape[1] % self.bs == 0
        n_blocks = prompt_ids.shape[1] // self.bs
        for i in range(n_blocks):
            block = prompt_ids[:, i * self.bs : (i + 1) * self.bs].to(self.device)
            # do not prefetch into generation from the last prompt block
            next_spec = (0, 3) if i + 1 < n_blocks else None
            _, kv = self._forward_block(block, phase=3, pos_start=i * self.bs, return_kv=True, next_spec=next_spec)
            self._append_cache(kv)

    @torch.no_grad()
    def generate_blocks(self, n_blocks: int) -> torch.Tensor:
        prompt_blocks = (self.cache_k[0].shape[2] if self.cache_k[0] is not None else 0) // self.bs
        out = []
        for b in range(n_blocks):
            block_idx = prompt_blocks + b
            pos_start = block_idx * self.bs
            xt = torch.full((1, self.bs), self.cfg.mask_token_id, dtype=torch.long, device=self.device)
            known = torch.zeros(self.bs, dtype=torch.bool, device=self.device)
            for s in range(4):
                next_spec = (0, s + 1) if s < 3 else (0, 3)
                logits, _ = self._forward_block(xt, phase=s, pos_start=pos_start, return_kv=False, next_spec=next_spec)
                probs = torch.softmax(logits.float(), dim=-1)
                conf = probs.max(dim=-1).values[0]
                pred = logits.argmax(dim=-1)[0]
                target_known = int(round(self.bs * (s + 1) / 4))
                n_to = max(0, target_known - int(known.sum()))
                if n_to > 0:
                    scores = conf.masked_fill(known, -float("inf"))
                    order = torch.argsort(scores, descending=True, stable=True)
                    sel = order[:n_to]
                    xt[0, sel] = pred[sel]
                    known[sel] = True
            if not bool(known.all()):
                scores = conf.masked_fill(known, -float("inf"))
                order = torch.argsort(scores, descending=True, stable=True)
                sel = order[: int((~known).sum())]
                xt[0, sel] = pred[sel]
                known[sel] = True
            # finalise the now-clean block and append its K/V
            final_next = (0, 0) if b + 1 < n_blocks else None
            _, kv = self._forward_block(xt, phase=3, pos_start=pos_start, return_kv=True, next_spec=final_next)
            self._append_cache(kv)
            out.append(xt[0].clone())
        return torch.cat(out, dim=0) if out else torch.empty(0, dtype=torch.long, device=self.device)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_system(name, model, cfg, provider, prompts, n_blocks, device="cuda"):
    engine = BlockDiffusionEngine(model, cfg, provider, device)
    total_tokens = 0
    total_s = 0.0
    all_tokens: List[int] = []
    per_prompt = []
    acc = {"hits": 0, "misses": 0, "bytes_moved": 0, "stall_s": 0.0,
           "fetch_calls": 0, "prefetch_calls": 0}
    for p in prompts:
        engine.reset_cache()
        # Prefill uses the clean-block path (phase 3).  PassChain must install
        # the phase-3 window first, then reset to phase 0 for generation.
        if hasattr(provider, "prepare_phase"):
            provider.prepare_phase(3)
        engine.prefill(p["ids"].to(device))
        if hasattr(provider, "prepare_phase"):
            provider.prepare_phase(0)
        provider.sync()
        provider.reset_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks = engine.generate_blocks(n_blocks)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        snap = provider.snapshot()
        for k in acc:
            acc[k] += snap[k]
        total_s += dt
        total_tokens += toks.numel()
        all_tokens.extend(toks.tolist())
        per_prompt.append(dt)
    access = acc["hits"] + acc["misses"]
    base_stats = provider.stats(total_tokens)
    stats = {
        "tokens": total_tokens,
        "hits": acc["hits"],
        "misses": acc["misses"],
        "hit_rate": acc["hits"] / max(access, 1),
        "stall_ms_per_token": 1000.0 * acc["stall_s"] / max(total_tokens, 1),
        "bytes_moved_MB_per_token": acc["bytes_moved"] / 1e6 / max(total_tokens, 1),
        "fetch_calls": acc["fetch_calls"],
        "prefetch_calls": acc["prefetch_calls"],
        "host_pool_GB": base_stats.get("host_pool_GB", 0.0),
        "gpu_slots_GB": base_stats.get("gpu_slots_GB", 0.0),
    }
    rec = {
        "system": name,
        "layout": cfg.layout,
        "tokens": total_tokens,
        "tok_per_s": total_tokens / max(total_s, 1e-9),
        "ms_per_token": 1000.0 * total_s / max(total_tokens, 1),
        "stall_ms_per_token": stats["stall_ms_per_token"],
        "dma_MB_per_token": stats["bytes_moved_MB_per_token"],
        "hit_rate": stats["hit_rate"],
        "fetch_calls": stats["fetch_calls"],
        "prefetch_calls": stats["prefetch_calls"],
        "host_pool_GB": stats["host_pool_GB"],
        "gpu_slots_GB": stats["gpu_slots_GB"],
        "tokens_out": all_tokens,
        "per_prompt_ms": [1000 * x for x in per_prompt],
    }
    print(
        f"  {name:28s} {rec['tok_per_s']:7.2f} tok/s | "
        f"stall {rec['stall_ms_per_token']:7.3f} ms/tok | "
        f"DMA {rec['dma_MB_per_token']:7.2f} MB/tok | hit {100*rec['hit_rate']:5.1f}%",
        flush=True,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/shards_150m")
    ap.add_argument("--results_dir", default=os.path.join("diffusion", "results"))
    ap.add_argument("--prompts_per_domain", type=int, default=5)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--generate_blocks", type=int, default=8)
    ap.add_argument("--layouts", default="d1,d1_chain,d2,d3a,d3b,d3c")
    ap.add_argument("--out", default=os.path.join("diffusion", "results", "inference_results.json"))
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    prompts = build_prompts(args.data_dir, args.prompt_len, args.prompts_per_domain)
    print(f"[bench] {len(prompts)} prompts x ({args.prompt_len} prompt + {args.generate_blocks*64} generated)", flush=True)
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    results = []
    for layout in layouts:
        ckpt_path = os.path.join(args.results_dir, layout, "ckpt_final.pt")
        if not os.path.exists(ckpt_path):
            print(f"[bench] missing {ckpt_path}, skipping", flush=True)
            continue
        model, cfg, ckpt = load_model(ckpt_path)
        print(
            f"\n[bench] {layout}: routed={cfg.n_pools*cfg.pool_size} stride={cfg.stride} "
            f"val_ppl={ckpt.get('val',{}).get('overall_ppl')}", flush=True,
        )
        resident = ResidentProvider(model, cfg)
        if cfg.router_mode == "full40":
            stream = SyncLRUProvider(model, cfg)
            stream_name = "sync LRU"
        elif cfg.router_mode == "window40":
            stream = VerticalSliceProvider(model, cfg)
            stream_name = "vertical-slice"
        else:
            stream = StreamingProvider(model, cfg, resident)
            stream_name = "FIFO ring"
        rec_res = run_system(f"{layout} all-resident", model, cfg, resident, prompts, args.generate_blocks)
        rec_str = run_system(f"{layout} {stream_name}", model, cfg, stream, prompts, args.generate_blocks)
        identical = rec_res["tokens_out"] == rec_str["tokens_out"]
        rec_res.pop("tokens_out"); rec_str.pop("tokens_out")
        rec_res["pct_of_resident"] = 100.0
        rec_str["pct_of_resident"] = 100.0 * rec_str["tok_per_s"] / max(rec_res["tok_per_s"], 1e-9)
        rec_res["bit_identical"] = True
        rec_str["bit_identical"] = identical
        rec_res["pad_bytes"] = 0
        rec_str["pad_bytes"] = stream.pad_bytes
        results.append({"resident": rec_res, "stream": rec_str, "bit_identical": identical})
        print(f"  bit-identical resident vs FIFO: {identical}", flush=True)
        del model, resident, stream
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
