"""Noise-phase sampling, block masks and the leak-free BD3-LM batch builder."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .config import DiffConfig, PHASE_RANGES


def _balanced_phases(rng: np.random.Generator, n_blocks_total: int, n_phases: int = 4) -> np.ndarray:
    """Stratified uniform phase assignment with (nearly) equal counts.

    Balancing keeps every compiled shape static: with B*8 blocks split as
    evenly as possible, the number of tokens in each phase is fixed, so the
    per-phase expert dispatch avoids data-dependent dynamic shapes.
    """
    base = n_blocks_total // n_phases
    phases = np.repeat(np.arange(n_phases, dtype=np.int64), base)
    rem = n_blocks_total - phases.size
    if rem:
        phases = np.concatenate([phases, rng.integers(0, n_phases, size=rem, dtype=np.int64)])
    rng.shuffle(phases)
    return phases


def sample_block_noise(
    rng: np.random.Generator,
    n_blocks: int,
    block_size: int,
    n_phases: int = 4,
    force_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (phases (n_blocks,), t (n_blocks,), mask (n_blocks, block_size))."""
    phases = _balanced_phases(rng, n_blocks, n_phases)
    t = np.empty(n_blocks, dtype=np.float64)
    for s in range(n_phases):
        lo, hi = PHASE_RANGES[s]
        sel = phases == s
        t[sel] = rng.uniform(lo, hi, size=int(sel.sum()))
    mask = rng.random((n_blocks, block_size)) < t[:, None]
    if force_one:
        empty = ~mask.any(axis=1)
        if empty.any():
            pos = rng.integers(0, block_size, size=int(empty.sum()))
            mask[np.nonzero(empty)[0], pos] = True
    return phases, t, mask


def build_sequence_batch(
    x0: np.ndarray,
    rng: np.random.Generator,
    cfg: DiffConfig,
    force_one: bool = True,
) -> Dict[str, np.ndarray]:
    """Build the x_0 || x_t training tensors for a micro-batch.

    x0: (B, seq_len) int64 clean tokens.
    Returns ids (B, 2*seq_len), phase_ids (B, 2*seq_len), targets (B, seq_len),
    mask (B, seq_len), t_tok (B, seq_len) plus the sorted-phase permutation
    (order, inv_order) and fixed phase sizes.
    """
    B, seq_len = x0.shape
    bs = cfg.block_size
    n_blocks = seq_len // bs
    phases, t, mask = sample_block_noise(rng, B * n_blocks, bs, cfg.n_phases, force_one)
    phases = phases.reshape(B, n_blocks)
    t = t.reshape(B, n_blocks)
    mask = mask.reshape(B, n_blocks, bs)

    noisy_phases = np.repeat(phases, bs, axis=1)                    # (B, seq_len)
    phase_ids = np.concatenate(
        [np.full((B, seq_len), 3, dtype=np.int64), noisy_phases], axis=1
    )
    xt = np.where(mask.reshape(B, seq_len), cfg.mask_token_id, x0)
    ids = np.concatenate([x0, xt], axis=1).astype(np.int64)
    t_tok = np.repeat(t, bs, axis=1).astype(np.float32)
    mask_flat = mask.reshape(B, seq_len)

    flat_phase = phase_ids.reshape(-1)
    order = np.argsort(flat_phase, kind="stable").astype(np.int64)
    inv_order = np.argsort(order, kind="stable").astype(np.int64)
    phase_sizes = tuple(int(np.count_nonzero(flat_phase == p)) for p in range(cfg.n_phases))

    return {
        "ids": ids,
        "phase_ids": phase_ids.astype(np.int64),
        "order": order,
        "inv_order": inv_order,
        "phase_sizes": phase_sizes,
        "targets": x0.astype(np.int64),
        "mask": mask_flat,
        "t_tok": t_tok,
        "phases": phases,
        "t": t,
    }


def batch_to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k in ("ids", "phase_ids", "order", "inv_order", "targets"):
        out[k] = torch.from_numpy(batch[k]).to(device=device, dtype=torch.long, non_blocking=True)
    out["mask"] = torch.from_numpy(batch["mask"].astype(np.bool_)).to(device=device, non_blocking=True)
    out["t_tok"] = torch.from_numpy(batch["t_tok"].astype(np.float32)).to(device=device, non_blocking=True)
    out["phase_sizes"] = batch["phase_sizes"]
    return out


def block_diffusion_mask_mod(cfg: DiffConfig):
    """The exact token-level connectivity rule (True = may attend)."""
    bs = cfg.block_size
    half = cfg.seq_len

    def mask_mod(b, h, qi, ki):
        qb = qi // bs
        kb = ki // bs
        qc = qi < half
        kc = ki < half
        bn = qb - cfg.n_blocks
        kn = kb - cfg.n_blocks
        clean = kc & (kb <= qb)
        noisy = (kc & (kb < bn)) | ((~kc) & (kn == bn))
        return torch.where(qc, clean, noisy)

    return mask_mod


def block_diffusion_block_mask(cfg: DiffConfig, device: torch.device):
    """FlexAttention block mask implementing the leak-free x0||xt attention rule.

    Clean block b   : keys = clean blocks 0..b        (bidirectional within b)
    Noisy block b   : keys = clean blocks 0..b-1      (never clean block b)
                        + own noisy block b           (bidirectional within b)

    ``separate_full_blocks=False`` is essential: the mask is not monotone at
    the 128-token kernel block granularity (our blocks are 64 tokens), so every
    partial kernel block must evaluate mask_mod elementwise for exactness.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    mask_mod = block_diffusion_mask_mod(cfg)
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=cfg.context_len,
        KV_LEN=cfg.context_len,
        BLOCK_SIZE=128,
        device=device,
        separate_full_blocks=False,
    )
