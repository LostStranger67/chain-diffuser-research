"""Block-diffusion MoE backbone: GQA + RoPE + phase-partitioned expert pools.

The only thing that changes between D1/D2/D3a/D3b/D3c is the expert-pool
layout.  Every MoE layer owns four lightweight routers (one per phase) and a
``bias[4,16]`` selection-only vector updated with the aux-loss-free rule.
Expert parameters live in one (or four) contiguous pool(s); a layer's 16
experts are a *slice* of that pool, so overlapping layers/phases share the
exact same parameter tensor and autograd accumulates into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from .config import DiffConfig


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def build_rope(dim: int, max_seq_len: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B,H,T,hd); cos/sin: (T, hd/2), already gathered for this block."""
    T = x.shape[-2]
    c, s = cos[:T][None, None], sin[:T][None, None]
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Attention(nn.Module):
    def __init__(self, cfg: DiffConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv_proj = nn.Linear(d, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin, block_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).view(B, T, 2, self.n_kv, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if block_mask is not None:
            o = flex_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                block_mask=block_mask, enable_gqa=(self.n_heads != self.n_kv),
            )
        else:
            o = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=(self.n_heads != self.n_kv)
            )
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(o)


# ---------------------------------------------------------------------------
# MoE with phase-specific expert pool slices
# ---------------------------------------------------------------------------
def _histogram(flat: torch.Tensor, n: int) -> torch.Tensor:
    """Fixed-shape counts (torch.compile-friendly; bincount breaks dynamo)."""
    counts = torch.zeros(n, dtype=torch.int64, device=flat.device)
    return counts.scatter_add_(0, flat, torch.ones_like(flat))


def _dispatch_padded(xf, top_idx, top_w, wg, wd, cap_factor=1.25):
    """Capacity-padded batched GEMM over 16 experts (dropless in practice).

    xf: (n,D); top_idx/top_w: (n,k); wg: (16,D,2F); wd: (16,F,D).
    Returns (routed (n,D), counts (16,), drop_frac scalar).
    """
    n = xf.shape[0]
    k = top_idx.shape[1]
    E = wg.shape[0]
    flat = top_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = _histogram(flat, E)
    tok = (order // k).to(torch.long)
    e_sorted = flat.index_select(0, order)
    mean = float(n * k) / E
    cap = int(math.ceil(mean * cap_factor)) + 8
    offs = counts.cumsum(0) - counts
    pos = torch.arange(n * k, device=xf.device) - offs[e_sorted]
    keep = pos < cap
    e_sorted, pos, tok = e_sorted[keep], pos[keep], tok[keep]
    w_sorted = top_w.reshape(-1).index_select(0, order)[keep]
    xe = torch.zeros(E, cap, xf.shape[1], device=xf.device, dtype=xf.dtype).index_put(
        (e_sorted, pos), xf.index_select(0, tok)
    )
    gu = torch.bmm(xe, wg)
    g, u = gu.chunk(2, dim=-1)
    oe = torch.bmm(F.silu(g) * u, wd)
    out = oe[e_sorted, pos] * w_sorted.to(oe.dtype).unsqueeze(-1)
    routed = torch.zeros(xf.shape[0], xf.shape[1], device=xf.device, dtype=out.dtype).index_add(0, tok, out)
    drop_frac = (keep.numel() - keep.sum()).float() / max(keep.numel(), 1)
    return routed, counts, drop_frac


def _dispatch_grouped(xf, top_idx, top_w, wg, wd):
    """Dropless sorted dispatch via torch._grouped_mm (16 experts per phase).

    Unlike capacity-padded bmm this cannot drop assignments when the router is
    initially unbalanced.  With only 16 groups the backward is acceptably fast
    (measured ~2.1 ms for n=2048 / ~6.1 ms for n=10240 on the RTX 3090).
    """
    n = xf.shape[0]
    k = top_idx.shape[1]
    flat = top_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = _histogram(flat, wg.shape[0])
    offs = counts.cumsum(0).to(torch.int32)
    tok = (order // k).to(torch.long)
    xe = xf.index_select(0, tok)
    gu = torch._grouped_mm(xe, wg, offs=offs)
    g, u = gu.chunk(2, dim=-1)
    oe = torch._grouped_mm(F.silu(g) * u, wd, offs=offs)
    w_sorted = top_w.reshape(-1).index_select(0, order).to(oe.dtype).unsqueeze(-1)
    routed = torch.zeros(n, xf.shape[1], device=xf.device, dtype=oe.dtype).index_add(
        0, tok, oe * w_sorted
    )
    return routed, counts, xf.new_zeros(())


class MoEFFN(nn.Module):
    def __init__(self, cfg: DiffConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.top_k = cfg.top_k
        self.n_phases = cfg.n_phases
        self.n_experts = cfg.experts_per_phase
        self.router_mode = cfg.router_mode
        if self.router_mode == "full40":
            # One standard router per layer; any pass (and clean x0) may use
            # any of the layer's 40 experts.
            self.n_experts_full = cfg.experts_per_layer
            self.router_w_full = nn.Parameter(torch.empty(cfg.d_model, self.n_experts_full))
            nn.init.normal_(self.router_w_full, std=0.02)
            self.register_buffer(
                "bias_full", torch.zeros(self.n_experts_full, dtype=torch.float32)
            )
        else:
            # Window/phase routers: Linear(384, 16) per pass.
            self.router_w = nn.Parameter(torch.empty(cfg.n_phases, self.n_experts, cfg.d_model))
            nn.init.normal_(self.router_w, std=0.02)
            self.register_buffer("bias", torch.zeros(cfg.n_phases, self.n_experts, dtype=torch.float32))
        self.shared = SwiGLU(cfg.d_model, cfg.d_ff_expert)

    def expert_slices(self, pool_wg, pool_wd, phase: int):
        if self.cfg.spec.kind == "independent_layers":
            p = self.layer_idx
            if self.router_mode == "full40":
                return pool_wg[p], pool_wd[p]
            start = self.cfg.window_starts[phase]
            return (
                pool_wg[p, start : start + self.cfg.experts_per_phase],
                pool_wd[p, start : start + self.cfg.experts_per_phase],
            )
        p = self.cfg.phase_to_pool[phase]
        s, e = self.cfg.pool_span(self.layer_idx)
        return pool_wg[p, s:e], pool_wd[p, s:e]

    def _dispatch(self, h, top_idx, top_w, wg, wd):
        if self.cfg.moe_impl == "padded":
            return _dispatch_padded(h, top_idx, top_w, wg.to(h.dtype), wd.to(h.dtype), cap_factor=10.0)
        if self.cfg.moe_impl in ("auto", "grouped") and hasattr(torch, "_grouped_mm") and h.is_cuda:
            hb = h.to(torch.bfloat16)
            y, counts, drop = _dispatch_grouped(
                hb, top_idx, top_w, wg.to(torch.bfloat16), wd.to(torch.bfloat16)
            )
            return y.to(h.dtype), counts, drop
        return _dispatch_padded(h, top_idx, top_w, wg.to(h.dtype), wd.to(h.dtype), cap_factor=10.0)

    def _forward_full(self, x, phase_ids, pool_wg, pool_wd):
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)
        z = xf.float() @ self.router_w_full               # (N,40)
        probs = torch.softmax(z, dim=-1)
        sel = z + self.bias_full
        top_idx = torch.topk(sel, self.top_k, dim=-1).indices
        top_w = torch.gather(probs, -1, top_idx)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
        wg, wd = self.expert_slices(pool_wg, pool_wd, 0)
        routed, _, drop = self._dispatch(xf, top_idx, top_w, wg, wd)
        y = routed.view(B, T, D) + self.shared(x)
        # Per-phase load histogram for reporting/CV; the single bias vector is
        # updated from the summed load across phases.
        counts = xf.new_zeros((self.n_phases, self.n_experts_full), dtype=torch.float32)
        phase_flat = phase_ids.reshape(-1)
        for p in range(self.n_phases):
            m = phase_flat == p
            counts[p] = _histogram(
                top_idx[m].reshape(-1), self.n_experts_full
            ).float()
        z_loss = z.logsumexp(dim=-1).pow(2).mean()
        return y, counts, drop, z_loss

    def _forward_window_combined(self, x, phase_ids, order, inv_order, phase_sizes, pool_wg, pool_wd):
        """PassChain fast path: one 40-group dispatch per layer, not four.

        The four phase routers still run independently, but their local top-k
        assignments are written into one phase-major (N, k) index tensor.  A
        single grouped GEMM over the layer's 40-expert pool replaces four
        separate 16-expert grouped GEMMs.
        """
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)
        hs = xf.index_select(0, order)
        top_idx_all = torch.empty((N, self.top_k), dtype=torch.long, device=xf.device)
        top_w_all = torch.empty((N, self.top_k), dtype=xf.dtype, device=xf.device)
        all_counts = []
        z_losses = []
        start = 0
        for p in range(self.n_phases):
            n_p = phase_sizes[p]
            end = start + n_p
            h_p = hs[start:end]
            z = h_p.float() @ self.router_w[p].t()
            probs = torch.softmax(z, dim=-1)
            sel = z + self.bias[p]
            top_idx = torch.topk(sel, self.top_k, dim=-1).indices
            top_w = torch.gather(probs, -1, top_idx)
            top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
            top_idx_all[start:end] = top_idx
            top_w_all[start:end] = top_w.to(xf.dtype)
            all_counts.append(_histogram(top_idx.reshape(-1), self.n_experts))
            z_losses.append(z.logsumexp(dim=-1).pow(2).mean())
            start = end
        # Full 40-expert pool for this independent layer.
        wg, wd = pool_wg[self.layer_idx], pool_wd[self.layer_idx]
        routed_sorted, _, drop = self._dispatch(hs, top_idx_all, top_w_all, wg, wd)
        routed = routed_sorted.index_select(0, inv_order).view(B, T, D)
        return (
            routed + self.shared(x),
            torch.stack(all_counts),
            drop,
            torch.stack(z_losses).mean(),
        )

    def forward(self, x, phase_ids, order, inv_order, phase_sizes, pool_wg, pool_wd):
        if self.router_mode == "full40":
            return self._forward_full(x, phase_ids, pool_wg, pool_wd)
        if self.router_mode == "window40" and self.cfg.spec.kind == "independent_layers":
            return self._forward_window_combined(
                x, phase_ids, order, inv_order, phase_sizes, pool_wg, pool_wd
            )
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)
        hs = xf.index_select(0, order)                       # phase-major
        out_sorted = torch.empty_like(hs)
        all_counts = []
        z_losses = []
        drop = xf.new_zeros(())
        start = 0
        for p in range(self.n_phases):
            n_p = phase_sizes[p]
            end = start + n_p
            h_p = hs[start:end]
            z = h_p.float() @ self.router_w[p].t()            # (n_p,16)
            probs = torch.softmax(z, dim=-1)
            sel = z + self.bias[p]
            top_idx = torch.topk(sel, self.top_k, dim=-1).indices
            top_w = torch.gather(probs, -1, top_idx)
            top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
            wg, wd = self.expert_slices(pool_wg, pool_wd, p)
            y_p, counts, drop_p = self._dispatch(h_p, top_idx, top_w, wg, wd)
            out_sorted[start:end] = y_p
            all_counts.append(counts)
            z_losses.append(z.logsumexp(dim=-1).pow(2).mean())
            drop = drop + drop_p
            start = end
        routed = out_sorted.index_select(0, inv_order).view(B, T, D)
        y = routed + self.shared(x)
        counts = torch.stack(all_counts)                      # (4,16)
        z_loss = torch.stack(z_losses).mean()
        return y, counts, drop, z_loss

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, gamma: float = 0.01) -> None:
        if self.router_mode == "full40":
            load = counts.float().sum(dim=0)
            self.bias_full.sub_(gamma * torch.sign(load - load.mean()))
            return
        load = counts.float()
        self.bias.sub_(gamma * torch.sign(load - load.mean(dim=-1, keepdim=True)))


class DenseFFN(nn.Module):
    def __init__(self, cfg: DiffConfig):
        super().__init__()
        self.body = SwiGLU(cfg.d_model, cfg.d_ff_dense)

    def forward(self, x):
        return self.body(x)


class Block(nn.Module):
    def __init__(self, cfg: DiffConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        if layer_idx >= cfg.moe_first_layer:
            self.ffn = MoEFFN(cfg, layer_idx - cfg.moe_first_layer)
            self.is_moe = True
        else:
            self.ffn = DenseFFN(cfg)
            self.is_moe = False

    def forward(self, x, cos, sin, block_mask, phase_ids, order, inv_order, phase_sizes, pool_wg, pool_wd):
        x = x + self.attn(self.norm1(x), cos, sin, block_mask)
        h = self.norm2(x)
        if self.is_moe:
            out, counts, drop, z_loss = self.ffn(
                h, phase_ids, order, inv_order, phase_sizes, pool_wg, pool_wd
            )
            return x + out, counts, drop, z_loss
        return x + self.ffn(h), None, None, x.new_zeros(())


class DiffusionTransformer(nn.Module):
    def __init__(self, cfg: DiffConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        # one or four contiguous expert pools
        self.pool_wg = nn.Parameter(
            torch.empty(cfg.n_pools, cfg.pool_size, cfg.d_model, 2 * cfg.d_ff_expert)
        )
        self.pool_wd = nn.Parameter(
            torch.empty(cfg.n_pools, cfg.pool_size, cfg.d_ff_expert, cfg.d_model)
        )
        nn.init.normal_(self.pool_wg, std=0.02)
        nn.init.normal_(self.pool_wd, std=0.02)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self._rope_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def rope(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (self.cfg.context_len, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope(
                self.cfg.head_dim, self.cfg.context_len, self.cfg.rope_theta, device
            )
        return self._rope_cache[key]

    def rope_positions(self, positions: torch.Tensor):
        cos, sin = self.rope(positions.device)
        return cos[positions], sin[positions]

    def num_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        routed_total = self.cfg.n_pools * self.cfg.pool_size * self.cfg.expert_params
        active_routed = self.cfg.n_moe_layers * self.cfg.top_k * self.cfg.expert_params
        return total, total - routed_total + active_routed

    def moe_layers(self) -> List[MoEFFN]:
        return [layer.ffn for layer in self.layers if layer.is_moe]

    def update_bias(self, counts_per_layer: List[torch.Tensor], gamma: float) -> None:
        for moe, counts in zip(self.moe_layers(), counts_per_layer):
            moe.update_bias(counts, gamma)

    def forward(self, ids, phase_ids, order, inv_order, phase_sizes, block_mask):
        B, T = ids.shape
        assert T == self.cfg.context_len, (T, self.cfg.context_len)
        pos = torch.arange(T, device=ids.device) % self.cfg.seq_len
        cos, sin = self.rope_positions(pos)
        x = self.embed(ids)
        counts = []
        drops = []
        z_losses = []
        for layer in self.layers:
            x, c, d, z = layer(
                x, cos, sin, block_mask, phase_ids, order, inv_order, phase_sizes,
                self.pool_wg, self.pool_wd,
            )
            if c is not None:
                counts.append(c)
                drops.append(d)
                z_losses.append(z)
        hidden = self.norm_f(x)
        out = {
            "hidden": hidden,
            "counts": torch.stack(counts) if counts else ids.new_zeros((0, 4, 16)),
            "drop_frac": torch.stack(drops).mean() if drops else ids.new_zeros(()),
            "router_z_loss": torch.stack(z_losses).mean() if z_losses else ids.new_zeros(()),
        }
        return out

    def lm_head(self, h):
        return F.linear(h, self.embed.weight)
