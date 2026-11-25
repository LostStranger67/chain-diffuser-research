"""Shared configuration and expert-pool layout math for the diffusion sweep.

The five runs differ *only* in `layout`; every other hyper-parameter is shared.
A phase pool is a contiguous parameter block of ``16 + 6 * stride`` experts.
MoE layer ``L`` (0..6) uses ``pool[L*stride : L*stride + 16]`` for its 16
experts.  D1 has a single pool shared by all four phases.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Dict, Tuple


# phase -> (t_low, t_high); t is the per-token mask probability of the block
PHASE_RANGES: Tuple[Tuple[float, float], ...] = (
    (0.75, 1.00),  # phase 0: coarsest
    (0.50, 0.75),  # phase 1: mid-coarse
    (0.25, 0.50),  # phase 2: mid-fine
    (0.05, 0.25),  # phase 3: finest
)


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    n_pools: int
    stride: int
    overlap_frac: float
    description: str
    # New independent-layer layouts (Unrestricted-40 / PassChain-40).  Old
    # layouts leave these at their phase-pool defaults.
    kind: str = "phase_pool"          # phase_pool | independent_layers
    experts_per_layer: int = 16
    window_starts: Tuple[int, ...] = ()
    router_mode: str = "phase16"      # phase16 | window40 | full40
    fixed_pool_size: int = 0
    total_routed_override: int = 0

    @property
    def pool_size(self) -> int:
        if self.fixed_pool_size:
            return self.fixed_pool_size
        return 16 + 6 * self.stride

    @property
    def total_routed_experts(self) -> int:
        if self.total_routed_override:
            return self.total_routed_override
        return self.n_pools * self.pool_size

    @property
    def fetch_per_layer(self) -> int:
        return self.stride

    @property
    def layer_split(self) -> Tuple[int, int, int]:
        """(shared with previous, unique, shared with next) for an interior layer."""
        prev = 16 - self.stride
        nx = 16 - self.stride
        uniq = 16 - prev - nx
        return prev, uniq, nx


LAYOUTS: Dict[str, LayoutSpec] = {
    "d1": LayoutSpec("d1", 1, 16, 0.00, "unpartitioned baseline (phases share 16/layer)"),
    "d1_chain": LayoutSpec(
        "d1_chain", 1, 8, 0.50,
        "single 64-expert pool shared by all phases, 50% layer overlap",
    ),
    "d2": LayoutSpec("d2", 4, 16, 0.00, "phase-partitioned (4 x 16/layer)"),
    "d3a": LayoutSpec("d3a", 4, 12, 0.25, "sliding-window chain, 25% overlap"),
    "d3b": LayoutSpec("d3b", 4, 10, 0.375, "sliding-window chain, 37.5% overlap"),
    "d3c": LayoutSpec("d3c", 4, 8, 0.50, "sliding-window chain, 50% overlap"),
    "unrestricted40": LayoutSpec(
        "unrestricted40", 7, 40, 0.00,
        "40 independent experts/layer, one full router/layer (280 routed)",
        kind="independent_layers", experts_per_layer=40,
        window_starts=(0, 0, 0, 0), router_mode="full40",
        fixed_pool_size=40, total_routed_override=280,
    ),
    "passchain40": LayoutSpec(
        "passchain40", 7, 8, 0.50,
        "40 independent experts/layer, per-pass 16-expert vertical slices (280 routed)",
        kind="independent_layers", experts_per_layer=40,
        window_starts=(0, 8, 16, 24), router_mode="window40",
        fixed_pool_size=40, total_routed_override=280,
    ),
    "passchain_3b": LayoutSpec(
        "passchain_3b", 15, 8, 0.50,
        "3B PassChain-40: 15 MoE layers x 40 routed experts (600 routed)",
        kind="independent_layers", experts_per_layer=40,
        window_starts=(0, 8, 16, 24), router_mode="window40",
        fixed_pool_size=40, total_routed_override=600,
    ),
}


@dataclass
class DiffConfig:
    # tokenizer / data
    vocab_size: int = 49152
    mask_token_id: int = 16
    # backbone
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff_dense: int = 1536
    moe_first_layer: int = 1
    d_ff_expert: int = 256
    top_k: int = 6
    experts_per_phase: int = 16
    n_phases: int = 4
    layout: str = "d1"
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    bias_gamma: float = 0.01
    moe_impl: str = "auto"  # auto | grouped | padded (padded is for CPU/tests)
    qk_norm: bool = False
    z_loss_weight: float = 0.0
    fp8: bool = False
    # block diffusion
    block_size: int = 64
    n_blocks: int = 8
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}; choose from {sorted(LAYOUTS)}")
        if self.n_phases != 4:
            raise ValueError("this experiment fixes n_phases=4")
        if self.experts_per_phase != 16:
            raise ValueError("this experiment fixes experts_per_phase=16")
        if self.block_size % 8 != 0:
            raise ValueError("block_size must be a multiple of 8 for the unmask schedule")

    @property
    def spec(self) -> LayoutSpec:
        return LAYOUTS[self.layout]

    @property
    def seq_len(self) -> int:
        return self.n_blocks * self.block_size

    @property
    def context_len(self) -> int:
        return 2 * self.seq_len

    @property
    def n_pools(self) -> int:
        # Each MoE layer owns its own 40-expert pool in the new A/B.
        if self.spec.kind == "independent_layers":
            return self.n_moe_layers
        return self.spec.n_pools

    @property
    def stride(self) -> int:
        return self.spec.stride

    @property
    def experts_per_layer(self) -> int:
        return self.spec.experts_per_layer

    @property
    def router_mode(self) -> str:
        return self.spec.router_mode

    @property
    def window_starts(self) -> Tuple[int, ...]:
        return self.spec.window_starts

    @property
    def counts_dim(self) -> int:
        # full40 uses a 40-wide load histogram; window/phase use 16-wide windows.
        return self.experts_per_layer if self.router_mode == "full40" else self.experts_per_phase

    @property
    def pool_size(self) -> int:
        return self.spec.pool_size

    @property
    def n_moe_layers(self) -> int:
        return max(0, self.n_layers - self.moe_first_layer)

    @property
    def phase_to_pool(self) -> Tuple[int, ...]:
        if self.spec.kind == "independent_layers":
            return (0, 0, 0, 0)
        if self.spec.n_pools == 1:
            return (0, 0, 0, 0)
        return (0, 1, 2, 3)

    @property
    def expert_params(self) -> int:
        return self.d_model * 2 * self.d_ff_expert + self.d_ff_expert * self.d_model

    @property
    def routed_experts(self) -> int:
        return self.n_pools * self.pool_size

    @property
    def routed_active_params(self) -> int:
        """Activated routed-expert parameters per token (excludes shared/backbone)."""
        return self.n_moe_layers * self.top_k * self.expert_params

    def estimate_params(self) -> dict:
        """Cheap analytical estimate used by the config gate."""
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * (
            2 * self.d_model * self.d_model
            + self.d_model * 2 * self.n_kv_heads * self.head_dim
            + (2 * self.head_dim + 2 * self.d_model if self.qk_norm else 2 * self.d_model)
        )
        dense = 3 * self.d_model * self.d_ff_dense
        routers = self.n_moe_layers * self.n_phases * (
            self.d_model * self.experts_per_phase + self.experts_per_phase
        )
        experts_total = self.n_moe_layers * (
            self.experts_per_layer * self.expert_params
            + self.expert_params
        )
        total = embed + attn + dense + routers + experts_total + self.d_model
        active_routed = self.routed_active_params
        active_total = total - experts_total + active_routed
        return {
            "total": total,
            "routed_experts": self.routed_experts,
            "routed_active": active_routed,
            "active_total": active_total,
            "expert_params_total": experts_total,
        }

    def pool_span(self, layer_idx: int) -> Tuple[int, int]:
        if self.spec.kind == "independent_layers":
            return 0, self.experts_per_layer
        start = layer_idx * self.stride
        return start, start + self.experts_per_phase

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layout_spec"] = asdict(self.spec)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DiffConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def layout_summary() -> str:
    lines = [
        "layout  pools stride overlap  pool_size total_routed fetch/layer  split(prev/uniq/next)",
        "------  ----- ------ -------  --------- ------------ -----------  -------------------",
    ]
    for k in ("d1", "d1_chain", "d2", "d3a", "d3b", "d3c", "unrestricted40", "passchain40", "passchain_3b"):
        s = LAYOUTS[k]
        if s.kind == "independent_layers":
            split = "layer-independent"
        else:
            p, u, n = s.layer_split
            split = f"{p:5d}/{u:3d}/{n:3d}"
        lines.append(
            f"{k:14s} {s.n_pools:5d} {s.stride:6d} {s.overlap_frac:6.1%}  "
            f"{s.pool_size:9d} {s.total_routed_experts:12d} {s.fetch_per_layer:11d}  {split}"
        )
    return "\n".join(lines)
