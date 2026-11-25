"""Optimizer and WSD schedule utilities for the 2026 PassChain training stack."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched Newton-Schulz orthogonalization of the last two dimensions.

    Supports 2D matrices and batched (..., m, n) tensors.  The expert pools are
    4D, so the caller reshapes them to (-1, m, n) before this function.
    """
    assert G.ndim >= 2
    orig_shape = G.shape
    m, n = G.shape[-2], G.shape[-1]
    X = G.reshape(-1, m, n)
    work_dtype = torch.bfloat16 if X.is_cuda else torch.float32
    X = X.to(work_dtype)
    transposed = m > n
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    # Keep the update scale stable for rectangular matrices.
    X = X * math.sqrt(max(1.0, m / n))
    return X.reshape(orig_shape).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalization for hidden 2D weights.

    All parameters in this optimizer must have ndim >= 2.  Batched expert pools
    are orthogonalized independently per expert matrix.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.02,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise ValueError("Muon requires parameters with ndim >= 2")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def build_param_groups(model: torch.nn.Module, lr_muon: float, lr_adamw: float,
                       weight_decay: float = 0.1) -> List[dict]:
    """Split parameters into Muon (hidden 2D/4D weights) and AdamW groups.

    AdamW owns embeddings, routers, 1D norms/biases, and any parameter whose
    name contains ``embed``/``router``/``norm`` or ``bias``.  Everything else
    with ndim >= 2 goes to Muon (attention projections, dense/shared MLPs, and
    the 4D routed-expert pools).
    """
    muon, adamw = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        seen.add(id(p))
        low = name.lower()
        is_adamw = (
            p.ndim < 2
            or "embed" in low
            or "router" in low
            or "norm" in low
            or low.endswith("bias")
        )
        (adamw if is_adamw else muon).append(p)
    # Defensive check: no silently dropped parameters.
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise RuntimeError("parameter-group split missed some trainable parameters")
    return [
        {"params": muon, "lr": lr_muon, "weight_decay": weight_decay, "name": "muon"},
        {"params": adamw, "lr": lr_adamw, "weight_decay": weight_decay, "name": "adamw"},
    ]


def wsd_lr(step: int, total_steps: int, peak_lr: float,
           warmup_frac: float = 0.02, decay_frac: float = 0.15,
           phase: str = "stable") -> float:
    """Warmup-Stable-Decay schedule.

    stable: linear warmup for ``warmup_frac`` then constant peak LR.
    decay:  linear warmup, constant through ``1-decay_frac``, then
            ``peak * (1 - sqrt(progress))`` to zero at the final step.
    """
    if phase not in ("stable", "decay"):
        raise ValueError(f"phase must be 'stable' or 'decay', got {phase!r}")
    total_steps = max(1, int(total_steps))
    step = max(0, int(step))
    warmup = max(1, int(round(total_steps * warmup_frac)))
    if step < warmup:
        return peak_lr * float(step + 1) / float(warmup)
    if phase == "stable":
        return float(peak_lr)
    decay_start = max(warmup, int(round(total_steps * (1.0 - decay_frac))))
    if step < decay_start:
        return float(peak_lr)
    progress = min(1.0, float(step + 1 - decay_start) / float(max(1, total_steps - decay_start)))
    return float(peak_lr * (1.0 - math.sqrt(progress)))


__all__ = ["Muon", "build_param_groups", "wsd_lr", "zeropower_via_newtonschulz5"]
