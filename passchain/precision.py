"""Precision helpers: native torchao FP8 on Blackwell/Hopper, BF16 fallback."""

from __future__ import annotations

import torch


def fp8_available(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(device)
    return cap[0] >= 9


def _fp8_module_filter(module: torch.nn.Module, fqn: str) -> bool:
    """Only safe, tensor-core-aligned linear projections are swapped.

    Keep the tied LM head and routers in high precision, and skip any linear
    whose input/output dimensions violate the FP8 GEMM multiple-of-16 rule.
    """
    if not isinstance(module, torch.nn.Linear):
        return False
    low = fqn.lower()
    if "lm_head" in low or "router" in low:
        return False
    return (module.in_features % 16 == 0) and (module.out_features % 16 == 0)


def enable_fp8(model: torch.nn.Module, enabled: bool = True) -> bool:
    """Swap safe ``nn.Linear`` layers with torchao Float8Linear for training.

    Runs before optimizer construction so the Muon/AdamW splitter sees the
    swapped modules (and their weight parameters).  On SM 8.6 or when torchao
    is unavailable, the model stays in BF16 and this returns False.
    """
    if not enabled:
        return False
    if not fp8_available():
        print("[precision] FP8 requested but GPU is SM < 9.0; using BF16 fallback", flush=True)
        return False
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except Exception as exc:  # pragma: no cover - depends on B200 image
        print(f"[precision] torchao unavailable ({exc}); using BF16 fallback", flush=True)
        return False
    try:
        converted = convert_to_float8_training(
            model,
            module_filter_fn=_fp8_module_filter,
            config=Float8LinearConfig(),
        )
        if converted is not model:  # root-level Linear is not our case
            raise RuntimeError("unexpected root-level model swap")
    except Exception as exc:  # pragma: no cover - Blackwell-specific
        print(f"[precision] torchao FP8 setup failed ({exc}); using BF16 fallback", flush=True)
        return False
    print("[precision] torchao convert_to_float8_training enabled (safe nn.Linear layers)", flush=True)
    return True


__all__ = ["enable_fp8", "fp8_available"]
