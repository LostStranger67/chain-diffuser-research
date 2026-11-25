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


def enable_fp8(model: torch.nn.Module, enabled: bool = True) -> bool:
    """Apply torchao dynamic FP8 to all nn.Linear projections.

    On SM >= 9.0 with torchao installed this switches attention, dense, and
    shared-expert Linear layers to float8_dynamic_activation_float8_weight.
    On SM 8.6 (RTX 3090) or when torchao is unavailable it leaves the model in
    BF16 and returns False.  The grouped routed-expert pool GEMMs remain BF16;
    they are streamed as raw contiguous expert rows, so torchao's module-level
    replacement does not apply to them.
    """
    if not enabled:
        return False
    if not fp8_available():
        print("[precision] FP8 requested but GPU is SM < 9.0; using BF16 fallback", flush=True)
        return False
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )
    except Exception as exc:  # pragma: no cover - depends on B200 image
        print(f"[precision] torchao unavailable ({exc}); using BF16 fallback", flush=True)
        return False
    try:
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(),
            filter_fn=lambda m, n: isinstance(m, torch.nn.Linear),
        )
    except Exception as exc:  # pragma: no cover - Blackwell-specific
        print(f"[precision] torchao FP8 setup failed ({exc}); using BF16 fallback", flush=True)
        return False
    print("[precision] torchao dynamic FP8 enabled for nn.Linear layers", flush=True)
    return True


__all__ = ["enable_fp8", "fp8_available"]
