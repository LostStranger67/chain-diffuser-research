#!/usr/bin/env python
"""Training entry point with a hard post-optimizer parameter-coverage gate."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passchain import training
from passchain.optim import assert_optimizer_coverage


def _assert_optimizer_coverage(model, opt_muon, opt_adamw):
    """Runs immediately after optimizer creation (hooked from training.train)."""
    stats = assert_optimizer_coverage(model, [opt_muon, opt_adamw], "muon+adamw")
    muon_numel = sum(p.numel() for g in opt_muon.param_groups for p in g["params"])
    adamw_numel = sum(p.numel() for g in opt_adamw.param_groups for p in g["params"])
    total_trainable = stats["total_numel"]
    print(
        f"[coverage] muon={muon_numel/1e6:.2f}M + adamw={adamw_numel/1e6:.2f}M "
        f"= {(muon_numel + adamw_numel)/1e6:.2f}M / trainable={total_trainable/1e6:.2f}M",
        flush=True,
    )
    assert muon_numel + adamw_numel == total_trainable, (
        f"optimizer coverage mismatch: {muon_numel} + {adamw_numel} != {total_trainable}"
    )


training.set_post_optimizer_hook(_assert_optimizer_coverage)

if __name__ == "__main__":
    training.main()
