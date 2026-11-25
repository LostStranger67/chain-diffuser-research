# Local verification gate

**Date:** 2026-11-25  
**Hardware:** RTX 3090 24 GB (SM 8.6), PyTorch 2.13.0+cu130

## `pytest -q`

`7 passed`:

- 3B PassChain-40 parameter estimate: 2.9266B total, 398.1M routed-active, 600 routed experts.
- PassChain 50% windows `0..15 / 8..23 / 16..31 / 24..39` and QK-Norm modules.
- Router + output-logit Z-loss finite and positive.
- Muon/AdamW parameter split disjoint and complete.
- WSD stable and `1 - sqrt(progress)` decay math.
- FP8 fallback on SM 8.6.
- Newton-Schulz orthogonalization shape/finiteness.

## 20-step debug smoke

Config: `configs/debug_50m.yaml` (actual 44.67M params; 19.60M active).

- First loss: `21.0611`
- Last loss: `11.8306`
- NaN/Inf: none
- Checkpoint: `ckpt_latest.pt` written at step 10 and step 20; `ckpt_final.pt` written.
- Resume check: step 10 → resume `ckpt_latest.pt` → step 20; loss continued `16.2099 → 11.9329`.
