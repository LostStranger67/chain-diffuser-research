# EXPERIMENTS.md — shared job board

Status: `TODO` / `RUNNING` / `DONE`. Update only the row you own.

## Leg 1 — RTX PRO 6000 (96 GB)

| Field | Value |
|---|---|
| Job | Base pretraining |
| Tokens | 0 → 7B |
| Config | `configs/passchain_3b_python.yaml` |
| Schedule | WSD **stable** (2% warmup, constant peak LR) |
| Checkpointing | `ckpt_latest.pt` every 30 min → private HF repo |
| Status | **TODO** |
| Notes | 3.0B total / ~400M routed-active PassChain-40; Python-heavy mix |

## Leg 2 — Modal B200 (192 GB)

| Field | Value |
|---|---|
| Job | Scale + cooldown |
| Tokens | 7B → 17B |
| Config | `configs/passchain_3b_python.yaml` |
| Schedule | WSD **stable** then `1 - sqrt(progress)` cooldown over the final 10–15% on clean Python instruction data |
| Checkpointing | Resume `ckpt_latest.pt` from HF; roll every 30 min; push `ckpt_final.pt` |
| Launcher | `modal run modal_train.py --phase stable` (then `--phase decay` for the cooldown segment) |
| Status | **TODO** |
| Notes | 4-hour hard cap (`timeout=14400`) inside Starter credit; rolling checkpoints make timeouts safe |

## Local gate

| Job | Command | Status |
|---|---|---|
| Unit tests | `pytest -q` | **DONE** — 7 passed |
| 20-step RTX 3090 smoke | `python scripts/train.py --config configs/debug_50m.yaml --steps 20` | **DONE** — loss 21.06 → 11.83, zero NaN |
