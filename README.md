# sliding-window-diffusion-moe

Standalone PassChain block-diffusion MoE research repository.

- **Model:** 16-layer, 1536-wide GQA backbone with QK-Norm; Layer 0 dense,
  Layers 1–15 PassChain-40 MoE (1 shared + 40 routed experts/layer, top-6,
  50% overlapping 16-expert vertical-slice windows).
- **3B config:** `configs/passchain_3b_python.yaml` — 2.93B total params,
  ~398M routed-active params/token (plus shared/backbone: ~604M total active).
- **Training:** Muon + AdamW, WSD (`stable` / `1-sqrt` decay), BF16,
  optional torchao FP8 on SM ≥ 9.0, router + output-logit Z-loss.
- **Data:** sequence-packed 2048-token blocks with EOS document boundaries;
  70% deduplicated Python, 20% FineWeb-Edu, 10% OpenMathInstruct.
- **Checkpointing:** atomic `ckpt_latest.pt` every 30 min + private HF upload.

## Quick start

```bash
pytest -q
python scripts/train.py --config configs/debug_50m.yaml --steps 20
```

## 3B / Modal

```bash
# Leg 1 on a provisioned RTX PRO 6000 or local machine:
python scripts/train.py --config configs/passchain_3b_python.yaml --phase stable --resume_from_hf

# Leg 2 on Modal B200 (4h cap):
modal run modal_train.py --phase stable
modal run modal_train.py --phase decay
```

Environment: `HF_REPO_ID` points at the private Hugging Face model repo;
`HF_TOKEN` supplies write access. No weights, shards, or logs belong in Git.

## Reports

Historical verified PassChain results are in `results/`:
`PASSCHAIN_REPORT.md`, `PASSCHAIN_1500M_REPORT.md`, `SCALE_1500M_REPORT.md`,
and the two scaling plots.
