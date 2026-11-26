# B200 throughput diagnosis

Observed dry run: **51.3 s / 1.05M-token step (~20.4k tok/s)**.

## Code audit

1. **Four grouped GEMMs per MoE layer.** The old `MoEFFN.forward` looped over
   the four phase windows and called `_dispatch_grouped` separately for each
   16-expert window. With 15 MoE layers and 32 gradient-accumulation
   micro-batches, that is **1,920 small grouped GEMMs per optimizer step**
   (plus 1,920 sorts), each with low arithmetic intensity.
2. **Nine Dynamo graph breaks from `torch.bincount`.** `_dispatch_grouped`,
   `_dispatch_padded`, and the per-phase load counters used data-dependent
   `bincount` shapes. `torch._dynamo.explain` reported 9 breaks / 10 graphs,
   preventing a single fused MoE graph.
3. **Dense Newton-Schulz over all expert matrices.** Muon orthogonalized every
   one of the 600 expert matrices every step. It was batched, but the full
   expert pool is much larger than the top-6 active set.
4. **Small micro-batches.** The initial B200 dry run used a conservative
   micro-batch, increasing launch/sort overhead and reducing GEMM efficiency.

No dense evaluation of unselected experts and no extra denoising forward passes
are present in training: each token is assigned one noise phase and top-6
routing.

## Changes

- `MoEFFN._forward_window_combined`: all four phase routers write into one
  phase-major `(N, 6)` index tensor; one **40-group** grouped GEMM runs per
  layer instead of four 16-group GEMMs. Grouped calls per step: **1,920 → 480**.
- Replaced `torch.bincount` with fixed-shape `scatter_add_` histograms.
  `torch._dynamo.explain` now reports **0 graph breaks / 1 graph** for the debug
  MoE forward.
- Muon already batches the 3D/4D expert tensors in BF16; added an explicit
  4D batched-NS regression test.
- B200 micro-batch default: **16** (8 on 96 GB cards); global batch remains 512.
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Added per-step `step_s`, `fwd_bwd_s`, `opt_s`, `tok/s` telemetry and
  `--log-every`.
- Warmup is anchored to the configured stable-step count, so short benchmarks
  do not collapse it to zero. Added `--no-resume`.

## Local RTX 3090 check

Debug config, 20 steps, no compile, same batch shape:

| | Wall | Wall tok/s | Steady tok/s |
|---|---:|---:|---:|
| Before | 10.835 s | 7,560 | — |
| After | 9.667 s | 8,474 | 15,432 |

Local debug telemetry after the change: `fwd_bwd_s=0.236`, `opt_s=0.030`.
The B200 split should be measured with:

```bash
modal run modal_train.py --phase stable --steps 5 --micro-batch 16 \
  --no-resume --log-every 1
```

and the per-step `fwd_bwd_s` / `opt_s` counters.  No B200 post-fix numbers are
claimed here.
