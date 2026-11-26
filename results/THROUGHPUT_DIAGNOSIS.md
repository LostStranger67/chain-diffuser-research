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

and the per-step `fwd_bwd_s` / `opt_s` counters.  The pre-fix baseline was
51.3 s/step; the post-fix measurement is below.

## Post-fix B200 dry run

After the combined dispatch + graph-break fix, a 5-step B200 run reported:

- **23.12 s/step** (45.4k tok/s), down from 51.3 s
- `opt_s = 0.185 s` (0.8%)
- `fwd_bwd_s = 22.935 s` (99.2%)
- ~358 ms per micro-batch at micro-batch 8

This still leaves substantial headroom. Use `--profile-step` on the next B200
run to identify the top CUDA kernels; the training loss is now chunked over
sequence tokens (`lm_chunk_tokens`) and `--no-fp8` is available for an A/B test.

## B200 `--profile-step` trace (post-fix, pre-optimization)

- `aten::index_add` / `indexFuncLargeIndex<BFloat16, ReduceAdd>`:
  **316.0 ms (23.2%, 60 calls @ 5.25 ms)**
- `flex_attention` backward: **174.0 ms (12.8%)** vs 38.9 ms forward
- Chunked CE with `lm_chunk_tokens=512`: 64 chunks/micro, 252 ms CPU,
  ~156 ms GPU, 7,115 kernel launches and Command Buffer Full (128.7 ms)
- `aten::_grouped_mm` 247 ms and `aten::_scaled_mm` 58 ms were already fast

### Fixes applied

- Replaced the MoE `index_add_` unpermute with a custom bijective
  `_UnpermuteSum` autograd function: forward uses `index_select(inv_order) +
  view(N,k,D).sum(1)`; backward uses `index_select(order)` on the expanded
  token gradient. A pytest compares forward, dX, and dWeights against the old
  `index_add_` implementation.
- Raised 3B `lm_chunk_tokens` **512 → 4096** (8 chunks per 16-micro-batch
  instead of 64) and reused one FP32 logits copy for both CE and Z-loss.
- FlexAttention audit: the installed kernel requires **BLOCK_SIZE=128**; a
  64-token kernel block raised `Q and KV block size must be divisible by
  BLOCK_M/BLOCK_N`.  We keep 128 and pass contiguous Q/K/V.  Custom
  `kernel_options` were not enabled without a B200 correctness/timing sweep.
- Added `--no-upload` to skip 23.7 GB + 11.7 GB uploads on profiling runs.
- `ckpt_final.pt` is now BF16 (~5.85 GB at 3B); `ckpt_latest.pt` remains
  full-precision with optimizer states for lossless resume.
