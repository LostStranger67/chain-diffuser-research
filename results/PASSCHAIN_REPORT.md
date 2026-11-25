# PassChain-40 vs Unrestricted-40 — iso-parameter block-diffusion A/B

Clean 150M-noisy-token A/B on `data/shards_150m/`, same 8-layer GQA backbone, Layer 0 dense, Layers 1–7 MoE, one shared expert (`d_ff=256`) plus **40 routed experts per layer** (`d_ff=256`, top-6).  **280 routed experts total, no layer-to-layer weight sharing.**  BD3-LM `x_0 || x_t` mask, B=64, 4 noise phases, 1,145 steps (150M noisy tokens), seed 42, `gamma_bias=0.01`, BF16 + `torch.compile`.

**Unrestricted-40:** one `Linear(384,40)` router + `b[40]` per layer; any denoising pass (and clean `x_0` tokens) may route top-6 to any of the 40 experts.

**PassChain-40:** one lightweight `Linear(384,16)` router + `b[phase,16]` per pass per layer.  Pass windows are hard-restricted with 50% overlap: pass 0 `0..15`, pass 1 `8..23`, pass 2 `16..31`, pass 3 `24..39` (clean `x_0` included in pass 3).  Experts `8..31` are shared by adjacent passes and receive accumulated gradients.

## Verification gate

- Both models: **280 routed experts**; routed-expert parameter count is **82,575,360 each** (280 × 294,912).  Total params differ by only **64,680 router parameters** (0.06%): Unrestricted-40 uses one 40-way router/layer, PassChain-40 four 16-way routers/layer.
- PassChain phase slices exactly match their 16-expert windows; overlapping experts `8..31` accumulate gradients correctly (`rel≈2.5e-6` gradient-accumulation check).
- Both streaming runtimes are **bit-identical** to all-resident inference on the same prompts.
- 20 BF16 training steps on each model: zero NaN/Inf.

## Side-by-side result

| Model | Routed experts | Total params | Active/token | Val PPL overall | Code PPL | Math PPL | ELBO | Load CV max | All-resident tok/s | Streaming tok/s | % of all-resident | DMA MB/tok | PCIe stall ms/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Unrestricted-40** | 280 | 108.54M | 38.35M | 112.1205 | 84.9184 | 23.5911 | 8.2542 | 1.0011 | 519.90 | 333.33 | 64.1% | 18.91 | 1.030 |
| **PassChain-40** | 280 | 108.61M | 38.42M | 125.7792 | 102.9139 | 26.5595 | 8.5524 | 0.3473 | 739.40 | 645.29 | 87.3% | 12.88 | 0.000 |

The streaming budget is **16 expert slots per layer × 7 MoE layers = 112 slots** (40% of the 280 routed experts resident at once).  DMA transfers are 3.1 MB padded rows from pinned host RAM; cuBLAS is deterministic.  Every streaming run was checked bit-identical to its all-resident output.

All-resident speed is the compute upper bound for each runtime.  Unrestricted-40 is lower because its full 40-way router activates a wider expert union per block, so the fixed 16-slot budget forces three deterministic 16-expert chunks at the all-resident upper bound as well.  PassChain-40 has one 16-expert window per pass and a single dispatch.

Runtime definitions: Unrestricted-40 uses a 16-slot **synchronous LRU** per layer; if the 64-token block activates more than 16 distinct experts, the union is processed in deterministic sorted chunks of at most 16.  PassChain-40 uses the **deterministic vertical-slice background swapper**: a dedicated non-blocking copy stream overwrites layer L's retiring experts with its next-pass experts as soon as layer L finishes the current pass, and the compute stream waits on layer L's copy-done event only when it reaches Layer L in the next pass.

## Training cost

| Model | Steps | Noisy tokens | Train time |
|---|---:|---:|---:|
| **Unrestricted-40** | 1145 | 150,077,440 | 39.7 min |
| **PassChain-40** | 1145 | 150,077,440 | 48.2 min |

## Findings

- PassChain-40 vs Unrestricted-40: overall PPL +12.18%, Code +21.19%, Math +12.58%, ELBO +3.61%.
- Load CV max: Unrestricted-40 1.0011, PassChain-40 0.3473.
- Streaming: Unrestricted-40 333.33 tok/s (64.1% of all-resident, 18.91 MB/tok); PassChain-40 645.29 tok/s (87.3%, 12.88 MB/tok).  PCIe stall is 1.030 vs 0.000 ms/tok.
- PassChain-40 moves -31.9% DMA relative to Unrestricted-40; the vertical-slice schedule hides the copy behind the seven-layer pass compute (measured 0.000 ms/tok stall).
- Trade-off: Unrestricted-40 wins quality (112.12 vs 125.78 overall PPL); PassChain-40 wins streaming (645.29 vs 333.33 tok/s, 87.3% vs 64.1% of all-resident, zero stall) and load balance (CV 0.347 vs 1.001).

## Caveats

- Both models have the same 280 routed experts and expert parameter count; the small total-parameter difference is exactly the router: Unrestricted-40 has one 40-way router per layer, PassChain-40 has four 16-way routers per layer.
- The sync-LRU Unrestricted runtime chunks a >16-expert activated union; this is required by the fixed 16-slot-per-layer VRAM budget and is counted in its DMA/stall.
- PassChain-40 prefetches 8 experts/layer after passes 0–2 and 16 experts/layer after pass 3, so the next block starts with experts 0..15.
- PPL is masked cross-entropy on the same held-out validation files used by the earlier diffusion runs.
