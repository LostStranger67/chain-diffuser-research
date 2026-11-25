# Vertical-Slice Pass-Chaining MoE (`PassChain`)

**Local Pilot Findings (150M–1.5B Tokens) & 3B Cloud Scaling Plan**

---

## 1. Project Goal & Core Thesis

**Objective:** Build, benchmark, and open-source a cache-efficient **Block Diffusion Mixture-of-Experts (MoE)** proof-of-concept capable of streaming from host RAM to a consumer GPU at near-native speed, then use the resulting 3B model and scaling benchmarks as collateral for **$10k+ compute research grants**.

**The Problem:** Standard autoregressive MoE models generate one token at a time. Because the router's choices for the next token are unpredictable, offloading experts from CPU RAM to GPU VRAM causes severe PCIe stalls and cache thrashing. Our autoregressive baseline (`AR R1`) crawled at **74.8 tok/s** with **2.101 ms/tok** of PCIe stall.

**The Solution (`PassChain`):** Block Diffusion (`BD3-LM`) drafts a 64-token block in parallel over 4 denoising passes. By hard-restricting each pass to a **50% overlapping sliding window** of experts per layer (a "vertical slice" across all layers) and running a deterministic background CUDA copy stream, we know 100% in advance which experts are needed next. As each layer finishes a pass, its retiring experts are evicted and the next pass's incoming experts are prefetched while the GPU computes downstream layers—eliminating PCIe stalls completely.

---

## 2. Local Experimental Findings (RTX 3090, 24GB)

All local experiments used the `SmolLM2-135M` tokenizer, an 8-layer GQA backbone (Layer 0 dense, Layers 1–7 MoE, `top-6` routing, `B = 64` block size, 4 noise phases), BF16 + `torch.compile`, and a 4-domain data mix (`60%` FineWeb-Edu, `20%` Cosmopedia, `15%` CodeParrot-clean, `5%` OpenMathInstruct-1). Every streaming benchmark uses `3.1 MB` padded DMA transfers from pinned host RAM and is verified **bit-identical** to all-resident inference.

### Phase A: Architecture Ablations (1.5B Tokens)

We first tested three layer-level streaming layouts at 1.5B noisy tokens (`11,445` steps) to isolate whether **phase partitioning** (disjoint expert pools per noise level) or **sliding-window overlap** drove efficiency:

| Model | Layout | Routed Experts | Val PPL (Overall) | Code PPL | Math PPL | FIFO tok/s | % All-Resident | DMA (MB/tok) | PCIe Stall (ms/tok) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`AR R1` (Ref)** | Autoregressive baseline | 64/layer | 10.72* | — | — | 74.8 | — | 41.50 | 2.101 |
| **`D1`** | Unpartitioned, 0% overlap | 112 | **35.84** | **20.20** | **8.54** | 508.1 | 71.8% | 27.12 | 0.002 |
| **`D1_chain`** | Unpartitioned, 50% layer overlap | 64 | 38.07 | 22.29 | 9.01 | 572.3 | 80.8% | **15.50** | **0.001** |
| **`D3c`** | 4× phase-partitioned + 50% overlap | 256 | 39.12 | 22.50 | 9.11 | **573.7** | **80.8%** | **15.50** | 0.002 |

**AR generation PPL is not directly comparable to masked-diffusion PPL.*

* **Key Takeaway:** Hard phase-partitioning with 0% pass overlap (`D3c`) failed to catch up even at 1.5B tokens (`+9.17%` worse PPL than `D1`). `D1_chain` beat `D3c` on every quality metric with 4× fewer experts.

---

### Phase B: Iso-Parameter Vertical-Slice A/B Test (1.5B Tokens)

To make a strictly fair apples-to-apples comparison without shrinking the model, we trained two **identical-size 40-expert/layer models (`280` routed experts, `~108.5M` total params, `~38.4M` active)** under the exact same **40% VRAM budget** (`16` expert slots per layer across all 7 MoE layers = `112` resident slots):

* **`Unrestricted-40` (Baseline):** Any pass can route `top-6` to any of the 40 experts. Served via a 16-slot/layer synchronous LRU cache.
* **`PassChain-40` (Ours):** Each pass is restricted to a 16-expert window with 50% overlap (`Pass 0: 0..15` $\rightarrow$ `Pass 1: 8..23` $\rightarrow$ `Pass 2: 16..31` $\rightarrow$ `Pass 3: 24..39`). Served via our deterministic vertical-slice background swapper.

#### 1.5B Scaling Curve (Quality vs. Tokens)

| Tokens | `Unrestricted-40` Overall | `PassChain-40` Overall | Δ Overall | `Unrestricted-40` Code | `PassChain-40` Code | Δ Code | `U` Load CV | `P` Load CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0.15B** | 100.28 | 105.83 | +5.53% | 75.13 | 77.41 | +3.04% | 1.017 | 0.342 |
| **0.50B** | 49.82 | 53.09 | +6.56% | 30.00 | 32.33 | +7.78% | 0.983 | 0.369 |
| **1.00B** | 37.88 | 40.82 | +7.77% | 21.67 | 23.57 | +8.80% | 0.981 | 0.359 |
| **1.50B** | **33.92** | 36.72 | **+8.27%** | **19.22** | 20.95 | **+8.99%** | 0.978 | **0.363** |

#### Final 1.5B Vertical-Slice Streaming Benchmark

| Model | All-Resident (tok/s) | Streaming (tok/s) | % of All-Resident | DMA (MB/tok) | PCIe Stall (ms/tok) | Bit-Identical |
| --- | --- | --- | --- | --- | --- | --- |
| **`Unrestricted-40` (Sync LRU)** | 568.73 | 388.78 | 68.4% | 15.21 | 0.805 | True |
| **`PassChain-40` (Vertical-Slice)** | **755.61** | **673.76** | **89.2%** | **12.88** | **0.000** | True |

#### Headline Findings

1. **1.73× Real Streaming Speedup:** `PassChain-40` streams at **673.8 tok/s** vs. `388.8 tok/s` for LRU—recovering **89.2% of native all-resident GPU speed** while keeping 60% of the model in CPU RAM.
2. **Zero PCIe Stall (`0.000 ms/tok`):** Background pass-swapping completely hides PCIe transfer latency (vs. `0.805 ms/tok` stall for LRU) and reduces total DMA traffic by **15.3%**.
3. **Faster Even All-Resident (`755.6` vs. `568.7 tok/s`):** Tighter load balancing inside the 16-expert windows (`CV = 0.363` vs. `0.978`) eliminates uneven expert batching overhead.
4. **Honest Quality Tradeoff:** Restricting each pass to `6-of-16` instead of `6-of-40` carries a stable **~8.3% overall PPL (`~9.0%` Code)** tax at 1.5B tokens. For memory-constrained hardware, trading ~8% PPL for 1.73× throughput and zero stalls is a strong Pareto win.

---

## 3. Future Experiment: 3B Python-Specialized Cloud PoC

A 3B model trained on 15B–20B tokens of general web text is forgettable; a **3B model specialized in Python coding** will actually solve real problems, produce objective **HumanEval / MBPP** scores, and generate a compelling live terminal demo (64-token Python functions materializing in 4 passes while streaming from CPU RAM).

### Target Architecture & Data

* **Model:** `PassChain` Block Diffusion MoE, **~3.0B total parameters / ~400M active per token**, keeping the verified vertical-slice 50% overlapping window design.
* **Data Mix (~15B–20B tokens):** `70%` Python code, `20%` FineWeb-Edu, `10%` Math/Reasoning, followed by a `500M`-token cooldown on clean Python instruction/solution pairs (`OpenCoder` / `Magicoder`).
* **Memory Budget:** Fits comfortably on both 96GB (RTX PRO 6000) and 192GB (B200) GPUs in BF16 / 8-bit AdamW with large batch sizes.

### Compute Allocation Roadmap

| Stage | Hardware | Budget / Time | Deliverable |
| --- | --- | --- | --- |
| **Leg 1: Base Pretraining** | **RTX PRO 6000 (96GB)** | **23 hrs** (free) | Train `0 → ~7B tokens` at peak LR; push rolling checkpoints to private Hugging Face repo every 30 min. |
| **Leg 2: Scale & Cooldown** | **Modal B200 (192GB)** | **~10–14 hrs** (2–3 × $30 free accounts) | Resume checkpoint in FP8; push `7B → ~17B–20B tokens` with cosine decay + 500M Python instruction cooldown. |
| **Leg 3: Grant Collateral** | **Local RTX 3090 (24GB)** | ~3 hrs (free) | 4-bit quantization, HumanEval / MBPP evaluation, vertical-slice streaming benchmark under a strict VRAM cap, and a 30-second live demo video. |

---

## 4. How We Collaborate (AI-Agent-First Workflow)

To keep manual effort minimal and prevent our AI agents from overwriting each other's work, we will spin this out into a clean, dedicated GitHub repository (`sliding-window-diffusion-moe`):

* **GitHub (`< 10 MB`):** Code (`src/`), immutable YAML configs (`configs/`), unit tests, small JSON metrics, plots, and Markdown reports. `.gitignore` strictly blocks `.pt`, `.bin`, and raw logs.
* **Private Hugging Face Repo (Free):** Stores tokenized `.bin` shards and rolling `.pt` checkpoints so your RTX PRO 6000 and my Modal B200 / 3090 can pull the exact same data and resume checkpoints seamlessly.
* **`AGENTS.md`:** A root rulebook read automatically by both of our AI agents (never edit `DONE` configs, always run `pytest` before GPU jobs, never commit checkpoints).
* **`EXPERIMENTS.md`:** A shared Markdown job board (`TODO` / `RUNNING` / `DONE`). Whenever you have free GPU hours, you just `git pull` and tell your agent: *"Read `AGENTS.md` and `EXPERIMENTS.md`, pick the next `TODO` job for the RTX PRO 6000, run it, sync checkpoints to HF, and push the results."*