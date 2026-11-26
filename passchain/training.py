"""PassChain 3B training stack: Muon/AdamW, WSD, FP8 fallback, HF rolling sync."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .config import DiffConfig
from .data import PackedTokenDataset, SyntheticTokenDataset
from .hf_sync import download_latest, upload_latest
from .masking import batch_to_torch, block_diffusion_block_mask, build_sequence_batch
from .model import DiffusionTransformer
from .optim import Muon, build_param_groups, wsd_lr
from .precision import enable_fp8

_POST_OPTIMIZER_HOOK = None


def set_post_optimizer_hook(fn):
    """Used by scripts/train.py to assert optimizer coverage immediately after creation."""
    global _POST_OPTIMIZER_HOOK
    _POST_OPTIMIZER_HOOK = fn


def load_yaml_config(path: str) -> Tuple[DiffConfig, Dict, Dict]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    model_over = raw.get("model", {}) or {}
    cfg = DiffConfig.from_dict({**DiffConfig().to_dict(), **model_over})
    train = raw.get("training", {}) or {}
    data = raw.get("data", {}) or {}
    return cfg, train, data


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def masked_ce(hidden_noisy, weight, targets, mask, t_tok, phase_noisy, chunk=2048):
    """Masked CE with 1/t ELBO weighting plus the output-logit Z statistic."""
    B, S, D = hidden_noisy.shape
    h = hidden_noisy.reshape(-1, D)
    tgt = targets.reshape(-1)
    m = mask.reshape(-1).to(torch.float32)
    t = t_tok.reshape(-1).to(torch.float32).clamp_min(0.05)
    ph = phase_noisy.reshape(-1)
    n = h.shape[0]
    n_masked = m.sum()
    ce_sum = h.new_zeros((), dtype=torch.float32)
    elbo_sum = h.new_zeros((), dtype=torch.float32)
    per_phase_ce = h.new_zeros((4,), dtype=torch.float32)
    per_phase_elbo = h.new_zeros((4,), dtype=torch.float32)
    per_phase_count = h.new_zeros((4,), dtype=torch.float32)
    z2_sum = h.new_zeros((), dtype=torch.float32)
    z_count = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        logits = F.linear(h[s:e], weight)
        ce = F.cross_entropy(logits.float(), tgt[s:e], reduction="none")
        m_c = m[s:e]
        ce_m = ce * m_c
        wce_m = ce_m / t[s:e]
        ce_sum = ce_sum + ce_m.sum()
        elbo_sum = elbo_sum + wce_m.sum()
        per_phase_ce = per_phase_ce.index_add(0, ph[s:e], ce_m)
        per_phase_elbo = per_phase_elbo.index_add(0, ph[s:e], wce_m)
        per_phase_count = per_phase_count.index_add(0, ph[s:e], m_c)
        z = torch.logsumexp(logits.float(), dim=-1)
        z2_sum = z2_sum + z.pow(2).sum()
        z_count += z.numel()
    loss = elbo_sum / n_masked.clamp_min(1.0)
    return loss, ce_sum, elbo_sum, per_phase_ce, per_phase_elbo, per_phase_count, n_masked, z2_sum, z_count


def train_step(model, cfg, d, block_mask, ce_chunk=2048):
    out = model(
        d["ids"], d["phase_ids"], d["order"], d["inv_order"],
        d["phase_sizes"], block_mask,
    )
    S = cfg.seq_len
    hidden = out["hidden"][:, S:]
    phase_noisy = d["phase_ids"][:, S:]
    ce_loss, ce_sum, elbo_sum, pp_ce, pp_elbo, pp_cnt, n_masked, z2_sum, z_count = masked_ce(
        hidden, _base_model(model).embed.weight, d["targets"], d["mask"], d["t_tok"],
        phase_noisy, ce_chunk,
    )
    output_z = z2_sum / max(z_count, 1)
    z_loss = cfg.z_loss_weight * (out["router_z_loss"] + output_z)
    return {
        "loss": ce_loss + z_loss,
        "ce_loss": ce_loss,
        "z_loss": z_loss,
        "router_z": out["router_z_loss"],
        "output_z": output_z,
        "counts": out["counts"],
        "drop": out["drop_frac"],
        "n_masked": n_masked,
        "ce_sum": ce_sum,
        "elbo_sum": elbo_sum,
        "pp_ce": pp_ce,
        "pp_elbo": pp_elbo,
        "pp_cnt": pp_cnt,
    }


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _base_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(path: str, model, opt_muon, opt_adamw, step: int, cfg,
                    rng: np.random.Generator, include_optimizer: bool = True,
                    extra: Optional[dict] = None):
    state = {
        "model": _to_cpu(_base_model(model).state_dict()),
        "model_config": cfg.to_dict(),
        "layout": cfg.layout,
        "step": int(step),
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    if include_optimizer:
        state["muon"] = _to_cpu(opt_muon.state_dict())
        state["adamw"] = _to_cpu(opt_adamw.state_dict())
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, opt_muon, opt_adamw, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _base_model(model).load_state_dict(ckpt["model"])
    if "muon" in ckpt:
        opt_muon.load_state_dict(ckpt["muon"])
    if "adamw" in ckpt:
        opt_adamw.load_state_dict(ckpt["adamw"])
    if ckpt.get("torch_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if ckpt.get("cuda_rng_state") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    step = int(ckpt.get("step", 0))
    return step, ckpt


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(args) -> Dict:
    cfg, train_cfg, data_cfg = load_yaml_config(args.config)
    if args.phase:
        train_cfg["phase"] = args.phase
    if args.micro_batch:
        train_cfg["micro_batch"] = int(args.micro_batch)
    if args.no_compile:
        train_cfg["compile"] = False
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    seq_len = cfg.seq_len
    global_batch = int(train_cfg.get("global_batch_seqs", 512))
    micro_batch = int(train_cfg.get("micro_batch", 1))
    if global_batch % micro_batch:
        raise ValueError("global_batch_seqs must be divisible by micro_batch")
    accum = global_batch // micro_batch
    tokens_per_step = global_batch * seq_len
    total_tokens = int(train_cfg.get("total_tokens", 17_000_000_000))
    total_steps = int(math.ceil(total_tokens / tokens_per_step))
    if args.steps:
        total_steps = int(args.steps)
    phase = train_cfg.get("phase", "stable")
    warmup_frac = float(train_cfg.get("warmup_frac", 0.02))
    decay_frac = float(train_cfg.get("decay_frac", 0.15))

    print(f"[train] layout={cfg.layout} seq_len={seq_len} context={cfg.context_len} "
          f"global_batch={global_batch} micro={micro_batch} accum={accum} "
          f"steps={total_steps} tokens/step={tokens_per_step:,} phase={phase}", flush=True)
    est = cfg.estimate_params()
    print(f"[train] estimate total={est['total']/1e9:.3f}B routed={est['routed_experts']} "
          f"routed_active={est['routed_active']/1e6:.1f}M active_total={est['active_total']/1e6:.1f}M",
          flush=True)

    model = DiffusionTransformer(cfg).to(device)
    total_p, active_p = model.num_params()
    print(f"[train] actual params total={total_p/1e6:.2f}M active/token={active_p/1e6:.2f}M", flush=True)
    fp8_enabled = enable_fp8(model, enabled=bool(cfg.fp8))

    lr_muon = float(train_cfg.get("lr_muon", 0.02))
    lr_adamw = float(train_cfg.get("lr_adamw", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.1))
    groups = build_param_groups(model, lr_muon, lr_adamw, wd)
    opt_muon = Muon(groups[0]["params"], lr=lr_muon, weight_decay=wd,
                    momentum=float(train_cfg.get("muon_momentum", 0.95)),
                    ns_steps=int(train_cfg.get("muon_ns_steps", 5)))
    opt_adamw = torch.optim.AdamW(
        groups[1]["params"], lr=lr_adamw, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=wd, fused=(device.type == "cuda"),
    )
    print(f"[train] Muon params={sum(p.numel() for p in groups[0]['params'])/1e6:.2f}M "
          f"AdamW params={sum(p.numel() for p in groups[1]['params'])/1e6:.2f}M", flush=True)
    if _POST_OPTIMIZER_HOOK is not None:
        _POST_OPTIMIZER_HOOK(model, opt_muon, opt_adamw)

    if train_cfg.get("compile", True):
        print("[train] torch.compile(model) ...", flush=True)
        model = torch.compile(model, dynamic=False, fullgraph=False)

    block_mask = block_diffusion_block_mask(cfg, device)
    if data_cfg.get("synthetic", False):
        train_ds = SyntheticTokenDataset(cfg.vocab_size, seq_len, seed=seed,
                                         length=max(4096, total_steps * global_batch + global_batch))
        print("[data] synthetic debug tokens", flush=True)
    else:
        data_dir = args.data_dir or data_cfg.get("shards_dir", "data/python_mix")
        train_ds = PackedTokenDataset(data_dir, seq_len=seq_len, seed=seed)
        print(f"[data] packed shards {data_dir} windows={len(train_ds):,}", flush=True)

    out_dir = args.out_dir or train_cfg.get("out_dir", "runs/passchain_3b")
    os.makedirs(out_dir, exist_ok=True)
    hf_repo_id = args.hf_repo_id or os.environ.get("HF_REPO_ID") or train_cfg.get("hf_repo_id")
    rng = np.random.default_rng(seed + 10_000)
    start_step = 0
    latest_path = os.path.join(out_dir, "ckpt_latest.pt")
    resume_path = args.resume_from
    if args.resume_from_hf:
        resume_path = download_latest(out_dir, hf_repo_id)
    if resume_path and os.path.exists(resume_path):
        start_step, _ = load_checkpoint(resume_path, model, opt_muon, opt_adamw, device)
        print(f"[train] resumed {resume_path} at step {start_step}", flush=True)

    save_every_steps = int(train_cfg.get("save_every_steps", 0) or 0)
    save_every_min = float(train_cfg.get("save_every_minutes", 30) or 0)
    last_save = time.time()
    log_interval = int(args.log_interval or train_cfg.get("log_interval", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    bias_gamma = float(cfg.bias_gamma)
    losses: List[float] = []
    t0 = time.time()

    for step in range(start_step, total_steps):
        cur_muon_lr = wsd_lr(step, total_steps, lr_muon, warmup_frac, decay_frac, phase)
        cur_adamw_lr = wsd_lr(step, total_steps, lr_adamw, warmup_frac, decay_frac, phase)
        for g in opt_muon.param_groups:
            g["lr"] = cur_muon_lr
        for g in opt_adamw.param_groups:
            g["lr"] = cur_adamw_lr

        micro = []
        n_masked_global = 0
        for m in range(accum):
            x0 = train_ds.get_batch(step * global_batch + m * micro_batch, micro_batch, wrap=True)
            b = build_sequence_batch(x0, rng, cfg)
            n_masked_global += int(b["mask"].sum())
            micro.append(batch_to_torch(b, device))

        counts_total = torch.zeros(
            (cfg.n_moe_layers, cfg.n_phases, cfg.counts_dim), dtype=torch.float32, device=device
        )
        ce_sum_g = torch.zeros((), device=device)
        z_sum_g = torch.zeros((), device=device)
        drop_max = 0.0
        for d in micro:
            n_m = int(d["mask"].sum().item())
            scale = float(n_m) / max(n_masked_global, 1)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = train_step(model, cfg, d, block_mask, ce_chunk=2048)
            (out["loss"] * scale).backward()
            ce_sum_g += out["ce_loss"].detach() * scale
            z_sum_g += out["z_loss"].detach() * scale
            counts_total += out["counts"].detach().float()
            drop_max = max(drop_max, float(out["drop"].detach()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt_muon.step()
        opt_adamw.step()
        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)
        if hasattr(model, "_orig_mod"):
            model._orig_mod.update_bias(list(counts_total), bias_gamma)
        else:
            model.update_bias(list(counts_total), bias_gamma)

        loss_val = float((ce_sum_g + z_sum_g).detach())
        losses.append(loss_val)
        if step % log_interval == 0 or step == total_steps - 1:
            elapsed = time.time() - t0
            print(f"[train] step {step+1}/{total_steps} lr_muon={cur_muon_lr:.3e} "
                  f"lr_adamw={cur_adamw_lr:.3e} ce={float(ce_sum_g):.4f} "
                  f"z={float(z_sum_g):.4f} loss={loss_val:.4f} drop={drop_max:.4f} "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

        do_save = (save_every_steps and (step + 1) % save_every_steps == 0)
        if save_every_min and (time.time() - last_save) >= save_every_min * 60:
            do_save = True
        if do_save:
            save_checkpoint(latest_path, model, opt_muon, opt_adamw, step + 1, cfg, rng,
                            include_optimizer=True, extra={"losses": losses[-20:]})
            last_save = time.time()
            print(f"[ckpt] saved {latest_path}", flush=True)
            upload_latest(latest_path, hf_repo_id, "ckpt_latest.pt")

    # final weights-only checkpoint + HF push
    final_path = os.path.join(out_dir, "ckpt_final.pt")
    save_checkpoint(final_path, model, opt_muon, opt_adamw, total_steps, cfg, rng,
                    include_optimizer=False, extra={"losses": losses[-20:]})
    upload_latest(final_path, hf_repo_id, "ckpt_final.pt")
    print(f"[ckpt] wrote {final_path}", flush=True)
    summary = {
        "config": args.config,
        "layout": cfg.layout,
        "phase": phase,
        "steps": total_steps,
        "start_step": start_step,
        "params_total": total_p,
        "params_active": active_p,
        "estimate": est,
        "fp8": fp8_enabled,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses": losses,
        "final": final_path,
        "latest": latest_path if os.path.exists(latest_path) else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", choices=["stable", "decay"], default=None)
    ap.add_argument("--steps", type=int, default=0, help="override target total steps")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--resume_from_hf", action="store_true")
    ap.add_argument("--hf_repo_id", default=None)
    ap.add_argument("--micro-batch", dest="micro_batch", type=int, default=0,
                    help="override config micro_batch; grad_accum is recomputed from global_batch")
    ap.add_argument("--log_interval", type=int, default=0)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
