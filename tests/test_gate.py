from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passchain.config import DiffConfig, LAYOUTS
from passchain.masking import batch_to_torch, block_diffusion_block_mask, build_sequence_batch
from passchain.model import DiffusionTransformer, MoEFFN
from passchain.optim import Muon, assert_optimizer_coverage, build_param_groups, wsd_lr, zeropower_via_newtonschulz5
from passchain.precision import _fp8_module_filter, enable_fp8, fp8_available
from passchain.training import load_yaml_config, masked_ce, train_step

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_cfg(**overrides):
    base = dict(
        vocab_size=512, d_model=64, n_layers=4, n_heads=2, n_kv_heads=1,
        head_dim=32, d_ff_dense=128, d_ff_expert=32, top_k=4,
        layout="passchain40", block_size=8, n_blocks=8, qk_norm=True,
        z_loss_weight=1e-4, moe_impl="padded",
    )
    base.update(overrides)
    return DiffConfig(**base)


def make_batch(cfg, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len), dtype=np.int64)
    b = build_sequence_batch(x0, rng, cfg)
    return batch_to_torch(b, torch.device(DEV))


def test_3b_parameter_counts():
    cfg, train, data = load_yaml_config(os.path.join(ROOT, "configs/passchain_3b_python.yaml"))
    est = cfg.estimate_params()
    assert cfg.layout == "passchain_3b"
    assert cfg.n_layers == 16 and cfg.n_moe_layers == 15
    assert cfg.d_model == 1536 and cfg.n_heads == 12 and cfg.n_kv_heads == 4
    assert cfg.routed_experts == 600
    assert 2.85e9 <= est["total"] <= 3.05e9, est["total"]
    assert 380e6 <= est["routed_active"] <= 420e6, est["routed_active"]
    assert cfg.window_starts == (0, 8, 16, 24)
    assert cfg.n_blocks * cfg.block_size == 2048


def test_passchain_windows_and_qknorm():
    cfg = tiny_cfg()
    m = DiffusionTransformer(cfg).to(DEV)
    moe = m.moe_layers()[0]
    for phase, start in enumerate(cfg.window_starts):
        wg, wd = moe.expert_slices(m.pool_wg, m.pool_wd, phase)
        assert wg.shape[0] == 16 and wd.shape[0] == 16
        assert torch.equal(wg, m.pool_wg[0, start:start + 16])
    assert m.layers[0].attn.q_norm is not None
    assert m.layers[0].attn.k_norm is not None
    d = make_batch(cfg)
    bm = block_diffusion_block_mask(cfg, torch.device(DEV))
    with torch.no_grad():
        out = m(d["ids"], d["phase_ids"], d["order"], d["inv_order"], d["phase_sizes"], bm)
    assert out["hidden"].shape == (2, cfg.context_len, cfg.d_model)
    assert math.isfinite(float(out["router_z_loss"]))
    assert float(out["router_z_loss"]) > 0


def test_z_loss_in_train_step():
    cfg = tiny_cfg()
    m = DiffusionTransformer(cfg).to(DEV)
    d = make_batch(cfg)
    bm = block_diffusion_block_mask(cfg, torch.device(DEV))
    out = train_step(m, cfg, d, bm, ce_chunk=64)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert float(out["z_loss"].detach()) > 0
    assert float(out["router_z"].detach()) > 0 and float(out["output_z"].detach()) > 0


def test_muon_adamw_split():
    cfg = tiny_cfg()
    m = DiffusionTransformer(cfg)
    groups = build_param_groups(m, lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.1)
    muon, adamw = groups[0]["params"], groups[1]["params"]
    all_params = {id(p) for p in m.parameters() if p.requires_grad}
    assert {id(p) for p in muon}.isdisjoint({id(p) for p in adamw})
    assert {id(p) for p in muon} | {id(p) for p in adamw} == all_params
    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    muon_names = {n for n, p in m.named_parameters() if id(p) in muon_ids}
    adamw_names = {n for n, p in m.named_parameters() if id(p) in adamw_ids}
    assert "embed.weight" in adamw_names
    assert any("router" in n for n in adamw_names)
    assert any("q_proj.weight" in n for n in muon_names)
    assert any("pool_wg" in n for n in muon_names)
    # A single Muon step must keep a hidden weight finite.
    before = m.layers[1].ffn.shared.gate_up.weight.detach().clone()
    loss = (m.layers[1].ffn.shared.gate_up.weight.float() ** 2).mean()
    loss.backward()
    opt = Muon(muon, lr=0.01)
    opt.step()
    after = m.layers[1].ffn.shared.gate_up.weight.detach()
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)


def test_swapped_linear_coverage():
    """A swapped/subclassed Float8Linear weight must never be dropped."""
    import torch.nn as nn

    class SwappedLinear(nn.Linear):
        pass

    class Float8Like(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight.detach().clone())

    cfg = tiny_cfg()
    m = DiffusionTransformer(cfg)
    old = m.layers[0].ffn.body.gate_up
    m.layers[0].ffn.body.gate_up = SwappedLinear(
        old.in_features, old.out_features, bias=False
    )
    m.layers[0].ffn.body.gate_up.weight = old.weight
    old_down = m.layers[1].ffn.shared.down
    m.layers[1].ffn.shared.down = Float8Like(old_down.weight)

    groups = build_param_groups(m, lr_muon=0.02, lr_adamw=3e-4)
    muon, adamw = groups[0]["params"], groups[1]["params"]
    all_trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert {id(p) for p in muon}.isdisjoint({id(p) for p in adamw})
    assert {id(p) for p in muon} | {id(p) for p in adamw} == all_trainable

    opt_muon = Muon(muon, lr=0.01)
    opt_adamw = torch.optim.AdamW(adamw, lr=3e-4)
    stats = assert_optimizer_coverage(m, [opt_muon, opt_adamw], "swapped")
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert stats["total_numel"] == total
    assert sum(p.numel() for p in muon) + sum(p.numel() for p in adamw) == total


def test_fp8_module_filter():
    import torch.nn as nn

    assert _fp8_module_filter(nn.Linear(16, 32), "layer.q_proj") is True
    assert _fp8_module_filter(nn.Linear(15, 32), "layer.q_proj") is False
    assert _fp8_module_filter(nn.Linear(16, 32), "lm_head") is False
    assert _fp8_module_filter(nn.Linear(16, 32), "layers.1.ffn.router_w") is False


def test_chunked_ce_matches_full():
    import torch.nn.functional as F

    torch.manual_seed(7)
    B, S, D, V = 2, 8, 16, 64
    h = torch.randn(B, S, D, dtype=torch.bfloat16)
    w = torch.randn(V, D, dtype=torch.bfloat16)
    tgt = torch.randint(0, V, (B, S))
    mask = torch.rand(B, S) < 0.6
    mask[0, 0] = True
    t = torch.rand(B, S) * 0.9 + 0.05
    phase = torch.zeros(B, S, dtype=torch.long)

    def reference(chunk):
        # Same math as masked_ce, explicitly materializing one chunk at a time.
        hf = h.reshape(-1, D)
        tf = tgt.reshape(-1)
        mf = mask.reshape(-1).float()
        tt = t.reshape(-1).float().clamp_min(0.05)
        ph = phase.reshape(-1)
        ce_sum = hf.new_zeros((), dtype=torch.float32)
        elbo_sum = hf.new_zeros((), dtype=torch.float32)
        z2 = hf.new_zeros((), dtype=torch.float32)
        for s0 in range(0, hf.shape[0], chunk):
            s1 = min(s0 + chunk, hf.shape[0])
            logits = F.linear(hf[s0:s1], w)
            ce = F.cross_entropy(logits.float(), tf[s0:s1], reduction="none")
            ce_m = ce * mf[s0:s1]
            ce_sum = ce_sum + ce_m.sum()
            elbo_sum = elbo_sum + (ce_m / tt[s0:s1]).sum()
            z2 = z2 + torch.logsumexp(logits.float(), dim=-1).pow(2).sum()
        return ce_sum, elbo_sum, z2

    full = masked_ce(h, w, tgt, mask, t, phase, chunk=128)
    small = masked_ce(h, w, tgt, mask, t, phase, chunk=3)
    ref_ce, ref_elbo, ref_z2 = reference(128)
    for got, ref, name in (
        (full[1], ref_ce, "full ce_sum"),
        (full[2], ref_elbo, "full elbo_sum"),
        (full[7], ref_z2, "full z2"),
        (small[1], ref_ce, "small ce_sum"),
        (small[2], ref_elbo, "small elbo_sum"),
        (small[7], ref_z2, "small z2"),
    ):
        assert torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2), (name, got, ref)
    assert torch.allclose(full[0].float(), small[0].float(), rtol=2e-2, atol=2e-2)


def test_wsd_schedule_math():
    total, peak = 1000, 1.0
    warmup = 20
    assert abs(wsd_lr(0, total, peak, 0.02, 0.15, "stable") - peak / warmup) < 1e-9
    assert abs(wsd_lr(warmup - 1, total, peak, 0.02, 0.15, "stable") - peak) < 1e-9
    assert abs(wsd_lr(500, total, peak, 0.02, 0.15, "stable") - peak) < 1e-9
    decay_start = int(round(total * 0.85))
    assert abs(wsd_lr(decay_start - 1, total, peak, 0.02, 0.15, "decay") - peak) < 1e-9
    assert abs(wsd_lr(total - 1, total, peak, 0.02, 0.15, "decay")) < 1e-9
    vals = [wsd_lr(s, total, peak, 0.02, 0.15, "decay") for s in range(decay_start, total)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
    # A 5-step benchmark must still use the configured stable-phase warmup.
    assert abs(wsd_lr(0, 5, peak, warmup_steps=100) - peak / 100) < 1e-9
    assert wsd_lr(0, 5, peak, warmup_steps=100) > 0
    with pytest.raises(ValueError):
        wsd_lr(0, total, peak, phase="bogus")


def test_fp8_fallback_on_rtx3090():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
        assert fp8_available() is False
        m = DiffusionTransformer(tiny_cfg())
        assert enable_fp8(m, enabled=True) is False


def test_orthogonalization_shape():
    g = torch.randn(8, 5)
    x = zeropower_via_newtonschulz5(g, steps=3)
    assert x.shape == g.shape and torch.isfinite(x).all()
    assert x.dtype == g.dtype


def test_muon_batches_4d_expert_tensor():
    # 2 experts x 3 x 8 x 4 simulates a batched expert-pool parameter.
    p = torch.nn.Parameter(torch.randn(2, 3, 8, 4))
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    Muon([p], lr=0.01).step()
    assert p.shape == before.shape
    assert torch.isfinite(p).all()
    assert not torch.equal(p, before)
