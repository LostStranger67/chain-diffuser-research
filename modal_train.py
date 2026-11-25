"""Modal B200 launcher for the PassChain 3B run (4-hour hard cap)."""

from __future__ import annotations

import glob
import os
import time

import modal

DEFAULT_HF_REPO_ID = "LostStranger67/passchain-3b-python"
DEFAULT_DATA_REPO = "LostStranger67/passchain-python-shards"
DATA_DIR = "/data/python_mix"
DEFAULT_MICRO_BATCH = 16

app = modal.App("passchain-3b-python")

# Mount this checkout into the image.  When running from CI, replace
# add_local_dir with an image.run_commands("git clone ...") step.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime",
    )
    # Ubuntu 24.04 images mark the system Python as externally managed (PEP 668).
    # This env var applies to Modal's build-time pip invocations.
    .env({
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    .apt_install("git")
    .pip_install(
        "torch==2.11.0",
        "torchao==0.18.0",
        "datasets",
        "huggingface_hub",
        "numpy",
        "pyyaml",
        "transformers",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace/passchain",
        ignore=["data/**", "runs/**", "**/*.pt", "**/*.bin", "**/*.log", ".git/**"],
    )
)

volume = modal.Volume.from_name("passchain-data", create_if_missing=True)


hf_secret = modal.Secret.from_name("huggingface-secret")


def _train_shards() -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "train_*.bin")))


def _sync_meta() -> None:
    """Make meta.json describe only the shards currently cached locally."""
    import json

    train_files = _train_shards()
    if len(train_files) < 2:
        raise RuntimeError(
            f"need at least 2 train_*.bin shards in {DATA_DIR}, found {len(train_files)}"
        )
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_tokens = sum(os.path.getsize(p) // 2 for p in train_files)
    val_tokens = os.path.getsize(val_path) // 2 if os.path.exists(val_path) else 0
    meta = {
        "train_files": [os.path.basename(p) for p in train_files],
        "val_files": {"val": "val.bin"} if val_tokens else {},
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "vocab_size": 49152,
        "eos_id": 0,
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
        "sequence_packing": True,
        "mix": {"python": 0.70, "fineweb_edu": 0.20, "openmath": 0.10},
        "shard_tokens": 50_000_000,
    }
    path = os.path.join(DATA_DIR, "meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)
    print(
        f"[modal] meta: {len(train_files)} train shards, {train_tokens/1e9:.2f}B train, "
        f"{val_tokens/1e6:.1f}M val",
        flush=True,
    )


def _ensure_val() -> None:
    """Auto-split the last train shard into val.bin when no val exists."""
    import numpy as np

    val_path = os.path.join(DATA_DIR, "val.bin")
    if os.path.exists(val_path):
        return
    shards = _train_shards()
    if len(shards) < 2:
        return
    src = shards[-1]
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    n = int(len(arr))
    if n < 2_000_000:
        raise RuntimeError(f"last shard too small to split: {src} ({n} tokens)")
    val_tokens = min(2_000_000, max(100_000, n // 100))
    head = np.asarray(arr[: n - val_tokens]).copy()
    tail = np.asarray(arr[n - val_tokens :]).copy()
    del arr
    head_tmp, val_tmp = src + ".tmp", val_path + ".tmp"
    head.tofile(head_tmp)
    tail.tofile(val_tmp)
    os.replace(head_tmp, src)
    os.replace(val_tmp, val_path)
    print(
        f"[modal] auto-split {os.path.basename(src)} -> val.bin "
        f"({val_tokens/1e6:.2f}M held-out tokens)",
        flush=True,
    )


def _download_shards() -> None:
    """Download the public shard dataset once into the Modal Volume."""
    from huggingface_hub import snapshot_download

    repo = os.environ.get("HF_DATA_REPO_ID", DEFAULT_DATA_REPO)
    for local_dir, patterns in (
        ("/data", ["python_mix/*"]),
        (DATA_DIR, ["*.bin", "meta.json"]),
    ):
        try:
            print(f"[modal] downloading shards {repo} -> {local_dir}", flush=True)
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=patterns,
            )
        except Exception as exc:
            print(f"[modal] shard download attempt failed: {exc}", flush=True)
        if len(_train_shards()) >= 2:
            break


def ensure_train_data() -> None:
    """Fast-start: use >=2 cached shards; never tokenize on the B200."""
    volume.reload()
    shards = _train_shards()
    print(f"[modal] cached train shards: {len(shards)}", flush=True)
    if len(shards) == 0:
        _download_shards()
        shards = _train_shards()
        print(f"[modal] shards after download: {len(shards)}", flush=True)
        if len(shards) >= 2:
            volume.commit()
    if len(shards) < 2:
        raise RuntimeError(
            "Need at least 2 train_*.bin shards. Run the CPU prep job first: "
            "modal run modal_train.py::prepare_data --train-tokens 17000000000"
        )
    _ensure_val()
    _sync_meta()
    volume.commit()
    print(f"[modal] fast-starting B200 with {len(_train_shards())} train shards", flush=True)


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=14400,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def prepare_data(train_tokens: int = 17_000_000_000, num_proc: int = 8, force: bool = False):
    """CPU-only tokenization: ~$0.38/hr instead of B200 pricing.

    Tokenizes with ``num_proc`` workers, uploads every completed shard to the
    public HF dataset repo, and commits the Modal Volume at each 1B-token mark.
    The B200 job can start as soon as two train shards exist.
    """
    import subprocess

    volume.reload()
    existing_tokens = sum(os.path.getsize(p) // 2 for p in _train_shards())
    if not force and existing_tokens >= train_tokens * 0.99:
        print(f"[prep] already have {existing_tokens/1e9:.2f}B tokens; skipping", flush=True)
        return
    repo = os.environ.get("HF_DATA_REPO_ID", DEFAULT_DATA_REPO)
    os.chdir("/workspace/passchain")
    cmd = [
        "python",
        "scripts/prepare_data.py",
        "--out_dir",
        DATA_DIR,
        "--train_tokens",
        str(train_tokens),
        "--num_proc",
        str(num_proc),
        "--hf_repo_id",
        repo,
    ]
    print("[prep] running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    committed = 0
    while proc.poll() is None:
        total = sum(os.path.getsize(p) // 2 for p in _train_shards())
        if total - committed >= 1_000_000_000:
            volume.commit()
            committed = total
            print(f"[prep] volume.commit at {total/1e9:.2f}B tokens", flush=True)
        time.sleep(30)
    if proc.wait() != 0:
        raise RuntimeError("CPU data preparation failed")
    volume.commit()
    print("[prep] data prep complete; volume committed", flush=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=14400,  # 4 hours, safely inside a $30 Starter credit
    secrets=[hf_secret],
    volumes={"/data": volume},
)
def train(phase: str = "stable", steps: int = 0, micro_batch: int = DEFAULT_MICRO_BATCH):
    import subprocess

    os.environ.setdefault("HF_REPO_ID", DEFAULT_HF_REPO_ID)
    os.chdir("/workspace/passchain")
    ensure_train_data()
    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        "configs/passchain_3b_python.yaml",
        "--phase",
        phase,
        "--resume_from_hf",
        "--data_dir",
        DATA_DIR,
        "--micro-batch",
        str(micro_batch),
    ]
    if steps:
        cmd += ["--steps", str(steps)]
    print("[modal] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("[modal] finished; ckpt_final.pt was pushed to Hugging Face", flush=True)


@app.local_entrypoint()
def main(phase: str = "stable", steps: int = 0, micro_batch: int = DEFAULT_MICRO_BATCH):
    train.remote(phase, steps, micro_batch)
