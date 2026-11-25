"""Modal B200 launcher for the PassChain 3B run (4-hour hard cap)."""

from __future__ import annotations

import os

import modal

DEFAULT_HF_REPO_ID = "LostStranger67/passchain-3b-python"
DEFAULT_DATA_REPO = "LostStranger67/chain-diffuser"
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


def ensure_data() -> None:
    """Populate /data/python_mix once, then reuse the Modal Volume forever.

    Preferred path: download prebuilt uint16 shards + meta.json from an HF
    *dataset* repo (default companion repo; override with HF_DATA_REPO_ID).
    Fallback: stream/tokenize the 70/20/10 mix on the first run.  Subsequent
    Modal runs see meta.json in the volume and start immediately.
    """
    import subprocess

    meta = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(meta):
        print(f"[modal] data cache hit: {DATA_DIR}", flush=True)
        return
    volume.reload()
    if os.path.exists(meta):
        print(f"[modal] data cache hit after reload: {DATA_DIR}", flush=True)
        return

    repo = os.environ.get("HF_DATA_REPO_ID", DEFAULT_DATA_REPO)
    try:
        from huggingface_hub import snapshot_download

        # Repo layouts we accept: shards at root, or a python_mix/ subfolder.
        for local_dir, patterns in (
            (DATA_DIR, ["*.bin", "meta.json"]),
            ("/data", ["python_mix/*"]),
        ):
            try:
                print(f"[modal] trying data snapshot {repo} -> {local_dir}", flush=True)
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=local_dir,
                    allow_patterns=patterns,
                )
            except Exception as exc:
                print(f"[modal] data snapshot attempt failed: {exc}", flush=True)
            if os.path.exists(meta):
                volume.commit()
                print(f"[modal] cached data snapshot at {DATA_DIR}", flush=True)
                return
    except Exception as exc:
        print(f"[modal] huggingface_hub data path unavailable: {exc}", flush=True)

    # No prebuilt data: stream/tokenize once.  The volume persists it.
    tokens = os.environ.get("PASSCHAIN_DATA_TOKENS", "17000000000")
    print(f"[modal] no prebuilt data; preparing {tokens} tokens on the volume ...", flush=True)
    subprocess.run(
        [
            "python",
            "scripts/prepare_data.py",
            "--out_dir",
            DATA_DIR,
            "--train_tokens",
            str(tokens),
        ],
        check=True,
    )
    volume.commit()
    print("[modal] data preparation complete and committed to volume", flush=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=14400,  # 4 hours, safely inside a $30 Starter credit
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
)
def train(phase: str = "stable", steps: int = 0, micro_batch: int = DEFAULT_MICRO_BATCH):
    import subprocess

    os.environ.setdefault("HF_REPO_ID", DEFAULT_HF_REPO_ID)
    os.chdir("/workspace/passchain")
    ensure_data()
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
