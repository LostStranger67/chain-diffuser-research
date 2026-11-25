"""Modal B200 launcher for the PassChain 3B run (4-hour hard cap)."""

from __future__ import annotations

import os

import modal

DEFAULT_HF_REPO_ID = "LostStranger67/passchain-3b-python"

app = modal.App("passchain-3b-python")

# Mount this checkout into the image.  When running from CI, replace
# add_local_dir with an image.run_commands("git clone ...") step.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
    )
    .apt_install("git")
    .pip_install(
        "torchao",
        "huggingface_hub",
        "datasets",
        "transformers",
        "pyyaml",
        "numpy",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace/passchain",
        ignore=["data/**", "runs/**", "**/*.pt", "**/*.bin", "**/*.log", ".git/**"],
    )
)

volume = modal.Volume.from_name("passchain-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=14400,  # 4 hours, safely inside a $30 Starter credit
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
)
def train(phase: str = "stable", steps: int = 0):
    import subprocess

    os.environ.setdefault("HF_REPO_ID", DEFAULT_HF_REPO_ID)
    os.chdir("/workspace/passchain")
    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        "configs/passchain_3b_python.yaml",
        "--phase",
        phase,
        "--resume_from_hf",
        "--data_dir",
        "/data/python_mix",
    ]
    if steps:
        cmd += ["--steps", str(steps)]
    print("[modal] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("[modal] finished; ckpt_final.pt was pushed to Hugging Face", flush=True)


@app.local_entrypoint()
def main(phase: str = "stable", steps: int = 0):
    train.remote(phase, steps)
