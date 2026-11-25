"""Optional private Hugging Face rolling-checkpoint sync."""

from __future__ import annotations

import os
from typing import Optional


def repo_id(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("HF_REPO_ID")


def download_latest(dest_dir: str, hf_repo_id: Optional[str] = None,
                    filename: str = "ckpt_latest.pt") -> Optional[str]:
    rid = repo_id(hf_repo_id)
    if not rid:
        print("[hf] HF_REPO_ID not set; skipping download", flush=True)
        return None
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"[hf] huggingface_hub unavailable ({exc}); skipping download", flush=True)
        return None
    os.makedirs(dest_dir, exist_ok=True)
    try:
        path = hf_hub_download(repo_id=rid, filename=filename, local_dir=dest_dir)
        print(f"[hf] downloaded {filename} from {rid} -> {path}", flush=True)
        return path
    except Exception as exc:
        print(f"[hf] download failed ({exc}); starting from scratch", flush=True)
        return None


def upload_latest(path: str, hf_repo_id: Optional[str] = None,
                  filename: str = "ckpt_latest.pt") -> bool:
    rid = repo_id(hf_repo_id)
    if not rid:
        return False
    try:
        from huggingface_hub import upload_file
    except Exception as exc:
        print(f"[hf] huggingface_hub unavailable ({exc}); checkpoint stays local", flush=True)
        return False
    try:
        upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=rid,
            repo_type="model",
            commit_message=f"rolling checkpoint {os.path.basename(path)}",
        )
        print(f"[hf] uploaded {filename} -> {rid}", flush=True)
        return True
    except Exception as exc:
        print(f"[hf] upload failed ({exc}); checkpoint stays local", flush=True)
        return False


__all__ = ["download_latest", "upload_latest", "repo_id"]
