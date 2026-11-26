#!/usr/bin/env python
"""Build streaming uint16 shards for the Python-heavy PassChain mix.

Default mix: 70% permissive Python, 20% FineWeb-Edu, 10% OpenMathInstruct.
Documents are tokenized by a multiprocessing pool; every document gets EOS;
shards are contiguous and padded only by the final partial shard.

Usage:
  python scripts/prepare_data.py --out_dir /data/python_mix \
      --train_tokens 17000000000 --num_proc 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
VOCAB_SIZE = 49152
EOS_ID = 0
DEFAULT_HF_DATA_REPO = "LostStranger67/passchain-python-shards"

_WORKER_TOK = None
_WORKER_EOS = EOS_ID


@dataclass
class Source:
    name: str
    share: float
    dataset: str
    config: str | None = None
    data_files: str | None = None
    fields: Sequence[str] = ("text",)
    combine: Sequence[str] | None = None
    min_chars: int = 0


def make_sources(args) -> List[Source]:
    return [
        Source("python", 0.70, args.python_dataset, fields=("content", "text"), min_chars=200),
        Source("fineweb_edu", 0.20, args.fineweb_dataset, config=args.fineweb_config,
               fields=("text",), min_chars=200),
        Source("openmath", 0.10, args.math_dataset,
               data_files="correct_solutions/train.jsonl",
               fields=("generated_solution",), combine=("question", "generated_solution")),
    ]


def iter_docs(src: Source) -> Iterator[str]:
    from datasets import load_dataset

    kwargs: Dict[str, object] = {"split": "train", "streaming": True}
    if src.config:
        ds = load_dataset(src.dataset, src.config, **kwargs)
    elif src.data_files:
        ds = load_dataset(src.dataset, data_files=src.data_files, **kwargs)
    else:
        ds = load_dataset(src.dataset, **kwargs)
    for row in ds:
        if src.combine:
            parts = [str(row.get(f) or "") for f in src.combine]
            doc = "\n\n".join(p for p in parts if p)
        else:
            doc = ""
            for f in src.fields:
                if row.get(f):
                    doc = str(row[f])
                    break
        if doc and len(doc) >= src.min_chars:
            yield doc


def _init_worker(tokenizer: str, eos_id: int) -> None:
    global _WORKER_TOK, _WORKER_EOS
    from transformers import AutoTokenizer

    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    _WORKER_EOS = int(eos_id)


def _tokenize_batch(docs: List[str]) -> List[np.ndarray]:
    """Worker: tokenize one document batch and append EOS to every document."""
    enc = _WORKER_TOK(docs, add_special_tokens=False)["input_ids"]
    out: List[np.ndarray] = []
    for ids in enc:
        if ids:
            arr = np.empty(len(ids) + 1, dtype=np.uint16)
            arr[:-1] = np.asarray(ids, dtype=np.uint16)
            arr[-1] = _WORKER_EOS
            out.append(arr)
    return out


class BatchDocStream:
    """Round-robin document batches weighted by the requested domain shares."""

    def __init__(self, sources: List[Source], batch_docs: int = 512):
        self.sources = sources
        self.batch_docs = batch_docs
        self.iters = {s.name: iter_docs(s) for s in sources}
        self.done = set()

    def batches(self) -> Iterator[List[str]]:
        while len(self.done) < len(self.sources):
            for s in self.sources:
                if s.name in self.done:
                    continue
                n = max(1, int(round(self.batch_docs * s.share)))
                docs: List[str] = []
                try:
                    for _ in range(n):
                        docs.append(next(self.iters[s.name]))
                except StopIteration:
                    self.done.add(s.name)
                if docs:
                    yield docs


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, shard_tokens: int,
                 start_idx: int = 0, existing_files: Sequence[str] = (),
                 existing_tokens: int = 0):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.parts: List[np.ndarray] = []
        self.buffered = 0
        self.idx = start_idx
        self.files: List[str] = list(existing_files)
        self.total_tokens = int(existing_tokens)

    def write(self, arr: np.ndarray) -> None:
        self.parts.append(arr)
        self.buffered += arr.size
        while self.buffered >= self.shard_tokens:
            cat = np.concatenate(self.parts)
            head, tail = cat[: self.shard_tokens], cat[self.shard_tokens :]
            self._flush(head)
            self.parts = [tail] if tail.size else []
            self.buffered = tail.size

    def _flush(self, arr: np.ndarray) -> None:
        name = f"{self.prefix}_{self.idx:05d}.bin"
        path = os.path.join(self.out_dir, name)
        arr.tofile(path)
        self.files.append(name)
        self.total_tokens += int(arr.size)
        print(f"  wrote {name} ({arr.size:,} tokens)", flush=True)
        self.idx += 1

    def close(self) -> int:
        if self.buffered:
            self._flush(np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0])
        self.parts, self.buffered = [], 0
        return self.total_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_tokens", type=int, default=17_000_000_000)
    ap.add_argument("--shard_tokens", type=int, default=50_000_000)
    ap.add_argument("--batch_docs", type=int, default=512)
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--hf_repo_id", default=DEFAULT_HF_DATA_REPO)
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--python_dataset", default="codeparrot/codeparrot-clean",
                    help="replace with bigcode/the-stack-v2-dedup or starcoderdata if you have access")
    ap.add_argument("--fineweb_dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--fineweb_config", default="sample-10BT")
    ap.add_argument("--math_dataset", default="nvidia/OpenMathInstruct-1")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.vocab_size != VOCAB_SIZE:
        raise SystemExit(f"expected vocab {VOCAB_SIZE}, got {tok.vocab_size}")
    eos_id = int(tok.eos_token_id if tok.eos_token_id is not None else EOS_ID)
    sources = make_sources(args)
    total_share = sum(s.share for s in sources)
    if abs(total_share - 1.0) > 1e-9:
        raise SystemExit(f"mix shares must sum to 1.0, got {total_share}")

    existing = sorted(glob.glob(os.path.join(args.out_dir, "train_*.bin")))
    existing_tokens = sum(os.path.getsize(p) // 2 for p in existing)
    start_idx = len(existing)
    writer = ShardWriter(
        args.out_dir, "train", args.shard_tokens,
        start_idx=start_idx, existing_files=[os.path.basename(p) for p in existing],
        existing_tokens=existing_tokens,
    )
    stream = BatchDocStream(sources, args.batch_docs)
    written = existing_tokens
    if written >= args.train_tokens:
        print(f"[prepare] already have {written:,} tokens; nothing to do", flush=True)
        writer.close()
        return
    t0 = time.time()
    print(
        f"[prepare] tokenizer={args.tokenizer} vocab={tok.vocab_size} eos={eos_id} "
        f"num_proc={args.num_proc} target={args.train_tokens:,}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=max(1, args.num_proc),
        initializer=_init_worker,
        initargs=(args.tokenizer, eos_id),
    ) as pool:
        try:
            mapped = pool.map(_tokenize_batch, stream.batches(), chunksize=1,
                              buffersize=max(1, args.num_proc * 2))
        except TypeError:  # Python < 3.14 lacks buffersize
            mapped = pool.map(_tokenize_batch, stream.batches(), chunksize=1)
        for arrays in mapped:
            if not arrays:
                continue
            arr = np.concatenate(arrays)
            take = min(arr.size, args.train_tokens - written)
            if take <= 0:
                break
            writer.write(arr[:take])
            written += take
            if written // 1_000_000_000 != (written - take) // 1_000_000_000:
                print(f"[prepare] {written/1e9:.2f}B / {args.train_tokens/1e9:.2f}B tokens "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if written >= args.train_tokens:
                break

    total = writer.close()
    meta = {
        "train_files": writer.files,
        "val_files": {},
        "train_tokens": int(total),
        "val_tokens": 0,
        "vocab_size": int(tok.vocab_size),
        "eos_id": eos_id,
        "tokenizer": args.tokenizer,
        "sequence_packing": True,
        "mix": {s.name: s.share for s in sources},
        "shard_tokens": args.shard_tokens,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if not args.no_upload and args.hf_repo_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset",
                            private=False, exist_ok=True)
            print(f"[prepare] upload_large_folder -> {args.hf_repo_id}", flush=True)
            api.upload_large_folder(
                folder_path=args.out_dir,
                repo_id=args.hf_repo_id,
                repo_type="dataset",
            )
        except Exception as exc:
            print(f"[prepare] large-folder upload failed: {exc}", flush=True)
    print(f"[prepare] wrote {len(writer.files)} shards / {total:,} tokens -> {args.out_dir}")


if __name__ == "__main__":
    main()
