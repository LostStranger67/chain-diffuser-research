#!/usr/bin/env python
"""Build streaming uint16 shards for the Python-heavy PassChain mix.

Default mix: 70% permissive Python, 20% FineWeb-Edu, 10% OpenMathInstruct.
Every document gets an EOS token; the writer packs tokens contiguously so the
training loader can read exact 2048-token windows with no padding.

Usage:
  python scripts/prepare_data.py --out_dir /data/python_mix \
      --train_tokens 17000000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
VOCAB_SIZE = 49152
EOS_ID = 0


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


class DomainStream:
    def __init__(self, src: Source, tok, eos_id: int, batch_docs: int = 256):
        self.src, self.tok, self.eos_id = src, tok, eos_id
        self.batch_docs = batch_docs
        self.docs = iter_docs(src)
        self.buf = np.empty(0, dtype=np.uint16)
        self.done = False
        self.emitted = 0
        self.n_docs = 0

    def _refill(self) -> None:
        if self.done:
            return
        batch: List[str] = []
        try:
            for _ in range(self.batch_docs):
                batch.append(next(self.docs))
        except StopIteration:
            self.done = True
        if not batch:
            return
        enc = self.tok(batch, add_special_tokens=False)["input_ids"]
        parts: List[np.ndarray] = []
        for ids in enc:
            if ids:
                parts.append(np.asarray(ids, dtype=np.uint16))
                parts.append(np.asarray([self.eos_id], dtype=np.uint16))
        if parts:
            self.buf = np.concatenate([self.buf] + parts)
        self.n_docs += len(batch)

    def take(self, n: int) -> np.ndarray:
        while self.buf.size < n and not self.done:
            self._refill()
        if self.buf.size < n:
            raise RuntimeError(
                f"{self.src.name} exhausted after {self.emitted:,} tokens; requested {n:,}"
            )
        out = self.buf[:n]
        self.buf = self.buf[n:]
        self.emitted += n
        return out


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, shard_tokens: int):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.parts: List[np.ndarray] = []
        self.buffered = 0
        self.idx = 0
        self.files: List[str] = []
        self.total_tokens = 0

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
    ap.add_argument("--block_tokens", type=int, default=100_000)
    ap.add_argument("--tokenizer", default=TOKENIZER)
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
    streams = {s.name: DomainStream(s, tok, eos_id) for s in sources}
    writer = ShardWriter(args.out_dir, "train", args.shard_tokens)
    written = 0
    t0 = time.time()
    print(f"[prepare] tokenizer={args.tokenizer} vocab={tok.vocab_size} eos={eos_id}")
    while written < args.train_tokens:
        need = args.train_tokens - written
        block = min(args.block_tokens, need)
        parts = []
        for s in sources:
            n = min(int(round(block * s.share)), args.train_tokens - written - sum(p.size for p in parts))
            if n > 0:
                parts.append(streams[s.name].take(n))
        if not parts:
            break
        arr = np.concatenate(parts)
        writer.write(arr)
        written += arr.size
        if written // 1_000_000_000 != (written - arr.size) // 1_000_000_000:
            print(f"[prepare] {written/1e9:.2f}B / {args.train_tokens/1e9:.2f}B tokens "
                  f"({time.time()-t0:.0f}s)", flush=True)
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
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] wrote {len(writer.files)} shards / {total:,} tokens -> {args.out_dir}")


if __name__ == "__main__":
    main()
