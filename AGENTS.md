# AGENTS.md — collaboration rules

These rules apply to every AI agent working in this repository.

1. **Never edit a DONE config.** `EXPERIMENTS.md` is the source of truth. A
   config marked `DONE` is immutable; create a new versioned config instead.
2. **Always run the verification gate before any GPU job.**
   `pytest -q` must pass, and the debug smoke test must pass, before launching
   a 3B run.
3. **Never commit checkpoints, weights, or data shards.** The repository is
   code + configs + small reports only. `*.pt`, `*.bin`, `*.safetensors`,
   `data/`, `.env`, and logs are ignored and must stay out of Git.
4. **Use rolling HF checkpoint sync.** Every training job must save
   `ckpt_latest.pt` atomically and upload it to the private HF repo
   (`LostStranger67/chain-diffuser`, env `HF_REPO_ID`) on the configured interval. Resume with
   `--resume_from_hf`.
5. **Report reproducible numbers.** Record the config hash, commit SHA, token
   count, hardware, and wall time in `EXPERIMENTS.md`; do not fabricate runs.
6. **Do not change another agent's RUNNING job.** Coordinate through
   `EXPERIMENTS.md` (TODO → RUNNING → DONE) before taking a GPU slot.
