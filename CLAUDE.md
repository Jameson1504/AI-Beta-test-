# Mythos — from-scratch GPT (PyTorch)

This repo contains three related sub-projects:

- `mythos/` — the main project: a modern decoder-only GPT (RMSNorm, RoPE,
  SwiGLU, KV-cache) with from-scratch BPE, DDP training, SFT, and a chat CLI.
- `ollama/` — a launcher + CLI bridge for comparing a local Ollama model with
  the Claude API. Ollama is NOT installed in web containers; don't claim these
  scripts run unless you actually ran them.
- `Anthropic/` — product notes on Claude models.

## Environment setup (fresh web containers are bare)

A SessionStart hook (`.claude/hooks/session-start.sh`) installs everything.
If imports still fail, run it manually — nothing works before this, not even
`import mythos`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU wheel, web containers have no GPU
pip install -r requirements.txt -r requirements-dev.txt
```

## Commands

- Tests: `python -m pytest -q` from the repo root (conftest.py handles the
  import path; covers `tests/test_model.py` and `tests/test_bpe_chat.py`,
  CPU-only, ~1–2 min).
- Toy training (CPU-verified): `python -m mythos.train --preset toy --data data/input.txt`
  — `--data` is required; there is no default.
- End-to-end demo: `bash scripts/demo.sh`.
- Training data is checked in (`data/input.txt`, `data/instructions.jsonl`) —
  no downloads needed.

## Repo conventions

- `main` is the integration branch. Do session work on `claude/<topic>`
  branches, but never leave finished work stranded — get it back into `main`.
  (Past sessions each produced an orphan branch that no later session could
  see; don't repeat that.)
- Check `PLAN.md` before starting new work: it tracks the roadmap. Phases 2–4
  (GPU-scale run, DPO + larger instruction set, eval harness + serving) are
  the open items — continue them rather than starting a parallel effort.
- When you add or move modules, update BOTH the roadmap status and the
  repository-layout blocks in `PLAN.md` and `README.md`.
- Never claim something "works" or is "verified" in docs or commit messages
  unless you ran it in this environment. If a dependency isn't available here
  (e.g. Ollama, a real GPU), say so explicitly instead.
- Use the project skills: `/verify-mythos` after changes to training/model
  code; `/continue-plan` to pick up the next roadmap item.
