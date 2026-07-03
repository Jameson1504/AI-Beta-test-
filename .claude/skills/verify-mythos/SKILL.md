---
name: verify-mythos
description: Verify the Mythos pipeline end-to-end - install deps if missing, run the test suite, then a short real training run asserting loss decreases. Use after changing anything under mythos/ or before claiming training/generation works.
---

# Verify Mythos

Run these steps in order. Stop and report at the first failure — do not mark
work as verified if any step fails.

1. **Deps** — if `python -c "import torch"` fails, run
   `.claude/hooks/session-start.sh` (or the pip commands from CLAUDE.md).
2. **Tests** — `python -m pytest -q` from the repo root. All tests must pass
   (they include KV-cache-vs-full-forward equivalence and single-batch
   overfitting).
3. **Real training smoke** — a short run must show loss clearly decreasing:
   ```bash
   python -m mythos.train --preset toy --data data/input.txt --steps 200 --out /tmp/verify-ckpt
   ```
   Check the logged loss: the final value must be well below the initial one
   (initial is ~5.5 on the included corpus).
4. **Generation** — sample from the checkpoint and confirm it emits text
   without crashing:
   ```bash
   python -m mythos.generate --ckpt /tmp/verify-ckpt --prompt "ROMEO:" --max-new-tokens 50
   ```
5. Report the actual numbers observed (initial/final loss), not just
   "it works". Clean up `/tmp/verify-ckpt`.
