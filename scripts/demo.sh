#!/usr/bin/env bash
# End-to-end demo: train a tiny model from scratch, then generate text.
# Runs on a CPU in a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Training a toy Mythos model on data/input.txt"
python -m mythos.train --preset toy --data data/input.txt --steps 2000

echo
echo "==> Sampling from the trained model"
python -m mythos.generate --ckpt checkpoints --prompt $'\n' \
  --max-new-tokens 300 --temperature 0.8 --top-k 50
