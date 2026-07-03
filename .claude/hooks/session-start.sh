#!/bin/bash
set -euo pipefail

# Bootstrap for Claude Code on the web: fresh containers have no ML stack,
# and nothing in this repo runs (not even `import mythos`) without torch.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Web containers are CPU-only; prefer the small CPU wheel, but some network
# policies block download.pytorch.org — fall back to PyPI's default build.
# Idempotent: skipped entirely when torch is already present.
if ! python -c "import torch" 2>/dev/null; then
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
    || pip install --quiet "torch>=2.1,<3"
fi

pip install --quiet -r requirements.txt -r requirements-dev.txt
