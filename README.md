# AI Beta Test — Ollama + Claude Bridge

Run a local Ollama model and optionally bridge it to Anthropic's Claude API.

## Requirements

- [Ollama](https://ollama.com/download) installed and on your `$PATH`
- Python 3.8+ (for the bridge script)
- An Anthropic API key (only needed for Claude backend)

## Quick start

```bash
# 1. Copy and fill in your environment variables
cp .env.example .env

# 2. Launch Ollama and pull/run a model (default: llama3.2)
chmod +x launch.sh
./launch.sh

# Or specify a different model
./launch.sh mistral
```

## Bridge script

Query Ollama, Claude, or both from the command line:

```bash
# Ask Ollama only (default)
python3 ollama_claude_bridge.py "What is the capital of France?"

# Ask Claude only
ANTHROPIC_API_KEY=sk-... python3 ollama_claude_bridge.py \
  --backend claude "Explain quantum entanglement."

# Side-by-side comparison
ANTHROPIC_API_KEY=sk-... python3 ollama_claude_bridge.py \
  --backend both "Write a haiku about code."
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Claude backend |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server address |
| `OLLAMA_PORT` | `11434` | Ollama server port (used by launch.sh) |
