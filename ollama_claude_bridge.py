#!/usr/bin/env python3
"""
Bridge between Ollama's local API and Anthropic's Claude API.
Sends a prompt to both and compares responses, or routes to either backend.
"""

import os
import json
import argparse
import urllib.request
import urllib.error

OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def _post(url: str, payload: dict, headers: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    result = _post(url, payload, headers)
    return result.get("response", "").strip()


def ask_claude(prompt: str, model: str = CLAUDE_MODEL) -> str:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    result = _post(url, payload, headers)
    return result["content"][0]["text"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Query Ollama, Claude, or both with a prompt."
    )
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude", "both"],
        default="ollama",
        help="Which backend to use (default: ollama)",
    )
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--claude-model", default=CLAUDE_MODEL)
    args = parser.parse_args()

    if args.backend in ("ollama", "both"):
        print(f"=== Ollama ({args.ollama_model}) ===")
        try:
            print(ask_ollama(args.prompt, model=args.ollama_model))
        except Exception as e:
            print(f"Ollama error: {e}")

    if args.backend in ("claude", "both"):
        print(f"\n=== Claude ({args.claude_model}) ===")
        try:
            print(ask_claude(args.prompt, model=args.claude_model))
        except Exception as e:
            print(f"Claude error: {e}")


if __name__ == "__main__":
    main()
