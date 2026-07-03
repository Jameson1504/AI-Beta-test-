"""Optional, opt-in cleanup pass through a local LLM (e.g. Ollama) for users
who want closer-to-Flow rewrite quality and have the hardware for it.

Disabled by default (`FlowConfig.use_local_llm_cleanup = False`). When
enabled, this talks to a model server on loopback only — it refuses to
contact anything but 127.0.0.1/localhost, so enabling it still can't send
audio or text off the machine.
"""

from __future__ import annotations

import json
import urllib.request

_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
_PROMPT_TEMPLATE = (
    "Rewrite the following raw speech-to-text transcript as clean, "
    "well-punctuated prose. Keep the meaning exactly the same, remove "
    "filler words and false starts, and output only the rewritten text "
    "with no commentary.\n\nTranscript: {text}"
)


def refine_with_local_llm(
    text: str,
    model: str,
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout: float = 10.0,
) -> str:
    """POSTs to a local Ollama-compatible `/api/generate` endpoint. Raises
    ValueError if `host` isn't loopback, so this can never be pointed at a
    remote server by config alone."""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"refuse to contact non-loopback host {host!r}: "
            "local LLM cleanup must stay on this machine"
        )

    payload = json.dumps(
        {
            "model": model,
            "prompt": _PROMPT_TEMPLATE.format(text=text),
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body.get("response", text).strip()
