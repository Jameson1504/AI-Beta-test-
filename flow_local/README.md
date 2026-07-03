# flow_local

A private, local clone of [Wispr Flow](https://wisprflow.ai)'s dictation
experience — hold a hotkey, talk, get clean text typed into whatever field
has focus. Unlike Flow, **no audio or text ever leaves your machine**: there
is no account, no server, and (in the default configuration) no network
client anywhere in the pipeline.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the research behind this — what
Flow's cloud pipeline actually does, stage by stage, and how each stage maps
to a local equivalent here.

## Pipeline

```
global hotkey (pynput) → mic capture (sounddevice) → local Whisper ASR
(faster-whisper) → rule-based cleanup → keystroke/clipboard injection (pynput)
```

## Install

```bash
pip install -r flow_local/requirements.txt
```

`sounddevice` needs PortAudio installed at the OS level (`apt install
libportaudio2` on Debian/Ubuntu, bundled on macOS/Windows wheels). `pynput`
needs Accessibility permission on macOS and an X11 (not pure-Wayland)
session on Linux to inject keystrokes system-wide.

The first run of a given `--model-size` downloads that Whisper checkpoint
once via `faster-whisper`/`huggingface_hub` and caches it locally
(`~/.cache/huggingface`); every run after that is offline.

## Run

```bash
python -m flow_local.cli                       # defaults: base.en, CPU, push-to-talk on ctrl+alt+space
python -m flow_local.cli --model-size small.en --device cuda   # bigger/faster model on a local GPU
python -m flow_local.cli --print-config         # see the effective config without starting
```

Hold the hotkey, speak, release — the cleaned-up transcript is typed at your
cursor. Say "scratch that" to backspace the last utterance, "new line" /
"new paragraph" for line breaks.

## Config

All defaults live in `flow_local/config.py` (`FlowConfig`). Pass
`--config path/to/config.json` (see `FlowConfig.save`/`.load`) or override
individual fields on the CLI (`--hotkey`, `--mode`, `--model-size`,
`--device`, `--injection-mode`).

## Privacy guarantee, concretely

- The default cleanup pass (`cleanup.py`) is deterministic text processing —
  no model, no network.
- `tests/test_flow_local_no_network.py` statically checks that none of the
  always-on modules (`audio`, `transcribe`, `cleanup`, `inject`, `hotkey`,
  `app`, `cli`) import an HTTP/socket client.
- The one opt-in exception is `llm_cleanup.py` (`use_local_llm_cleanup =
  True` in config), for routing cleanup through a local model server (e.g.
  Ollama). It refuses to contact any host other than `127.0.0.1`/`localhost`,
  so turning it on still can't send anything off the machine.

## Limitations vs. Wispr Flow

- Cleanup is rule-based by default, not a fine-tuned LLM rewrite — it won't
  match Flow's tone-adaptive polish out of the box. Enable
  `use_local_llm_cleanup` with a local Ollama model if you want to trade
  latency/hardware for closer-to-Flow quality.
- Injection is keystroke/clipboard simulation, not the OS accessibility
  layer — it works in effectively all apps but can't detect the surrounding
  field's existing content the way Flow's "View Diff" can.
- No mobile app, no per-app tone profiles, no personalization/vocabulary
  learning yet.
