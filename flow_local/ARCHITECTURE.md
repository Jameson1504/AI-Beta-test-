# Wispr Flow: architecture research, and the local-only design it informs

## What Wispr Flow actually does

Wispr Flow ("Flow") is a commercial voice-dictation app (macOS, Windows, iOS,
Android) that lets you talk into any text field and get back clean, formatted
text — not a raw transcript. Public sources describe a five-stage pipeline:

1. **Activation** — a global hotkey (push-to-talk or toggle) or, in
   hands-free mode, voice-activity detection starts capture. Hotkeys and even
   mouse buttons are user-bindable to actions: dictate, Command Mode,
   Transforms, paste-last-transcript, cancel.
2. **Audio capture** — the client streams mic audio in small chunks rather
   than waiting for silence, so transcription can start before you stop
   talking.
3. **ASR (speech → raw text)** — audio is sent to Wispr's cloud, where a
   Whisper-family model (served on Baseten's GPU infra) turns it into a raw
   transcript, filler words and false starts included.
4. **LLM cleanup (raw → "what you'd have typed")** — the raw transcript is
   passed through a fine-tuned Llama-based model that strips filler words,
   fixes punctuation/capitalization, repairs false starts, and adapts tone to
   the active app (an email reads differently than a Slack message or a code
   comment). Wispr states the full pipeline — ASR through Llama cleanup —
   runs end-to-end in under ~700ms.
5. **Injection** — the finished text is written into whatever field has
   focus, primarily through the OS accessibility layer (macOS Accessibility
   API / Windows UI Automation / Android AccessibilityService), with a
   clipboard-paste fallback for apps that don't expose it. This is what makes
   it work "in any app" instead of one integration at a time.

Two knobs affect data handling but **not** the architecture: *Cloud Sync*
(whether transcripts are stored after processing) and *Privacy Mode* (process
audio in real time and discard it immediately). Even with both maximally
strict, **audio still leaves the device** — Flow has no offline mode. That's
the specific gap this project closes.

Sources: [Wispr Flow — technical challenges](https://wisprflow.ai/post/technical-challenges) ·
[Wispr Flow — data controls](https://wisprflow.ai/data-controls) ·
[Wispr Flow — privacy mode & data retention](https://docs.wisprflow.ai/articles/6274675613-privacy-mode-data-retention) ·
[Wispr Flow on Baseten](https://www.baseten.co/resources/customers/wispr-flow/) ·
[Wispr Flow — keyboard & hotkeys](https://docs.wisprflow.ai/articles/2612050838-supported-unsupported-keyboard-hotkey-shortcuts) ·
[Wispr Flow — Wikipedia](https://en.wikipedia.org/wiki/Wispr_Flow)

## Mapping each stage to a fully local substitute

`flow_local/` reimplements the same five stages with nothing that ever opens
a socket:

| Stage | Wispr Flow | `flow_local` |
|---|---|---|
| Activation | Global hotkey, OS-registered | `pynput` global hotkey listener (`hotkey.py`), push-to-talk or toggle |
| Capture | Streamed mic chunks to cloud | `sounddevice` mic capture straight into a local ring buffer (`audio.py`) — nothing leaves the process |
| ASR | Cloud Whisper on Baseten GPUs | `faster-whisper` (CTranslate2 Whisper) running on-device, CPU or local GPU, INT8/FP16 (`transcribe.py`) |
| Cleanup | Fine-tuned Llama, cloud-hosted | Deterministic local rules by default (filler-word strip, punctuation/casing) (`cleanup.py`); optional pluggable local LLM pass via a loopback-only Ollama call, off by default |
| Injection | OS accessibility API + clipboard fallback | Simulated keystrokes via `pynput`, or clipboard-paste fallback (`inject.py`) |

The one deliberate simplification: Flow's cleanup step is a fine-tuned LLM
with per-app tone adaptation; reproducing that quality locally would need a
model most laptops can't run at dictation latency. `cleanup.py` uses fast
deterministic rules instead, with a documented, opt-in hook to route through
a local LLM (e.g. Ollama on `127.0.0.1`) for anyone who wants closer-to-Flow
quality and has the hardware for it. Nothing in the default path requires
network access, an account, or a subscription.

See `README.md` in this directory for setup, config, and the privacy
guarantee (`flow_local` never imports an HTTP client for its default path —
verified by `tests/test_no_network.py`).
