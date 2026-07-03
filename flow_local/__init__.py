"""flow_local: a private, fully local voice-dictation daemon.

Hotkey -> mic capture -> local Whisper transcription -> local cleanup ->
keystroke/clipboard injection. No audio or text ever leaves the machine.
"""

__version__ = "0.1.0"
