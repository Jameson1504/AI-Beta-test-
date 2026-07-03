"""Global hotkey activation (push-to-talk or toggle), OS-registered via
pynput so it fires regardless of which app has focus — the same idea as
Wispr Flow's OS-level hotkey registration, done locally.

The press/release state machine (`_HotkeyStateMachine`) is kept independent
of pynput's key types so it can be unit-tested on any machine, including
ones with no display server (pynput itself requires one to import its
keyboard backend)."""

from __future__ import annotations

from typing import Callable, FrozenSet, Hashable


class _HotkeyStateMachine:
    """Tracks which keys are down and fires on_start/on_stop when `target`
    transitions in/out of "fully held", per `mode`. Keys can be any
    hashable value — real usage passes pynput key objects, tests pass
    plain strings."""

    def __init__(
        self,
        target: FrozenSet[Hashable],
        mode: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ):
        self.target = target
        self.mode = mode
        self.on_start = on_start
        self.on_stop = on_stop
        self.pressed: set = set()
        self.recording = False
        self._combo_down = False

    def _combo_held(self) -> bool:
        return self.target <= self.pressed

    def press(self, key: Hashable) -> None:
        self.pressed.add(key)
        if not self._combo_held() or self._combo_down:
            return
        self._combo_down = True

        if self.mode == "toggle":
            self.recording = not self.recording
            (self.on_start if self.recording else self.on_stop)()
        else:
            self.recording = True
            self.on_start()

    def release(self, key: Hashable) -> None:
        self.pressed.discard(key)
        if self._combo_held():
            return
        self._combo_down = False
        if self.mode == "push_to_talk" and self.recording:
            self.recording = False
            self.on_stop()


def parse_hotkey(spec: str, keyboard_module) -> FrozenSet[Hashable]:
    """Parses a config string like "<ctrl>+<alt>+space" into a frozenset of
    pynput key objects, using pynput's own `Key`/`KeyCode` types."""
    keys = []
    for part in (p.strip() for p in spec.split("+")):
        if part.startswith("<") and part.endswith(">"):
            keys.append(getattr(keyboard_module.Key, part[1:-1]))
        else:
            keys.append(keyboard_module.KeyCode.from_char(part))
    return frozenset(keys)


class HotkeyListener:
    def __init__(self, config, on_start: Callable[[], None], on_stop: Callable[[], None]):
        self.config = config
        self.on_start = on_start
        self.on_stop = on_stop
        self._listener = None
        self._machine: _HotkeyStateMachine | None = None

    def start(self) -> None:
        from pynput import keyboard  # noqa: local import — needs a display server

        target = parse_hotkey(self.config.hotkey, keyboard)
        self._machine = _HotkeyStateMachine(target, self.config.mode, self.on_start, self.on_stop)

        def on_press(key):
            self._machine.press(self._listener.canonical(key))

        def on_release(key):
            self._machine.release(self._listener.canonical(key))

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
