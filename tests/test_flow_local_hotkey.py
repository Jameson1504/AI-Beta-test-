"""Tests for the hotkey press/release state machine, using plain strings
as stand-in keys so this runs with no display server (pynput itself needs
one just to import, which this container doesn't have)."""

from flow_local.hotkey import _HotkeyStateMachine

TARGET = frozenset({"ctrl", "alt", "space"})


def test_push_to_talk_starts_on_full_combo_and_stops_on_release():
    events = []
    m = _HotkeyStateMachine(TARGET, "push_to_talk", lambda: events.append("start"), lambda: events.append("stop"))
    m.press("ctrl"); m.press("alt"); m.press("space")
    assert events == ["start"]
    m.release("space")
    assert events == ["start", "stop"]


def test_push_to_talk_ignores_key_repeat_while_held():
    events = []
    m = _HotkeyStateMachine(TARGET, "push_to_talk", lambda: events.append("start"), lambda: events.append("stop"))
    m.press("ctrl"); m.press("alt"); m.press("space")
    m.press("space")  # OS key-repeat while still held
    m.press("space")
    assert events == ["start"]


def test_push_to_talk_partial_release_does_not_stop():
    events = []
    m = _HotkeyStateMachine(TARGET, "push_to_talk", lambda: events.append("start"), lambda: events.append("stop"))
    m.press("ctrl"); m.press("alt"); m.press("space")
    m.release("alt")  # combo no longer fully held -> should stop
    assert events == ["start", "stop"]


def test_toggle_starts_then_stops_on_next_full_press():
    events = []
    m = _HotkeyStateMachine(TARGET, "toggle", lambda: events.append("start"), lambda: events.append("stop"))
    m.press("ctrl"); m.press("alt"); m.press("space")
    assert events == ["start"]
    m.release("space"); m.release("alt"); m.release("ctrl")
    assert events == ["start"]  # releasing must not stop in toggle mode
    m.press("ctrl"); m.press("alt"); m.press("space")
    assert events == ["start", "stop"]


def test_unrelated_keys_do_not_trigger():
    events = []
    m = _HotkeyStateMachine(TARGET, "push_to_talk", lambda: events.append("start"), lambda: events.append("stop"))
    m.press("ctrl"); m.press("a")
    assert events == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
