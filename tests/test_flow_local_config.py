"""Tests for FlowConfig defaults and save/load round trip."""

import os
import tempfile

from flow_local.config import FlowConfig


def test_defaults_are_fully_local():
    cfg = FlowConfig()
    assert cfg.use_local_llm_cleanup is False
    assert cfg.device == "cpu"
    assert cfg.mode in ("push_to_talk", "toggle")


def test_save_load_round_trip():
    cfg = FlowConfig(hotkey="<ctrl>+<shift>+d", model_size="small.en", device="cuda")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        cfg.save(path)
        loaded = FlowConfig.load(path)
    assert loaded == cfg


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
