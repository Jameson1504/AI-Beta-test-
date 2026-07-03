"""Tests for flow_local's transcript cleanup and command parsing — the pure
text-processing core, testable with no audio hardware or model weights."""

from flow_local.cleanup import TranscriptProcessor
from flow_local.config import FlowConfig


def processor(**overrides):
    return TranscriptProcessor(FlowConfig(**overrides))


def test_strips_filler_words():
    result = processor().process("um so I think, uh, this works")
    assert "um" not in result.text.lower().split()
    assert "uh" not in result.text.lower().split()


def test_capitalizes_and_punctuates():
    result = processor().process("this is a test")
    assert result.text == "This is a test."


def test_leaves_existing_terminal_punctuation():
    result = processor().process("is this a test?")
    assert result.text == "Is this a test?"


def test_capitalizes_standalone_i():
    result = processor().process("i think i am ready")
    assert " I " in f" {result.text} "


def test_filler_removal_can_be_disabled():
    result = processor(remove_filler_words=False, auto_punctuate_capitalize=False).process("um hello")
    assert result.text == "um hello"


def test_scratch_that_is_a_command_not_insert():
    result = processor().process("scratch that")
    assert result.action == "clear_last"
    assert result.text == ""


def test_new_paragraph_command():
    result = processor().process("New Paragraph")
    assert result.action == "new_paragraph"


def test_normal_speech_is_not_mistaken_for_a_command():
    result = processor().process("please scratch that itch for me")
    assert result.action == "insert"


def test_empty_transcript_round_trips_empty():
    result = processor().process("")
    assert result.action == "insert"
    assert result.text == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
