"""The opt-in local-LLM cleanup pass must refuse anything but loopback,
so enabling it can never send text off the machine."""

from flow_local.llm_cleanup import refine_with_local_llm


def test_refuses_non_loopback_host():
    threw = False
    try:
        refine_with_local_llm("hello", model="llama3.2", host="example.com")
    except ValueError:
        threw = True
    assert threw, "expected ValueError for non-loopback host"


def test_allows_loopback_hosts_past_the_guard():
    # We don't actually hit the network here (no server running in tests) —
    # just confirm the host guard doesn't reject valid loopback hosts before
    # the (separate, expected-to-fail-in-CI) connection attempt.
    for host in ("127.0.0.1", "localhost"):
        try:
            refine_with_local_llm("hello", model="llama3.2", host=host, timeout=0.05)
        except ValueError:
            raise AssertionError(f"loopback host {host!r} should pass the guard")
        except OSError:
            pass  # expected: no local Ollama server running in this environment


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
