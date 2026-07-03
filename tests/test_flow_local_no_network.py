"""Static guarantee: the always-on flow_local modules never import an
HTTP/socket client. The only network-capable module is llm_cleanup.py,
which is opt-in (FlowConfig.use_local_llm_cleanup, default False) and
restricted to loopback hosts (see test_flow_local_llm_cleanup.py)."""

import ast
from pathlib import Path

FLOW_LOCAL_DIR = Path(__file__).resolve().parent.parent / "flow_local"
ALWAYS_ON_MODULES = [
    "audio.py", "transcribe.py", "cleanup.py", "inject.py",
    "hotkey.py", "app.py", "cli.py", "config.py",
]
NETWORK_MODULES = {
    "socket", "requests", "urllib", "urllib.request", "http", "http.client",
    "httpx", "aiohttp", "websocket", "websockets", "ftplib", "smtplib",
}


def _imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_always_on_modules_import_no_network_client():
    offenders = {}
    for filename in ALWAYS_ON_MODULES:
        imported = _imported_modules(FLOW_LOCAL_DIR / filename)
        hit = imported & NETWORK_MODULES
        if hit:
            offenders[filename] = hit
    assert not offenders, f"network-capable imports found: {offenders}"


def test_llm_cleanup_module_exists_and_is_the_only_exception():
    assert (FLOW_LOCAL_DIR / "llm_cleanup.py").exists()
    imported = _imported_modules(FLOW_LOCAL_DIR / "llm_cleanup.py")
    assert imported & NETWORK_MODULES, "expected llm_cleanup.py to be the module that talks over the network"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
