"""Entrypoint: `python -m flow_local.cli [--config path] [--print-config]`"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import FlowConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flow_local", description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="path to a JSON config file")
    parser.add_argument("--hotkey", type=str, default=None)
    parser.add_argument("--mode", choices=["push_to_talk", "toggle"], default=None)
    parser.add_argument("--model-size", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--injection-mode", choices=["type", "paste"], default=None)
    parser.add_argument("--print-config", action="store_true", help="print the effective config and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def load_config(args: argparse.Namespace) -> FlowConfig:
    config = FlowConfig.load(args.config) if args.config else FlowConfig()
    overrides = {
        "hotkey": args.hotkey,
        "mode": args.mode,
        "model_size": args.model_size,
        "device": args.device,
        "injection_mode": args.injection_mode,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    config = load_config(args)

    if args.print_config:
        import json
        from dataclasses import asdict

        print(json.dumps(asdict(config), indent=2))
        return 0

    from .app import FlowApp

    FlowApp(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
