"""Minimal SPINNV2 compiler command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.frontend.onnx_importer import import_onnx
from compiler.packager.spk_writer import write_spk
from compiler.target.profile import load_target_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spinnv2.compiler",
        description="SPINNV2 ahead-of-time model compiler.",
    )
    parser.add_argument(
        "--print-target",
        metavar="NAME",
        help="Print a bundled target profile as JSON and exit.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List bundled target profiles.",
    )
    subparsers = parser.add_subparsers(dest="command")
    compile_parser = subparsers.add_parser("compile", help="Compile an ONNX model to SPK.")
    compile_parser.add_argument("model", help="Input ONNX model path.")
    compile_parser.add_argument("-o", "--output", required=True, help="Output SPK path.")
    compile_parser.add_argument("--target", default="cpu_ref", help="Target profile name or JSON path.")
    return parser


def list_targets() -> list[str]:
    profiles_dir = Path(__file__).resolve().parent / "target" / "profiles"
    return sorted(path.stem for path in profiles_dir.glob("*.json"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_targets:
        for name in list_targets():
            print(name)
        return 0

    if args.print_target:
        profile = load_target_profile(args.print_target)
        print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "compile":
        profile = load_target_profile(args.target)
        graph = import_onnx(args.model)
        write_spk(graph, args.output, profile)
        print(f"Wrote {args.output}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
