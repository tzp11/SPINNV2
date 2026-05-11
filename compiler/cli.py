"""Minimal SPINNV2 compiler command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.frontend.onnx_importer import import_onnx
from compiler.packager.spk_writer import write_spk
from compiler.passes.manager import DEFAULT_PIPELINE, run_pass_pipeline, write_pass_stats_json
from compiler.planner.kernel_spec import select_kernel_specs
from compiler.planner.memory_plan import plan_memory
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
    compile_parser.add_argument("--memory-plan-csv", help="Optional memory plan CSV output path.")
    compile_parser.add_argument(
        "--disable-passes",
        action="store_true",
        help="Disable the default M3 graph optimization pipeline.",
    )
    compile_parser.add_argument(
        "--pass-pipeline",
        default=",".join(DEFAULT_PIPELINE),
        help="Comma-separated M3 pass names. Defaults to the full M3 pipeline.",
    )
    compile_parser.add_argument("--pass-stats-json", help="Optional M3 pass statistics JSON output path.")
    compile_parser.add_argument("--external-inputs", action="store_true", help="Do not allocate graph inputs in activation arena.")
    compile_parser.add_argument("--external-outputs", action="store_true", help="Do not allocate graph outputs in activation arena.")
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
        pass_names = [name.strip() for name in args.pass_pipeline.split(",") if name.strip()]
        pass_results = run_pass_pipeline(
            graph,
            pipeline=pass_names,
            enabled=not args.disable_passes,
        )
        if args.pass_stats_json:
            write_pass_stats_json(pass_results, args.pass_stats_json)
        kernel_plan = select_kernel_specs(graph, profile)
        memory_plan = plan_memory(
            graph,
            max_arena_bytes=int(profile["memory"]["activation_arena_max"]),
            alloc_input=not args.external_inputs,
            alloc_output=not args.external_outputs,
        )
        write_spk(
            graph,
            args.output,
            profile,
            memory_plan=memory_plan,
            kernel_plan=kernel_plan,
            memory_plan_csv=args.memory_plan_csv,
        )
        print(f"Wrote {args.output}")
        if pass_results:
            print("Passes:", " ".join(f"{result.name}:{result.changed}" for result in pass_results))
        print(
            "Kernels:",
            f"scratch={kernel_plan.scratch_arena_bytes}",
            f"fallbacks={kernel_plan.fallback_count}",
        )
        print(
            "Memory:",
            f"naive={memory_plan.naive_activation_bytes}",
            f"planned={memory_plan.planned_activation_bytes}",
            f"reduction={memory_plan.memory_reduction_ratio:.4f}",
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
