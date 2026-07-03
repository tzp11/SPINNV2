"""Minimal SPINNV2 compiler command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.codegen.c_codegen import generate_c_from_spk, generate_c_inline
from compiler.frontend.onnx_importer import import_onnx
from compiler.packager.spk_writer import write_spk
from compiler.passes.convert_fp16 import convert_weights_fp16
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
    compile_parser.add_argument("--fp16-weights", action="store_true",
                                help="Convert FP32 weights to FP16 for ~50%% model size reduction.")
    compile_parser.add_argument("-O", "--optimization-level", type=int, choices=[0, 1, 2], default=None,
                                help="Optimization level: 0 (none), 1 (basic), 2 (full). Overrides --pass-pipeline.")
    compile_parser.add_argument("--protection-plan", help="Optional reliability ProtectionPlan JSON path.")
    codegen_parser = subparsers.add_parser("codegen", help="Generate static C deployment files from SPK.")
    codegen_parser.add_argument("spk", help="Input SPK package path.")
    codegen_parser.add_argument("--out-dir", required=True, help="Output directory for generated C files.")
    codegen_parser.add_argument("--name", default="model", help="C symbol/file prefix.")
    codegen_parser.add_argument("--runtime-dir", default="runtime", help="Path to the SPINNV2 runtime source directory.")
    codegen_parser.add_argument("--external-weights", action="store_true", help="Store weights in a separate .bin file instead of embedding in the C array.")
    codegen_parser.add_argument("--inline", action="store_true", help="Generate inline kernel calls instead of SPK blob + runtime dispatch.")
    quantize_parser = subparsers.add_parser("quantize", help="Quantize an ONNX model to INT8 SPK.")
    quantize_parser.add_argument("model", help="Input ONNX model path.")
    quantize_parser.add_argument("-o", "--output", required=True, help="Output INT8 SPK path.")
    quantize_parser.add_argument("--target", default="cpu_generic", help="Target profile name or JSON path.")
    quantize_parser.add_argument("--calibration-data", required=True, help="Path to calibration data directory (numpy .npy files).")
    quantize_parser.add_argument("--num-samples", type=int, default=100, help="Number of calibration samples.")
    quantize_parser.add_argument("--disable-passes", action="store_true", help="Disable graph optimization passes.")
    quantize_parser.add_argument("--pass-pipeline", default=",".join(DEFAULT_PIPELINE), help="Comma-separated pass names.")
    quantize_parser.add_argument("--calibration-method", choices=["minmax", "entropy"], default="minmax",
                                 help="Calibration method: minmax (default) or entropy (KL-divergence).")
    quantize_parser.add_argument("-O", "--optimization-level", type=int, choices=[0, 1, 2], default=None,
                                 help="Optimization level: 0 (none), 1 (basic), 2 (full). Overrides --pass-pipeline.")
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
        from compiler.passes.manager import pipeline_for_level

        profile = load_target_profile(args.target)
        graph = import_onnx(args.model)
        if args.optimization_level is not None:
            pass_names = pipeline_for_level(args.optimization_level)
            enabled = len(pass_names) > 0
        elif args.disable_passes:
            pass_names = []
            enabled = False
        else:
            pass_names = [name.strip() for name in args.pass_pipeline.split(",") if name.strip()]
            enabled = True
        pass_results = run_pass_pipeline(
            graph,
            pipeline=pass_names if enabled else None,
            enabled=enabled,
            profile=profile,
        )
        if args.pass_stats_json:
            write_pass_stats_json(pass_results, args.pass_stats_json)
        fp16_count = 0
        if args.fp16_weights:
            fp16_count = convert_weights_fp16(graph)
        kernel_plan = select_kernel_specs(graph, profile)
        memory_plan = plan_memory(
            graph,
            max_arena_bytes=int(profile["memory"]["activation_arena_max"]),
            alloc_input=not args.external_inputs,
            alloc_output=not args.external_outputs,
        )
        protection_plan = None
        if args.protection_plan:
            from compiler.reliability.protection_plan import load_protection_plan
            protection_plan = load_protection_plan(args.protection_plan)
        write_spk(
            graph,
            args.output,
            profile,
            memory_plan=memory_plan,
            kernel_plan=kernel_plan,
            memory_plan_csv=args.memory_plan_csv,
            protection_plan=protection_plan,
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
        if fp16_count > 0:
            print(f"FP16: converted {fp16_count} weight tensors")
        return 0

    if args.command == "codegen":
        if args.inline:
            generate_c_inline(args.spk, args.out_dir, name=args.name, runtime_dir=args.runtime_dir, external_weights=args.external_weights)
            print(f"Wrote inline C deployment to {args.out_dir}")
        else:
            generate_c_from_spk(args.spk, args.out_dir, name=args.name, runtime_dir=args.runtime_dir, external_weights=args.external_weights)
            print(f"Wrote generated C deployment to {args.out_dir}")
        return 0

    if args.command == "quantize":
        import numpy as np
        from compiler.passes.manager import pipeline_for_level
        from compiler.quantization.calibrate import calibrate_from_graph
        from compiler.quantization.quantize_weights import (
            quantize_conv_weights,
            build_activation_quant_params,
        )

        profile = load_target_profile(args.target)
        graph = import_onnx(args.model)
        if args.optimization_level is not None:
            pass_names = pipeline_for_level(args.optimization_level)
            enabled = len(pass_names) > 0
        elif args.disable_passes:
            pass_names = []
            enabled = False
        else:
            pass_names = [name.strip() for name in args.pass_pipeline.split(",") if name.strip()]
            enabled = True
        run_pass_pipeline(graph, pipeline=pass_names if enabled else None, enabled=enabled, profile=profile)

        cal_dir = Path(args.calibration_data)
        cal_files = sorted(cal_dir.glob("*.npy"))[:args.num_samples]
        if not cal_files:
            print(f"No .npy files found in {cal_dir}")
            return 1
        calibration_data = [np.load(str(f)) for f in cal_files]

        calibration = calibrate_from_graph(graph, calibration_data, args.model, method=args.calibration_method)
        weight_quant_params = quantize_conv_weights(graph)
        activation_quant_params = build_activation_quant_params(graph, calibration)
        all_quant_params = weight_quant_params + activation_quant_params

        kernel_plan = select_kernel_specs(graph, profile)
        memory_plan = plan_memory(
            graph,
            max_arena_bytes=int(profile["memory"]["activation_arena_max"]),
        )
        write_spk(
            graph,
            args.output,
            profile,
            memory_plan=memory_plan,
            kernel_plan=kernel_plan,
            quant_params=all_quant_params,
        )
        int8_count = sum(1 for s in kernel_plan.specs if s.kernel_kind == "int8_im2col_gemm")
        print(f"Wrote INT8 SPK: {args.output}")
        print(f"Quantized: {len(weight_quant_params)} weight tensors, {len(activation_quant_params)} activation tensors")
        print(f"INT8 Conv kernels: {int8_count}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
