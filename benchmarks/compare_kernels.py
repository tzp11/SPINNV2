#!/usr/bin/env python3
"""Compare reference and optimized KernelSpec target profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from compiler.frontend.onnx_importer import import_onnx
from compiler.packager.spk_writer import write_spk
from compiler.passes.manager import run_pass_pipeline
from compiler.planner.kernel_spec import select_kernel_specs
from compiler.planner.memory_plan import plan_memory
from compiler.target.profile import load_target_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Input ONNX model path.")
    parser.add_argument("--ref-target", default="cpu_ref", help="Reference target profile.")
    parser.add_argument("--opt-target", default="cpu_generic", help="Optimized target profile.")
    parser.add_argument("--out-dir", default="build/kernel_compare", help="Output directory.")
    parser.add_argument("--input-bin", help="Optional fp32 input binary for runtime comparison.")
    parser.add_argument("--runtime-runner", default="build/runtime/spkv2_run", help="Path to spkv2_run.")
    parser.add_argument("--runs", type=int, default=10, help="Runtime runs per variant when --input-bin is set.")
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model": str(model_path),
        "reference": _compile_variant(model_path, args.ref_target, out_dir / "reference.spk"),
        "optimized": _compile_variant(model_path, args.opt_target, out_dir / "optimized.spk"),
    }
    if "error" not in report["reference"] and "error" not in report["optimized"]:
        report["delta"] = {
            "spk_size_bytes": report["optimized"]["spk_size_bytes"] - report["reference"]["spk_size_bytes"],
            "scratch_arena_bytes": report["optimized"]["scratch_arena_bytes"]
            - report["reference"]["scratch_arena_bytes"],
            "planned_activation_bytes": report["optimized"]["planned_activation_bytes"]
            - report["reference"]["planned_activation_bytes"],
        }
    if args.input_bin:
        _add_runtime_comparison(report, Path(args.input_bin), Path(args.runtime_runner), max(1, args.runs), out_dir)

    report_path = out_dir / "kernel_compare.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def _compile_variant(model_path: Path, target: str, out_path: Path) -> dict:
    try:
        profile = load_target_profile(target)
        graph = import_onnx(model_path)
        run_pass_pipeline(graph)
        kernel_plan = select_kernel_specs(graph, profile)
        memory_plan = plan_memory(
            graph,
            max_arena_bytes=int(profile["memory"]["activation_arena_max"]),
        )
        write_spk(graph, out_path, profile, memory_plan=memory_plan, kernel_plan=kernel_plan)
    except Exception as exc:  # noqa: BLE001 - benchmark reports should keep going.
        return {"target": target, "error": str(exc)}

    return {
        "target": profile["name"],
        "spk_size_bytes": out_path.stat().st_size,
        "planned_activation_bytes": memory_plan.planned_activation_bytes,
        "scratch_arena_bytes": kernel_plan.scratch_arena_bytes,
        "kernel_fallback_count": kernel_plan.fallback_count,
        "kernel_specs": [spec.__dict__ for spec in kernel_plan.specs],
    }


def _add_runtime_comparison(report: dict, input_path: Path, runner: Path, runs: int, out_dir: Path) -> None:
    outputs: dict[str, np.ndarray] = {}
    runtime: dict[str, dict] = {}
    for key, spk_name in (("reference", "reference.spk"), ("optimized", "optimized.spk")):
        if "error" in report[key]:
            runtime[key] = {"error": "compile failed"}
            continue
        output_path = out_dir / f"{key}.out.bin"
        timings = []
        try:
            for _ in range(runs):
                start = time.perf_counter()
                subprocess.run([str(runner), str(out_dir / spk_name), str(input_path), str(output_path)], check=True)
                timings.append((time.perf_counter() - start) * 1000.0)
            outputs[key] = np.frombuffer(output_path.read_bytes(), dtype=np.float32).copy()
            runtime[key] = {
                "runs": runs,
                "time_ms_avg": float(np.mean(timings)),
                "time_ms_p50": float(np.percentile(timings, 50)),
                "time_ms_p90": float(np.percentile(timings, 90)),
            }
        except Exception as exc:  # noqa: BLE001 - benchmark reports should keep going.
            runtime[key] = {"error": str(exc)}
    if "reference" in outputs and "optimized" in outputs:
        diff = np.abs(outputs["reference"] - outputs["optimized"])
        runtime["output_compare"] = {
            "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
            "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        }
    report["runtime"] = runtime


if __name__ == "__main__":
    raise SystemExit(main())
