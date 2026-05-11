#!/usr/bin/env python3
"""Compare SPK compiler output with and without M3 graph passes."""

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
    parser.add_argument("--target", default="cpu_ref", help="Target profile name or JSON path.")
    parser.add_argument("--out-dir", default="build/pass_compare", help="Output directory.")
    parser.add_argument("--input-bin", help="Optional fp32 input binary for runtime comparison.")
    parser.add_argument("--runtime-runner", default="build/runtime/spkv2_run", help="Path to spkv2_run.")
    parser.add_argument("--runs", type=int, default=5, help="Runtime runs per variant when --input-bin is set.")
    parser.add_argument("--external-inputs", action="store_true", help="Do not allocate graph inputs in activation arena.")
    parser.add_argument("--external-outputs", action="store_true", help="Do not allocate graph outputs in activation arena.")
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = load_target_profile(args.target)
    report = {
        "model": str(model_path),
        "target": profile["name"],
        "without_passes": _compile_variant(model_path, profile, out_dir / "without_passes.spk", False, args),
        "with_passes": _compile_variant(model_path, profile, out_dir / "with_passes.spk", True, args),
    }
    before = report["without_passes"]
    after = report["with_passes"]
    if "error" not in before and "error" not in after:
        report["delta"] = {
            "node_count": after["node_count"] - before["node_count"],
            "spk_size_bytes": after["spk_size_bytes"] - before["spk_size_bytes"],
            "planned_activation_bytes": after["planned_activation_bytes"] - before["planned_activation_bytes"],
            "scratch_arena_bytes": after["scratch_arena_bytes"] - before["scratch_arena_bytes"],
        }
    if args.input_bin:
        _add_runtime_comparison(report, Path(args.input_bin), Path(args.runtime_runner), max(1, args.runs), out_dir)

    report_path = out_dir / "pass_compare.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def _compile_variant(model_path: Path, profile: dict, out_path: Path, enabled: bool, args) -> dict:
    try:
        graph = import_onnx(model_path)
        nodes_before = len(graph.nodes)
        pass_results = run_pass_pipeline(graph, enabled=enabled)
        kernel_plan = select_kernel_specs(graph, profile)
        memory_plan = plan_memory(
            graph,
            max_arena_bytes=int(profile["memory"]["activation_arena_max"]),
            alloc_input=not args.external_inputs,
            alloc_output=not args.external_outputs,
        )
        write_spk(graph, out_path, profile, memory_plan=memory_plan, kernel_plan=kernel_plan)
    except Exception as exc:  # noqa: BLE001 - this report should capture compiler failures.
        return {"enabled": enabled, "error": str(exc)}

    return {
        "enabled": enabled,
        "node_count_before": nodes_before,
        "node_count": len(graph.nodes),
        "spk_size_bytes": out_path.stat().st_size,
        "naive_activation_bytes": memory_plan.naive_activation_bytes,
        "planned_activation_bytes": memory_plan.planned_activation_bytes,
        "scratch_arena_bytes": kernel_plan.scratch_arena_bytes,
        "kernel_fallback_count": kernel_plan.fallback_count,
        "memory_reduction_ratio": memory_plan.memory_reduction_ratio,
        "pass_stats": [result.__dict__ for result in pass_results],
    }


def _add_runtime_comparison(report: dict, input_path: Path, runner: Path, runs: int, out_dir: Path) -> None:
    outputs: dict[str, np.ndarray] = {}
    runtime: dict[str, dict] = {}
    for key in ("without_passes", "with_passes"):
        variant = report[key]
        if "error" in variant:
            runtime[key] = {"error": "compile failed"}
            continue
        spk_path = out_dir / ("without_passes.spk" if key == "without_passes" else "with_passes.spk")
        output_path = out_dir / f"{key}.out.bin"
        timings = []
        try:
            for _ in range(runs):
                start = time.perf_counter()
                subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)
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
    if "without_passes" in outputs and "with_passes" in outputs:
        diff = np.abs(outputs["without_passes"] - outputs["with_passes"])
        runtime["output_compare"] = {
            "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
            "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        }
    report["runtime"] = runtime


if __name__ == "__main__":
    raise SystemExit(main())
