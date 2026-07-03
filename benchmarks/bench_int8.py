#!/usr/bin/env python3
"""INT8 vs FP32 benchmark: compile both variants, run spkv2_bench, compare latency."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.frontend.onnx_importer import import_onnx
from compiler.passes.manager import DEFAULT_PIPELINE, run_pass_pipeline
from compiler.planner.kernel_spec import select_kernel_specs
from compiler.planner.memory_plan import plan_memory
from compiler.packager.spk_writer import write_spk
from compiler.target.profile import load_target_profile
from compiler.quantization.calibrate import calibrate_from_graph
from compiler.quantization.quantize_weights import quantize_conv_weights, build_activation_quant_params

BENCH = ROOT / "build" / "runtime" / "spkv2_bench"

MODELS = {
    "resnet101": ROOT / "build" / "models" / "resnet101.onnx",
    "yolov10n": ROOT / "build" / "models" / "yolov10n.onnx",
}


def get_input_shape(onnx_path: Path) -> list[int]:
    import onnx
    model = onnx.load(str(onnx_path))
    shape = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        shape.append(int(dim.dim_value) if dim.dim_value > 0 else 1)
    return shape


def compile_fp32(onnx_path: Path, spk_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile",
         str(onnx_path), "-o", str(spk_path), "--target", "cpu_generic"],
        check=True, capture_output=True, cwd=str(ROOT),
    )


def compile_int8(onnx_path: Path, spk_path: Path, cal_data: list[np.ndarray]) -> int:
    profile = load_target_profile("cpu_generic")
    graph = import_onnx(str(onnx_path))
    run_pass_pipeline(graph, pipeline=list(DEFAULT_PIPELINE), enabled=True, profile=profile)
    calibration = calibrate_from_graph(graph, cal_data, str(onnx_path))
    weight_qp = quantize_conv_weights(graph)
    act_qp = build_activation_quant_params(graph, calibration)
    all_qp = weight_qp + act_qp
    kernel_plan = select_kernel_specs(graph, profile)
    memory_plan = plan_memory(graph, max_arena_bytes=int(profile["memory"]["activation_arena_max"]))
    write_spk(graph, str(spk_path), profile, memory_plan=memory_plan, kernel_plan=kernel_plan, quant_params=all_qp)
    return sum(1 for s in kernel_plan.specs if s.kernel_kind == "int8_im2col_gemm")


def bench_spk(spk_path: Path, input_path: Path, output_path: Path,
              warmup: int = 10, runs: int = 50) -> dict | None:
    proc = subprocess.run(
        [str(BENCH), str(spk_path), str(input_path), str(output_path),
         "--warmup", str(warmup), "--runs", str(runs)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  BENCH FAILED: {proc.stderr[:200]}")
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="resnet101,yolov10n")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--cal-samples", type=int, default=5)
    parser.add_argument("--out-dir", default="build/bench_int8")
    args = parser.parse_args()

    if not BENCH.exists():
        print("Building runtime...")
        subprocess.run(
            ["cmake", "-S", str(ROOT / "runtime"), "-B", str(ROOT / "build" / "runtime"),
             "-DCMAKE_BUILD_TYPE=Release"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["cmake", "--build", str(ROOT / "build" / "runtime"), "-j4"],
            check=True, capture_output=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}

    print(f"\n{'Model':<16} {'Variant':<8} {'avg_ms':>10} {'p50_ms':>10} {'p90_ms':>10} {'min_ms':>10}")
    print("-" * 70)

    for name in model_names:
        if name not in MODELS or not MODELS[name].exists():
            print(f"{name:<16} ONNX NOT FOUND at {MODELS.get(name, '?')}")
            continue

        onnx_path = MODELS[name]
        shape = get_input_shape(onnx_path)
        np.random.seed(42)
        x = np.random.randn(*shape).astype(np.float32) * 0.1
        cal_data = [np.random.randn(*shape).astype(np.float32) * 0.1 for _ in range(args.cal_samples)]

        mdir = out_dir / name
        mdir.mkdir(parents=True, exist_ok=True)

        inp_path = mdir / "input.bin"
        out_path = mdir / "output.bin"
        inp_path.write_bytes(x.tobytes())

        fp32_spk = mdir / f"{name}_fp32.spk"
        int8_spk = mdir / f"{name}_int8.spk"

        print(f"\n  Compiling {name} FP32...", end=" ", flush=True)
        compile_fp32(onnx_path, fp32_spk)
        print(f"done ({fp32_spk.stat().st_size} bytes)")

        print(f"  Compiling {name} INT8...", end=" ", flush=True)
        n_int8 = compile_int8(onnx_path, int8_spk, cal_data)
        print(f"done ({int8_spk.stat().st_size} bytes, {n_int8} INT8 kernels)")

        fp32_bench = bench_spk(fp32_spk, inp_path, out_path, args.warmup, args.runs)
        int8_bench = bench_spk(int8_spk, inp_path, out_path, args.warmup, args.runs)

        entry = {"fp32": fp32_bench, "int8": int8_bench, "int8_kernels": n_int8}
        results[name] = entry

        for variant, r in [("FP32", fp32_bench), ("INT8", int8_bench)]:
            if r:
                avg = r.get("avg_ms", 0)
                p50 = r.get("p50_ms", 0)
                p90 = r.get("p90_ms", 0)
                mn = r.get("min_ms", 0)
                print(f"{name:<16} {variant:<8} {avg:>10.3f} {p50:>10.3f} {p90:>10.3f} {mn:>10.3f}")
            else:
                print(f"{name:<16} {variant:<8} FAILED")

        if fp32_bench and int8_bench:
            fp32_p50 = fp32_bench.get("p50_ms", 1)
            int8_p50 = int8_bench.get("p50_ms", 1)
            speedup = fp32_p50 / int8_p50 if int8_p50 > 0 else 0
            print(f"  >>> Speedup: {speedup:.2f}x (p50 FP32={fp32_p50:.3f}ms → INT8={int8_p50:.3f}ms)")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }
    report_path = out_dir / "int8_bench_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
