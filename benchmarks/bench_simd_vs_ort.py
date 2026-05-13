#!/usr/bin/env python3
"""
In-process benchmark: SPINNV2 ref vs SPINNV2 SIMD vs ONNX Runtime.

Eliminates process-startup overhead:
  - SPINNV2: spkv2_bench loads once, warms up, then times N runs of spkv2_run()
  - ORT:     Python session.run() with same warmup + N runs protocol

Models: resnet101, yolov10n (full-size inputs)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = {
    "resnet101": Path("/home/tzp/work/SPINN/SPINN/run_time/resnet101.onnx"),
    "yolov10n": Path("/home/tzp/work/SPINN/SPINN/run_time/yolov10n.onnx"),
}

# Frozen REF baseline from previous benchmark (no need to re-run every time)
REF_BASELINE = {
    "resnet101": {"avg_ms": 18142.82, "min_ms": 17553.75, "p50_ms": 18132.63, "p90_ms": 19021.39, "max_ms": 19021.39},
    "yolov10n":  {"avg_ms": 9453.08,  "min_ms": 8867.98,  "p50_ms": 9309.81,  "p90_ms": 10576.99, "max_ms": 10576.99},
}

WARMUP = 3
RUNS = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="resnet101,yolov10n")
    parser.add_argument("--out-dir", default="build/bench_simd_vs_ort")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build runtime
    subprocess.run(
        ["cmake", "-S", "runtime", "-B", "build/runtime", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=str(ROOT), check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build/runtime", f"-j{4}"],
        cwd=str(ROOT), check=True,
    )
    bench_bin = ROOT / "build" / "runtime" / "spkv2_bench"
    if not bench_bin.exists():
        print(f"ERROR: {bench_bin} not found", file=sys.stderr)
        return 1

    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name not in DEFAULT_MODELS:
            print(f"SKIP unknown model: {name}")
            continue
        model_path = DEFAULT_MODELS[name]
        if not model_path.exists():
            print(f"SKIP {name}: {model_path} not found")
            continue
        mdir = out_root / name
        mdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Model: {name}  ({model_path})")
        print(f"{'='*60}")
        results[name] = bench_model(
            name, model_path, mdir, bench_bin,
            warmup=args.warmup, runs=args.runs,
        )

    report_path = out_root / "bench_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {report_path}")

    # Pretty table
    print(f"\n{'Model':<14} {'ORT avg(ms)':>12} {'REF avg(ms)':>12} {'SIMD avg(ms)':>13} {'SIMD/ORT':>10} {'Speedup':>10}")
    print("-" * 75)
    for name, r in results.items():
        ort_avg = r.get("ort", {}).get("avg_ms", float("nan"))
        ref_avg = r.get("ref", {}).get("avg_ms", float("nan"))
        simd_avg = r.get("simd", {}).get("avg_ms", float("nan"))
        ratio = simd_avg / ort_avg if ort_avg > 0 else float("nan")
        speedup = ref_avg / simd_avg if simd_avg > 0 else float("nan")
        print(f"{name:<14} {ort_avg:>12.2f} {ref_avg:>12.2f} {simd_avg:>13.2f} {ratio:>10.2f}x {speedup:>9.2f}x")
    return 0


def bench_model(
    name: str, model_path: Path, mdir: Path, bench_bin: Path,
    warmup: int, runs: int,
) -> dict:
    result: dict = {}

    # --- Prepare input ---
    input_array = make_input(model_path)
    input_path = mdir / "input.bin"
    input_path.write_bytes(np.ascontiguousarray(input_array).tobytes())
    print(f"  Input shape: {input_array.shape}  ({input_array.nbytes} bytes)")

    # --- ORT benchmark ---
    print("  [ORT] benchmarking ...")
    result["ort"] = bench_ort(model_path, input_array, warmup=warmup, runs=runs)
    print(f"  [ORT] avg={result['ort']['avg_ms']:.2f}ms  min={result['ort']['min_ms']:.2f}ms  p50={result['ort']['p50_ms']:.2f}ms")

    # --- SPINNV2 ref (frozen baseline) ---
    if name in REF_BASELINE:
        result["ref"] = REF_BASELINE[name]
        print(f"  [REF] (frozen baseline) avg={result['ref']['avg_ms']:.2f}ms")

    # --- SPINNV2 SIMD ---
    print("  [SIMD] compiling ...")
    simd_spk = mdir / f"{name}_simd.spk"
    simd_out = mdir / f"{name}_simd_output.bin"
    if compile_spk(model_path, simd_spk, target="cpu_generic"):
        print("  [SIMD] benchmarking ...")
        result["simd"] = bench_spkv2(bench_bin, simd_spk, input_path, simd_out, warmup=warmup, runs=runs)
        if result["simd"]:
            print(f"  [SIMD] avg={result['simd']['avg_ms']:.2f}ms  min={result['simd']['min_ms']:.2f}ms  p50={result['simd']['p50_ms']:.2f}ms")
        else:
            print("  [SIMD] runtime FAILED")
    else:
        print("  [SIMD] compile FAILED")

    # --- Numerical comparison ---
    ort_output = result["ort"].get("output")
    for key, out_path in [("simd", simd_out)]:
        if ort_output is not None and out_path.exists() and key in result:
            sp = np.fromfile(out_path, dtype=np.float32)
            if sp.size == ort_output.size:
                diff = np.abs(sp - ort_output.reshape(-1))
                cos = float(np.dot(sp, ort_output.reshape(-1)) / (np.linalg.norm(sp) * np.linalg.norm(ort_output) + 1e-12))
                result[key]["vs_ort"] = {
                    "max_abs_error": float(diff.max()),
                    "mean_abs_error": float(diff.mean()),
                    "cosine_similarity": cos,
                }
                print(f"  [{key.upper()} vs ORT] max_abs={diff.max():.6e}  cos={cos:.6f}")

    # Remove numpy array from JSON output
    if "output" in result.get("ort", {}):
        del result["ort"]["output"]
    return result


def make_input(model_path: Path) -> np.ndarray:
    model = onnx.load(model_path)
    shape = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        shape.append(int(dim.dim_value))
    total = int(np.prod(shape))
    if shape[-1] == 640:
        return np.linspace(0.0, 1.0, num=total, dtype=np.float32).reshape(shape)
    return np.linspace(-1.0, 1.0, num=total, dtype=np.float32).reshape(shape)


def bench_ort(model_path: Path, input_array: np.ndarray, warmup: int, runs: int) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        session.run(None, {input_name: input_array})

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_array})
        timings.append((time.perf_counter() - t0) * 1000.0)

    output = session.run(None, {input_name: input_array})[0]
    return _stats(timings, output=np.asarray(output, dtype=np.float32))


def compile_spk(model_path: Path, spk_path: Path, target: str) -> bool:
    cmd = [
        sys.executable, "-m", "spinnv2.compiler", "compile",
        str(model_path), "-o", str(spk_path), "--target", target,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    compile error: {proc.stderr[:500]}")
    return proc.returncode == 0


def bench_spkv2(bench_bin: Path, spk_path: Path, input_path: Path, output_path: Path,
                warmup: int, runs: int) -> dict | None:
    cmd = [
        str(bench_bin), str(spk_path), str(input_path), str(output_path),
        "--warmup", str(warmup), "--runs", str(runs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    bench error: {proc.stderr[:500]}")
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"    bad JSON: {proc.stdout[:300]}")
        return None
    return data


def _stats(timings: list[float], output: np.ndarray | None = None) -> dict:
    arr = np.asarray(timings)
    result = {
        "runs": len(timings),
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(arr.max()),
    }
    if output is not None:
        result["output"] = output
    return result


if __name__ == "__main__":
    raise SystemExit(main())
