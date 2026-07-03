#!/usr/bin/env python3
"""
In-process benchmark: SPINNV2 (1-thread / multi-thread) vs ORT (1-thread / multi-thread).

Eliminates process-startup overhead:
  - SPINNV2: spkv2_bench loads once, warms up, then times N runs of spkv2_run()
  - ORT:     Python session.run() with same warmup + N runs protocol

Models: resnet101, yolov10n (full-size inputs)

Output table columns:
  ORT-1T | ORT-MT | SPINNV2-1T | SPINNV2-MT | SIMD-MT/ORT-1T | SIMD-MT/ORT-MT
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
    "resnet101": ROOT / "build" / "models" / "resnet101.onnx",
    "yolov10n": ROOT / "build" / "models" / "yolov10n.onnx",
}

# Frozen REF baseline from previous benchmark (no need to re-run every time)
REF_BASELINE = {
    "resnet101": {"avg_ms": 18142.82, "min_ms": 17553.75, "p50_ms": 18132.63, "p90_ms": 19021.39, "max_ms": 19021.39},
    "yolov10n":  {"avg_ms": 9453.08,  "min_ms": 8867.98,  "p50_ms": 9309.81,  "p90_ms": 10576.99, "max_ms": 10576.99},
}

WARMUP = 10
RUNS = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="resnet101,yolov10n")
    parser.add_argument("--out-dir", default="build/bench_simd_vs_ort")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--profile", action="store_true",
                        help="Collect per-layer profiling (SPKV2_PROFILE=1)")
    parser.add_argument("--skip-1t", action="store_true",
                        help="Skip single-threaded SPINNV2 benchmark (saves time)")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
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
            profile=args.profile, skip_1t=args.skip_1t,
        )

    report_path = out_root / "bench_report.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "runs": args.runs,
        "models": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {report_path}")

    # Pretty table
    hdr = f"{'Model':<14} {'ORT-1T':>10} {'ORT-MT':>10} {'SIMD-1T':>10} {'SIMD-MT':>10} {'SIMD/ORT-1T':>12} {'SIMD/ORT-MT':>12} {'cos_sim':>9}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for name, r in results.items():
        if r is None:
            r = {}
        ort_1t  = (r.get("ort_1t")  or {}).get("avg_ms", float("nan"))
        ort_mt  = (r.get("ort_mt")  or {}).get("avg_ms", float("nan"))
        simd_1t = (r.get("simd_1t") or {}).get("avg_ms", float("nan"))
        simd_mt = (r.get("simd_mt") or {}).get("avg_ms", float("nan"))
        ratio_1t = simd_mt / ort_1t if ort_1t > 0 else float("nan")
        ratio_mt = simd_mt / ort_mt if ort_mt > 0 else float("nan")
        cos = r.get("simd_mt", {}).get("vs_ort", {}).get("cosine_similarity", float("nan"))
        print(f"{name:<14} {ort_1t:>10.2f} {ort_mt:>10.2f} {simd_1t:>10.2f} {simd_mt:>10.2f}"
              f" {ratio_1t:>12.3f}x {ratio_mt:>12.3f}x {cos:>9.6f}")
    return 0


def bench_model(
    name: str, model_path: Path, mdir: Path, bench_bin: Path,
    warmup: int, runs: int, profile: bool = False, skip_1t: bool = False,
) -> dict:
    result: dict = {}

    # --- Prepare input ---
    input_array = make_input(model_path)
    input_path = mdir / "input.bin"
    input_path.write_bytes(np.ascontiguousarray(input_array).tobytes())
    print(f"  Input shape: {input_array.shape}  ({input_array.nbytes} bytes)")

    # --- ORT 1-thread ---
    print("  [ORT-1T] benchmarking ...")
    result["ort_1t"] = bench_ort(model_path, input_array, warmup=warmup, runs=runs, num_threads=1)
    r1t = result["ort_1t"]
    print(f"  [ORT-1T] avg={r1t['avg_ms']:.2f}ms  min={r1t['min_ms']:.2f}ms  p50={r1t['p50_ms']:.2f}ms")

    # --- ORT multi-thread ---
    print("  [ORT-MT] benchmarking ...")
    result["ort_mt"] = bench_ort(model_path, input_array, warmup=warmup, runs=runs, num_threads=0)
    rmt = result["ort_mt"]
    print(f"  [ORT-MT] avg={rmt['avg_ms']:.2f}ms  min={rmt['min_ms']:.2f}ms  p50={rmt['p50_ms']:.2f}ms")

    # --- SPINNV2 ref (frozen baseline) ---
    if name in REF_BASELINE:
        result["ref"] = REF_BASELINE[name]
        print(f"  [REF]    (frozen baseline) avg={result['ref']['avg_ms']:.2f}ms")

    # --- Compile SPINNV2 SPK ---
    print("  [SIMD] compiling ...")
    simd_spk = mdir / f"{name}_simd.spk"
    compiled = compile_spk(model_path, simd_spk, target="cpu_generic")

    if compiled:
        # --- SPINNV2 multi-thread ---
        simd_mt_out = mdir / f"{name}_simd_mt_output.bin"
        print("  [SIMD-MT] benchmarking ...")
        result["simd_mt"] = bench_spkv2(
            bench_bin, simd_spk, input_path, simd_mt_out,
            warmup=warmup, runs=runs, threads=0,
        )
        if result["simd_mt"]:
            s = result["simd_mt"]
            print(f"  [SIMD-MT] avg={s['avg_ms']:.2f}ms  min={s['min_ms']:.2f}ms  p50={s['p50_ms']:.2f}ms")
        else:
            print("  [SIMD-MT] runtime FAILED")

        # --- SPINNV2 single-thread ---
        if not skip_1t:
            simd_1t_out = mdir / f"{name}_simd_1t_output.bin"
            print("  [SIMD-1T] benchmarking ...")
            result["simd_1t"] = bench_spkv2(
                bench_bin, simd_spk, input_path, simd_1t_out,
                warmup=warmup, runs=runs, threads=1,
            )
            if result["simd_1t"]:
                s = result["simd_1t"]
                print(f"  [SIMD-1T] avg={s['avg_ms']:.2f}ms  min={s['min_ms']:.2f}ms  p50={s['p50_ms']:.2f}ms")
            else:
                print("  [SIMD-1T] runtime FAILED")
    else:
        print("  [SIMD] compile FAILED")

    # --- Numerical comparison (SIMD-MT vs ORT-1T) ---
    ort_output = result["ort_1t"].get("output")
    for key, out_path in [
        ("simd_mt", mdir / f"{name}_simd_mt_output.bin"),
        ("simd_1t", mdir / f"{name}_simd_1t_output.bin"),
    ]:
        if key not in result or result[key] is None:
            continue
        if ort_output is not None and out_path.exists():
            sp = np.fromfile(out_path, dtype=np.float32)
            if sp.size == ort_output.size:
                diff = np.abs(sp - ort_output.reshape(-1))
                cos = float(np.dot(sp, ort_output.reshape(-1)) /
                            (np.linalg.norm(sp) * np.linalg.norm(ort_output) + 1e-12))
                result[key]["vs_ort"] = {
                    "max_abs_error": float(diff.max()),
                    "mean_abs_error": float(diff.mean()),
                    "cosine_similarity": cos,
                }
                print(f"  [{key.upper()} vs ORT-1T] max_abs={diff.max():.6e}  cos={cos:.6f}")

    # --- Per-layer profiling ---
    if profile and compiled and simd_spk.exists():
        print("  [PROFILE] collecting per-layer timing ...")
        profile_text = bench_spkv2_profile(bench_bin, simd_spk, input_path, warmup=warmup, runs=runs)
        if profile_text:
            profile_path = mdir / f"{name}_profile.txt"
            profile_path.write_text(profile_text)
            result["profile_path"] = str(profile_path)
            print(f"  [PROFILE] saved to {profile_path}")

    # --- Memory plan ---
    spk_json = simd_spk.with_suffix(".spk.json")
    if spk_json.exists():
        meta = json.loads(spk_json.read_text())
        mem = meta.get("memory", {})
        result["memory"] = {
            "planned_activation_bytes": mem.get("planned_activation_bytes", 0),
            "naive_activation_bytes": mem.get("naive_activation_bytes", 0),
            "reduction_ratio": mem.get("memory_reduction_ratio", 0),
            "scratch_bytes": meta.get("metadata", {}).get("scratch_arena_bytes", 0),
        }
        planned_mb = result["memory"]["planned_activation_bytes"] / (1024 * 1024)
        scratch_mb = result["memory"]["scratch_bytes"] / (1024 * 1024)
        print(f"  [MEM] planned={planned_mb:.1f}MB  scratch={scratch_mb:.1f}MB"
              f"  reduction={result['memory']['reduction_ratio']:.1%}")

    # Remove numpy array from JSON output
    for k in ("ort_1t", "ort_mt"):
        if "output" in result.get(k, {}):
            del result[k]["output"]
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


def bench_ort(model_path: Path, input_array: np.ndarray, warmup: int, runs: int,
              num_threads: int = 1) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads  # 0 = use all cores
    opts.inter_op_num_threads = num_threads
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
                warmup: int, runs: int, threads: int = 0) -> dict | None:
    cmd = [
        str(bench_bin), str(spk_path), str(input_path), str(output_path),
        "--warmup", str(warmup), "--runs", str(runs),
    ]
    if threads > 0:
        cmd += ["--threads", str(threads)]
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


def bench_spkv2_profile(bench_bin: Path, spk_path: Path, input_path: Path,
                        warmup: int, runs: int) -> str | None:
    cmd = [
        str(bench_bin), str(spk_path), str(input_path), "/dev/null",
        "--warmup", str(warmup), "--runs", str(runs),
    ]
    env = {**subprocess.os.environ, "SPKV2_PROFILE": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return None
    return proc.stderr


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
