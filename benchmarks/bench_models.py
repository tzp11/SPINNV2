#!/usr/bin/env python3
"""Per-model benchmark: compile model_zoo models, run spkv2_bench, collect JSON results."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.e2e.model_zoo import write_model, ALL_MODELS


def compile_spk(onnx_path: Path, spk_path: Path) -> bool:
    cmd = [sys.executable, "-m", "spinnv2.compiler", "compile", str(onnx_path), "-o", str(spk_path),
           "--target", "cpu_generic"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode == 0


def bench_spk(bench_bin: Path, spk_path: Path, input_path: Path, output_path: Path,
              warmup: int = 5, runs: int = 50) -> dict | None:
    cmd = [str(bench_bin), str(spk_path), str(input_path), str(output_path),
           "--warmup", str(warmup), "--runs", str(runs)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def get_input_shape(onnx_path: Path) -> list[int]:
    import onnx
    model = onnx.load(str(onnx_path))
    shape = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        shape.append(int(dim.dim_value) if dim.dim_value > 0 else 1)
    return shape


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/bench_models")
    parser.add_argument("--models", default=",".join(ALL_MODELS),
                        help=f"Comma-separated model names. Available: {','.join(ALL_MODELS)}")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_bin = ROOT / "build" / "runtime" / "spkv2_bench"
    if not bench_bin.exists():
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
    if not bench_bin.exists():
        print(f"ERROR: {bench_bin} not found", file=sys.stderr)
        return 1

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}

    print(f"{'Model':<24} {'avg_ms':>10} {'min_ms':>10} {'p50_ms':>10} {'p90_ms':>10}")
    print("-" * 70)

    for name in model_names:
        if name not in ALL_MODELS:
            print(f"{name:<24} UNKNOWN MODEL")
            continue

        mdir = out_dir / name
        mdir.mkdir(parents=True, exist_ok=True)

        onnx_path = mdir / f"{name}.onnx"
        write_model(name, onnx_path)

        spk_path = mdir / f"{name}.spk"
        if not compile_spk(onnx_path, spk_path):
            print(f"{name:<24} COMPILE FAILED")
            continue

        input_shape = get_input_shape(onnx_path)
        inp = np.random.randn(*input_shape).astype(np.float32) * 0.1
        input_path = mdir / "input.bin"
        input_path.write_bytes(inp.tobytes())
        output_path = mdir / "output.bin"

        r = bench_spk(bench_bin, spk_path, input_path, output_path,
                      warmup=args.warmup, runs=args.runs)
        if r:
            results[name] = r
            avg = r.get("avg_ms", 0)
            mn = r.get("min_ms", 0)
            p50 = r.get("p50_ms", 0)
            p90 = r.get("p90_ms", 0)
            print(f"{name:<24} {avg:>10.3f} {mn:>10.3f} {p50:>10.3f} {p90:>10.3f}")
        else:
            print(f"{name:<24} BENCH FAILED")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "runs": args.runs,
        "models": model_names,
        "results": results,
    }
    report_path = out_dir / "bench_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
