#!/usr/bin/env python3
"""Per-op benchmark: compile single-op models from test suite, run spkv2_bench, collect JSON."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.e2e.test_single_ops import (
    _make_conv_model, _make_gemm_model, _make_relu_model,
    _make_sigmoid_model, _make_softmax_model,
)


def _make_models() -> dict[str, tuple]:
    return {
        "conv_3x3s1p1":  (_make_conv_model(3, 16, 32, 3, 1, 1), (1, 3, 16, 16)),
        "conv_1x1":      (_make_conv_model(3, 16, 1, 1, 0),     (1, 3, 16, 16)),
        "conv_dw":       (_make_conv_model(16, 16, 32, 3, 1, 1, group=16), (1, 16, 16, 16)),
        "gemm":          (_make_gemm_model(64, 128, bias=True),  (1, 64)),
        "relu":          (_make_relu_model((1, 64, 16, 16)),     (1, 64, 16, 16)),
        "sigmoid":       (_make_sigmoid_model((1, 64, 16, 16)),  (1, 64, 16, 16)),
    }


def compile_spk(onnx_path: Path, spk_path: Path) -> bool:
    cmd = [sys.executable, "-m", "compiler.cli", "compile", str(onnx_path), "-o", str(spk_path),
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


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/bench_ops")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_bin = ROOT / "runtime" / "build" / "spkv2_bench"
    if not bench_bin.exists():
        print("Building runtime...")
        subprocess.run(
            ["cmake", "-S", str(ROOT / "runtime"), "-B", str(ROOT / "runtime" / "build"),
             "-DCMAKE_BUILD_TYPE=Release"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["cmake", "--build", str(ROOT / "runtime" / "build"), "-j4"],
            check=True, capture_output=True,
        )
    if not bench_bin.exists():
        print(f"ERROR: {bench_bin} not found", file=sys.stderr)
        return 1

    import onnx
    models = _make_models()
    results = {}

    print(f"{'Op':<20} {'avg_ms':>10} {'min_ms':>10} {'p50_ms':>10}")
    print("-" * 55)

    for name, (model, input_shape) in models.items():
        mdir = out_dir / name
        mdir.mkdir(parents=True, exist_ok=True)

        onnx_path = mdir / "model.onnx"
        onnx.save(model, str(onnx_path))

        spk_path = mdir / "model.spk"
        if not compile_spk(onnx_path, spk_path):
            print(f"{name:<20} COMPILE FAILED")
            continue

        inp = np.random.randn(*input_shape).astype(np.float32)
        input_path = mdir / "input.bin"
        input_path.write_bytes(inp.tobytes())
        output_path = mdir / "output.bin"

        r = bench_spk(bench_bin, spk_path, input_path, output_path,
                      warmup=args.warmup, runs=args.runs)
        if r:
            results[name] = r
            print(f"{name:<20} {r['avg_ms']:>10.3f} {r['min_ms']:>10.3f} {r['p50_ms']:>10.3f}")
        else:
            print(f"{name:<20} BENCH FAILED")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }
    report_path = out_dir / "bench_ops_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
