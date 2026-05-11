#!/usr/bin/env python3
"""Collect runtime latency for generated validation models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.model_zoo import make_input, write_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mnist,lenet")
    parser.add_argument("--out-dir", default="build/latency_benchmark")
    parser.add_argument("--target", default="cpu_ref")
    parser.add_argument("--runtime-runner", default="build/runtime/spkv2_run")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"target": args.target, "runs": args.runs, "models": {}}
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        report["models"][name] = run_one(name, out_root / name, args.target, Path(args.runtime_runner), args.runs)

    report_path = out_root / "latency_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def run_one(name: str, out_dir: Path, target: str, runner: Path, runs: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.onnx"
    spk_path = out_dir / f"{name}.spk"
    input_path = out_dir / "input.bin"
    output_path = out_dir / "output.bin"
    write_model(name, model_path)
    subprocess.run(["python", "-m", "spinnv2.compiler", "compile", str(model_path), "-o", str(spk_path), "--target", target], check=True)
    input_path.write_bytes(np.ascontiguousarray(make_input(name)).tobytes())
    timings = []
    for _ in range(max(1, runs)):
        started = time.perf_counter()
        subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)
        timings.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "run_time_ms_avg": float(values.mean()),
        "run_time_ms_p50": float(np.percentile(values, 50)),
        "run_time_ms_p90": float(np.percentile(values, 90)),
        "throughput_fps": float(1000.0 / values.mean()) if values.mean() > 0 else 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
