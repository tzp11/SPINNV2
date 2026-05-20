#!/usr/bin/env python3
"""Benchmark precompiled SPINNV2 artifacts against ONNX Runtime.

This script intentionally does not build the runtime and does not compile SPK
files. It only times execution of existing .spk/input.bin pairs plus ORT.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="resnet101,yolov10n")
    parser.add_argument("--artifact-dir", default="build/m6_opt1_correctness")
    parser.add_argument("--out-dir", default="build/bench_precompiled_vs_ort")
    parser.add_argument("--bench-bin", default="build/runtime/spkv2_bench")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--ort-threads", type=int, default=1)
    args = parser.parse_args()

    bench_bin = ROOT / args.bench_bin
    if not bench_bin.exists():
        raise SystemExit(f"missing SPINNV2 bench binary: {bench_bin}")

    artifact_root = ROOT / args.artifact_dir
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    report = {}
    for name in selected:
        if name not in DEFAULT_MODELS:
            print(f"SKIP unknown model: {name}")
            continue
        model_path = DEFAULT_MODELS[name]
        model_dir = artifact_root / name
        spk_path = model_dir / f"{name}.spk"
        input_path = model_dir / "input.bin"
        if not spk_path.exists() or not input_path.exists():
            raise SystemExit(f"missing precompiled artifacts for {name}: {spk_path}, {input_path}")

        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print(f"SPK:   {spk_path}")
        print(f"Input: {input_path}")
        print(f"{'=' * 70}")

        input_array = make_input(model_path)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        ort_stats = bench_ort(model_path, input_array, args.warmup, args.runs, args.ort_threads)
        spinn_stats = bench_spinnv2(
            bench_bin,
            spk_path,
            input_path,
            out_dir / "spinnv2_output.bin",
            args.warmup,
            args.runs,
        )

        ratios = ratio_stats(spinn_stats, ort_stats)
        report[name] = {
            "model_path": str(model_path),
            "spk_path": str(spk_path),
            "input_path": str(input_path),
            "ort": ort_stats,
            "spinnv2": spinn_stats,
            "spinnv2_over_ort": ratios,
        }
        print_model_summary(name, ort_stats, spinn_stats, ratios)

    report_path = out_root / "bench_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0


def make_input(model_path: Path) -> np.ndarray:
    model = onnx.load(model_path)
    shape = [int(dim.dim_value) for dim in model.graph.input[0].type.tensor_type.shape.dim]
    total = int(np.prod(shape))
    if shape[-1] == 640:
        return np.linspace(0.0, 1.0, num=total, dtype=np.float32).reshape(shape)
    return np.linspace(-1.0, 1.0, num=total, dtype=np.float32).reshape(shape)


def bench_ort(
    model_path: Path,
    input_array: np.ndarray,
    warmup: int,
    runs: int,
    threads: int,
) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        session.run(None, {input_name: input_array})

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {input_name: input_array})
        timings.append((time.perf_counter() - started) * 1000.0)
    return stats(timings)


def bench_spinnv2(
    bench_bin: Path,
    spk_path: Path,
    input_path: Path,
    output_path: Path,
    warmup: int,
    runs: int,
) -> dict:
    proc = subprocess.run(
        [
            str(bench_bin),
            str(spk_path),
            str(input_path),
            str(output_path),
            "--warmup",
            str(warmup),
            "--runs",
            str(runs),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"SPINNV2 benchmark failed for {spk_path}:\n{proc.stderr}")
    return json.loads(proc.stdout)


def stats(timings: list[float]) -> dict:
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "runs": int(arr.size),
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(arr.max()),
    }


def ratio_stats(spinnv2: dict, ort_stats: dict) -> dict:
    keys = ("avg_ms", "min_ms", "p50_ms", "p90_ms", "max_ms")
    return {key.replace("_ms", "_ratio"): spinnv2[key] / ort_stats[key] for key in keys}


def print_model_summary(name: str, ort_stats: dict, spinnv2: dict, ratios: dict) -> None:
    print(
        f"{name:<10} "
        f"ORT avg/min/p50={ort_stats['avg_ms']:.2f}/{ort_stats['min_ms']:.2f}/{ort_stats['p50_ms']:.2f} ms  "
        f"SPINNV2 avg/min/p50={spinnv2['avg_ms']:.2f}/{spinnv2['min_ms']:.2f}/{spinnv2['p50_ms']:.2f} ms  "
        f"ratio avg/min/p50={ratios['avg_ratio']:.3f}/{ratios['min_ratio']:.3f}/{ratios['p50_ratio']:.3f}x"
    )


if __name__ == "__main__":
    raise SystemExit(main())
