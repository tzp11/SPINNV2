#!/usr/bin/env python3
"""Benchmark INT8 vs FP32 inference on ResNet-101 and YOLOv10n.

Usage:
    python3 scripts/bench_int8_vs_fp32.py [--runs N]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.frontend.onnx_importer import import_onnx
from compiler.passes.manager import DEFAULT_PIPELINE, run_pass_pipeline
from compiler.planner.kernel_spec import select_kernel_specs
from compiler.planner.memory_plan import plan_memory
from compiler.packager.spk_writer import write_spk
from compiler.target.profile import load_target_profile
from compiler.quantization.calibrate import calibrate_from_graph
from compiler.quantization.quantize_weights import quantize_conv_weights, build_activation_quant_params

RUNNER = ROOT / "build" / "runtime" / "spkv2_run"
MODELS_DIR = ROOT / "build" / "models"

import onnx


def get_input_shape(onnx_path: Path) -> list[int]:
    model = onnx.load(str(onnx_path))
    return [int(d.dim_value) if d.dim_value > 0 else 1
            for d in model.graph.input[0].type.tensor_type.shape.dim]


def compile_fp32_spk(onnx_path: Path, spk_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile",
         str(onnx_path), "-o", str(spk_path), "--target", "cpu_generic"],
        check=True, capture_output=True, cwd=str(ROOT),
    )


def compile_int8_spk(onnx_path: Path, spk_path: Path, calibration_data: list[np.ndarray]) -> int:
    profile = load_target_profile("cpu_generic")
    graph = import_onnx(str(onnx_path))
    run_pass_pipeline(graph, pipeline=list(DEFAULT_PIPELINE), enabled=True, profile=profile)
    calibration = calibrate_from_graph(graph, calibration_data, str(onnx_path))
    weight_qp = quantize_conv_weights(graph)
    act_qp = build_activation_quant_params(graph, calibration)
    all_qp = weight_qp + act_qp
    kernel_plan = select_kernel_specs(graph, profile)
    memory_plan = plan_memory(graph, max_arena_bytes=int(profile["memory"]["activation_arena_max"]))
    write_spk(graph, str(spk_path), profile, memory_plan=memory_plan, kernel_plan=kernel_plan, quant_params=all_qp)
    return sum(1 for s in kernel_plan.specs if s.kernel_kind == "int8_im2col_gemm")


def run_spk_timed(spk_path: Path, input_bin: Path, output_bin: Path, runs: int) -> list[float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = subprocess.run(
            [str(RUNNER), str(spk_path), str(input_bin), str(output_bin)],
            capture_output=True,
        )
        t1 = time.perf_counter()
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.decode()[:200]}", file=sys.stderr)
            return []
        times.append(t1 - t0)
    return times


def bench_model(name: str, onnx_path: Path, runs: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not onnx_path.exists():
        print(f"  SKIP: {onnx_path} not found")
        return

    shape = get_input_shape(onnx_path)
    np.random.seed(42)
    x = np.random.randn(*shape).astype(np.float32) * 0.1

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fp32_spk = td / f"{name}_fp32.spk"
        int8_spk = td / f"{name}_int8.spk"
        input_bin = td / "input.bin"
        output_fp32 = td / "output_fp32.bin"
        output_int8 = td / "output_int8.bin"

        input_bin.write_bytes(np.ascontiguousarray(x, dtype=np.float32).tobytes())

        # Compile FP32
        print(f"  Compiling FP32 SPK...")
        compile_fp32_spk(onnx_path, fp32_spk)
        fp32_size = fp32_spk.stat().st_size

        # Compile INT8
        print(f"  Compiling INT8 SPK...")
        cal_data = [np.random.randn(*shape).astype(np.float32) * 0.1 for _ in range(5)]
        n_int8 = compile_int8_spk(onnx_path, int8_spk, cal_data)
        int8_size = int8_spk.stat().st_size

        print(f"  INT8 Conv kernels: {n_int8}")
        print(f"  FP32 SPK size: {fp32_size / 1024 / 1024:.2f} MB")
        print(f"  INT8 SPK size: {int8_size / 1024 / 1024:.2f} MB")
        print(f"  Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")

        # Warmup
        print(f"  Warmup (1 run each)...")
        run_spk_timed(fp32_spk, input_bin, output_fp32, 1)
        run_spk_timed(int8_spk, input_bin, output_int8, 1)

        # Benchmark FP32
        print(f"  Benchmarking FP32 ({runs} runs)...")
        fp32_times = run_spk_timed(fp32_spk, input_bin, output_fp32, runs)
        if not fp32_times:
            print("  FP32 benchmark FAILED")
            return

        # Benchmark INT8
        print(f"  Benchmarking INT8 ({runs} runs)...")
        int8_times = run_spk_timed(int8_spk, input_bin, output_int8, runs)
        if not int8_times:
            print("  INT8 benchmark FAILED")
            return

        fp32_med = np.median(fp32_times) * 1000
        int8_med = np.median(int8_times) * 1000
        fp32_min = np.min(fp32_times) * 1000
        int8_min = np.min(int8_times) * 1000

        print(f"\n  Results:")
        print(f"  {'':20s} {'Median (ms)':>12s} {'Min (ms)':>12s}")
        print(f"  {'FP32':20s} {fp32_med:12.2f} {fp32_min:12.2f}")
        print(f"  {'INT8':20s} {int8_med:12.2f} {int8_min:12.2f}")
        speedup = fp32_med / int8_med
        label = "faster" if speedup > 1 else "slower"
        ratio = speedup if speedup > 1 else 1 / speedup
        print(f"  INT8 is {ratio:.2f}x {label} than FP32")

        # Correctness: cosine similarity
        fp32_out = np.fromfile(str(output_fp32), dtype=np.float32)
        int8_out = np.fromfile(str(output_int8), dtype=np.float32)
        if fp32_out.size == int8_out.size and fp32_out.size > 0:
            cos = float(np.dot(fp32_out, int8_out) / (
                np.linalg.norm(fp32_out) * np.linalg.norm(int8_out) + 1e-10))
            print(f"  Cosine similarity (INT8 vs FP32): {cos:.6f}")
        else:
            print(f"  Output size mismatch: FP32={fp32_out.size}, INT8={int8_out.size}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark INT8 vs FP32")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    args = parser.parse_args()

    print(f"SPINNV2 INT8 vs FP32 Benchmark")
    print(f"Runner: {RUNNER}")
    print(f"Runs: {args.runs}")

    bench_model("ResNet-101", MODELS_DIR / "resnet101.onnx", args.runs)
    bench_model("YOLOv10n", MODELS_DIR / "yolov10n.onnx", args.runs)

    print(f"\n{'='*60}")
    print(f"  Done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
