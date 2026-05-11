#!/usr/bin/env python3
"""Run M6 large-model compile/runtime validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


DEFAULT_MODELS = {
    "resnet101": "/home/tzp/work/SPINN/SPINN/run_time/resnet101.onnx",
    "yolov10n": "/home/tzp/work/SPINN/SPINN/run_time/yolov10n.onnx",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(DEFAULT_MODELS), action="append")
    parser.add_argument("--out-dir", default="build/m6_models")
    parser.add_argument("--target", default="cpu_ref")
    parser.add_argument("--runtime-runner", default="build/runtime/spkv2_run")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    selected = args.model or sorted(DEFAULT_MODELS)
    report = {"target": args.target, "models": {}}
    out_root = Path(args.out_dir)
    for name in selected:
        report["models"][name] = run_model(
            name,
            Path(DEFAULT_MODELS[name]),
            out_root / name,
            args.target,
            Path(args.runtime_runner),
            skip_run=args.skip_run,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "m6_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def run_model(
    name: str,
    model_path: Path,
    out_dir: Path,
    target: str,
    runner: Path,
    *,
    skip_run: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "model_path": str(model_path),
        "op_counts": op_counts(model_path),
    }
    spk_path = out_dir / f"{name}.spk"
    compile_cmd = [
        "python",
        "-m",
        "spinnv2.compiler",
        "compile",
        str(model_path),
        "-o",
        str(spk_path),
        "--target",
        target,
        "--memory-plan-csv",
        str(out_dir / "memory_plan.csv"),
        "--pass-stats-json",
        str(out_dir / "pass_stats.json"),
    ]
    started = time.perf_counter()
    compile_proc = subprocess.run(compile_cmd, text=True, capture_output=True)
    report["compile"] = {
        "returncode": compile_proc.returncode,
        "time_s": time.perf_counter() - started,
        "stdout": compile_proc.stdout,
        "stderr": compile_proc.stderr,
    }
    if compile_proc.returncode != 0 or skip_run:
        return report

    debug = json.loads(spk_path.with_suffix(spk_path.suffix + ".json").read_text(encoding="utf-8"))
    report["spk_size_bytes"] = spk_path.stat().st_size
    report["memory"] = debug.get("memory", {})

    input_array = make_input(model_path)
    input_path = out_dir / "input.bin"
    input_path.write_bytes(np.ascontiguousarray(input_array).tobytes())

    ort_output = run_ort(model_path, input_array)
    ort_path = out_dir / "ort_output.bin"
    ort_path.write_bytes(np.ascontiguousarray(ort_output).astype(np.float32).tobytes())

    output_path = out_dir / "output.bin"
    started = time.perf_counter()
    run_proc = subprocess.run(
        [str(runner), str(spk_path), str(input_path), str(output_path)],
        text=True,
        capture_output=True,
    )
    report["runtime"] = {
        "returncode": run_proc.returncode,
        "time_s": time.perf_counter() - started,
        "stdout": run_proc.stdout,
        "stderr": run_proc.stderr,
    }
    if run_proc.returncode != 0:
        return report

    sp_output = np.fromfile(output_path, dtype=np.float32).reshape(ort_output.shape)
    report["compare"] = compare_outputs(name, sp_output, ort_output)
    return report


def op_counts(model_path: Path) -> dict[str, int]:
    model = onnx.load(model_path)
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def make_input(model_path: Path) -> np.ndarray:
    model = onnx.load(model_path)
    shape = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        shape.append(int(dim.dim_value))
    total = int(np.prod(shape))
    if shape[-1] == 640:
        return np.linspace(0.0, 1.0, num=total, dtype=np.float32).reshape(shape)
    return np.linspace(-1.0, 1.0, num=total, dtype=np.float32).reshape(shape)


def run_ort(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: input_array})[0]
    return np.asarray(output, dtype=np.float32)


def compare_outputs(name: str, sp_output: np.ndarray, ort_output: np.ndarray) -> dict:
    diff = np.abs(sp_output - ort_output)
    result = {
        "shape": list(ort_output.shape),
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
    }
    if name == "resnet101":
        result["top1_equal"] = bool(sp_output.reshape(-1).argmax() == ort_output.reshape(-1).argmax())
    if name.startswith("yolo"):
        sp = sp_output.reshape(-1, 6)
        ort = ort_output.reshape(-1, 6)
        result["score_max_abs_error"] = float(np.abs(sp[:, 4] - ort[:, 4]).max())
        result["score_mean_abs_error"] = float(np.abs(sp[:, 4] - ort[:, 4]).mean())
        for count in (10, 20, 50):
            rows = min(count, sp.shape[0])
            row_diff = np.abs(sp[:rows] - ort[:rows])
            result[f"top{rows}_max_abs_error"] = float(row_diff.max())
            result[f"top{rows}_mean_abs_error"] = float(row_diff.mean())
            result[f"top{rows}_same_class_count"] = int(
                np.sum(sp[:rows, 5].astype(np.int32) == ort[:rows, 5].astype(np.int32))
            )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
