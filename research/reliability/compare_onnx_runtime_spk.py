"""Compare an ONNX model output with the compiled SPINNV2 C runtime output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from research.reliability.metrics.task_metrics import detection_failure
from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-shape", type=int, nargs="+", required=True)
    parser.add_argument("--task", choices=("raw", "detection"), default="raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    values = np.fromfile(args.input, dtype=np.float32).reshape(args.input_shape)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    reference = np.asarray(session.run(None, {session.get_inputs()[0].name: values})[0], dtype=np.float32).reshape(-1)
    with RuntimeDriver(args.library, args.spk) as runtime:
        actual, stats = runtime.run(values)
    delta = np.abs(actual - reference)
    report = {
        "onnx": args.onnx,
        "spk": args.spk,
        "input": args.input,
        "element_count": int(reference.size),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "all_finite": bool(np.all(np.isfinite(actual))),
        "runtime_stats": stats,
    }
    if args.task == "detection":
        consequence = detection_failure(reference.reshape(-1, 6), actual.reshape(-1, 6))
        report["detection_agreement"] = {
            "critical_failure_vs_onnx": consequence.critical_failure,
            "missed_targets": consequence.missed_targets,
            "false_positives": consequence.false_positives,
            "class_changes": consequence.class_changes,
            "localization_degradation": consequence.localization_degradation,
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
