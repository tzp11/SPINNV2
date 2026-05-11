#!/usr/bin/env python3
"""Run generated-model ONNX -> SPK -> Runtime numerical validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.model_zoo import ALL_MODELS, SMALL_MODELS, TOY_MODELS, make_input, output_tolerance, write_model


MODEL_GROUPS = {
    "toy": TOY_MODELS,
    "all-small": SMALL_MODELS,
    "all": ALL_MODELS,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="toy", help="Model name or group: toy, all-small, all.")
    parser.add_argument("--all-small", action="store_true", help="Run the small validation set.")
    parser.add_argument("--all", action="store_true", help="Run the complete generated validation set.")
    parser.add_argument("--out-dir", default="build/e2e_validation")
    parser.add_argument("--target", default="cpu_ref")
    parser.add_argument("--runtime-runner", default="build/runtime/spkv2_run")
    args = parser.parse_args()

    group = "all" if args.all else "all-small" if args.all_small else args.model
    selected = list(MODEL_GROUPS.get(group, (group,)))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)

    report = {"target": args.target, "models": {}}
    for name in selected:
        report["models"][name] = run_one(name, out_root / name, args.target, Path(args.runtime_runner))

    report_path = out_root / "e2e_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def run_one(name: str, out_dir: Path, target: str, runner: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.onnx"
    spk_path = out_dir / f"{name}.spk"
    input_path = out_dir / "input.bin"
    output_path = out_dir / "output.bin"
    write_model(name, model_path)

    subprocess.run(
        [
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
        ],
        check=True,
    )
    x = make_input(name)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())
    expected = run_ort(model_path, x)
    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)
    actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
    diff = np.abs(actual - expected)
    rtol, atol = output_tolerance(name)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    debug = json.loads(spk_path.with_suffix(spk_path.suffix + ".json").read_text(encoding="utf-8"))
    return {
        "nodes": len(debug["nodes"]),
        "spk_size_bytes": spk_path.stat().st_size,
        "naive_activation_bytes": debug["memory"]["naive_activation_bytes"],
        "planned_activation_bytes": debug["memory"]["planned_activation_bytes"],
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
    }


def run_ort(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return np.asarray(session.run(None, {session.get_inputs()[0].name: input_array})[0], dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
