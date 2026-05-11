#!/usr/bin/env python3
"""Run generated-C deployment validation for generated models."""

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

from tests.e2e.model_zoo import make_input, output_tolerance, write_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mnist,lenet")
    parser.add_argument("--out-dir", default="build/codegen_validation")
    parser.add_argument("--target", default="cpu_generic")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"target": args.target, "models": {}}
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        report["models"][name] = run_one(name, out_root / name, args.target)

    report_path = out_root / "codegen_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def run_one(name: str, out_dir: Path, target: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.onnx"
    spk_path = out_dir / f"{name}.spk"
    gen_dir = out_dir / "generated"
    build_dir = out_dir / "build"
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
            "--external-inputs",
            "--external-outputs",
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "codegen",
            str(spk_path),
            "--out-dir",
            str(gen_dir),
            "--name",
            name,
            "--runtime-dir",
            "runtime",
        ],
        check=True,
    )
    subprocess.run(["cmake", "-S", str(gen_dir), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)
    x = make_input(name)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())
    subprocess.run([str(build_dir / f"{name}_main_test"), str(input_path), str(output_path)], check=True)
    actual = np.fromfile(output_path, dtype=np.float32)
    expected = run_ort(model_path, x).reshape(-1)
    rtol, atol = output_tolerance(name)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    diff = np.abs(actual - expected)
    return {
        "generated_c_size_bytes": (gen_dir / f"{name}.c").stat().st_size,
        "runtime_binary_size_bytes": (build_dir / f"{name}_main_test").stat().st_size,
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
    }


def run_ort(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return np.asarray(session.run(None, {session.get_inputs()[0].name: input_array})[0], dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
