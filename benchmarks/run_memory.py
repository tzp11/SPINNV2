#!/usr/bin/env python3
"""Collect memory-planning metrics for generated validation models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.model_zoo import write_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mnist,lenet,resnet18")
    parser.add_argument("--out-dir", default="build/memory_benchmark")
    parser.add_argument("--target", default="cpu_ref")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"target": args.target, "models": {}}
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        report["models"][name] = run_one(name, out_root / name, args.target)

    report_path = out_root / "memory_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


def run_one(name: str, out_dir: Path, target: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.onnx"
    spk_path = out_dir / f"{name}.spk"
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
        ],
        check=True,
    )
    debug = json.loads(spk_path.with_suffix(spk_path.suffix + ".json").read_text(encoding="utf-8"))
    return {
        "spk_size_bytes": spk_path.stat().st_size,
        "naive_activation_bytes": debug["memory"]["naive_activation_bytes"],
        "planned_activation_bytes": debug["memory"]["planned_activation_bytes"],
        "memory_reduction_ratio": debug["memory"]["memory_reduction_ratio"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
