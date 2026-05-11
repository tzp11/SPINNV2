#!/usr/bin/env python3
"""Check M6 report thresholds and frozen experiment metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_MODELS = ("resnet101", "yolov10n")
REQUIRED_TARGET = "cpu_ref"
RESNET_TOP1_REQUIRED = True
YOLO_SCORE_MAX_ABS_LIMIT = 1e-5
YOLO_TOP10_MAX_ABS_LIMIT = 1e-2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Input m6_report.json produced by benchmarks/run_m6_models.py.")
    parser.add_argument("--allow-skip-run", action="store_true", help="Allow reports that only compiled models.")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    errors = check_report(report, allow_skip_run=args.allow_skip_run)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("M6 reproducibility check passed")
    return 0


def check_report(report: dict[str, Any], *, allow_skip_run: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("target") != REQUIRED_TARGET:
        errors.append(f"target must be {REQUIRED_TARGET}, got {report.get('target')!r}")

    models = report.get("models", {})
    for model_name in REQUIRED_MODELS:
        if model_name not in models:
            errors.append(f"missing model report: {model_name}")
            continue
        model = models[model_name]
        compile_info = model.get("compile", {})
        if compile_info.get("returncode") != 0:
            errors.append(f"{model_name} compile failed: {compile_info.get('stderr', '')}")
            continue
        if allow_skip_run:
            continue
        runtime = model.get("runtime", {})
        if runtime.get("returncode") != 0:
            errors.append(f"{model_name} runtime failed: {runtime.get('stderr', '')}")
            continue
        compare = model.get("compare", {})
        if model_name == "resnet101" and compare.get("top1_equal") is not RESNET_TOP1_REQUIRED:
            errors.append("resnet101 top1 must match ORT")
        if model_name == "yolov10n":
            score_error = float(compare.get("score_max_abs_error", float("inf")))
            top10_error = float(compare.get("top10_max_abs_error", float("inf")))
            if score_error > YOLO_SCORE_MAX_ABS_LIMIT:
                errors.append(f"yolov10n score max error {score_error} exceeds {YOLO_SCORE_MAX_ABS_LIMIT}")
            if top10_error > YOLO_TOP10_MAX_ABS_LIMIT:
                errors.append(f"yolov10n top10 max error {top10_error} exceeds {YOLO_TOP10_MAX_ABS_LIMIT}")

        memory = model.get("memory", {})
        naive = int(memory.get("naive_activation_bytes", 0))
        planned = int(memory.get("planned_activation_bytes", 0))
        if naive <= 0 or planned <= 0:
            errors.append(f"{model_name} memory metrics are missing")
        elif planned >= naive:
            errors.append(f"{model_name} planned memory must be smaller than naive memory")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
