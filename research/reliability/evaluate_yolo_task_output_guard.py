"""Evaluate a lightweight task-output guard for YOLO runtime faults.

The guard checks final detection tensors for semantic invalidity and simulates
rerun recovery for transient faults by replacing guarded invalid outputs with
the cached clean output for the same sample.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.metrics.task_metrics import target_aware_detection_failure
from research.reliability.prepare_yolo_runtime_sample import letterbox
from research.reliability.runtime_driver import RuntimeDriver


def _load_targets(label_path: Path, preprocessing: dict) -> np.ndarray:
    height, width = preprocessing["original_shape"]
    ratio = float(preprocessing["scale"])
    left, top = preprocessing["padding_xy"]
    targets = []
    for line in label_path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, box_width, box_height = [float(value) for value in line.split()]
        x1 = (cx - box_width / 2.0) * width * ratio + left
        y1 = (cy - box_height / 2.0) * height * ratio + top
        x2 = (cx + box_width / 2.0) * width * ratio + left
        y2 = (cy + box_height / 2.0) * height * ratio + top
        targets.append([x1, y1, x2, y2, class_id])
    return np.asarray(targets, dtype=np.float32).reshape((-1, 5))


def _guard_reasons(output: np.ndarray, *, image_size: int, num_classes: int, coord_margin: float) -> list[str]:
    values = output.reshape(-1, 6)
    reasons: list[str] = []
    if not np.isfinite(values).all():
        reasons.append("nonfinite")
        return reasons
    boxes = values[:, :4]
    conf = values[:, 4]
    cls = values[:, 5]
    if np.any(conf < -1e-5) or np.any(conf > 1.0 + 1e-5):
        reasons.append("confidence_out_of_range")
    if np.any(cls < -0.5) or np.any(cls > (num_classes - 1) + 0.5):
        reasons.append("class_out_of_range")
    if np.any(boxes < -coord_margin) or np.any(boxes > image_size + coord_margin):
        reasons.append("box_coordinate_out_of_range")
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(widths < -1e-5) or np.any(heights < -1e-5):
        reasons.append("invalid_box_order")
    return reasons


def _event_from_record(record: dict[str, Any]) -> FaultEvent:
    event = record["fault_event"]
    return FaultEvent(
        model_id=str(event["model_id"]),
        sample_id=str(event["sample_id"]),
        node_id=int(event["node_id"]),
        tensor_id=int(event["tensor_id"]),
        element_index=int(event["element_index"]),
        bit_index=int(event["bit_index"]),
        invocation_index=int(event.get("invocation_index", 1)),
        seed=int(event["seed"]),
    )


def _load_event_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prepare_samples(records: list[dict[str, Any]], images: Path, labels: Path, runtime: RuntimeDriver) -> dict[str, dict]:
    sample_ids = sorted({record["fault_event"]["sample_id"] for record in records})
    samples = {}
    for sample_id in sample_ids:
        image = cv2.imread(str(images / f"{sample_id}.jpg"))
        if image is None:
            continue
        values, preprocessing = letterbox(image)
        baseline, _ = runtime.run(values)
        targets = _load_targets(labels / f"{sample_id}.txt", preprocessing)
        samples[sample_id] = {
            "input": values,
            "baseline": baseline,
            "targets": targets,
        }
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--coord-margin", type=float, default=64.0)
    args = parser.parse_args()

    records = _load_event_records(Path(args.events_jsonl))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    reason_counts = Counter()
    observations = []
    with RuntimeDriver(args.library, args.spk) as runtime:
        samples = _prepare_samples(records, Path(args.images), Path(args.labels), runtime)
        for index, record in enumerate(records):
            event = _event_from_record(record)
            sample = samples.get(event.sample_id)
            if not sample:
                continue
            totals["events"] += 1
            baseline = sample["baseline"]
            targets = sample["targets"]
            try:
                faulted, stats = runtime.run(sample["input"], event)
                reasons = _guard_reasons(
                    faulted,
                    image_size=args.image_size,
                    num_classes=args.num_classes,
                    coord_margin=args.coord_margin,
                )
                execution_error = None
                execution_error_critical = False
            except RuntimeError as error:
                faulted = baseline
                stats = {}
                reasons = ["controlled_execution_error"]
                execution_error = str(error)
                execution_error_critical = True
            guarded = bool(reasons)
            if guarded:
                totals["guard_triggered"] += 1
                reason_counts.update(reasons)
                recovered_output = baseline
            else:
                recovered_output = faulted
            fault_consequence = target_aware_detection_failure(
                baseline.reshape(-1, 6),
                faulted.reshape(-1, 6),
                targets,
                confidence_threshold=args.confidence_threshold,
            )
            guarded_consequence = target_aware_detection_failure(
                baseline.reshape(-1, 6),
                recovered_output.reshape(-1, 6),
                targets,
                confidence_threshold=args.confidence_threshold,
            )
            faulted_critical = bool(fault_consequence.critical_failure or execution_error_critical)
            totals["faulted_critical_failures"] += int(faulted_critical)
            totals["guarded_critical_failures"] += int(guarded_consequence.critical_failure)
            totals["mitigated_failures"] += int(
                faulted_critical and not guarded_consequence.critical_failure
            )
            totals["false_positive_clean_guard"] += int(
                bool(
                    _guard_reasons(
                        baseline,
                        image_size=args.image_size,
                        num_classes=args.num_classes,
                        coord_margin=args.coord_margin,
                    )
                )
            )
            for name in ("detected_faults", "recovered_faults", "unrecovered_faults", "rerun_count"):
                totals[f"runtime_{name}"] += int(stats.get(name, 0))
            observations.append(
                {
                    "sequence": index,
                    "fault_event": asdict(event),
                    "guarded": guarded,
                    "guard_reasons": reasons,
                    "execution_error": execution_error,
                    "faulted_critical_failure": faulted_critical,
                    "guarded_critical_failure": bool(guarded_consequence.critical_failure),
                    "faulted_severity": float(fault_consequence.severity),
                    "guarded_severity": float(guarded_consequence.severity),
                }
            )

    events = totals["events"]
    faulted_failures = totals["faulted_critical_failures"]
    guarded_failures = totals["guarded_critical_failures"]
    report = {
        "status": "ready",
        "mechanism": "task_output_guard_rerun",
        "source_events": str(args.events_jsonl),
        "fault_model": "replay_existing_runtime_fault_events",
        "guard_policy": {
            "checks": [
                "nonfinite",
                "confidence_out_of_range",
                "class_out_of_range",
                "box_coordinate_out_of_range",
                "invalid_box_order",
                "controlled_execution_error",
            ],
            "coord_margin": args.coord_margin,
            "image_size": args.image_size,
            "num_classes": args.num_classes,
            "recovery": "rerun_clean_output_under_single-transient-fault_model",
        },
        "totals": dict(totals),
        "guard_reasons": dict(reason_counts),
        "rates": {
            "guard_trigger_rate": totals["guard_triggered"] / events if events else 0.0,
            "faulted_critical_failure_rate": faulted_failures / events if events else 0.0,
            "guarded_critical_failure_rate": guarded_failures / events if events else 0.0,
            "observed_reduction_ratio": (
                (faulted_failures - guarded_failures) / faulted_failures if faulted_failures else 0.0
            ),
            "clean_false_positive_rate_upper_bound": totals["false_positive_clean_guard"] / events if events else 0.0,
        },
        "observations": observations,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"totals": report["totals"], "rates": report["rates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
