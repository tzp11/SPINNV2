"""Evaluate task-utility consistency guards for YOLO runtime faults."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from research.reliability.evaluate_yolo_task_output_guard import _event_from_record, _guard_reasons, _load_targets
from research.reliability.metrics.task_metrics import target_aware_detection_failure
from research.reliability.prepare_yolo_runtime_sample import letterbox
from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--coord-margin", type=float, default=64.0)
    parser.add_argument("--count-delta-ratio", type=float, default=0.25)
    parser.add_argument("--confidence-sum-delta-ratio", type=float, default=0.30)
    parser.add_argument("--class-hist-delta-ratio", type=float, default=0.30)
    args = parser.parse_args()

    records = _load_event_records(Path(args.events_jsonl), args.max_events)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    current_reason_counts = Counter()
    utility_reason_counts = Counter()
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
                current_reasons = _guard_reasons(
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
                current_reasons = ["controlled_execution_error"]
                execution_error = str(error)
                execution_error_critical = True
            utility_reasons = _utility_reasons(baseline, faulted, args)
            current_guarded = bool(current_reasons)
            utility_guarded = bool(current_reasons or utility_reasons)
            current_reason_counts.update(current_reasons)
            utility_reason_counts.update(utility_reasons)
            totals["current_guard_triggered"] += int(current_guarded)
            totals["utility_guard_triggered"] += int(utility_guarded)

            fault_consequence = target_aware_detection_failure(
                baseline.reshape(-1, 6),
                faulted.reshape(-1, 6),
                targets,
                confidence_threshold=args.confidence_threshold,
            )
            current_recovered = baseline if current_guarded else faulted
            utility_recovered = baseline if utility_guarded else faulted
            current_consequence = target_aware_detection_failure(
                baseline.reshape(-1, 6),
                current_recovered.reshape(-1, 6),
                targets,
                confidence_threshold=args.confidence_threshold,
            )
            utility_consequence = target_aware_detection_failure(
                baseline.reshape(-1, 6),
                utility_recovered.reshape(-1, 6),
                targets,
                confidence_threshold=args.confidence_threshold,
            )
            faulted_critical = bool(fault_consequence.critical_failure or execution_error_critical)
            totals["faulted_critical_failures"] += int(faulted_critical)
            totals["current_guarded_critical_failures"] += int(current_consequence.critical_failure)
            totals["utility_guarded_critical_failures"] += int(utility_consequence.critical_failure)
            totals["current_mitigated_failures"] += int(faulted_critical and not current_consequence.critical_failure)
            totals["utility_mitigated_failures"] += int(faulted_critical and not utility_consequence.critical_failure)
            totals["current_false_positive_clean_guard"] += int(
                bool(
                    _guard_reasons(
                        baseline,
                        image_size=args.image_size,
                        num_classes=args.num_classes,
                        coord_margin=args.coord_margin,
                    )
                )
            )
            totals["utility_false_positive_clean_guard"] += int(
                bool(_utility_reasons(baseline, baseline, args))
            )
            observations.append(
                {
                    "sequence": index,
                    "fault_event": asdict(event),
                    "current_guarded": current_guarded,
                    "utility_guarded": utility_guarded,
                    "current_guard_reasons": current_reasons,
                    "utility_guard_reasons": utility_reasons,
                    "execution_error": execution_error,
                    "faulted_critical_failure": faulted_critical,
                    "current_guarded_critical_failure": bool(current_consequence.critical_failure),
                    "utility_guarded_critical_failure": bool(utility_consequence.critical_failure),
                    "faulted_severity": float(fault_consequence.severity),
                    "utility_guarded_severity": float(utility_consequence.severity),
                    "runtime_stats": {key: int(stats.get(key, 0)) for key in ("detected_faults", "recovered_faults", "unrecovered_faults", "rerun_count")},
                }
            )
            if totals["events"] % 1000 == 0:
                print(json.dumps({"events": totals["events"]}), flush=True)

    events = totals["events"]
    faulted_failures = totals["faulted_critical_failures"]
    current_failures = totals["current_guarded_critical_failures"]
    utility_failures = totals["utility_guarded_critical_failures"]
    current_reduction = (faulted_failures - current_failures) / faulted_failures if faulted_failures else 0.0
    utility_reduction = (faulted_failures - utility_failures) / faulted_failures if faulted_failures else 0.0
    current_trigger = totals["current_guard_triggered"] / events if events else 0.0
    utility_trigger = totals["utility_guard_triggered"] / events if events else 0.0
    report = {
        "status": "measured",
        "mechanism": "task_utility_consistency_guard",
        "source_events": str(args.events_jsonl),
        "guard_policy": {
            "base_checks": [
                "nonfinite",
                "confidence_out_of_range",
                "class_out_of_range",
                "box_coordinate_out_of_range",
                "invalid_box_order",
                "controlled_execution_error",
            ],
            "utility_checks": [
                "detection_count_delta",
                "high_confidence_count_delta",
                "confidence_sum_delta",
                "class_histogram_delta",
                "box_area_distribution_delta",
            ],
            "recovery": "rerun_clean_output_under_single-transient-fault_model",
        },
        "totals": dict(totals),
        "current_guard_reasons": dict(current_reason_counts),
        "utility_guard_reasons": dict(utility_reason_counts),
        "rates": {
            "current_reduction_ratio": current_reduction,
            "utility_reduction_ratio": utility_reduction,
            "reduction_gain_vs_current": utility_reduction - current_reduction,
            "current_guard_trigger_rate": current_trigger,
            "utility_guard_trigger_rate": utility_trigger,
            "trigger_reduction_vs_current": (
                (current_trigger - utility_trigger) / current_trigger if current_trigger else 0.0
            ),
            "current_clean_false_positive_rate": totals["current_false_positive_clean_guard"] / events if events else 0.0,
            "utility_clean_false_positive_rate": totals["utility_false_positive_clean_guard"] / events if events else 0.0,
        },
        "observations": observations,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".csv").write_text(_summary_csv(report), encoding="utf-8")
    print(json.dumps({"totals": report["totals"], "rates": report["rates"]}, ensure_ascii=False))
    return 0


def _utility_reasons(baseline: np.ndarray, faulted: np.ndarray, args) -> list[str]:
    base = baseline.reshape(-1, 6)
    fault = faulted.reshape(-1, 6)
    if not np.isfinite(fault).all():
        return ["nonfinite_utility"]
    base_keep = base[base[:, 4] >= args.confidence_threshold]
    fault_keep = fault[fault[:, 4] >= args.confidence_threshold]
    base_high = base[base[:, 4] >= args.high_confidence_threshold]
    fault_high = fault[fault[:, 4] >= args.high_confidence_threshold]
    reasons = []
    if _count_delta(len(base_keep), len(fault_keep), args.count_delta_ratio, minimum=5):
        reasons.append("detection_count_delta")
    if _count_delta(len(base_high), len(fault_high), args.count_delta_ratio, minimum=3):
        reasons.append("high_confidence_count_delta")
    base_conf = float(base_keep[:, 4].sum()) if len(base_keep) else 0.0
    fault_conf = float(fault_keep[:, 4].sum()) if len(fault_keep) else 0.0
    if abs(base_conf - fault_conf) > max(1.0, abs(base_conf) * args.confidence_sum_delta_ratio):
        reasons.append("confidence_sum_delta")
    if _class_hist_delta(base_keep, fault_keep, args.num_classes) > max(3.0, len(base_keep) * args.class_hist_delta_ratio):
        reasons.append("class_histogram_delta")
    if _area_distribution_delta(base_keep, fault_keep):
        reasons.append("box_area_distribution_delta")
    return reasons


def _count_delta(base_count: int, fault_count: int, ratio: float, *, minimum: int) -> bool:
    return abs(base_count - fault_count) > max(minimum, int(round(base_count * ratio)))


def _class_hist_delta(base: np.ndarray, fault: np.ndarray, num_classes: int) -> float:
    base_hist = np.bincount(np.clip(base[:, 5].round().astype(int), 0, num_classes - 1), minlength=num_classes) if len(base) else np.zeros(num_classes)
    fault_hist = np.bincount(np.clip(fault[:, 5].round().astype(int), 0, num_classes - 1), minlength=num_classes) if len(fault) else np.zeros(num_classes)
    return float(np.abs(base_hist - fault_hist).sum())


def _area_distribution_delta(base: np.ndarray, fault: np.ndarray) -> bool:
    if len(base) == 0 or len(fault) == 0:
        return False
    base_area = np.maximum(0.0, base[:, 2] - base[:, 0]) * np.maximum(0.0, base[:, 3] - base[:, 1])
    fault_area = np.maximum(0.0, fault[:, 2] - fault[:, 0]) * np.maximum(0.0, fault[:, 3] - fault[:, 1])
    base_median = float(np.median(base_area)) if base_area.size else 0.0
    fault_median = float(np.median(fault_area)) if fault_area.size else 0.0
    return abs(base_median - fault_median) > max(128.0, abs(base_median) * 1.0)


def _load_event_records(path: Path, max_events: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= max_events:
                    break
    return records


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
        samples[sample_id] = {"input": values, "baseline": baseline, "targets": targets}
    return samples


def _summary_csv(report: dict) -> str:
    rates = report["rates"]
    return (
        "method,current_reduction_ratio,utility_reduction_ratio,reduction_gain_vs_current,"
        "current_guard_trigger_rate,utility_guard_trigger_rate,trigger_reduction_vs_current,"
        "current_clean_false_positive_rate,utility_clean_false_positive_rate\n"
        f"task_utility_consistency_guard,{rates['current_reduction_ratio']},{rates['utility_reduction_ratio']},"
        f"{rates['reduction_gain_vs_current']},{rates['current_guard_trigger_rate']},{rates['utility_guard_trigger_rate']},"
        f"{rates['trigger_reduction_vs_current']},{rates['current_clean_false_positive_rate']},"
        f"{rates['utility_clean_false_positive_rate']}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
