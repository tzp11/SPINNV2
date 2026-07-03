"""Render a DIOR detection fault and its protected recovery as three panels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.metrics.task_metrics import target_aware_detection_failure
from research.reliability.prepare_yolo_runtime_sample import letterbox
from research.reliability.runtime_driver import RuntimeDriver
from research.reliability.screen_yolo_runtime_faults import _load_targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize one task-critical YOLO runtime fault and recovery.")
    parser.add_argument("--library", required=True)
    parser.add_argument("--baseline-spk", required=True)
    parser.add_argument("--protected-spk", required=True)
    parser.add_argument("--injections", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=30)
    args = parser.parse_args()

    selected = {int(node["node_id"]) for node in json.loads(Path(args.plan).read_text(encoding="utf-8"))["nodes"]}
    source_record = _select_fault(Path(args.injections), selected)
    event = FaultEvent(**source_record["fault_event"])
    image_path = Path(args.images) / f"{event.sample_id}.jpg"
    label_path = Path(args.labels) / f"{event.sample_id}.txt"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    values, preprocessing = letterbox(image)
    targets = _load_targets(label_path, preprocessing)
    with RuntimeDriver(args.library, args.baseline_spk) as baseline_runtime:
        clean, _ = baseline_runtime.run(values)
        faulted, unprotected_stats = baseline_runtime.run(values, event)
    with RuntimeDriver(args.library, args.protected_spk) as protected_runtime:
        recovered, protected_stats = protected_runtime.run(values, event)

    clean_values = clean.reshape((-1, 6))
    faulted_values = faulted.reshape((-1, 6))
    recovered_values = recovered.reshape((-1, 6))
    fault_consequence = target_aware_detection_failure(
        clean_values, faulted_values, targets, confidence_threshold=args.confidence_threshold
    )
    recovered_consequence = target_aware_detection_failure(
        clean_values, recovered_values, targets, confidence_threshold=args.confidence_threshold
    )
    canvas = cv2.cvtColor((values[0].transpose(1, 2, 0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    panels = [
        _draw_panel(canvas, targets, clean_values, "Clean baseline", args.confidence_threshold, args.max_detections),
        _draw_panel(canvas, targets, faulted_values, "Bit-flip fault", args.confidence_threshold, args.max_detections),
        _draw_panel(canvas, targets, recovered_values, "Protected recovery", args.confidence_threshold, args.max_detections),
    ]
    combined = cv2.hconcat(panels)
    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), combined)
    report = {
        "image": str(image_path),
        "fault_event": asdict(event),
        "source_failure_mode": source_record["failure_mode"],
        "unprotected_consequence": asdict(fault_consequence),
        "protected_consequence": asdict(recovered_consequence),
        "unprotected_stats": unprotected_stats,
        "protected_stats": protected_stats,
        "output_image": str(output_image),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _select_fault(path: Path, selected_nodes: set[int]) -> dict:
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if (
                record["failure_mode"] == "task_output_failure"
                and int(record["node_id"]) in selected_nodes
            ):
                return record
    raise ValueError("no protected task-output failure found in injection log")


def _draw_panel(
    canvas: np.ndarray,
    targets: np.ndarray,
    detections: np.ndarray,
    title: str,
    confidence_threshold: float,
    max_detections: int,
) -> np.ndarray:
    output = canvas.copy()
    for target in targets:
        x1, y1, x2, y2 = [int(round(value)) for value in target[:4]]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 190, 0), 2)
    active = detections[detections[:, 4] >= confidence_threshold]
    active = active[np.argsort(-active[:, 4])][:max_detections]
    for prediction in active:
        x1, y1, x2, y2 = [int(round(value)) for value in prediction[:4]]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 130, 0), 2)
        cv2.putText(
            output,
            f"{int(prediction[5])}:{prediction[4]:.2f}",
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 130, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(output, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
