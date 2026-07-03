"""Screen deployment-domain YOLO runtime outputs with task-aware faults."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from research.reliability.injection.bitflip import sample_fault_event
from research.reliability.metrics.task_metrics import target_aware_detection_failure
from research.reliability.prepare_yolo_runtime_sample import letterbox
from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--spk-debug", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--events-per-sample", type=int, default=64)
    parser.add_argument("--sampling", choices=("activation_prior", "stratified"), default="activation_prior")
    parser.add_argument("--injections-per-node", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    candidates = _runtime_candidates(Path(args.spk_debug))
    rng = np.random.default_rng(args.seed)
    weights = np.asarray([entry["activation_bytes"] for entry in candidates], dtype=np.float64)
    weights /= weights.sum()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations = 0
    critical_failures = 0
    eligible_samples = 0
    eligible_sample_ids: list[str] = []
    failure_modes: Counter[str] = Counter()
    reliability_totals: Counter[str] = Counter()
    eligible_records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    with RuntimeDriver(args.library, args.spk) as runtime:
        for image_path in sorted(Path(args.images).glob("*.jpg"))[: args.max_samples]:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            values, preprocessing = letterbox(image)
            baseline, _ = runtime.run(values)
            targets = _load_targets(Path(args.labels) / f"{image_path.stem}.txt", preprocessing)
            clean = target_aware_detection_failure(
                baseline.reshape(-1, 6),
                baseline.reshape(-1, 6),
                targets,
                confidence_threshold=args.confidence_threshold,
            )
            if clean.baseline_true_positives == 0:
                continue
            eligible_samples += 1
            eligible_sample_ids.append(image_path.stem)
            eligible_records.append((image_path.stem, values, baseline, targets))
        with output_path.open("w", encoding="utf-8") as sink:
            if args.sampling == "activation_prior":
                jobs = (
                    (candidates[int(rng.choice(len(candidates), p=weights))], record)
                    for record in eligible_records
                    for _repeat in range(args.events_per_sample)
                )
            else:
                jobs = (
                    (candidate, eligible_records[(index * args.injections_per_node + repeat) % len(eligible_records)])
                    for index, candidate in enumerate(candidates)
                    for repeat in range(args.injections_per_node)
                ) if eligible_records else ()
            for candidate, (sample_id, values, baseline, targets) in jobs:
                event = sample_fault_event(
                    model_id="yolov10n_dior",
                    sample_id=sample_id,
                    node_id=candidate["node_id"],
                    tensor_id=candidate["tensor_id"],
                    element_count=candidate["element_count"],
                    seed=args.seed + observations,
                )
                try:
                    faulted, stats = runtime.run(values, event)
                    consequence = target_aware_detection_failure(
                        baseline.reshape(-1, 6),
                        faulted.reshape(-1, 6),
                        targets,
                        confidence_threshold=args.confidence_threshold,
                    )
                    failure_mode = "task_output_failure" if consequence.critical_failure else "none"
                    critical_failure = consequence.critical_failure
                    severity = consequence.severity
                    baseline_true_positives = consequence.baseline_true_positives
                    execution_error = None
                except RuntimeError as error:
                    stats = {}
                    failure_mode = "controlled_execution_error"
                    critical_failure = True
                    severity = 1.0
                    baseline_true_positives = 0
                    execution_error = str(error)
                sink.write(
                    json.dumps(
                        {
                            "node_id": candidate["node_id"],
                            "tensor_id": candidate["tensor_id"],
                            "activation_bytes": candidate["activation_bytes"],
                            "critical_failure": critical_failure,
                            "severity": severity,
                            "baseline_true_positives": baseline_true_positives,
                            "failure_mode": failure_mode,
                            "execution_error": execution_error,
                            "reliability_stats": stats,
                            "fault_event": asdict(event),
                        }
                    )
                    + "\n"
                )
                observations += 1
                critical_failures += int(critical_failure)
                failure_modes[failure_mode] += 1
                reliability_totals.update(stats)
    summary = {
        "status": "ready" if observations else "no_baseline_true_positives_for_task_aware_screening",
        "candidate_nodes": len(candidates),
        "eligible_samples": eligible_samples,
        "eligible_sample_ids": eligible_sample_ids,
        "observations": observations,
        "critical_failures": critical_failures,
        "failure_modes": dict(failure_modes),
        "reliability_stats": dict(reliability_totals),
        "sampling": args.sampling,
        "events_per_sample": args.events_per_sample if args.sampling == "activation_prior" else None,
        "injections_per_node": args.injections_per_node if args.sampling == "stratified" else None,
        "confidence_threshold": args.confidence_threshold,
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _runtime_candidates(debug_path: Path) -> list[dict]:
    graph = json.loads(debug_path.read_text(encoding="utf-8"))
    tensors = {tensor["id"]: tensor for tensor in graph["tensors"]}
    candidates = []
    for node in graph["nodes"]:
        if not node["outputs"]:
            continue
        tensor = tensors[node["outputs"][0]]
        if tensor["role"] not in {"activation", "output"}:
            continue
        size_bytes = int(tensor["size_bytes"])
        candidates.append(
            {
                "node_id": int(node["id"]),
                "tensor_id": int(tensor["id"]),
                "element_count": size_bytes // 4,
                "activation_bytes": size_bytes,
            }
        )
    if not candidates:
        raise ValueError("SPK debug metadata does not contain injectable output tensors")
    return candidates


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


if __name__ == "__main__":
    raise SystemExit(main())
