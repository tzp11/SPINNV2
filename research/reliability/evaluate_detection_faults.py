"""Monte Carlo detection-task fault evaluation for compiled SPINNV2 plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.metrics.task_metrics import detection_failure, DetectionFailure
from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare protected and unprotected runtime fault outcomes for detection models.")
    parser.add_argument("--library", required=True)
    parser.add_argument("--baseline-spk", required=True)
    parser.add_argument("--protected-spk", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--test-images", required=True, help="Directory of test images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-shape", default="1,300,6", help="Expected output shape")
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--events", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_shape = tuple(int(x) for x in args.output_shape.split(","))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    protected_modes = {int(node["node_id"]): node["mode"] for node in plan["nodes"]}
    candidates = _load_candidates(Path(args.ranked_csv))

    image_dir = Path(args.test_images)
    image_paths = sorted(image_dir.glob("*.jpg"))[:args.test_samples]
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))[:args.test_samples]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    generator = random.Random(args.seed)
    totals = {
        "events": 0,
        "unprotected_critical_failures": 0,
        "protected_critical_failures": 0,
        "mitigated_critical_failures": 0,
        "new_protected_failures": 0,
        "selected_node_events": 0,
        "selected_node_unprotected_critical_failures": 0,
        "selected_node_protected_critical_failures": 0,
        "detected_faults": 0,
        "recovered_faults": 0,
        "unrecovered_faults": 0,
        "rerun_count": 0,
        "total_missed_targets": 0,
        "total_false_positives": 0,
        "total_class_changes": 0,
    }
    observations = []

    with RuntimeDriver(args.library, args.baseline_spk) as baseline_rt, \
         RuntimeDriver(args.library, args.protected_spk) as protected_rt:

        baselines = _collect_baselines(baseline_rt, image_paths, args.imgsz, output_shape, args.confidence_threshold)
        print(f"Collected {len(baselines)} baseline images with detections")
        if not baselines:
            raise RuntimeError("No baseline images produced detections")

        weights = [c["activation_bytes"] for c in candidates]

        for seq in range(args.events):
            sample_id, values, baseline_dets = generator.choice(baselines)
            candidate = generator.choices(candidates, weights=weights, k=1)[0]
            event = FaultEvent(
                model_id=plan["model_id"],
                sample_id=sample_id,
                node_id=candidate["runtime_node_id"],
                tensor_id=candidate["runtime_tensor_id"],
                element_index=generator.randrange(candidate["activation_bytes"] // 4),
                bit_index=generator.randrange(32),
                invocation_index=1,
                seed=args.seed + seq,
            )

            raw_unprot, _ = baseline_rt.run(values, event)
            raw_prot, stats = protected_rt.run(values, event)

            unprot_dets = raw_unprot.reshape(output_shape)
            prot_dets = raw_prot.reshape(output_shape)

            unprot_result = detection_failure(
                baseline_dets, unprot_dets[0],
                confidence_threshold=args.confidence_threshold,
                match_iou=args.match_iou,
            )
            prot_result = detection_failure(
                baseline_dets, prot_dets[0],
                confidence_threshold=args.confidence_threshold,
                match_iou=args.match_iou,
            )

            is_selected = event.node_id in protected_modes
            totals["events"] += 1
            totals["unprotected_critical_failures"] += int(unprot_result.critical_failure)
            totals["protected_critical_failures"] += int(prot_result.critical_failure)
            totals["mitigated_critical_failures"] += int(unprot_result.critical_failure and not prot_result.critical_failure)
            totals["new_protected_failures"] += int(prot_result.critical_failure and not unprot_result.critical_failure)
            totals["selected_node_events"] += int(is_selected)
            totals["selected_node_unprotected_critical_failures"] += int(is_selected and unprot_result.critical_failure)
            totals["selected_node_protected_critical_failures"] += int(is_selected and prot_result.critical_failure)
            for name in ("detected_faults", "recovered_faults", "unrecovered_faults", "rerun_count"):
                totals[name] += int(stats[name])
            totals["total_missed_targets"] += unprot_result.missed_targets
            totals["total_false_positives"] += unprot_result.false_positives
            totals["total_class_changes"] += unprot_result.class_changes

            observations.append({
                "sequence": seq,
                "sample_id": sample_id,
                "node_id": event.node_id,
                "tensor_id": event.tensor_id,
                "element_index": event.element_index,
                "bit_index": event.bit_index,
                "protection_mode": protected_modes.get(event.node_id, "none"),
                "unprotected_critical_failure": unprot_result.critical_failure,
                "protected_critical_failure": prot_result.critical_failure,
                "unprotected_severity": unprot_result.severity,
                "protected_severity": prot_result.severity,
                "unprotected_missed": unprot_result.missed_targets,
                "unprotected_fp": unprot_result.false_positives,
                "unprotected_class_changes": unprot_result.class_changes,
                "protected_missed": prot_result.missed_targets,
                "protected_fp": prot_result.false_positives,
                "protected_class_changes": prot_result.class_changes,
                "stats": stats,
            })

            if (seq + 1) % 100 == 0:
                print(f"  {seq+1}/{args.events} events, "
                      f"unprot_fail={totals['unprotected_critical_failures']}, "
                      f"prot_fail={totals['protected_critical_failures']}, "
                      f"mitigated={totals['mitigated_critical_failures']}")

    total_events = totals["events"]
    baseline_failures = totals["unprotected_critical_failures"]
    selected_failures = totals["selected_node_unprotected_critical_failures"]
    report = {
        "model_id": plan["model_id"],
        "task": "detection",
        "platform_profile": plan.get("platform_profile"),
        "fault_model": {
            "type": "one_random_fp32_output_bit_flip_per_inference",
            "runtime_object_prior": "activation_bytes_weighted",
            "bit_prior": "uniform_all_32_bits",
            "seed": args.seed,
        },
        "detection_config": {
            "confidence_threshold": args.confidence_threshold,
            "match_iou": args.match_iou,
        },
        "sampling": {
            "requested_test_prefix": args.test_samples,
            "baseline_images_with_detections": len(baselines),
            "events": total_events,
        },
        "totals": totals,
        "rates": {
            "unprotected_critical_failure_rate": baseline_failures / total_events if total_events else 0.0,
            "protected_critical_failure_rate": totals["protected_critical_failures"] / total_events if total_events else 0.0,
            "observed_risk_reduction_ratio": (
                (baseline_failures - totals["protected_critical_failures"]) / baseline_failures
                if baseline_failures else 0.0
            ),
            "selected_node_event_rate": totals["selected_node_events"] / total_events if total_events else 0.0,
            "selected_node_observed_reduction_ratio": (
                (selected_failures - totals["selected_node_protected_critical_failures"]) / selected_failures
                if selected_failures else 0.0
            ),
        },
        "observations": observations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sampling": report["sampling"], "totals": totals, "rates": report["rates"]}))
    return 0


def _load_candidates(path: Path) -> list[dict]:
    table = pd.read_csv(path)
    table = table[table["retained_after_passes"].astype(str).str.lower().eq("true")]
    return [
        {
            "runtime_node_id": int(row.runtime_node_id),
            "runtime_tensor_id": int(row.runtime_tensor_id),
            "activation_bytes": int(row.activation_bytes),
        }
        for row in table.itertuples(index=False)
    ]


def _collect_baselines(
    runtime: RuntimeDriver,
    image_paths: list[Path],
    imgsz: int,
    output_shape: tuple[int, ...],
    confidence_threshold: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    results = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        arr = np.ascontiguousarray(arr)
        raw, _ = runtime.run(arr)
        dets = raw.reshape(output_shape)[0]
        valid = dets[dets[:, 4] >= confidence_threshold]
        if len(valid) > 0:
            results.append((path.stem, arr, dets))
    return results


if __name__ == "__main__":
    raise SystemExit(main())
