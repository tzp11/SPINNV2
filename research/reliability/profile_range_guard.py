"""Calibrate and assess range guards on selected ResNet50 runtime outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights, resnet50
import yaml

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.injection.torch_injector import capture_candidate_output, infer_with_fault_and_capture
from research.reliability.profiling.range_guard import RangeGuardRecord, calibrate_scalar_bounds, outside_scalar_bounds
from research.reliability.profiling.risk_profile import wilson_interval


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile scalar range-guard coverage for high-risk runtime objects.")
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--injections", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--holdout-samples", type=int, default=32)
    parser.add_argument("--margin-ratio", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    dataset = EuroSAT(root=str(data_root), transform=weights.transforms())
    calibration = Subset(dataset, splits["val"][: args.calibration_samples])
    holdout = Subset(
        dataset,
        splits["val"][args.calibration_samples : args.calibration_samples + args.holdout_samples],
    )
    test = Subset(dataset, splits["test"])
    device = torch.device(args.device)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(_load_state_dict(Path(args.checkpoint)))
    model.to(device).eval()

    ranked = pd.read_csv(args.ranked_csv)
    retained = ranked[ranked["retained_after_passes"].astype(str).str.lower().eq("true")].head(args.top_k)
    observations = _read_selected_observations(Path(args.injections), set(int(value) for value in retained["node_id"]))
    records = []
    for position, row in enumerate(retained.itertuples(index=False), start=1):
        node_id = int(row.node_id)
        minima = []
        maxima = []
        for image, _label in calibration:
            values = capture_candidate_output(
                model,
                image.unsqueeze(0).to(device),
                module_name=row.module_name,
                invocation_index=int(row.invocation_index),
            )
            minima.append(float(values.min().item()))
            maxima.append(float(values.max().item()))
        bounds = calibrate_scalar_bounds(minima, maxima, margin_ratio=args.margin_ratio)
        false_positives = 0
        for image, _label in holdout:
            values = capture_candidate_output(
                model,
                image.unsqueeze(0).to(device),
                module_name=row.module_name,
                invocation_index=int(row.invocation_index),
            ).cpu().numpy()
            false_positives += int(outside_scalar_bounds(values, bounds))
        detected = 0
        detected_critical = 0
        critical = 0
        node_observations = observations.get(node_id, [])
        for observation in node_observations:
            sample_index = int(observation["fault_event"]["sample_id"].rsplit("_", 1)[1])
            image, _label = test[sample_index]
            _output, activation = infer_with_fault_and_capture(
                model,
                image.unsqueeze(0).to(device),
                module_name=row.module_name,
                event=FaultEvent(**observation["fault_event"]),
            )
            was_detected = outside_scalar_bounds(activation.cpu().numpy(), bounds)
            detected += int(was_detected)
            if observation["critical_failure"]:
                critical += 1
                detected_critical += int(was_detected)
        coverage_ci = wilson_interval(detected_critical, critical) if critical else (0.0, 0.0)
        false_positive_ci = wilson_interval(false_positives, len(holdout)) if len(holdout) else (0.0, 0.0)
        records.append(
            RangeGuardRecord(
                node_id=node_id,
                runtime_node_id=int(row.runtime_node_id),
                runtime_tensor_id=int(row.runtime_tensor_id),
                module_name=row.module_name,
                invocation_index=int(row.invocation_index),
                lower_bound=bounds.lower_bound,
                upper_bound=bounds.upper_bound,
                critical_faults=critical,
                detected_critical_faults=detected_critical,
                critical_coverage=(detected_critical / critical) if critical else 0.0,
                critical_coverage_ci_low=coverage_ci[0],
                critical_coverage_ci_high=coverage_ci[1],
                injected_faults=len(node_observations),
                detected_faults=detected,
                false_positive_samples=false_positives,
                clean_holdout_samples=len(holdout),
                false_positive_rate=(false_positives / len(holdout)) if len(holdout) else 0.0,
                false_positive_ci_low=false_positive_ci[0],
                false_positive_ci_high=false_positive_ci[1],
            )
        )
        print(
            json.dumps(
                {
                    "profiled": position,
                    "candidate": f"{row.module_name}@{int(row.invocation_index)}",
                    "critical_coverage": records[-1].critical_coverage,
                    "critical_coverage_ci_low": records[-1].critical_coverage_ci_low,
                    "false_positive_rate": records[-1].false_positive_rate,
                    "false_positive_ci_high": records[-1].false_positive_ci_high,
                }
            ),
            flush=True,
        )
    report = {
        "model_id": "resnet50_eurosat_gpu_prevalidation",
        "status": "python_semantic_calibration_pending_runtime_threshold_validation",
        "calibration_samples": len(calibration),
        "clean_holdout_samples": len(holdout),
        "margin_ratio": args.margin_ratio,
        "records": [record.as_dict() for record in records],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _read_selected_observations(path: Path, selected: set[int]) -> dict[int, list[dict]]:
    grouped = {node_id: [] for node_id in selected}
    with path.open(encoding="utf-8") as source:
        for line in source:
            observation = json.loads(line)
            node_id = int(observation["node_id"])
            if node_id in grouped:
                grouped[node_id].append(observation)
    return grouped


def _load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
