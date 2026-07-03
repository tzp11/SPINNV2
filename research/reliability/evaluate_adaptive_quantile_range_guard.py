"""Evaluate adaptive quantile range guards on ResNet50 semantic activations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights, resnet50
import yaml

from research.reliability.injection.bitflip import sample_fault_event
from research.reliability.injection.torch_injector import capture_candidate_output, infer_with_fault_and_capture
from research.reliability.mechanism_screening import decide_adaptive_range
from research.reliability.metrics.task_metrics import classification_failure


POLICIES = {
    "minmax": None,
    "q99.9": 0.999,
    "q99.5": 0.995,
    "q99": 0.990,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--holdout-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--injections-per-node", type=int, default=32)
    parser.add_argument("--sample-values-per-activation", type=int, default=8192)
    parser.add_argument("--margin-ratio", type=float, default=0.02)
    parser.add_argument("--alarm-margin", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    dataset = EuroSAT(root=str(data_root), transform=weights.transforms())
    calibration = Subset(dataset, splits["val"][: args.calibration_samples])
    holdout = Subset(dataset, splits["val"][args.calibration_samples : args.calibration_samples + args.holdout_samples])
    test = Subset(dataset, splits["test"][: args.test_samples])

    device = torch.device(args.device)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(_load_state_dict(Path(args.checkpoint)))
    model.to(device).eval()

    ranked = pd.read_csv(args.ranked_csv)
    retained = ranked[ranked["retained_after_passes"].astype(str).str.lower().eq("true")].head(args.top_k)
    rows = []
    for index, row in enumerate(retained.itertuples(index=False), start=1):
        print(json.dumps({"stage": "node", "index": index, "node_id": int(row.node_id)}), flush=True)
        node_rows = _evaluate_node(model, device, row, calibration, holdout, test, args, int(config["experiment"]["seed"]))
        rows.extend(node_rows)

    frame = pd.DataFrame(rows)
    summary_rows = []
    for policy, group in frame.groupby("policy"):
        critical = int(group["critical_fault"].sum())
        detected_critical = int((group["critical_fault"] & group["detected"]).sum())
        clean = int(group["clean_holdout_samples"].sum())
        false_positives = int(group["false_positive_samples"].sum())
        coverage = detected_critical / critical if critical else 0.0
        false_positive_rate = false_positives / clean if clean else 0.0
        minmax = frame[frame["policy"] == "minmax"]
        minmax_critical = int(minmax["critical_fault"].sum())
        minmax_detected_critical = int((minmax["critical_fault"] & minmax["detected"]).sum())
        minmax_coverage = minmax_detected_critical / minmax_critical if minmax_critical else 0.0
        gain = coverage - minmax_coverage
        decision = decide_adaptive_range(false_positive_rate=false_positive_rate, coverage_gain=gain)
        summary_rows.append(
            {
                "policy": policy,
                "critical_faults": critical,
                "detected_critical_faults": detected_critical,
                "critical_coverage": coverage,
                "false_positive_samples": false_positives,
                "clean_holdout_samples": clean,
                "false_positive_rate": false_positive_rate,
                "coverage_gain_vs_minmax": gain,
                "decision": decision.decision,
                "reason": decision.reason,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "measured",
        "model_id": "resnet50_eurosat",
        "method": "adaptive_quantile_range_guard",
        "config": {
            "top_k": args.top_k,
            "calibration_samples": args.calibration_samples,
            "holdout_samples": args.holdout_samples,
            "test_samples": args.test_samples,
            "injections_per_node": args.injections_per_node,
            "sample_values_per_activation": args.sample_values_per_activation,
            "margin_ratio": args.margin_ratio,
            "alarm_margin": args.alarm_margin,
        },
        "summary": summary_rows,
        "records": rows,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".csv").write_text(pd.DataFrame(summary_rows).to_csv(index=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "summary": summary_rows}, ensure_ascii=False))
    return 0


def _evaluate_node(model, device, row, calibration, holdout, test, args, base_seed: int) -> list[dict]:
    samples = []
    clean_minima = []
    clean_maxima = []
    rng = np.random.default_rng(base_seed + int(row.node_id))
    for sample_index, (image, _label) in enumerate(calibration):
        activation = capture_candidate_output(
            model,
            image.unsqueeze(0).to(device),
            module_name=row.module_name,
            invocation_index=int(row.invocation_index),
        ).cpu().numpy()
        flat = activation.reshape(-1)
        clean_minima.append(float(np.min(flat)))
        clean_maxima.append(float(np.max(flat)))
        take = min(args.sample_values_per_activation, flat.size)
        indices = rng.choice(flat.size, size=take, replace=False)
        samples.append(flat[indices].astype(np.float32, copy=False))
    sampled = np.concatenate(samples) if samples else np.asarray([], dtype=np.float32)
    bounds = {name: _bounds(name, sampled, clean_minima, clean_maxima, args.margin_ratio) for name in POLICIES}

    holdout_violations = {name: [] for name in POLICIES}
    for image, _label in holdout:
        activation = capture_candidate_output(
            model,
            image.unsqueeze(0).to(device),
            module_name=row.module_name,
            invocation_index=int(row.invocation_index),
        ).cpu().numpy()
        for name, bound in bounds.items():
            holdout_violations[name].append(_violation_fraction(activation, bound))
    alarm_thresholds = {
        name: float(np.max(values) + args.alarm_margin) if values else args.alarm_margin
        for name, values in holdout_violations.items()
    }
    false_positive_samples = {
        name: int(sum(value > alarm_thresholds[name] for value in values))
        for name, values in holdout_violations.items()
    }

    correct_samples = []
    for sample_index, (image, label) in enumerate(test):
        with torch.inference_mode():
            baseline = model(image.unsqueeze(0).to(device)).cpu().numpy()[0]
        if int(baseline.argmax()) == int(label):
            correct_samples.append((sample_index, image, int(label), baseline))
    if not correct_samples:
        raise RuntimeError("no correct test samples available for adaptive range guard evaluation")

    rows = []
    for repeat in range(args.injections_per_node):
        sample_index, image, label, baseline = correct_samples[repeat % len(correct_samples)]
        seed = base_seed + int(row.node_id) * 100000 + repeat
        event = sample_fault_event(
            model_id="resnet50_eurosat",
            sample_id=f"test_{sample_index:06d}",
            node_id=int(row.node_id),
            tensor_id=int(row.tensor_id),
            element_count=int(row.activation_bytes // 4),
            seed=seed,
            invocation_index=int(row.invocation_index),
        )
        faulted, activation = infer_with_fault_and_capture(
            model,
            image.unsqueeze(0).to(device),
            module_name=row.module_name,
            event=event,
        )
        consequence = classification_failure(baseline, faulted.cpu().numpy()[0], label)
        for name, bound in bounds.items():
            violation = _violation_fraction(activation.cpu().numpy(), bound)
            rows.append(
                {
                    "semantic_node_id": int(row.node_id),
                    "runtime_node_id": int(row.runtime_node_id),
                    "tensor_id": int(row.runtime_tensor_id),
                    "module_name": row.module_name,
                    "policy": name,
                    "lower_bound": bound[0],
                    "upper_bound": bound[1],
                    "alarm_threshold": alarm_thresholds[name],
                    "violation_fraction": violation,
                    "detected": bool(violation > alarm_thresholds[name]),
                    "critical_fault": bool(consequence.critical_failure),
                    "severity": float(consequence.severity),
                    "false_positive_samples": false_positive_samples[name],
                    "clean_holdout_samples": len(holdout),
                    "fault_event": asdict(event),
                }
            )
    return rows


def _bounds(policy: str, sampled: np.ndarray, minima: list[float], maxima: list[float], margin_ratio: float) -> tuple[float, float]:
    if policy == "minmax":
        lower = float(np.min(minima))
        upper = float(np.max(maxima))
    else:
        q = POLICIES[policy]
        assert q is not None
        lower = float(np.quantile(sampled, 1.0 - q))
        upper = float(np.quantile(sampled, q))
    span = max(upper - lower, abs(lower), abs(upper), 1e-6)
    margin = max(span * margin_ratio, 1e-6)
    return lower - margin, upper + margin


def _violation_fraction(values: np.ndarray, bound: tuple[float, float]) -> float:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        return 1.0
    lower, upper = bound
    return float(np.mean((array < lower) | (array > upper)))


def _load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
