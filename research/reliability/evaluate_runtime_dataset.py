"""Evaluate a compiled protection plan on a clean EuroSAT runtime split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights
import yaml

from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure clean SPINNV2 plan behavior on a fixed EuroSAT split.")
    parser.add_argument("--config", default="research/reliability/configs/windows_prevalidation.yaml")
    parser.add_argument("--library", required=True)
    parser.add_argument("--baseline-spk", required=True)
    parser.add_argument("--protected-spk", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    source_indices = splits[args.split][args.offset : args.offset + args.max_samples]
    dataset = Subset(
        EuroSAT(root=str(data_root), transform=ResNet50_Weights.IMAGENET1K_V2.transforms()),
        source_indices,
    )
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    range_nodes = {
        int(node["node_id"]): node
        for node in plan["nodes"]
        if node["mode"] == "range_guard_rerun"
    }
    aggregate_ranges: dict[int, dict] = {}
    stats_total = {
        "injected_faults": 0,
        "detected_faults": 0,
        "recovered_faults": 0,
        "unrecovered_faults": 0,
        "rerun_count": 0,
    }
    baseline_correct = 0
    protected_correct = 0
    matching_predictions = 0
    clean_alarm_samples = 0
    max_abs_difference = 0.0
    baseline_times = []
    protected_times = []
    sample_rows = []
    with RuntimeDriver(args.library, args.baseline_spk) as baseline, RuntimeDriver(
        args.library, args.protected_spk
    ) as protected:
        for image, _label in list(dataset)[: args.warmup_samples]:
            values = np.ascontiguousarray(image.unsqueeze(0).numpy(), dtype=np.float32)
            baseline.run(values)
            protected.run(values)
        for local_index, (image, label) in enumerate(dataset):
            values = np.ascontiguousarray(image.unsqueeze(0).numpy(), dtype=np.float32)
            start = time.perf_counter()
            baseline_output, _baseline_stats = baseline.run(values)
            baseline_times.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            protected_output, protected_stats = protected.run(values)
            protected_times.append((time.perf_counter() - start) * 1000.0)
            observations = protected.range_observations(list(range_nodes))
            _accumulate_ranges(aggregate_ranges, observations)
            for name in stats_total:
                stats_total[name] += int(protected_stats[name])
            baseline_class = int(baseline_output.argmax())
            protected_class = int(protected_output.argmax())
            baseline_correct += int(baseline_class == int(label))
            protected_correct += int(protected_class == int(label))
            matching_predictions += int(baseline_class == protected_class)
            clean_alarm_samples += int(protected_stats["detected_faults"] > 0)
            difference = float(np.max(np.abs(protected_output - baseline_output)))
            max_abs_difference = max(max_abs_difference, difference)
            sample_rows.append(
                {
                    "sample_id": f"{args.split}_{args.offset + local_index:06d}",
                    "dataset_index": int(source_indices[local_index]),
                    "label": int(label),
                    "baseline_class": baseline_class,
                    "protected_class": protected_class,
                    "max_abs_difference": difference,
                    "stats": protected_stats,
                }
            )
    observations = _annotate_ranges(aggregate_ranges, range_nodes)
    count = len(dataset)
    report = {
        "model_id": plan["model_id"],
        "platform_profile": plan.get("platform_profile"),
        "evaluation": {
            "split": args.split,
            "offset": args.offset,
            "samples": count,
            "warmup_samples": min(args.warmup_samples, count),
        },
        "clean_baseline": {
            "accuracy": baseline_correct / count if count else 0.0,
            "latency_ms": _timing_summary(baseline_times),
        },
        "clean_protected": {
            "accuracy": protected_correct / count if count else 0.0,
            "prediction_agreement_with_baseline": matching_predictions / count if count else 0.0,
            "max_abs_vs_unprotected_reference": max_abs_difference,
            "false_alarm_samples": clean_alarm_samples,
            "false_alarm_rate": clean_alarm_samples / count if count else 0.0,
            "stats": stats_total,
            "latency_ms": _timing_summary(protected_times),
            "range_observations": observations,
            "range_violation_node_ids": [
                item["node_id"] for item in observations if item["outside_configured_bounds"]
            ],
        },
        "latency_overhead": {
            "avg_ms": float(np.mean(protected_times) - np.mean(baseline_times)) if count else 0.0,
            "avg_ratio": (
                float(np.mean(protected_times) / np.mean(baseline_times) - 1.0)
                if count and np.mean(baseline_times)
                else 0.0
            ),
        },
        "samples": sample_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("evaluation", "clean_baseline", "clean_protected", "latency_overhead")}))
    return 0


def _accumulate_ranges(aggregate: dict[int, dict], observations: list[dict]) -> None:
    for item in observations:
        node_id = int(item["node_id"])
        if node_id not in aggregate:
            aggregate[node_id] = dict(item)
            continue
        aggregate[node_id]["observed_min"] = min(aggregate[node_id]["observed_min"], item["observed_min"])
        aggregate[node_id]["observed_max"] = max(aggregate[node_id]["observed_max"], item["observed_max"])
        aggregate[node_id]["observations"] += item["observations"]


def _annotate_ranges(aggregate: dict[int, dict], configured: dict[int, dict]) -> list[dict]:
    records = []
    for node_id in sorted(configured):
        if node_id not in aggregate:
            continue
        item = aggregate[node_id]
        bounds = configured[node_id]
        item["configured_lower_bound"] = float(bounds["lower_bound"])
        item["configured_upper_bound"] = float(bounds["upper_bound"])
        item["outside_configured_bounds"] = bool(
            item["observed_min"] < item["configured_lower_bound"]
            or item["observed_max"] > item["configured_upper_bound"]
        )
        records.append(item)
    return records


def _timing_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
