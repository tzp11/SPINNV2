"""Monte Carlo task-level fault evaluation for compiled SPINNV2 plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from torchvision.models import ResNet50_Weights
import yaml

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare protected and unprotected runtime fault outcomes.")
    parser.add_argument("--config", default="research/reliability/configs/macos_reliability.yaml")
    parser.add_argument("--library", required=True)
    parser.add_argument("--baseline-spk", required=True)
    parser.add_argument("--protected-spk", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--events", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    protected_modes = {int(node["node_id"]): node["mode"] for node in plan["nodes"]}
    candidates = _load_candidates(Path(args.ranked_csv))
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(config["datasets"]["eurosat"]["root"])
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    dataset = Subset(
        EuroSAT(root=str(data_root), transform=ResNet50_Weights.IMAGENET1K_V2.transforms()),
        splits["test"][: args.test_samples],
    )
    generator = random.Random(args.seed)
    observations = []
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
    }
    with RuntimeDriver(args.library, args.baseline_spk) as baseline, RuntimeDriver(
        args.library, args.protected_spk
    ) as protected:
        correct_inputs = _collect_correct_inputs(baseline, dataset)
        if not correct_inputs:
            raise RuntimeError("no baseline-correct test samples available")
        weights = [candidate["activation_bytes"] for candidate in candidates]
        for sequence in range(args.events):
            sample_id, values, label = generator.choice(correct_inputs)
            candidate = generator.choices(candidates, weights=weights, k=1)[0]
            event = FaultEvent(
                model_id=plan["model_id"],
                sample_id=sample_id,
                node_id=candidate["runtime_node_id"],
                tensor_id=candidate["runtime_tensor_id"],
                element_index=generator.randrange(candidate["activation_bytes"] // 4),
                bit_index=generator.randrange(32),
                invocation_index=1,
                seed=args.seed + sequence,
            )
            unprotected_output, _ = baseline.run(values, event)
            protected_output, stats = protected.run(values, event)
            unprotected_class = int(unprotected_output.argmax())
            protected_class = int(protected_output.argmax())
            unprotected_failure = unprotected_class != label
            protected_failure = protected_class != label
            is_selected = event.node_id in protected_modes
            totals["events"] += 1
            totals["unprotected_critical_failures"] += int(unprotected_failure)
            totals["protected_critical_failures"] += int(protected_failure)
            totals["mitigated_critical_failures"] += int(unprotected_failure and not protected_failure)
            totals["new_protected_failures"] += int(protected_failure and not unprotected_failure)
            totals["selected_node_events"] += int(is_selected)
            totals["selected_node_unprotected_critical_failures"] += int(is_selected and unprotected_failure)
            totals["selected_node_protected_critical_failures"] += int(is_selected and protected_failure)
            for name in ("detected_faults", "recovered_faults", "unrecovered_faults", "rerun_count"):
                totals[name] += int(stats[name])
            observations.append(
                {
                    "sequence": sequence,
                    "sample_id": sample_id,
                    "label": label,
                    "node_id": event.node_id,
                    "tensor_id": event.tensor_id,
                    "element_index": event.element_index,
                    "bit_index": event.bit_index,
                    "protection_mode": protected_modes.get(event.node_id, "none"),
                    "unprotected_class": unprotected_class,
                    "protected_class": protected_class,
                    "unprotected_critical_failure": unprotected_failure,
                    "protected_critical_failure": protected_failure,
                    "stats": stats,
                }
            )
    total_events = totals["events"]
    baseline_failures = totals["unprotected_critical_failures"]
    selected_failures = totals["selected_node_unprotected_critical_failures"]
    report = {
        "model_id": plan["model_id"],
        "platform_profile": plan.get("platform_profile"),
        "fault_model": {
            "type": "one_random_fp32_output_bit_flip_per_inference",
            "runtime_object_prior": "activation_bytes_weighted",
            "bit_prior": "uniform_all_32_bits",
            "seed": args.seed,
        },
        "sampling": {
            "requested_test_prefix": args.test_samples,
            "baseline_correct_samples": len(correct_inputs),
            "events": total_events,
        },
        "totals": totals,
        "rates": {
            "unprotected_critical_failure_rate": baseline_failures / total_events if total_events else 0.0,
            "protected_critical_failure_rate": totals["protected_critical_failures"] / total_events if total_events else 0.0,
            "observed_risk_reduction_ratio": (
                (baseline_failures - totals["protected_critical_failures"]) / baseline_failures
                if baseline_failures
                else 0.0
            ),
            "selected_node_event_rate": totals["selected_node_events"] / total_events if total_events else 0.0,
            "selected_node_observed_reduction_ratio": (
                (selected_failures - totals["selected_node_protected_critical_failures"]) / selected_failures
                if selected_failures
                else 0.0
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
    candidates = []
    for row in table.itertuples(index=False):
        candidates.append(
            {
                "runtime_node_id": int(row.runtime_node_id),
                "runtime_tensor_id": int(row.runtime_tensor_id),
                "activation_bytes": int(row.activation_bytes),
            }
        )
    return candidates


def _collect_correct_inputs(runtime: RuntimeDriver, dataset: Subset) -> list[tuple[str, np.ndarray, int]]:
    correct = []
    for local_index, (image, label) in enumerate(dataset):
        values = np.ascontiguousarray(image.unsqueeze(0).numpy(), dtype=np.float32)
        output, _ = runtime.run(values)
        if int(output.argmax()) == int(label):
            correct.append((f"test_{local_index:06d}", values, int(label)))
    return correct


if __name__ == "__main__":
    raise SystemExit(main())
