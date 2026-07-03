"""Validate one compiled ProtectionPlan against a deterministic runtime fault."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.reliability.injection.bitflip import FaultEvent
from research.reliability.runtime_driver import RuntimeDriver


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a protected SPK with and without one deterministic fault.")
    parser.add_argument("--library", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-output", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--node-id", required=True, type=int)
    parser.add_argument("--tensor-id", required=True, type=int)
    parser.add_argument("--element-index", required=True, type=int)
    parser.add_argument("--bit-index", required=True, type=int)
    parser.add_argument("--invocation-index", default=1, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()
    input_data = np.fromfile(args.input, dtype=np.float32).reshape((1, 3, 224, 224))
    reference = np.fromfile(args.reference_output, dtype=np.float32)
    event = FaultEvent(
        args.model_id,
        args.sample_id,
        args.node_id,
        args.tensor_id,
        args.element_index,
        args.bit_index,
        args.invocation_index,
        args.seed,
    )
    plan_nodes = json.loads(Path(args.plan).read_text(encoding="utf-8"))["nodes"]
    range_node_ids = [
        int(node["node_id"])
        for node in plan_nodes
        if node["mode"] == "range_guard_rerun"
    ]
    with RuntimeDriver(args.library, args.spk) as runtime:
        clean_output, clean_stats = runtime.run(input_data)
        clean_range_observations = runtime.range_observations(range_node_ids)
        faulted_output, faulted_stats = runtime.run(input_data, event)
    configured_ranges = {
        int(node["node_id"]): node for node in plan_nodes if node["mode"] == "range_guard_rerun"
    }
    for observation in clean_range_observations:
        configured = configured_ranges[observation["node_id"]]
        observation["configured_lower_bound"] = configured["lower_bound"]
        observation["configured_upper_bound"] = configured["upper_bound"]
        observation["outside_configured_bounds"] = bool(
            observation["observed_min"] < configured["lower_bound"]
            or observation["observed_max"] > configured["upper_bound"]
        )
    report = {
        "model_id": args.model_id,
        "sample_id": args.sample_id,
        "fault_event": {
            "node_id": args.node_id,
            "tensor_id": args.tensor_id,
            "element_index": args.element_index,
            "bit_index": args.bit_index,
            "invocation_index": args.invocation_index,
        },
        "clean_protected": {
            "predicted_class": int(clean_output.argmax()),
            "max_abs_vs_unprotected_reference": float(np.max(np.abs(clean_output - reference))),
            "stats": clean_stats,
            "range_observations": clean_range_observations,
            "range_violation_node_ids": [
                item["node_id"] for item in clean_range_observations if item["outside_configured_bounds"]
            ],
        },
        "faulted_protected": {
            "predicted_class": int(faulted_output.argmax()),
            "max_abs_vs_unprotected_reference": float(np.max(np.abs(faulted_output - reference))),
            "stats": faulted_stats,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
