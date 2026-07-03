"""Build YOLO DMR candidates from task-level runtime risk and measured costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.reliability.profile_protection_costs import (
    _interpolate_primitive_cost,
    _run_primitive_benchmark,
    _run_runtime_profile,
)


def build_candidates(
    risk_records: list[dict],
    node_costs: dict[int, dict],
    primitive: list[dict],
    *,
    protectable_node_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    candidates = []
    details = []
    excluded = []
    for record in risk_records:
        probability = float(record["critical_probability"])
        if probability <= 0.0:
            continue
        node_id = int(record["node_id"])
        if protectable_node_ids is not None and node_id not in protectable_node_ids:
            excluded.append(
                {
                    "node_id": node_id,
                    "tensor_id": int(record["tensor_id"]),
                    "critical_failures": int(record["critical_failures"]),
                    "reason": "current DMR runtime supports exactly one output per protected node",
                }
            )
            continue
        if node_id not in node_costs:
            raise RuntimeError(f"runtime profile is missing node {node_id}")
        tensor_bytes = int(record["activation_bytes"])
        primitive_cost = _interpolate_primitive_cost(primitive, tensor_bytes)
        latency_ms = float(node_costs[node_id]["avg_ms"]) + primitive_cost["dmr_buffer_ms"]
        contribution = float(record["exposure_ratio"]) * probability
        candidate = {
            "node_id": node_id,
            "tensor_id": int(record["tensor_id"]),
            "mode": "dmr_compare_rerun",
            "risk_reduction": contribution,
            "latency_overhead_ms": latency_ms,
            "extra_memory_bytes": tensor_bytes * 2,
        }
        candidates.append(candidate)
        details.append(
            {
                **candidate,
                "critical_failures": int(record["critical_failures"]),
                "critical_probability": probability,
                "exposure_ratio": float(record["exposure_ratio"]),
                "risk_reduction_basis": "activation_bytes_exposure_ratio * stratified_critical_probability",
            }
        )
    return candidates, details, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DMR candidates for detection task protection optimization.")
    parser.add_argument("--risk", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--spk-debug", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runtime-bench", required=True)
    parser.add_argument("--primitive-bench", required=True)
    parser.add_argument("--runtime-warmup", type=int, default=3)
    parser.add_argument("--runtime-runs", type=int, default=10)
    parser.add_argument("--output-candidates", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--baseline-metrics", help="JSON with native vulnerability metrics from compute_baseline_metrics.py.")
    args = parser.parse_args()

    risk_records = json.loads(Path(args.risk).read_text(encoding="utf-8"))
    debug = json.loads(Path(args.spk_debug).read_text(encoding="utf-8"))
    protectable_node_ids = {int(node["id"]) for node in debug["nodes"] if len(node["outputs"]) == 1}
    primitive = _run_primitive_benchmark(Path(args.primitive_bench))
    node_costs, runtime_output = _run_runtime_profile(
        Path(args.runtime_bench),
        Path(args.spk),
        Path(args.input),
        warmup=args.runtime_warmup,
        runs=args.runtime_runs,
    )
    candidates, details, excluded = build_candidates(
        risk_records,
        node_costs,
        primitive,
        protectable_node_ids=protectable_node_ids,
    )

    if args.baseline_metrics:
        bm = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
        for candidate in candidates:
            node_metrics = bm.get(str(candidate["node_id"]), {})
            for key in ("macs_fraction", "activation_magnitude", "taylor_importance"):
                if key in node_metrics:
                    candidate[key] = node_metrics[key]

    baseline_expected_risk = sum(
        float(record["exposure_ratio"]) * float(record["critical_probability"]) for record in risk_records
    )
    profile = {
        "objective": "minimize expected critical detection-failure probability under activation-byte-weighted runtime-object prior",
        "baseline_expected_critical_failure_probability": baseline_expected_risk,
        "protectable_observed_risk_nodes": len(candidates),
        "excluded_observed_risk_nodes": excluded,
        "cost_model": {
            "dmr": "measured node re-execution time plus measured two-copy-and-compare buffer time",
            "runtime_profile_output": runtime_output,
            "runtime_warmup": args.runtime_warmup,
            "runtime_runs": args.runtime_runs,
            "primitive_measurements": primitive,
        },
        "modes": details,
    }
    candidates_path = Path(args.output_candidates)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    profile_path = Path(args.output_profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_modes": len(candidates),
                "excluded_nodes": len(excluded),
                "baseline_expected_critical_failure_probability": baseline_expected_risk,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
