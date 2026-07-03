"""Build protection candidates from measured task risk and runtime costs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Create mode candidates for budget-aware protection optimization.")
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--range-report", help="Range profiling report JSON (optional, enables range_guard candidates).")
    parser.add_argument("--spk", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runtime-bench", required=True, help="Path to spkv2_bench executable.")
    parser.add_argument("--primitive-bench", required=True, help="Path to spkv2_protection_bench executable.")
    parser.add_argument("--runtime-warmup", type=int, default=3)
    parser.add_argument("--runtime-runs", type=int, default=10)
    parser.add_argument("--output-candidates", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--baseline-metrics", help="JSON with native vulnerability metrics from compute_baseline_metrics.py.")
    args = parser.parse_args()

    ranked = pd.read_csv(args.ranked_csv)
    ranked = ranked[ranked["retained_after_passes"].astype(str).str.lower().eq("true")].copy()
    ranked["runtime_node_id"] = ranked["runtime_node_id"].astype(int)
    ranked["runtime_tensor_id"] = ranked["runtime_tensor_id"].astype(int)
    total_bytes = float(ranked["activation_bytes"].sum())
    ranked["runtime_fault_prior"] = ranked["activation_bytes"] / total_bytes
    if "critical_probability" in ranked.columns:
        ranked["critical_failure_contribution"] = ranked["runtime_fault_prior"] * ranked["critical_probability"]
    else:
        ranked["critical_failure_contribution"] = ranked["runtime_fault_prior"]

    primitive = _run_primitive_benchmark(Path(args.primitive_bench))
    node_costs, runtime_output = _run_runtime_profile(
        Path(args.runtime_bench),
        Path(args.spk),
        Path(args.input),
        warmup=args.runtime_warmup,
        runs=args.runtime_runs,
    )
    range_records = {}
    if args.range_report:
        range_data = json.loads(Path(args.range_report).read_text(encoding="utf-8"))
        range_records = {
            int(item["node_id"]): item
            for item in range_data.get("records", range_data.get("observations", []))
        }

    candidates = []
    details = []
    for row in ranked.itertuples(index=False):
        runtime_node_id = int(row.runtime_node_id)
        tensor_bytes = int(row.activation_bytes)
        if runtime_node_id not in node_costs:
            continue
        primitive_cost = _interpolate_primitive_cost(primitive, tensor_bytes)
        contribution = float(row.critical_failure_contribution)
        critical_prob = float(row.critical_probability) if hasattr(row, "critical_probability") and pd.notna(row.critical_probability) else None
        dmr_latency = node_costs[runtime_node_id]["avg_ms"] + primitive_cost["dmr_buffer_ms"]
        candidate_entry = {
            "node_id": runtime_node_id,
            "tensor_id": int(row.runtime_tensor_id),
            "mode": "dmr_compare_rerun",
            "risk_reduction": contribution,
            "latency_overhead_ms": dmr_latency,
            "extra_memory_bytes": tensor_bytes * 2,
        }
        if critical_prob is not None:
            candidate_entry["critical_probability"] = critical_prob
        candidates.append(candidate_entry)
        details.append(
            {
                "runtime_node_id": runtime_node_id,
                "runtime_tensor_id": int(row.runtime_tensor_id),
                "mode": "dmr_compare_rerun",
                "critical_failure_contribution": contribution,
                "latency_overhead_ms": dmr_latency,
                "extra_memory_bytes": tensor_bytes * 2,
            }
        )
        range_record = range_records.get(runtime_node_id)
        if range_record is None:
            continue
        coverage = float(range_record.get("critical_coverage_ci_low", range_record.get("coverage", 0.5)))
        range_reduction = contribution * coverage
        false_positive_upper = float(range_record.get("false_positive_ci_high", range_record.get("false_positive_rate", 0.0)))
        range_latency = primitive_cost["range_scan_ms"] + false_positive_upper * (
            node_costs[runtime_node_id]["avg_ms"] + primitive_cost["range_scan_ms"]
        )
        candidates.append(
            {
                "node_id": runtime_node_id,
                "tensor_id": int(row.runtime_tensor_id),
                "mode": "range_guard_rerun",
                "risk_reduction": range_reduction,
                "latency_overhead_ms": range_latency,
                "extra_memory_bytes": 0,
                "lower_bound": float(range_record["lower_bound"]),
                "upper_bound": float(range_record["upper_bound"]),
                **({"critical_probability": critical_prob} if critical_prob is not None else {}),
            }
        )
        details.append(
            {
                "runtime_node_id": runtime_node_id,
                "runtime_tensor_id": int(row.runtime_tensor_id),
                "mode": "range_guard_rerun",
                "critical_failure_contribution": contribution,
                "estimated_critical_failure_reduction": range_reduction,
                "latency_overhead_ms": range_latency,
                "extra_memory_bytes": 0,
                "coverage": coverage,
                "false_positive_overhead_rate": false_positive_upper,
                "lower_bound": float(range_record["lower_bound"]),
                "upper_bound": float(range_record["upper_bound"]),
            }
        )

    if args.baseline_metrics:
        bm = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
        for candidate in candidates:
            node_metrics = bm.get(str(candidate["node_id"]), {})
            for key in ("macs_fraction", "activation_magnitude", "taylor_importance"):
                if key in node_metrics:
                    candidate[key] = node_metrics[key]

    profile = {
        "objective": "minimize expected critical task-failure probability under activation-byte-weighted runtime-object prior",
        "baseline_expected_critical_failure_probability": float(ranked["critical_failure_contribution"].sum()),
        "protectable_runtime_object_count": int(len(ranked)),
        "range_profiled_object_count": len(range_records),
        "cost_model": {
            "dmr": "baseline node re-execution + measured two-copy-and-compare buffer time",
            "range_guard": "measured scalar finite-and-bound scan time; rerun excluded from no-fault budget",
            "runtime_warmup": args.runtime_warmup,
            "runtime_runs": args.runtime_runs,
            "primitive_measurements": primitive,
        },
        "modes": details,
    }
    Path(args.output_candidates).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_candidates).write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    Path(args.output_profile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_profile).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "protectable_runtime_objects": len(ranked),
                "candidate_modes": len(candidates),
                "baseline_expected_failure_probability": profile["baseline_expected_critical_failure_probability"],
            }
        )
    )
    return 0


def _run_primitive_benchmark(executable: Path) -> list[dict]:
    completed = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)["measurements"]


def _run_runtime_profile(
    executable: Path,
    spk: Path,
    input_path: Path,
    *,
    warmup: int,
    runs: int,
) -> tuple[dict[int, dict], str]:
    environment = os.environ.copy()
    environment["SPKV2_PROFILE"] = "1"
    completed = subprocess.run(
        [
            str(executable),
            str(spk),
            str(input_path),
            "--warmup", str(warmup),
            "--runs", str(runs),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    marker = "node_id,op_type,avg_ms,total_ms,count\n"
    if marker not in completed.stderr:
        raise RuntimeError("SPKV2 runtime did not emit node profile CSV on stderr")
    table = completed.stderr.split(marker, 1)[1].split("\n\n", 1)[0]
    rows = csv.DictReader(table.splitlines(), fieldnames=["node_id", "op_type", "avg_ms", "total_ms", "count"])
    nodes = {
        int(row["node_id"]): {"op_type": row["op_type"], "avg_ms": float(row["avg_ms"])}
        for row in rows
    }
    return nodes, completed.stdout


def _interpolate_primitive_cost(measurements: list[dict], tensor_bytes: int) -> dict[str, float]:
    ordered = sorted(measurements, key=lambda item: item["tensor_bytes"])
    x = np.array([item["tensor_bytes"] for item in ordered], dtype=float)
    return {
        "range_scan_ms": float(np.interp(tensor_bytes, x, [item["range_scan_ms"] for item in ordered])),
        "dmr_buffer_ms": float(np.interp(tensor_bytes, x, [item["dmr_buffer_ms"] for item in ordered])),
    }


if __name__ == "__main__":
    raise SystemExit(main())
