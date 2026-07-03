"""Recalibrate range-guard thresholds from deployment-runtime observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand range bounds using clean SPINNV2 runtime calibration runs.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runtime-reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-ratio", default=0.05, type=float)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    observed = {}
    for report_path in args.runtime_reports:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for item in report["clean_protected"]["range_observations"]:
            node_id = int(item["node_id"])
            if node_id not in observed:
                observed[node_id] = [float(item["observed_min"]), float(item["observed_max"])]
            else:
                observed[node_id][0] = min(observed[node_id][0], float(item["observed_min"]))
                observed[node_id][1] = max(observed[node_id][1], float(item["observed_max"]))
    adjusted = []
    for node in plan["nodes"]:
        if node["mode"] != "range_guard_rerun" or int(node["node_id"]) not in observed:
            continue
        before = [float(node["lower_bound"]), float(node["upper_bound"])]
        runtime_min, runtime_max = observed[int(node["node_id"])]
        lower = min(before[0], runtime_min)
        upper = max(before[1], runtime_max)
        span = max(upper - lower, abs(lower), abs(upper), 1e-6)
        margin = span * args.margin_ratio
        node["lower_bound"] = lower - margin
        node["upper_bound"] = upper + margin
        adjusted.append(
            {
                "node_id": int(node["node_id"]),
                "bounds_before": before,
                "runtime_observed": [runtime_min, runtime_max],
                "bounds_after": [node["lower_bound"], node["upper_bound"]],
            }
        )
    plan["runtime_range_recalibration"] = {
        "status": "runtime_calibrated_requires_disjoint_evaluation",
        "margin_ratio": args.margin_ratio,
        "runtime_reports": args.runtime_reports,
        "adjusted_nodes": adjusted,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"adjusted_range_nodes": len(adjusted), "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
