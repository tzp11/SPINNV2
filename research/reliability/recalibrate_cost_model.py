"""Conservatively scale predicted protection costs using clean runtime feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale candidate latency costs from measured plan overhead.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cost-profile", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--output-candidates", required=True)
    parser.add_argument("--output-profile", required=True)
    args = parser.parse_args()

    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    profile = json.loads(Path(args.cost_profile).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    report = json.loads(Path(args.runtime_report).read_text(encoding="utf-8"))
    candidate_by_key = {(int(item["node_id"]), item["mode"]): item for item in candidates}
    predicted_overhead = sum(
        candidate_by_key[(int(item["node_id"]), item["mode"])]["latency_overhead_ms"]
        for item in plan["nodes"]
    )
    measured_overhead = float(report["latency_overhead"]["avg_ms"])
    if predicted_overhead <= 0.0:
        raise ValueError("selected plan has no predicted overhead to recalibrate")
    measured_scale = measured_overhead / predicted_overhead
    conservative_scale = max(1.0, measured_scale)
    adjusted = []
    for item in candidates:
        updated = dict(item)
        updated["latency_overhead_ms"] = float(item["latency_overhead_ms"]) * conservative_scale
        adjusted.append(updated)
    profile["cost_model"]["runtime_feedback"] = {
        "source_plan": args.plan,
        "source_runtime_report": args.runtime_report,
        "predicted_plan_overhead_ms": predicted_overhead,
        "measured_plan_overhead_ms": measured_overhead,
        "measured_scale": measured_scale,
        "applied_conservative_scale": conservative_scale,
        "policy": "uniform_latency_scale_never_below_one",
    }
    profile["modes"] = [
        {
            **item,
            "latency_overhead_ms": float(item["latency_overhead_ms"]) * conservative_scale,
        }
        for item in profile["modes"]
    ]
    output_candidates = Path(args.output_candidates)
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.write_text(json.dumps(adjusted, ensure_ascii=False, indent=2), encoding="utf-8")
    output_profile = Path(args.output_profile)
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "predicted_plan_overhead_ms": predicted_overhead,
                "measured_plan_overhead_ms": measured_overhead,
                "applied_conservative_scale": conservative_scale,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
