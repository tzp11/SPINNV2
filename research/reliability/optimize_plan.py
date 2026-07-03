"""CLI for producing a protected-execution plan from measured candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_ahmadilivani,
    optimize_aspis,
    optimize_bounded_loss_memory_ilp,
    optimize_filr,
    optimize_greedy,
    optimize_ilp,
    optimize_ilp_static,
    optimize_ilp_v2,
    optimize_ilp_v3,
    optimize_ranger,
    optimize_ruospo,
    optimize_uniform_dmr,
    optimize_vf_greedy,
    write_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize a SPINNV2 reliability ProtectionPlan.")
    parser.add_argument("candidates", help="JSON list of measured candidate protection modes.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--platform-profile", default="macos_arm64_neon")
    parser.add_argument("--fault-prior", default="activation_bytes_weighted_single_bit")
    parser.add_argument("--workload-scope")
    parser.add_argument("--latency-budget-ms", required=True, type=float)
    parser.add_argument("--memory-budget-bytes", required=True, type=int)
    parser.add_argument("--method", choices=("ilp", "greedy", "bounded-loss-memory", "filr", "ranger", "vf-greedy", "uniform-dmr", "aspis", "ruospo", "ahmadilivani", "ilp-static", "ilp-v2", "ilp-v3"), default="ilp")
    parser.add_argument("--bounded-loss-tolerance", type=float, default=0.05)
    parser.add_argument("--calibration-metadata",
                        help="Calibration metadata JSON for ILP-v2 (provides CF node IDs and per-node RG coverage).")
    args = parser.parse_args()
    raw = json.loads(open(args.candidates, encoding="utf-8").read())
    candidates = [CandidateMode(**item) for item in raw]
    if args.method == "ilp":
        result = optimize_ilp(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "greedy":
        result = optimize_greedy(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "filr":
        result = optimize_filr(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "ranger":
        result = optimize_ranger(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "vf-greedy":
        result = optimize_vf_greedy(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "uniform-dmr":
        result = optimize_uniform_dmr(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "aspis":
        result = optimize_aspis(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "ruospo":
        result = optimize_ruospo(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "ahmadilivani":
        result = optimize_ahmadilivani(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "ilp-static":
        result = optimize_ilp_static(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    elif args.method == "ilp-v2":
        cf_node_ids = None
        per_node_rg_coverage = None
        if args.calibration_metadata:
            import json as _json
            meta = _json.loads(Path(args.calibration_metadata).read_text(encoding="utf-8"))
            cf_node_ids = {
                int(nid) for nid, nd in meta.get("per_node", {}).items()
                if nd.get("critical_failures", 0) > 0
            }
            per_node_rg_coverage = {
                int(nid): float(v) for nid, v in meta.get("per_node_rg_coverage", {}).items()
            }
        result = optimize_ilp_v2(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
            cf_node_ids=cf_node_ids,
            per_node_rg_coverage=per_node_rg_coverage,
        )
    elif args.method == "ilp-v3":
        per_node_rg_coverage = None
        if args.calibration_metadata:
            import json as _json
            meta = _json.loads(Path(args.calibration_metadata).read_text(encoding="utf-8"))
            per_node_rg_coverage = {
                int(nid): float(v) for nid, v in meta.get("per_node_rg_coverage", {}).items()
            }
        result = optimize_ilp_v3(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
            per_node_rg_coverage=per_node_rg_coverage,
        )
    else:
        result = optimize_bounded_loss_memory_ilp(
            candidates,
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
            risk_loss_tolerance=args.bounded_loss_tolerance,
        )
    plan = result.to_protection_plan(
        model_id=args.model_id,
        platform_profile=args.platform_profile,
        latency_budget_ms=args.latency_budget_ms,
        memory_budget_bytes=args.memory_budget_bytes,
        fault_prior=args.fault_prior,
        workload_scope=args.workload_scope,
    )
    write_plan(plan, args.output)
    print(json.dumps({"method": result.method, "selected": len(result.selected), "risk_reduction": result.total_risk_reduction}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
