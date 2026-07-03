"""Build a DMR plan for runtime objects that caused controlled execution errors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def build_plan(
    injection_path: str | Path,
    debug_path: str | Path,
    *,
    model_id: str,
    platform_profile: str,
    memory_budget_bytes: int,
    fault_prior: str = "stratified_runtime_objects",
    workload_scope: str | None = None,
) -> dict:
    failures: Counter[int] = Counter()
    with Path(injection_path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("failure_mode") == "controlled_execution_error":
                failures[int(record["node_id"])] += 1
    graph = json.loads(Path(debug_path).read_text(encoding="utf-8"))
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    tensors = {int(tensor["id"]): tensor for tensor in graph["tensors"]}
    selected = []
    skipped = []
    for node_id, error_count in sorted(failures.items(), key=lambda item: (-item[1], item[0])):
        tensor_id = int(nodes[node_id]["outputs"][0])
        extra_memory = 2 * int(tensors[tensor_id]["size_bytes"])
        candidate = {
            "node_id": node_id,
            "tensor_id": tensor_id,
            "mode": "dmr_compare_rerun",
            "observed_execution_errors": error_count,
            "extra_memory_bytes": extra_memory,
        }
        if extra_memory <= memory_budget_bytes:
            selected.append(candidate)
        else:
            skipped.append(candidate)
    plan = {
        "version": 1,
        "model_id": model_id,
        "platform_profile": platform_profile,
        "fault_prior": fault_prior,
        "budgets": {"extra_memory_bytes": memory_budget_bytes},
        "optimizer": {
            "method": "protect_observed_control_path_failures_with_dmr",
            "observed_execution_errors_covered": sum(item["observed_execution_errors"] for item in selected),
            "observed_execution_errors_skipped": sum(item["observed_execution_errors"] for item in skipped),
            "peak_extra_memory_bytes": max((item["extra_memory_bytes"] for item in selected), default=0),
        },
        "nodes": [
            {"node_id": item["node_id"], "tensor_id": item["tensor_id"], "mode": item["mode"]}
            for item in selected
        ],
        "skipped_nodes": skipped,
    }
    if workload_scope is not None:
        plan["workload_scope"] = workload_scope
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--injections", required=True)
    parser.add_argument("--spk-debug", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--platform-profile", default="windows_x64_avx2_release")
    parser.add_argument("--fault-prior", default="stratified_runtime_objects")
    parser.add_argument("--workload-scope")
    parser.add_argument("--memory-budget-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = build_plan(
        args.injections,
        args.spk_debug,
        model_id=args.model_id,
        platform_profile=args.platform_profile,
        memory_budget_bytes=args.memory_budget_bytes,
        fault_prior=args.fault_prior,
        workload_scope=args.workload_scope,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
