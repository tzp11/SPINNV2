"""Summarize YOLO grouped protection evidence from existing runtime reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="artifacts/reports/paper_assets/yolov10n_grouped_protection_comparison.json")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    base_dir = root / "artifacts/reports/yolov10n_dior_full_e30_b16"
    docs = {
        "control_path_dmr": _read_json(base_dir / "control_dmr_comparison/yolov10n_control_dmr_comparison.json"),
        "ilp_dmr_activation_prior": _read_json(
            base_dir / "ilp_dmr_activation_prior_comparison/yolov10n_control_dmr_comparison.json"
        ),
        "ilp_dmr_stratified": _read_json(
            base_dir / "ilp_dmr_stratified_comparison/yolov10n_control_dmr_comparison.json"
        ),
    }
    groups: list[dict[str, Any]] = []
    control = docs["control_path_dmr"]
    if control:
        groups.append(
            {
                "group": "control_path_nodes",
                "mechanism": "dmr_compare_rerun",
                "sampling": control.get("fault_sampling"),
                "protected_scope": "Gather/GatherElements and execution-control-sensitive nodes",
                "baseline_critical_failures": control["baseline"]["critical_failures"],
                "protected_critical_failures": control["control_dmr"]["critical_failures"],
                "reduction_ratio": control["critical_failure_reduction_ratio"],
                "extra_memory_bytes": control["control_dmr"]["extra_memory_bytes"],
                "latency_overhead_ms": control["latency_overhead_ms"],
                "interpretation": "isolates control-path crash/error mitigation; not sufficient for task-output faults",
            }
        )
    activation = docs["ilp_dmr_activation_prior"]
    if activation:
        groups.append(
            {
                "group": "activation_prior_budget_nodes",
                "mechanism": "budgeted_ilp_dmr",
                "sampling": activation.get("fault_sampling"),
                "protected_scope": "40 budget-selected single-output runtime nodes",
                "baseline_critical_failures": activation["baseline"]["critical_failures"],
                "protected_critical_failures": activation["budget_ilp_dmr"]["critical_failures"],
                "reduction_ratio": activation["critical_failure_reduction_ratio"],
                "extra_memory_bytes": activation["budget_ilp_dmr"]["extra_memory_bytes"],
                "latency_overhead_ms": activation["latency_overhead_ms"],
                "interpretation": "realistic activation-byte prior; protection effective but reduction is moderate",
            }
        )
    stratified = docs["ilp_dmr_stratified"]
    if stratified:
        groups.append(
            {
                "group": "stratified_budget_nodes",
                "mechanism": "budgeted_ilp_dmr",
                "sampling": stratified.get("fault_sampling"),
                "protected_scope": "40 budget-selected single-output runtime nodes under stratified node coverage",
                "baseline_critical_failures": stratified["baseline"]["critical_failures"],
                "protected_critical_failures": stratified["budget_ilp_dmr"]["critical_failures"],
                "reduction_ratio": stratified["critical_failure_reduction_ratio"],
                "extra_memory_bytes": stratified["budget_ilp_dmr"]["extra_memory_bytes"],
                "latency_overhead_ms": stratified["latency_overhead_ms"],
                "interpretation": "shows protection can remove many high-risk/control-sensitive failures when nodes are evenly covered",
            }
        )
    report = {
        "model": "yolov10n_dior_full_e30_b16",
        "status": "grouped_protection_summary_from_existing_runtime_reports",
        "task_output_guard_status": "not_implemented; grouped protection comparison is used as the minimum P5 deliverable",
        "groups": groups,
        "thesis_claim": (
            "YOLOv10n evidence currently supports task-level protection effectiveness and control-path mitigation. "
            "It should not be used alone to claim ILP superiority over all simple policies."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "groups": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
