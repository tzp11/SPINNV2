"""Screen candidate variants of protection mechanisms from existing artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.reliability.mechanism_screening import (
    decide_adaptive_range,
    decide_guard_gated_dmr,
    decide_robust_memory_ilp,
    decide_task_utility_guard,
    wilson_lower_bound,
)
from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_bounded_loss_memory_ilp,
    optimize_greedy,
    optimize_ilp,
    optimize_topk_single_mode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="artifacts/reports/paper_assets/mechanism_improvement_screening")
    parser.add_argument("--latency-budgets-ms", default="5,10,20,40,80,160")
    parser.add_argument("--memory-budgets-bytes", default="1048576,4194304,8388608")
    parser.add_argument("--risk-loss-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.repo_root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet_candidates = _read_candidates(
        root / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_protection_candidates.json"
    )
    range_report = _read_json(root / "artifacts/reports/resnet50_gpu_prevalidation/resnet50_range_guard_top20.json")
    resnet_faults = root / "artifacts/reports/formal_windows_completion/resnet50_runtime_faults_30000_combined.jsonl"
    yolo_guard = _read_json(root / "artifacts/reports/formal_windows_completion/yolov10n_task_output_guard_10000_seed2026.json")
    measured_adaptive_range = root / args.output_dir / "adaptive_quantile_range_guard_measured.json"
    measured_task_utility = root / args.output_dir / "task_utility_guard_yolov10n_measured.json"

    dmr_rows, dmr_summary = _screen_guard_gated_dmr(resnet_candidates, range_report)
    _write_table(out_dir / "dmr_guard_gated_comparison", dmr_rows, dmr_summary)

    adaptive_rows, adaptive_summary = _screen_adaptive_range(range_report, measured_adaptive_range)
    _write_table(out_dir / "adaptive_quantile_range_guard", adaptive_rows, adaptive_summary)

    task_rows, task_summary = _screen_task_utility_guard(yolo_guard, measured_task_utility)
    _write_table(out_dir / "task_utility_guard_yolov10n", task_rows, task_summary)

    robust_rows, robust_summary = _screen_robust_memory_ilp(
        resnet_candidates,
        resnet_faults,
        latency_budgets=[float(value) for value in args.latency_budgets_ms.split(",")],
        memory_budgets=[int(value) for value in args.memory_budgets_bytes.split(",")],
        risk_loss_tolerance=args.risk_loss_tolerance,
    )
    _write_table(out_dir / "confidence_robust_bounded_loss_ilp", robust_rows, robust_summary)

    summary = _decision_markdown(
        {
            "Guard-Gated DMR": dmr_summary,
            "Adaptive Quantile Range Guard": adaptive_summary,
            "Task-Utility Consistency Guard": task_summary,
            "Confidence-Robust Bounded-Loss Memory ILP": robust_summary,
        }
    )
    (out_dir / "screening_decision_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summaries": 4}, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_candidates(path: Path) -> list[CandidateMode]:
    return [CandidateMode(**item) for item in _read_json(path)]


def _write_table(prefix: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    pd.DataFrame(csv_rows).to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _candidate_map(candidates: list[CandidateMode]) -> dict[tuple[int, str], CandidateMode]:
    return {(candidate.node_id, candidate.mode): candidate for candidate in candidates}


def _screen_guard_gated_dmr(candidates: list[CandidateMode], range_report: dict[str, Any]) -> tuple[list[dict], dict]:
    by_key = _candidate_map(candidates)
    rows = []
    total_always_risk = 0.0
    total_gated_risk = 0.0
    total_always_latency = 0.0
    total_gated_latency = 0.0
    max_always_memory = 0
    max_gated_memory = 0
    for record in range_report.get("records", []):
        node_id = int(record["runtime_node_id"])
        dmr = by_key.get((node_id, "dmr_compare_rerun"))
        guard = by_key.get((node_id, "range_guard_rerun"))
        if not dmr or not guard:
            continue
        coverage = float(record.get("critical_coverage_ci_low", record.get("critical_coverage", 0.0)))
        false_positive = float(record.get("false_positive_ci_high", record.get("false_positive_rate", 0.0)))
        gated_risk = dmr.risk_reduction * coverage
        gated_latency = guard.latency_overhead_ms + false_positive * dmr.latency_overhead_ms
        latency_reduction = 1.0 - gated_latency / dmr.latency_overhead_ms if dmr.latency_overhead_ms else 0.0
        recovery_retention = gated_risk / dmr.risk_reduction if dmr.risk_reduction else 0.0
        decision = decide_guard_gated_dmr(
            latency_reduction=latency_reduction,
            recovery_retention=recovery_retention,
        )
        rows.append(
            {
                "runtime_node_id": node_id,
                "tensor_id": dmr.tensor_id,
                "always_dmr_risk_reduction": dmr.risk_reduction,
                "gated_dmr_risk_reduction": gated_risk,
                "always_dmr_latency_ms": dmr.latency_overhead_ms,
                "gated_dmr_expected_latency_ms": gated_latency,
                "latency_reduction": latency_reduction,
                "recovery_retention": recovery_retention,
                "always_dmr_extra_memory_bytes": dmr.extra_memory_bytes,
                "gated_dmr_peak_extra_memory_bytes": dmr.extra_memory_bytes,
                "decision": decision.decision,
                "reason": decision.reason,
            }
        )
        total_always_risk += dmr.risk_reduction
        total_gated_risk += gated_risk
        total_always_latency += dmr.latency_overhead_ms
        total_gated_latency += gated_latency
        max_always_memory = max(max_always_memory, dmr.extra_memory_bytes)
        max_gated_memory = max(max_gated_memory, dmr.extra_memory_bytes)
    overall_latency_reduction = 1.0 - total_gated_latency / total_always_latency if total_always_latency else 0.0
    overall_retention = total_gated_risk / total_always_risk if total_always_risk else 0.0
    decision = decide_guard_gated_dmr(
        latency_reduction=overall_latency_reduction,
        recovery_retention=overall_retention,
    )
    return rows, {
        "mechanism": "guard_gated_dmr",
        "evidence": "existing ResNet50 range guard profile and measured protection candidates",
        "nodes_evaluated": len(rows),
        "latency_reduction": overall_latency_reduction,
        "recovery_retention": overall_retention,
        "always_dmr_peak_extra_memory_bytes": max_always_memory,
        "gated_dmr_peak_extra_memory_bytes": max_gated_memory,
        "decision": decision.decision,
        "reason": decision.reason,
    }


def _screen_adaptive_range(range_report: dict[str, Any], measured_path: Path) -> tuple[list[dict], dict]:
    if measured_path.exists():
        measured = _read_json(measured_path)
        rows = []
        best_decision = None
        for item in measured.get("summary", []):
            decision = decide_adaptive_range(
                false_positive_rate=float(item.get("false_positive_rate", 1.0)),
                coverage_gain=float(item.get("coverage_gain_vs_minmax", 0.0)),
            )
            row = dict(item)
            row["status"] = "measured"
            row["decision"] = decision.decision
            row["reason"] = decision.reason
            rows.append(row)
            if best_decision is None or _decision_rank(decision.decision) > _decision_rank(best_decision.decision):
                best_decision = decision
        best = max(rows, key=lambda row: _decision_rank(row["decision"])) if rows else {}
        return rows, {
            "mechanism": "adaptive_quantile_range_guard",
            "evidence": str(measured_path),
            "policies_evaluated": len(rows),
            "best_policy": best.get("policy"),
            "best_critical_coverage": best.get("critical_coverage"),
            "best_false_positive_rate": best.get("false_positive_rate"),
            "best_coverage_gain_vs_minmax": best.get("coverage_gain_vs_minmax"),
            "decision": best.get("decision", "reject"),
            "reason": best.get("reason", "no measured policy passed the threshold"),
        }
    rows = []
    minmax = range_report.get("records", [])
    for record in minmax:
        baseline_coverage = float(record.get("critical_coverage", 0.0))
        baseline_fp = float(record.get("false_positive_rate", 0.0))
        for policy in ("minmax", "q99.9", "q99.5", "q99"):
            if policy == "minmax":
                coverage = baseline_coverage
                fp = baseline_fp
                status = "measured_existing_profile"
                gain = 0.0
            else:
                coverage = None
                fp = None
                status = "needs_raw_activation_replay"
                gain = 0.0
            decision = (
                decide_adaptive_range(false_positive_rate=baseline_fp, coverage_gain=gain)
                if policy == "minmax"
                else None
            )
            rows.append(
                {
                    "runtime_node_id": int(record["runtime_node_id"]),
                    "semantic_node_id": int(record["node_id"]),
                    "threshold_policy": policy,
                    "critical_coverage": coverage,
                    "false_positive_rate": fp,
                    "coverage_gain_vs_minmax": gain,
                    "status": status,
                    "decision": decision.decision if decision else "reject",
                    "reason": decision.reason if decision else "raw activation extrema/quantiles are not present in existing artifacts",
                }
            )
    summary = {
        "mechanism": "adaptive_quantile_range_guard",
        "evidence": "existing minmax profile only; quantile policies require activation replay",
        "nodes_evaluated": len(minmax),
        "decision": "reject",
        "reason": "quantile variants were not adopted because existing artifacts do not contain raw clean/faulted activation extrema needed for a real measurement",
    }
    return rows, summary


def _screen_task_utility_guard(yolo_guard: dict[str, Any], measured_path: Path) -> tuple[list[dict], dict]:
    rates = yolo_guard.get("rates", {})
    totals = yolo_guard.get("totals", {})
    baseline_reduction = float(rates.get("observed_reduction_ratio", 0.0))
    baseline_trigger = float(rates.get("guard_trigger_rate", 0.0))
    baseline_fp = float(rates.get("clean_false_positive_rate_upper_bound", 0.0))
    if measured_path.exists():
        measured = _read_json(measured_path)
        measured_rates = measured.get("rates", {})
        fp = float(measured_rates.get("utility_clean_false_positive_rate", 1.0))
        gain = float(measured_rates.get("reduction_gain_vs_current", 0.0))
        trigger_reduction = float(measured_rates.get("trigger_reduction_vs_current", 0.0))
        decision = decide_task_utility_guard(
            false_positive_rate=fp,
            reduction_gain=gain,
            trigger_reduction=trigger_reduction,
        )
        rows = [
            {
                "policy": "current_task_output_guard",
                "status": "measured",
                "events": int(measured.get("totals", {}).get("events", 0)),
                "critical_failure_reduction_ratio": float(measured_rates.get("current_reduction_ratio", baseline_reduction)),
                "guard_trigger_rate": float(measured_rates.get("current_guard_trigger_rate", baseline_trigger)),
                "clean_false_positive_rate": float(measured_rates.get("current_clean_false_positive_rate", baseline_fp)),
                "reduction_gain_vs_current": 0.0,
                "trigger_reduction_vs_current": 0.0,
                "decision": "keep_as_ablation",
                "reason": "measured baseline task output guard",
            },
            {
                "policy": "task_utility_consistency_guard",
                "status": "measured",
                "events": int(measured.get("totals", {}).get("events", 0)),
                "critical_failure_reduction_ratio": float(measured_rates.get("utility_reduction_ratio", 0.0)),
                "guard_trigger_rate": float(measured_rates.get("utility_guard_trigger_rate", 0.0)),
                "clean_false_positive_rate": fp,
                "reduction_gain_vs_current": gain,
                "trigger_reduction_vs_current": trigger_reduction,
                "decision": decision.decision,
                "reason": decision.reason,
            },
        ]
        return rows, {
            "mechanism": "task_utility_consistency_guard",
            "evidence": str(measured_path),
            "baseline_task_output_guard_reduction": float(measured_rates.get("current_reduction_ratio", baseline_reduction)),
            "task_utility_guard_reduction": float(measured_rates.get("utility_reduction_ratio", 0.0)),
            "reduction_gain_vs_current": gain,
            "utility_guard_trigger_rate": float(measured_rates.get("utility_guard_trigger_rate", 0.0)),
            "utility_clean_false_positive_rate": fp,
            "decision": decision.decision,
            "reason": decision.reason,
        }
    rows = [
        {
            "policy": "current_task_output_guard",
            "status": "measured_existing_profile",
            "events": int(totals.get("events", 0)),
            "critical_failure_reduction_ratio": baseline_reduction,
            "guard_trigger_rate": baseline_trigger,
            "clean_false_positive_rate": baseline_fp,
            "reduction_gain_vs_current": 0.0,
            "trigger_reduction_vs_current": 0.0,
            "decision": "keep_as_ablation",
            "reason": "current guard is measured and useful, but not a new task-utility variant",
        },
        {
            "policy": "task_utility_consistency_guard",
            "status": "needs_faulted_output_replay",
            "events": int(totals.get("events", 0)),
            "critical_failure_reduction_ratio": None,
            "guard_trigger_rate": None,
            "clean_false_positive_rate": None,
            "reduction_gain_vs_current": None,
            "trigger_reduction_vs_current": None,
            "decision": "reject",
            "reason": "existing report stores guard decisions and severity but not enough output statistics to verify target-count/confidence/box-distribution rules",
        },
    ]
    decision = decide_task_utility_guard(
        false_positive_rate=baseline_fp,
        reduction_gain=0.0,
        trigger_reduction=0.0,
    )
    summary = {
        "mechanism": "task_utility_consistency_guard",
        "baseline_task_output_guard_reduction": baseline_reduction,
        "baseline_guard_trigger_rate": baseline_trigger,
        "decision": decision.decision,
        "reason": "new task-utility rules require replaying faulted YOLO outputs; current measured task output guard remains an ablation result",
    }
    return rows, summary


def _screen_robust_memory_ilp(
    candidates: list[CandidateMode],
    faults_jsonl: Path,
    *,
    latency_budgets: list[float],
    memory_budgets: list[int],
    risk_loss_tolerance: float,
) -> tuple[list[dict], dict]:
    factors = _node_robustness_factors(faults_jsonl)
    original_by_key = {(candidate.node_id, candidate.mode): candidate for candidate in candidates}
    robust_candidates = [
        CandidateMode(
            node_id=candidate.node_id,
            tensor_id=candidate.tensor_id,
            mode=candidate.mode,
            risk_reduction=candidate.risk_reduction * factors.get(candidate.node_id, 0.0),
            latency_overhead_ms=candidate.latency_overhead_ms,
            extra_memory_bytes=candidate.extra_memory_bytes,
            lower_bound=candidate.lower_bound,
            upper_bound=candidate.upper_bound,
        )
        for candidate in candidates
    ]
    rows = []
    best_decision = None
    for memory in memory_budgets:
        for latency in latency_budgets:
            ilp = optimize_ilp(candidates, latency_budget_ms=latency, memory_budget_bytes=memory)
            greedy = optimize_greedy(candidates, latency_budget_ms=latency, memory_budget_bytes=memory)
            topk = optimize_topk_single_mode(
                candidates,
                mode="dmr_compare_rerun",
                latency_budget_ms=latency,
                memory_budget_bytes=memory,
            )
            bounded = optimize_bounded_loss_memory_ilp(
                candidates,
                latency_budget_ms=latency,
                memory_budget_bytes=memory,
                risk_loss_tolerance=risk_loss_tolerance,
            )
            robust = optimize_bounded_loss_memory_ilp(
                robust_candidates,
                latency_budget_ms=latency,
                memory_budget_bytes=memory,
                risk_loss_tolerance=risk_loss_tolerance,
            )
            robust_original = [original_by_key[(choice.node_id, choice.mode)] for choice in robust.selected]
            robust_mean_risk = sum(choice.risk_reduction for choice in robust_original)
            robust_robust_risk = sum(choice.risk_reduction for choice in robust.selected)
            risk_retention = robust_mean_risk / ilp.total_risk_reduction if ilp.total_risk_reduction else 1.0
            memory_saving = (
                (ilp.total_extra_memory_bytes - robust.total_extra_memory_bytes) / ilp.total_extra_memory_bytes
                if ilp.total_extra_memory_bytes
                else 0.0
            )
            decision = decide_robust_memory_ilp(
                risk_retention=risk_retention,
                memory_saving_ratio=memory_saving,
            )
            if best_decision is None or _decision_rank(decision.decision) > _decision_rank(best_decision.decision):
                best_decision = decision
            rows.append(
                {
                    "latency_budget_ms": latency,
                    "peak_memory_budget_bytes": memory,
                    "ilp_risk_reduction": ilp.total_risk_reduction,
                    "greedy_risk_reduction": greedy.total_risk_reduction,
                    "topk_dmr_risk_reduction": topk.total_risk_reduction,
                    "bounded_loss_risk_reduction": bounded.total_risk_reduction,
                    "robust_mean_risk_reduction": robust_mean_risk,
                    "robust_lower_bound_objective": robust_robust_risk,
                    "risk_retention_vs_ilp": risk_retention,
                    "ilp_peak_extra_memory_bytes": ilp.total_extra_memory_bytes,
                    "robust_peak_extra_memory_bytes": robust.total_extra_memory_bytes,
                    "memory_saving_ratio_vs_ilp": memory_saving,
                    "robust_latency_ms": robust.total_latency_overhead_ms,
                    "robust_selected_count": len(robust.selected),
                    "decision": decision.decision,
                    "reason": decision.reason,
                }
            )
    best_row = max(rows, key=lambda row: _decision_rank(row["decision"]))
    return rows, {
        "mechanism": "confidence_robust_bounded_loss_memory_ilp",
        "evidence": "ResNet50 30000-event formal runtime fault log and measured protection candidates",
        "risk_loss_tolerance": risk_loss_tolerance,
        "best_latency_budget_ms": best_row["latency_budget_ms"],
        "best_peak_memory_budget_bytes": best_row["peak_memory_budget_bytes"],
        "best_risk_retention_vs_ilp": best_row["risk_retention_vs_ilp"],
        "best_memory_saving_ratio_vs_ilp": best_row["memory_saving_ratio_vs_ilp"],
        "decision": best_row["decision"],
        "reason": best_row["reason"],
    }


def _node_robustness_factors(path: Path) -> dict[int, float]:
    counts: dict[int, int] = defaultdict(int)
    failures: dict[int, int] = defaultdict(int)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            node_id = int(record["node_id"])
            counts[node_id] += 1
            failures[node_id] += int(bool(record.get("unprotected_critical_failure", False)))
    factors = {}
    for node_id, count in counts.items():
        fail = failures[node_id]
        mean = fail / count if count else 0.0
        lower = wilson_lower_bound(fail, count)
        factors[node_id] = lower / mean if mean > 0.0 else 0.0
    return factors


def _decision_rank(decision: str) -> int:
    return {"reject": 0, "keep_as_ablation": 1, "adopt_as_main": 2}.get(decision, 0)


def _decision_markdown(summaries: dict[str, dict[str, Any]]) -> str:
    lines = ["# 保护机制改进筛选结论", ""]
    for name, summary in summaries.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- decision: `{summary.get('decision')}`",
                f"- reason: {summary.get('reason')}",
                "",
            ]
        )
        for key, value in summary.items():
            if key in {"decision", "reason", "mechanism", "evidence"}:
                continue
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
