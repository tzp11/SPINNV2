"""Export paper-ready summaries and figures from final runtime validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.reliability.profiling.risk_profile import wilson_interval


def main() -> int:
    parser = argparse.ArgumentParser(description="Create final runtime validation summary artifacts.")
    parser.add_argument("--fault-report", required=True)
    parser.add_argument("--clean-report", required=True)
    parser.add_argument("--budget-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    fault = json.loads(Path(args.fault_report).read_text(encoding="utf-8"))
    clean = json.loads(Path(args.clean_report).read_text(encoding="utf-8"))
    budget = json.loads(Path(args.budget_summary).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = int(fault["totals"]["events"])
    baseline_failures = int(fault["totals"]["unprotected_critical_failures"])
    protected_failures = int(fault["totals"]["protected_critical_failures"])
    base_rate = baseline_failures / total
    protected_rate = protected_failures / total
    base_ci = wilson_interval(baseline_failures, total)
    protected_ci = wilson_interval(protected_failures, total)
    reduction_interval = _paired_bootstrap_reduction(fault["observations"], args.bootstrap_repeats, args.seed)
    final_plan = next(
        row
        for row in budget["representative_results"]
        if row["method"] == "ilp"
    )
    summary = {
        "runtime_backend": clean["platform_profile"],
        "clean_evaluation": {
            "samples": clean["evaluation"]["samples"],
            "baseline_accuracy": clean["clean_baseline"]["accuracy"],
            "protected_accuracy": clean["clean_protected"]["accuracy"],
            "prediction_agreement": clean["clean_protected"]["prediction_agreement_with_baseline"],
            "false_alarm_rate": clean["clean_protected"]["false_alarm_rate"],
            "baseline_latency_ms": clean["clean_baseline"]["latency_ms"]["avg"],
            "protected_latency_ms": clean["clean_protected"]["latency_ms"]["avg"],
            "latency_overhead_ms": clean["latency_overhead"]["avg_ms"],
            "latency_overhead_ratio": clean["latency_overhead"]["avg_ratio"],
        },
        "fault_evaluation": {
            "events": total,
            "unprotected_critical_failures": baseline_failures,
            "protected_critical_failures": protected_failures,
            "unprotected_failure_rate": base_rate,
            "unprotected_failure_rate_wilson95": list(base_ci),
            "protected_failure_rate": protected_rate,
            "protected_failure_rate_wilson95": list(protected_ci),
            "observed_reduction_ratio": (base_rate - protected_rate) / base_rate if base_rate else 0.0,
            "paired_bootstrap_reduction_ratio_95": list(reduction_interval),
            "detected_faults": fault["totals"]["detected_faults"],
            "recovered_faults": fault["totals"]["recovered_faults"],
        },
        "budget_optimization": {
            "latency_budget_ms": budget["representative_budget"]["latency_overhead_ms"],
            "extra_memory_budget_bytes": budget["representative_budget"]["peak_extra_memory_bytes"],
            "predicted_risk_reduction_ratio": final_plan["mitigation_ratio"],
            "selected_count": final_plan["selected_count"],
            "used_peak_extra_memory_bytes": final_plan["used_peak_extra_memory_bytes"],
        },
    }
    (output_dir / "resnet50_final_runtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_failure_rates(base_rate, protected_rate, base_ci, protected_ci, output_dir / "resnet50_runtime_fault_mitigation.png")
    _plot_latency(clean, output_dir / "resnet50_runtime_latency_overhead.png")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _paired_bootstrap_reduction(observations: list[dict], repeats: int, seed: int) -> tuple[float, float]:
    baseline = np.asarray([int(item["unprotected_critical_failure"]) for item in observations], dtype=float)
    protected = np.asarray([int(item["protected_critical_failure"]) for item in observations], dtype=float)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = generator.integers(0, len(baseline), size=len(baseline))
        before = baseline[sample].mean()
        if before > 0:
            estimates.append((before - protected[sample].mean()) / before)
    if not estimates:
        return 0.0, 0.0
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _plot_failure_rates(
    baseline: float,
    protected: float,
    baseline_ci: tuple[float, float],
    protected_ci: tuple[float, float],
    output: Path,
) -> None:
    values = np.asarray([baseline, protected]) * 100.0
    intervals = [baseline_ci, protected_ci]
    lower = np.asarray([values[index] - interval[0] * 100.0 for index, interval in enumerate(intervals)])
    upper = np.asarray([interval[1] * 100.0 - values[index] for index, interval in enumerate(intervals)])
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Unprotected", "Protected"], values, color=["#c44e52", "#4c72b0"])
    axis.errorbar([0, 1], values, yerr=[lower, upper], fmt="none", ecolor="black", capsize=5)
    axis.set_ylabel("Critical task failure rate (%)")
    axis.set_title("ResNet50 runtime single-bit fault mitigation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_latency(clean: dict, output: Path) -> None:
    values = [
        clean["clean_baseline"]["latency_ms"]["avg"],
        clean["clean_protected"]["latency_ms"]["avg"],
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Unprotected", "Protected"], values, color=["#8172b3", "#55a868"])
    axis.set_ylabel("Mean latency (ms)")
    axis.set_title("ResNet50 protected inference overhead")
    axis.grid(axis="y", alpha=0.25)
    axis.text(1, values[1], f"+{clean['latency_overhead']['avg_ms']:.2f} ms", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
