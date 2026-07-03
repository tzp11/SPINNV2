"""Compare a YOLO control-path protection plan against an unprotected runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-injections", required=True)
    parser.add_argument("--protected-injections", required=True)
    parser.add_argument("--baseline-bench", required=True)
    parser.add_argument("--protected-bench", required=True)
    parser.add_argument("--protected-summary", help="Optional screen summary containing runtime recovery counters.")
    parser.add_argument("--extra-memory-bytes", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--workload-scope",
        default="DIOR prevalidation subset, not full-task thesis result",
    )
    parser.add_argument("--fault-sampling", default="stratified runtime-object injection")
    parser.add_argument("--protected-label", default="control_dmr")
    parser.add_argument("--title", default="YOLOv10n control-path protection")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    baseline_modes, baseline_total = _modes(args.baseline_injections)
    protected_modes, protected_total = _modes(args.protected_injections)
    if baseline_total != protected_total:
        raise ValueError("baseline and protected injection runs must contain the same event count")
    baseline_bench = json.loads(Path(args.baseline_bench).read_text(encoding="ascii"))
    protected_bench = json.loads(Path(args.protected_bench).read_text(encoding="ascii"))
    baseline_failures = baseline_total - baseline_modes["none"]
    protected_failures = protected_total - protected_modes["none"]
    reduction_ci_low, reduction_ci_high = _paired_reduction_ci(
        args.baseline_injections,
        args.protected_injections,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    latency_delta = float(protected_bench["avg_ms"] - baseline_bench["avg_ms"])
    report = {
        "workload_scope": args.workload_scope,
        "fault_sampling": args.fault_sampling,
        "baseline": {
            "critical_failures": baseline_failures,
            "failure_modes": dict(baseline_modes),
            "latency_avg_ms": float(baseline_bench["avg_ms"]),
        },
        args.protected_label: {
            "critical_failures": protected_failures,
            "failure_modes": dict(protected_modes),
            "latency_avg_ms": float(protected_bench["avg_ms"]),
            "extra_memory_bytes": args.extra_memory_bytes,
        },
        "critical_failure_reduction": baseline_failures - protected_failures,
        "critical_failure_reduction_ratio": (baseline_failures - protected_failures) / baseline_failures,
        "critical_failure_reduction_ratio_bootstrap_95_ci": [reduction_ci_low, reduction_ci_high],
        "latency_overhead_ms": latency_delta,
        "latency_overhead_ratio": latency_delta / float(baseline_bench["avg_ms"]),
    }
    if args.protected_summary:
        protected_summary = json.loads(Path(args.protected_summary).read_text(encoding="utf-8"))
        report[args.protected_label]["reliability_stats"] = protected_summary.get("reliability_stats", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "yolov10n_control_dmr_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_failure_modes(
        baseline_modes,
        protected_modes,
        output_dir / "yolov10n_control_dmr_failure_modes.png",
        title=args.title,
    )
    print(json.dumps(report, indent=2))
    return 0


def _modes(path: str) -> tuple[Counter[str], int]:
    modes: Counter[str] = Counter()
    total = 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                modes[json.loads(line).get("failure_mode", "unclassified")] += 1
                total += 1
    return modes, total


def _paired_reduction_ci(baseline_path: str, protected_path: str, *, iterations: int, seed: int) -> tuple[float, float]:
    def failures(path: str) -> np.ndarray:
        with Path(path).open(encoding="utf-8") as source:
            return np.asarray(
                [bool(json.loads(line)["critical_failure"]) for line in source if line.strip()],
                dtype=np.int8,
            )

    baseline = failures(baseline_path)
    protected = failures(protected_path)
    if baseline.shape != protected.shape:
        raise ValueError("baseline and protected injection runs must pair the same events")
    generator = np.random.default_rng(seed)
    ratios = []
    for _ in range(iterations):
        sampled = generator.integers(0, len(baseline), size=len(baseline))
        sampled_baseline = int(baseline[sampled].sum())
        if sampled_baseline == 0:
            continue
        sampled_protected = int(protected[sampled].sum())
        ratios.append((sampled_baseline - sampled_protected) / sampled_baseline)
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def _plot_failure_modes(baseline: Counter[str], protected: Counter[str], path: Path, *, title: str) -> None:
    modes = ["task_output_failure", "controlled_execution_error"]
    labels = ["Task output failure", "Controlled execution error"]
    x = range(len(modes))
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.bar([value - 0.18 for value in x], [baseline[mode] for mode in modes], width=0.36, label="Unprotected")
    axis.bar([value + 0.18 for value in x], [protected[mode] for mode in modes], width=0.36, label="Control-path DMR")
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Critical failures in stratified injections")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
