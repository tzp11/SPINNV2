"""Export a compact task-aware YOLO runtime fault-screening report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk", required=True)
    parser.add_argument("--screen-summary", required=True)
    parser.add_argument("--injections", help="Optional JSONL observations for failure-mode breakdown.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--workload-scope",
        default="DIOR prevalidation subset, not full-task thesis result",
    )
    parser.add_argument("--title", default="YOLOv10n DIOR runtime fault screening")
    args = parser.parse_args()

    records = pd.DataFrame(json.loads(Path(args.risk).read_text(encoding="utf-8")))
    summary = json.loads(Path(args.screen_summary).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / "yolov10n_runtime_risk_ranked.csv", index=False)
    top = records.head(args.top_k).iloc[::-1]
    fig, axis = plt.subplots(figsize=(8.5, max(4.5, 0.34 * len(top))))
    labels = [f"node {node} / tensor {tensor}" for node, tensor in zip(top["node_id"], top["tensor_id"])]
    axis.barh(labels, top["risk"], color="#295f9e")
    axis.set_xlabel("Normalized task-level risk score")
    axis.set_title(args.title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "yolov10n_runtime_risk_topk.png", dpi=220)
    plt.close(fig)
    report = {
        "workload_scope": args.workload_scope,
        "observations": int(summary["observations"]),
        "eligible_samples": int(summary["eligible_samples"]),
        "critical_failures": int(summary["critical_failures"]),
        "critical_failure_rate": float(summary["critical_failures"] / summary["observations"]),
        "sampled_nodes": int(len(records)),
        "highest_risk_node": int(records.iloc[0]["node_id"]),
        "highest_risk_score": float(records.iloc[0]["risk"]),
        "highest_node_critical_probability": float(records.iloc[0]["critical_probability"]),
    }
    if args.injections:
        modes = Counter()
        with Path(args.injections).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    modes[json.loads(line).get("failure_mode", "unclassified")] += 1
        report["failure_modes"] = dict(modes)
    (output_dir / "yolov10n_runtime_risk_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
