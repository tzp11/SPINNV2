"""Export ranked risk tables and a compact screening figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk", required=True)
    parser.add_argument("--runtime-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    risk = pd.DataFrame(json.loads(Path(args.risk).read_text(encoding="utf-8")))
    mappings = pd.DataFrame(json.loads(Path(args.runtime_map).read_text(encoding="utf-8"))["mappings"])
    columns = [
        "node_id",
        "module_name",
        "invocation_index",
        "retained_after_passes",
        "runtime_node_id",
        "runtime_tensor_id",
    ]
    ranked = risk.merge(mappings[columns], on="node_id", how="left")
    ranked["candidate"] = ranked["module_name"] + "@" + ranked["invocation_index"].astype(str)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_dir / "resnet50_risk_ranked.csv", index=False)
    top = ranked.head(args.top_k).iloc[::-1]
    colors = ["#295f9e" if retained else "#a6a6a6" for retained in top["retained_after_passes"]]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(top))))
    ax.barh(top["candidate"], top["risk"], color=colors)
    ax.set_xlabel("Task-level risk score")
    ax.set_ylabel("Runtime output candidate")
    ax.set_title("ResNet50 EuroSAT fault screening (top candidates)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "resnet50_risk_topk.png", dpi=200)
    plt.close(fig)
    summary = {
        "candidate_count": int(len(ranked)),
        "retained_after_passes": int(ranked["retained_after_passes"].sum()),
        "top_candidate": ranked.iloc[0]["candidate"],
        "top_risk": float(ranked.iloc[0]["risk"]),
        "top_critical_probability": float(ranked.iloc[0]["critical_probability"]),
        "top_runtime_node_id": int(ranked.iloc[0]["runtime_node_id"]),
        "top_runtime_tensor_id": int(ranked.iloc[0]["runtime_tensor_id"]),
    }
    (output_dir / "resnet50_risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
