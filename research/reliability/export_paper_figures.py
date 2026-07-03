"""Generate paper-facing figures from aggregated reliability reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _barh_risk(csv_path: Path, out_path: Path, title: str, *, id_col: str, topn: int = 20) -> None:
    table = pd.read_csv(csv_path)
    table = table[table[id_col].notna()].copy()
    table = table.sort_values("risk", ascending=False).head(topn)
    labels = [str(int(value)) for value in table[id_col]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], table["risk"].to_numpy()[::-1], color="#4C78A8")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Runtime node id")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _method_comparison(csv_path: Path, out_path: Path, model: str, title: str) -> None:
    table = pd.read_csv(csv_path)
    subset = table[(table["model"] == model) & (table["comparison_type"] == "predicted_budget_selection")].copy()
    subset = subset.sort_values("mitigation_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(subset["method"], subset["mitigation_ratio"], color="#59A14F")
    ax.set_ylabel("Mitigation ratio")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _stability_plot(csv_path: Path, out_path: Path, title: str) -> None:
    table = pd.read_csv(csv_path)
    metrics = [
        ("top5_overlap_vs_full", "Top-5 overlap"),
        ("top10_overlap_vs_full", "Top-10 overlap"),
        ("top20_overlap_vs_full", "Top-20 overlap"),
        ("spearman_vs_full", "Spearman"),
    ]
    values = [table[col].mean() for col, _label in metrics]
    errors = [table[col].std(ddof=0) for col, _label in metrics]
    labels = [label for _col, label in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, yerr=errors, color="#F28E2B", capsize=4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Bootstrap mean ± std")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _yolo_grouped(json_path: Path, out_path: Path) -> None:
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    groups = doc["groups"]
    labels = [group["group"] for group in groups]
    values = [group["reduction_ratio"] for group in groups]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color="#B07AA1")
    ax.set_ylabel("Critical-failure reduction ratio")
    ax.set_title("YOLOv10n DIOR grouped protection comparison")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _formal_completion(csv_path: Path, out_path: Path) -> None:
    table = pd.read_csv(csv_path)
    labels = [f"{row.model}\n{row.protected_method}" for row in table.itertuples(index=False)]
    values = table["critical_failure_reduction_ratio"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#76B7B2")
    ax.set_ylabel("Critical-failure reduction ratio")
    ax.set_title("Formal Windows completion experiments")
    ax.set_ylim(0, max(0.8, float(values.max()) + 0.1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="artifacts/reports/paper_assets")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _barh_risk(
        root / "artifacts/reports/resnet50_gpu_prevalidation/resnet50_risk_ranked.csv",
        out_dir / "ch3_risk_rank_resnet50.png",
        "ResNet50 EuroSAT runtime-object risk ranking",
        id_col="runtime_node_id",
    )
    _barh_risk(
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/stratified/yolov10n_runtime_risk_ranked.csv",
        out_dir / "ch3_risk_rank_yolov10n.png",
        "YOLOv10n DIOR runtime-object risk ranking",
        id_col="node_id",
    )
    _method_comparison(
        out_dir / "baseline_comparison.csv",
        out_dir / "ch5_resnet50_method_comparison.png",
        "resnet50_eurosat",
        "ResNet50 budgeted protection method comparison",
    )
    _method_comparison(
        out_dir / "baseline_comparison.csv",
        out_dir / "ch5_yolov10n_method_comparison.png",
        "yolov10n_dior_full_e30_b16",
        "YOLOv10n budgeted DMR method comparison",
    )
    _stability_plot(
        out_dir / "risk_stability_resnet50_formal.csv"
        if (out_dir / "risk_stability_resnet50_formal.csv").exists()
        else out_dir / "risk_stability_resnet50.csv",
        out_dir / "ch3_risk_stability_resnet50.png",
        "ResNet50 risk-ranking bootstrap stability",
    )
    _stability_plot(
        out_dir / "risk_stability_yolov10n_formal.csv"
        if (out_dir / "risk_stability_yolov10n_formal.csv").exists()
        else out_dir / "risk_stability_yolov10n.csv",
        out_dir / "ch3_risk_stability_yolov10n.png",
        "YOLOv10n risk-ranking bootstrap stability",
    )
    _yolo_grouped(
        out_dir / "yolov10n_grouped_protection_comparison.json",
        out_dir / "ch5_yolov10n_grouped_protection.png",
    )
    formal_summary = out_dir / "formal_windows_completion_summary.csv"
    figure_count = 7
    if formal_summary.exists():
        _formal_completion(formal_summary, out_dir / "ch5_formal_windows_completion_summary.png")
        figure_count += 1
    print(json.dumps({"out_dir": str(out_dir), "figures": figure_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
