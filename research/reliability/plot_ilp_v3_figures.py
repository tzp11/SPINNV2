"""Generate paper figures: proposed method vs baselines + ablations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS_ORDER = [
    "ilp-v3", "ranger",
    "ilp", "ilp-static", "ilp-v2",
    "filr", "aspis", "ruospo", "ahmadilivani", "uniform-dmr",
]

METHOD_LABELS = {
    "ilp-v3": "Proposed",
    "ilp": "w/o coverage floor",
    "ilp-static": "w/o calibration",
    "ilp-v2": "Two-tier",
    "ranger": "Ranger [Chen 2021]",
    "filr": "FILR [Chen 2019]",
    "aspis": "ASPIS [Mahmoud 2021]",
    "ruospo": "Ruospo [2023]",
    "ahmadilivani": "Ahmadilivani [2024]",
    "uniform-dmr": "Uniform-DMR",
}

METHOD_COLORS = {
    "ilp-v3": "#D4380D",
    "ilp": "#E15759",
    "ilp-static": "#FF9DA7",
    "ilp-v2": "#F28E2B",
    "ranger": "#4E79A7",
    "filr": "#59A14F",
    "aspis": "#EDC948",
    "ruospo": "#B07AA1",
    "ahmadilivani": "#76B7B2",
    "uniform-dmr": "#9C755F",
}

METHOD_MARKERS = {
    "ilp-v3": "o",
    "ilp": "D",
    "ilp-static": "h",
    "ilp-v2": "*",
    "ranger": "s",
    "filr": "^",
    "aspis": "v",
    "ruospo": "<",
    "ahmadilivani": "P",
    "uniform-dmr": "X",
}

MODELS = {
    "resnet50": {
        "label": "ResNet50 EuroSAT",
        "mc_dir": "artifacts/experiments/macos_reliability/resnet50/mc_results",
        "budgets": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "events": 5000,
    },
    "yolov10n": {
        "label": "YOLOv10n DIOR",
        "mc_dir": "artifacts/experiments/macos_reliability/yolov10n_dior/mc_results",
        "budgets": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "events": 5000,
    },
}


def load_all_results(root: Path) -> dict[str, dict[str, list[tuple[float, int]]]]:
    results = {}
    for model_key, cfg in MODELS.items():
        mc_dir = root / cfg["mc_dir"]
        results[model_key] = {}
        for method in METHODS_ORDER:
            data = []
            for budget in cfg["budgets"]:
                fname = f"mc{cfg['events']}_obsfloor_{method}_{budget}ms.json"
                fpath = mc_dir / fname
                if not fpath.exists():
                    continue
                d = json.loads(fpath.read_text(encoding="utf-8"))
                cf = d["totals"]["protected_critical_failures"]
                data.append((budget, cf))
            if data:
                results[model_key][method] = data
        baseline_f = list(mc_dir.glob(f"mc{cfg['events']}_obsfloor_ranger_*.json"))
        if baseline_f:
            d = json.loads(baseline_f[0].read_text(encoding="utf-8"))
            results[model_key]["_baseline_cf"] = d["totals"]["unprotected_critical_failures"]
            results[model_key]["_events"] = d["totals"]["events"]
    return results


def plot_cf_comparison(results: dict, model_key: str, cfg: dict, out_path: Path) -> None:
    """Main figure: Proposed vs baselines only (no ablations)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    baseline_cf = results.get("_baseline_cf", 0)

    main_methods = ["ilp-v3", "ranger", "filr", "aspis", "ruospo", "ahmadilivani", "uniform-dmr"]
    for method in main_methods:
        if method not in results:
            continue
        data = sorted(results[method])
        budgets = [d[0] for d in data]
        cfs = [d[1] for d in data]
        lw = 3.0 if method == "ilp-v3" else 1.5
        ms = 9 if method == "ilp-v3" else 6
        zorder = 10 if method == "ilp-v3" else 5
        ax.plot(budgets, cfs,
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                label=METHOD_LABELS[method], linewidth=lw, markersize=ms,
                zorder=zorder)

    ax.axhline(y=baseline_cf, color="gray", linestyle="--", alpha=0.5,
               label=f"Unprotected (CF={baseline_cf})")
    ax.set_xlabel("Latency budget (ms)", fontsize=12)
    ax.set_ylabel("Critical failures (5000 events)", fontsize=12)
    ax.set_title(f"{cfg['label']}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(cfg["budgets"])
    ax.set_ylim(bottom=-2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(results: dict, model_key: str, cfg: dict, out_path: Path) -> None:
    """Ablation figure: Proposed vs variants with components removed."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    baseline_cf = results.get("_baseline_cf", 0)

    ablation_methods = ["ilp-v3", "ilp", "ilp-static", "ilp-v2", "ranger"]
    ablation_labels = {
        "ilp-v3": "Proposed (full)",
        "ilp": "w/o coverage floor",
        "ilp-static": "w/o calibration data",
        "ilp-v2": "Two-tier (CF-node focused)",
        "ranger": "Ranger [Chen 2021]",
    }

    for method in ablation_methods:
        if method not in results:
            continue
        data = sorted(results[method])
        budgets = [d[0] for d in data]
        cfs = [d[1] for d in data]
        lw = 3.0 if method == "ilp-v3" else 1.8
        ms = 10 if method == "ilp-v3" else 7
        ls = "--" if method == "ranger" else "-"
        ax.plot(budgets, cfs,
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                label=ablation_labels[method], linewidth=lw, markersize=ms,
                linestyle=ls)

    ax.axhline(y=baseline_cf, color="gray", linestyle="--", alpha=0.5,
               label=f"Unprotected (CF={baseline_cf})")
    ax.set_xlabel("Latency budget (ms)", fontsize=12)
    ax.set_ylabel("Critical failures (5000 events)", fontsize=12)
    ax.set_title(f"{cfg['label']} — Ablation Study", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(cfg["budgets"])
    ax.set_ylim(bottom=-2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model(all_results: dict, out_path: Path) -> None:
    """2x2 figure: top row = main comparison, bottom row = ablation."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    main_methods = ["ilp-v3", "ranger", "filr", "aspis", "ruospo", "ahmadilivani", "uniform-dmr"]
    ablation_methods = ["ilp-v3", "ilp", "ilp-static", "ilp-v2", "ranger"]
    ablation_labels = {
        "ilp-v3": "Proposed (full)",
        "ilp": "w/o coverage floor",
        "ilp-static": "w/o calibration data",
        "ilp-v2": "Two-tier (CF-node focused)",
        "ranger": "Ranger [Chen 2021]",
    }

    for col, (model_key, cfg) in enumerate(MODELS.items()):
        results = all_results[model_key]
        baseline_cf = results.get("_baseline_cf", 0)

        # Top: main comparison
        ax = axes[0, col]
        for method in main_methods:
            if method not in results:
                continue
            data = sorted(results[method])
            lw = 3.0 if method == "ilp-v3" else 1.3
            ms = 9 if method == "ilp-v3" else 5
            zorder = 10 if method == "ilp-v3" else 5
            ax.plot([d[0] for d in data], [d[1] for d in data],
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=lw, markersize=ms,
                    zorder=zorder)
        ax.axhline(y=baseline_cf, color="gray", linestyle="--", alpha=0.5,
                   label=f"Unprotected (CF={baseline_cf})")
        ax.set_xlabel("Latency budget (ms)", fontsize=11)
        ax.set_ylabel("Critical failures", fontsize=11)
        ax.set_title(f"{cfg['label']}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks(cfg["budgets"])
        ax.set_ylim(bottom=-2)
        ax.grid(alpha=0.3)

        # Bottom: ablation
        ax = axes[1, col]
        for method in ablation_methods:
            if method not in results:
                continue
            data = sorted(results[method])
            lw = 3.0 if method == "ilp-v3" else 1.8
            ms = 10 if method == "ilp-v3" else 7
            ls = "--" if method == "ranger" else "-"
            ax.plot([d[0] for d in data], [d[1] for d in data],
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=ablation_labels[method], linewidth=lw, markersize=ms,
                    linestyle=ls)
        ax.axhline(y=baseline_cf, color="gray", linestyle="--", alpha=0.5,
                   label=f"Unprotected (CF={baseline_cf})")
        ax.set_xlabel("Latency budget (ms)", fontsize=11)
        ax.set_ylabel("Critical failures", fontsize=11)
        ax.set_title(f"{cfg['label']} — Ablation", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(cfg["budgets"])
        ax.set_ylim(bottom=-2)
        ax.grid(alpha=0.3)

    fig.tight_layout(h_pad=3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path("/Users/tzp/work/c/SPINNV2")
    all_results = load_all_results(root)

    for model_key, cfg in MODELS.items():
        results = all_results[model_key]
        fig_dir = root / f"artifacts/experiments/macos_reliability/{model_key if model_key != 'yolov10n' else 'yolov10n_dior'}/figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_cf_comparison(results, model_key, cfg,
                          fig_dir / f"{model_key}_main_comparison.png")
        plot_ablation(results, model_key, cfg,
                     fig_dir / f"{model_key}_ablation.png")
        print(f"{model_key}: main_comparison.png + ablation.png")

    cross_dir = root / "artifacts/experiments/macos_reliability/figures"
    cross_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_model(all_results, cross_dir / "cross_model_full.png")
    print("cross_model_full.png")


if __name__ == "__main__":
    main()
