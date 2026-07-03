"""Cross-method, cross-calibration MC result analysis and figure generation.

Reads mc1000_{cal}_{method}_{budget}ms.json files and produces:
  1. Summary CSV with all results
  2. Budget sweep curves (CF rate vs budget, one per model)
  3. Calibration ablation curves (ILP only, across 3 strategies)
  4. Protection node heatmaps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def _wilson_upper(cf: int, events: int, z: float = 1.96) -> float:
    if events == 0:
        return 1.0
    p_hat = cf / events
    denom = 1 + z * z / events
    centre = p_hat + z * z / (2 * events)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / events + z * z / (4 * events * events))
    return min(1.0, (centre + spread) / denom)


METHODS = ["ilp", "ilp-static", "ilp-v2", "ranger", "filr", "aspis", "ruospo", "ahmadilivani", "uniform-dmr"]
CALIBRATIONS = ["cal", "obs", "obsfloor"]
BUDGETS = [0.5, 1.0, 2.0, 4.0, 8.0]

METHOD_LABELS = {
    "ilp": "ILP (ours)",
    "ilp-static": "ILP-static (Taylor proxy)",
    "ilp-v2": "ILP-v2 (two-tier)",
    "ranger": "Ranger",
    "filr": "FILR",
    "aspis": "ASPIS",
    "ruospo": "Ruospo",
    "ahmadilivani": "Ahmadilivani",
    "uniform-dmr": "Uniform-DMR",
}

CAL_LABELS = {
    "cal": "Wilson",
    "obs": "Observed",
    "obsfloor": "ObsFloor",
}

METHOD_COLORS = {
    "ilp": "#E15759",
    "ilp-static": "#FF9DA7",
    "ilp-v2": "#D4380D",
    "ranger": "#4E79A7",
    "filr": "#59A14F",
    "aspis": "#F28E2B",
    "ruospo": "#B07AA1",
    "ahmadilivani": "#76B7B2",
    "uniform-dmr": "#9C755F",
}

METHOD_MARKERS = {
    "ilp": "o",
    "ilp-static": "h",
    "ilp-v2": "*",
    "ranger": "s",
    "filr": "^",
    "aspis": "D",
    "ruospo": "v",
    "ahmadilivani": "P",
    "uniform-dmr": "X",
}


def load_results(mc_dir: Path) -> list[dict]:
    rows = []
    for cal in CALIBRATIONS:
        for method in METHODS:
            for budget in BUDGETS:
                fname = "mc1000_{}_{}_{}ms.json".format(cal, method, budget)
                fpath = mc_dir / fname
                if not fpath.exists():
                    continue
                d = json.loads(fpath.read_text(encoding="utf-8"))
                t = d.get("totals", {})
                r = d.get("rates", {})
                rows.append({
                    "calibration": cal,
                    "method": method,
                    "budget_ms": budget,
                    "events": t.get("events", 0),
                    "unprotected_cf": t.get("unprotected_critical_failures", 0),
                    "protected_cf": t.get("protected_critical_failures", 0),
                    "mitigated_cf": t.get("mitigated_critical_failures", 0),
                    "new_failures": t.get("new_protected_failures", 0),
                    "detected": t.get("detected_faults", 0),
                    "recovered": t.get("recovered_faults", 0),
                    "cf_rate": r.get("protected_critical_failure_rate", 0.0),
                    "risk_reduction": r.get("observed_risk_reduction_ratio", 0.0),
                })
    return rows


def load_cf_nodes(mc_dir: Path) -> set[int]:
    for f in sorted(mc_dir.glob("mc1000_cal_ranger_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cf_nodes = set()
        for obs in d.get("observations", []):
            if obs.get("unprotected_critical_failure"):
                cf_nodes.add(obs["node_id"])
        if cf_nodes:
            return cf_nodes
    return set()


def load_plan_nodes(plan_path: Path) -> list[dict]:
    if not plan_path.exists():
        return []
    d = json.loads(plan_path.read_text(encoding="utf-8"))
    return d.get("nodes", [])


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    import csv
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_budget_sweep(rows: list[dict], cal: str, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = [r for r in rows if r["calibration"] == cal]
    if not subset:
        plt.close(fig)
        return

    baseline_cf = subset[0]["unprotected_cf"]

    for method in METHODS:
        mdata = [r for r in subset if r["method"] == method]
        if not mdata:
            continue
        mdata.sort(key=lambda x: x["budget_ms"])
        budgets = [r["budget_ms"] for r in mdata]
        cf_rates = [r["protected_cf"] / r["events"] * 100 for r in mdata]
        ax.plot(budgets, cf_rates,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2, markersize=7)

    baseline_rate = baseline_cf / subset[0]["events"] * 100
    ax.axhline(y=baseline_rate, color="gray", linestyle="--", alpha=0.5,
               label="Unprotected ({:.1f}%)".format(baseline_rate))

    ax.set_xlabel("Latency budget (ms)", fontsize=12)
    ax.set_ylabel("Critical failure rate (%)", fontsize=12)
    ax.set_title("{} — {} calibration".format(model_name, CAL_LABELS[cal]), fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(BUDGETS)
    ax.set_ylim(bottom=-0.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_ablation(rows: list[dict], model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    cal_colors = {"cal": "#E15759", "obs": "#59A14F", "obsfloor": "#4E79A7"}
    cal_markers = {"cal": "o", "obs": "s", "obsfloor": "^"}

    for cal in CALIBRATIONS:
        mdata = [r for r in rows if r["calibration"] == cal and r["method"] == "ilp"]
        if not mdata:
            continue
        mdata.sort(key=lambda x: x["budget_ms"])
        budgets = [r["budget_ms"] for r in mdata]
        pro_cf = [r["protected_cf"] for r in mdata]
        ax.plot(budgets, pro_cf,
                marker=cal_markers[cal], color=cal_colors[cal],
                label="ILP — {}".format(CAL_LABELS[cal]),
                linewidth=2.5, markersize=8)

    ranger = [r for r in rows if r["calibration"] == "cal" and r["method"] == "ranger"]
    if ranger:
        ranger.sort(key=lambda x: x["budget_ms"])
        ax.plot([r["budget_ms"] for r in ranger],
                [r["protected_cf"] for r in ranger],
                marker="s", color="#999999", linestyle="--",
                label="Ranger (reference)", linewidth=1.5, markersize=6)

    ax.set_xlabel("Latency budget (ms)", fontsize=12)
    ax.set_ylabel("Protected critical failures", fontsize=12)
    ax.set_title("{} — ILP calibration ablation".format(model_name), fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(BUDGETS)
    ax.set_ylim(bottom=-0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_node_heatmap(plans_dir: Path, budget: float, cal: str,
                      cf_nodes: set[int], model_name: str, out_path: Path,
                      max_node: int | None = None) -> None:
    node_data = {}
    for method in METHODS:
        plan_file = plans_dir / "{}_{}_{:.1f}ms.json".format(method, cal, budget)
        nodes = load_plan_nodes(plan_file)
        node_data[method] = {}
        for n in nodes:
            mode = "RG" if n.get("mode") == "range_guard_rerun" else "DMR"
            node_data[method][n["node_id"]] = mode

    all_nodes = set()
    for nd in node_data.values():
        all_nodes.update(nd.keys())
    all_nodes.update(cf_nodes)
    if not all_nodes:
        return

    if max_node is None:
        max_node = max(all_nodes)
    node_ids = sorted(n for n in all_nodes if n <= max_node)

    matrix = np.zeros((len(METHODS), len(node_ids)))
    for i, method in enumerate(METHODS):
        for j, nid in enumerate(node_ids):
            if nid in node_data.get(method, {}):
                mode = node_data[method][nid]
                matrix[i, j] = 1 if mode == "RG" else 2
            else:
                matrix[i, j] = 0

    cmap = mcolors.ListedColormap(["#F0F0F0", "#4E79A7", "#E15759"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_width = max(10, len(node_ids) * 0.2)
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)

    step = max(1, len(node_ids) // 30)
    xtick_pos = list(range(0, len(node_ids), step))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([str(node_ids[p]) for p in xtick_pos], fontsize=7, rotation=90)

    for j, nid in enumerate(node_ids):
        if nid in cf_nodes:
            ax.axvline(x=j, color="red", linewidth=0.3, alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F0F0F0", edgecolor="gray", label="Unprotected"),
        Patch(facecolor="#4E79A7", label="Range Guard"),
        Patch(facecolor="#E15759", label="DMR"),
        Patch(facecolor="white", edgecolor="red", linewidth=1, label="CF node (validation)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.0, -0.15), ncol=4)

    ax.set_xlabel("Node ID", fontsize=11)
    ax.set_title("{} — {:.1f}ms budget, {} calibration".format(
        model_name, budget, CAL_LABELS[cal]), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_sweep_and_ablation(
    rows_r50: list[dict], rows_yolo: list[dict], out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = [
        (axes[0, 0], rows_r50, "obsfloor", "ResNet50 EuroSAT (ObsFloor)"),
        (axes[0, 1], rows_yolo, "obsfloor", "YOLOv10n DIOR (ObsFloor)"),
    ]
    for ax, rows, cal, title in configs:
        subset = [r for r in rows if r["calibration"] == cal]
        if not subset:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        baseline_rate = subset[0]["unprotected_cf"] / subset[0]["events"] * 100
        for method in METHODS:
            mdata = sorted([r for r in subset if r["method"] == method],
                           key=lambda x: x["budget_ms"])
            if not mdata:
                continue
            ax.plot([r["budget_ms"] for r in mdata],
                    [r["protected_cf"] / r["events"] * 100 for r in mdata],
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=2, markersize=7)
        ax.axhline(y=baseline_rate, color="gray", linestyle="--", alpha=0.5,
                   label="Unprotected ({:.1f}%)".format(baseline_rate))
        ax.set_xlabel("Latency budget (ms)", fontsize=11)
        ax.set_ylabel("Critical failure rate (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(BUDGETS)
        ax.set_ylim(bottom=-0.1)
        ax.grid(alpha=0.3)

    cal_colors = {"cal": "#E15759", "obs": "#59A14F", "obsfloor": "#4E79A7"}
    cal_markers = {"cal": "o", "obs": "s", "obsfloor": "^"}
    ablation_configs = [
        (axes[1, 0], rows_r50, "ResNet50 — ILP calibration ablation"),
        (axes[1, 1], rows_yolo, "YOLOv10n — ILP calibration ablation"),
    ]
    for ax, rows, title in ablation_configs:
        for cal in CALIBRATIONS:
            mdata = sorted([r for r in rows if r["calibration"] == cal and r["method"] == "ilp"],
                           key=lambda x: x["budget_ms"])
            if not mdata:
                continue
            ax.plot([r["budget_ms"] for r in mdata],
                    [r["protected_cf"] for r in mdata],
                    marker=cal_markers[cal], color=cal_colors[cal],
                    label="ILP — {}".format(CAL_LABELS[cal]),
                    linewidth=2.5, markersize=8)
        ranger = sorted([r for r in rows if r["calibration"] == "cal" and r["method"] == "ranger"],
                        key=lambda x: x["budget_ms"])
        if ranger:
            ax.plot([r["budget_ms"] for r in ranger],
                    [r["protected_cf"] for r in ranger],
                    marker="s", color="#999999", linestyle="--",
                    label="Ranger (ref)", linewidth=1.5, markersize=6)
        ax.set_xlabel("Latency budget (ms)", fontsize=11)
        ax.set_ylabel("Protected critical failures", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xticks(BUDGETS)
        ax.set_ylim(bottom=-0.5)
        ax.grid(alpha=0.3)

    fig.tight_layout(h_pad=3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_accuracy(
    metadata_dir: Path,
    mc_dir: Path,
    model_name: str,
    out_path: Path,
) -> None:
    """Scatter: predicted CP vs actual validation CF rate, per smoothing strategy."""
    meta_obs = metadata_dir / "calibration_metadata_obs.json"
    meta_obsfloor = metadata_dir / "calibration_metadata_obsfloor.json"
    if not meta_obs.exists() or not meta_obsfloor.exists():
        return

    obs_data = json.loads(meta_obs.read_text(encoding="utf-8"))
    obsfloor_data = json.loads(meta_obsfloor.read_text(encoding="utf-8"))
    per_node_raw = obs_data["per_node"]

    # Actual validation CF rates: aggregate from seed=2027 ranger MC (highest budget)
    actual_cf: dict[int, int] = {}
    actual_events: dict[int, int] = {}
    for fname in sorted(mc_dir.glob("mc1000_cal_ranger_*.json")):
        d = json.loads(fname.read_text(encoding="utf-8"))
        for obs in d.get("observations", []):
            nid = obs["node_id"]
            actual_events[nid] = actual_events.get(nid, 0) + 1
            if obs["unprotected_critical_failure"]:
                actual_cf[nid] = actual_cf.get(nid, 0) + 1
        break  # just use highest budget file (last in sorted order is smallest, use max budget)
    # re-read to get highest budget
    actual_cf = {}
    actual_events = {}
    ranger_files = sorted(mc_dir.glob("mc1000_cal_ranger_*.json"),
                          key=lambda p: float(p.stem.split("_")[-1].rstrip("ms")))
    if ranger_files:
        d = json.loads(ranger_files[-1].read_text(encoding="utf-8"))
        for obs in d.get("observations", []):
            nid = obs["node_id"]
            actual_events[nid] = actual_events.get(nid, 0) + 1
            if obs["unprotected_critical_failure"]:
                actual_cf[nid] = actual_cf.get(nid, 0) + 1

    node_ids = [int(n) for n in per_node_raw]

    # Build per-strategy predicted CP
    strategies = {
        "Wilson": {},
        "Observed": {},
        "ObsFloor": {},
    }
    obs_per_node = obs_data["per_node"]
    obsfloor_per_node = obsfloor_data["per_node"]
    for nid_str, nd in per_node_raw.items():
        nid = int(nid_str)
        ev = nd["events"]
        cf = nd["critical_failures"]
        strategies["Wilson"][nid] = _wilson_upper(cf, ev)
        strategies["Observed"][nid] = obs_per_node[nid_str]["calibrated_critical_probability"]
        strategies["ObsFloor"][nid] = obsfloor_per_node[nid_str]["calibrated_critical_probability"]

    colors_strat = {"Wilson": "#E15759", "Observed": "#59A14F", "ObsFloor": "#4E79A7"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("{} — Risk Model Calibration Accuracy".format(model_name), fontsize=14, fontweight="bold")

    for ax, (strat_name, pred_cp_map) in zip(axes, strategies.items()):
        xs, ys, colors = [], [], []
        for nid in sorted(pred_cp_map):
            if nid not in actual_events:
                continue
            actual_rate = actual_cf.get(nid, 0) / actual_events[nid]
            pred = pred_cp_map[nid]
            xs.append(actual_rate)
            ys.append(pred)
            has_cf = actual_cf.get(nid, 0) > 0
            colors.append(colors_strat[strat_name] if has_cf else "#AAAAAA")

        ax.scatter(xs, ys, c=colors, s=60, alpha=0.8, zorder=3)
        if xs:
            max_val = max(max(xs), max(ys)) * 1.1
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1, label="y=x (perfect)")
            rmse = (sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5
            ax.text(0.97, 0.05, "RMSE={:.4f}".format(rmse), transform=ax.transAxes,
                    ha="right", fontsize=9, color="black")
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_strat[strat_name],
                   markersize=8, label="CF observed"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#AAAAAA",
                   markersize=8, label="No CF observed"),
            Line2D([0], [0], linestyle="--", color="k", alpha=0.4, label="y=x (perfect)"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
        ax.set_xlabel("Actual CF rate (seed=2027)", fontsize=10)
        ax.set_ylabel("Predicted CP", fontsize=10)
        ax.set_title(strat_name, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name (resnet50, yolov10n_dior)")
    parser.add_argument("--mc-dir", required=True, help="MC results directory")
    parser.add_argument("--plans-dir", required=True, help="Plans directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures and CSV")
    parser.add_argument("--model-label", help="Human-readable model name for titles")
    parser.add_argument("--max-heatmap-node", type=int, help="Limit heatmap to this max node_id")
    parser.add_argument("--combined-mc-dir", help="Second model MC dir for combined figure")
    parser.add_argument("--combined-out", help="Output path for combined 2x2 figure")
    args = parser.parse_args()

    mc_dir = Path(args.mc_dir)
    plans_dir = Path(args.plans_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_label = args.model_label or args.model

    rows = load_results(mc_dir)
    print("Loaded {} MC results".format(len(rows)))

    write_summary_csv(rows, out_dir / "mc_summary_{}.csv".format(args.model))
    print("Wrote summary CSV")

    cf_nodes = load_cf_nodes(mc_dir)
    print("Validation CF nodes: {} ({})".format(sorted(cf_nodes), len(cf_nodes)))

    for cal in CALIBRATIONS:
        cal_rows = [r for r in rows if r["calibration"] == cal]
        if not cal_rows:
            continue
        plot_budget_sweep(rows, cal, model_label, out_dir / "sweep_{}_{}.png".format(args.model, cal))
        print("Plotted budget sweep: {}".format(cal))

    plot_calibration_ablation(rows, model_label, out_dir / "ablation_{}.png".format(args.model))
    print("Plotted calibration ablation")

    for cal in ["obsfloor"]:
        for budget in [2.0, 4.0]:
            plot_node_heatmap(
                plans_dir, budget, cal, cf_nodes, model_label,
                out_dir / "heatmap_{}_{}_{:.0f}ms.png".format(args.model, cal, budget),
                max_node=args.max_heatmap_node,
            )
            print("Plotted heatmap: {} {:.0f}ms".format(cal, budget))

    if args.combined_mc_dir and args.combined_out:
        rows2 = load_results(Path(args.combined_mc_dir))
        # rows2 = ResNet50, rows = YOLOv10n (combined-mc-dir is always r50 when called from yolo)
        plot_combined_sweep_and_ablation(rows2, rows, Path(args.combined_out))
        print("Plotted combined 2x2 figure")

    plot_calibration_accuracy(
        metadata_dir=Path(args.mc_dir).parent,
        mc_dir=mc_dir,
        model_name=model_label,
        out_path=out_dir / "calibration_accuracy_{}.png".format(args.model),
    )
    print("Plotted calibration accuracy")

    print("Done. {} figures in {}".format(
        len(list(out_dir.glob("*.png"))), out_dir))


if __name__ == "__main__":
    main()
