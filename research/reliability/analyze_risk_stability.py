"""Bootstrap risk-ranking stability from existing fault observations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_observations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "observations" in doc:
        return doc["observations"]
    raise ValueError(f"Unsupported observation format: {path}")


def _load_activation_map(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    result: dict[int, int] = {}
    for row in table.itertuples(index=False):
        node_value = getattr(row, "runtime_node_id", getattr(row, "node_id", -1))
        if pd.isna(node_value):
            continue
        node = int(node_value)
        if node < 0:
            continue
        result[node] = int(getattr(row, "activation_bytes", 4))
    return result


def _normalize(records: list[dict[str, Any]], activation_map: dict[int, int]) -> list[dict[str, Any]]:
    normalized = []
    for item in records:
        node_id = int(item["node_id"])
        tensor_id = int(item.get("tensor_id", -1))
        critical = bool(item.get("critical_failure", item.get("unprotected_critical_failure", False)))
        severity = float(item.get("severity", 1.0 if critical else 0.0))
        activation = int(item.get("activation_bytes", activation_map.get(node_id, 4)))
        normalized.append(
            {
                "node_id": node_id,
                "tensor_id": tensor_id,
                "critical_failure": critical,
                "severity": severity,
                "activation_bytes": max(activation, 4),
            }
        )
    return normalized


def _risk_table(records: list[dict[str, Any]], weights: tuple[float, float, float]) -> pd.DataFrame:
    critical_w, severity_w, exposure_w = weights
    grouped: dict[int, dict[str, Any]] = {}
    total_activation = sum({(r["node_id"], r["activation_bytes"]) for r in records}, start=()) if False else None
    activation_by_node: dict[int, int] = {}
    for r in records:
        activation_by_node[r["node_id"]] = max(activation_by_node.get(r["node_id"], 0), r["activation_bytes"])
    total_activation_bytes = sum(activation_by_node.values()) or 1
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["node_id"]].append(record)
    rows = []
    for node_id, items in buckets.items():
        count = len(items)
        failures = sum(int(item["critical_failure"]) for item in items)
        mean_severity = float(np.mean([item["severity"] for item in items])) if items else 0.0
        exposure = activation_by_node[node_id] / total_activation_bytes
        risk = critical_w * (failures / count) + severity_w * mean_severity + exposure_w * exposure
        rows.append(
            {
                "node_id": node_id,
                "observations": count,
                "critical_failures": failures,
                "mean_severity": mean_severity,
                "activation_bytes": activation_by_node[node_id],
                "exposure": exposure,
                "risk": risk,
            }
        )
    return pd.DataFrame(rows).sort_values("risk", ascending=False).reset_index(drop=True)


def _rank_map(table: pd.DataFrame) -> dict[int, int]:
    return {int(row.node_id): idx for idx, row in enumerate(table.itertuples(index=False), start=1)}


def _spearman(a: dict[int, int], b: dict[int, int]) -> float:
    nodes = sorted(set(a) & set(b))
    if len(nodes) < 2:
        return 1.0
    av = np.array([a[n] for n in nodes], dtype=np.float64)
    bv = np.array([b[n] for n in nodes], dtype=np.float64)
    if av.std() == 0 or bv.std() == 0:
        return 1.0
    return float(np.corrcoef(av, bv)[0, 1])


def analyze(
    observations: list[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    seed: int,
    topk_values: list[int],
    weights: tuple[float, float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    full = _risk_table(observations, weights)
    full_rank = _rank_map(full)
    full_top = {k: set(full.head(k)["node_id"].astype(int).tolist()) for k in topk_values}
    rows = []
    n = len(observations)
    for repeat in range(bootstrap_repeats):
        sample_indices = rng.integers(0, n, size=n)
        sample = [observations[int(i)] for i in sample_indices]
        table = _risk_table(sample, weights)
        rank = _rank_map(table)
        row: dict[str, Any] = {
            "repeat": repeat,
            "spearman_vs_full": _spearman(full_rank, rank),
            "highest_risk_node": int(table.iloc[0]["node_id"]) if not table.empty else None,
        }
        for k in topk_values:
            sample_top = set(table.head(k)["node_id"].astype(int).tolist())
            denom = len(full_top[k] | sample_top) or 1
            row[f"top{k}_jaccard_vs_full"] = len(full_top[k] & sample_top) / denom
            row[f"top{k}_overlap_vs_full"] = len(full_top[k] & sample_top) / max(len(full_top[k]), 1)
        rows.append(row)
    detail = pd.DataFrame(rows)
    summary = {
        "observations": n,
        "nodes": int(full["node_id"].nunique()),
        "bootstrap_repeats": bootstrap_repeats,
        "seed": seed,
        "weights": {
            "critical": weights[0],
            "severity": weights[1],
            "exposure": weights[2],
        },
        "full_top_nodes": {f"top{k}": [int(x) for x in full.head(k)["node_id"].tolist()] for k in topk_values},
        "metrics": {
            column: {
                "mean": float(detail[column].mean()),
                "std": float(detail[column].std(ddof=0)),
                "min": float(detail[column].min()),
                "max": float(detail[column].max()),
            }
            for column in detail.columns
            if column != "repeat" and pd.api.types.is_numeric_dtype(detail[column])
        },
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--activation-map-csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--topk", default="5,10,20")
    parser.add_argument("--critical-weight", type=float, default=0.65)
    parser.add_argument("--severity-weight", type=float, default=0.25)
    parser.add_argument("--exposure-weight", type=float, default=0.10)
    args = parser.parse_args()

    activation_map = _load_activation_map(Path(args.activation_map_csv) if args.activation_map_csv else None)
    observations = _normalize(_load_observations(Path(args.observations)), activation_map)
    topk = [int(value) for value in args.topk.split(",") if value.strip()]
    detail, summary = analyze(
        observations,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
        topk_values=topk,
        weights=(args.critical_weight, args.severity_weight, args.exposure_weight),
    )
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(out_csv), "json": str(out_json), "summary": summary["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
