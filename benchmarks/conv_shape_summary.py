#!/usr/bin/env python3
"""Summarize Conv shapes and selected kernel kinds from a SPK debug JSON."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spk_json", type=Path, help="Path to model.spk.json debug file")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    data = json.loads(args.spk_json.read_text(encoding="utf-8"))
    tensors = {t["id"]: t for t in data.get("tensors", [])}
    specs = {}
    for spec in data.get("metadata", {}).get("kernel_specs", []):
        if spec.get("node_id") not in specs:
            specs[spec["node_id"]] = spec

    rows: list[tuple[str, str]] = []
    categories: collections.Counter[str] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    for node in data.get("nodes", []):
        if node.get("op_type") != "Conv":
            continue
        x = tensors[node["inputs"][0]]
        w = tensors[node["inputs"][1]]
        y = tensors[node["outputs"][0]]
        attrs = node.get("attrs", {})
        strides = tuple(attrs.get("strides", [1, 1]))
        pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(attrs.get("dilations", [1, 1]))
        group = int(attrs.get("group", 1))
        k = tuple(w["shape"][2:4])
        spatial = y["shape"][2] * y["shape"][3] if len(y["shape"]) == 4 else 1
        kind = specs.get(node["id"], {}).get("kernel_kind", "unknown")
        key = (
            f"k={k},s={strides},p={pads},d={dilations},g={group},"
            f"ic={x['shape'][1]},oc={w['shape'][0]},sp={spatial}"
        )
        rows.append((kind, key))
        kinds[kind] += 1

        if group == x["shape"][1] == w["shape"][0]:
            categories["depthwise"] += 1
        elif k == (1, 1) and strides == (1, 1) and pads == (0, 0, 0, 0) and group == 1:
            categories["pointwise_1x1"] += 1
        elif k == (3, 3) and strides == (1, 1) and pads == (1, 1, 1, 1) and group == 1:
            categories["std_3x3_s1p1"] += 1
        elif k == (3, 3) and strides == (2, 2) and pads == (1, 1, 1, 1) and group == 1:
            categories["std_3x3_s2p1"] += 1
        else:
            categories["other"] += 1

    print(f"model={data.get('model_name')} convs={len(rows)}")
    print("kernel_kinds:")
    for kind, count in kinds.most_common():
        print(f"  {kind}: {count}")
    print("categories:")
    for category, count in categories.most_common():
        print(f"  {category}: {count}")
    print("top_shapes:")
    for (kind, key), count in collections.Counter(rows).most_common(args.top):
        print(f"  {count:4d} {kind:18s} {key}")


if __name__ == "__main__":
    main()
