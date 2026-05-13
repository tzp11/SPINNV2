"""M3 graph optimization pass pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from compiler.ir.graph import Graph
from compiler.passes import transforms


@dataclass
class PassResult:
    name: str
    changed: int
    nodes_before: int
    nodes_after: int
    tensors_before: int
    tensors_after: int


PassFn = Callable[[Graph], int]

AVAILABLE_PASSES: dict[str, PassFn] = {
    "EliminateIdentityDropout": transforms.eliminate_identity_dropout,
    "ConstantFold": transforms.constant_fold,
    "FuseConvBatchNorm": transforms.fuse_conv_batchnorm,
    "FuseConvRelu": transforms.fuse_conv_relu,
    "FuseConvSilu": transforms.fuse_conv_silu,
    "FuseAddRelu": transforms.fuse_add_relu,
    "EliminateDead": transforms.eliminate_dead,
}

DEFAULT_PIPELINE = [
    "EliminateIdentityDropout",
    "ConstantFold",
    "FuseConvBatchNorm",
    "FuseConvRelu",
    "FuseConvSilu",
    "FuseAddRelu",
    "EliminateDead",
]


def run_default_pass_pipeline(graph: Graph, *, enabled: bool = True) -> list[PassResult]:
    return run_pass_pipeline(graph, enabled=enabled)


def run_pass_pipeline(
    graph: Graph,
    *,
    pipeline: list[str] | None = None,
    enabled: bool = True,
) -> list[PassResult]:
    if not enabled:
        graph.metadata["pass_stats"] = []
        return []

    pass_names = pipeline or DEFAULT_PIPELINE

    results: list[PassResult] = []
    for name in pass_names:
        if name not in AVAILABLE_PASSES:
            raise ValueError(f"unknown pass: {name}")
        fn = AVAILABLE_PASSES[name]
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)
        changed = fn(graph)
        transforms.rebuild_graph_links(graph)
        nodes_after = len(graph.nodes)
        tensors_after = len(graph.tensors)
        results.append(
            PassResult(
                name=name,
                changed=changed,
                nodes_before=nodes_before,
                nodes_after=nodes_after,
                tensors_before=tensors_before,
                tensors_after=tensors_after,
            )
        )

    graph.metadata["pass_stats"] = [result.__dict__ for result in results]
    return results


def write_pass_stats_json(results: list[PassResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
