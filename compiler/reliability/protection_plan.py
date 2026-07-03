"""Validation and binary encoding for resource-budget protection plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

from compiler.ir.graph import Graph
from compiler.ir import types


PROTECTION_RECORD_STRUCT = struct.Struct("<IIHHffQ")
MODE_CODES = {
    "none": 0,
    "range_guard_rerun": 1,
    "dmr_compare_rerun": 2,
}


@dataclass(frozen=True)
class ProtectionEntry:
    node_id: int
    tensor_id: int
    mode: str
    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    compare_atol: float = 0.0
    compare_rtol: float = 0.0


@dataclass(frozen=True)
class ProtectionPlan:
    version: int
    model_id: str
    nodes: tuple[ProtectionEntry, ...]
    platform_profile: str = ""
    fault_prior: str = ""
    budgets: dict | None = None

    def validated_for_graph(self, graph: Graph) -> "ProtectionPlan":
        seen: set[int] = set()
        for entry in self.nodes:
            if entry.mode not in MODE_CODES:
                raise ValueError(f"unsupported protection mode: {entry.mode}")
            if entry.node_id < 0 or entry.node_id >= len(graph.nodes):
                raise ValueError(f"unknown protection node_id: {entry.node_id}")
            if entry.node_id in seen:
                raise ValueError(f"duplicate protection node_id: {entry.node_id}")
            seen.add(entry.node_id)
            node = graph.nodes[entry.node_id]
            if len(node.outputs) != 1:
                raise ValueError(f"protected node {entry.node_id} must have exactly one output")
            if entry.tensor_id != node.outputs[0]:
                raise ValueError(f"tensor {entry.tensor_id} is not output of node {entry.node_id}")
            tensor = graph.tensor(entry.tensor_id)
            if tensor.dtype != types.DTYPE_FP32:
                raise ValueError(f"protected tensor {entry.tensor_id} must be fp32")
            if entry.mode == "range_guard_rerun" and entry.lower_bound > entry.upper_bound:
                raise ValueError(f"invalid range for node {entry.node_id}")
        return self

    def scratch_bytes(self, graph: Graph) -> int:
        return max(
            (2 * graph.tensor(entry.tensor_id).size_bytes for entry in self.nodes if entry.mode == "dmr_compare_rerun"),
            default=0,
        )

    def to_bytes(self, graph: Graph, *, scratch_offset: int) -> bytes:
        self.validated_for_graph(graph)
        blob = bytearray()
        for entry in self.nodes:
            lower = entry.lower_bound if math.isfinite(entry.lower_bound) else -3.4028235e38
            upper = entry.upper_bound if math.isfinite(entry.upper_bound) else 3.4028235e38
            blob.extend(
                PROTECTION_RECORD_STRUCT.pack(
                    entry.node_id,
                    entry.tensor_id,
                    MODE_CODES[entry.mode],
                    0,
                    lower,
                    upper,
                    scratch_offset if entry.mode == "dmr_compare_rerun" else 0,
                )
            )
        return bytes(blob)

    def as_dict(self) -> dict:
        nodes = []
        for entry in self.nodes:
            item = {
                "node_id": entry.node_id,
                "tensor_id": entry.tensor_id,
                "mode": entry.mode,
            }
            if entry.mode == "range_guard_rerun":
                item["lower_bound"] = entry.lower_bound if math.isfinite(entry.lower_bound) else None
                item["upper_bound"] = entry.upper_bound if math.isfinite(entry.upper_bound) else None
            if entry.mode == "dmr_compare_rerun":
                item["compare_atol"] = entry.compare_atol
                item["compare_rtol"] = entry.compare_rtol
            nodes.append(item)
        return {
            "version": self.version,
            "model_id": self.model_id,
            "platform_profile": self.platform_profile,
            "fault_prior": self.fault_prior,
            "budgets": self.budgets or {},
            "nodes": nodes,
        }


def load_protection_plan(path: str | Path) -> ProtectionPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("ProtectionPlan version must be 1")
    entries = tuple(
        ProtectionEntry(
            node_id=int(item["node_id"]),
            tensor_id=int(item["tensor_id"]),
            mode=str(item["mode"]),
            lower_bound=float(item.get("lower_bound", -math.inf)),
            upper_bound=float(item.get("upper_bound", math.inf)),
            compare_atol=float(item.get("compare_atol", 0.0)),
            compare_rtol=float(item.get("compare_rtol", 0.0)),
        )
        for item in payload.get("nodes", [])
    )
    return ProtectionPlan(
        version=1,
        model_id=str(payload.get("model_id", "")),
        platform_profile=str(payload.get("platform_profile", "")),
        fault_prior=str(payload.get("fault_prior", "")),
        budgets=dict(payload.get("budgets", {})),
        nodes=entries,
    )
