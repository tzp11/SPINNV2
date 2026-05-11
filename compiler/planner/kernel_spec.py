"""KernelSpec selection and scratch estimation for M4."""

from __future__ import annotations

from dataclasses import dataclass

from compiler.ir.graph import Graph, Node
from compiler.ir import types


BACKEND_REF = "ref"
BACKEND_CPU = "cpu"

KIND_REFERENCE = "reference"
KIND_DIRECT = "direct"
KIND_IM2COL_GEMM = "im2col_gemm"

DTYPE_FP32 = "fp32"
LAYOUT_NCHW = "NCHW"
WEIGHT_LAYOUT_OIHW = "OIHW"


@dataclass
class KernelSpec:
    id: int
    node_id: int
    op_type: str
    kernel_kind: str
    backend: str
    dtype: str = DTYPE_FP32
    layout: str = LAYOUT_NCHW
    weight_layout: str = WEIGHT_LAYOUT_OIHW
    scratch_offset: int = 0
    scratch_bytes: int = 0
    fallback_kernel_spec_id: int = 0xFFFFFFFF
    required_features: list[str] | None = None
    selected_by_fallback: bool = False


@dataclass
class KernelPlan:
    specs: list[KernelSpec]
    by_node: dict[int, KernelSpec]
    scratch_arena_bytes: int
    fallback_count: int


def select_kernel_specs(graph: Graph, target_profile: dict) -> KernelPlan:
    specs: list[KernelSpec] = []
    by_node: dict[int, KernelSpec] = {}
    fallback_count = 0
    scratch_peak = 0

    for node in graph.nodes:
        op_support = target_profile["ops"].get(node.op_type, [])
        if not op_support:
            raise ValueError(f"target profile does not support op: {node.op_type}")

        selected = _select_primary_spec(graph, node, op_support)
        ref_spec = _make_spec(graph, node, KIND_REFERENCE, BACKEND_REF)
        if selected is None:
            if "ref" not in op_support:
                raise ValueError(f"no supported kernel for op: {node.op_type}")
            selected = ref_spec
            selected.selected_by_fallback = True
            fallback_count += 1

        selected.id = len(specs)
        specs.append(selected)
        by_node[node.id] = selected

        if selected.kernel_kind != KIND_REFERENCE and "ref" in op_support:
            fallback = ref_spec
            fallback.id = len(specs)
            selected.fallback_kernel_spec_id = fallback.id
            specs.append(fallback)

        scratch_peak = max(scratch_peak, selected.scratch_bytes)

    scratch_limit = int(target_profile["memory"]["scratch_arena_max"])
    if scratch_peak > scratch_limit:
        raise MemoryError(f"planned scratch arena {scratch_peak} exceeds target limit {scratch_limit}")

    graph.metadata["kernel_specs"] = [_spec_json(spec) for spec in specs]
    graph.metadata["kernel_fallback_count"] = fallback_count
    graph.metadata["scratch_arena_bytes"] = scratch_peak

    return KernelPlan(
        specs=specs,
        by_node=by_node,
        scratch_arena_bytes=scratch_peak,
        fallback_count=fallback_count,
    )


def _select_primary_spec(graph: Graph, node: Node, op_support: list[str]) -> KernelSpec | None:
    if node.op_type == "Conv" and KIND_IM2COL_GEMM in op_support:
        return _make_spec(graph, node, KIND_IM2COL_GEMM, BACKEND_CPU)
    if node.op_type == "Gemm" and KIND_DIRECT in op_support:
        return _make_spec(graph, node, KIND_DIRECT, BACKEND_CPU)
    if KIND_REFERENCE in op_support or "ref" in op_support:
        return _make_spec(graph, node, KIND_REFERENCE, BACKEND_REF)
    return None


def _make_spec(graph: Graph, node: Node, kernel_kind: str, backend: str) -> KernelSpec:
    return KernelSpec(
        id=0,
        node_id=node.id,
        op_type=node.op_type,
        kernel_kind=kernel_kind,
        backend=backend,
        scratch_bytes=_estimate_scratch_bytes(graph, node, kernel_kind),
        required_features=[types.DTYPE_FP32],
    )


def _estimate_scratch_bytes(graph: Graph, node: Node, kernel_kind: str) -> int:
    if node.op_type != "Conv" or kernel_kind != KIND_IM2COL_GEMM:
        return 0
    x = graph.tensors[node.inputs[0]]
    w = graph.tensors[node.inputs[1]]
    if len(x.shape) != 4 or len(w.shape) != 4:
        return 0
    channels = x.shape[1]
    kernel_h = w.shape[2]
    kernel_w = w.shape[3]
    return _align(channels * kernel_h * kernel_w * 4, 16)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _spec_json(spec: KernelSpec) -> dict:
    return {
        "id": spec.id,
        "node_id": spec.node_id,
        "op_type": spec.op_type,
        "kernel_kind": spec.kernel_kind,
        "backend": spec.backend,
        "dtype": spec.dtype,
        "layout": spec.layout,
        "weight_layout": spec.weight_layout,
        "scratch_offset": spec.scratch_offset,
        "scratch_bytes": spec.scratch_bytes,
        "fallback_kernel_spec_id": spec.fallback_kernel_spec_id,
        "required_features": spec.required_features or [],
        "selected_by_fallback": spec.selected_by_fallback,
    }
