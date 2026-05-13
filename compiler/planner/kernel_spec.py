"""KernelSpec selection and scratch estimation for M4."""

from __future__ import annotations

from dataclasses import dataclass

from compiler.ir.graph import Graph, Node
from compiler.ir import types


BACKEND_REF = "ref"
BACKEND_CPU = "cpu"
BACKEND_SIMD = "simd"

KIND_REFERENCE = "reference"
KIND_DIRECT = "direct"
KIND_IM2COL_GEMM = "im2col_gemm"
KIND_POINTWISE_1X1 = "pointwise_1x1"
KIND_DEPTHWISE_DIRECT = "depthwise_direct"
KIND_WINOGRAD_3X3S1 = "winograd_3x3s1"
KIND_CONV3X3S2_DIRECT = "conv3x3s2_direct"

# Target-profile op-support tokens that map to (backend, kind) pairs
SIMD_IM2COL_GEMM = "simd_im2col_gemm"
SIMD_POINTWISE_1X1 = "simd_pointwise_1x1"
SIMD_DEPTHWISE_DIRECT = "simd_depthwise_direct"
SIMD_WINOGRAD_3X3S1 = "simd_winograd_3x3s1"
SIMD_CONV3X3S2_DIRECT = "simd_conv3x3s2_direct"
SIMD_DIRECT = "simd_direct"
SIMD_REF = "simd"

# Max im2col buffer used by the SIMD conv kernel (32 MiB)
_SIMD_TILE_BYTES = 32 * 1024 * 1024

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
    # SIMD paths (preferred when target advertises them)
    if node.op_type == "Conv":
        conv_kind = _select_simd_conv_kind(graph, node, op_support)
        if conv_kind is not None:
            return _make_spec(graph, node, conv_kind, BACKEND_SIMD)
    if node.op_type in {"Gemm", "MatMul"} and SIMD_DIRECT in op_support:
        return _make_spec(graph, node, KIND_DIRECT, BACKEND_SIMD)
    if SIMD_REF in op_support and node.op_type in {
        "Add", "Mul", "Sub", "Div", "Relu", "Sigmoid", "Transpose",
        "ReduceMax", "ReduceMean", "Softmax",
    }:
        return _make_spec(graph, node, KIND_REFERENCE, BACKEND_SIMD)
    # CPU paths (fallback from SIMD)
    if node.op_type == "Conv" and KIND_IM2COL_GEMM in op_support:
        return _make_spec(graph, node, KIND_IM2COL_GEMM, BACKEND_CPU)
    if node.op_type == "Gemm" and KIND_DIRECT in op_support:
        return _make_spec(graph, node, KIND_DIRECT, BACKEND_CPU)
    if KIND_REFERENCE in op_support or "ref" in op_support:
        return _make_spec(graph, node, KIND_REFERENCE, BACKEND_REF)
    return None


def _select_simd_conv_kind(graph: Graph, node: Node, op_support: list[str]) -> str | None:
    x = graph.tensors[node.inputs[0]] if node.inputs else None
    w = graph.tensors[node.inputs[1]] if len(node.inputs) > 1 else None
    if x is None or w is None or len(x.shape) != 4 or len(w.shape) != 4:
        return KIND_IM2COL_GEMM if SIMD_IM2COL_GEMM in op_support else None

    group = int(node.attrs.get("group", 1))
    strides = _int_list(node.attrs.get("strides", [1, 1]), 2, 1)
    pads = _int_list(node.attrs.get("pads", [0, 0, 0, 0]), 4, 0)
    dilations = _int_list(node.attrs.get("dilations", [1, 1]), 2, 1)
    k_h, k_w = int(w.shape[2]), int(w.shape[3])
    in_c, out_c = int(x.shape[1]), int(w.shape[0])

    if (
        SIMD_POINTWISE_1X1 in op_support
        and group == 1
        and k_h == 1
        and k_w == 1
        and strides == [1, 1]
        and pads == [0, 0, 0, 0]
        and dilations == [1, 1]
    ):
        return KIND_POINTWISE_1X1

    if (
        SIMD_DEPTHWISE_DIRECT in op_support
        and group == in_c
        and group == out_c
        and k_h == 3
        and k_w == 3
        and dilations == [1, 1]
    ):
        return KIND_DEPTHWISE_DIRECT

    # Winograd F(2,3) is implemented in the runtime for controlled testing, but
    # not selected by default yet. Current A/B on YOLOv10n and ResNet101 shows
    # it is slower than the existing packed im2col+SGEMM path because transform
    # and per-alpha GEMM overhead dominate.

    # Keep the stride-2 direct kind in the format/runtime for experiments, but
    # do not select it by default. The gather-heavy direct path is slower than
    # the existing im2col+SGEMM path on YOLO downsample layers.

    if SIMD_IM2COL_GEMM in op_support:
        return KIND_IM2COL_GEMM
    return None


def _make_spec(graph: Graph, node: Node, kernel_kind: str, backend: str) -> KernelSpec:
    return KernelSpec(
        id=0,
        node_id=node.id,
        op_type=node.op_type,
        kernel_kind=kernel_kind,
        backend=backend,
        scratch_bytes=_estimate_scratch_bytes(graph, node, kernel_kind, backend),
        required_features=[types.DTYPE_FP32],
    )


def _estimate_scratch_bytes(graph: Graph, node: Node, kernel_kind: str, backend: str = BACKEND_REF) -> int:
    if node.op_type != "Conv":
        return 0
    if kernel_kind in {KIND_POINTWISE_1X1, KIND_DEPTHWISE_DIRECT, KIND_CONV3X3S2_DIRECT}:
        return 0
    if kernel_kind == KIND_WINOGRAD_3X3S1:
        x = graph.tensors[node.inputs[0]]
        w = graph.tensors[node.inputs[1]]
        y = graph.tensors[node.outputs[0]]
        if len(x.shape) != 4 or len(w.shape) != 4 or len(y.shape) != 4:
            return 0
        tile_count = ((y.shape[2] + 1) // 2) * ((y.shape[3] + 1) // 2)
        channels = x.shape[1]
        out_c = w.shape[0]
        return _align(min(16 * (channels + out_c) * tile_count * 4, _SIMD_TILE_BYTES), 16)
    if kernel_kind != KIND_IM2COL_GEMM:
        return 0
    x = graph.tensors[node.inputs[0]]
    w = graph.tensors[node.inputs[1]]
    if len(x.shape) != 4 or len(w.shape) != 4:
        return 0
    channels = x.shape[1]
    kernel_h = w.shape[2]
    kernel_w = w.shape[3]
    K = channels * kernel_h * kernel_w

    if backend == BACKEND_SIMD:
        # SIMD path needs a tile of the full im2col matrix: K × tile_n floats
        y = graph.tensors[node.outputs[0]]
        spatial = y.shape[2] * y.shape[3] if len(y.shape) == 4 else 1
        full_bytes = K * spatial * 4
        return _align(min(full_bytes, _SIMD_TILE_BYTES), 16)

    # Original CPU im2col: one column vector
    return _align(K * 4, 16)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _int_list(value, length: int, default: int) -> list[int]:
    result = [int(v) for v in value]
    return (result + [default] * length)[:length]


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
