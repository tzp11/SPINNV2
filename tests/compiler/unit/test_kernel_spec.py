from __future__ import annotations

import pytest

from compiler.ir import types
from compiler.ir.graph import Graph, Node, Tensor
from compiler.planner.kernel_spec import select_kernel_specs
from compiler.target.profile import load_target_profile


def test_cpu_ref_selects_reference_kernels_only():
    graph = _conv_gemm_graph()
    plan = select_kernel_specs(graph, load_target_profile("cpu_ref"))

    selected = [plan.by_node[node.id] for node in graph.nodes]
    assert [spec.backend for spec in selected] == ["ref", "ref"]
    assert [spec.kernel_kind for spec in selected] == ["reference", "reference"]
    assert plan.scratch_arena_bytes == 0
    assert plan.fallback_count == 0


def test_cpu_generic_selects_optimized_conv_and_gemm_with_ref_fallbacks():
    graph = _conv_gemm_graph()
    plan = select_kernel_specs(graph, load_target_profile("cpu_generic"))

    conv_spec = plan.by_node[0]
    gemm_spec = plan.by_node[1]
    assert (conv_spec.backend, conv_spec.kernel_kind) == ("simd", "pointwise_1x1")
    assert conv_spec.scratch_bytes == 0
    assert conv_spec.fallback_kernel_spec_id != 0xFFFFFFFF
    assert (gemm_spec.backend, gemm_spec.kernel_kind) == ("simd", "direct")
    assert gemm_spec.fallback_kernel_spec_id != 0xFFFFFFFF
    assert plan.scratch_arena_bytes == 0
    assert graph.metadata["kernel_fallback_count"] == 0


def test_cpu_generic_selects_shape_specific_conv_kinds():
    profile = load_target_profile("cpu_generic")

    cases = [
        (_conv_only_graph([1, 16, 8, 8], [32, 16, 1, 1], [1, 32, 8, 8], {"kernel_shape": [1, 1]}), "pointwise_1x1"),
        (_conv_only_graph([1, 8, 8, 8], [8, 1, 3, 3], [1, 8, 8, 8], {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "group": 8}), "depthwise_direct"),
        (_conv_only_graph([1, 16, 16, 16], [16, 16, 3, 3], [1, 16, 16, 16], {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}), "im2col_gemm"),
        (_conv_only_graph([1, 8, 8, 8], [16, 8, 3, 3], [1, 16, 4, 4], {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2]}), "im2col_gemm"),
        (_conv_only_graph([1, 8, 8, 8], [16, 8, 5, 5], [1, 16, 8, 8], {"kernel_shape": [5, 5], "pads": [2, 2, 2, 2]}), "im2col_gemm"),
    ]

    for graph, expected_kind in cases:
        plan = select_kernel_specs(graph, profile)
        assert plan.by_node[0].kernel_kind == expected_kind


def test_scratch_budget_violation_raises():
    graph = _conv_only_graph(
        [1, 8, 8, 8],
        [16, 8, 5, 5],
        [1, 16, 8, 8],
        {"kernel_shape": [5, 5], "pads": [2, 2, 2, 2]},
    )
    profile = load_target_profile("cpu_generic")
    profile["memory"]["scratch_arena_max"] = 8

    with pytest.raises(MemoryError):
        select_kernel_specs(graph, profile)


def _conv_gemm_graph() -> Graph:
    graph = Graph(model_name="kernels")
    for tensor in [
        Tensor(0, "input", types.DTYPE_FP32, [1, 1, 2, 2], types.ROLE_INPUT),
        Tensor(1, "conv_w", types.DTYPE_FP32, [1, 1, 1, 1], types.ROLE_WEIGHT, data=b"\x00" * 4),
        Tensor(2, "conv_out", types.DTYPE_FP32, [1, 1, 2, 2], types.ROLE_ACTIVATION),
        Tensor(3, "gemm_w", types.DTYPE_FP32, [4, 2], types.ROLE_WEIGHT, data=b"\x00" * 32),
        Tensor(4, "output", types.DTYPE_FP32, [1, 2], types.ROLE_OUTPUT),
    ]:
        graph.add_tensor(tensor)
    graph.inputs = [0]
    graph.outputs = [4]
    graph.add_node(Node(0, "Conv", [0, 1], [2], {"kernel_shape": [1, 1]}))
    graph.add_node(Node(1, "Gemm", [2, 3], [4]))
    return graph


def _conv_only_graph(x_shape: list[int], w_shape: list[int], y_shape: list[int], attrs: dict) -> Graph:
    graph = Graph(model_name="conv_only")
    graph.add_tensor(Tensor(0, "input", types.DTYPE_FP32, x_shape, types.ROLE_INPUT))
    graph.add_tensor(Tensor(1, "conv_w", types.DTYPE_FP32, w_shape, types.ROLE_WEIGHT, data=b"\x00" * (4 * _numel(w_shape))))
    graph.add_tensor(Tensor(2, "output", types.DTYPE_FP32, y_shape, types.ROLE_OUTPUT))
    graph.inputs = [0]
    graph.outputs = [2]
    graph.add_node(Node(0, "Conv", [0, 1], [2], attrs))
    return graph


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
