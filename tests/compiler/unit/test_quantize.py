"""Unit tests for INT8 quantization pipeline."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from compiler.ir import types
from compiler.ir.graph import Graph, Node, Tensor
from compiler.quantization.quantize_weights import QuantParam, quantize_conv_weights


def _add_tensor(
    graph: Graph,
    name: str,
    shape: list[int],
    role: str,
    value: np.ndarray | None = None,
    dtype: str = types.DTYPE_FP32,
) -> None:
    data = None
    if value is not None:
        if dtype == types.DTYPE_FP32:
            data = np.ascontiguousarray(value, dtype=np.float32).tobytes()
        else:
            data = value.tobytes()
    graph.add_tensor(
        Tensor(
            id=len(graph.tensors),
            name=name,
            dtype=dtype,
            shape=shape,
            role=role,
            data=data,
        )
    )


def test_size_bytes_int8():
    t = Tensor(id=0, name="w", dtype=types.DTYPE_INT8, shape=[2, 3, 3, 3], role=types.ROLE_WEIGHT)
    assert t.size_bytes == 2 * 3 * 3 * 3


def test_size_bytes_fp32():
    t = Tensor(id=0, name="w", dtype=types.DTYPE_FP32, shape=[2, 3, 3, 3], role=types.ROLE_WEIGHT)
    assert t.size_bytes == 2 * 3 * 3 * 3 * 4


def test_size_bytes_fp16():
    t = Tensor(id=0, name="w", dtype="fp16", shape=[2, 3], role=types.ROLE_WEIGHT)
    assert t.size_bytes == 2 * 3 * 2  # 6 elements * 2 bytes


def test_quantize_conv_weights_basic():
    """Quantize a simple Conv weight tensor and verify dtype, scales, round-trip error."""
    graph = Graph(model_name="quant_test")
    _add_tensor(graph, "input", [1, 3, 8, 8], types.ROLE_INPUT)
    w_data = np.random.randn(16, 3, 3, 3).astype(np.float32)
    _add_tensor(graph, "weight", [16, 3, 3, 3], types.ROLE_WEIGHT, w_data)
    _add_tensor(graph, "bias", [16], types.ROLE_WEIGHT, np.zeros(16, dtype=np.float32))
    _add_tensor(graph, "output", [1, 16, 6, 6], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [3]
    graph.add_node(Node(0, "Conv", [0, 1, 2], [3], {"kernel_shape": [3, 3]}))

    quant_params = quantize_conv_weights(graph)

    assert len(quant_params) == 1
    qp = quant_params[0]
    assert qp.tensor_id == 1
    assert qp.scheme == "per_channel"
    assert qp.quant_axis == 0
    assert qp.num_channels == 16
    assert len(qp.scales) == 16
    assert all(s > 0 for s in qp.scales)
    assert all(zp == 0 for zp in qp.zero_points)

    w_tensor = graph.tensors[1]
    assert w_tensor.dtype == types.DTYPE_INT8
    assert w_tensor.size_bytes == 16 * 3 * 3 * 3

    w_q = np.frombuffer(w_tensor.data, dtype=np.int8).reshape(16, 3, 3, 3)
    w_deq = np.zeros_like(w_data)
    for c in range(16):
        w_deq[c] = w_q[c].astype(np.float32) * qp.scales[c]
    max_err = np.max(np.abs(w_data - w_deq))
    max_val = np.max(np.abs(w_data))
    assert max_err < max_val / 127.0 + 1e-6


def test_quantize_conv_weights_no_conv():
    """Non-Conv graph should produce no quant params."""
    graph = Graph(model_name="no_conv")
    _add_tensor(graph, "input", [1, 4], types.ROLE_INPUT)
    _add_tensor(graph, "output", [1, 4], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [1]
    graph.add_node(Node(0, "Relu", [0], [1]))

    quant_params = quantize_conv_weights(graph)
    assert len(quant_params) == 0


def test_quantize_conv_weights_shared_weight():
    """Two Conv nodes sharing the same weight tensor should only quantize once."""
    graph = Graph(model_name="shared_weight")
    _add_tensor(graph, "input", [1, 3, 8, 8], types.ROLE_INPUT)
    w_data = np.random.randn(8, 3, 1, 1).astype(np.float32)
    _add_tensor(graph, "weight", [8, 3, 1, 1], types.ROLE_WEIGHT, w_data)
    _add_tensor(graph, "conv1_out", [1, 8, 8, 8], types.ROLE_ACTIVATION)
    _add_tensor(graph, "conv2_out", [1, 8, 8, 8], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [3]
    graph.add_node(Node(0, "Conv", [0, 1], [2], {"kernel_shape": [1, 1]}))
    graph.add_node(Node(1, "Conv", [0, 1], [3], {"kernel_shape": [1, 1]}))

    quant_params = quantize_conv_weights(graph)
    assert len(quant_params) == 1


def test_feature_mask_int8():
    from compiler.packager.spk_writer import _feature_mask
    assert _feature_mask(["fp32"]) == 1
    assert _feature_mask(["int8"]) == 2
    assert _feature_mask(["fp32", "int8"]) == 3
    assert _feature_mask([]) == 0


def test_kernel_spec_int8_selection():
    """When Conv weight is INT8, kernel spec should select int8_im2col_gemm."""
    from compiler.planner.kernel_spec import select_kernel_specs

    graph = Graph(model_name="int8_conv")
    _add_tensor(graph, "input", [1, 3, 8, 8], types.ROLE_INPUT)
    w_q = np.random.randint(-128, 127, size=(8, 3, 3, 3), dtype=np.int8)
    _add_tensor(graph, "weight", [8, 3, 3, 3], types.ROLE_WEIGHT, w_q, dtype=types.DTYPE_INT8)
    _add_tensor(graph, "output", [1, 8, 6, 6], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [2]
    graph.add_node(Node(0, "Conv", [0, 1], [2], {"kernel_shape": [3, 3]}))

    profile = {
        "features": ["fp32", "int8"],
        "backends": ["ref", "simd"],
        "ops": {"Conv": ["simd_int8_im2col_gemm", "simd_im2col_gemm", "ref"]},
        "memory": {"activation_arena_max": 67108864, "scratch_arena_max": 33554432},
    }
    plan = select_kernel_specs(graph, profile)
    spec = plan.by_node[0]
    assert spec.kernel_kind == "int8_im2col_gemm"
    assert spec.dtype == "int8"
    assert spec.scratch_bytes > 0


def test_kernel_spec_fp32_unchanged():
    """FP32 Conv weights should still select im2col_gemm, not INT8."""
    from compiler.planner.kernel_spec import select_kernel_specs

    graph = Graph(model_name="fp32_conv")
    _add_tensor(graph, "input", [1, 3, 8, 8], types.ROLE_INPUT)
    w_fp32 = np.random.randn(8, 3, 3, 3).astype(np.float32)
    _add_tensor(graph, "weight", [8, 3, 3, 3], types.ROLE_WEIGHT, w_fp32)
    _add_tensor(graph, "output", [1, 8, 6, 6], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [2]
    graph.add_node(Node(0, "Conv", [0, 1], [2], {"kernel_shape": [3, 3]}))

    profile = {
        "features": ["fp32", "int8"],
        "backends": ["ref", "simd"],
        "ops": {"Conv": ["simd_int8_im2col_gemm", "simd_im2col_gemm", "ref"]},
        "memory": {"activation_arena_max": 67108864, "scratch_arena_max": 33554432},
    }
    plan = select_kernel_specs(graph, profile)
    spec = plan.by_node[0]
    assert spec.kernel_kind == "im2col_gemm"
    assert spec.dtype == "fp32"


def test_quant_section_serialization():
    """Verify SECTION_QUANTIZATION round-trips correctly."""
    from compiler.packager.spk_writer import _quantization_section_bytes, QUANT_PARAM_STRUCT

    qp = QuantParam(
        tensor_id=5,
        scheme="per_channel",
        quant_axis=0,
        num_channels=3,
        scales=[0.1, 0.2, 0.3],
        zero_points=[0, 0, 0],
    )
    data = _quantization_section_bytes([qp])

    rec = QUANT_PARAM_STRUCT.unpack_from(data, 0)
    assert rec[0] == 5    # tensor_id
    assert rec[1] == 2    # per_channel
    assert rec[2] == 0    # quant_axis
    assert rec[3] == 3    # num_channels

    data_offset = rec[4]
    scales = []
    for i in range(3):
        s = struct.unpack_from("<f", data, data_offset + i * 4)[0]
        scales.append(round(s, 6))
    assert scales == [0.1, 0.2, 0.3]
