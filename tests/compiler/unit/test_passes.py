from __future__ import annotations

import numpy as np

from compiler.ir import types
from compiler.ir.graph import Graph, Node, Tensor
from compiler.passes.manager import run_pass_pipeline, pipeline_for_target
from compiler.passes import transforms


def test_eliminate_identity_dropout_rewires_internal_tensor():
    graph = Graph(model_name="identity")
    _add_tensor(graph, "input", [1, 4], types.ROLE_INPUT)
    _add_tensor(graph, "id_out", [1, 4], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 4], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [2]
    graph.add_node(Node(0, "Identity", [0], [1]))
    graph.add_node(Node(1, "Relu", [1], [2]))

    changed = transforms.eliminate_identity_dropout(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Relu"]
    assert graph.nodes[0].inputs == graph.inputs
    assert graph.outputs == [1]


def test_constant_fold_internal_add_replaces_node_with_weight_tensor():
    graph = Graph(model_name="const_add")
    _add_tensor(graph, "a", [1, 2], types.ROLE_WEIGHT, np.array([[1.0, 2.0]], dtype=np.float32))
    _add_tensor(graph, "b", [1, 2], types.ROLE_WEIGHT, np.array([[3.0, 4.0]], dtype=np.float32))
    _add_tensor(graph, "sum", [1, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 2], types.ROLE_OUTPUT)
    graph.outputs = [3]
    graph.add_node(Node(0, "Add", [0, 1], [2]))
    graph.add_node(Node(1, "Relu", [2], [3]))

    changed = transforms.constant_fold(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Relu"]
    folded = graph.tensors[graph.nodes[0].inputs[0]]
    assert folded.role == types.ROLE_WEIGHT
    np.testing.assert_allclose(_tensor_array(folded), np.array([[4.0, 6.0]], dtype=np.float32))


def test_fuse_conv_batchnorm_updates_conv_weights_and_bias():
    graph = Graph(model_name="conv_bn")
    _add_tensor(graph, "input", [1, 1, 2, 2], types.ROLE_INPUT)
    _add_tensor(graph, "w", [2, 1, 1, 1], types.ROLE_WEIGHT, np.array([[[[2.0]]], [[[4.0]]]], dtype=np.float32))
    _add_tensor(graph, "b", [2], types.ROLE_WEIGHT, np.array([0.5, -0.5], dtype=np.float32))
    _add_tensor(graph, "conv_out", [1, 2, 2, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "scale", [2], types.ROLE_WEIGHT, np.array([3.0, 5.0], dtype=np.float32))
    _add_tensor(graph, "bn_bias", [2], types.ROLE_WEIGHT, np.array([0.25, -0.25], dtype=np.float32))
    _add_tensor(graph, "mean", [2], types.ROLE_WEIGHT, np.array([1.0, 2.0], dtype=np.float32))
    _add_tensor(graph, "var", [2], types.ROLE_WEIGHT, np.array([8.0, 24.0], dtype=np.float32))
    _add_tensor(graph, "output", [1, 2, 2, 2], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [8]
    graph.add_node(Node(0, "Conv", [0, 1, 2], [3], {"kernel_shape": [1, 1]}))
    graph.add_node(Node(1, "BatchNormalization", [3, 4, 5, 6, 7], [8], {"epsilon": 1e-5}))

    changed = transforms.fuse_conv_batchnorm(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Conv"]
    conv = graph.nodes[0]
    fused_w = _tensor_array(graph.tensors[conv.inputs[1]])
    fused_b = _tensor_array(graph.tensors[conv.inputs[2]])
    factor = np.array([3.0, 5.0], dtype=np.float32) / np.sqrt(np.array([8.0, 24.0], dtype=np.float32) + 1e-5)
    np.testing.assert_allclose(fused_w.reshape(2), np.array([2.0, 4.0], dtype=np.float32) * factor, rtol=1e-6)
    np.testing.assert_allclose(
        fused_b,
        factor * (np.array([0.5, -0.5], dtype=np.float32) - np.array([1.0, 2.0], dtype=np.float32))
        + np.array([0.25, -0.25], dtype=np.float32),
        rtol=1e-6,
    )
    assert graph.tensors[conv.outputs[0]].role == types.ROLE_OUTPUT


def test_fuse_conv_relu_uses_fused_activation_attr():
    graph = Graph(model_name="conv_relu")
    _add_tensor(graph, "input", [1, 1, 2, 2], types.ROLE_INPUT)
    _add_tensor(graph, "w", [1, 1, 1, 1], types.ROLE_WEIGHT, np.ones((1, 1, 1, 1), dtype=np.float32))
    _add_tensor(graph, "conv_out", [1, 1, 2, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 1, 2, 2], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [3]
    graph.add_node(Node(0, "Conv", [0, 1], [2], {"kernel_shape": [1, 1]}))
    graph.add_node(Node(1, "Relu", [2], [3]))

    changed = transforms.fuse_conv_relu(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Conv"]
    assert graph.nodes[0].attrs["fused_activation"] == "Relu"
    assert graph.tensors[graph.nodes[0].outputs[0]].role == types.ROLE_OUTPUT


def test_fuse_conv_silu_removes_sigmoid_mul_pair():
    graph = Graph(model_name="conv_silu")
    _add_tensor(graph, "input", [1, 1, 2, 2], types.ROLE_INPUT)
    _add_tensor(graph, "w", [1, 1, 1, 1], types.ROLE_WEIGHT, np.ones((1, 1, 1, 1), dtype=np.float32))
    _add_tensor(graph, "conv_out", [1, 1, 2, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "sigmoid_out", [1, 1, 2, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 1, 2, 2], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [4]
    graph.add_node(Node(0, "Conv", [0, 1], [2], {"kernel_shape": [1, 1]}))
    graph.add_node(Node(1, "Sigmoid", [2], [3]))
    graph.add_node(Node(2, "Mul", [2, 3], [4]))

    changed = transforms.fuse_conv_silu(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Conv"]
    assert graph.nodes[0].attrs["fused_activation"] == "Silu"
    assert graph.tensors[graph.nodes[0].outputs[0]].role == types.ROLE_OUTPUT


def test_fuse_add_relu_uses_fused_activation_attr():
    graph = Graph(model_name="add_relu")
    _add_tensor(graph, "a", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "b", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "add_out", [1, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 2], types.ROLE_OUTPUT)
    graph.inputs = [0, 1]
    graph.outputs = [3]
    graph.add_node(Node(0, "Add", [0, 1], [2]))
    graph.add_node(Node(1, "Relu", [2], [3]))

    changed = transforms.fuse_add_relu(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Add"]
    assert graph.nodes[0].attrs["fused_activation"] == "Relu"
    assert graph.tensors[graph.nodes[0].outputs[0]].role == types.ROLE_OUTPUT


def test_fuse_add_relu_keeps_multi_consumer_relu():
    graph = Graph(model_name="add_relu_multi")
    _add_tensor(graph, "a", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "b", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "add_out", [1, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "relu_out", [1, 2], types.ROLE_OUTPUT)
    _add_tensor(graph, "other_out", [1, 2], types.ROLE_OUTPUT)
    graph.inputs = [0, 1]
    graph.outputs = [3, 4]
    graph.add_node(Node(0, "Add", [0, 1], [2]))
    graph.add_node(Node(1, "Relu", [2], [3]))
    graph.add_node(Node(2, "Mul", [2, 2], [4]))

    changed = transforms.fuse_add_relu(graph)

    assert changed == 0
    assert [node.op_type for node in graph.nodes] == ["Add", "Relu", "Mul"]


def test_eliminate_dead_keeps_only_output_dependencies():
    graph = Graph(model_name="dead_branch")
    _add_tensor(graph, "input", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "live", [1, 2], types.ROLE_OUTPUT)
    _add_tensor(graph, "dead", [1, 2], types.ROLE_ACTIVATION)
    graph.inputs = [0]
    graph.outputs = [1]
    graph.add_node(Node(0, "Relu", [0], [1]))
    graph.add_node(Node(1, "Relu", [0], [2]))

    changed = transforms.eliminate_dead(graph)

    assert changed == 1
    assert [node.op_type for node in graph.nodes] == ["Relu"]
    assert graph.nodes[0].outputs == graph.outputs


def test_default_pipeline_records_pass_stats():
    graph = Graph(model_name="pipeline")
    _add_tensor(graph, "input", [1, 4], types.ROLE_INPUT)
    _add_tensor(graph, "id_out", [1, 4], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 4], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [2]
    graph.add_node(Node(0, "Identity", [0], [1]))
    graph.add_node(Node(1, "Relu", [1], [2]))

    results = run_pass_pipeline(graph)

    assert results[0].name == "EliminateIdentityDropout"
    assert results[0].changed == 1
    assert graph.metadata["pass_stats"][0]["nodes_before"] == 2


def _add_tensor(
    graph: Graph,
    name: str,
    shape: list[int],
    role: str,
    value: np.ndarray | None = None,
) -> None:
    graph.add_tensor(
        Tensor(
            id=len(graph.tensors),
            name=name,
            dtype=types.DTYPE_FP32,
            shape=shape,
            role=role,
            data=None if value is None else np.ascontiguousarray(value, dtype=np.float32).tobytes(),
        )
    )


def _tensor_array(tensor: Tensor) -> np.ndarray:
    return np.frombuffer(tensor.data or b"", dtype=np.float32).copy().reshape(tensor.shape)


def test_pipeline_for_target_includes_conv_add_with_simd():
    profile = {"backends": ["ref", "cpu", "simd"]}
    pipeline = pipeline_for_target(profile)
    assert "FuseConvAdd" in pipeline


def test_pipeline_for_target_excludes_conv_add_without_simd():
    profile = {"backends": ["ref"]}
    pipeline = pipeline_for_target(profile)
    assert "FuseConvAdd" not in pipeline
    assert "FuseConvRelu" in pipeline


def test_pipeline_for_target_no_profile_includes_all():
    pipeline = pipeline_for_target(None)
    assert "FuseConvAdd" in pipeline
    assert "FuseConvRelu" in pipeline


def test_eliminate_noop_transpose_removes_inverse_pair():
    graph = Graph(model_name="transpose_noop")
    _add_tensor(graph, "input", [1, 3, 4, 5], types.ROLE_INPUT)
    _add_tensor(graph, "t1_out", [1, 4, 5, 3], types.ROLE_ACTIVATION)
    _add_tensor(graph, "t2_out", [1, 3, 4, 5], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 3, 4, 5], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [3]
    graph.add_node(Node(0, "Transpose", [0], [1], {"perm": [0, 2, 3, 1]}))
    graph.add_node(Node(1, "Transpose", [1], [2], {"perm": [0, 3, 1, 2]}))
    graph.add_node(Node(2, "Relu", [2], [3]))

    changed = transforms.eliminate_noop_transpose(graph)

    assert changed == 2
    assert [n.op_type for n in graph.nodes] == ["Relu"]
    assert graph.nodes[0].inputs == graph.inputs


def test_eliminate_noop_transpose_keeps_non_inverse():
    graph = Graph(model_name="transpose_keep")
    _add_tensor(graph, "input", [1, 3, 4, 5], types.ROLE_INPUT)
    _add_tensor(graph, "t1_out", [1, 4, 5, 3], types.ROLE_ACTIVATION)
    _add_tensor(graph, "t2_out", [1, 5, 3, 4], types.ROLE_ACTIVATION)
    _add_tensor(graph, "output", [1, 5, 3, 4], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [3]
    graph.add_node(Node(0, "Transpose", [0], [1], {"perm": [0, 2, 3, 1]}))
    graph.add_node(Node(1, "Transpose", [1], [2], {"perm": [0, 2, 3, 1]}))
    graph.add_node(Node(2, "Relu", [2], [3]))

    changed = transforms.eliminate_noop_transpose(graph)

    assert changed == 0
    assert len(graph.nodes) == 3


def test_share_constants_deduplicates_identical_weights():
    graph = Graph(model_name="share_const")
    data = np.array([1.0, 2.0], dtype=np.float32)
    _add_tensor(graph, "input", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "w1", [2], types.ROLE_WEIGHT, data)
    _add_tensor(graph, "w2", [2], types.ROLE_WEIGHT, data.copy())
    _add_tensor(graph, "add1_out", [1, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "add2_out", [1, 2], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [4]
    graph.add_node(Node(0, "Add", [0, 1], [3]))
    graph.add_node(Node(1, "Add", [3, 2], [4]))

    changed = transforms.share_constants(graph)

    assert changed == 1
    assert graph.nodes[1].inputs[1] == graph.nodes[0].inputs[1]


def test_share_constants_keeps_different_weights():
    graph = Graph(model_name="share_diff")
    _add_tensor(graph, "input", [1, 2], types.ROLE_INPUT)
    _add_tensor(graph, "w1", [2], types.ROLE_WEIGHT, np.array([1.0, 2.0], dtype=np.float32))
    _add_tensor(graph, "w2", [2], types.ROLE_WEIGHT, np.array([3.0, 4.0], dtype=np.float32))
    _add_tensor(graph, "add1_out", [1, 2], types.ROLE_ACTIVATION)
    _add_tensor(graph, "add2_out", [1, 2], types.ROLE_OUTPUT)
    graph.inputs = [0]
    graph.outputs = [4]
    graph.add_node(Node(0, "Add", [0, 1], [3]))
    graph.add_node(Node(1, "Add", [3, 2], [4]))

    changed = transforms.share_constants(graph)

    assert changed == 0
