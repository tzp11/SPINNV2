"""SIR graph transforms for M3."""

from __future__ import annotations

import numpy as np

from compiler.ir.graph import Graph, Node, Tensor
from compiler.ir import types


def rebuild_graph_links(graph: Graph) -> None:
    for tensor in graph.tensors:
        tensor.producer = None
        tensor.consumers = []
    for node_index, node in enumerate(graph.nodes):
        node.id = node_index
        node.execution_index = node_index
        for input_id in node.inputs:
            graph.tensors[input_id].consumers.append(node.id)
        for output_id in node.outputs:
            graph.tensors[output_id].producer = node.id


def eliminate_identity_dropout(graph: Graph) -> int:
    changed = 0
    new_nodes: list[Node] = []
    for node in graph.nodes:
        if node.op_type in {"Identity", "Dropout"} and node.inputs and node.outputs:
            if any(output_id in graph.outputs for output_id in node.outputs):
                new_nodes.append(node)
                continue
            _redirect_tensor(graph, node.outputs[0], node.inputs[0])
            changed += 1
            continue
        new_nodes.append(node)
    graph.nodes = new_nodes
    _compact_graph(graph)
    return changed


def constant_fold(graph: Graph) -> int:
    changed = 0
    new_nodes: list[Node] = []
    for node in graph.nodes:
        if any(output_id in graph.outputs for output_id in node.outputs):
            new_nodes.append(node)
            continue
        if node.op_type in {"Add", "Relu", "Flatten"} and all(_is_const(graph, tid) for tid in node.inputs):
            value = _eval_constant_node(graph, node)
            if value is not None:
                out = graph.tensors[node.outputs[0]]
                out.data = np.ascontiguousarray(value, dtype=np.float32).tobytes()
                out.shape = [int(dim) for dim in value.shape]
                out.role = types.ROLE_WEIGHT
                changed += 1
                continue
        new_nodes.append(node)
    graph.nodes = new_nodes
    _compact_graph(graph)
    return changed


def fuse_conv_batchnorm(graph: Graph) -> int:
    rebuild_graph_links(graph)
    changed = 0
    remove_nodes: set[int] = set()
    for bn in list(graph.nodes):
        if bn.op_type != "BatchNormalization" or len(bn.inputs) < 5:
            continue
        conv_out = bn.inputs[0]
        conv_node_id = graph.tensors[conv_out].producer
        if conv_node_id is None:
            continue
        conv = graph.nodes[conv_node_id]
        if conv.op_type != "Conv":
            continue
        if _active_consumers(graph, conv_out, ignore_node=bn.id) != 0:
            continue
        if not all(_is_const(graph, tid) for tid in bn.inputs[1:5]):
            continue

        x_id, w_id = conv.inputs[0], conv.inputs[1]
        old_w = _tensor_array(graph, w_id)
        scale = _tensor_array(graph, bn.inputs[1])
        bias = _tensor_array(graph, bn.inputs[2])
        mean = _tensor_array(graph, bn.inputs[3])
        var = _tensor_array(graph, bn.inputs[4])
        eps = float(bn.attrs.get("epsilon", 1e-5))

        if old_w.ndim != 4:
            continue
        oc = old_w.shape[0]
        factor = scale.reshape(oc) / np.sqrt(var.reshape(oc) + eps)
        new_w = old_w * factor.reshape(oc, 1, 1, 1)

        if len(conv.inputs) >= 3 and _is_const(graph, conv.inputs[2]):
            old_b = _tensor_array(graph, conv.inputs[2]).reshape(oc)
            b_id = conv.inputs[2]
        else:
            old_b = np.zeros((oc,), dtype=np.float32)
            b_id = _add_weight_tensor(graph, f"{graph.tensors[w_id].name}_bn_bias", old_b)
            conv.inputs.append(b_id)

        new_b = factor.reshape(oc) * (old_b - mean.reshape(oc)) + bias.reshape(oc)
        _set_tensor_array(graph, w_id, new_w)
        _set_tensor_array(graph, b_id, new_b)

        conv.outputs[0] = bn.outputs[0]
        graph.tensors[bn.outputs[0]].producer = conv.id
        remove_nodes.add(bn.id)
        changed += 1

    if remove_nodes:
        graph.nodes = [node for node in graph.nodes if node.id not in remove_nodes]
        _compact_graph(graph)
    return changed


def fuse_conv_relu(graph: Graph) -> int:
    rebuild_graph_links(graph)
    changed = 0
    remove_nodes: set[int] = set()
    for relu in list(graph.nodes):
        if relu.op_type != "Relu" or not relu.inputs or not relu.outputs:
            continue
        conv_out = relu.inputs[0]
        conv_node_id = graph.tensors[conv_out].producer
        if conv_node_id is None:
            continue
        conv = graph.nodes[conv_node_id]
        if conv.op_type != "Conv":
            continue
        if _active_consumers(graph, conv_out, ignore_node=relu.id) != 0:
            continue
        conv.attrs["fused_activation"] = "Relu"
        conv.outputs[0] = relu.outputs[0]
        remove_nodes.add(relu.id)
        changed += 1

    if remove_nodes:
        graph.nodes = [node for node in graph.nodes if node.id not in remove_nodes]
        _compact_graph(graph)
    return changed


def fuse_conv_silu(graph: Graph) -> int:
    rebuild_graph_links(graph)
    changed = 0
    remove_nodes: set[int] = set()
    for conv in list(graph.nodes):
        if conv.op_type != "Conv" or not conv.outputs:
            continue
        conv_out = conv.outputs[0]
        consumers = [graph.nodes[node_id] for node_id in graph.tensors[conv_out].consumers]
        if len(consumers) != 2:
            continue

        sigmoid = next((node for node in consumers if node.op_type == "Sigmoid"), None)
        mul = next((node for node in consumers if node.op_type == "Mul"), None)
        if sigmoid is None or mul is None or not sigmoid.outputs or not mul.outputs:
            continue

        sigmoid_out = sigmoid.outputs[0]
        if _active_consumers(graph, sigmoid_out, ignore_node=mul.id) != 0:
            continue
        if not ({conv_out, sigmoid_out} <= set(mul.inputs)):
            continue

        conv.attrs["fused_activation"] = "Silu"
        conv.outputs[0] = mul.outputs[0]
        graph.tensors[mul.outputs[0]].producer = conv.id
        remove_nodes.update({sigmoid.id, mul.id})
        changed += 1

    if remove_nodes:
        graph.nodes = [node for node in graph.nodes if node.id not in remove_nodes]
        _compact_graph(graph)
    return changed


def fuse_add_relu(graph: Graph) -> int:
    rebuild_graph_links(graph)
    changed = 0
    remove_nodes: set[int] = set()
    for relu in list(graph.nodes):
        if relu.op_type != "Relu" or not relu.inputs or not relu.outputs:
            continue
        add_out = relu.inputs[0]
        add_node_id = graph.tensors[add_out].producer
        if add_node_id is None:
            continue
        add = graph.nodes[add_node_id]
        if add.op_type != "Add":
            continue
        if _active_consumers(graph, add_out, ignore_node=relu.id) != 0:
            continue
        add.attrs["fused_activation"] = "Relu"
        add.outputs[0] = relu.outputs[0]
        graph.tensors[relu.outputs[0]].producer = add.id
        remove_nodes.add(relu.id)
        changed += 1

    if remove_nodes:
        graph.nodes = [node for node in graph.nodes if node.id not in remove_nodes]
        _compact_graph(graph)
    return changed


def eliminate_dead(graph: Graph) -> int:
    rebuild_graph_links(graph)
    live_tensors = set(graph.outputs)
    live_nodes: set[int] = set()
    changed = 0

    worklist = list(graph.outputs)
    while worklist:
        tensor_id = worklist.pop()
        producer = graph.tensors[tensor_id].producer
        if producer is None or producer in live_nodes:
            continue
        live_nodes.add(producer)
        for input_id in graph.nodes[producer].inputs:
            if input_id not in live_tensors:
                live_tensors.add(input_id)
                worklist.append(input_id)

    before = len(graph.nodes)
    graph.nodes = [node for node in graph.nodes if node.id in live_nodes]
    changed += before - len(graph.nodes)
    _compact_graph(graph)
    return changed


def _redirect_tensor(graph: Graph, old_id: int, new_id: int) -> None:
    for node in graph.nodes:
        node.inputs = [new_id if tid == old_id else tid for tid in node.inputs]
    graph.outputs = [new_id if tid == old_id else tid for tid in graph.outputs]


def _compact_graph(graph: Graph) -> None:
    rebuild_graph_links(graph)
    used = set(graph.inputs) | set(graph.outputs)
    for node in graph.nodes:
        used.update(node.inputs)
        used.update(node.outputs)

    mapping: dict[int, int] = {}
    new_tensors: list[Tensor] = []
    for old_id, tensor in enumerate(graph.tensors):
        if old_id not in used:
            continue
        new_id = len(new_tensors)
        mapping[old_id] = new_id
        new_tensors.append(
            Tensor(
                id=new_id,
                name=tensor.name,
                dtype=tensor.dtype,
                shape=list(tensor.shape),
                role=tensor.role,
                layout=tensor.layout,
                data=tensor.data,
            )
        )

    graph.tensors = new_tensors
    graph.inputs = [mapping[tid] for tid in graph.inputs if tid in mapping]
    graph.outputs = [mapping[tid] for tid in graph.outputs if tid in mapping]
    for node_id, node in enumerate(graph.nodes):
        node.id = node_id
        node.execution_index = node_id
        node.inputs = [mapping[tid] for tid in node.inputs if tid in mapping]
        node.outputs = [mapping[tid] for tid in node.outputs if tid in mapping]
    rebuild_graph_links(graph)


def _active_consumers(graph: Graph, tensor_id: int, *, ignore_node: int) -> int:
    return sum(1 for node_id in graph.tensors[tensor_id].consumers if node_id != ignore_node)


def _is_const(graph: Graph, tensor_id: int) -> bool:
    tensor = graph.tensors[tensor_id]
    return tensor.data is not None and tensor.role in {types.ROLE_WEIGHT, types.ROLE_CONSTANT}


def _tensor_array(graph: Graph, tensor_id: int) -> np.ndarray:
    tensor = graph.tensors[tensor_id]
    return np.frombuffer(tensor.data or b"", dtype=np.float32).copy().reshape(tensor.shape)


def _set_tensor_array(graph: Graph, tensor_id: int, value: np.ndarray) -> None:
    tensor = graph.tensors[tensor_id]
    value = np.ascontiguousarray(value, dtype=np.float32)
    tensor.shape = [int(dim) for dim in value.shape]
    tensor.data = value.tobytes()
    tensor.role = types.ROLE_WEIGHT


def _add_weight_tensor(graph: Graph, name: str, value: np.ndarray) -> int:
    value = np.ascontiguousarray(value, dtype=np.float32)
    tensor_id = len(graph.tensors)
    graph.add_tensor(
        Tensor(
            id=tensor_id,
            name=name,
            dtype=types.DTYPE_FP32,
            shape=[int(dim) for dim in value.shape],
            role=types.ROLE_WEIGHT,
            data=value.tobytes(),
        )
    )
    return tensor_id


def _eval_constant_node(graph: Graph, node: Node) -> np.ndarray | None:
    values = [_tensor_array(graph, tid) for tid in node.inputs]
    if node.op_type == "Add":
        return values[0] + values[1]
    if node.op_type == "Relu":
        return np.maximum(values[0], 0.0)
    if node.op_type == "Flatten":
        axis = int(node.attrs.get("axis", 1))
        shape = values[0].shape
        left = int(np.prod(shape[:axis]))
        right = int(np.prod(shape[axis:]))
        return values[0].reshape(left, right)
    return None
