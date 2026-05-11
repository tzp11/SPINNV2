"""Minimal ONNX to SIR importer for M1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from compiler.ir.graph import Graph, Node, Tensor
from compiler.ir import types


def import_onnx(path: str | Path) -> Graph:
    model = onnx.load(path)
    graph_proto = model.graph
    graph = Graph(model_name=graph_proto.name or Path(path).stem)

    value_shapes = _collect_value_shapes(model)
    initializers = {init.name: numpy_helper.to_array(init).astype(np.float32) for init in graph_proto.initializer}
    tensor_ids: dict[str, int] = {}

    def ensure_tensor(name: str, role: str | None = None, data: bytes | None = None) -> int:
        if name in tensor_ids:
            tensor = graph.tensors[tensor_ids[name]]
            if role == types.ROLE_OUTPUT:
                tensor.role = types.ROLE_OUTPUT
            return tensor.id

        shape = value_shapes.get(name)
        if shape is None and name in initializers:
            shape = list(initializers[name].shape)
        if shape is None:
            raise ValueError(f"missing static shape for tensor: {name}")

        tensor_role = role or (types.ROLE_WEIGHT if name in initializers else types.ROLE_ACTIVATION)
        tensor_data = data
        if tensor_data is None and name in initializers:
            tensor_data = np.ascontiguousarray(initializers[name], dtype=np.float32).tobytes()

        tensor_id = len(graph.tensors)
        graph.add_tensor(
            Tensor(
                id=tensor_id,
                name=name,
                dtype=types.DTYPE_FP32,
                shape=[int(dim) for dim in shape],
                role=tensor_role,
                data=tensor_data,
            )
        )
        tensor_ids[name] = tensor_id
        return tensor_id

    initializer_names = set(initializers)
    for value in graph_proto.input:
        if value.name in initializer_names:
            ensure_tensor(value.name, types.ROLE_WEIGHT)
            continue
        tensor_id = ensure_tensor(value.name, types.ROLE_INPUT)
        graph.inputs.append(tensor_id)

    for init_name in initializers:
        ensure_tensor(init_name, types.ROLE_WEIGHT)

    for node_proto in graph_proto.node:
        if node_proto.op_type not in types.SUPPORTED_IMPORT_OPS:
            raise ValueError(f"unsupported import op: {node_proto.op_type}")

        input_ids = [ensure_tensor(name) for name in node_proto.input if name]
        output_ids = [ensure_tensor(name) for name in node_proto.output if name]
        node_id = len(graph.nodes)
        graph.add_node(
            Node(
                id=node_id,
                op_type=node_proto.op_type,
                inputs=input_ids,
                outputs=output_ids,
                attrs=_extract_attrs(node_proto),
                execution_index=node_id,
            )
        )

    for value in graph_proto.output:
        tensor_id = ensure_tensor(value.name, types.ROLE_OUTPUT)
        if tensor_id not in graph.outputs:
            graph.outputs.append(tensor_id)

    return graph


def _collect_value_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    shapes: dict[str, list[int]] = {}

    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                raise ValueError(f"dynamic shape is not supported in M1: {value.name}")
        shapes[value.name] = dims

    return shapes


def _extract_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            attrs[attr.name] = value.decode("utf-8")
        elif isinstance(value, (list, tuple)):
            attrs[attr.name] = [int(v) if isinstance(v, np.integer) else v for v in value]
        elif isinstance(value, np.ndarray):
            attrs[attr.name] = value.tolist()
        elif isinstance(value, np.integer):
            attrs[attr.name] = int(value)
        elif isinstance(value, np.floating):
            attrs[attr.name] = float(value)
        else:
            attrs[attr.name] = value
    return attrs
