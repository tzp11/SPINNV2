"""M1 binary SPK writer."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from compiler.ir.graph import Graph, Node, Tensor
from compiler.ir import types


SPKV2_MAGIC = 0x32564B50
VERSION_MAJOR = 0
VERSION_MINOR = 1

SECTION_METADATA = 1
SECTION_TARGET_PROFILE = 2
SECTION_TENSOR_TABLE = 3
SECTION_NODE_TABLE = 4
SECTION_ATTRIBUTES = 5
SECTION_WEIGHTS = 6
SECTION_STRING_TABLE = 10

DTYPE_CODES = {types.DTYPE_FP32: 1}
ROLE_CODES = {
    types.ROLE_INPUT: 1,
    types.ROLE_OUTPUT: 2,
    types.ROLE_WEIGHT: 3,
    types.ROLE_ACTIVATION: 4,
    types.ROLE_CONSTANT: 5,
}
MEMORY_CLASS_CODES = {
    types.ROLE_INPUT: 1,
    types.ROLE_OUTPUT: 2,
    types.ROLE_WEIGHT: 3,
    types.ROLE_ACTIVATION: 4,
    types.ROLE_CONSTANT: 3,
}
OP_CODES = {
    "Add": 1,
    "Conv": 2,
    "Flatten": 3,
    "Gemm": 4,
    "MaxPool": 5,
    "Relu": 6,
    "Softmax": 7,
}

HEADER_STRUCT = struct.Struct("<IHHHHIIIIIIQQQII")
SECTION_STRUCT = struct.Struct("<IIQQII")
TENSOR_STRUCT = struct.Struct("<IHHHH8IQQII")
NODE_STRUCT = struct.Struct("<IHHHH8I4IIIII")
ATTR_STRUCT = struct.Struct("<Iii4i2i2i2i2if")


def write_spk(graph: Graph, out_path: str | Path, target_profile: dict) -> None:
    out_path = Path(out_path)
    sections: list[tuple[int, bytes, int]] = []

    strings = _build_string_table(graph)
    weights, weight_offsets = _build_weights(graph)
    attrs, attr_offsets = _build_attrs(graph)

    sections.append((SECTION_METADATA, _metadata_bytes(graph), 1))
    sections.append((SECTION_TARGET_PROFILE, json.dumps(target_profile, sort_keys=True).encode("utf-8"), 1))
    sections.append((SECTION_TENSOR_TABLE, _tensor_table_bytes(graph, strings.offsets, weight_offsets), 4))
    sections.append((SECTION_NODE_TABLE, _node_table_bytes(graph, attr_offsets), 4))
    sections.append((SECTION_ATTRIBUTES, attrs, 4))
    sections.append((SECTION_WEIGHTS, weights, 16))
    sections.append((SECTION_STRING_TABLE, strings.blob, 1))

    section_count = len(sections)
    header_size = HEADER_STRUCT.size
    directory_size = section_count * SECTION_STRUCT.size
    offset = header_size + directory_size
    directory = bytearray()
    payload = bytearray()

    for kind, data, alignment in sections:
        aligned_offset = _align(offset, alignment)
        if aligned_offset > offset:
            payload.extend(b"\x00" * (aligned_offset - offset))
            offset = aligned_offset
        directory.extend(SECTION_STRUCT.pack(kind, 0, offset, len(data), alignment, 0))
        payload.extend(data)
        offset += len(data)

    header = HEADER_STRUCT.pack(
        SPKV2_MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        0,
        header_size,
        section_count,
        0,
        len(graph.tensors),
        len(graph.nodes),
        len(graph.inputs),
        len(graph.outputs),
        sum(t.size_bytes for t in graph.tensors if t.role == types.ROLE_WEIGHT),
        0,
        0,
        _stable_profile_hash(target_profile),
        0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + bytes(directory) + bytes(payload))
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(_debug_json(graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class _StringTable:
    def __init__(self) -> None:
        self.blob = bytearray(b"\x00")
        self.offsets: dict[str, int] = {"": 0}

    def add(self, text: str) -> int:
        if text in self.offsets:
            return self.offsets[text]
        offset = len(self.blob)
        self.blob.extend(text.encode("utf-8") + b"\x00")
        self.offsets[text] = offset
        return offset


def _build_string_table(graph: Graph) -> _StringTable:
    table = _StringTable()
    for tensor in graph.tensors:
        table.add(tensor.name)
    return table


def _build_weights(graph: Graph) -> tuple[bytes, dict[int, int]]:
    blob = bytearray()
    offsets: dict[int, int] = {}
    for tensor in graph.tensors:
        if tensor.role != types.ROLE_WEIGHT:
            continue
        blob.extend(b"\x00" * (_align(len(blob), 16) - len(blob)))
        offsets[tensor.id] = len(blob)
        if tensor.data is None:
            raise ValueError(f"weight tensor missing data: {tensor.name}")
        blob.extend(tensor.data)
    return bytes(blob), offsets


def _build_attrs(graph: Graph) -> tuple[bytes, dict[int, tuple[int, int]]]:
    blob = bytearray()
    offsets: dict[int, tuple[int, int]] = {}
    for node in graph.nodes:
        blob.extend(b"\x00" * (_align(len(blob), 4) - len(blob)))
        offset = len(blob)
        data = _attr_bytes(node)
        blob.extend(data)
        offsets[node.id] = (offset, len(data))
    return bytes(blob), offsets


def _attr_bytes(node: Node) -> bytes:
    attrs = node.attrs
    axis = int(attrs.get("axis", 1))
    kernel = _int_list(attrs.get("kernel_shape", [1, 1]), 2, 1)
    strides = _int_list(attrs.get("strides", [1, 1]), 2, 1)
    pads = _int_list(attrs.get("pads", [0, 0, 0, 0]), 4, 0)
    dilations = _int_list(attrs.get("dilations", [1, 1]), 2, 1)
    group = int(attrs.get("group", 1))
    alpha = float(attrs.get("alpha", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    return ATTR_STRUCT.pack(
        OP_CODES[node.op_type],
        axis,
        group,
        pads[0],
        pads[1],
        pads[2],
        pads[3],
        strides[0],
        strides[1],
        kernel[0],
        kernel[1],
        dilations[0],
        dilations[1],
        trans_a,
        trans_b,
        alpha,
    )


def _tensor_table_bytes(graph: Graph, string_offsets: dict[str, int], weight_offsets: dict[int, int]) -> bytes:
    blob = bytearray()
    for tensor in graph.tensors:
        shape = tensor.shape[:8] + [1] * (8 - len(tensor.shape))
        data_offset = weight_offsets.get(tensor.id, 0)
        blob.extend(
            TENSOR_STRUCT.pack(
                tensor.id,
                DTYPE_CODES[tensor.dtype],
                ROLE_CODES[tensor.role],
                len(tensor.shape),
                MEMORY_CLASS_CODES[tensor.role],
                *shape,
                tensor.size_bytes,
                data_offset,
                string_offsets[tensor.name],
                0,
            )
        )
    return bytes(blob)


def _node_table_bytes(graph: Graph, attr_offsets: dict[int, tuple[int, int]]) -> bytes:
    blob = bytearray()
    for node in graph.nodes:
        inputs = node.inputs[:8] + [0] * (8 - len(node.inputs))
        outputs = node.outputs[:4] + [0] * (4 - len(node.outputs))
        attr_offset, attr_size = attr_offsets[node.id]
        blob.extend(
            NODE_STRUCT.pack(
                node.id,
                OP_CODES[node.op_type],
                0,
                len(node.inputs),
                len(node.outputs),
                *inputs,
                *outputs,
                attr_offset,
                attr_size,
                0,
                0,
            )
        )
    return bytes(blob)


def _metadata_bytes(graph: Graph) -> bytes:
    metadata = {
        "model_name": graph.model_name,
        "sir_version": "0.1",
        "inputs": graph.inputs,
        "outputs": graph.outputs,
    }
    return json.dumps(metadata, sort_keys=True).encode("utf-8")


def _debug_json(graph: Graph) -> dict:
    return {
        "model_name": graph.model_name,
        "inputs": graph.inputs,
        "outputs": graph.outputs,
        "tensors": [
            {"id": t.id, "name": t.name, "shape": t.shape, "role": t.role, "size_bytes": t.size_bytes}
            for t in graph.tensors
        ],
        "nodes": [
            {"id": n.id, "op_type": n.op_type, "inputs": n.inputs, "outputs": n.outputs, "attrs": n.attrs}
            for n in graph.nodes
        ],
    }


def _int_list(value, length: int, default: int) -> list[int]:
    result = [int(v) for v in value]
    return (result + [default] * length)[:length]


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _stable_profile_hash(profile: dict) -> int:
    data = json.dumps(profile, sort_keys=True).encode("utf-8")
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value
