"""M1 binary SPK writer."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from compiler.ir.graph import Graph, Node, Tensor
from compiler.ir import types
from compiler.planner.kernel_spec import KernelPlan, KernelSpec, select_kernel_specs
from compiler.planner.memory_plan import MemoryPlan, plan_memory, write_memory_plan_csv
from compiler.reliability.protection_plan import ProtectionPlan


SPKV2_MAGIC = 0x32564B50
VERSION_MAJOR = 0
VERSION_MINOR = 3

SECTION_METADATA = 1
SECTION_TARGET_PROFILE = 2
SECTION_TENSOR_TABLE = 3
SECTION_NODE_TABLE = 4
SECTION_ATTRIBUTES = 5
SECTION_WEIGHTS = 6
SECTION_MEMORY_PLAN = 7
SECTION_KERNEL_SPEC = 8
SECTION_STRING_TABLE = 10
SECTION_QUANTIZATION = 9
SECTION_CHECKSUM = 12
SECTION_PROTECTION_PLAN = 13

DTYPE_CODES = {types.DTYPE_FP32: 1, types.DTYPE_INT8: 2, types.DTYPE_FP16: 3}
LAYOUT_CODES = {"NCHW": 1}
WEIGHT_LAYOUT_CODES = {"OIHW": 1}
BACKEND_CODES = {"ref": 1, "cpu": 2, "simd": 3, "delegate": 4}
KERNEL_KIND_CODES = {
    "reference": 1,
    "direct": 2,
    "im2col_gemm": 3,
    "packed_gemm": 4,
    "pointwise_1x1": 5,
    "depthwise_direct": 6,
    "winograd_3x3s1": 7,
    "conv3x3s2_direct": 8,
    "winograd_f43": 9,
    "int8_im2col_gemm": 10,
    "bnns_fp32": 11,
}
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
    "EXTERNAL": 5,
}
OP_CODES = {
    "Add": 1,
    "Conv": 2,
    "Flatten": 3,
    "Gemm": 4,
    "MaxPool": 5,
    "Relu": 6,
    "Softmax": 7,
    "Mul": 8,
    "Sub": 9,
    "Div": 10,
    "Mod": 11,
    "Sigmoid": 12,
    "Reshape": 13,
    "Transpose": 14,
    "Concat": 15,
    "Split": 16,
    "ReduceMean": 17,
    "ReduceMax": 18,
    "MatMul": 19,
    "Resize": 20,
    "Tile": 21,
    "Unsqueeze": 22,
    "GatherElements": 23,
    "TopK": 24,
    "Cast": 25,
    "Slice": 26,
    "Gather": 27,
    "AveragePool": 28,
    "GlobalAveragePool": 29,
    "Clip": 30,
    "LeakyRelu": 31,
}

HEADER_STRUCT = struct.Struct("<IHHHHIIIIIIQQQII")
SECTION_STRUCT = struct.Struct("<IIQQII")
TENSOR_STRUCT = struct.Struct("<IHHHH8IQQII")
NODE_STRUCT = struct.Struct("<IHHHH8I4IIIII")
ATTR_STRUCT = struct.Struct("<Iiii4i2i2i2i2i2fii8i3i")
MEMORY_PLAN_STRUCT = struct.Struct("<IHHIQQII")
KERNEL_SPEC_STRUCT = struct.Struct("<IIHHHHHHQQII")
QUANT_PARAM_STRUCT = struct.Struct("<IHHI Q")  # tensor_id(u32), scheme(u16), quant_axis(u16), num_channels(u32), data_offset(u64)


def write_spk(
    graph: Graph,
    out_path: str | Path,
    target_profile: dict,
    *,
    memory_plan: MemoryPlan | None = None,
    kernel_plan: KernelPlan | None = None,
    memory_plan_csv: str | Path | None = None,
    quant_params: list | None = None,
    protection_plan: ProtectionPlan | None = None,
) -> None:
    out_path = Path(out_path)
    _validate_runtime_ops(graph)
    sections: list[tuple[int, bytes, int]] = []
    if kernel_plan is None:
        kernel_plan = select_kernel_specs(graph, target_profile)
    if memory_plan is None:
        memory_plan = plan_memory(
            graph,
            max_arena_bytes=int(target_profile["memory"]["activation_arena_max"]),
        )
    if memory_plan_csv is not None:
        write_memory_plan_csv(memory_plan, memory_plan_csv)

    strings = _build_string_table(graph)
    weights, weight_offsets = _build_weights(graph)
    attrs, attr_offsets = _build_attrs(graph)

    sections.append((SECTION_METADATA, _metadata_bytes(graph), 1))
    sections.append((SECTION_TARGET_PROFILE, json.dumps(target_profile, sort_keys=True).encode("utf-8"), 1))
    sections.append((SECTION_TENSOR_TABLE, _tensor_table_bytes(graph, strings.offsets, weight_offsets, memory_plan), 4))
    sections.append((SECTION_NODE_TABLE, _node_table_bytes(graph, attr_offsets, kernel_plan), 4))
    sections.append((SECTION_ATTRIBUTES, attrs, 4))
    sections.append((SECTION_WEIGHTS, weights, 4096))
    sections.append((SECTION_MEMORY_PLAN, _memory_plan_bytes(graph, memory_plan), 4))
    sections.append((SECTION_KERNEL_SPEC, _kernel_spec_bytes(kernel_plan.specs), 4))
    if quant_params:
        sections.append((SECTION_QUANTIZATION, _quantization_section_bytes(quant_params), 4))
    if protection_plan is not None:
        protection_plan.validated_for_graph(graph)
        prot_blob = protection_plan.to_bytes(graph, scratch_offset=kernel_plan.scratch_arena_bytes)
        sections.append((SECTION_PROTECTION_PLAN, prot_blob, 8))
    sections.append((SECTION_STRING_TABLE, strings.blob, 1))
    sections.append((SECTION_CHECKSUM, b"\x00\x00\x00\x00", 4))

    section_count = len(sections)
    header_size = HEADER_STRUCT.size
    directory_size = section_count * SECTION_STRUCT.size
    offset = header_size + directory_size
    directory = bytearray()
    payload = bytearray()
    checksum_offset = 0

    for kind, data, alignment in sections:
        aligned_offset = _align(offset, alignment)
        if aligned_offset > offset:
            payload.extend(b"\x00" * (aligned_offset - offset))
            offset = aligned_offset
        if kind == SECTION_CHECKSUM:
            checksum_offset = offset
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
        memory_plan.planned_activation_bytes,
        kernel_plan.scratch_arena_bytes + (protection_plan.scratch_bytes(graph) if protection_plan else 0),
        _stable_profile_hash(target_profile),
        1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    package = bytearray(header + bytes(directory) + bytes(payload))
    checksum = _fnv1a32(package[:checksum_offset])
    package[checksum_offset : checksum_offset + 4] = struct.pack("<I", checksum)
    out_path.write_bytes(package)
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(_debug_json(graph, memory_plan), ensure_ascii=False, indent=2),
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
    beta = float(attrs.get("beta", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    fused_activation = {"Relu": 1, "Silu": 2}.get(attrs.get("fused_activation"), 0)
    extra_values: list[int] = []
    if node.op_type == "Transpose":
        extra_values = [int(v) for v in attrs.get("perm", [])]
    elif node.op_type in {"ReduceMean", "ReduceMax"} and "axes" in attrs:
        extra_values = [int(v) for v in attrs.get("axes", [])]
    extra = (extra_values + [0] * 8)[:8]
    keepdims = int(attrs.get("keepdims", 1))
    largest = int(attrs.get("largest", 1))
    sorted_attr = int(attrs.get("sorted", 1))
    cast_to = int(attrs.get("to", 1))
    return ATTR_STRUCT.pack(
        OP_CODES[node.op_type],
        axis,
        group,
        fused_activation,
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
        beta,
        len(extra_values),
        keepdims,
        *extra,
        largest,
        sorted_attr,
        cast_to,
    )


def _tensor_table_bytes(
    graph: Graph,
    string_offsets: dict[str, int],
    weight_offsets: dict[int, int],
    memory_plan: MemoryPlan,
) -> bytes:
    blob = bytearray()
    for tensor in graph.tensors:
        shape = tensor.shape[:8] + [1] * (8 - len(tensor.shape))
        plan_entry = memory_plan.entries[tensor.id]
        data_offset = weight_offsets.get(tensor.id, plan_entry.offset)
        memory_class = _memory_class_code(plan_entry.memory_class)
        blob.extend(
            TENSOR_STRUCT.pack(
                tensor.id,
                DTYPE_CODES[tensor.dtype],
                ROLE_CODES[tensor.role],
                len(tensor.shape),
                memory_class,
                *shape,
                tensor.size_bytes,
                data_offset,
                string_offsets[tensor.name],
                0,
            )
        )
    return bytes(blob)


def _memory_plan_bytes(graph: Graph, memory_plan: MemoryPlan) -> bytes:
    blob = bytearray()
    for tensor in graph.tensors:
        entry = memory_plan.entries[tensor.id]
        blob.extend(
            MEMORY_PLAN_STRUCT.pack(
                entry.tensor_id,
                _memory_class_code(entry.memory_class),
                16,
                0,
                entry.offset,
                entry.size,
                entry.first_use if entry.first_use >= 0 else 0xFFFFFFFF,
                entry.last_use if entry.last_use >= 0 else 0xFFFFFFFF,
            )
        )
    return bytes(blob)


def _kernel_spec_bytes(specs: list[KernelSpec]) -> bytes:
    blob = bytearray()
    for spec in specs:
        blob.extend(
            KERNEL_SPEC_STRUCT.pack(
                spec.id,
                spec.node_id,
                KERNEL_KIND_CODES[spec.kernel_kind],
                BACKEND_CODES[spec.backend],
                DTYPE_CODES[spec.dtype],
                LAYOUT_CODES[spec.layout],
                WEIGHT_LAYOUT_CODES[spec.weight_layout],
                0,
                spec.scratch_offset,
                spec.scratch_bytes,
                spec.fallback_kernel_spec_id,
                _feature_mask(spec.required_features or []),
            )
        )
    return bytes(blob)


def _node_table_bytes(
    graph: Graph,
    attr_offsets: dict[int, tuple[int, int]],
    kernel_plan: KernelPlan,
) -> bytes:
    blob = bytearray()
    for node in graph.nodes:
        inputs = node.inputs[:8] + [0] * (8 - len(node.inputs))
        outputs = node.outputs[:4] + [0] * (4 - len(node.outputs))
        attr_offset, attr_size = attr_offsets[node.id]
        kernel_spec = kernel_plan.by_node[node.id]
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
                kernel_spec.id,
                kernel_spec.scratch_bytes,
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


def _debug_json(graph: Graph, memory_plan: MemoryPlan) -> dict:
    return {
        "model_name": graph.model_name,
        "inputs": graph.inputs,
        "outputs": graph.outputs,
        "metadata": graph.metadata,
        "memory": {
            "naive_activation_bytes": memory_plan.naive_activation_bytes,
            "planned_activation_bytes": memory_plan.planned_activation_bytes,
            "memory_reduction_ratio": memory_plan.memory_reduction_ratio,
            "alloc_input": memory_plan.alloc_input,
            "alloc_output": memory_plan.alloc_output,
        },
        "tensors": [
            {
                "id": t.id,
                "name": t.name,
                "shape": t.shape,
                "role": t.role,
                "size_bytes": t.size_bytes,
                "memory": memory_plan.entries[t.id].__dict__,
            }
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


def _memory_class_code(memory_class: str) -> int:
    if memory_class == "INPUT":
        return 1
    if memory_class == "OUTPUT":
        return 2
    if memory_class == "WEIGHT":
        return 3
    if memory_class == "ACTIVATION_ARENA":
        return 4
    if memory_class == "EXTERNAL":
        return 5
    raise ValueError(f"unknown memory class: {memory_class}")


def _stable_profile_hash(profile: dict) -> int:
    data = json.dumps(profile, sort_keys=True).encode("utf-8")
    return _fnv1a32(data)


def _fnv1a32(data: bytes | bytearray) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _validate_runtime_ops(graph: Graph) -> None:
    unsupported = sorted({node.op_type for node in graph.nodes if node.op_type not in OP_CODES})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"graph still contains unsupported runtime op(s): {names}")


def _feature_mask(features: list[str]) -> int:
    mask = 0
    if types.DTYPE_FP32 in features:
        mask |= 1
    if types.DTYPE_INT8 in features:
        mask |= 2
    if types.DTYPE_FP16 in features:
        mask |= 4
    return mask


_QUANT_SCHEME_CODES = {"per_tensor": 1, "per_channel": 2}


def _quantization_section_bytes(quant_params: list) -> bytes:
    """Serialize SECTION_QUANTIZATION: array of Spkv2QuantParamRecord
    followed by scale/zero_point float arrays."""
    import struct as _struct

    record_count = len(quant_params)
    records_size = record_count * QUANT_PARAM_STRUCT.size

    data_blob = bytearray()
    data_offsets: list[int] = []

    for qp in quant_params:
        offset = records_size + len(data_blob)
        data_offsets.append(offset)
        for s in qp.scales:
            data_blob.extend(_struct.pack("<f", s))
        for zp in qp.zero_points:
            data_blob.extend(_struct.pack("<i", zp))

    records = bytearray()
    for i, qp in enumerate(quant_params):
        records.extend(QUANT_PARAM_STRUCT.pack(
            qp.tensor_id,
            _QUANT_SCHEME_CODES.get(qp.scheme, 0),
            qp.quant_axis,
            qp.num_channels,
            data_offsets[i],
        ))

    return bytes(records + data_blob)
