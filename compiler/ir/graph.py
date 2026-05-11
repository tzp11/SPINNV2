"""Minimal SIR graph model for M1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compiler.ir import types


@dataclass
class Tensor:
    id: int
    name: str
    dtype: str
    shape: list[int]
    role: str
    layout: str = types.LAYOUT_NCHW
    producer: int | None = None
    consumers: list[int] = field(default_factory=list)
    data: bytes | None = None

    @property
    def elem_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def size_bytes(self) -> int:
        if self.dtype != types.DTYPE_FP32:
            raise ValueError(f"unsupported dtype in M1: {self.dtype}")
        return self.elem_count * 4


@dataclass
class Node:
    id: int
    op_type: str
    inputs: list[int]
    outputs: list[int]
    attrs: dict[str, Any] = field(default_factory=dict)
    execution_index: int = 0


@dataclass
class Graph:
    model_name: str
    tensors: list[Tensor] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def tensor(self, tensor_id: int) -> Tensor:
        return self.tensors[tensor_id]

    def add_tensor(self, tensor: Tensor) -> None:
        if tensor.id != len(self.tensors):
            raise ValueError("M1 tensor IDs must be dense and ordered")
        self.tensors.append(tensor)

    def add_node(self, node: Node) -> None:
        if node.id != len(self.nodes):
            raise ValueError("M1 node IDs must be dense and ordered")
        self.nodes.append(node)
        for input_id in node.inputs:
            self.tensors[input_id].consumers.append(node.id)
        for output_id in node.outputs:
            self.tensors[output_id].producer = node.id

