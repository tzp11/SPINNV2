"""Static activation memory planning for SPINNV2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compiler.ir.graph import Graph, Tensor
from compiler.ir import types


ALIGNMENT = 16
GRAPH_END_EXTRA = 1

INPLACE_SHAPE_ONLY_OPS = {"Reshape", "Flatten", "Unsqueeze", "Cast"}
INPLACE_ELEMENTWISE_OPS = {"Relu"}


@dataclass
class MemoryPlanEntry:
    tensor_id: int
    name: str
    size: int
    aligned_size: int
    first_use: int
    last_use: int
    offset: int
    memory_class: str


@dataclass
class MemoryPlan:
    entries: dict[int, MemoryPlanEntry]
    naive_activation_bytes: int
    planned_activation_bytes: int
    alloc_input: bool = True
    alloc_output: bool = True

    @property
    def memory_reduction_ratio(self) -> float:
        if self.naive_activation_bytes == 0:
            return 0.0
        return 1.0 - (self.planned_activation_bytes / self.naive_activation_bytes)


def plan_memory(
    graph: Graph,
    *,
    max_arena_bytes: int = 0,
    alloc_input: bool = True,
    alloc_output: bool = True,
) -> MemoryPlan:
    lifetimes = analyze_lifetimes(graph)
    inplace_map = _find_inplace_aliases(graph, alloc_input, alloc_output)
    concat_map = _find_concat_aliases(graph, alloc_input, alloc_output)

    for alias_tid, src_tid in inplace_map.items():
        src_first, src_last = lifetimes[src_tid]
        alias_first, alias_last = lifetimes[alias_tid]
        lifetimes[src_tid] = (min(src_first, alias_first), max(src_last, alias_last))

    for alias_tid, (parent_tid, _byte_off) in concat_map.items():
        parent_first, parent_last = lifetimes[parent_tid]
        alias_first, alias_last = lifetimes[alias_tid]
        lifetimes[parent_tid] = (min(parent_first, alias_first), max(parent_last, alias_last))

    planned_tensors = [
        tensor
        for tensor in graph.tensors
        if _should_allocate(tensor, alloc_input=alloc_input, alloc_output=alloc_output)
        and tensor.id not in inplace_map
        and tensor.id not in concat_map
    ]

    naive = sum(_align(tensor.size_bytes) for tensor in planned_tensors)
    entries: dict[int, MemoryPlanEntry] = {}
    free_blocks: list[tuple[int, int]] = []
    active: list[tuple[int, int, int]] = []  # tensor_id, offset, size
    peak = 0

    ordered = sorted(planned_tensors, key=lambda t: (lifetimes[t.id][0], t.id))
    for tensor in ordered:
        first, _last = lifetimes[tensor.id]
        still_active: list[tuple[int, int, int]] = []
        for active_tensor_id, active_offset, active_size in active:
            active_last = lifetimes[active_tensor_id][1]
            if active_last < first:
                _insert_free_block(free_blocks, active_offset, active_size)
            else:
                still_active.append((active_tensor_id, active_offset, active_size))
        active = still_active

        aligned_size = _align(tensor.size_bytes)
        offset = _alloc_best_fit(free_blocks, aligned_size)
        if offset is None:
            offset = peak
            peak += aligned_size
        active.append((tensor.id, offset, aligned_size))
        first_use, last_use = lifetimes[tensor.id]
        entries[tensor.id] = MemoryPlanEntry(
            tensor_id=tensor.id,
            name=tensor.name,
            size=tensor.size_bytes,
            aligned_size=aligned_size,
            first_use=first_use,
            last_use=last_use,
            offset=offset,
            memory_class=_memory_class(tensor, alloc_input=alloc_input, alloc_output=alloc_output),
        )

    for tensor in graph.tensors:
        if tensor.id in entries:
            continue
        if tensor.id in inplace_map:
            src_tid = inplace_map[tensor.id]
            src_entry = entries[src_tid]
            first_use, last_use = lifetimes.get(tensor.id, (-1, -1))
            entries[tensor.id] = MemoryPlanEntry(
                tensor_id=tensor.id,
                name=tensor.name,
                size=tensor.size_bytes,
                aligned_size=_align(tensor.size_bytes),
                first_use=first_use,
                last_use=last_use,
                offset=src_entry.offset,
                memory_class=_memory_class(tensor, alloc_input=alloc_input, alloc_output=alloc_output),
            )
            continue
        if tensor.id in concat_map:
            parent_tid, byte_offset = concat_map[tensor.id]
            parent_entry = entries[parent_tid]
            first_use, last_use = lifetimes.get(tensor.id, (-1, -1))
            entries[tensor.id] = MemoryPlanEntry(
                tensor_id=tensor.id,
                name=tensor.name,
                size=tensor.size_bytes,
                aligned_size=_align(tensor.size_bytes),
                first_use=first_use,
                last_use=last_use,
                offset=parent_entry.offset + byte_offset,
                memory_class=_memory_class(tensor, alloc_input=alloc_input, alloc_output=alloc_output),
            )
            continue
        first_use, last_use = lifetimes.get(tensor.id, (-1, -1))
        entries[tensor.id] = MemoryPlanEntry(
            tensor_id=tensor.id,
            name=tensor.name,
            size=tensor.size_bytes,
            aligned_size=_align(tensor.size_bytes),
            first_use=first_use,
            last_use=last_use,
            offset=0,
            memory_class=_memory_class(tensor, alloc_input=alloc_input, alloc_output=alloc_output),
        )

    if max_arena_bytes > 0 and peak > max_arena_bytes:
        raise MemoryError(
            f"planned activation arena {peak} exceeds target limit {max_arena_bytes}"
        )

    return MemoryPlan(
        entries=entries,
        naive_activation_bytes=naive,
        planned_activation_bytes=peak,
        alloc_input=alloc_input,
        alloc_output=alloc_output,
    )


def analyze_lifetimes(graph: Graph) -> dict[int, tuple[int, int]]:
    graph_end = len(graph.nodes) + GRAPH_END_EXTRA
    lifetimes: dict[int, tuple[int, int]] = {}
    output_ids = set(graph.outputs)

    for tensor in graph.tensors:
        if tensor.role == types.ROLE_WEIGHT:
            lifetimes[tensor.id] = (-1, graph_end)
            continue

        if tensor.producer is None:
            first = 0
        else:
            first = tensor.producer

        if tensor.id in output_ids:
            last = graph_end
        elif tensor.consumers:
            last = max(tensor.consumers)
        else:
            last = first

        lifetimes[tensor.id] = (first, last)

    return lifetimes


def write_memory_plan_csv(plan: MemoryPlan, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["tensor_id,name,size,aligned_size,first_use,last_use,offset,memory_class"]
    for tensor_id in sorted(plan.entries):
        entry = plan.entries[tensor_id]
        lines.append(
            f"{entry.tensor_id},{entry.name},{entry.size},{entry.aligned_size},"
            f"{entry.first_use},{entry.last_use},{entry.offset},{entry.memory_class}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _should_allocate(tensor: Tensor, *, alloc_input: bool, alloc_output: bool) -> bool:
    if tensor.role in {types.ROLE_WEIGHT, types.ROLE_CONSTANT}:
        return False
    if tensor.role == types.ROLE_INPUT and not alloc_input:
        return False
    if tensor.role == types.ROLE_OUTPUT and not alloc_output:
        return False
    return tensor.size_bytes > 0


def _memory_class(tensor: Tensor, *, alloc_input: bool, alloc_output: bool) -> str:
    if tensor.role == types.ROLE_WEIGHT:
        return "WEIGHT"
    if tensor.role == types.ROLE_CONSTANT:
        return "WEIGHT"
    if tensor.role == types.ROLE_INPUT and not alloc_input:
        return "EXTERNAL"
    if tensor.role == types.ROLE_OUTPUT and not alloc_output:
        return "EXTERNAL"
    if tensor.role == types.ROLE_INPUT:
        return "INPUT"
    if tensor.role == types.ROLE_OUTPUT:
        return "OUTPUT"
    return "ACTIVATION_ARENA"


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _alloc_best_fit(free_blocks: list[tuple[int, int]], size: int) -> int | None:
    best_index: int | None = None
    best_size: int | None = None
    for i, (_offset, block_size) in enumerate(free_blocks):
        if block_size >= size and (best_size is None or block_size < best_size):
            best_index = i
            best_size = block_size
    if best_index is None:
        return None

    offset, block_size = free_blocks.pop(best_index)
    if block_size > size:
        _insert_free_block(free_blocks, offset + size, block_size - size)
    return offset


def _insert_free_block(free_blocks: list[tuple[int, int]], offset: int, size: int) -> None:
    if size <= 0:
        return
    free_blocks.append((offset, size))
    free_blocks.sort()

    merged: list[tuple[int, int]] = []
    for block_offset, block_size in free_blocks:
        if not merged:
            merged.append((block_offset, block_size))
            continue
        prev_offset, prev_size = merged[-1]
        if prev_offset + prev_size == block_offset:
            merged[-1] = (prev_offset, prev_size + block_size)
        else:
            merged.append((block_offset, block_size))
    free_blocks[:] = merged


def _find_inplace_aliases(
    graph: Graph, alloc_input: bool, alloc_output: bool
) -> dict[int, int]:
    aliases: dict[int, int] = {}
    output_ids = set(graph.outputs)
    input_ids = set(graph.inputs)
    for node in graph.nodes:
        is_shape_only = node.op_type in INPLACE_SHAPE_ONLY_OPS
        is_elementwise = node.op_type in INPLACE_ELEMENTWISE_OPS
        if not is_shape_only and not is_elementwise:
            continue
        if len(node.inputs) < 1 or len(node.outputs) < 1:
            continue
        in_tid = node.inputs[0]
        out_tid = node.outputs[0]
        in_tensor = graph.tensors[in_tid]
        out_tensor = graph.tensors[out_tid]
        if in_tensor.size_bytes != out_tensor.size_bytes:
            continue
        if not _should_allocate(out_tensor, alloc_input=alloc_input, alloc_output=alloc_output):
            continue
        if not _should_allocate(in_tensor, alloc_input=alloc_input, alloc_output=alloc_output):
            continue
        if out_tid in output_ids and in_tid in input_ids:
            continue
        if is_elementwise and len(in_tensor.consumers) > 1:
            continue
        src = aliases.get(in_tid, in_tid)
        if is_elementwise and src in input_ids:
            continue
        aliases[out_tid] = src
    return aliases


def _find_concat_aliases(
    graph: Graph, alloc_input: bool, alloc_output: bool
) -> dict[int, tuple[int, int]]:
    """Find Concat inputs that can be sub-region aliases of the Concat output.

    Returns {input_tensor_id: (concat_output_tensor_id, byte_offset)}.
    """
    aliases: dict[int, tuple[int, int]] = {}
    input_ids = set(graph.inputs)
    used_as_concat_alias: set[int] = set()

    for node in graph.nodes:
        if node.op_type != "Concat":
            continue
        if len(node.inputs) < 2 or len(node.outputs) < 1:
            continue

        out_tid = node.outputs[0]
        out_tensor = graph.tensors[out_tid]
        if not _should_allocate(out_tensor, alloc_input=alloc_input, alloc_output=alloc_output):
            continue

        axis = int(node.attrs.get("axis", 0))
        rank = len(out_tensor.shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            continue

        outer = 1
        for i in range(axis):
            outer *= out_tensor.shape[i]
        if outer != 1:
            continue

        inner = 1
        for i in range(axis + 1, rank):
            inner *= out_tensor.shape[i]

        eligible = True
        axis_offset = 0
        input_offsets: list[tuple[int, int]] = []

        for in_tid in node.inputs:
            in_tensor = graph.tensors[in_tid]
            if not _should_allocate(in_tensor, alloc_input=alloc_input, alloc_output=alloc_output):
                eligible = False
                break
            if in_tid in input_ids:
                eligible = False
                break
            if in_tid in used_as_concat_alias:
                eligible = False
                break

            byte_offset = axis_offset * inner * 4
            if byte_offset % ALIGNMENT != 0:
                eligible = False
                break

            input_offsets.append((in_tid, byte_offset))
            axis_offset += in_tensor.shape[axis]

        if not eligible:
            continue

        for in_tid, byte_offset in input_offsets:
            aliases[in_tid] = (out_tid, byte_offset)
            used_as_concat_alias.add(in_tid)

    return aliases

