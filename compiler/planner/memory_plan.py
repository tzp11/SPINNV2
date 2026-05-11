"""Static activation memory planning for SPINNV2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compiler.ir.graph import Graph, Tensor
from compiler.ir import types


ALIGNMENT = 16
GRAPH_END_EXTRA = 1


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
    planned_tensors = [
        tensor
        for tensor in graph.tensors
        if _should_allocate(tensor, alloc_input=alloc_input, alloc_output=alloc_output)
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

