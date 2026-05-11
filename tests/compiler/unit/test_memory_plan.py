from pathlib import Path

import pytest

from compiler.ir.graph import Graph, Node, Tensor
from compiler.ir import types
from compiler.planner.memory_plan import analyze_lifetimes, plan_memory, write_memory_plan_csv


def test_lifetime_extends_graph_outputs():
    graph = _branch_graph()
    lifetimes = analyze_lifetimes(graph)
    graph_end = len(graph.nodes) + 1
    assert lifetimes[1] == (0, 1)
    assert lifetimes[4] == (3, graph_end)


def test_best_fit_reuses_non_overlapping_tensor_memory():
    graph = _branch_graph()
    plan = plan_memory(graph)
    # T1 and T3 have non-overlapping lifetimes and identical sizes.
    assert plan.entries[1].offset == plan.entries[3].offset
    assert plan.planned_activation_bytes < plan.naive_activation_bytes


def test_external_io_is_not_allocated():
    graph = _branch_graph()
    plan = plan_memory(graph, alloc_input=False, alloc_output=False)
    assert plan.entries[0].memory_class == "EXTERNAL"
    assert plan.entries[4].memory_class == "EXTERNAL"
    assert plan.entries[0].offset == 0
    assert plan.entries[4].offset == 0


def test_memory_budget_violation_raises():
    graph = _branch_graph()
    with pytest.raises(MemoryError):
        plan_memory(graph, max_arena_bytes=16)


def test_write_memory_plan_csv(tmp_path: Path):
    graph = _branch_graph()
    plan = plan_memory(graph)
    csv_path = tmp_path / "memory_plan.csv"
    write_memory_plan_csv(plan, csv_path)
    text = csv_path.read_text(encoding="utf-8")
    assert "tensor_id,name,size,aligned_size,first_use,last_use,offset,memory_class" in text
    assert "t4" in text


def _branch_graph() -> Graph:
    graph = Graph(model_name="branch")
    for tensor in [
        Tensor(0, "input", types.DTYPE_FP32, [1, 4], types.ROLE_INPUT),
        Tensor(1, "t1", types.DTYPE_FP32, [1, 4], types.ROLE_ACTIVATION),
        Tensor(2, "t2", types.DTYPE_FP32, [1, 4], types.ROLE_ACTIVATION),
        Tensor(3, "t3", types.DTYPE_FP32, [1, 4], types.ROLE_ACTIVATION),
        Tensor(4, "t4", types.DTYPE_FP32, [1, 4], types.ROLE_OUTPUT),
    ]:
        graph.add_tensor(tensor)
    graph.inputs = [0]
    graph.outputs = [4]
    graph.add_node(Node(0, "Relu", [0], [1]))
    graph.add_node(Node(1, "Relu", [1], [2]))
    graph.add_node(Node(2, "Relu", [2], [3]))
    graph.add_node(Node(3, "Relu", [3], [4]))
    return graph
