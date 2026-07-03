"""Negative-path tests for the ONNX importer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from compiler.frontend.onnx_importer import import_onnx


def _save_model(model: onnx.ModelProto) -> Path:
    path = Path(tempfile.mktemp(suffix=".onnx"))
    onnx.save(model, str(path))
    return path


def test_unsupported_op_raises():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Erf", ["input"], ["output"])
    graph = helper.make_graph([node], "unsupported_op", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = _save_model(model)
    try:
        with pytest.raises(ValueError, match="unsupported import op: Erf"):
            import_onnx(path)
    finally:
        path.unlink(missing_ok=True)


def test_dynamic_shape_raises():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 3, 8, 8])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "dynamic_shape", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = _save_model(model)
    try:
        with pytest.raises(ValueError, match="dynamic shape"):
            import_onnx(path)
    finally:
        path.unlink(missing_ok=True)


def test_valid_model_imports_successfully():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "valid_model", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = _save_model(model)
    try:
        sir = import_onnx(path)
        assert len(sir.nodes) == 1
        assert sir.nodes[0].op_type == "Relu"
    finally:
        path.unlink(missing_ok=True)
