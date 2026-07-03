"""Convert FP32 weight tensors to FP16 for mixed-precision storage."""

from __future__ import annotations

import numpy as np

from compiler.ir import types
from compiler.ir.graph import Graph


def convert_weights_fp16(graph: Graph) -> int:
    """Convert all FP32 weight tensors to FP16. Returns count converted."""
    count = 0
    for tensor in graph.tensors:
        if (
            tensor.role == types.ROLE_WEIGHT
            and tensor.dtype == types.DTYPE_FP32
            and tensor.data is not None
            and len(tensor.shape) >= 2
        ):
            arr = np.frombuffer(tensor.data, dtype=np.float32)
            tensor.data = arr.astype(np.float16).tobytes()
            tensor.dtype = types.DTYPE_FP16
            count += 1
    return count
