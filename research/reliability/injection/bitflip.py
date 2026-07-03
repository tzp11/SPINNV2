"""Deterministic single-bit FP32 fault injection utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np


@dataclass(frozen=True)
class FaultEvent:
    model_id: str
    sample_id: str
    node_id: int
    tensor_id: int
    element_index: int
    bit_index: int
    invocation_index: int = 1
    seed: int = 2026

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FaultEvent":
        return cls(**json.loads(text))


def flip_fp32_bit(values: np.ndarray, element_index: int, bit_index: int, *, copy: bool = True) -> np.ndarray:
    """Flip one IEEE-754 bit in a contiguous FP32 tensor."""
    if values.dtype != np.float32:
        raise TypeError("fault injection requires float32 arrays")
    if bit_index < 0 or bit_index >= 32:
        raise ValueError("bit_index must be in [0, 31]")
    result = np.array(values, dtype=np.float32, copy=copy, order="C")
    flat = result.reshape(-1)
    if element_index < 0 or element_index >= flat.size:
        raise IndexError("element_index is outside tensor bounds")
    bits = flat.view(np.uint32)
    bits[element_index] ^= np.uint32(1 << bit_index)
    return result


def sample_fault_event(
    *,
    model_id: str,
    sample_id: str,
    node_id: int,
    tensor_id: int,
    element_count: int,
    seed: int,
    invocation_index: int = 1,
) -> FaultEvent:
    if element_count <= 0:
        raise ValueError("element_count must be positive")
    rng = np.random.default_rng(seed)
    return FaultEvent(
        model_id=model_id,
        sample_id=sample_id,
        node_id=node_id,
        tensor_id=tensor_id,
        element_index=int(rng.integers(0, element_count)),
        bit_index=int(rng.integers(0, 32)),
        invocation_index=invocation_index,
        seed=seed,
    )
