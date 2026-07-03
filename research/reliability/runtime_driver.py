"""ctypes binding for repeated fault-injected execution against the C runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from research.reliability.injection.bitflip import FaultEvent


class CSpkv2FaultEvent(ctypes.Structure):
    _fields_ = [
        ("node_id", ctypes.c_uint32),
        ("tensor_id", ctypes.c_uint32),
        ("element_index", ctypes.c_uint64),
        ("bit_index", ctypes.c_uint8),
        ("invocation_index", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("enabled", ctypes.c_int),
    ]


class CSpkv2ReliabilityStats(ctypes.Structure):
    _fields_ = [
        ("injected_faults", ctypes.c_uint64),
        ("detected_faults", ctypes.c_uint64),
        ("recovered_faults", ctypes.c_uint64),
        ("unrecovered_faults", ctypes.c_uint64),
        ("rerun_count", ctypes.c_uint64),
    ]

class CSpkv2RangeObservation(ctypes.Structure):
    _fields_ = [
        ("node_id", ctypes.c_uint32),
        ("tensor_id", ctypes.c_uint32),
        ("observed_min", ctypes.c_float),
        ("observed_max", ctypes.c_float),
        ("observations", ctypes.c_uint64),
    ]


class RuntimeDriver:
    def __init__(self, library_path: str | Path, spk_path: str | Path) -> None:
        self.lib = ctypes.CDLL(str(library_path))
        self.ctx = ctypes.c_void_p()
        self._configure()
        self._check(self.lib.spkv2_load_file(str(spk_path).encode(), ctypes.byref(self.ctx)), "load")
        self._check(self.lib.spkv2_prepare(self.ctx, None, 0), "prepare")

    def close(self) -> None:
        if self.ctx:
            self.lib.spkv2_free(self.ctx)
            self.ctx = ctypes.c_void_p()

    def __enter__(self) -> "RuntimeDriver":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def run(self, input_data: np.ndarray, event: FaultEvent | None = None) -> tuple[np.ndarray, dict]:
        values = np.ascontiguousarray(input_data, dtype=np.float32)
        self._check(self.lib.spkv2_set_input(self.ctx, 0, values.ctypes.data, values.nbytes), "set input")
        self.lib.spkv2_reset_reliability_stats(self.ctx)
        self.lib.spkv2_reset_range_observations(self.ctx)
        if event is None:
            self.lib.spkv2_clear_fault_event(self.ctx)
        else:
            c_event = CSpkv2FaultEvent(
                event.node_id,
                event.tensor_id,
                event.element_index,
                event.bit_index,
                event.invocation_index,
                event.seed,
                1,
            )
            self._check(self.lib.spkv2_set_fault_event(self.ctx, ctypes.byref(c_event)), "set event")
        self._check(self.lib.spkv2_run(self.ctx), "run")
        size = ctypes.c_size_t()
        self._check(self.lib.spkv2_get_output_size(self.ctx, 0, ctypes.byref(size)), "output size")
        output = np.empty(size.value // 4, dtype=np.float32)
        self._check(self.lib.spkv2_get_output(self.ctx, 0, output.ctypes.data, output.nbytes), "get output")
        stats = CSpkv2ReliabilityStats()
        self._check(self.lib.spkv2_get_reliability_stats(self.ctx, ctypes.byref(stats)), "stats")
        return output, {name: int(getattr(stats, name)) for name, _kind in stats._fields_}

    def range_observations(self, node_ids: list[int]) -> list[dict]:
        records = []
        for node_id in node_ids:
            observation = CSpkv2RangeObservation()
            self._check(
                self.lib.spkv2_get_range_observation(self.ctx, node_id, ctypes.byref(observation)),
                "range observation",
            )
            if observation.observations:
                records.append(
                    {
                        "node_id": int(observation.node_id),
                        "tensor_id": int(observation.tensor_id),
                        "observed_min": float(observation.observed_min),
                        "observed_max": float(observation.observed_max),
                        "observations": int(observation.observations),
                    }
                )
        return records

    def _configure(self) -> None:
        self.lib.spkv2_load_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.spkv2_prepare.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.spkv2_set_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.spkv2_set_fault_event.argtypes = [ctypes.c_void_p, ctypes.POINTER(CSpkv2FaultEvent)]
        self.lib.spkv2_get_output_size.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.spkv2_get_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.spkv2_get_reliability_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(CSpkv2ReliabilityStats)]
        self.lib.spkv2_get_range_observation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(CSpkv2RangeObservation),
        ]
        self.lib.spkv2_free.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _check(code: int, action: str) -> None:
        if code != 0:
            raise RuntimeError(f"SPINNV2 runtime {action} failed with {code}")
