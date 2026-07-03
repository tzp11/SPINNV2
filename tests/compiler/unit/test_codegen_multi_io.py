"""Tests for multi-input/output codegen support."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from compiler.codegen.c_codegen import SpkInfo, _inspect_spk, _header_text


SPKV2_MAGIC = 0x32564B50
HEADER_STRUCT = struct.Struct("<IHHHHIIIIIIQQQII")
SECTION_STRUCT = struct.Struct("<IIQQII")
TENSOR_STRUCT = struct.Struct("<IHHHH8IQQII")

SECTION_TENSOR_TABLE = 3
SECTION_WEIGHTS = 6
SECTION_CHECKSUM = 12

ROLE_INPUT = 1
ROLE_OUTPUT = 2
ROLE_WEIGHT = 3
ROLE_ACTIVATION = 4


def _make_spk(
    input_sizes: list[int],
    output_sizes: list[int],
    weight_sizes: list[int] | None = None,
) -> bytes:
    """Build a minimal valid SPK binary with the given I/O tensor sizes."""
    if weight_sizes is None:
        weight_sizes = [64]

    tensors: list[tuple[int, int, int]] = []
    for sz in input_sizes:
        tensors.append((ROLE_INPUT, sz, 1))
    for sz in weight_sizes:
        tensors.append((ROLE_WEIGHT, sz, 4))
    for sz in output_sizes:
        tensors.append((ROLE_OUTPUT, sz, 4))

    num_tensors = len(tensors)
    tensor_data = bytearray()
    for i, (role, size_bytes, rank) in enumerate(tensors):
        shape = [1] * 8
        tensor_data.extend(TENSOR_STRUCT.pack(
            i, 1, role, rank, 4,
            *shape,
            size_bytes, 0,
            0, 0,
        ))

    weight_blob = b"\x00" * sum(weight_sizes)
    checksum_data = b"\x00\x00\x00\x00"

    sections = [
        (SECTION_TENSOR_TABLE, bytes(tensor_data), 4),
        (SECTION_WEIGHTS, weight_blob, 16),
        (SECTION_CHECKSUM, checksum_data, 4),
    ]

    section_count = len(sections)
    header_size = HEADER_STRUCT.size
    directory_size = section_count * SECTION_STRUCT.size
    offset = header_size + directory_size

    directory = bytearray()
    payload = bytearray()
    for kind, data, alignment in sections:
        aligned = (offset + alignment - 1) // alignment * alignment
        if aligned > offset:
            payload.extend(b"\x00" * (aligned - offset))
            offset = aligned
        directory.extend(SECTION_STRUCT.pack(kind, 0, offset, len(data), alignment, 0))
        payload.extend(data)
        offset += len(data)

    num_inputs = len(input_sizes)
    num_outputs = len(output_sizes)
    header = HEADER_STRUCT.pack(
        SPKV2_MAGIC, 0, 2, 0, header_size, section_count, 0,
        num_tensors, 0, num_inputs, num_outputs,
        sum(weight_sizes), 1024, 0, 0, 1,
    )

    return bytes(header) + bytes(directory) + bytes(payload)


class TestInspectSpkMultiIO:
    def test_single_io(self):
        spk = _make_spk([100], [200])
        info = _inspect_spk(spk)
        assert info.input_sizes == [100]
        assert info.output_sizes == [200]
        assert info.input_size == 100
        assert info.output_size == 200

    def test_multi_output(self):
        spk = _make_spk([400], [100, 200])
        info = _inspect_spk(spk)
        assert info.input_sizes == [400]
        assert info.output_sizes == [100, 200]
        assert info.input_size == 400
        assert info.output_size == 100

    def test_multi_input_multi_output(self):
        spk = _make_spk([50, 60], [70, 80, 90])
        info = _inspect_spk(spk)
        assert info.input_sizes == [50, 60]
        assert info.output_sizes == [70, 80, 90]

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="at least one input"):
            _make_spk([], [100])
            _inspect_spk(_make_spk([], [100]))

    def test_no_output_raises(self):
        with pytest.raises(ValueError, match="at least one input"):
            _inspect_spk(_make_spk([100], []))


class TestHeaderTextMultiIO:
    def _make_info(self, input_sizes, output_sizes):
        return SpkInfo(
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            activation_arena_bytes=1024,
            scratch_arena_bytes=512,
            checksum=0xDEADBEEF,
        )

    def test_single_io_backward_compat(self):
        info = self._make_info([100], [200])
        header = _header_text("model", info)
        assert "MODEL_INPUT_SIZE_0" in header
        assert "MODEL_OUTPUT_SIZE_0" in header
        assert "MODEL_INPUT_SIZE MODEL_INPUT_SIZE_0" in header
        assert "MODEL_OUTPUT_SIZE MODEL_OUTPUT_SIZE_0" in header
        assert "MODEL_INPUT_COUNT 1" in header
        assert "MODEL_OUTPUT_COUNT 1" in header
        assert "model_run(const void *input, void *output)" in header
        assert "model_run_multi" in header

    def test_multi_output_macros(self):
        info = self._make_info([400], [100, 200])
        header = _header_text("net", info)
        assert "NET_OUTPUT_COUNT 2" in header
        assert "NET_OUTPUT_SIZE_0" in header
        assert "NET_OUTPUT_SIZE_1" in header
        assert "NET_INPUT_COUNT 1" in header

    def test_multi_input_macros(self):
        info = self._make_info([50, 60], [70])
        header = _header_text("m", info)
        assert "M_INPUT_COUNT 2" in header
        assert "M_INPUT_SIZE_0" in header
        assert "M_INPUT_SIZE_1" in header
        assert "M_OUTPUT_COUNT 1" in header

    def test_run_multi_declaration(self):
        info = self._make_info([100], [200])
        header = _header_text("model", info)
        assert "model_run_multi(const void *const *inputs" in header


class TestSpkInfoProperties:
    def test_input_size_property(self):
        info = SpkInfo(input_sizes=[100, 200], output_sizes=[300], activation_arena_bytes=0, scratch_arena_bytes=0, checksum=0)
        assert info.input_size == 100

    def test_output_size_property(self):
        info = SpkInfo(input_sizes=[100], output_sizes=[300, 400], activation_arena_bytes=0, scratch_arena_bytes=0, checksum=0)
        assert info.output_size == 300
