"""Tests for SPK format 0.3 (version bump + 4KB weight alignment)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from compiler.codegen.c_codegen import HEADER_STRUCT, SECTION_STRUCT, SPKV2_MAGIC
from compiler.frontend.onnx_importer import import_onnx
from compiler.packager.spk_writer import write_spk
from compiler.passes.manager import run_pass_pipeline
from tests.e2e.test_m1_e2e import _write_tiny_cnn

SECTION_WEIGHTS = 6

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROFILE_PATH = _PROJECT_ROOT / "compiler" / "target" / "profiles" / "cpu_generic.json"


def _compile_to_spk(tmp_path: Path) -> bytes:
    model_path = tmp_path / "tiny.onnx"
    spk_path = tmp_path / "tiny.spk"
    _write_tiny_cnn(model_path)
    graph = import_onnx(model_path)
    run_pass_pipeline(graph)
    profile = json.loads(_PROFILE_PATH.read_text())
    write_spk(graph, spk_path, target_profile=profile)
    return spk_path.read_bytes()


def _parse_header(data: bytes) -> dict:
    header = HEADER_STRUCT.unpack_from(data, 0)
    return {
        "magic": header[0],
        "version_major": header[1],
        "version_minor": header[2],
        "header_size": header[4],
        "section_count": header[5],
    }


def _find_section(data: bytes, hdr: dict, kind: int) -> dict | None:
    for i in range(hdr["section_count"]):
        entry_offset = hdr["header_size"] + i * SECTION_STRUCT.size
        k, flags, offset, size, alignment, _ = SECTION_STRUCT.unpack_from(data, entry_offset)
        if k == kind:
            return {"offset": offset, "size": size, "alignment": alignment}
    return None


def test_version_minor_is_3(tmp_path):
    data = _compile_to_spk(tmp_path)
    hdr = _parse_header(data)
    assert hdr["magic"] == SPKV2_MAGIC
    assert hdr["version_major"] == 0
    assert hdr["version_minor"] == 3


def test_weight_section_4k_aligned(tmp_path):
    data = _compile_to_spk(tmp_path)
    hdr = _parse_header(data)
    ws = _find_section(data, hdr, SECTION_WEIGHTS)
    assert ws is not None, "no WEIGHTS section found"
    assert ws["alignment"] == 4096, f"expected alignment=4096, got {ws['alignment']}"
    assert ws["offset"] % 4096 == 0, f"WEIGHTS offset {ws['offset']} is not 4KB aligned"
