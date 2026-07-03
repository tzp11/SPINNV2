"""Run an SPK package integrity boundary experiment.

The experiment flips one bit in a non-checksum section of an SPK file and then
tries to load/run it through the runtime CLI. A non-zero runtime exit is the
expected outcome because the SPK loader should reject the corrupted package via
its checksum boundary.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEADER_STRUCT = struct.Struct("<IHHHHIIIIQQQII")
SECTION_STRUCT = struct.Struct("<IIQQII")
SPKV2_MAGIC = 0x32564B50
SECTION_CHECKSUM = 12
SECTION_WEIGHTS = 6


def _sections(data: bytes) -> list[dict[str, int]]:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("SPK file is smaller than the fixed header.")
    fields = HEADER_STRUCT.unpack_from(data, 0)
    magic = fields[0]
    if magic != SPKV2_MAGIC:
        raise ValueError(f"Invalid SPK magic: 0x{magic:08x}")
    header_size = fields[4]
    section_count = fields[5]
    offset = header_size
    sections = []
    for _ in range(section_count):
        if offset + SECTION_STRUCT.size > len(data):
            raise ValueError("Section table extends past end of file.")
        kind, flags, sec_offset, size, alignment, reserved = SECTION_STRUCT.unpack_from(data, offset)
        sections.append(
            {
                "kind": kind,
                "flags": flags,
                "offset": sec_offset,
                "size": size,
                "alignment": alignment,
                "reserved": reserved,
            }
        )
        offset += SECTION_STRUCT.size
    return sections


def _choose_flip(sections: list[dict[str, int]]) -> tuple[int, dict[str, int]]:
    candidates = [s for s in sections if s["kind"] == SECTION_WEIGHTS and s["size"] > 0]
    if not candidates:
        candidates = [s for s in sections if s["kind"] != SECTION_CHECKSUM and s["size"] > 0]
    if not candidates:
        raise ValueError("No non-checksum section can be corrupted.")
    section = max(candidates, key=lambda item: item["size"])
    return int(section["offset"] + section["size"] // 2), section


def run_experiment(spk: Path, runner: Path, input_bin: Path, out_dir: Path, bit_index: int) -> dict[str, Any]:
    data = bytearray(spk.read_bytes())
    sections = _sections(data)
    byte_offset, section = _choose_flip(sections)
    if byte_offset >= len(data):
        raise ValueError("Chosen byte offset is outside file.")

    before = data[byte_offset]
    data[byte_offset] ^= 1 << bit_index
    after = data[byte_offset]

    out_dir.mkdir(parents=True, exist_ok=True)
    corrupt_spk = out_dir / f"{spk.stem}_corrupt_bit{bit_index}.spk"
    output_bin = out_dir / f"{spk.stem}_corrupt_output.bin"
    corrupt_spk.write_bytes(data)

    cmd = [str(runner), str(corrupt_spk), str(input_bin), str(output_bin)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_spk": str(spk),
        "corrupt_spk": str(corrupt_spk),
        "runtime_runner": str(runner),
        "input_bin": str(input_bin),
        "flipped_section": section,
        "byte_offset": byte_offset,
        "bit_index": bit_index,
        "byte_before": before,
        "byte_after": after,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "checksum_rejected": proc.returncode != 0,
        "interpretation": (
            "loader rejected corrupted SPK as expected"
            if proc.returncode != 0
            else "corrupted SPK unexpectedly loaded and ran"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spk", required=True, help="Source SPK package.")
    parser.add_argument("--runner", required=True, help="spkv2_run executable.")
    parser.add_argument("--input-bin", required=True, help="Input tensor binary for the model.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/reports/boundary_experiments",
        help="Output directory.",
    )
    parser.add_argument("--bit-index", type=int, default=0, choices=range(8), help="Bit index inside the selected byte.")
    parser.add_argument("--report", default="spk_integrity_boundary.json", help="Report filename.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    result = run_experiment(
        spk=Path(args.spk),
        runner=Path(args.runner),
        input_bin=Path(args.input_bin),
        out_dir=out_dir,
        bit_index=args.bit_index,
    )
    report = out_dir / args.report
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report), "checksum_rejected": result["checksum_rejected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
