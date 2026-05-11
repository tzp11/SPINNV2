"""Generate static C deployment wrappers from an SPK package."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path


HEADER_STRUCT = struct.Struct("<IHHHHIIIIIIQQQII")
SECTION_STRUCT = struct.Struct("<IIQQII")
TENSOR_STRUCT = struct.Struct("<IHHHH8IQQII")

SPKV2_MAGIC = 0x32564B50
SECTION_TENSOR_TABLE = 3
ROLE_INPUT = 1
ROLE_OUTPUT = 2


@dataclass
class SpkInfo:
    input_size: int
    output_size: int
    activation_arena_bytes: int
    scratch_arena_bytes: int
    checksum: int


def generate_c_from_spk(
    spk_path: str | Path,
    out_dir: str | Path,
    *,
    name: str = "model",
    runtime_dir: str | Path = "runtime",
) -> None:
    spk_path = Path(spk_path)
    out_dir = Path(out_dir)
    symbol = _sanitize_symbol(name)
    data = spk_path.read_bytes()
    info = _inspect_spk(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{symbol}.h").write_text(_header_text(symbol, info), encoding="utf-8")
    (out_dir / f"{symbol}.c").write_text(_source_text(symbol, data, info), encoding="utf-8")
    (out_dir / "main_test.c").write_text(_main_test_text(symbol), encoding="utf-8")
    (out_dir / "CMakeLists.txt").write_text(
        _cmake_text(symbol, Path(runtime_dir).resolve()),
        encoding="utf-8",
    )


def _inspect_spk(data: bytes) -> SpkInfo:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("SPK file is smaller than header")
    header = HEADER_STRUCT.unpack_from(data, 0)
    if header[0] != SPKV2_MAGIC:
        raise ValueError("invalid SPK magic")
    header_size = header[4]
    section_count = header[5]
    num_tensors = header[7]
    activation_arena_bytes = header[12]
    scratch_arena_bytes = header[13]

    tensor_offset = None
    tensor_size = None
    for i in range(section_count):
        entry_offset = header_size + i * SECTION_STRUCT.size
        kind, _flags, offset, size, _alignment, _reserved = SECTION_STRUCT.unpack_from(data, entry_offset)
        if kind == SECTION_TENSOR_TABLE:
            tensor_offset = offset
            tensor_size = size
            break
    if tensor_offset is None or tensor_size is None:
        raise ValueError("SPK missing tensor table")
    if tensor_size < num_tensors * TENSOR_STRUCT.size:
        raise ValueError("SPK tensor table is truncated")

    input_size = None
    output_size = None
    for i in range(num_tensors):
        record = TENSOR_STRUCT.unpack_from(data, tensor_offset + i * TENSOR_STRUCT.size)
        role = record[2]
        size_bytes = record[13]
        if role == ROLE_INPUT and input_size is None:
            input_size = size_bytes
        if role == ROLE_OUTPUT and output_size is None:
            output_size = size_bytes
    if input_size is None or output_size is None:
        raise ValueError("SPK must contain at least one input and one output")

    return SpkInfo(
        input_size=int(input_size),
        output_size=int(output_size),
        activation_arena_bytes=int(activation_arena_bytes),
        scratch_arena_bytes=int(scratch_arena_bytes),
        checksum=_fnv1a32(data),
    )


def _header_text(symbol: str, info: SpkInfo) -> str:
    guard = f"{symbol.upper()}_H"
    return f"""#ifndef {guard}
#define {guard}

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define {symbol.upper()}_INPUT_SIZE ((size_t){info.input_size}u)
#define {symbol.upper()}_OUTPUT_SIZE ((size_t){info.output_size}u)
#define {symbol.upper()}_ACTIVATION_ARENA_SIZE ((size_t){max(info.activation_arena_bytes, 1)}u)
#define {symbol.upper()}_SCRATCH_ARENA_SIZE ((size_t){max(info.scratch_arena_bytes, 1)}u)
#define {symbol.upper()}_SPK_CHECKSUM 0x{info.checksum:08x}u

int {symbol}_init(void);
int {symbol}_run(const void *input, void *output);
int {symbol}_run_checked(const void *input, size_t input_size, void *output, size_t output_size);
int {symbol}_verify_checksum(const void *data, size_t size);
void {symbol}_free(void);

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""


def _source_text(symbol: str, data: bytes, info: SpkInfo) -> str:
    bytes_literal = _bytes_literal(data)
    activation_size = max(info.activation_arena_bytes, 1)
    scratch_size = max(info.scratch_arena_bytes, 1)
    return f"""#include "{symbol}.h"

#include "spkv2_runtime.h"

#include <stdint.h>

static const unsigned char g_{symbol}_spk[] = {{
{bytes_literal}
}};

static unsigned char g_{symbol}_activation_arena[{activation_size}];
static unsigned char g_{symbol}_scratch_arena[{scratch_size}];
static Spkv2Context *g_{symbol}_ctx;

static uint32_t {symbol}_fnv1a32(const unsigned char *data, size_t size) {{
    uint32_t value = 2166136261u;
    for (size_t i = 0; i < size; i++) {{
        value ^= data[i];
        value *= 16777619u;
    }}
    return value;
}}

int {symbol}_verify_checksum(const void *data, size_t size) {{
    if (!data || size == 0) return -1;
    return {symbol}_fnv1a32((const unsigned char *)data, size) == {symbol.upper()}_SPK_CHECKSUM ? 0 : -2;
}}

int {symbol}_init(void) {{
    if (g_{symbol}_ctx) return 0;
    if ({symbol}_verify_checksum(g_{symbol}_spk, sizeof(g_{symbol}_spk)) != 0) return -10;
    int rc = spkv2_load_memory(g_{symbol}_spk, sizeof(g_{symbol}_spk), &g_{symbol}_ctx);
    if (rc != 0) return rc;
    rc = spkv2_prepare_with_scratch(
        g_{symbol}_ctx,
        g_{symbol}_activation_arena,
        sizeof(g_{symbol}_activation_arena),
        g_{symbol}_scratch_arena,
        sizeof(g_{symbol}_scratch_arena));
    if (rc != 0) {{
        spkv2_free(g_{symbol}_ctx);
        g_{symbol}_ctx = 0;
    }}
    return rc;
}}

int {symbol}_run_checked(const void *input, size_t input_size, void *output, size_t output_size) {{
    if (!input || !output) return -1;
    if (input_size != {symbol.upper()}_INPUT_SIZE || output_size != {symbol.upper()}_OUTPUT_SIZE) return -2;
    int rc = {symbol}_init();
    if (rc != 0) return rc;
    rc = spkv2_bind_input(g_{symbol}_ctx, 0, (void *)input, input_size);
    if (rc != 0) return rc;
    rc = spkv2_bind_output(g_{symbol}_ctx, 0, output, output_size);
    if (rc != 0) return rc;
    return spkv2_run(g_{symbol}_ctx);
}}

int {symbol}_run(const void *input, void *output) {{
    return {symbol}_run_checked(input, {symbol.upper()}_INPUT_SIZE, output, {symbol.upper()}_OUTPUT_SIZE);
}}

void {symbol}_free(void) {{
    spkv2_free(g_{symbol}_ctx);
    g_{symbol}_ctx = 0;
}}
"""


def _main_test_text(symbol: str) -> str:
    return f"""#include "{symbol}.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned char *read_file(const char *path, size_t *out_size) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    if (size < 0) {{
        fclose(fp);
        return NULL;
    }}
    unsigned char *data = (unsigned char *)malloc((size_t)size);
    if (!data) {{
        fclose(fp);
        return NULL;
    }}
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {{
        free(data);
        fclose(fp);
        return NULL;
    }}
    fclose(fp);
    *out_size = (size_t)size;
    return data;
}}

static int write_file(const char *path, const unsigned char *data, size_t size) {{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    int ok = fwrite(data, 1, size, fp) == size;
    fclose(fp);
    return ok ? 0 : -1;
}}

int main(int argc, char **argv) {{
    if (argc != 3) {{
        fprintf(stderr, "Usage: %s input.bin output.bin\\n", argv[0]);
        return 2;
    }}
    size_t input_size = 0;
    unsigned char *input = read_file(argv[1], &input_size);
    if (!input) {{
        fprintf(stderr, "failed to read input\\n");
        return 1;
    }}
    unsigned char *output = (unsigned char *)malloc({symbol.upper()}_OUTPUT_SIZE);
    if (!output) {{
        free(input);
        return 1;
    }}
    int rc = {symbol}_run_checked(input, input_size, output, {symbol.upper()}_OUTPUT_SIZE);
    if (rc != 0) {{
        fprintf(stderr, "model run failed: %d\\n", rc);
        free(input);
        free(output);
        return 1;
    }}
    rc = write_file(argv[2], output, {symbol.upper()}_OUTPUT_SIZE);
    free(input);
    free(output);
    {symbol}_free();
    return rc == 0 ? 0 : 1;
}}
"""


def _cmake_text(symbol: str, runtime_dir: Path) -> str:
    runtime = str(runtime_dir).replace("\\", "/")
    return f"""cmake_minimum_required(VERSION 3.16)

project({symbol}_generated C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory("{runtime}" spkv2_runtime_build)

add_library({symbol}_model STATIC {symbol}.c)
target_include_directories({symbol}_model PUBLIC ${{CMAKE_CURRENT_SOURCE_DIR}})
target_link_libraries({symbol}_model PUBLIC spkv2_runtime)

add_executable({symbol}_main_test main_test.c)
target_link_libraries({symbol}_main_test PRIVATE {symbol}_model)
"""


def _bytes_literal(data: bytes) -> str:
    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines)


def _fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _sanitize_symbol(name: str) -> str:
    symbol = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not symbol or symbol[0].isdigit():
        symbol = f"model_{symbol}"
    return symbol
