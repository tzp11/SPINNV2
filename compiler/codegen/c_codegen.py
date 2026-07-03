"""Generate static C deployment wrappers from an SPK package."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path


HEADER_STRUCT = struct.Struct("<IHHHHIIIIIIQQQII")
SECTION_STRUCT = struct.Struct("<IIQQII")
TENSOR_STRUCT = struct.Struct("<IHHHH8IQQII")
NODE_STRUCT = struct.Struct("<IHHHH8I4IIIII")
ATTR_STRUCT = struct.Struct("<Iiii4i2i2i2i2i2fii8i3i")
KERNEL_SPEC_STRUCT = struct.Struct("<IIHHHHHHQQII")
MEMORY_PLAN_STRUCT = struct.Struct("<IHHIQQII")

SPKV2_MAGIC = 0x32564B50
SECTION_TENSOR_TABLE = 3
SECTION_NODE_TABLE = 4
SECTION_ATTRIBUTES = 5
SECTION_WEIGHTS = 6
SECTION_MEMORY_PLAN = 7
SECTION_KERNEL_SPEC = 8
ROLE_INPUT = 1
ROLE_OUTPUT = 2
ROLE_WEIGHT = 3

# (op_type, backend, kernel_kind) → C function name
# Mirrors REGISTRY in runtime/kernels/reference/kernels.c
_KERNEL_FN = {
    (2, 3, 3): "kernel_conv_simd",      (2, 3, 5): "kernel_conv_simd",
    (2, 3, 6): "kernel_conv_simd",      (2, 3, 7): "kernel_conv_simd",
    (2, 3, 9): "kernel_conv_simd",      (2, 3, 8): "kernel_conv_simd",
    (2, 3, 10): "kernel_conv_int8",     (2, 3, 11): "kernel_conv_bnns",
    (4, 3, 2): "kernel_gemm_simd",      (19, 3, 2): "kernel_matmul_simd",
    (1, 3, 1): "kernel_add_simd",       (8, 3, 1): "kernel_mul_simd",
    (9, 3, 1): "kernel_sub_simd",       (10, 3, 1): "kernel_div_simd",
    (6, 3, 1): "kernel_relu_simd",      (12, 3, 1): "kernel_sigmoid_simd",
    (14, 3, 1): "kernel_transpose_simd",(18, 3, 1): "kernel_reduce_simd",
    (17, 3, 1): "kernel_reduce_simd",   (7, 3, 1): "kernel_softmax_simd",
    (5, 3, 1): "kernel_maxpool_simd",   (20, 3, 1): "kernel_resize_simd",
    (28, 3, 1): "kernel_avgpool",       (29, 3, 1): "kernel_global_avgpool",
    (30, 3, 1): "kernel_clip",          (31, 3, 1): "kernel_leakyrelu",
    (1, 1, 1): "kernel_add",            (25, 1, 1): "kernel_copy",
    (15, 1, 1): "kernel_concat",        (2, 1, 1): "kernel_conv",
    (2, 2, 3): "kernel_conv_im2col",    (10, 1, 1): "kernel_binary",
    (3, 1, 1): "kernel_flatten",        (23, 1, 1): "kernel_gather_elements",
    (4, 1, 1): "kernel_gemm",           (4, 2, 2): "kernel_gemm_cpu_direct",
    (19, 1, 1): "kernel_matmul",        (5, 1, 1): "kernel_maxpool",
    (11, 1, 1): "kernel_binary",        (8, 1, 1): "kernel_binary",
    (18, 1, 1): "kernel_reduce",        (17, 1, 1): "kernel_reduce",
    (6, 1, 1): "kernel_relu",           (13, 1, 1): "kernel_copy",
    (20, 1, 1): "kernel_resize",        (12, 1, 1): "kernel_sigmoid",
    (7, 1, 1): "kernel_softmax",        (16, 1, 1): "kernel_split",
    (9, 1, 1): "kernel_binary",         (21, 1, 1): "kernel_tile",
    (24, 1, 1): "kernel_topk",          (14, 1, 1): "kernel_transpose",
    (22, 1, 1): "kernel_unsqueeze",     (26, 1, 1): "kernel_slice",
    (27, 1, 1): "kernel_gather",        (28, 1, 1): "kernel_avgpool",
    (29, 1, 1): "kernel_global_avgpool",(30, 1, 1): "kernel_clip",
    (31, 1, 1): "kernel_leakyrelu",
}

_KERNEL_GUARD: dict[str, str] = {}
for _fn in [
    "kernel_conv_simd", "kernel_gemm_simd", "kernel_matmul_simd",
    "kernel_add_simd", "kernel_mul_simd", "kernel_sub_simd",
    "kernel_div_simd", "kernel_relu_simd", "kernel_sigmoid_simd",
    "kernel_transpose_simd", "kernel_reduce_simd",
    "kernel_softmax_simd", "kernel_maxpool_simd", "kernel_resize_simd",
]:
    _KERNEL_GUARD[_fn] = "#if defined(__AVX2__) || defined(__ARM_NEON)"
_KERNEL_GUARD["kernel_conv_int8"] = (
    "#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)"
)
_KERNEL_GUARD["kernel_conv_bnns"] = "#ifdef __APPLE__"


@dataclass
class SpkInfo:
    input_sizes: list[int]
    output_sizes: list[int]
    activation_arena_bytes: int
    scratch_arena_bytes: int
    checksum: int
    weight_bytes: int = 0

    @property
    def input_size(self) -> int:
        return self.input_sizes[0]

    @property
    def output_size(self) -> int:
        return self.output_sizes[0]


def generate_c_from_spk(
    spk_path: str | Path,
    out_dir: str | Path,
    *,
    name: str = "model",
    runtime_dir: str | Path = "runtime",
    external_weights: bool = False,
) -> None:
    spk_path = Path(spk_path)
    out_dir = Path(out_dir)
    symbol = _sanitize_symbol(name)
    data = spk_path.read_bytes()
    info = _inspect_spk(data)

    out_dir.mkdir(parents=True, exist_ok=True)

    if external_weights and info.weight_bytes > 0:
        spk_no_weights, weights_blob = _extract_weights(data)
        (out_dir / f"{symbol}_weights.bin").write_bytes(weights_blob)
        info.checksum = _fnv1a32(spk_no_weights)
        (out_dir / f"{symbol}.h").write_text(
            _header_text(symbol, info, external_weights=True), encoding="utf-8"
        )
        (out_dir / f"{symbol}.c").write_text(
            _source_text_external(symbol, spk_no_weights, info), encoding="utf-8"
        )
    else:
        (out_dir / f"{symbol}.h").write_text(
            _header_text(symbol, info, external_weights=False), encoding="utf-8"
        )
        (out_dir / f"{symbol}.c").write_text(
            _source_text(symbol, data, info), encoding="utf-8"
        )

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
    weight_bytes = 0
    for i in range(section_count):
        entry_offset = header_size + i * SECTION_STRUCT.size
        kind, _flags, offset, size, _alignment, _reserved = SECTION_STRUCT.unpack_from(data, entry_offset)
        if kind == SECTION_TENSOR_TABLE:
            tensor_offset = offset
            tensor_size = size
        if kind == SECTION_WEIGHTS:
            weight_bytes = size
    if tensor_offset is None or tensor_size is None:
        raise ValueError("SPK missing tensor table")
    if tensor_size < num_tensors * TENSOR_STRUCT.size:
        raise ValueError("SPK tensor table is truncated")

    input_sizes: list[int] = []
    output_sizes: list[int] = []
    for i in range(num_tensors):
        record = TENSOR_STRUCT.unpack_from(data, tensor_offset + i * TENSOR_STRUCT.size)
        role = record[2]
        size_bytes = record[13]
        if role == ROLE_INPUT:
            input_sizes.append(int(size_bytes))
        elif role == ROLE_OUTPUT:
            output_sizes.append(int(size_bytes))
    if not input_sizes or not output_sizes:
        raise ValueError("SPK must contain at least one input and one output")

    return SpkInfo(
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        activation_arena_bytes=int(activation_arena_bytes),
        scratch_arena_bytes=int(scratch_arena_bytes),
        checksum=_fnv1a32(data),
        weight_bytes=int(weight_bytes),
    )


def _header_text(symbol: str, info: SpkInfo, *, external_weights: bool = False) -> str:
    guard = f"{symbol.upper()}_H"
    init_decl = (
        f"int {symbol}_init(const char *weights_path);"
        if external_weights
        else f"int {symbol}_init(void);"
    )
    sym = symbol.upper()

    size_macros = []
    for i, sz in enumerate(info.input_sizes):
        size_macros.append(f"#define {sym}_INPUT_SIZE_{i} ((size_t){sz}u)")
    for i, sz in enumerate(info.output_sizes):
        size_macros.append(f"#define {sym}_OUTPUT_SIZE_{i} ((size_t){sz}u)")
    size_macros.append(f"#define {sym}_INPUT_COUNT {len(info.input_sizes)}")
    size_macros.append(f"#define {sym}_OUTPUT_COUNT {len(info.output_sizes)}")
    size_macros.append(f"#define {sym}_INPUT_SIZE {sym}_INPUT_SIZE_0")
    size_macros.append(f"#define {sym}_OUTPUT_SIZE {sym}_OUTPUT_SIZE_0")
    size_macros_str = "\n".join(size_macros)

    return f"""#ifndef {guard}
#define {guard}

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

{size_macros_str}
#define {sym}_ACTIVATION_ARENA_SIZE ((size_t){max(info.activation_arena_bytes, 1)}u)
#define {sym}_SCRATCH_ARENA_SIZE ((size_t){max(info.scratch_arena_bytes, 1)}u)
#define {sym}_SPK_CHECKSUM 0x{info.checksum:08x}u

{init_decl}
int {symbol}_run(const void *input, void *output);
int {symbol}_run_checked(const void *input, size_t input_size, void *output, size_t output_size);
int {symbol}_run_multi(const void *const *inputs, const size_t *input_sizes, int num_inputs,
                       void *const *outputs, const size_t *output_sizes, int num_outputs);
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

int {symbol}_run_multi(const void *const *inputs, const size_t *input_sizes, int num_inputs,
                       void *const *outputs, const size_t *output_sizes, int num_outputs) {{
    if (!inputs || !outputs || !input_sizes || !output_sizes) return -1;
    int rc = {symbol}_init();
    if (rc != 0) return rc;
    for (int i = 0; i < num_inputs; i++) {{
        rc = spkv2_bind_input(g_{symbol}_ctx, i, (void *)inputs[i], input_sizes[i]);
        if (rc != 0) return rc;
    }}
    for (int i = 0; i < num_outputs; i++) {{
        rc = spkv2_bind_output(g_{symbol}_ctx, i, outputs[i], output_sizes[i]);
        if (rc != 0) return rc;
    }}
    return spkv2_run(g_{symbol}_ctx);
}}

void {symbol}_free(void) {{
    spkv2_free(g_{symbol}_ctx);
    g_{symbol}_ctx = 0;
}}
"""


def _extract_weights(data: bytes) -> tuple[bytes, bytes]:
    header = HEADER_STRUCT.unpack_from(data, 0)
    header_size = header[4]
    section_count = header[5]
    for i in range(section_count):
        entry_offset = header_size + i * SECTION_STRUCT.size
        kind, _flags, offset, size, _alignment, _reserved = SECTION_STRUCT.unpack_from(data, entry_offset)
        if kind == SECTION_WEIGHTS:
            weights = data[offset : offset + size]
            spk_no_weights = bytearray(data)
            spk_no_weights[offset : offset + size] = b"\x00" * size
            return bytes(spk_no_weights), weights
    return data, b""


def _source_text_external(symbol: str, data: bytes, info: SpkInfo) -> str:
    bytes_literal = _bytes_literal(data)
    activation_size = max(info.activation_arena_bytes, 1)
    scratch_size = max(info.scratch_arena_bytes, 1)
    return f"""#include "{symbol}.h"

#include "spkv2_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char g_{symbol}_spk[] = {{
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

int {symbol}_init(const char *weights_path) {{
    if (g_{symbol}_ctx) return 0;
    FILE *fp = fopen(weights_path, "rb");
    if (!fp) return -20;
    fseek(fp, 0, SEEK_END);
    long wsize = ftell(fp);
    rewind(fp);
    if (wsize <= 0) {{ fclose(fp); return -21; }}
    unsigned char *weights = (unsigned char *)malloc((size_t)wsize);
    if (!weights) {{ fclose(fp); return -22; }}
    if (fread(weights, 1, (size_t)wsize, fp) != (size_t)wsize) {{
        free(weights);
        fclose(fp);
        return -23;
    }}
    fclose(fp);

    /* Patch weights into the SPK buffer (zeroed weight section) */
    /* Find weight section offset from header */
    uint16_t header_sz = *(uint16_t *)(g_{symbol}_spk + 8);
    uint32_t section_count = *(uint32_t *)(g_{symbol}_spk + 10);
    for (uint32_t i = 0; i < section_count; i++) {{
        uint32_t *entry = (uint32_t *)(g_{symbol}_spk + header_sz + i * 24);
        if (entry[0] == 6) {{ /* SECTION_WEIGHTS */
            uint64_t offset = *(uint64_t *)(entry + 2);
            uint64_t size = *(uint64_t *)(entry + 4);
            if ((size_t)wsize == (size_t)size) {{
                memcpy(g_{symbol}_spk + offset, weights, (size_t)wsize);
            }}
            break;
        }}
    }}
    free(weights);

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
    if (!g_{symbol}_ctx) return -3;
    int rc = spkv2_bind_input(g_{symbol}_ctx, 0, (void *)input, input_size);
    if (rc != 0) return rc;
    rc = spkv2_bind_output(g_{symbol}_ctx, 0, output, output_size);
    if (rc != 0) return rc;
    return spkv2_run(g_{symbol}_ctx);
}}

int {symbol}_run(const void *input, void *output) {{
    return {symbol}_run_checked(input, {symbol.upper()}_INPUT_SIZE, output, {symbol.upper()}_OUTPUT_SIZE);
}}

int {symbol}_run_multi(const void *const *inputs, const size_t *input_sizes, int num_inputs,
                       void *const *outputs, const size_t *output_sizes, int num_outputs) {{
    if (!inputs || !outputs || !input_sizes || !output_sizes) return -1;
    if (!g_{symbol}_ctx) return -3;
    int rc;
    for (int i = 0; i < num_inputs; i++) {{
        rc = spkv2_bind_input(g_{symbol}_ctx, i, (void *)inputs[i], input_sizes[i]);
        if (rc != 0) return rc;
    }}
    for (int i = 0; i < num_outputs; i++) {{
        rc = spkv2_bind_output(g_{symbol}_ctx, i, outputs[i], output_sizes[i]);
        if (rc != 0) return rc;
    }}
    return spkv2_run(g_{symbol}_ctx);
}}

void {symbol}_free(void) {{
    spkv2_free(g_{symbol}_ctx);
    g_{symbol}_ctx = 0;
}}
"""


def _main_test_text(symbol: str) -> str:
    return _main_test_text_ex(symbol, external_weights=False)


def _main_test_text_ex(symbol: str, *, external_weights: bool = False) -> str:
    if external_weights:
        usage = f'fprintf(stderr, "Usage: %s input.bin output.bin weights.bin\\n", argv[0]);'
        argc_check = "argc != 4"
        init_block = f"""    int init_rc = {symbol}_init(argv[3]);
    if (init_rc != 0) {{
        fprintf(stderr, "init failed: %d\\n", init_rc);
        return 1;
    }}
"""
    else:
        usage = f'fprintf(stderr, "Usage: %s input.bin output.bin\\n", argv[0]);'
        argc_check = "argc != 3"
        init_block = ""

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
    if ({argc_check}) {{
        {usage}
        return 2;
    }}
{init_block}    size_t input_size = 0;
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


# ---------------------------------------------------------------------------
# Inline codegen: direct kernel calls, no SPK loader / registry
# ---------------------------------------------------------------------------

def _parse_spk_full(data: bytes) -> dict:
    hdr = HEADER_STRUCT.unpack_from(data, 0)
    if hdr[0] != SPKV2_MAGIC:
        raise ValueError("invalid SPK magic")
    header_size, section_count = hdr[4], hdr[5]
    num_tensors, num_nodes = hdr[7], hdr[8]
    activation_arena, scratch_arena = int(hdr[12]), int(hdr[13])

    secs: dict[int, tuple[int, int]] = {}
    for i in range(section_count):
        off = header_size + i * SECTION_STRUCT.size
        kind, _, offset, size, _, _ = SECTION_STRUCT.unpack_from(data, off)
        secs[kind] = (int(offset), int(size))

    tensors = []
    if SECTION_TENSOR_TABLE in secs:
        o, _ = secs[SECTION_TENSOR_TABLE]
        for i in range(num_tensors):
            tensors.append(TENSOR_STRUCT.unpack_from(data, o + i * TENSOR_STRUCT.size))

    nodes = []
    if SECTION_NODE_TABLE in secs:
        o, _ = secs[SECTION_NODE_TABLE]
        for i in range(num_nodes):
            nodes.append(NODE_STRUCT.unpack_from(data, o + i * NODE_STRUCT.size))

    attrs_raw = b""
    if SECTION_ATTRIBUTES in secs:
        o, s = secs[SECTION_ATTRIBUTES]
        attrs_raw = data[o : o + s]

    weights_raw = b""
    if SECTION_WEIGHTS in secs:
        o, s = secs[SECTION_WEIGHTS]
        weights_raw = data[o : o + s]

    specs = []
    if SECTION_KERNEL_SPEC in secs:
        o, s = secs[SECTION_KERNEL_SPEC]
        for i in range(s // KERNEL_SPEC_STRUCT.size):
            specs.append(
                KERNEL_SPEC_STRUCT.unpack_from(data, o + i * KERNEL_SPEC_STRUCT.size)
            )

    in_ids = [t[0] for t in tensors if t[2] == ROLE_INPUT]
    out_ids = [t[0] for t in tensors if t[2] == ROLE_OUTPUT]

    return {
        "num_tensors": num_tensors,
        "num_nodes": num_nodes,
        "activation_arena_bytes": activation_arena,
        "scratch_arena_bytes": scratch_arena,
        "tensor_records": tensors,
        "node_records": nodes,
        "attrs_raw": attrs_raw,
        "weights_raw": weights_raw,
        "kernel_specs": specs,
        "input_ids": in_ids,
        "output_ids": out_ids,
        "input_sizes": [int(t[13]) for t in tensors if t[2] == ROLE_INPUT],
        "output_sizes": [int(t[13]) for t in tensors if t[2] == ROLE_OUTPUT],
    }


def _c_tensor_init(rec: tuple) -> str:
    tid, dtype, role, rank, mc = rec[0], rec[1], rec[2], rec[3], rec[4]
    shape = ", ".join(str(s) for s in rec[5:13])
    return (
        f"    {{{tid}, {dtype}, {role}, {rank}, {mc}, "
        f"{{{shape}}}, {rec[13]}ULL, {rec[14]}ULL, {rec[15]}, {rec[16]}}}"
    )


def _c_node_init(rec: tuple) -> str:
    nid, op, fl, ic, oc = rec[0], rec[1], rec[2], rec[3], rec[4]
    ins = ", ".join(str(x) for x in rec[5:13])
    outs = ", ".join(str(x) for x in rec[13:17])
    return (
        f"    {{{nid}, {op}, {fl}, {ic}, {oc}, "
        f"{{{ins}}}, {{{outs}}}, {rec[17]}, {rec[18]}, {rec[19]}u, {rec[20]}}}"
    )


def _c_ks_init(rec: tuple) -> str:
    return (
        f"    {{{rec[0]}, {rec[1]}, {rec[2]}, {rec[3]}, {rec[4]}, "
        f"{rec[5]}, {rec[6]}, {rec[7]}, {rec[8]}ULL, {rec[9]}ULL, "
        f"{rec[10]}u, {rec[11]}}}"
    )


def _inline_source_text(sym: str, spk: dict, *, external_weights: bool = False) -> str:
    SYM = sym.upper()
    nt = spk["num_tensors"]
    nn = spk["num_nodes"]
    nk = len(spk["kernel_specs"])
    act_sz = max(spk["activation_arena_bytes"], 1)
    scratch_sz = max(spk["scratch_arena_bytes"], 1)
    in_ids = spk["input_ids"]
    out_ids = spk["output_ids"]

    used_fns: set[str] = set()
    node_kernels: list[tuple[str | None, str, str]] = []
    for rec in spk["node_records"]:
        fn, guard, scratch = _resolve_node_kernel(rec, spk["kernel_specs"], sym)
        node_kernels.append((fn, guard, scratch))
        if fn:
            used_fns.add(fn)

    extern_lines: list[str] = []
    for fn in sorted(used_fns):
        g = _KERNEL_GUARD.get(fn, "")
        decl = f"extern int {fn}(Spkv2Context *, const Spkv2NodeRecord *, void *);"
        if g:
            extern_lines.extend([g, decl, "#endif"])
        else:
            extern_lines.append(decl)

    exec_lines: list[str] = []
    for i, (fn, guard, scratch) in enumerate(node_kernels):
        if fn and guard:
            exec_lines.append(guard)
            exec_lines.append(
                f"    rc = {fn}(&g_{sym}_ctx, &g_{sym}_nr[{i}], {scratch});"
            )
            exec_lines.append("#else")
            exec_lines.append(
                f"    rc = spkv2_execute_node(&g_{sym}_ctx, &g_{sym}_nr[{i}]);"
            )
            exec_lines.append("#endif")
        elif fn:
            exec_lines.append(
                f"    rc = {fn}(&g_{sym}_ctx, &g_{sym}_nr[{i}], {scratch});"
            )
        else:
            exec_lines.append(
                f"    rc = spkv2_execute_node(&g_{sym}_ctx, &g_{sym}_nr[{i}]);"
            )
        exec_lines.append("    if (rc != 0) return rc;")

    multi_bind: list[str] = []
    for idx, tid in enumerate(in_ids):
        sz = next(r[13] for r in spk["tensor_records"] if r[0] == tid)
        multi_bind.append(
            f"    if (num_inputs > {idx}) {{\n"
            f"        if (input_sizes[{idx}] != {sz}u) return -2;\n"
            f"        g_{sym}_ts[{tid}].data = (uint8_t *)inputs[{idx}];\n"
            f"    }}"
        )
    for idx, tid in enumerate(out_ids):
        sz = next(r[13] for r in spk["tensor_records"] if r[0] == tid)
        multi_bind.append(
            f"    if (num_outputs > {idx}) {{\n"
            f"        if (output_sizes[{idx}] != {sz}u) return -2;\n"
            f"        g_{sym}_ts[{tid}].data = (uint8_t *)outputs[{idx}];\n"
            f"    }}"
        )

    w_decl = ""
    w_ptr = "NULL"
    w_sz = "0"
    if external_weights:
        w_decl = f"static uint8_t *g_{sym}_loaded_w;\nstatic size_t g_{sym}_loaded_w_sz;\n"
        w_ptr = f"g_{sym}_loaded_w"
        w_sz = f"g_{sym}_loaded_w_sz"
    elif spk["weights_raw"]:
        w_decl = (
            f"static const unsigned char g_{sym}_w[] = {{\n"
            f"{_bytes_literal(spk['weights_raw'])}\n}};\n"
        )
        w_ptr = f"g_{sym}_w"
        w_sz = f"sizeof(g_{sym}_w)"

    a_decl = ""
    a_ptr = "NULL"
    a_sz = "0"
    if spk["attrs_raw"]:
        a_decl = (
            f"static const unsigned char g_{sym}_a[] = {{\n"
            f"{_bytes_literal(spk['attrs_raw'])}\n}};\n"
        )
        a_ptr = f"g_{sym}_a"
        a_sz = f"sizeof(g_{sym}_a)"

    ks_decl = ""
    ks_ptr = "NULL"
    if nk > 0:
        ks_lines = ",\n".join(_c_ks_init(r) for r in spk["kernel_specs"])
        ks_decl = (
            f"static const Spkv2KernelSpecRecord g_{sym}_ks[{nk}] = {{\n"
            f"{ks_lines}\n}};\n"
        )
        ks_ptr = f"g_{sym}_ks"

    tr_lines = ",\n".join(_c_tensor_init(r) for r in spk["tensor_records"])
    nr_lines = ",\n".join(_c_node_init(r) for r in spk["node_records"])
    out_id_str = ", ".join(str(x) for x in out_ids)
    nc_sz = max(nn, 1)

    # Include stdio for external weights file loading
    extra_include = '#include <stdio.h>\n' if external_weights else ''

    # Generate init function signature and body
    if external_weights:
        init_sig = f"int {sym}_init(const char *weights_path)"
        init_load_weights = f"""    FILE *fp = fopen(weights_path, "rb");
    if (!fp) return -20;
    fseek(fp, 0, SEEK_END);
    long wsize = ftell(fp);
    rewind(fp);
    if (wsize <= 0) {{ fclose(fp); return -21; }}
    g_{sym}_loaded_w = (uint8_t *)malloc((size_t)wsize);
    if (!g_{sym}_loaded_w) {{ fclose(fp); return -22; }}
    if (fread(g_{sym}_loaded_w, 1, (size_t)wsize, fp) != (size_t)wsize) {{
        free(g_{sym}_loaded_w); g_{sym}_loaded_w = NULL;
        fclose(fp); return -23;
    }}
    fclose(fp);
    g_{sym}_loaded_w_sz = (size_t)wsize;"""
        run_init_call = ""
        run_ready_check = f"    if (!g_{sym}_ready) return -3;\n"
        free_weights = f"""    if (g_{sym}_loaded_w) {{
        free(g_{sym}_loaded_w);
        g_{sym}_loaded_w = NULL;
    }}"""
    else:
        init_sig = f"int {sym}_init(void)"
        init_load_weights = ""
        run_init_call = f"    int rc = {sym}_init();\n    if (rc != 0) return rc;\n"
        run_ready_check = ""
        free_weights = ""

    # For external weights, weight pointer uses loaded buffer (non-const cast)
    if external_weights:
        w_bind_data = f"(uint8_t *)g_{sym}_loaded_w"
    else:
        w_bind_data = f"(uint8_t *){w_ptr}"

    return f'''#include "{sym}.h"

#include "spkv2_format.h"
#include "context.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
{extra_include}
{chr(10).join(extern_lines)}

{w_decl}
static const Spkv2TensorRecord g_{sym}_tr[{nt}] = {{
{tr_lines}
}};

static const Spkv2NodeRecord g_{sym}_nr[{nn}] = {{
{nr_lines}
}};

{a_decl}
{ks_decl}
static uint32_t g_{sym}_oids[] = {{{out_id_str}}};

static unsigned char g_{sym}_arena[{act_sz}];
static unsigned char g_{sym}_scratch[{scratch_sz}];

static Spkv2TensorState g_{sym}_ts[{nt}];
static void *g_{sym}_nc[{nc_sz}];
static void (*g_{sym}_ncd[{nc_sz}])(void *);
static Spkv2Context g_{sym}_ctx;
static int g_{sym}_ready;

int {sym}_verify_checksum(const void *data, size_t size) {{
    (void)data; (void)size;
    return 0;
}}

{init_sig} {{
    if (g_{sym}_ready) return 0;
{init_load_weights}
    memset(&g_{sym}_ctx, 0, sizeof(g_{sym}_ctx));
    g_{sym}_ctx.header.num_tensors = {nt};
    g_{sym}_ctx.header.num_nodes = {nn};
    g_{sym}_ctx.header.activation_arena_bytes = {act_sz};
    g_{sym}_ctx.header.scratch_arena_bytes = {scratch_sz};
    g_{sym}_ctx.tensor_records = g_{sym}_tr;
    g_{sym}_ctx.node_records = g_{sym}_nr;
    g_{sym}_ctx.attrs = (const uint8_t *){a_ptr};
    g_{sym}_ctx.attrs_size = {a_sz};
    g_{sym}_ctx.weights = (const uint8_t *){w_ptr};
    g_{sym}_ctx.weights_size = {w_sz};
    g_{sym}_ctx.kernel_spec_records = {ks_ptr};
    g_{sym}_ctx.kernel_spec_count = {nk};
    g_{sym}_ctx.output_ids = g_{sym}_oids;
    g_{sym}_ctx.output_count = {len(out_ids)};
    g_{sym}_ctx.tensors = g_{sym}_ts;
    g_{sym}_ctx.scratch = g_{sym}_scratch;
    g_{sym}_ctx.scratch_size = sizeof(g_{sym}_scratch);
    g_{sym}_ctx.arena_size = sizeof(g_{sym}_arena);
    g_{sym}_ctx.node_cache = g_{sym}_nc;
    g_{sym}_ctx.node_cache_dtors = g_{sym}_ncd;
    g_{sym}_ctx.node_cache_count = {nc_sz};
    for (uint32_t i = 0; i < {nt}; i++) {{
        g_{sym}_ts[i].record = &g_{sym}_tr[i];
        if (g_{sym}_tr[i].role == {ROLE_WEIGHT}) {{
            g_{sym}_ts[i].data = {w_bind_data} + g_{sym}_tr[i].data_offset;
        }} else if (g_{sym}_tr[i].memory_class == 5) {{
            g_{sym}_ts[i].data = NULL;
        }} else {{
            g_{sym}_ts[i].data = g_{sym}_arena + g_{sym}_tr[i].data_offset;
        }}
    }}
    g_{sym}_ready = 1;
    return 0;
}}

static int {sym}_exec(void) {{
    int rc;
{chr(10).join(exec_lines)}
    return 0;
}}

int {sym}_run(const void *input, void *output) {{
{run_init_call}{run_ready_check}    g_{sym}_ts[{in_ids[0]}].data = (uint8_t *)input;
    g_{sym}_ts[{out_ids[0]}].data = (uint8_t *)output;
    return {sym}_exec();
}}

int {sym}_run_checked(const void *input, size_t input_size,
                      void *output, size_t output_size) {{
    if (!input || !output) return -1;
    if (input_size != {SYM}_INPUT_SIZE || output_size != {SYM}_OUTPUT_SIZE) return -2;
    return {sym}_run(input, output);
}}

int {sym}_run_multi(const void *const *inputs, const size_t *input_sizes, int num_inputs,
                    void *const *outputs, const size_t *output_sizes, int num_outputs) {{
    if (!inputs || !outputs || !input_sizes || !output_sizes) return -1;
{run_ready_check}{chr(10).join(multi_bind)}
    return {sym}_exec();
}}

void {sym}_free(void) {{
    if (!g_{sym}_ready) return;
    for (uint32_t i = 0; i < {nc_sz}; i++) {{
        if (g_{sym}_nc[i]) {{
            if (g_{sym}_ncd[i])
                g_{sym}_ncd[i](g_{sym}_nc[i]);
            else
                free(g_{sym}_nc[i]);
            g_{sym}_nc[i] = NULL;
        }}
    }}
{free_weights}
    g_{sym}_ready = 0;
}}
'''


def _resolve_node_kernel(
    node_rec: tuple, specs: list[tuple], sym: str
) -> tuple[str | None, str, str]:
    op_type = node_rec[1]
    ks_id = node_rec[19]
    if ks_id == 0xFFFFFFFF or ks_id >= len(specs):
        return None, "", "NULL"
    spec = specs[ks_id]
    backend, kind = spec[3], spec[2]
    s_off, s_sz = int(spec[8]), int(spec[9])
    fn = _KERNEL_FN.get((op_type, backend, kind))
    if fn is None:
        fb_id = spec[10]
        if fb_id != 0xFFFFFFFF and fb_id < len(specs):
            fb = specs[fb_id]
            fn = _KERNEL_FN.get((op_type, fb[3], fb[2]))
            if fn:
                s_off, s_sz = int(fb[8]), int(fb[9])
    if fn is None:
        return None, "", "NULL"
    guard = _KERNEL_GUARD.get(fn, "")
    scratch = f"g_{sym}_scratch + {s_off}" if s_sz > 0 else "NULL"
    return fn, guard, scratch


def _cmake_text_inline(symbol: str, runtime_dir: Path) -> str:
    runtime = str(runtime_dir).replace("\\", "/")
    return f"""cmake_minimum_required(VERSION 3.16)

project({symbol}_generated C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory("{runtime}" spkv2_runtime_build)

add_library({symbol}_model STATIC {symbol}.c)
target_include_directories({symbol}_model
    PUBLIC ${{CMAKE_CURRENT_SOURCE_DIR}}
    PRIVATE "{runtime}/core"
)
target_link_libraries({symbol}_model PUBLIC spkv2_runtime)

add_executable({symbol}_main_test main_test.c)
target_link_libraries({symbol}_main_test PRIVATE {symbol}_model)
"""


def generate_c_inline(
    spk_path: str | Path,
    out_dir: str | Path,
    *,
    name: str = "model",
    runtime_dir: str | Path = "runtime",
    external_weights: bool = False,
) -> None:
    spk_path = Path(spk_path)
    out_dir = Path(out_dir)
    sym = _sanitize_symbol(name)
    data = spk_path.read_bytes()
    spk = _parse_spk_full(data)
    info = SpkInfo(
        input_sizes=spk["input_sizes"],
        output_sizes=spk["output_sizes"],
        activation_arena_bytes=spk["activation_arena_bytes"],
        scratch_arena_bytes=spk["scratch_arena_bytes"],
        checksum=0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if external_weights and spk["weights_raw"]:
        (out_dir / f"{sym}_weights.bin").write_bytes(spk["weights_raw"])
    (out_dir / f"{sym}.h").write_text(
        _header_text(sym, info, external_weights=external_weights), encoding="utf-8"
    )
    (out_dir / f"{sym}.c").write_text(
        _inline_source_text(sym, spk, external_weights=external_weights), encoding="utf-8"
    )
    (out_dir / "main_test.c").write_text(
        _main_test_text_ex(sym, external_weights=external_weights), encoding="utf-8"
    )
    (out_dir / "CMakeLists.txt").write_text(
        _cmake_text_inline(sym, Path(runtime_dir).resolve()), encoding="utf-8"
    )
