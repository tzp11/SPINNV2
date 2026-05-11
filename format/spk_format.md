# SPK Format Specification

SPK is the section-based deployment package format used by SPINNV2 Runtime.

## Goals

1. Keep runtime parsing simple.
2. Store graph, weights, memory plan, KernelSpec, and target metadata together.
3. Support static validation before execution.
4. Allow optional debug sections to be stripped for deployment.

## Header

The C representation is defined in `runtime/include/spkv2_format.h`.

Required header fields:

| Field | Meaning |
|---|---|
| `magic` | `SPKV2_MAGIC` |
| `version_major/minor` | Format version. Current binary writer emits `0.2`. |
| `endianness` | Encoded byte order |
| `section_count` | Number of section directory entries |
| `num_tensors` | Tensor count |
| `num_nodes` | Node count |
| `weight_bytes` | Total weight bytes |
| `activation_arena_bytes` | Required activation arena size |
| `scratch_arena_bytes` | Required scratch arena size |
| `target_profile_hash` | Target profile identity |
| `checksum_type` | Checksum algorithm ID |

## Sections

Initial section IDs:

| ID | Section |
|---|---|
| 1 | Metadata |
| 2 | Target Profile |
| 3 | Tensor Table |
| 4 | Node Table |
| 5 | Attributes |
| 6 | Weights |
| 7 | Memory Plan |
| 8 | KernelSpec |
| 9 | Quantization |
| 10 | String Table |
| 11 | Debug |
| 12 | Checksum |

## Validation Rules

Runtime loader must validate:

1. Header magic and version.
2. Section table bounds.
3. Section offset and size overflow.
4. Required sections exist.
5. Arena sizes fit user-provided memory.
6. Tensor memory offsets do not exceed arena sizes.
7. KernelSpec entries can be mapped to compiled kernels or fallback kernels.

## M0 Status

M0 defines the format constants and documentation only. Full binary serialization starts in M1.

## M1 Binary Tables

M1 implements a compact binary subset:

```text
Header
Section Directory
Metadata JSON
Target Profile JSON
Tensor Table
Node Table
Attribute Table
Weight Blob
String Table
```

The runtime parses Tensor Table and Node Table directly. Debug JSON remains compiler-side output and is not required by the C runtime.

### Tensor Record

The C definition is `Spkv2TensorRecord` in `runtime/include/spkv2_format.h`.

M1 tensor records include:

```text
id
dtype
role
rank
memory_class
shape[8]
size_bytes
data_offset
name_offset
```

For M1, `data_offset` is only used for weight tensors and points into the Weight section.

### Node Record

The C definition is `Spkv2NodeRecord`.

M1 node records include:

```text
id
op_type
input_count
output_count
inputs[8]
outputs[4]
attr_offset
attr_size
kernel_spec_id
scratch_bytes
```

`kernel_spec_id` and `scratch_bytes` are reserved in M1 and become active in M2-M4.

### Attribute Record

The C definition is `Spkv2AttrRecord`. M1 uses a single fixed attribute record for the supported reference kernels. This is intentionally simple and will be split into op-specific compact attributes after the SPK format stabilizes.

M3 extends the fixed attribute record with `fused_activation`. The current value
`1` means Conv output applies fused Relu in the reference runtime; `0` means no
fused activation.

M6 keeps the fixed record shape but extends it for large-model operators:

```text
extra_count
keepdims
extra[8]        # Transpose perm or Reduce axes
largest
sorted
cast_to
reserved
```

The extension is used by ResNet101/YOLOv10n operators such as `Transpose`,
`ReduceMean`, `ReduceMax`, `TopK`, and `Cast`.

## M2 Memory Plan

M2 activates the Memory Plan section and header arena size fields.

Compiler writes:

```text
header.activation_arena_bytes
Memory Plan Section
TensorRecord.data_offset for non-weight tensors
memory_plan.csv optional sidecar
debug JSON memory summary
```

For M2, `TensorRecord.data_offset` has two meanings:

```text
weight tensor:
    offset into Weight section

non-weight tensor:
    offset into activation arena
```

Runtime `prepare` uses `header.activation_arena_bytes` to validate the provided arena size. Then it binds each non-weight tensor as:

```text
tensor.data = arena_base + tensor.data_offset
```

If a tensor is marked `SPKV2_MEMORY_EXTERNAL`, Runtime leaves its data pointer unset until the user calls an external bind API.

The C definition for Memory Plan entries is `Spkv2MemoryPlanRecord`.

M2 memory plan records include:

```text
tensor_id
memory_class
alignment
offset
size
first_use
last_use
```

The Memory Plan section is primarily for verification, debugging, and paper experiments. Runtime uses Tensor Table offsets for the fast path.

## M4 KernelSpec

M4 activates the KernelSpec section and the node/header scratch fields.

Compiler writes:

```text
header.scratch_arena_bytes
NodeRecord.kernel_spec_id
NodeRecord.scratch_bytes
KernelSpec Section
debug JSON metadata.kernel_specs
debug JSON metadata.kernel_fallback_count
debug JSON metadata.scratch_arena_bytes
```

The C definition for KernelSpec entries is `Spkv2KernelSpecRecord`.

M4 KernelSpec records include:

```text
id
node_id
kernel_kind
backend
dtype
layout
weight_layout
flags
scratch_offset
scratch_bytes
fallback_kernel_spec_id
required_feature_mask
```

Runtime maps `NodeRecord.kernel_spec_id` to a KernelSpec record, then dispatches
through its compiled kernel registry using `op_type + backend + kernel_kind`. If
the selected kernel is not present, Runtime follows `fallback_kernel_spec_id`.
Current backend codes are `ref=1` and `cpu=2`; current kernel kind codes are
`reference=1`, `direct=2`, and `im2col_gemm=3`.

## M5 Checksum And Codegen

M5 enables `header.checksum_type = 1`, meaning FNV-1a 32-bit. The Checksum
section stores the checksum of all package bytes before the Checksum section.
The Checksum section is written last, so runtime verification covers the header,
directory, and all model payload sections that precede it.

Runtime rejects packages with checksum type `1` when the Checksum section is
missing, truncated, or mismatched.

M5 codegen embeds the complete SPK package as a `static const unsigned char[]`
inside generated `model.c`. Generated code also stores a checksum of that
embedded blob and verifies it before calling `spkv2_load_memory`.

Generated deployment files are:

```text
model.h
model.c
main_test.c
CMakeLists.txt
```

Generated `model.c` owns static activation and scratch arenas, then calls
`spkv2_prepare_with_scratch` and binds caller-provided input/output buffers via
`spkv2_bind_input` and `spkv2_bind_output`.
