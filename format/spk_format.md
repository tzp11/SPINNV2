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
| `version_major/minor` | Format version |
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
