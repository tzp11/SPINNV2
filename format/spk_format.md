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

