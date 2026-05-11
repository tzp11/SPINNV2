#ifndef SPKV2_FORMAT_H
#define SPKV2_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPKV2_MAGIC 0x32564B50u /* 'PKV2' little-endian */
#define SPKV2_VERSION_MAJOR 0u
#define SPKV2_VERSION_MINOR 1u

typedef enum {
    SPKV2_SECTION_METADATA = 1,
    SPKV2_SECTION_TARGET_PROFILE = 2,
    SPKV2_SECTION_TENSOR_TABLE = 3,
    SPKV2_SECTION_NODE_TABLE = 4,
    SPKV2_SECTION_ATTRIBUTES = 5,
    SPKV2_SECTION_WEIGHTS = 6,
    SPKV2_SECTION_MEMORY_PLAN = 7,
    SPKV2_SECTION_KERNEL_SPEC = 8,
    SPKV2_SECTION_QUANTIZATION = 9,
    SPKV2_SECTION_STRING_TABLE = 10,
    SPKV2_SECTION_DEBUG = 11,
    SPKV2_SECTION_CHECKSUM = 12
} Spkv2SectionKind;

typedef struct {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint16_t endianness;
    uint16_t header_size;
    uint32_t section_count;
    uint32_t model_flags;
    uint32_t num_tensors;
    uint32_t num_nodes;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint64_t weight_bytes;
    uint64_t activation_arena_bytes;
    uint64_t scratch_arena_bytes;
    uint32_t target_profile_hash;
    uint32_t checksum_type;
} Spkv2Header;

typedef struct {
    uint32_t kind;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;
    uint32_t alignment;
    uint32_t reserved;
} Spkv2SectionEntry;

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_FORMAT_H */

