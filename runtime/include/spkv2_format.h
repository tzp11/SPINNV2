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

typedef enum {
    SPKV2_DTYPE_FP32 = 1
} Spkv2DType;

typedef enum {
    SPKV2_ROLE_INPUT = 1,
    SPKV2_ROLE_OUTPUT = 2,
    SPKV2_ROLE_WEIGHT = 3,
    SPKV2_ROLE_ACTIVATION = 4,
    SPKV2_ROLE_CONSTANT = 5
} Spkv2TensorRole;

typedef enum {
    SPKV2_MEMORY_INPUT = 1,
    SPKV2_MEMORY_OUTPUT = 2,
    SPKV2_MEMORY_WEIGHT = 3,
    SPKV2_MEMORY_ACTIVATION_ARENA = 4,
    SPKV2_MEMORY_EXTERNAL = 5
} Spkv2MemoryClass;

typedef enum {
    SPKV2_OP_ADD = 1,
    SPKV2_OP_CONV = 2,
    SPKV2_OP_FLATTEN = 3,
    SPKV2_OP_GEMM = 4,
    SPKV2_OP_MAXPOOL = 5,
    SPKV2_OP_RELU = 6,
    SPKV2_OP_SOFTMAX = 7
} Spkv2OpType;

#pragma pack(push, 1)

typedef struct {
    uint32_t id;
    uint16_t dtype;
    uint16_t role;
    uint16_t rank;
    uint16_t memory_class;
    uint32_t shape[8];
    uint64_t size_bytes;
    uint64_t data_offset;
    uint32_t name_offset;
    uint32_t reserved;
} Spkv2TensorRecord;

typedef struct {
    uint32_t id;
    uint16_t op_type;
    uint16_t flags;
    uint16_t input_count;
    uint16_t output_count;
    uint32_t inputs[8];
    uint32_t outputs[4];
    uint32_t attr_offset;
    uint32_t attr_size;
    uint32_t kernel_spec_id;
    uint32_t scratch_bytes;
} Spkv2NodeRecord;

typedef struct {
    uint32_t op_type;
    int32_t axis;
    int32_t group;
    int32_t pads[4];
    int32_t strides[2];
    int32_t kernel_shape[2];
    int32_t dilations[2];
    int32_t trans_a;
    int32_t trans_b;
    float alpha;
} Spkv2AttrRecord;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_FORMAT_H */
