#ifndef SPKV2_FORMAT_H
#define SPKV2_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPKV2_MAGIC 0x32564B50u /* 'PKV2' little-endian */
#define SPKV2_VERSION_MAJOR 0u
#define SPKV2_VERSION_MINOR 3u

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
    SPKV2_SECTION_CHECKSUM = 12,
    SPKV2_SECTION_PROTECTION_PLAN = 13
} Spkv2SectionKind;

#pragma pack(push, 1)

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
    SPKV2_DTYPE_FP32 = 1,
    SPKV2_DTYPE_INT8 = 2,
    SPKV2_DTYPE_FP16 = 3
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
    SPKV2_OP_SOFTMAX = 7,
    SPKV2_OP_MUL = 8,
    SPKV2_OP_SUB = 9,
    SPKV2_OP_DIV = 10,
    SPKV2_OP_MOD = 11,
    SPKV2_OP_SIGMOID = 12,
    SPKV2_OP_RESHAPE = 13,
    SPKV2_OP_TRANSPOSE = 14,
    SPKV2_OP_CONCAT = 15,
    SPKV2_OP_SPLIT = 16,
    SPKV2_OP_REDUCEMEAN = 17,
    SPKV2_OP_REDUCEMAX = 18,
    SPKV2_OP_MATMUL = 19,
    SPKV2_OP_RESIZE = 20,
    SPKV2_OP_TILE = 21,
    SPKV2_OP_UNSQUEEZE = 22,
    SPKV2_OP_GATHERELEMENTS = 23,
    SPKV2_OP_TOPK = 24,
    SPKV2_OP_CAST = 25,
    SPKV2_OP_SLICE = 26,
    SPKV2_OP_GATHER = 27,
    SPKV2_OP_AVERAGEPOOL = 28,
    SPKV2_OP_GLOBALAVGPOOL = 29,
    SPKV2_OP_CLIP = 30,
    SPKV2_OP_LEAKYRELU = 31
} Spkv2OpType;

typedef enum {
    SPKV2_BACKEND_REF = 1,
    SPKV2_BACKEND_CPU = 2,
    SPKV2_BACKEND_SIMD = 3,
    SPKV2_BACKEND_DELEGATE = 4
} Spkv2Backend;

typedef enum {
    SPKV2_KERNEL_REFERENCE = 1,
    SPKV2_KERNEL_DIRECT = 2,
    SPKV2_KERNEL_IM2COL_GEMM = 3,
    SPKV2_KERNEL_PACKED_GEMM = 4,
    SPKV2_KERNEL_POINTWISE_1X1 = 5,
    SPKV2_KERNEL_DEPTHWISE_DIRECT = 6,
    SPKV2_KERNEL_WINOGRAD_3X3S1 = 7,
    SPKV2_KERNEL_CONV3X3S2_DIRECT = 8,
    SPKV2_KERNEL_WINOGRAD_F43 = 9,
    SPKV2_KERNEL_INT8_IM2COL_GEMM = 10,
    SPKV2_KERNEL_BNNS_FP32 = 11
} Spkv2KernelKind;

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
    int32_t fused_activation;
    int32_t pads[4];
    int32_t strides[2];
    int32_t kernel_shape[2];
    int32_t dilations[2];
    int32_t trans_a;
    int32_t trans_b;
    float alpha;
    float beta;
    int32_t extra_count;
    int32_t keepdims;
    int32_t extra[8];
    int32_t largest;
    int32_t sorted;
    int32_t cast_to;
} Spkv2AttrRecord;

typedef struct {
    uint32_t tensor_id;
    uint16_t memory_class;
    uint16_t alignment;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;
    uint32_t first_use;
    uint32_t last_use;
} Spkv2MemoryPlanRecord;

typedef struct {
    uint32_t id;
    uint32_t node_id;
    uint16_t kernel_kind;
    uint16_t backend;
    uint16_t dtype;
    uint16_t layout;
    uint16_t weight_layout;
    uint16_t flags;
    uint64_t scratch_offset;
    uint64_t scratch_bytes;
    uint32_t fallback_kernel_spec_id;
    uint32_t required_feature_mask;
} Spkv2KernelSpecRecord;

/* Per-channel quantization parameters (SECTION_QUANTIZATION).
 * One record per quantized tensor. scale/zero_point arrays follow
 * the record header at data_offset within the section. */
typedef enum {
    SPKV2_QUANT_NONE = 0,
    SPKV2_QUANT_PER_TENSOR = 1,
    SPKV2_QUANT_PER_CHANNEL = 2
} Spkv2QuantScheme;

typedef struct {
    uint32_t tensor_id;
    uint16_t scheme;        /* Spkv2QuantScheme */
    uint16_t quant_axis;    /* channel axis for per-channel */
    uint32_t num_channels;  /* 1 for per-tensor, C for per-channel */
    uint64_t data_offset;   /* offset to scale[]/zero_point[] arrays in section */
} Spkv2QuantParamRecord;

typedef enum {
    SPKV2_PROTECT_NONE = 0,
    SPKV2_PROTECT_RANGE_GUARD_RERUN = 1,
    SPKV2_PROTECT_DMR_COMPARE_RERUN = 2
} Spkv2ProtectionMode;

typedef struct {
    uint32_t node_id;
    uint32_t tensor_id;
    uint16_t mode;
    uint16_t flags;
    float lower_bound;
    float upper_bound;
    uint64_t scratch_offset;
} Spkv2ProtectionRecord;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_FORMAT_H */
