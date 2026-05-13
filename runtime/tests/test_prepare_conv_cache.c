#include "context.h"
#include "spkv2_format.h"
#include "spkv2_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int main(void) {
    enum { SECTION_COUNT = 5 };
    uint8_t buffer[2048];
    memset(buffer, 0, sizeof(buffer));

    Spkv2Header *header = (Spkv2Header *)buffer;
    header->magic = SPKV2_MAGIC;
    header->version_major = SPKV2_VERSION_MAJOR;
    header->version_minor = SPKV2_VERSION_MINOR;
    header->header_size = sizeof(Spkv2Header);
    header->section_count = SECTION_COUNT;
    header->num_tensors = 3;
    header->num_nodes = 1;
    header->num_inputs = 1;
    header->num_outputs = 1;
    header->weight_bytes = 8u * 4u * sizeof(float);
    header->activation_arena_bytes = 0;
    header->scratch_arena_bytes = 0;

    Spkv2SectionEntry *sections = (Spkv2SectionEntry *)(buffer + sizeof(Spkv2Header));
    size_t offset = sizeof(Spkv2Header) + SECTION_COUNT * sizeof(Spkv2SectionEntry);

    sections[0].kind = SPKV2_SECTION_TENSOR_TABLE;
    sections[0].offset = offset;
    sections[0].size = 3u * sizeof(Spkv2TensorRecord);
    offset += (size_t)sections[0].size;

    sections[1].kind = SPKV2_SECTION_NODE_TABLE;
    sections[1].offset = offset;
    sections[1].size = sizeof(Spkv2NodeRecord);
    offset += (size_t)sections[1].size;

    sections[2].kind = SPKV2_SECTION_ATTRIBUTES;
    sections[2].offset = offset;
    sections[2].size = sizeof(Spkv2AttrRecord);
    offset += (size_t)sections[2].size;

    offset = align_up(offset, 16);
    sections[3].kind = SPKV2_SECTION_WEIGHTS;
    sections[3].offset = offset;
    sections[3].size = header->weight_bytes;
    offset += (size_t)sections[3].size;

    sections[4].kind = SPKV2_SECTION_KERNEL_SPEC;
    sections[4].offset = offset;
    sections[4].size = sizeof(Spkv2KernelSpecRecord);
    offset += (size_t)sections[4].size;

    Spkv2TensorRecord *tensors = (Spkv2TensorRecord *)(buffer + sections[0].offset);
    tensors[0].id = 0;
    tensors[0].dtype = SPKV2_DTYPE_FP32;
    tensors[0].role = SPKV2_ROLE_INPUT;
    tensors[0].rank = 4;
    tensors[0].memory_class = SPKV2_MEMORY_EXTERNAL;
    tensors[0].shape[0] = 1;
    tensors[0].shape[1] = 4;
    tensors[0].shape[2] = 2;
    tensors[0].shape[3] = 2;
    tensors[0].size_bytes = 1u * 4u * 2u * 2u * sizeof(float);

    tensors[1].id = 1;
    tensors[1].dtype = SPKV2_DTYPE_FP32;
    tensors[1].role = SPKV2_ROLE_WEIGHT;
    tensors[1].rank = 4;
    tensors[1].memory_class = SPKV2_MEMORY_WEIGHT;
    tensors[1].shape[0] = 8;
    tensors[1].shape[1] = 4;
    tensors[1].shape[2] = 1;
    tensors[1].shape[3] = 1;
    tensors[1].size_bytes = header->weight_bytes;
    tensors[1].data_offset = 0;

    tensors[2].id = 2;
    tensors[2].dtype = SPKV2_DTYPE_FP32;
    tensors[2].role = SPKV2_ROLE_OUTPUT;
    tensors[2].rank = 4;
    tensors[2].memory_class = SPKV2_MEMORY_EXTERNAL;
    tensors[2].shape[0] = 1;
    tensors[2].shape[1] = 8;
    tensors[2].shape[2] = 2;
    tensors[2].shape[3] = 2;
    tensors[2].size_bytes = 1u * 8u * 2u * 2u * sizeof(float);

    Spkv2NodeRecord *node = (Spkv2NodeRecord *)(buffer + sections[1].offset);
    node->id = 0;
    node->op_type = SPKV2_OP_CONV;
    node->input_count = 2;
    node->output_count = 1;
    node->inputs[0] = 0;
    node->inputs[1] = 1;
    node->outputs[0] = 2;
    node->attr_offset = 0;
    node->attr_size = sizeof(Spkv2AttrRecord);
    node->kernel_spec_id = 0;
    node->scratch_bytes = 0;

    Spkv2AttrRecord *attr = (Spkv2AttrRecord *)(buffer + sections[2].offset);
    attr->op_type = SPKV2_OP_CONV;
    attr->group = 1;
    attr->strides[0] = 1;
    attr->strides[1] = 1;
    attr->kernel_shape[0] = 1;
    attr->kernel_shape[1] = 1;
    attr->dilations[0] = 1;
    attr->dilations[1] = 1;

    Spkv2KernelSpecRecord *spec = (Spkv2KernelSpecRecord *)(buffer + sections[4].offset);
    spec->id = 0;
    spec->node_id = 0;
    spec->kernel_kind = SPKV2_KERNEL_POINTWISE_1X1;
    spec->backend = SPKV2_BACKEND_SIMD;
    spec->dtype = SPKV2_DTYPE_FP32;
    spec->layout = 1;
    spec->weight_layout = 1;
    spec->fallback_kernel_spec_id = 0xFFFFFFFFu;

    Spkv2Context *ctx = NULL;
    if (spkv2_load_memory(buffer, offset, &ctx) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    if (spkv2_prepare(ctx, NULL, 0) != 0) {
        fprintf(stderr, "prepare failed\n");
        spkv2_free(ctx);
        return 1;
    }

    if (!ctx->node_cache || !ctx->node_cache[0]) {
        fprintf(stderr, "prepare should create packed conv cache\n");
        spkv2_free(ctx);
        return 1;
    }

    spkv2_free(ctx);
    return 0;
}
