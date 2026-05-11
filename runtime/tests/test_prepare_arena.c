#include "spkv2_format.h"
#include "spkv2_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { SECTION_COUNT = 4 };
    uint8_t buffer[512];
    memset(buffer, 0, sizeof(buffer));

    Spkv2Header *header = (Spkv2Header *)buffer;
    header->magic = SPKV2_MAGIC;
    header->version_major = SPKV2_VERSION_MAJOR;
    header->version_minor = SPKV2_VERSION_MINOR;
    header->header_size = sizeof(Spkv2Header);
    header->section_count = SECTION_COUNT;
    header->num_tensors = 1;
    header->num_nodes = 0;
    header->num_inputs = 1;
    header->num_outputs = 0;
    header->activation_arena_bytes = 16;

    Spkv2SectionEntry *sections = (Spkv2SectionEntry *)(buffer + sizeof(Spkv2Header));
    size_t offset = sizeof(Spkv2Header) + SECTION_COUNT * sizeof(Spkv2SectionEntry);

    sections[0].kind = SPKV2_SECTION_TENSOR_TABLE;
    sections[0].offset = offset;
    sections[0].size = sizeof(Spkv2TensorRecord);
    offset += sizeof(Spkv2TensorRecord);

    sections[1].kind = SPKV2_SECTION_NODE_TABLE;
    sections[1].offset = offset;
    sections[1].size = 0;

    sections[2].kind = SPKV2_SECTION_ATTRIBUTES;
    sections[2].offset = offset;
    sections[2].size = 0;

    sections[3].kind = SPKV2_SECTION_WEIGHTS;
    sections[3].offset = offset;
    sections[3].size = 0;

    Spkv2TensorRecord *tensor = (Spkv2TensorRecord *)(buffer + sections[0].offset);
    tensor->id = 0;
    tensor->dtype = SPKV2_DTYPE_FP32;
    tensor->role = SPKV2_ROLE_INPUT;
    tensor->rank = 1;
    tensor->memory_class = SPKV2_MEMORY_INPUT;
    tensor->shape[0] = 4;
    tensor->size_bytes = 16;

    Spkv2Context *ctx = NULL;
    if (spkv2_load_memory(buffer, sizeof(buffer), &ctx) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    uint8_t too_small[8];
    if (spkv2_prepare(ctx, too_small, sizeof(too_small)) != -2) {
        fprintf(stderr, "small arena should fail\n");
        spkv2_free(ctx);
        return 1;
    }

    uint8_t enough[16];
    if (spkv2_prepare(ctx, enough, sizeof(enough)) != 0) {
        fprintf(stderr, "sufficient arena should pass\n");
        spkv2_free(ctx);
        return 1;
    }

    spkv2_free(ctx);
    return 0;
}

