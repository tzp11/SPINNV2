#include "context.h"
#include "spkv2_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const Spkv2SectionEntry *find_section(const Spkv2Context *ctx, uint32_t kind) {
    for (uint32_t i = 0; i < ctx->header.section_count; i++) {
        if (ctx->sections[i].kind == kind) {
            return &ctx->sections[i];
        }
    }
    return NULL;
}

static int section_bounds_ok(size_t model_size, const Spkv2SectionEntry *section) {
    if (section->offset > model_size) return 0;
    if (section->size > model_size - section->offset) return 0;
    return 1;
}

int spkv2_load_file(const char *path, Spkv2Context **out_ctx) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long size = ftell(fp);
    if (size <= 0) {
        fclose(fp);
        return -1;
    }
    rewind(fp);
    uint8_t *data = (uint8_t *)malloc((size_t)size);
    if (!data) {
        fclose(fp);
        return -1;
    }
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    int rc = spkv2_load_memory(data, (size_t)size, out_ctx);
    if (rc != 0) {
        free(data);
        return rc;
    }
    (*out_ctx)->owned_model = data;
    (*out_ctx)->owned_model_size = (size_t)size;
    return 0;
}

int spkv2_load_memory(const void *data, size_t size, Spkv2Context **out_ctx) {
    if (!data || !out_ctx || size < sizeof(Spkv2Header)) return -1;

    Spkv2Context *ctx = (Spkv2Context *)calloc(1, sizeof(Spkv2Context));
    if (!ctx) return -1;

    memcpy(&ctx->header, data, sizeof(Spkv2Header));
    if (ctx->header.magic != SPKV2_MAGIC) {
        free(ctx);
        return -2;
    }
    if (ctx->header.version_major != SPKV2_VERSION_MAJOR) {
        free(ctx);
        return -2;
    }

    size_t directory_offset = ctx->header.header_size;
    size_t directory_size = (size_t)ctx->header.section_count * sizeof(Spkv2SectionEntry);
    if (directory_offset > size || directory_size > size - directory_offset) {
        free(ctx);
        return -3;
    }

    ctx->model_data = (const uint8_t *)data;
    ctx->model_size = size;
    ctx->sections = (const Spkv2SectionEntry *)(ctx->model_data + directory_offset);

    for (uint32_t i = 0; i < ctx->header.section_count; i++) {
        if (!section_bounds_ok(size, &ctx->sections[i])) {
            free(ctx);
            return -3;
        }
    }

    const Spkv2SectionEntry *tensor_sec = find_section(ctx, SPKV2_SECTION_TENSOR_TABLE);
    const Spkv2SectionEntry *node_sec = find_section(ctx, SPKV2_SECTION_NODE_TABLE);
    const Spkv2SectionEntry *attr_sec = find_section(ctx, SPKV2_SECTION_ATTRIBUTES);
    const Spkv2SectionEntry *weight_sec = find_section(ctx, SPKV2_SECTION_WEIGHTS);
    const Spkv2SectionEntry *memory_plan_sec = find_section(ctx, SPKV2_SECTION_MEMORY_PLAN);
    if (!tensor_sec || !node_sec || !attr_sec || !weight_sec) {
        free(ctx);
        return -4;
    }
    if (tensor_sec->size < ctx->header.num_tensors * sizeof(Spkv2TensorRecord)) {
        free(ctx);
        return -4;
    }
    if (node_sec->size < ctx->header.num_nodes * sizeof(Spkv2NodeRecord)) {
        free(ctx);
        return -4;
    }

    ctx->tensor_records = (const Spkv2TensorRecord *)(ctx->model_data + tensor_sec->offset);
    ctx->node_records = (const Spkv2NodeRecord *)(ctx->model_data + node_sec->offset);
    if (memory_plan_sec && memory_plan_sec->size >= ctx->header.num_tensors * sizeof(Spkv2MemoryPlanRecord)) {
        ctx->memory_plan_records = (const Spkv2MemoryPlanRecord *)(ctx->model_data + memory_plan_sec->offset);
        ctx->memory_plan_count = (size_t)(memory_plan_sec->size / sizeof(Spkv2MemoryPlanRecord));
    }
    ctx->attrs = ctx->model_data + attr_sec->offset;
    ctx->attrs_size = (size_t)attr_sec->size;
    ctx->weights = ctx->model_data + weight_sec->offset;
    ctx->weights_size = (size_t)weight_sec->size;
    ctx->tensors = (Spkv2TensorState *)calloc(ctx->header.num_tensors, sizeof(Spkv2TensorState));
    if (!ctx->tensors) {
        free(ctx);
        return -1;
    }

    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        ctx->tensors[i].record = &ctx->tensor_records[i];
    }

    *out_ctx = ctx;
    return 0;
}

void spkv2_free(Spkv2Context *ctx) {
    if (!ctx) return;
    free(ctx->owned_arena);
    free(ctx->tensors);
    free(ctx->owned_model);
    free(ctx);
}
