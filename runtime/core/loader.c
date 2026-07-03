#include "context.h"
#include "spkv2_platform.h"
#include "spkv2_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

static uint32_t fnv1a32(const uint8_t *data, size_t size) {
    uint32_t value = 2166136261u;
    for (size_t i = 0; i < size; i++) {
        value ^= data[i];
        value *= 16777619u;
    }
    return value;
}

static void skip_json_ws(const char **p, const char *end) {
    while (*p < end && isspace((unsigned char)**p)) (*p)++;
}

static const char *find_bytes(const char *begin, const char *end, const char *needle, size_t needle_size) {
    if (needle_size == 0) return begin;
    for (const char *p = begin; p + needle_size <= end; p++) {
        if (memcmp(p, needle, needle_size) == 0) return p;
    }
    return NULL;
}

static int parse_metadata_outputs(Spkv2Context *ctx, const Spkv2SectionEntry *metadata_sec) {
    if (!metadata_sec || metadata_sec->size == 0 || ctx->header.num_outputs == 0) return 0;

    const char *begin = (const char *)(ctx->model_data + metadata_sec->offset);
    const char *end = begin + metadata_sec->size;
    const char *key = "\"outputs\"";
    const char *p = begin;
    while (p + strlen(key) <= end) {
        const char *match = find_bytes(p, end, key, strlen(key));
        if (!match) return 0;
        p = match + strlen(key);
        skip_json_ws(&p, end);
        if (p >= end || *p != ':') continue;
        p++;
        skip_json_ws(&p, end);
        if (p >= end || *p != '[') continue;
        p++;

        uint32_t *ids = (uint32_t *)spkv2_platform_calloc(ctx->header.num_outputs, sizeof(uint32_t));
        if (!ids) return -1;
        size_t count = 0;
        for (;;) {
            skip_json_ws(&p, end);
            if (p >= end) break;
            if (*p == ']') {
                p++;
                break;
            }
            char *next = NULL;
            unsigned long value = strtoul(p, &next, 10);
            if (next == p || value >= ctx->header.num_tensors || count >= ctx->header.num_outputs) {
                spkv2_platform_free(ids);
                return 0;
            }
            ids[count++] = (uint32_t)value;
            p = next;
            skip_json_ws(&p, end);
            if (p < end && *p == ',') {
                p++;
                continue;
            }
            if (p < end && *p == ']') {
                p++;
                break;
            }
            spkv2_platform_free(ids);
            return 0;
        }
        if (count == ctx->header.num_outputs) {
            ctx->output_ids = ids;
            ctx->output_count = count;
        } else {
            spkv2_platform_free(ids);
        }
        return 0;
    }
    return 0;
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
    uint8_t *data = (uint8_t *)spkv2_platform_malloc((size_t)size);
    if (!data) {
        fclose(fp);
        return -1;
    }
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {
        spkv2_platform_free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    int rc = spkv2_load_memory(data, (size_t)size, out_ctx);
    if (rc != 0) {
        spkv2_platform_free(data);
        return rc;
    }
    (*out_ctx)->owned_model = data;
    (*out_ctx)->owned_model_size = (size_t)size;
    return 0;
}

int spkv2_load_memory(const void *data, size_t size, Spkv2Context **out_ctx) {
    if (!data || !out_ctx || size < sizeof(Spkv2Header)) return -1;

    Spkv2Context *ctx = (Spkv2Context *)spkv2_platform_calloc(1, sizeof(Spkv2Context));
    if (!ctx) return -1;

    memcpy(&ctx->header, data, sizeof(Spkv2Header));
    if (ctx->header.magic != SPKV2_MAGIC) {
        spkv2_platform_free(ctx);
        return -2;
    }
    if (ctx->header.version_major != SPKV2_VERSION_MAJOR) {
        spkv2_platform_free(ctx);
        return -2;
    }

    size_t directory_offset = ctx->header.header_size;
    size_t directory_size = (size_t)ctx->header.section_count * sizeof(Spkv2SectionEntry);
    if (directory_offset > size || directory_size > size - directory_offset) {
        spkv2_platform_free(ctx);
        return -3;
    }

    ctx->model_data = (const uint8_t *)data;
    ctx->model_size = size;
    ctx->sections = (const Spkv2SectionEntry *)(ctx->model_data + directory_offset);

    for (uint32_t i = 0; i < ctx->header.section_count; i++) {
        if (!section_bounds_ok(size, &ctx->sections[i])) {
            spkv2_platform_free(ctx);
            return -3;
        }
    }

    const Spkv2SectionEntry *tensor_sec = find_section(ctx, SPKV2_SECTION_TENSOR_TABLE);
    const Spkv2SectionEntry *node_sec = find_section(ctx, SPKV2_SECTION_NODE_TABLE);
    const Spkv2SectionEntry *metadata_sec = find_section(ctx, SPKV2_SECTION_METADATA);
    const Spkv2SectionEntry *attr_sec = find_section(ctx, SPKV2_SECTION_ATTRIBUTES);
    const Spkv2SectionEntry *weight_sec = find_section(ctx, SPKV2_SECTION_WEIGHTS);
    const Spkv2SectionEntry *memory_plan_sec = find_section(ctx, SPKV2_SECTION_MEMORY_PLAN);
    const Spkv2SectionEntry *kernel_spec_sec = find_section(ctx, SPKV2_SECTION_KERNEL_SPEC);
    const Spkv2SectionEntry *protection_sec = find_section(ctx, SPKV2_SECTION_PROTECTION_PLAN);
    const Spkv2SectionEntry *checksum_sec = find_section(ctx, SPKV2_SECTION_CHECKSUM);
    if (!tensor_sec || !node_sec || !attr_sec || !weight_sec) {
        spkv2_platform_free(ctx);
        return -4;
    }
    if (tensor_sec->size < ctx->header.num_tensors * sizeof(Spkv2TensorRecord)) {
        spkv2_platform_free(ctx);
        return -4;
    }
    if (node_sec->size < ctx->header.num_nodes * sizeof(Spkv2NodeRecord)) {
        spkv2_platform_free(ctx);
        return -4;
    }
    if (ctx->header.checksum_type == 1) {
        if (!checksum_sec || checksum_sec->size < sizeof(uint32_t)) {
            spkv2_platform_free(ctx);
            return -5;
        }
        uint32_t expected = 0;
        memcpy(&expected, ctx->model_data + checksum_sec->offset, sizeof(expected));
        uint32_t actual = fnv1a32(ctx->model_data, (size_t)checksum_sec->offset);
        if (actual != expected) {
            spkv2_platform_free(ctx);
            return -6;
        }
    }

    ctx->tensor_records = (const Spkv2TensorRecord *)(ctx->model_data + tensor_sec->offset);
    ctx->node_records = (const Spkv2NodeRecord *)(ctx->model_data + node_sec->offset);
    if (memory_plan_sec && memory_plan_sec->size >= ctx->header.num_tensors * sizeof(Spkv2MemoryPlanRecord)) {
        ctx->memory_plan_records = (const Spkv2MemoryPlanRecord *)(ctx->model_data + memory_plan_sec->offset);
        ctx->memory_plan_count = (size_t)(memory_plan_sec->size / sizeof(Spkv2MemoryPlanRecord));
    }
    if (kernel_spec_sec && kernel_spec_sec->size >= ctx->header.num_nodes * sizeof(Spkv2KernelSpecRecord)) {
        ctx->kernel_spec_records = (const Spkv2KernelSpecRecord *)(ctx->model_data + kernel_spec_sec->offset);
        ctx->kernel_spec_count = (size_t)(kernel_spec_sec->size / sizeof(Spkv2KernelSpecRecord));
    }

    /* Parse optional SECTION_QUANTIZATION (INT8 models only) */
    const Spkv2SectionEntry *quant_sec = find_section(ctx, SPKV2_SECTION_QUANTIZATION);
    if (quant_sec && quant_sec->size >= sizeof(Spkv2QuantParamRecord)) {
        const uint8_t *qbase = ctx->model_data + quant_sec->offset;
        const Spkv2QuantParamRecord *first = (const Spkv2QuantParamRecord *)qbase;
        if (first->data_offset > 0) {
            size_t count = first->data_offset / sizeof(Spkv2QuantParamRecord);
            if (count > 0 && count * sizeof(Spkv2QuantParamRecord) <= quant_sec->size) {
                ctx->quant_param_records = (const Spkv2QuantParamRecord *)qbase;
                ctx->quant_param_count = count;
                ctx->quant_data = qbase;
                ctx->quant_data_size = (size_t)quant_sec->size;
            }
        }
    }

    if (protection_sec && protection_sec->size >= sizeof(Spkv2ProtectionRecord)) {
        ctx->protection_records = (const Spkv2ProtectionRecord *)(ctx->model_data + protection_sec->offset);
        ctx->protection_count = (size_t)(protection_sec->size / sizeof(Spkv2ProtectionRecord));
    }

    ctx->attrs = ctx->model_data + attr_sec->offset;
    ctx->attrs_size = (size_t)attr_sec->size;
    ctx->weights = ctx->model_data + weight_sec->offset;
    ctx->weights_size = (size_t)weight_sec->size;
    ctx->tensors = (Spkv2TensorState *)spkv2_platform_calloc(ctx->header.num_tensors, sizeof(Spkv2TensorState));
    if (!ctx->tensors) {
        spkv2_platform_free(ctx);
        return -1;
    }

    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        ctx->tensors[i].record = &ctx->tensor_records[i];
    }

    int metadata_rc = parse_metadata_outputs(ctx, metadata_sec);
    if (metadata_rc != 0) {
        spkv2_platform_free(ctx->tensors);
        spkv2_platform_free(ctx);
        return metadata_rc;
    }

    ctx->node_cache_count = ctx->header.num_nodes;
    if (ctx->node_cache_count > 0) {
        ctx->node_cache = (void **)spkv2_platform_calloc(ctx->node_cache_count, sizeof(void *));
        ctx->node_cache_dtors = (void (**)(void *))spkv2_platform_calloc(ctx->node_cache_count, sizeof(void (*)(void *)));
        ctx->node_invocation_counts = (uint32_t *)spkv2_platform_calloc(ctx->node_cache_count, sizeof(uint32_t));
        ctx->range_observations = (Spkv2RangeObservation *)spkv2_platform_calloc(
            ctx->node_cache_count, sizeof(Spkv2RangeObservation));
        if (!ctx->node_cache || !ctx->node_invocation_counts || !ctx->range_observations) {
            spkv2_platform_free(ctx->range_observations);
            spkv2_platform_free(ctx->node_invocation_counts);
            spkv2_platform_free(ctx->node_cache_dtors);
            spkv2_platform_free(ctx->node_cache);
            spkv2_platform_free(ctx->tensors);
            spkv2_platform_free(ctx);
            return -1;
        }
    }

    *out_ctx = ctx;
    return 0;
}

void spkv2_free(Spkv2Context *ctx) {
    if (!ctx) return;
    if (ctx->node_cache) {
        for (size_t i = 0; i < ctx->node_cache_count; i++) {
            if (!ctx->node_cache[i]) continue;
            if (ctx->node_cache_dtors && ctx->node_cache_dtors[i])
                ctx->node_cache_dtors[i](ctx->node_cache[i]);
            else
                spkv2_platform_free(ctx->node_cache[i]);
        }
        spkv2_platform_free(ctx->node_cache);
    }
    spkv2_platform_free(ctx->node_cache_dtors);
    spkv2_platform_free(ctx->owned_scratch);
    spkv2_platform_free(ctx->owned_arena);
    spkv2_platform_free(ctx->output_ids);
    spkv2_platform_free(ctx->tensors);
    spkv2_platform_free(ctx->range_observations);
    spkv2_platform_free(ctx->node_invocation_counts);
    spkv2_platform_free(ctx->owned_model);
    spkv2_platform_free(ctx);
}
