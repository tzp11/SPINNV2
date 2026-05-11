#include "context.h"
#include "spkv2_runtime.h"

#include <stdlib.h>
#include <string.h>

static uint64_t align16(uint64_t value) {
    return (value + 15u) & ~15u;
}

int spkv2_prepare(Spkv2Context *ctx, void *arena, size_t arena_size) {
    if (!ctx) return -1;

    uint64_t required = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role == SPKV2_ROLE_WEIGHT) continue;
        required += align16(record->size_bytes);
    }

    if (arena) {
        if (arena_size < required) return -2;
        ctx->owned_arena = NULL;
        ctx->arena_size = arena_size;
    } else {
        arena = calloc(1, (size_t)required);
        if (!arena && required > 0) return -1;
        ctx->owned_arena = (uint8_t *)arena;
        ctx->arena_size = (size_t)required;
    }

    uint8_t *cursor = (uint8_t *)arena;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role == SPKV2_ROLE_WEIGHT) {
            if (record->data_offset > ctx->weights_size ||
                record->size_bytes > ctx->weights_size - record->data_offset) {
                return -3;
            }
            ctx->tensors[i].data = (uint8_t *)(ctx->weights + record->data_offset);
        } else {
            ctx->tensors[i].data = cursor;
            cursor += align16(record->size_bytes);
        }
    }
    return 0;
}

int spkv2_set_input(Spkv2Context *ctx, int index, const void *data, size_t size) {
    if (!ctx || !data || index < 0) return -1;
    int current = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_INPUT) continue;
        if (current == index) {
            if (size != record->size_bytes) return -2;
            memcpy(ctx->tensors[i].data, data, size);
            return 0;
        }
        current++;
    }
    return -3;
}

int spkv2_get_output(Spkv2Context *ctx, int index, void *data, size_t size) {
    if (!ctx || !data || index < 0) return -1;
    int current = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_OUTPUT) continue;
        if (current == index) {
            if (size != record->size_bytes) return -2;
            memcpy(data, ctx->tensors[i].data, size);
            return 0;
        }
        current++;
    }
    return -3;
}

int spkv2_get_output_size(Spkv2Context *ctx, int index, size_t *out_size) {
    if (!ctx || !out_size || index < 0) return -1;
    int current = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_OUTPUT) continue;
        if (current == index) {
            *out_size = (size_t)record->size_bytes;
            return 0;
        }
        current++;
    }
    return -3;
}

int spkv2_run(Spkv2Context *ctx) {
    if (!ctx) return -1;
    for (uint32_t i = 0; i < ctx->header.num_nodes; i++) {
        int rc = spkv2_execute_node(ctx, &ctx->node_records[i]);
        if (rc != 0) return rc;
    }
    return 0;
}
