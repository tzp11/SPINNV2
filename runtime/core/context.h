#ifndef SPKV2_CONTEXT_H
#define SPKV2_CONTEXT_H

#include "spkv2_format.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const Spkv2TensorRecord *record;
    uint8_t *data;
} Spkv2TensorState;

typedef struct Spkv2Context {
    uint8_t *owned_model;
    size_t owned_model_size;
    const uint8_t *model_data;
    size_t model_size;

    Spkv2Header header;
    const Spkv2SectionEntry *sections;

    const Spkv2TensorRecord *tensor_records;
    const Spkv2NodeRecord *node_records;
    const Spkv2MemoryPlanRecord *memory_plan_records;
    size_t memory_plan_count;
    const uint8_t *attrs;
    size_t attrs_size;
    const uint8_t *weights;
    size_t weights_size;

    Spkv2TensorState *tensors;
    uint8_t *owned_arena;
    size_t arena_size;
} Spkv2Context;

int spkv2_execute_node(Spkv2Context *ctx, const Spkv2NodeRecord *node);

#endif /* SPKV2_CONTEXT_H */
