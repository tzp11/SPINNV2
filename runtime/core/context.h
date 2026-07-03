#ifndef SPKV2_CONTEXT_H
#define SPKV2_CONTEXT_H

#include "spkv2_format.h"
#include "spkv2_runtime.h"

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
    const Spkv2KernelSpecRecord *kernel_spec_records;
    size_t kernel_spec_count;
    const Spkv2ProtectionRecord *protection_records;
    size_t protection_count;
    const uint8_t *attrs;
    size_t attrs_size;
    const uint8_t *weights;
    size_t weights_size;
    uint32_t *output_ids;
    size_t output_count;

    Spkv2TensorState *tensors;
    uint8_t *owned_arena;
    size_t arena_size;
    uint8_t *owned_scratch;
    uint8_t *scratch;
    size_t scratch_size;

    void **node_cache;       /* per-node opaque cache (e.g. packed weights) */
    void (**node_cache_dtors)(void*); /* per-node custom destructor (NULL = platform_free) */
    size_t node_cache_count;

    const Spkv2QuantParamRecord *quant_param_records;
    size_t quant_param_count;
    const uint8_t *quant_data;
    size_t quant_data_size;

    Spkv2FaultEvent fault_event;
    Spkv2ReliabilityStats reliability_stats;
    Spkv2RangeObservation *range_observations;
    uint32_t *node_invocation_counts;
} Spkv2Context;

int spkv2_execute_node(Spkv2Context *ctx, const Spkv2NodeRecord *node);

static inline const Spkv2QuantParamRecord *
spkv2_find_quant_param(const Spkv2Context *ctx, uint32_t tensor_id) {
    for (size_t i = 0; i < ctx->quant_param_count; i++)
        if (ctx->quant_param_records[i].tensor_id == tensor_id)
            return &ctx->quant_param_records[i];
    return NULL;
}

static inline const float *
spkv2_quant_scales(const Spkv2Context *ctx, const Spkv2QuantParamRecord *qp) {
    if (!qp || !ctx->quant_data) return NULL;
    return (const float *)(ctx->quant_data + qp->data_offset);
}

#endif /* SPKV2_CONTEXT_H */
