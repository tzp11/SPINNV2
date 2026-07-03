#ifndef SPKV2_RUNTIME_H
#define SPKV2_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include "spkv2_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Spkv2Context Spkv2Context;

typedef struct {
    uint32_t node_id;
    uint32_t tensor_id;
    uint64_t element_index;
    uint8_t bit_index;
    uint32_t invocation_index;
    uint64_t seed;
    int enabled;
} Spkv2FaultEvent;

typedef struct {
    uint64_t injected_faults;
    uint64_t detected_faults;
    uint64_t recovered_faults;
    uint64_t unrecovered_faults;
    uint64_t rerun_count;
} Spkv2ReliabilityStats;

typedef struct {
    uint32_t node_id;
    uint32_t tensor_id;
    float observed_min;
    float observed_max;
    uint64_t observations;
} Spkv2RangeObservation;

typedef int (*Spkv2NodeKernelFn)(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);

const char *spkv2_runtime_version(void);

int spkv2_load_file(const char *path, Spkv2Context **out_ctx);
int spkv2_load_memory(const void *data, size_t size, Spkv2Context **out_ctx);
int spkv2_prepare(Spkv2Context *ctx, void *arena, size_t arena_size);
int spkv2_prepare_with_scratch(
    Spkv2Context *ctx,
    void *arena,
    size_t arena_size,
    void *scratch,
    size_t scratch_size);
int spkv2_set_input(Spkv2Context *ctx, int index, const void *data, size_t size);
int spkv2_bind_input(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_bind_output(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_get_output_size(Spkv2Context *ctx, int index, size_t *out_size);
int spkv2_get_output(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_run(Spkv2Context *ctx);
int spkv2_set_fault_event(Spkv2Context *ctx, const Spkv2FaultEvent *event);
void spkv2_clear_fault_event(Spkv2Context *ctx);
int spkv2_get_reliability_stats(Spkv2Context *ctx, Spkv2ReliabilityStats *stats);
void spkv2_reset_reliability_stats(Spkv2Context *ctx);
int spkv2_get_range_observation(Spkv2Context *ctx, uint32_t node_id, Spkv2RangeObservation *observation);
void spkv2_reset_range_observations(Spkv2Context *ctx);
void spkv2_profile_dump(void);
void spkv2_free(Spkv2Context *ctx);

int spkv2_register_kernel(uint16_t op_type, uint16_t backend, uint16_t kernel_kind,
                           Spkv2NodeKernelFn fn, int priority);
void spkv2_clear_user_kernels(void);

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_RUNTIME_H */
