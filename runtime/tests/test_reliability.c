#include "spkv2_runtime.h"

#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: spkv2_reliability_test protected.spk\n");
        return 2;
    }
    Spkv2Context *ctx = NULL;
    if (spkv2_load_file(argv[1], &ctx) != 0 || spkv2_prepare(ctx, NULL, 0) != 0) {
        fprintf(stderr, "failed to load or prepare protected model\n");
        return 1;
    }
    float input[4] = {-1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    Spkv2FaultEvent event = {0};
    event.node_id = 0;
    event.tensor_id = 1;
    event.element_index = 1;
    event.bit_index = 31;
    event.invocation_index = 1;
    event.seed = 2026;
    event.enabled = 1;
    if (spkv2_set_input(ctx, 0, input, sizeof(input)) != 0 ||
        spkv2_set_fault_event(ctx, &event) != 0 ||
        spkv2_run(ctx) != 0 ||
        spkv2_get_output(ctx, 0, output, sizeof(output)) != 0) {
        fprintf(stderr, "protected execution failed\n");
        spkv2_free(ctx);
        return 1;
    }
    Spkv2ReliabilityStats stats;
    if (spkv2_get_reliability_stats(ctx, &stats) != 0 ||
        stats.injected_faults != 1 || stats.detected_faults != 1 ||
        stats.recovered_faults != 1 || stats.rerun_count != 1) {
        fprintf(stderr, "unexpected reliability stats\n");
        spkv2_free(ctx);
        return 1;
    }
    if (fabsf(output[0] - 0.0f) > 1e-6f || fabsf(output[1] - 2.0f) > 1e-6f ||
        fabsf(output[2] - 3.0f) > 1e-6f || fabsf(output[3] - 4.0f) > 1e-6f) {
        fprintf(stderr, "fault was not recovered\n");
        spkv2_free(ctx);
        return 1;
    }
    spkv2_free(ctx);
    return 0;
}
