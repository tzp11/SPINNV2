#include "spkv2_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned char *read_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    if (size < 0) { fclose(fp); return NULL; }
    unsigned char *data = (unsigned char *)malloc((size_t)size);
    if (!data) { fclose(fp); return NULL; }
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *out_size = (size_t)size;
    return data;
}

static int write_file(const char *path, const unsigned char *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    int ok = fwrite(data, 1, size, fp) == size;
    fclose(fp);
    return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s model.spk input.bin output.bin node tensor element bit invocation\n", argv[0]);
        return 2;
    }
    Spkv2Context *ctx = NULL;
    size_t input_size = 0;
    unsigned char *input = read_file(argv[2], &input_size);
    if (!input || spkv2_load_file(argv[1], &ctx) != 0 || spkv2_prepare(ctx, NULL, 0) != 0) {
        fprintf(stderr, "failed to prepare model or input\n");
        free(input);
        return 1;
    }
    Spkv2FaultEvent event = {0};
    event.node_id = (uint32_t)strtoul(argv[4], NULL, 10);
    event.tensor_id = (uint32_t)strtoul(argv[5], NULL, 10);
    event.element_index = (uint64_t)strtoull(argv[6], NULL, 10);
    event.bit_index = (uint8_t)strtoul(argv[7], NULL, 10);
    event.invocation_index = (uint32_t)strtoul(argv[8], NULL, 10);
    event.enabled = 1;
    if (spkv2_set_input(ctx, 0, input, input_size) != 0 ||
        spkv2_set_fault_event(ctx, &event) != 0 ||
        spkv2_run(ctx) != 0) {
        fprintf(stderr, "fault-injected execution failed\n");
        free(input);
        spkv2_free(ctx);
        return 1;
    }
    size_t output_size = 0;
    if (spkv2_get_output_size(ctx, 0, &output_size) != 0) {
        free(input);
        spkv2_free(ctx);
        return 1;
    }
    unsigned char *output = (unsigned char *)malloc(output_size);
    if (!output || spkv2_get_output(ctx, 0, output, output_size) != 0 ||
        write_file(argv[3], output, output_size) != 0) {
        free(output);
        free(input);
        spkv2_free(ctx);
        return 1;
    }
    Spkv2ReliabilityStats stats = {0};
    spkv2_get_reliability_stats(ctx, &stats);
    printf("{\"injected_faults\":%llu,\"detected_faults\":%llu,\"recovered_faults\":%llu,"
           "\"unrecovered_faults\":%llu,\"rerun_count\":%llu}\n",
           (unsigned long long)stats.injected_faults,
           (unsigned long long)stats.detected_faults,
           (unsigned long long)stats.recovered_faults,
           (unsigned long long)stats.unrecovered_faults,
           (unsigned long long)stats.rerun_count);
    free(output);
    free(input);
    spkv2_free(ctx);
    return 0;
}
