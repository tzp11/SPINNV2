/*
 * spkv2_bench — in-process micro-benchmark.
 *
 * Loads & prepares the SPK package once, sets the input once, then runs
 * the inference loop `runs` times (after `warmup` warm-up iterations).
 * Reports avg / min / p50 / p90 / max in milliseconds — identical
 * protocol to what we use for ORT in the Python wrapper.
 *
 *   Usage:
 *     spkv2_bench model.spk input.bin [output.bin] [--warmup N] [--runs N]
 */

#include "spkv2_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

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
        free(data); fclose(fp); return NULL;
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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s model.spk input.bin [output.bin] [--warmup N] [--runs N]\n",
            argv[0]);
        return 2;
    }
    const char *spk_path = argv[1];
    const char *in_path  = argv[2];
    const char *out_path = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    int warmup = 3, runs = 20;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
    }

    Spkv2Context *ctx = NULL;
    double t_load = now_ms();
    if (spkv2_load_file(spk_path, &ctx) != 0) {
        fprintf(stderr, "failed to load SPK\n");
        return 1;
    }
    t_load = now_ms() - t_load;

    double t_prep = now_ms();
    if (spkv2_prepare(ctx, NULL, 0) != 0) {
        fprintf(stderr, "failed to prepare runtime\n");
        spkv2_free(ctx); return 1;
    }
    t_prep = now_ms() - t_prep;

    size_t input_size = 0;
    unsigned char *input = read_file(in_path, &input_size);
    if (!input) { fprintf(stderr, "failed to read input\n"); spkv2_free(ctx); return 1; }
    if (spkv2_set_input(ctx, 0, input, input_size) != 0) {
        fprintf(stderr, "failed to set input\n");
        free(input); spkv2_free(ctx); return 1;
    }

    /* Warm-up (re-set input each run — memory planner may reuse input arena) */
    for (int i = 0; i < warmup; i++) {
        spkv2_set_input(ctx, 0, input, input_size);
        if (spkv2_run(ctx) != 0) {
            fprintf(stderr, "warmup run %d failed\n", i);
            free(input); spkv2_free(ctx); return 1;
        }
    }

    /* Timed runs */
    double *timings = (double *)malloc((size_t)runs * sizeof(double));
    if (!timings) { free(input); spkv2_free(ctx); return 1; }
    for (int i = 0; i < runs; i++) {
        spkv2_set_input(ctx, 0, input, input_size);
        double t0 = now_ms();
        if (spkv2_run(ctx) != 0) {
            fprintf(stderr, "timed run %d failed\n", i);
            free(timings); free(input); spkv2_free(ctx); return 1;
        }
        timings[i] = now_ms() - t0;
    }
    free(input);

    /* Optional output dump (from last run) – dump ALL outputs */
    if (out_path) {
        FILE *fp = fopen(out_path, "wb");
        if (fp) {
            for (int oi = 0; ; oi++) {
                size_t output_size = 0;
                if (spkv2_get_output_size(ctx, oi, &output_size) != 0) break;
                unsigned char *output = (unsigned char *)malloc(output_size);
                if (output && spkv2_get_output(ctx, oi, output, output_size) == 0)
                    fwrite(output, 1, output_size, fp);
                free(output);
            }
            fclose(fp);
        }
    }

    /* Stats */
    double sum = 0.0, mn = timings[0], mx = timings[0];
    for (int i = 0; i < runs; i++) {
        sum += timings[i];
        if (timings[i] < mn) mn = timings[i];
        if (timings[i] > mx) mx = timings[i];
    }
    qsort(timings, (size_t)runs, sizeof(double), cmp_double);
    double p50 = timings[runs / 2];
    double p90 = timings[(runs * 9) / 10];
    if (p90 >= runs) p90 = timings[runs - 1];

    printf("{\n");
    printf("  \"load_ms\": %.4f,\n", t_load);
    printf("  \"prepare_ms\": %.4f,\n", t_prep);
    printf("  \"warmup\": %d,\n", warmup);
    printf("  \"runs\": %d,\n", runs);
    printf("  \"min_ms\": %.4f,\n", mn);
    printf("  \"avg_ms\": %.4f,\n", sum / runs);
    printf("  \"p50_ms\": %.4f,\n", p50);
    printf("  \"p90_ms\": %.4f,\n", p90);
    printf("  \"max_ms\": %.4f\n", mx);
    printf("}\n");

    spkv2_profile_dump();
    free(timings);
    spkv2_free(ctx);
    return 0;
}
