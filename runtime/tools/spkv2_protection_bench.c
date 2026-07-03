/*
 * Measure the scalar protection primitives used by execute_protected_node().
 * The output is JSON so experiment drivers can build cost models directly.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

static volatile int s_sink;

static double now_ms(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static int output_in_range(const float *values, size_t count, float lower, float upper) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i]) || values[i] < lower || values[i] > upper) return 0;
    }
    return 1;
}

static int repetitions_for(size_t bytes) {
    const size_t target_bytes = 256u * 1024u * 1024u;
    int repetitions = (int)(target_bytes / (bytes > 0 ? bytes : 1u));
    if (repetitions < 12) repetitions = 12;
    if (repetitions > 20000) repetitions = 20000;
    return repetitions;
}

int main(void) {
    static const size_t sizes[] = {
        40u, 100352u, 200704u, 401408u, 802816u, 1605632u, 3211264u
    };
    const size_t size_count = sizeof(sizes) / sizeof(sizes[0]);
    printf("{\"measurements\":[\n");
    for (size_t index = 0; index < size_count; index++) {
        size_t bytes = sizes[index];
        size_t count = bytes / sizeof(float);
        int repetitions = repetitions_for(bytes);
        float *values = (float *)calloc(count, sizeof(float));
        float *first = (float *)malloc(bytes);
        float *second = (float *)malloc(bytes);
        if (!values || !first || !second) {
            free(values);
            free(first);
            free(second);
            return 1;
        }
        double t0 = now_ms();
        for (int repeat = 0; repeat < repetitions; repeat++) {
            s_sink += output_in_range(values, count, -1.0f, 1.0f);
        }
        double scan_ms = (now_ms() - t0) / repetitions;
        t0 = now_ms();
        for (int repeat = 0; repeat < repetitions; repeat++) {
            memcpy(first, values, bytes);
            memcpy(second, values, bytes);
            s_sink += memcmp(first, second, bytes) == 0;
        }
        double dmr_buffer_ms = (now_ms() - t0) / repetitions;
        printf(
            "  {\"tensor_bytes\":%llu,\"repetitions\":%d,\"range_scan_ms\":%.9f,\"dmr_buffer_ms\":%.9f}%s\n",
            (unsigned long long)bytes,
            repetitions,
            scan_ms,
            dmr_buffer_ms,
            index + 1 == size_count ? "" : ","
        );
        free(values);
        free(first);
        free(second);
    }
    printf("]}\n");
    return s_sink == -1;
}
