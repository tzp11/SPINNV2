#include "spkv2_platform.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

/* ── Thread count ── */

static int g_num_threads = 0;

/* ── GEMM tiling ── */

static int g_gemm_kc = 256;
static int g_gemm_nc = 256;

/* ── Parallel threshold ── */

static int g_parallel_threshold_ns = 50000; /* 50µs */

/* ── Core topology ── */

static Spkv2CoreTopology g_topology = {0};

/* ── Initialization ── */

static void _init_platform(void) __attribute__((constructor));
static void _init_platform(void) {
    const char *env;

    env = getenv("SPKV2_NUM_THREADS");
    if (env) {
        int n = atoi(env);
        if (n >= 0) g_num_threads = n;
    }

    env = getenv("SPKV2_GEMM_KC");
    if (env) {
        int v = atoi(env);
        if (v >= 64) g_gemm_kc = v;
    }
    env = getenv("SPKV2_GEMM_NC");
    if (env) {
        int v = atoi(env);
        if (v >= 64) g_gemm_nc = v;
    }

    /* Auto-tune from cache size env vars if set */
    env = getenv("SPKV2_L1_CACHE");
    const char *env2 = getenv("SPKV2_L2_CACHE");
    if (env && env2) {
        spkv2_auto_tune_gemm(atoi(env), atoi(env2));
    }

    env = getenv("SPKV2_PARALLEL_THRESHOLD");
    if (env) {
        int v = atoi(env);
        if (v >= 0) g_parallel_threshold_ns = v;
    }
}

/* ── Thread count API ── */

void spkv2_set_num_threads(int n) {
    if (n < 0) n = 0;
    g_num_threads = n;
}

int spkv2_get_num_threads(void) { return g_num_threads; }

/* ── GEMM tiling API ── */

void spkv2_set_gemm_tiling(int kc, int nc) {
    if (kc >= 64) g_gemm_kc = kc;
    if (nc >= 64) g_gemm_nc = nc;
}

int spkv2_get_gemm_kc(void) { return g_gemm_kc; }
int spkv2_get_gemm_nc(void) { return g_gemm_nc; }

void spkv2_auto_tune_gemm(int l1_bytes, int l2_bytes) {
    if (l1_bytes <= 0 || l2_bytes <= 0) return;

    /* A panel of MR×KC should fit in ~1/3 of L1.
     * MR is arch-dependent (12 for NEON, 6 for AVX2).
     * Use 12 as the conservative (larger) value. */
    int mr = 12;
    int kc = l1_bytes / (3 * mr * (int)sizeof(float));
    kc = (kc / 64) * 64;
    if (kc < 64) kc = 64;
    if (kc > 512) kc = 512;

    /* B panel of KC×NC should fit in ~1/2 of L2. */
    int nc = l2_bytes / (2 * kc * (int)sizeof(float));
    nc = (nc / 64) * 64;
    if (nc < 64) nc = 64;
    if (nc > 1024) nc = 1024;

    g_gemm_kc = kc;
    g_gemm_nc = nc;
}

/* ── Parallel threshold API ── */

void spkv2_set_parallel_cost_threshold(int threshold_ns) {
    if (threshold_ns >= 0) g_parallel_threshold_ns = threshold_ns;
}

int spkv2_get_parallel_cost_threshold(void) { return g_parallel_threshold_ns; }

int spkv2_should_parallelize(int count, int cost_per_unit_ns) {
    int nthreads = g_num_threads;
    if (nthreads == 1) return 0;
    if (count <= 1) return 0;
    long long total = (long long)count * cost_per_unit_ns;
    return total > g_parallel_threshold_ns;
}

/* ── Core topology ── */

#if defined(__linux__) && defined(__ARM_NEON) && !defined(__APPLE__)

static int _read_int_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int val = -1;
    if (fscanf(f, "%d", &val) != 1) val = -1;
    fclose(f);
    return val;
}

void spkv2_detect_topology(void) {
    if (g_topology.detected) return;
    g_topology.detected = 1;

    int freqs[32];
    int ncpus = 0;
    for (int i = 0; i < 32; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        int freq = _read_int_file(path);
        if (freq < 0) break;
        freqs[i] = freq;
        ncpus++;
    }
    if (ncpus < 2) {
        g_topology.num_big = ncpus;
        g_topology.num_little = 0;
        for (int i = 0; i < ncpus; i++)
            g_topology.big_core_ids[i] = i;
        return;
    }

    /* Find the max frequency to distinguish big from little */
    int max_freq = 0;
    for (int i = 0; i < ncpus; i++)
        if (freqs[i] > max_freq) max_freq = freqs[i];

    /* Cores with ≥80% of max freq are "big", rest are "little" */
    int threshold = max_freq * 4 / 5;
    g_topology.num_big = 0;
    g_topology.num_little = 0;
    for (int i = 0; i < ncpus; i++) {
        if (freqs[i] >= threshold) {
            if (g_topology.num_big < 16)
                g_topology.big_core_ids[g_topology.num_big++] = i;
        } else {
            if (g_topology.num_little < 16)
                g_topology.little_core_ids[g_topology.num_little++] = i;
        }
    }

    /* If all cores have the same frequency, treat all as big */
    if (g_topology.num_little == 0 || g_topology.num_big == 0) {
        g_topology.num_big = ncpus;
        g_topology.num_little = 0;
        for (int i = 0; i < ncpus && i < 16; i++)
            g_topology.big_core_ids[i] = i;
    }
}

#include <sched.h>

void spkv2_set_affinity(int core_type) {
    if (!g_topology.detected) spkv2_detect_topology();

    cpu_set_t mask;
    CPU_ZERO(&mask);

    const int *ids;
    int count;
    switch (core_type) {
    case SPKV2_CORES_BIG:
        ids = g_topology.big_core_ids;
        count = g_topology.num_big;
        break;
    case SPKV2_CORES_LITTLE:
        ids = g_topology.little_core_ids;
        count = g_topology.num_little;
        if (count == 0) {
            ids = g_topology.big_core_ids;
            count = g_topology.num_big;
        }
        break;
    default: /* SPKV2_CORES_ALL */
        for (int i = 0; i < g_topology.num_big; i++)
            CPU_SET(g_topology.big_core_ids[i], &mask);
        for (int i = 0; i < g_topology.num_little; i++)
            CPU_SET(g_topology.little_core_ids[i], &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
        return;
    }

    for (int i = 0; i < count; i++)
        CPU_SET(ids[i], &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

#else /* Apple / non-Linux */

void spkv2_detect_topology(void) {
    if (g_topology.detected) return;
    g_topology.detected = 1;
#if defined(__APPLE__)
    int ncpu = 1;
    size_t len = sizeof(ncpu);
    sysctlbyname("hw.ncpu", &ncpu, &len, NULL, 0);
    g_topology.num_big = ncpu;
#else
    g_topology.num_big = 1;
#endif
    g_topology.num_little = 0;
    for (int i = 0; i < g_topology.num_big && i < 16; i++)
        g_topology.big_core_ids[i] = i;
}

void spkv2_set_affinity(int core_type) {
    (void)core_type;
    /* No-op on Apple (QoS handles this) and unsupported platforms */
}

#endif

const Spkv2CoreTopology *spkv2_get_topology(void) {
    if (!g_topology.detected) spkv2_detect_topology();
    return &g_topology;
}

/* ── Memory allocation ── */

void *spkv2_platform_malloc(size_t size) { return malloc(size); }
void *spkv2_platform_calloc(size_t count, size_t size) { return calloc(count, size); }
void spkv2_platform_free(void *ptr) { free(ptr); }
