#ifndef SPKV2_PLATFORM_H
#define SPKV2_PLATFORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *spkv2_platform_malloc(size_t size);
void *spkv2_platform_calloc(size_t count, size_t size);
void spkv2_platform_free(void *ptr);

/* Thread count control.
 * 0 = auto (all available cores via GCD).
 * 1 = single-threaded (no dispatch overhead).
 * Initialised from SPKV2_NUM_THREADS env var at startup. */
void spkv2_set_num_threads(int n);
int  spkv2_get_num_threads(void);

/* ── GEMM tiling parameters (non-Apple ARM) ──
 * MR/NR are compile-time (tied to micro-kernel register allocation).
 * KC/NC control the outer GEBP tiling loops and are runtime-configurable.
 * Defaults: KC=256, NC=256. */
void spkv2_set_gemm_tiling(int kc, int nc);
int  spkv2_get_gemm_kc(void);
int  spkv2_get_gemm_nc(void);

/* Auto-tune KC/NC based on cache sizes (bytes).
 * KC sized so A panel (MR×KC) fits in L1.
 * NC sized so B panel (KC×NC) fits in L2.
 * Also reads SPKV2_L1_CACHE / SPKV2_L2_CACHE env vars. */
void spkv2_auto_tune_gemm(int l1_bytes, int l2_bytes);

/* ── Cost-based parallel threshold ──
 * Only parallelize when total_cost = count * cost_per_unit > threshold.
 * cost_per_unit is in nanoseconds. threshold defaults to 50000 (50µs).
 * On Apple, GCD has its own scheduler so this primarily affects
 * non-Apple platforms using OpenMP/pthreads. */
void spkv2_set_parallel_cost_threshold(int threshold_ns);
int  spkv2_get_parallel_cost_threshold(void);
int  spkv2_should_parallelize(int count, int cost_per_unit_ns);

/* ── big.LITTLE core topology ──
 * Detects heterogeneous core layout on Linux ARM (via sysfs).
 * On Apple/other platforms, all cores are treated as homogeneous. */
#define SPKV2_CORES_ALL    0
#define SPKV2_CORES_BIG    1
#define SPKV2_CORES_LITTLE 2

typedef struct {
    int num_big;
    int num_little;
    int big_core_ids[16];
    int little_core_ids[16];
    int detected;
} Spkv2CoreTopology;

void spkv2_detect_topology(void);
const Spkv2CoreTopology *spkv2_get_topology(void);
void spkv2_set_affinity(int core_type);

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_PLATFORM_H */
