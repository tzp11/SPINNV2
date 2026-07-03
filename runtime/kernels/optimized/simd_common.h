#ifndef SPKV2_SIMD_COMMON_H
#define SPKV2_SIMD_COMMON_H

#include "context.h"
#include "spkv2_format.h"

#if defined(__AVX2__) || defined(__ARM_NEON)

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <stddef.h>
#include <stdint.h>

#define SIMD_MIN(a, b) ((a) < (b) ? (a) : (b))

/* Arch-neutral utility functions */
size_t simd_elem_count(const Spkv2TensorRecord *r);
int simd_get_attr(const Spkv2Context *ctx, const Spkv2NodeRecord *node, Spkv2AttrRecord *attr);
void fused_activation_pass(float *data, size_t count, int act_type);
float apply_activation_scalar_simd(float x, int act_type);
const Spkv2KernelSpecRecord *simd_node_spec(const Spkv2Context *ctx, const Spkv2NodeRecord *node);
size_t simd_broadcast_index(const Spkv2TensorRecord *in_rec, const Spkv2TensorRecord *out_rec, size_t out_index);
int same_shape(const Spkv2TensorRecord *a, const Spkv2TensorRecord *b);

/* AVX2-specific */
#if defined(__AVX2__)
__m256 apply_activation_avx2(__m256 x, int act_type);
__m256 fast_exp_avx2(__m256 x);
__m256 sigmoid_avx2(__m256 x);
float hmax_avx2(__m256 v);
float hsum_avx2(__m256 v);
#endif

/* NEON-specific */
#if defined(__ARM_NEON)
float32x4_t apply_activation_neon(float32x4_t x, int act_type);
float32x4_t fast_exp_neon(float32x4_t x);
float32x4_t sigmoid_neon(float32x4_t x);
static inline float hmax_neon(float32x4_t v) { return vmaxvq_f32(v); }
static inline float hsum_neon(float32x4_t v) { return vaddvq_f32(v); }

/* Apple GCD parallel-for with single-thread fallback. */
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include "spkv2_platform.h"
static inline void spkv2_apply(size_t count, void (^block)(size_t)) {
    if (count == 0) return;
    if (spkv2_get_num_threads() == 1 || count == 1) {
        for (size_t i = 0; i < count; i++) block(i);
    } else {
        dispatch_apply(count, DISPATCH_APPLY_AUTO, block);
    }
}

/* Cost-aware variant: only parallelize when total cost exceeds threshold. */
static inline void spkv2_apply_cost(size_t count, int cost_per_unit_ns,
                                    void (^block)(size_t)) {
    if (count == 0) return;
    if (!spkv2_should_parallelize((int)count, cost_per_unit_ns)) {
        for (size_t i = 0; i < count; i++) block(i);
    } else {
        dispatch_apply(count, DISPATCH_APPLY_AUTO, block);
    }
}
#else
/* Non-Apple: OpenMP-based parallel-for with cost threshold.
 * Blocks (^) are not available, so use a serial loop when
 * cost doesn't justify threading overhead. The actual OMP
 * parallelism is done inline at call sites with #pragma omp. */
#include "spkv2_platform.h"
#endif /* __APPLE__ */

#endif

#endif /* __AVX2__ || __ARM_NEON */
#endif /* SPKV2_SIMD_COMMON_H */
