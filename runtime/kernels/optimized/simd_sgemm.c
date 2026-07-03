#include "simd_sgemm.h"

#if defined(__AVX2__)

#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ================================================================== */
/*  GEBP SGEMM – C[M×N] += A[M×K] · B[K×N]                           */
/*                                                                     */
/*  Ported from SPINN v1 gemm_kernel.c with enhancements:              */
/*   - A (weights) pre-packed offline via node_cache                   */
/*   - B packed into NR-wide panels per KC tile                        */
/*   - 6×16 micro-kernel with 4× K-unroll and prefetch                */
/*   - N-tiling (TILE_NC) for better L2 utilisation                    */
/*   - 2D M×N parallel scheduling via OMP                              */
/* ================================================================== */

#define SGEMM_MR 6
#define SGEMM_NR 16
#define SGEMM_KC 256          /* K-tile */
#define SGEMM_NC 128          /* N-tile */
#define OMP_MIN_M 24          /* skip OMP for tiny M */
#define PF_AHEAD 8            /* prefetch lookahead steps */



static void pack_B_panel(const float *B, int ldb, int kc, int nr, float *Bp)
{
    for (int k = 0; k < kc; k++) {
        const float *src = B + (size_t)k * ldb;
        int j = 0;
        for (; j + 7 < nr; j += 8)
            _mm256_storeu_ps(Bp + j, _mm256_loadu_ps(src + j));
        for (; j < nr; j++)
            Bp[j] = src[j];
        for (; j < SGEMM_NR; j++)
            Bp[j] = 0.0f;
        Bp += SGEMM_NR;
    }
}



float *sgemm_pack_a_impl(int M, int K, const float *A, int lda)
{
    int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
    size_t total = (size_t)num_m_blocks * K * SGEMM_MR;
    float *pa;
    if (posix_memalign((void **)&pa, 32, total * sizeof(float)) != 0)
        return NULL;

    float *dst = pa;
    for (int m0 = 0; m0 < M; m0 += SGEMM_MR) {
        int cm = SIMD_MIN(SGEMM_MR, M - m0);
        for (int k = 0; k < K; k++) {
            int m = 0;
            for (; m < cm; m++)
                dst[m] = A[(m0 + m) * (size_t)lda + k];
            for (; m < SGEMM_MR; m++)
                dst[m] = 0.0f;
            dst += SGEMM_MR;
        }
    }
    return pa;
}



#define KERNEL_STEP_PA(KK) \
    bL = _mm256_loadu_ps(pb + (KK) * SGEMM_NR); \
    bR = _mm256_loadu_ps(pb + (KK) * SGEMM_NR + 8); \
    va = _mm256_set1_ps(pa[(KK)*SGEMM_MR + 0]); c0L = _mm256_fmadd_ps(va, bL, c0L); c0R = _mm256_fmadd_ps(va, bR, c0R); \
    va = _mm256_set1_ps(pa[(KK)*SGEMM_MR + 1]); c1L = _mm256_fmadd_ps(va, bL, c1L); c1R = _mm256_fmadd_ps(va, bR, c1R); \
    va = _mm256_set1_ps(pa[(KK)*SGEMM_MR + 2]); c2L = _mm256_fmadd_ps(va, bL, c2L); c2R = _mm256_fmadd_ps(va, bR, c2R); \
    va = _mm256_set1_ps(pa[(KK)*SGEMM_MR + 3]); c3L = _mm256_fmadd_ps(va, bL, c3L); c3R = _mm256_fmadd_ps(va, bR, c3R); \
    va = _mm256_set1_ps(pa[(KK)*SGEMM_MR + 4]); c4L = _mm256_fmadd_ps(va, bL, c4L); c4R = _mm256_fmadd_ps(va, bR, c4R); \
    va = _mm256_set1_ps(pa[(KK)*SGEMM_MR + 5]); c5L = _mm256_fmadd_ps(va, bL, c5L); c5R = _mm256_fmadd_ps(va, bR, c5R);

#define PREFETCH_PA(KK) \
    _mm_prefetch((const char*)(pa + ((KK) + PF_AHEAD) * SGEMM_MR), _MM_HINT_T0); \
    _mm_prefetch((const char*)(pb + ((KK) + PF_AHEAD) * SGEMM_NR), _MM_HINT_T0);



static void micro_6x16_packed_a(const float * __restrict__ pa,
                                 const float * __restrict__ pb,
                                 float * __restrict__ C, int ldc,
                                 int ck, int actual_m, int actual_n,
                                 int zero_mode)
{
    __m256 c0L, c0R, c1L, c1R, c2L, c2R, c3L, c3R, c4L, c4R, c5L, c5R;
    __m256 bL, bR, va;

    int is_edge = (actual_m < SGEMM_MR || actual_n < SGEMM_NR);

    if (is_edge) {
        c0L=_mm256_setzero_ps(); c0R=_mm256_setzero_ps();
        c1L=_mm256_setzero_ps(); c1R=_mm256_setzero_ps();
        c2L=_mm256_setzero_ps(); c2R=_mm256_setzero_ps();
        c3L=_mm256_setzero_ps(); c3R=_mm256_setzero_ps();
        c4L=_mm256_setzero_ps(); c4R=_mm256_setzero_ps();
        c5L=_mm256_setzero_ps(); c5R=_mm256_setzero_ps();
    } else if (zero_mode) {
        c0L=_mm256_setzero_ps(); c0R=_mm256_setzero_ps();
        c1L=_mm256_setzero_ps(); c1R=_mm256_setzero_ps();
        c2L=_mm256_setzero_ps(); c2R=_mm256_setzero_ps();
        c3L=_mm256_setzero_ps(); c3R=_mm256_setzero_ps();
        c4L=_mm256_setzero_ps(); c4R=_mm256_setzero_ps();
        c5L=_mm256_setzero_ps(); c5R=_mm256_setzero_ps();
    } else {
        c0L=_mm256_loadu_ps(C);            c0R=_mm256_loadu_ps(C+8);
        c1L=_mm256_loadu_ps(C+ldc);        c1R=_mm256_loadu_ps(C+ldc+8);
        c2L=_mm256_loadu_ps(C+2*ldc);      c2R=_mm256_loadu_ps(C+2*ldc+8);
        c3L=_mm256_loadu_ps(C+3*ldc);      c3R=_mm256_loadu_ps(C+3*ldc+8);
        c4L=_mm256_loadu_ps(C+4*ldc);      c4R=_mm256_loadu_ps(C+4*ldc+8);
        c5L=_mm256_loadu_ps(C+5*ldc);      c5R=_mm256_loadu_ps(C+5*ldc+8);
    }

    int k = 0;
    for (; k + 3 < ck; k += 4) {
        PREFETCH_PA(0);
        KERNEL_STEP_PA(0);
        KERNEL_STEP_PA(1);
        PREFETCH_PA(2);
        KERNEL_STEP_PA(2);
        KERNEL_STEP_PA(3);
        pa += 4 * SGEMM_MR;
        pb += 4 * SGEMM_NR;
    }
    for (; k < ck; k++) {
        KERNEL_STEP_PA(0);
        pa += SGEMM_MR;
        pb += SGEMM_NR;
    }

    if (is_edge) {
        float out_block[SGEMM_MR * SGEMM_NR] __attribute__((aligned(32)));
        _mm256_storeu_ps(out_block,      c0L); _mm256_storeu_ps(out_block+8,      c0R);
        _mm256_storeu_ps(out_block+16,   c1L); _mm256_storeu_ps(out_block+16+8,   c1R);
        _mm256_storeu_ps(out_block+32,   c2L); _mm256_storeu_ps(out_block+32+8,   c2R);
        _mm256_storeu_ps(out_block+48,   c3L); _mm256_storeu_ps(out_block+48+8,   c3R);
        _mm256_storeu_ps(out_block+64,   c4L); _mm256_storeu_ps(out_block+64+8,   c4R);
        _mm256_storeu_ps(out_block+80,   c5L); _mm256_storeu_ps(out_block+80+8,   c5R);
        for (int m = 0; m < actual_m; m++)
            for (int n = 0; n < actual_n; n++) {
                if (zero_mode) C[m * ldc + n]  = out_block[m * 16 + n];
                else           C[m * ldc + n] += out_block[m * 16 + n];
            }
    } else {
        _mm256_storeu_ps(C,         c0L); _mm256_storeu_ps(C+8,         c0R);
        _mm256_storeu_ps(C+ldc,     c1L); _mm256_storeu_ps(C+ldc+8,     c1R);
        _mm256_storeu_ps(C+2*ldc,   c2L); _mm256_storeu_ps(C+2*ldc+8,   c2R);
        _mm256_storeu_ps(C+3*ldc,   c3L); _mm256_storeu_ps(C+3*ldc+8,   c3R);
        _mm256_storeu_ps(C+4*ldc,   c4L); _mm256_storeu_ps(C+4*ldc+8,   c4R);
        _mm256_storeu_ps(C+5*ldc,   c5L); _mm256_storeu_ps(C+5*ldc+8,   c5R);
    }
}


#undef KERNEL_STEP_PA
#undef PREFETCH_PA



#define KERNEL_STEP_UNPACK(KK) \
    bL = _mm256_loadu_ps(pb + (KK) * SGEMM_NR); \
    bR = _mm256_loadu_ps(pb + (KK) * SGEMM_NR + 8); \
    va = _mm256_broadcast_ss(&a0[KK]); c0L = _mm256_fmadd_ps(va, bL, c0L); c0R = _mm256_fmadd_ps(va, bR, c0R); \
    va = _mm256_broadcast_ss(&a1[KK]); c1L = _mm256_fmadd_ps(va, bL, c1L); c1R = _mm256_fmadd_ps(va, bR, c1R); \
    va = _mm256_broadcast_ss(&a2[KK]); c2L = _mm256_fmadd_ps(va, bL, c2L); c2R = _mm256_fmadd_ps(va, bR, c2R); \
    va = _mm256_broadcast_ss(&a3[KK]); c3L = _mm256_fmadd_ps(va, bL, c3L); c3R = _mm256_fmadd_ps(va, bR, c3R); \
    va = _mm256_broadcast_ss(&a4[KK]); c4L = _mm256_fmadd_ps(va, bL, c4L); c4R = _mm256_fmadd_ps(va, bR, c4R); \
    va = _mm256_broadcast_ss(&a5[KK]); c5L = _mm256_fmadd_ps(va, bL, c5L); c5R = _mm256_fmadd_ps(va, bR, c5R);



static void micro_6x16_unpacked(const float *A, int lda,
                                  const float *pb,
                                  float *C, int ldc,
                                  int ck, int actual_m, int actual_n,
                                  int zero_mode)
{
    __m256 c0L, c0R, c1L, c1R, c2L, c2R, c3L, c3R, c4L, c4R, c5L, c5R;
    __m256 bL, bR, va;

    int is_edge = (actual_m < SGEMM_MR || actual_n < SGEMM_NR);

    if (is_edge || zero_mode) {
        c0L=_mm256_setzero_ps(); c0R=_mm256_setzero_ps();
        c1L=_mm256_setzero_ps(); c1R=_mm256_setzero_ps();
        c2L=_mm256_setzero_ps(); c2R=_mm256_setzero_ps();
        c3L=_mm256_setzero_ps(); c3R=_mm256_setzero_ps();
        c4L=_mm256_setzero_ps(); c4R=_mm256_setzero_ps();
        c5L=_mm256_setzero_ps(); c5R=_mm256_setzero_ps();
    } else {
        c0L=_mm256_loadu_ps(C);       c0R=_mm256_loadu_ps(C+8);
        c1L=_mm256_loadu_ps(C+ldc);   c1R=_mm256_loadu_ps(C+ldc+8);
        c2L=_mm256_loadu_ps(C+2*ldc); c2R=_mm256_loadu_ps(C+2*ldc+8);
        c3L=_mm256_loadu_ps(C+3*ldc); c3R=_mm256_loadu_ps(C+3*ldc+8);
        c4L=_mm256_loadu_ps(C+4*ldc); c4R=_mm256_loadu_ps(C+4*ldc+8);
        c5L=_mm256_loadu_ps(C+5*ldc); c5R=_mm256_loadu_ps(C+5*ldc+8);
    }

    const float *a0 = A, *a1 = A + lda, *a2 = A + 2*(size_t)lda;
    const float *a3 = A + 3*(size_t)lda, *a4 = A + 4*(size_t)lda, *a5 = A + 5*(size_t)lda;

    int k = 0;
    for (; k + 3 < ck; k += 4) {
        KERNEL_STEP_UNPACK(k+0); KERNEL_STEP_UNPACK(k+1);
        KERNEL_STEP_UNPACK(k+2); KERNEL_STEP_UNPACK(k+3);
    }
    for (; k < ck; k++) {
        KERNEL_STEP_UNPACK(k);
    }

    if (is_edge) {
        float out_block[SGEMM_MR * SGEMM_NR] __attribute__((aligned(32)));
        _mm256_storeu_ps(out_block,    c0L); _mm256_storeu_ps(out_block+8,    c0R);
        _mm256_storeu_ps(out_block+16, c1L); _mm256_storeu_ps(out_block+16+8, c1R);
        _mm256_storeu_ps(out_block+32, c2L); _mm256_storeu_ps(out_block+32+8, c2R);
        _mm256_storeu_ps(out_block+48, c3L); _mm256_storeu_ps(out_block+48+8, c3R);
        _mm256_storeu_ps(out_block+64, c4L); _mm256_storeu_ps(out_block+64+8, c4R);
        _mm256_storeu_ps(out_block+80, c5L); _mm256_storeu_ps(out_block+80+8, c5R);
        for (int m = 0; m < actual_m; m++)
            for (int n = 0; n < actual_n; n++) {
                if (zero_mode) C[m * ldc + n]  = out_block[m * 16 + n];
                else           C[m * ldc + n] += out_block[m * 16 + n];
            }
    } else {
        _mm256_storeu_ps(C,       c0L); _mm256_storeu_ps(C+8,       c0R);
        _mm256_storeu_ps(C+ldc,   c1L); _mm256_storeu_ps(C+ldc+8,   c1R);
        _mm256_storeu_ps(C+2*ldc, c2L); _mm256_storeu_ps(C+2*ldc+8, c2R);
        _mm256_storeu_ps(C+3*ldc, c3L); _mm256_storeu_ps(C+3*ldc+8, c3R);
        _mm256_storeu_ps(C+4*ldc, c4L); _mm256_storeu_ps(C+4*ldc+8, c4R);
        _mm256_storeu_ps(C+5*ldc, c5L); _mm256_storeu_ps(C+5*ldc+8, c5R);
    }
}


#undef KERNEL_STEP_UNPACK

/* Thread-local B-pack buffer (grows on demand, never freed) */


static _Thread_local float *s_bpack_buf = NULL;
static _Thread_local size_t s_bpack_floats = 0;



static float *get_bpack_buf(size_t need)
{
    if (s_bpack_floats < need) {
        free(s_bpack_buf);
        if (posix_memalign((void **)&s_bpack_buf, 32, need * sizeof(float)) != 0) {
            s_bpack_buf = NULL; s_bpack_floats = 0; return NULL;
        }
        s_bpack_floats = need;
    }
    return s_bpack_buf;
}


void sgemm_nn_packed_a_impl_run(int M, int N, int K,
                                const float *packed_a,
                                const float *B, int ldb,
                                float *C, int ldc,
                                int allow_parallel)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    int use_par = 0;
#ifdef _OPENMP
    use_par = allow_parallel && (M >= OMP_MIN_M);
#endif

    float *B_shared = get_bpack_buf((size_t)SGEMM_KC * SGEMM_NC);
    if (!B_shared) return;

    for (int n0 = 0; n0 < N; n0 += SGEMM_NC) {
        int cn = SIMD_MIN(SGEMM_NC, N - n0);
        int num_n_blocks = (cn + SGEMM_NR - 1) / SGEMM_NR;

        for (int k0 = 0; k0 < K; k0 += SGEMM_KC) {
            int ck = SIMD_MIN(SGEMM_KC, K - k0);
            int zm = 0;

            /* Pack B columns for this NC×KC tile */
            for (int nj = 0; nj < cn; nj += SGEMM_NR) {
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);
                pack_B_panel(B + (size_t)k0 * ldb + n0 + nj, ldb, ck, cnr,
                             B_shared + (size_t)nj * ck);
            }

            int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
            int total_tasks = num_m_blocks * num_n_blocks;

            #pragma omp parallel for schedule(static) if(use_par)
            for (int task = 0; task < total_tasks; task++) {
                int mi = task / num_n_blocks;
                int ni = task % num_n_blocks;
                int m0 = mi * SGEMM_MR;
                int nj = ni * SGEMM_NR;
                int cm = SIMD_MIN(SGEMM_MR, M - m0);
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);

                const float *pa = packed_a + (size_t)mi * K * SGEMM_MR + (size_t)k0 * SGEMM_MR;
                const float *pb = B_shared + (size_t)nj * ck;

                micro_6x16_packed_a(pa, pb, C + (size_t)m0 * ldc + n0 + nj, ldc,
                                     ck, cm, cnr, zm);
            }
        }
    }
}



void sgemm_nn_packed_a(int M, int N, int K,
                                const float *packed_a,
                                const float *B, int ldb,
                                float *C, int ldc)
{
    sgemm_nn_packed_a_impl_run(M, N, K, packed_a, B, ldb, C, ldc, 1);
}



void sgemm_nn(int M, int N, int K,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float *C, int ldc)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    int use_par = 0;
#ifdef _OPENMP
    use_par = (M >= OMP_MIN_M);
#endif

    float *B_shared = get_bpack_buf((size_t)SGEMM_KC * SGEMM_NC);
    if (!B_shared) {
        /* scalar fallback */
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float s = 0;
                for (int k = 0; k < K; k++) s += A[(size_t)m*lda+k] * B[(size_t)k*ldb+n];
                C[(size_t)m*ldc+n] += s;
            }
        return;
    }

    for (int n0 = 0; n0 < N; n0 += SGEMM_NC) {
        int cn = SIMD_MIN(SGEMM_NC, N - n0);
        int num_n_blocks = (cn + SGEMM_NR - 1) / SGEMM_NR;

        for (int k0 = 0; k0 < K; k0 += SGEMM_KC) {
            int ck = SIMD_MIN(SGEMM_KC, K - k0);
            int zm = 0;

            for (int nj = 0; nj < cn; nj += SGEMM_NR) {
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);
                pack_B_panel(B + (size_t)k0 * ldb + n0 + nj, ldb, ck, cnr,
                             B_shared + (size_t)nj * ck);
            }

            int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
            int total_tasks = num_m_blocks * num_n_blocks;

            #pragma omp parallel for schedule(static) if(use_par)
            for (int task = 0; task < total_tasks; task++) {
                int mi = task / num_n_blocks;
                int ni = task % num_n_blocks;
                int m0 = mi * SGEMM_MR;
                int nj = ni * SGEMM_NR;
                int cm = SIMD_MIN(SGEMM_MR, M - m0);
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);

                const float *pb = B_shared + (size_t)nj * ck;

                micro_6x16_unpacked(A + (size_t)m0 * lda + k0, lda, pb,
                                     C + (size_t)m0 * ldc + n0 + nj, ldc,
                                     ck, cm, cnr, zm);
            }
        }
    }
}


/* ================================================================== */
/*  NEON GEBP SGEMM – 12×8 micro-kernel for AArch64                   */
/* ================================================================== */

#elif defined(__ARM_NEON)

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif
#include "spkv2_platform.h"

#define SGEMM_MR 12
#define SGEMM_NR 8
#define SGEMM_KC 256
#define SGEMM_NC 256
#define OMP_MIN_M 24
#define PF_AHEAD 8

/* Runtime-configurable KC/NC for non-Apple ARM GEBP loops.
 * Falls back to compile-time defaults if not set. */
#define GEBP_KC (spkv2_get_gemm_kc())
#define GEBP_NC (spkv2_get_gemm_nc())



static void pack_B_panel(const float *B, int ldb, int kc, int nr, float *Bp)
{
    for (int k = 0; k < kc; k++) {
        const float *src = B + (size_t)k * ldb;
        if (nr == SGEMM_NR) {
            vst1q_f32(Bp, vld1q_f32(src));
            vst1q_f32(Bp + 4, vld1q_f32(src + 4));
        } else {
            int j = 0;
            for (; j + 3 < nr; j += 4)
                vst1q_f32(Bp + j, vld1q_f32(src + j));
            for (; j < nr; j++)
                Bp[j] = src[j];
            for (; j < SGEMM_NR; j++)
                Bp[j] = 0.0f;
        }
        Bp += SGEMM_NR;
    }
}



float *sgemm_pack_a_impl(int M, int K, const float *A, int lda)
{
    int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
    size_t total = (size_t)num_m_blocks * K * SGEMM_MR;
    float *pa;
    if (posix_memalign((void **)&pa, 32, total * sizeof(float)) != 0)
        return NULL;

    float *dst = pa;
    for (int m0 = 0; m0 < M; m0 += SGEMM_MR) {
        int cm = SIMD_MIN(SGEMM_MR, M - m0);
        for (int k = 0; k < K; k++) {
            int m = 0;
            for (; m < cm; m++)
                dst[m] = A[(m0 + m) * (size_t)lda + k];
            for (; m < SGEMM_MR; m++)
                dst[m] = 0.0f;
            dst += SGEMM_MR;
        }
    }
    return pa;
}


float *sgemm_pack_a_fp16(int M, int K, const void *A_fp16, int lda)
{
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    const _Float16 *A = (const _Float16 *)A_fp16;
    int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
    size_t total = (size_t)num_m_blocks * K * SGEMM_MR;
    float *pa;
    if (posix_memalign((void **)&pa, 32, total * sizeof(float)) != 0)
        return NULL;

    float *dst = pa;
    for (int m0 = 0; m0 < M; m0 += SGEMM_MR) {
        int cm = SIMD_MIN(SGEMM_MR, M - m0);
        for (int k = 0; k < K; k++) {
            int m = 0;
            for (; m < cm; m++)
                dst[m] = (float)A[(m0 + m) * (size_t)lda + k];
            for (; m < SGEMM_MR; m++)
                dst[m] = 0.0f;
            dst += SGEMM_MR;
        }
    }
    return pa;
#else
    (void)M; (void)K; (void)A_fp16; (void)lda;
    return NULL;
#endif
}



static void micro_12x8_packed_a(const float * __restrict__ pa,
                                const float * __restrict__ pb,
                                float * __restrict__ C, int ldc,
                                int ck, int actual_m, int actual_n,
                                int zero_mode)
{
    float32x4_t c0L  = vdupq_n_f32(0), c0R  = vdupq_n_f32(0);
    float32x4_t c1L  = vdupq_n_f32(0), c1R  = vdupq_n_f32(0);
    float32x4_t c2L  = vdupq_n_f32(0), c2R  = vdupq_n_f32(0);
    float32x4_t c3L  = vdupq_n_f32(0), c3R  = vdupq_n_f32(0);
    float32x4_t c4L  = vdupq_n_f32(0), c4R  = vdupq_n_f32(0);
    float32x4_t c5L  = vdupq_n_f32(0), c5R  = vdupq_n_f32(0);
    float32x4_t c6L  = vdupq_n_f32(0), c6R  = vdupq_n_f32(0);
    float32x4_t c7L  = vdupq_n_f32(0), c7R  = vdupq_n_f32(0);
    float32x4_t c8L  = vdupq_n_f32(0), c8R  = vdupq_n_f32(0);
    float32x4_t c9L  = vdupq_n_f32(0), c9R  = vdupq_n_f32(0);
    float32x4_t c10L = vdupq_n_f32(0), c10R = vdupq_n_f32(0);
    float32x4_t c11L = vdupq_n_f32(0), c11R = vdupq_n_f32(0);

    int is_edge = (actual_m < SGEMM_MR || actual_n < SGEMM_NR);

    if (!is_edge && !zero_mode) {
        c0L  = vld1q_f32(C);             c0R  = vld1q_f32(C + 4);
        c1L  = vld1q_f32(C + ldc);       c1R  = vld1q_f32(C + ldc + 4);
        c2L  = vld1q_f32(C + 2*ldc);     c2R  = vld1q_f32(C + 2*ldc + 4);
        c3L  = vld1q_f32(C + 3*ldc);     c3R  = vld1q_f32(C + 3*ldc + 4);
        c4L  = vld1q_f32(C + 4*ldc);     c4R  = vld1q_f32(C + 4*ldc + 4);
        c5L  = vld1q_f32(C + 5*ldc);     c5R  = vld1q_f32(C + 5*ldc + 4);
        c6L  = vld1q_f32(C + 6*ldc);     c6R  = vld1q_f32(C + 6*ldc + 4);
        c7L  = vld1q_f32(C + 7*ldc);     c7R  = vld1q_f32(C + 7*ldc + 4);
        c8L  = vld1q_f32(C + 8*ldc);     c8R  = vld1q_f32(C + 8*ldc + 4);
        c9L  = vld1q_f32(C + 9*ldc);     c9R  = vld1q_f32(C + 9*ldc + 4);
        c10L = vld1q_f32(C + 10*ldc);    c10R = vld1q_f32(C + 10*ldc + 4);
        c11L = vld1q_f32(C + 11*ldc);    c11R = vld1q_f32(C + 11*ldc + 4);
    }

    int k = 0;
    for (; k + 1 < ck; k += 2) {
        if (k + PF_AHEAD < ck) {
            __builtin_prefetch(pa + PF_AHEAD * SGEMM_MR, 0, 3);
            __builtin_prefetch(pb + PF_AHEAD * SGEMM_NR, 0, 3);
        }
        /* k+0 */
        {
            float32x4_t bL = vld1q_f32(pb);
            float32x4_t bR = vld1q_f32(pb + 4);
            float32x4_t aV0 = vld1q_f32(pa);
            float32x4_t aV1 = vld1q_f32(pa + 4);
            float32x4_t aV2 = vld1q_f32(pa + 8);
            c0L  = vfmaq_laneq_f32(c0L,  bL, aV0, 0); c0R  = vfmaq_laneq_f32(c0R,  bR, aV0, 0);
            c1L  = vfmaq_laneq_f32(c1L,  bL, aV0, 1); c1R  = vfmaq_laneq_f32(c1R,  bR, aV0, 1);
            c2L  = vfmaq_laneq_f32(c2L,  bL, aV0, 2); c2R  = vfmaq_laneq_f32(c2R,  bR, aV0, 2);
            c3L  = vfmaq_laneq_f32(c3L,  bL, aV0, 3); c3R  = vfmaq_laneq_f32(c3R,  bR, aV0, 3);
            c4L  = vfmaq_laneq_f32(c4L,  bL, aV1, 0); c4R  = vfmaq_laneq_f32(c4R,  bR, aV1, 0);
            c5L  = vfmaq_laneq_f32(c5L,  bL, aV1, 1); c5R  = vfmaq_laneq_f32(c5R,  bR, aV1, 1);
            c6L  = vfmaq_laneq_f32(c6L,  bL, aV1, 2); c6R  = vfmaq_laneq_f32(c6R,  bR, aV1, 2);
            c7L  = vfmaq_laneq_f32(c7L,  bL, aV1, 3); c7R  = vfmaq_laneq_f32(c7R,  bR, aV1, 3);
            c8L  = vfmaq_laneq_f32(c8L,  bL, aV2, 0); c8R  = vfmaq_laneq_f32(c8R,  bR, aV2, 0);
            c9L  = vfmaq_laneq_f32(c9L,  bL, aV2, 1); c9R  = vfmaq_laneq_f32(c9R,  bR, aV2, 1);
            c10L = vfmaq_laneq_f32(c10L, bL, aV2, 2); c10R = vfmaq_laneq_f32(c10R, bR, aV2, 2);
            c11L = vfmaq_laneq_f32(c11L, bL, aV2, 3); c11R = vfmaq_laneq_f32(c11R, bR, aV2, 3);
            pa += SGEMM_MR; pb += SGEMM_NR;
        }
        /* k+1 */
        {
            float32x4_t bL = vld1q_f32(pb);
            float32x4_t bR = vld1q_f32(pb + 4);
            float32x4_t aV0 = vld1q_f32(pa);
            float32x4_t aV1 = vld1q_f32(pa + 4);
            float32x4_t aV2 = vld1q_f32(pa + 8);
            c0L  = vfmaq_laneq_f32(c0L,  bL, aV0, 0); c0R  = vfmaq_laneq_f32(c0R,  bR, aV0, 0);
            c1L  = vfmaq_laneq_f32(c1L,  bL, aV0, 1); c1R  = vfmaq_laneq_f32(c1R,  bR, aV0, 1);
            c2L  = vfmaq_laneq_f32(c2L,  bL, aV0, 2); c2R  = vfmaq_laneq_f32(c2R,  bR, aV0, 2);
            c3L  = vfmaq_laneq_f32(c3L,  bL, aV0, 3); c3R  = vfmaq_laneq_f32(c3R,  bR, aV0, 3);
            c4L  = vfmaq_laneq_f32(c4L,  bL, aV1, 0); c4R  = vfmaq_laneq_f32(c4R,  bR, aV1, 0);
            c5L  = vfmaq_laneq_f32(c5L,  bL, aV1, 1); c5R  = vfmaq_laneq_f32(c5R,  bR, aV1, 1);
            c6L  = vfmaq_laneq_f32(c6L,  bL, aV1, 2); c6R  = vfmaq_laneq_f32(c6R,  bR, aV1, 2);
            c7L  = vfmaq_laneq_f32(c7L,  bL, aV1, 3); c7R  = vfmaq_laneq_f32(c7R,  bR, aV1, 3);
            c8L  = vfmaq_laneq_f32(c8L,  bL, aV2, 0); c8R  = vfmaq_laneq_f32(c8R,  bR, aV2, 0);
            c9L  = vfmaq_laneq_f32(c9L,  bL, aV2, 1); c9R  = vfmaq_laneq_f32(c9R,  bR, aV2, 1);
            c10L = vfmaq_laneq_f32(c10L, bL, aV2, 2); c10R = vfmaq_laneq_f32(c10R, bR, aV2, 2);
            c11L = vfmaq_laneq_f32(c11L, bL, aV2, 3); c11R = vfmaq_laneq_f32(c11R, bR, aV2, 3);
            pa += SGEMM_MR; pb += SGEMM_NR;
        }
    }
    for (; k < ck; k++) {
        float32x4_t bL = vld1q_f32(pb);
        float32x4_t bR = vld1q_f32(pb + 4);
        float32x4_t aV0 = vld1q_f32(pa);
        float32x4_t aV1 = vld1q_f32(pa + 4);
        float32x4_t aV2 = vld1q_f32(pa + 8);
        c0L  = vfmaq_laneq_f32(c0L,  bL, aV0, 0); c0R  = vfmaq_laneq_f32(c0R,  bR, aV0, 0);
        c1L  = vfmaq_laneq_f32(c1L,  bL, aV0, 1); c1R  = vfmaq_laneq_f32(c1R,  bR, aV0, 1);
        c2L  = vfmaq_laneq_f32(c2L,  bL, aV0, 2); c2R  = vfmaq_laneq_f32(c2R,  bR, aV0, 2);
        c3L  = vfmaq_laneq_f32(c3L,  bL, aV0, 3); c3R  = vfmaq_laneq_f32(c3R,  bR, aV0, 3);
        c4L  = vfmaq_laneq_f32(c4L,  bL, aV1, 0); c4R  = vfmaq_laneq_f32(c4R,  bR, aV1, 0);
        c5L  = vfmaq_laneq_f32(c5L,  bL, aV1, 1); c5R  = vfmaq_laneq_f32(c5R,  bR, aV1, 1);
        c6L  = vfmaq_laneq_f32(c6L,  bL, aV1, 2); c6R  = vfmaq_laneq_f32(c6R,  bR, aV1, 2);
        c7L  = vfmaq_laneq_f32(c7L,  bL, aV1, 3); c7R  = vfmaq_laneq_f32(c7R,  bR, aV1, 3);
        c8L  = vfmaq_laneq_f32(c8L,  bL, aV2, 0); c8R  = vfmaq_laneq_f32(c8R,  bR, aV2, 0);
        c9L  = vfmaq_laneq_f32(c9L,  bL, aV2, 1); c9R  = vfmaq_laneq_f32(c9R,  bR, aV2, 1);
        c10L = vfmaq_laneq_f32(c10L, bL, aV2, 2); c10R = vfmaq_laneq_f32(c10R, bR, aV2, 2);
        c11L = vfmaq_laneq_f32(c11L, bL, aV2, 3); c11R = vfmaq_laneq_f32(c11R, bR, aV2, 3);
        pa += SGEMM_MR; pb += SGEMM_NR;
    }

    if (is_edge) {
        float out_block[SGEMM_MR * SGEMM_NR] __attribute__((aligned(32)));
        float *t = out_block;
        vst1q_f32(t, c0L);  vst1q_f32(t + 4, c0R);  t += SGEMM_NR;
        vst1q_f32(t, c1L);  vst1q_f32(t + 4, c1R);  t += SGEMM_NR;
        vst1q_f32(t, c2L);  vst1q_f32(t + 4, c2R);  t += SGEMM_NR;
        vst1q_f32(t, c3L);  vst1q_f32(t + 4, c3R);  t += SGEMM_NR;
        vst1q_f32(t, c4L);  vst1q_f32(t + 4, c4R);  t += SGEMM_NR;
        vst1q_f32(t, c5L);  vst1q_f32(t + 4, c5R);  t += SGEMM_NR;
        vst1q_f32(t, c6L);  vst1q_f32(t + 4, c6R);  t += SGEMM_NR;
        vst1q_f32(t, c7L);  vst1q_f32(t + 4, c7R);  t += SGEMM_NR;
        vst1q_f32(t, c8L);  vst1q_f32(t + 4, c8R);  t += SGEMM_NR;
        vst1q_f32(t, c9L);  vst1q_f32(t + 4, c9R);  t += SGEMM_NR;
        vst1q_f32(t, c10L); vst1q_f32(t + 4, c10R); t += SGEMM_NR;
        vst1q_f32(t, c11L); vst1q_f32(t + 4, c11R);
        for (int m = 0; m < actual_m; m++)
            for (int n = 0; n < actual_n; n++) {
                if (zero_mode) C[m * ldc + n]  = out_block[m * SGEMM_NR + n];
                else           C[m * ldc + n] += out_block[m * SGEMM_NR + n];
            }
    } else {
        vst1q_f32(C, c0L);          vst1q_f32(C + 4, c0R);
        vst1q_f32(C+ldc, c1L);      vst1q_f32(C+ldc+4, c1R);
        vst1q_f32(C+2*ldc, c2L);    vst1q_f32(C+2*ldc+4, c2R);
        vst1q_f32(C+3*ldc, c3L);    vst1q_f32(C+3*ldc+4, c3R);
        vst1q_f32(C+4*ldc, c4L);    vst1q_f32(C+4*ldc+4, c4R);
        vst1q_f32(C+5*ldc, c5L);    vst1q_f32(C+5*ldc+4, c5R);
        vst1q_f32(C+6*ldc, c6L);    vst1q_f32(C+6*ldc+4, c6R);
        vst1q_f32(C+7*ldc, c7L);    vst1q_f32(C+7*ldc+4, c7R);
        vst1q_f32(C+8*ldc, c8L);    vst1q_f32(C+8*ldc+4, c8R);
        vst1q_f32(C+9*ldc, c9L);    vst1q_f32(C+9*ldc+4, c9R);
        vst1q_f32(C+10*ldc, c10L);  vst1q_f32(C+10*ldc+4, c10R);
        vst1q_f32(C+11*ldc, c11L);  vst1q_f32(C+11*ldc+4, c11R);
    }
}



static void micro_12x8_unpacked(const float *A, int lda,
                                 const float *pb,
                                 float *C, int ldc,
                                 int ck, int actual_m, int actual_n,
                                 int zero_mode)
{
    float32x4_t c0L  = vdupq_n_f32(0), c0R  = vdupq_n_f32(0);
    float32x4_t c1L  = vdupq_n_f32(0), c1R  = vdupq_n_f32(0);
    float32x4_t c2L  = vdupq_n_f32(0), c2R  = vdupq_n_f32(0);
    float32x4_t c3L  = vdupq_n_f32(0), c3R  = vdupq_n_f32(0);
    float32x4_t c4L  = vdupq_n_f32(0), c4R  = vdupq_n_f32(0);
    float32x4_t c5L  = vdupq_n_f32(0), c5R  = vdupq_n_f32(0);
    float32x4_t c6L  = vdupq_n_f32(0), c6R  = vdupq_n_f32(0);
    float32x4_t c7L  = vdupq_n_f32(0), c7R  = vdupq_n_f32(0);
    float32x4_t c8L  = vdupq_n_f32(0), c8R  = vdupq_n_f32(0);
    float32x4_t c9L  = vdupq_n_f32(0), c9R  = vdupq_n_f32(0);
    float32x4_t c10L = vdupq_n_f32(0), c10R = vdupq_n_f32(0);
    float32x4_t c11L = vdupq_n_f32(0), c11R = vdupq_n_f32(0);

    int is_edge = (actual_m < SGEMM_MR || actual_n < SGEMM_NR);

    if (!is_edge && !zero_mode) {
        c0L  = vld1q_f32(C);             c0R  = vld1q_f32(C + 4);
        c1L  = vld1q_f32(C + ldc);       c1R  = vld1q_f32(C + ldc + 4);
        c2L  = vld1q_f32(C + 2*ldc);     c2R  = vld1q_f32(C + 2*ldc + 4);
        c3L  = vld1q_f32(C + 3*ldc);     c3R  = vld1q_f32(C + 3*ldc + 4);
        c4L  = vld1q_f32(C + 4*ldc);     c4R  = vld1q_f32(C + 4*ldc + 4);
        c5L  = vld1q_f32(C + 5*ldc);     c5R  = vld1q_f32(C + 5*ldc + 4);
        c6L  = vld1q_f32(C + 6*ldc);     c6R  = vld1q_f32(C + 6*ldc + 4);
        c7L  = vld1q_f32(C + 7*ldc);     c7R  = vld1q_f32(C + 7*ldc + 4);
        c8L  = vld1q_f32(C + 8*ldc);     c8R  = vld1q_f32(C + 8*ldc + 4);
        c9L  = vld1q_f32(C + 9*ldc);     c9R  = vld1q_f32(C + 9*ldc + 4);
        c10L = vld1q_f32(C + 10*ldc);    c10R = vld1q_f32(C + 10*ldc + 4);
        c11L = vld1q_f32(C + 11*ldc);    c11R = vld1q_f32(C + 11*ldc + 4);
    }

    const float *a0  = A,                  *a1  = A + lda;
    const float *a2  = A + 2*(size_t)lda,  *a3  = A + 3*(size_t)lda;
    const float *a4  = A + 4*(size_t)lda,  *a5  = A + 5*(size_t)lda;
    const float *a6  = A + 6*(size_t)lda,  *a7  = A + 7*(size_t)lda;
    const float *a8  = A + 8*(size_t)lda,  *a9  = A + 9*(size_t)lda;
    const float *a10 = A + 10*(size_t)lda, *a11 = A + 11*(size_t)lda;

    int k = 0;
    for (; k + 1 < ck; k += 2) {
        /* k+0 */
        {
            float32x4_t bL = vld1q_f32(pb + k * SGEMM_NR);
            float32x4_t bR = vld1q_f32(pb + k * SGEMM_NR + 4);
            float32x4_t va;
            va = vld1q_dup_f32(&a0[k]);  c0L  = vfmaq_f32(c0L,  va, bL); c0R  = vfmaq_f32(c0R,  va, bR);
            va = vld1q_dup_f32(&a1[k]);  c1L  = vfmaq_f32(c1L,  va, bL); c1R  = vfmaq_f32(c1R,  va, bR);
            va = vld1q_dup_f32(&a2[k]);  c2L  = vfmaq_f32(c2L,  va, bL); c2R  = vfmaq_f32(c2R,  va, bR);
            va = vld1q_dup_f32(&a3[k]);  c3L  = vfmaq_f32(c3L,  va, bL); c3R  = vfmaq_f32(c3R,  va, bR);
            va = vld1q_dup_f32(&a4[k]);  c4L  = vfmaq_f32(c4L,  va, bL); c4R  = vfmaq_f32(c4R,  va, bR);
            va = vld1q_dup_f32(&a5[k]);  c5L  = vfmaq_f32(c5L,  va, bL); c5R  = vfmaq_f32(c5R,  va, bR);
            va = vld1q_dup_f32(&a6[k]);  c6L  = vfmaq_f32(c6L,  va, bL); c6R  = vfmaq_f32(c6R,  va, bR);
            va = vld1q_dup_f32(&a7[k]);  c7L  = vfmaq_f32(c7L,  va, bL); c7R  = vfmaq_f32(c7R,  va, bR);
            va = vld1q_dup_f32(&a8[k]);  c8L  = vfmaq_f32(c8L,  va, bL); c8R  = vfmaq_f32(c8R,  va, bR);
            va = vld1q_dup_f32(&a9[k]);  c9L  = vfmaq_f32(c9L,  va, bL); c9R  = vfmaq_f32(c9R,  va, bR);
            va = vld1q_dup_f32(&a10[k]); c10L = vfmaq_f32(c10L, va, bL); c10R = vfmaq_f32(c10R, va, bR);
            va = vld1q_dup_f32(&a11[k]); c11L = vfmaq_f32(c11L, va, bL); c11R = vfmaq_f32(c11R, va, bR);
        }
        /* k+1 */
        {
            float32x4_t bL = vld1q_f32(pb + (k+1) * SGEMM_NR);
            float32x4_t bR = vld1q_f32(pb + (k+1) * SGEMM_NR + 4);
            float32x4_t va;
            va = vld1q_dup_f32(&a0[k+1]);  c0L  = vfmaq_f32(c0L,  va, bL); c0R  = vfmaq_f32(c0R,  va, bR);
            va = vld1q_dup_f32(&a1[k+1]);  c1L  = vfmaq_f32(c1L,  va, bL); c1R  = vfmaq_f32(c1R,  va, bR);
            va = vld1q_dup_f32(&a2[k+1]);  c2L  = vfmaq_f32(c2L,  va, bL); c2R  = vfmaq_f32(c2R,  va, bR);
            va = vld1q_dup_f32(&a3[k+1]);  c3L  = vfmaq_f32(c3L,  va, bL); c3R  = vfmaq_f32(c3R,  va, bR);
            va = vld1q_dup_f32(&a4[k+1]);  c4L  = vfmaq_f32(c4L,  va, bL); c4R  = vfmaq_f32(c4R,  va, bR);
            va = vld1q_dup_f32(&a5[k+1]);  c5L  = vfmaq_f32(c5L,  va, bL); c5R  = vfmaq_f32(c5R,  va, bR);
            va = vld1q_dup_f32(&a6[k+1]);  c6L  = vfmaq_f32(c6L,  va, bL); c6R  = vfmaq_f32(c6R,  va, bR);
            va = vld1q_dup_f32(&a7[k+1]);  c7L  = vfmaq_f32(c7L,  va, bL); c7R  = vfmaq_f32(c7R,  va, bR);
            va = vld1q_dup_f32(&a8[k+1]);  c8L  = vfmaq_f32(c8L,  va, bL); c8R  = vfmaq_f32(c8R,  va, bR);
            va = vld1q_dup_f32(&a9[k+1]);  c9L  = vfmaq_f32(c9L,  va, bL); c9R  = vfmaq_f32(c9R,  va, bR);
            va = vld1q_dup_f32(&a10[k+1]); c10L = vfmaq_f32(c10L, va, bL); c10R = vfmaq_f32(c10R, va, bR);
            va = vld1q_dup_f32(&a11[k+1]); c11L = vfmaq_f32(c11L, va, bL); c11R = vfmaq_f32(c11R, va, bR);
        }
    }
    for (; k < ck; k++) {
        float32x4_t bL = vld1q_f32(pb + k * SGEMM_NR);
        float32x4_t bR = vld1q_f32(pb + k * SGEMM_NR + 4);
        float32x4_t va;
        va = vld1q_dup_f32(&a0[k]);  c0L  = vfmaq_f32(c0L,  va, bL); c0R  = vfmaq_f32(c0R,  va, bR);
        va = vld1q_dup_f32(&a1[k]);  c1L  = vfmaq_f32(c1L,  va, bL); c1R  = vfmaq_f32(c1R,  va, bR);
        va = vld1q_dup_f32(&a2[k]);  c2L  = vfmaq_f32(c2L,  va, bL); c2R  = vfmaq_f32(c2R,  va, bR);
        va = vld1q_dup_f32(&a3[k]);  c3L  = vfmaq_f32(c3L,  va, bL); c3R  = vfmaq_f32(c3R,  va, bR);
        va = vld1q_dup_f32(&a4[k]);  c4L  = vfmaq_f32(c4L,  va, bL); c4R  = vfmaq_f32(c4R,  va, bR);
        va = vld1q_dup_f32(&a5[k]);  c5L  = vfmaq_f32(c5L,  va, bL); c5R  = vfmaq_f32(c5R,  va, bR);
        va = vld1q_dup_f32(&a6[k]);  c6L  = vfmaq_f32(c6L,  va, bL); c6R  = vfmaq_f32(c6R,  va, bR);
        va = vld1q_dup_f32(&a7[k]);  c7L  = vfmaq_f32(c7L,  va, bL); c7R  = vfmaq_f32(c7R,  va, bR);
        va = vld1q_dup_f32(&a8[k]);  c8L  = vfmaq_f32(c8L,  va, bL); c8R  = vfmaq_f32(c8R,  va, bR);
        va = vld1q_dup_f32(&a9[k]);  c9L  = vfmaq_f32(c9L,  va, bL); c9R  = vfmaq_f32(c9R,  va, bR);
        va = vld1q_dup_f32(&a10[k]); c10L = vfmaq_f32(c10L, va, bL); c10R = vfmaq_f32(c10R, va, bR);
        va = vld1q_dup_f32(&a11[k]); c11L = vfmaq_f32(c11L, va, bL); c11R = vfmaq_f32(c11R, va, bR);
    }

    if (is_edge) {
        float out_block[SGEMM_MR * SGEMM_NR] __attribute__((aligned(32)));
        float *t = out_block;
        vst1q_f32(t, c0L);  vst1q_f32(t + 4, c0R);  t += SGEMM_NR;
        vst1q_f32(t, c1L);  vst1q_f32(t + 4, c1R);  t += SGEMM_NR;
        vst1q_f32(t, c2L);  vst1q_f32(t + 4, c2R);  t += SGEMM_NR;
        vst1q_f32(t, c3L);  vst1q_f32(t + 4, c3R);  t += SGEMM_NR;
        vst1q_f32(t, c4L);  vst1q_f32(t + 4, c4R);  t += SGEMM_NR;
        vst1q_f32(t, c5L);  vst1q_f32(t + 4, c5R);  t += SGEMM_NR;
        vst1q_f32(t, c6L);  vst1q_f32(t + 4, c6R);  t += SGEMM_NR;
        vst1q_f32(t, c7L);  vst1q_f32(t + 4, c7R);  t += SGEMM_NR;
        vst1q_f32(t, c8L);  vst1q_f32(t + 4, c8R);  t += SGEMM_NR;
        vst1q_f32(t, c9L);  vst1q_f32(t + 4, c9R);  t += SGEMM_NR;
        vst1q_f32(t, c10L); vst1q_f32(t + 4, c10R); t += SGEMM_NR;
        vst1q_f32(t, c11L); vst1q_f32(t + 4, c11R);
        for (int m = 0; m < actual_m; m++)
            for (int n = 0; n < actual_n; n++) {
                if (zero_mode) C[m * ldc + n]  = out_block[m * SGEMM_NR + n];
                else           C[m * ldc + n] += out_block[m * SGEMM_NR + n];
            }
    } else {
        vst1q_f32(C, c0L);          vst1q_f32(C + 4, c0R);
        vst1q_f32(C+ldc, c1L);      vst1q_f32(C+ldc+4, c1R);
        vst1q_f32(C+2*ldc, c2L);    vst1q_f32(C+2*ldc+4, c2R);
        vst1q_f32(C+3*ldc, c3L);    vst1q_f32(C+3*ldc+4, c3R);
        vst1q_f32(C+4*ldc, c4L);    vst1q_f32(C+4*ldc+4, c4R);
        vst1q_f32(C+5*ldc, c5L);    vst1q_f32(C+5*ldc+4, c5R);
        vst1q_f32(C+6*ldc, c6L);    vst1q_f32(C+6*ldc+4, c6R);
        vst1q_f32(C+7*ldc, c7L);    vst1q_f32(C+7*ldc+4, c7R);
        vst1q_f32(C+8*ldc, c8L);    vst1q_f32(C+8*ldc+4, c8R);
        vst1q_f32(C+9*ldc, c9L);    vst1q_f32(C+9*ldc+4, c9R);
        vst1q_f32(C+10*ldc, c10L);  vst1q_f32(C+10*ldc+4, c10R);
        vst1q_f32(C+11*ldc, c11L);  vst1q_f32(C+11*ldc+4, c11R);
    }
}



/* Thread-local B-pack buffer (grows on demand, never freed) */

static _Thread_local float *s_bpack_buf = NULL;
static _Thread_local size_t s_bpack_floats = 0;



static float *get_bpack_buf(size_t need)
{
    if (s_bpack_floats < need) {
        free(s_bpack_buf);
        if (posix_memalign((void **)&s_bpack_buf, 32, need * sizeof(float)) != 0) {
            s_bpack_buf = NULL; s_bpack_floats = 0; return NULL;
        }
        s_bpack_floats = need;
    }
    return s_bpack_buf;
}


void sgemm_nn_packed_a_impl_run(int M, int N, int K,
                                const float *packed_a,
                                const float *B, int ldb,
                                float *C, int ldc,
                                int allow_parallel)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    int tile_kc = GEBP_KC;
    int tile_nc = GEBP_NC;
    int use_par = 0;
#ifdef _OPENMP
    use_par = allow_parallel && spkv2_should_parallelize(
        (M + SGEMM_MR - 1) / SGEMM_MR, tile_kc * SGEMM_NR * 2);
#endif

    float *B_shared = get_bpack_buf((size_t)tile_kc * tile_nc);
    if (!B_shared) return;

    for (int n0 = 0; n0 < N; n0 += tile_nc) {
        int cn = SIMD_MIN(tile_nc, N - n0);
        int num_n_blocks = (cn + SGEMM_NR - 1) / SGEMM_NR;

        for (int k0 = 0; k0 < K; k0 += tile_kc) {
            int ck = SIMD_MIN(tile_kc, K - k0);
            int zm = 0;

            /* Pack B columns for this NC×KC tile */
            for (int nj = 0; nj < cn; nj += SGEMM_NR) {
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);
                pack_B_panel(B + (size_t)k0 * ldb + n0 + nj, ldb, ck, cnr,
                             B_shared + (size_t)nj * ck);
            }

            int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
            int total_tasks = num_m_blocks * num_n_blocks;

            #pragma omp parallel for schedule(static) if(use_par)
            for (int task = 0; task < total_tasks; task++) {
                int mi = task / num_n_blocks;
                int ni = task % num_n_blocks;
                int m0 = mi * SGEMM_MR;
                int nj = ni * SGEMM_NR;
                int cm = SIMD_MIN(SGEMM_MR, M - m0);
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);

                const float *pa = packed_a + (size_t)mi * K * SGEMM_MR + (size_t)k0 * SGEMM_MR;
                const float *pb = B_shared + (size_t)nj * ck;

                micro_12x8_packed_a(pa, pb, C + (size_t)m0 * ldc + n0 + nj, ldc,
                                    ck, cm, cnr, zm);
            }
        }
    }
}



void sgemm_nn_packed_a(int M, int N, int K,
                                const float *packed_a,
                                const float *B, int ldb,
                                float *C, int ldc)
{
    sgemm_nn_packed_a_impl_run(M, N, K, packed_a, B, ldb, C, ldc, 1);
}



void sgemm_nn(int M, int N, int K,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float *C, int ldc)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

#if defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, lda, B, ldb, 1.0f, C, ldc);
    return;
#endif
    /* Fallback NEON GEBP for non-Apple ARM */
    int tile_kc = GEBP_KC;
    int tile_nc = GEBP_NC;
    int use_par = 0;
#ifdef _OPENMP
    use_par = spkv2_should_parallelize(
        (M + SGEMM_MR - 1) / SGEMM_MR, tile_kc * SGEMM_NR * 2);
#endif

    float *B_shared = get_bpack_buf((size_t)tile_kc * tile_nc);
    if (!B_shared) {
        /* scalar fallback */
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float s = 0;
                for (int k = 0; k < K; k++) s += A[(size_t)m*lda+k] * B[(size_t)k*ldb+n];
                C[(size_t)m*ldc+n] += s;
            }
        return;
    }

    for (int n0 = 0; n0 < N; n0 += tile_nc) {
        int cn = SIMD_MIN(tile_nc, N - n0);
        int num_n_blocks = (cn + SGEMM_NR - 1) / SGEMM_NR;

        for (int k0 = 0; k0 < K; k0 += tile_kc) {
            int ck = SIMD_MIN(tile_kc, K - k0);
            int zm = 0;

            for (int nj = 0; nj < cn; nj += SGEMM_NR) {
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);
                pack_B_panel(B + (size_t)k0 * ldb + n0 + nj, ldb, ck, cnr,
                             B_shared + (size_t)nj * ck);
            }

            int num_m_blocks = (M + SGEMM_MR - 1) / SGEMM_MR;
            int total_tasks = num_m_blocks * num_n_blocks;

            #pragma omp parallel for schedule(static) if(use_par)
            for (int task = 0; task < total_tasks; task++) {
                int mi = task / num_n_blocks;
                int ni = task % num_n_blocks;
                int m0 = mi * SGEMM_MR;
                int nj = ni * SGEMM_NR;
                int cm = SIMD_MIN(SGEMM_MR, M - m0);
                int cnr = SIMD_MIN(SGEMM_NR, cn - nj);

                const float *pb = B_shared + (size_t)nj * ck;

                micro_12x8_unpacked(A + (size_t)m0 * lda + k0, lda, pb,
                                    C + (size_t)m0 * ldc + n0 + nj, ldc,
                                    ck, cm, cnr, zm);
            }
        }
    }
}


#endif /* __AVX2__ || __ARM_NEON */
