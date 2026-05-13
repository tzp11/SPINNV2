#include "simd_sgemm.h"

#ifdef __AVX2__

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


#endif /* __AVX2__ */
