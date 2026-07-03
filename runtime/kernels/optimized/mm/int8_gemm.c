/*
 * INT8 GEMM: C_int32[M x N] += A_int8[M x K] * B_int8[K x N]
 *
 * MR=8, NR=8 micro-kernel using vdotq_laneq_s32 (NEON dot product).
 * Requires ARMv8.2-A+dotprod (Apple Silicon M1+ is ARMv8.4+).
 *
 * A is pre-packed in MR-wide panels.
 * B is packed on-the-fly per KC tile into NR-wide panels.
 * Output is int32 accumulator -- caller dequantizes.
 */

#include "int8_gemm.h"

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

/*
 * Pack A (weights) into MR-wide panels for INT8 GEMM.
 * K is padded to a multiple of 4 for vdotq_s32.
 *
 * Layout per K-step of 4:
 *   [row0_k0..k3, row1_k0..k3, ..., row(MR-1)_k0..k3] = MR*4 bytes
 *
 * Returns allocated buffer (caller owns and must free).
 */
int8_t *i8gemm_pack_a(int M, int K, const int8_t *A, int lda) {
    int K4 = (K + 3) & ~3;
    int panels = (M + I8GEMM_MR - 1) / I8GEMM_MR;
    size_t packed_size = (size_t)panels * I8GEMM_MR * K4;
    int8_t *pa = (int8_t *)calloc(packed_size, 1);
    if (!pa) return NULL;

    int8_t *dst = pa;
    for (int i = 0; i < M; i += I8GEMM_MR) {
        int mr = (M - i < I8GEMM_MR) ? (M - i) : I8GEMM_MR;
        for (int k = 0; k < K4; k += 4) {
            for (int r = 0; r < I8GEMM_MR; r++) {
                if (r < mr && k < K) {
                    int remain = K - k;
                    if (remain >= 4) {
                        memcpy(dst, &A[(i + r) * lda + k], 4);
                    } else {
                        memcpy(dst, &A[(i + r) * lda + k], remain);
                        memset(dst + remain, 0, 4 - remain);
                    }
                } else {
                    memset(dst, 0, 4);
                }
                dst += 4;
            }
        }
    }
    return pa;
}

/*
 * Pack a B panel of size kc x nr into NR-wide format.
 * Same layout as A: per K-step of 4, store NR rows of 4 bytes each.
 */
static void pack_b_panel(const int8_t *B, int ldb, int kc, int nr, int8_t *Bp) {
    int kc4 = (kc + 3) & ~3;
    for (int k = 0; k < kc4; k += 4) {
        for (int j = 0; j < I8GEMM_NR; j++) {
            if (j < nr && k < kc) {
                int remain = kc - k;
                if (remain >= 4) {
                    Bp[0] = B[(k + 0) * ldb + j];
                    Bp[1] = B[(k + 1) * ldb + j];
                    Bp[2] = B[(k + 2) * ldb + j];
                    Bp[3] = B[(k + 3) * ldb + j];
                } else {
                    for (int r = 0; r < remain; r++)
                        Bp[r] = B[(k + r) * ldb + j];
                    for (int r = remain; r < 4; r++)
                        Bp[r] = 0;
                }
            } else {
                memset(Bp, 0, 4);
            }
            Bp += 4;
        }
    }
}

/*
 * 8x8 micro-kernel: accumulate A_packed[MR x kc] * B_packed[kc x NR] into C[MR x NR].
 *
 * Register allocation:
 *   16 accumulators: c0L..c7L (rows 0-7, cols 0-3), c0R..c7R (rows 0-7, cols 4-7)
 *   2 A loads: aLo (rows 0-3), aHi (rows 4-7)
 *   2 B loads: bLo (cols 0-3), bHi (cols 4-7)
 *   Total: 20 out of 32 NEON registers
 */
static inline void micro_8x8(int kc, const int8_t *pa, const int8_t *pb,
                              int32_t *C, int ldc, int mr, int nr) {
    int32x4_t c0L = vdupq_n_s32(0), c0R = vdupq_n_s32(0);
    int32x4_t c1L = vdupq_n_s32(0), c1R = vdupq_n_s32(0);
    int32x4_t c2L = vdupq_n_s32(0), c2R = vdupq_n_s32(0);
    int32x4_t c3L = vdupq_n_s32(0), c3R = vdupq_n_s32(0);
    int32x4_t c4L = vdupq_n_s32(0), c4R = vdupq_n_s32(0);
    int32x4_t c5L = vdupq_n_s32(0), c5R = vdupq_n_s32(0);
    int32x4_t c6L = vdupq_n_s32(0), c6R = vdupq_n_s32(0);
    int32x4_t c7L = vdupq_n_s32(0), c7R = vdupq_n_s32(0);

    int kc4 = (kc + 3) & ~3;
    for (int k = 0; k < kc4; k += 4) {
        /* A: MR=8 rows, 4 bytes each = 32 bytes = 2 x int8x16_t */
        int8x16_t aLo = vld1q_s8(pa);       /* rows 0-3, 4 bytes each */
        int8x16_t aHi = vld1q_s8(pa + 16);  /* rows 4-7, 4 bytes each */

        /* B: NR=8 cols, 4 bytes each = 32 bytes = 2 x int8x16_t */
        int8x16_t bLo = vld1q_s8(pb);       /* cols 0-3, 4 bytes each */
        int8x16_t bHi = vld1q_s8(pb + 16);  /* cols 4-7, 4 bytes each */

        /* row 0 x all cols */
        c0L = vdotq_laneq_s32(c0L, bLo, aLo, 0);
        c0R = vdotq_laneq_s32(c0R, bHi, aLo, 0);
        /* row 1 */
        c1L = vdotq_laneq_s32(c1L, bLo, aLo, 1);
        c1R = vdotq_laneq_s32(c1R, bHi, aLo, 1);
        /* row 2 */
        c2L = vdotq_laneq_s32(c2L, bLo, aLo, 2);
        c2R = vdotq_laneq_s32(c2R, bHi, aLo, 2);
        /* row 3 */
        c3L = vdotq_laneq_s32(c3L, bLo, aLo, 3);
        c3R = vdotq_laneq_s32(c3R, bHi, aLo, 3);
        /* row 4 */
        c4L = vdotq_laneq_s32(c4L, bLo, aHi, 0);
        c4R = vdotq_laneq_s32(c4R, bHi, aHi, 0);
        /* row 5 */
        c5L = vdotq_laneq_s32(c5L, bLo, aHi, 1);
        c5R = vdotq_laneq_s32(c5R, bHi, aHi, 1);
        /* row 6 */
        c6L = vdotq_laneq_s32(c6L, bLo, aHi, 2);
        c6R = vdotq_laneq_s32(c6R, bHi, aHi, 2);
        /* row 7 */
        c7L = vdotq_laneq_s32(c7L, bLo, aHi, 3);
        c7R = vdotq_laneq_s32(c7R, bHi, aHi, 3);

        pa += I8GEMM_MR * 4;  /* 32 bytes */
        pb += I8GEMM_NR * 4;  /* 32 bytes */
    }

    /* Store accumulators back to C, handling edge cases */
    int32x4_t *rows[8] = { &c0L, &c1L, &c2L, &c3L, &c4L, &c5L, &c6L, &c7L };
    int32x4_t *rows_r[8] = { &c0R, &c1R, &c2R, &c3R, &c4R, &c5R, &c6R, &c7R };

    for (int r = 0; r < mr; r++) {
        int32_t *Cr = C + r * ldc;
        if (nr >= 4) {
            int32x4_t old = vld1q_s32(Cr);
            vst1q_s32(Cr, vaddq_s32(old, *rows[r]));
        } else {
            int32_t tmp[4];
            vst1q_s32(tmp, *rows[r]);
            for (int j = 0; j < nr && j < 4; j++)
                Cr[j] += tmp[j];
        }
        if (nr > 4) {
            int rn = nr - 4;
            if (rn >= 4) {
                int32x4_t old = vld1q_s32(Cr + 4);
                vst1q_s32(Cr + 4, vaddq_s32(old, *rows_r[r]));
            } else {
                int32_t tmp[4];
                vst1q_s32(tmp, *rows_r[r]);
                for (int j = 0; j < rn; j++)
                    Cr[4 + j] += tmp[j];
            }
        }
    }
}

/*
 * INT8 GEMM with pre-packed A (weights).
 * C[M x N] += A_packed[M x K] * B[K x N]
 * C is int32, caller zeros it before first call.
 *
 * Uses GEBP tiling: N-tiles of NC, K-tiles of KC.
 */
void i8gemm_nn_packed_a(int M, int N, int K,
                        const int8_t *packed_a,
                        const int8_t *B, int ldb,
                        int32_t *C, int ldc) {
    int K4 = (K + 3) & ~3;
    int panels_m = (M + I8GEMM_MR - 1) / I8GEMM_MR;
    size_t panel_stride = (size_t)I8GEMM_MR * K4;

    /* Allocate B packing buffer: KC * NR bytes */
    int kc4_max = (I8GEMM_KC + 3) & ~3;
    int8_t *Bp = (int8_t *)malloc((size_t)kc4_max * I8GEMM_NR);
    if (!Bp) return;

    for (int jc = 0; jc < N; jc += I8GEMM_NC) {
        int nc = (N - jc < I8GEMM_NC) ? (N - jc) : I8GEMM_NC;

        for (int pc = 0; pc < K; pc += I8GEMM_KC) {
            int kc = (K - pc < I8GEMM_KC) ? (K - pc) : I8GEMM_KC;
            int kc4 = (kc + 3) & ~3;

            for (int jr = 0; jr < nc; jr += I8GEMM_NR) {
                int nr = (nc - jr < I8GEMM_NR) ? (nc - jr) : I8GEMM_NR;

                /* Pack B panel: B[pc:pc+kc, jc+jr:jc+jr+nr] */
                pack_b_panel(&B[pc * ldb + jc + jr], ldb, kc, nr, Bp);

                for (int ic = 0; ic < M; ic += I8GEMM_MR) {
                    int mr = (M - ic < I8GEMM_MR) ? (M - ic) : I8GEMM_MR;

                    /* A panel for rows ic..ic+mr, K offset pc */
                    int panel_idx = ic / I8GEMM_MR;
                    const int8_t *pa = packed_a + panel_idx * panel_stride + (size_t)pc * I8GEMM_MR;

                    micro_8x8(kc, pa, Bp,
                              &C[ic * ldc + jc + jr], ldc,
                              mr, nr);
                }
            }
        }
    }

    free(Bp);
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
