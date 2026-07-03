#ifndef SPKV2_SIMD_SGEMM_H
#define SPKV2_SIMD_SGEMM_H

#include "simd_common.h"

#if defined(__AVX2__) || defined(__ARM_NEON)

#if defined(__AVX2__)
#define SGEMM_MR 6
#define SGEMM_NR 16
#elif defined(__ARM_NEON)
#define SGEMM_MR 12
#define SGEMM_NR 8
#endif
#define SGEMM_KC 256
#define SGEMM_NC 256
#define OMP_MIN_M 24
#define PF_AHEAD 8

float *sgemm_pack_a_impl(int M, int K, const float *A, int lda);
float *sgemm_pack_a_fp16(int M, int K, const void *A_fp16, int lda);
void sgemm_nn_packed_a_impl_run(int M, int N, int K, const float *packed_a, const float *B, int ldb, float *C, int ldc, int allow_parallel);
void sgemm_nn_packed_a(int M, int N, int K, const float *PA, const float *B, int ldb, float *C, int ldc);
void sgemm_nn(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);

#endif /* __AVX2__ || __ARM_NEON */

#endif /* SPKV2_SIMD_SGEMM_H */
