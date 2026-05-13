#ifndef SPKV2_SIMD_SGEMM_H
#define SPKV2_SIMD_SGEMM_H

#include "simd_common.h"

#ifdef __AVX2__

#define SGEMM_MR 6
#define SGEMM_NR 16
#define SGEMM_KC 256
#define SGEMM_NC 128
#define OMP_MIN_M 24
#define PF_AHEAD 8

float *sgemm_pack_a_impl(int M, int K, const float *A, int lda);
void sgemm_nn_packed_a_impl_run(int M, int N, int K, const float *packed_a, const float *B, int ldb, float *C, int ldc, int allow_parallel);
void sgemm_nn_packed_a(int M, int N, int K, const float *PA, const float *B, int ldb, float *C, int ldc);
void sgemm_nn(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);

#endif /* __AVX2__ */

#endif /* SPKV2_SIMD_SGEMM_H */
