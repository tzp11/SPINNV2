#ifndef SPKV2_INT8_GEMM_H
#define SPKV2_INT8_GEMM_H

#include <stdint.h>
#include <stddef.h>

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

#define I8GEMM_MR 8
#define I8GEMM_NR 8
#define I8GEMM_KC 256
#define I8GEMM_NC 256

int8_t *i8gemm_pack_a(int M, int K, const int8_t *A, int lda);
void i8gemm_nn_packed_a(int M, int N, int K,
                        const int8_t *packed_a,
                        const int8_t *B, int ldb,
                        int32_t *C, int ldc);

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */

#endif /* SPKV2_INT8_GEMM_H */
