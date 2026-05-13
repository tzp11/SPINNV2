#ifndef SPKV2_SIMD_KERNELS_H
#define SPKV2_SIMD_KERNELS_H

#include "simd_common.h"

#ifdef __AVX2__

int kernel_conv_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_gemm_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_matmul_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_add_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_mul_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_sub_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_div_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_relu_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_sigmoid_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_softmax_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_reduce_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_transpose_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);

#endif /* __AVX2__ */

#endif /* SPKV2_SIMD_KERNELS_H */
