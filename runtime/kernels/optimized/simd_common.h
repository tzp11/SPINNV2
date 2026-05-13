#ifndef SPKV2_SIMD_COMMON_H
#define SPKV2_SIMD_COMMON_H

#include "context.h"
#include "spkv2_format.h"

#ifdef __AVX2__

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define SIMD_MIN(a, b) ((a) < (b) ? (a) : (b))

size_t simd_elem_count(const Spkv2TensorRecord *r);
int simd_get_attr(const Spkv2Context *ctx, const Spkv2NodeRecord *node, Spkv2AttrRecord *attr);
void fused_activation_pass(float *data, size_t count, int act_type);
float apply_activation_scalar_simd(float x, int act_type);
__m256 apply_activation_avx2(__m256 x, int act_type);
const Spkv2KernelSpecRecord *simd_node_spec(const Spkv2Context *ctx, const Spkv2NodeRecord *node);
size_t simd_broadcast_index(const Spkv2TensorRecord *in_rec, const Spkv2TensorRecord *out_rec, size_t out_index);
int same_shape(const Spkv2TensorRecord *a, const Spkv2TensorRecord *b);
__m256 fast_exp_avx2(__m256 x);
__m256 sigmoid_avx2(__m256 x);
float hmax_avx2(__m256 v);
float hsum_avx2(__m256 v);

#endif /* __AVX2__ */

#endif /* SPKV2_SIMD_COMMON_H */
