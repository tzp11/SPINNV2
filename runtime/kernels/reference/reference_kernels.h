#ifndef SPKV2_REFERENCE_KERNELS_H
#define SPKV2_REFERENCE_KERNELS_H

#include "kernel_common.h"

int kernel_add(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_binary(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_relu(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_sigmoid(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_flatten(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_copy(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_gemm(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_gemm_cpu_direct(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_softmax(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_conv(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_conv_im2col(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_maxpool(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_transpose(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_concat(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_split(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_reduce(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_matmul(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_resize(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_tile(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_unsqueeze(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_gather_elements(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_topk(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_slice(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_gather(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_avgpool(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_global_avgpool(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_clip(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
int kernel_leakyrelu(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);

#endif /* SPKV2_REFERENCE_KERNELS_H */
