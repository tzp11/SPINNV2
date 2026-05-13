#ifndef SPKV2_REFERENCE_KERNEL_COMMON_H
#define SPKV2_REFERENCE_KERNEL_COMMON_H

#include "context.h"

#include <stddef.h>
#include <stdint.h>

typedef int (*NodeKernelFn)(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);

size_t spkv2_kernel_elem_count(const Spkv2TensorRecord *record);
int spkv2_kernel_normalize_axis(int axis, uint16_t rank);
size_t spkv2_kernel_linear_to_coords(size_t linear, const Spkv2TensorRecord *record, uint32_t coords[8]);
size_t spkv2_kernel_coords_to_linear(const Spkv2TensorRecord *record, const uint32_t coords[8]);
size_t spkv2_kernel_broadcast_index(const Spkv2TensorRecord *in_rec, const Spkv2TensorRecord *out_rec, size_t out_index);
int spkv2_kernel_tensor_axis_values(
    const Spkv2Context *ctx,
    const Spkv2NodeRecord *node,
    const Spkv2AttrRecord *attr,
    uint16_t rank,
    int axes[8],
    int *axis_count);
int spkv2_kernel_axis_in_set(int axis, const int axes[8], int axis_count);
int spkv2_kernel_get_attr(const Spkv2Context *ctx, const Spkv2NodeRecord *node, Spkv2AttrRecord *attr);
float spkv2_kernel_apply_fused_activation_scalar(float value, int activation);

#endif /* SPKV2_REFERENCE_KERNEL_COMMON_H */
