#include "kernel_common.h"

#include <math.h>
#include <string.h>

size_t spkv2_kernel_elem_count(const Spkv2TensorRecord *record) {
    size_t count = 1;
    for (uint16_t i = 0; i < record->rank; i++) {
        count *= record->shape[i];
    }
    return count;
}

int spkv2_kernel_normalize_axis(int axis, uint16_t rank) {
    if (axis < 0) axis += (int)rank;
    return axis;
}

size_t spkv2_kernel_linear_to_coords(size_t linear, const Spkv2TensorRecord *record, uint32_t coords[8]) {
    (void)linear;
    for (int i = (int)record->rank - 1; i >= 0; i--) {
        uint32_t dim = record->shape[i];
        coords[i] = dim == 0 ? 0 : (uint32_t)(linear % dim);
        linear = dim == 0 ? 0 : linear / dim;
    }
    return linear;
}

size_t spkv2_kernel_coords_to_linear(const Spkv2TensorRecord *record, const uint32_t coords[8]) {
    size_t index = 0;
    for (uint16_t i = 0; i < record->rank; i++) {
        index = index * record->shape[i] + coords[i];
    }
    return index;
}

size_t spkv2_kernel_broadcast_index(const Spkv2TensorRecord *in_rec, const Spkv2TensorRecord *out_rec, size_t out_index) {
    uint32_t out_coords[8] = {0};
    uint32_t in_coords[8] = {0};
    spkv2_kernel_linear_to_coords(out_index, out_rec, out_coords);
    int rank_delta = (int)out_rec->rank - (int)in_rec->rank;
    for (uint16_t i = 0; i < in_rec->rank; i++) {
        uint32_t coord = out_coords[i + rank_delta];
        in_coords[i] = in_rec->shape[i] == 1 ? 0 : coord;
    }
    return spkv2_kernel_coords_to_linear(in_rec, in_coords);
}

int spkv2_kernel_tensor_axis_values(
    const Spkv2Context *ctx,
    const Spkv2NodeRecord *node,
    const Spkv2AttrRecord *attr,
    uint16_t rank,
    int axes[8],
    int *axis_count) {
    *axis_count = 0;
    if (node->input_count > 1) {
        const Spkv2TensorRecord *axes_rec = ctx->tensors[node->inputs[1]].record;
        const float *axes_data = (const float *)ctx->tensors[node->inputs[1]].data;
        size_t count = spkv2_kernel_elem_count(axes_rec);
        if (count > 8) return -11;
        for (size_t i = 0; i < count; i++) {
            axes[*axis_count] = spkv2_kernel_normalize_axis((int)axes_data[i], rank);
            (*axis_count)++;
        }
        return 0;
    }
    if (attr->extra_count > 0) {
        if (attr->extra_count > 8) return -11;
        for (int i = 0; i < attr->extra_count; i++) {
            axes[*axis_count] = spkv2_kernel_normalize_axis(attr->extra[i], rank);
            (*axis_count)++;
        }
        return 0;
    }
    for (uint16_t i = 0; i < rank; i++) {
        axes[*axis_count] = (int)i;
        (*axis_count)++;
    }
    return 0;
}

int spkv2_kernel_axis_in_set(int axis, const int axes[8], int axis_count) {
    for (int i = 0; i < axis_count; i++) {
        if (axes[i] == axis) return 1;
    }
    return 0;
}

int spkv2_kernel_get_attr(const Spkv2Context *ctx, const Spkv2NodeRecord *node, Spkv2AttrRecord *attr) {
    if (node->attr_offset > ctx->attrs_size ||
        node->attr_size > ctx->attrs_size - node->attr_offset ||
        node->attr_size < sizeof(Spkv2AttrRecord)) {
        return -10;
    }
    memcpy(attr, ctx->attrs + node->attr_offset, sizeof(Spkv2AttrRecord));
    return 0;
}

float spkv2_kernel_apply_fused_activation_scalar(float value, int activation) {
    if (activation == 1) {
        return value > 0.0f ? value : 0.0f;
    }
    if (activation == 2) {
        return value / (1.0f + expf(-value));
    }
    return value;
}

