#include "reference_kernels.h"

int kernel_conv(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int N = (int)x_rec->shape[0];
    int C = (int)x_rec->shape[1];
    int H = (int)x_rec->shape[2];
    int W = (int)x_rec->shape[3];
    int M = (int)w_rec->shape[0];
    int group = attr.group > 0 ? attr.group : 1;
    if (C % group != 0 || M % group != 0) return -12;
    int C_per_group = C / group;
    int M_per_group = M / group;
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            int g = m / M_per_group;
            int c_begin = g * C_per_group;
            int c_end = c_begin + C_per_group;
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias ? bias[m] : 0.0f;
                    for (int c = c_begin; c < c_end; c++) {
                        int wc = c - c_begin;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * attr.strides[0] + kh * attr.dilations[0] - attr.pads[0];
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * attr.strides[1] + kw * attr.dilations[1] - attr.pads[1];
                                if (iw < 0 || iw >= W) continue;
                                size_t xi = ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw;
                                size_t wi = ((size_t)m * C_per_group * kH * kW) + ((size_t)wc * kH * kW) + ((size_t)kh * kW) + kw;
                                sum += x[xi] * w[wi];
                            }
                        }
                    }
                    sum = spkv2_kernel_apply_fused_activation_scalar(sum, attr.fused_activation);
                    y[((size_t)n * M * outH * outW) + ((size_t)m * outH * outW) + ((size_t)oh * outW) + ow] = sum;
                }
            }
        }
    }
    return 0;
}

