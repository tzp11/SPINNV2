#include "reference_kernels.h"

int kernel_conv_im2col(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (attr.group != 1) return -99;
    if (!scratch && node->scratch_bytes > 0) return -13;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    float *patch = (float *)scratch;

    int N = (int)x_rec->shape[0];
    int C = (int)x_rec->shape[1];
    int H = (int)x_rec->shape[2];
    int W = (int)x_rec->shape[3];
    int M = (int)w_rec->shape[0];
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];
    int patch_len = C * kH * kW;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int pi = 0;
                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * attr.strides[0] + kh * attr.dilations[0] - attr.pads[0];
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * attr.strides[1] + kw * attr.dilations[1] - attr.pads[1];
                            float value = 0.0f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                value = x[((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw];
                            }
                            patch[pi++] = value;
                        }
                    }
                }
                for (int m = 0; m < M; m++) {
                    const float *w_row = w + (size_t)m * patch_len;
                    float sum = bias ? bias[m] : 0.0f;
                    for (int k = 0; k < patch_len; k++) {
                        sum += patch[k] * w_row[k];
                    }
                    sum = spkv2_kernel_apply_fused_activation_scalar(sum, attr.fused_activation);
                    y[((size_t)n * M * outH * outW) + ((size_t)m * outH * outW) + ((size_t)oh * outW) + ow] = sum;
                }
            }
        }
    }
    return 0;
}

