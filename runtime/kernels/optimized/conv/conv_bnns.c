/*
 * conv_bnns.c — Apple BNNS FP32 convolution kernel
 *
 * Caches BNNSFilter in node_cache on first invocation;
 * subsequent calls reuse it via BNNSFilterApply.
 * Only compiled on Apple platforms.
 */

#ifdef __APPLE__

#include "simd_common.h"
#include "context.h"
#include "spkv2_format.h"

#include <string.h>
#include <math.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <Accelerate/Accelerate.h>

static void bnns_filter_dtor(void *ptr) {
    if (ptr) BNNSFilterDestroy((BNNSFilter)ptr);
}

static BNNSActivationFunction map_activation(int act_type) {
    switch (act_type) {
        case 1: return BNNSActivationFunctionRectifiedLinear;
        default: return BNNSActivationFunctionIdentity;
    }
}

int kernel_conv_bnns(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;

    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;

    int N_batch = (int)x_rec->shape[0];
    int C_in  = (int)x_rec->shape[1];
    int H     = (int)x_rec->shape[2];
    int W     = (int)x_rec->shape[3];
    int C_out = (int)w_rec->shape[0];
    int kH    = (int)w_rec->shape[2];
    int kW    = (int)w_rec->shape[3];
    int outH  = (int)y_rec->shape[2];
    int outW  = (int)y_rec->shape[3];
    int spatial = outH * outW;

    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2
                        ? (const float *)ctx->tensors[node->inputs[2]].data
                        : NULL;
    const float *residual = node->input_count >= 4
                            ? (const float *)ctx->tensors[node->inputs[3]].data
                            : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    /* Create and cache BNNSFilter on first call */
    if (ctx->node_cache && node->id < ctx->node_cache_count && !ctx->node_cache[node->id]) {
        int act_for_bnns = attr.fused_activation;
        /* SiLU (act_type==2) not natively supported by BNNS — apply manually after */
        if (act_for_bnns == 2) act_for_bnns = 0;
        /* When a residual is present the activation must be deferred until after
         * the residual add: correct order is relu(conv(x) + residual), not
         * relu(conv(x)) + residual.  Create the BNNS filter as identity here
         * and apply the activation manually in the batch loop below. */
        if (node->input_count >= 4 && act_for_bnns == 1) act_for_bnns = 0;

        BNNSLayerParametersConvolution p;
        memset(&p, 0, sizeof(p));

        p.i_desc.layout = BNNSDataLayoutImageCHW;
        p.i_desc.size[0] = (size_t)W;
        p.i_desc.size[1] = (size_t)H;
        p.i_desc.size[2] = (size_t)C_in;
        p.i_desc.data_type = BNNSDataTypeFloat32;

        p.w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
        p.w_desc.size[0] = (size_t)kW;
        p.w_desc.size[1] = (size_t)kH;
        p.w_desc.size[2] = (size_t)C_in;
        p.w_desc.size[3] = (size_t)C_out;
        p.w_desc.data_type = (w_rec->dtype == SPKV2_DTYPE_FP16)
            ? BNNSDataTypeFloat16 : BNNSDataTypeFloat32;
        p.w_desc.data = (void *)w;

        p.o_desc.layout = BNNSDataLayoutImageCHW;
        p.o_desc.size[0] = (size_t)outW;
        p.o_desc.size[1] = (size_t)outH;
        p.o_desc.size[2] = (size_t)C_out;
        p.o_desc.data_type = BNNSDataTypeFloat32;

        if (bias) {
            p.bias.layout = BNNSDataLayoutVector;
            p.bias.size[0] = (size_t)C_out;
            p.bias.data_type = BNNSDataTypeFloat32;
            p.bias.data = (void *)bias;
        }

        p.x_stride = (size_t)attr.strides[1];
        p.y_stride = (size_t)attr.strides[0];
        p.x_padding = (size_t)attr.pads[1];
        p.y_padding = (size_t)attr.pads[0];
        p.x_dilation_stride = (size_t)attr.dilations[1];
        p.y_dilation_stride = (size_t)attr.dilations[0];

        p.activation.function = map_activation(act_for_bnns);

        BNNSFilterParameters fp;
        memset(&fp, 0, sizeof(fp));
        /* Honour spkv2_get_num_threads(): 0 = let BNNS choose (all cores),
         * 1 = force single-thread, N > 1 = limit to N threads. */
        int nt = spkv2_get_num_threads();
        fp.n_threads = (nt > 0) ? (size_t)nt : 0;
        BNNSFilter f = BNNSFilterCreateLayerConvolution(&p, &fp);
        if (!f) return -1;

        ctx->node_cache[node->id] = (void *)f;
        if (ctx->node_cache_dtors)
            ctx->node_cache_dtors[node->id] = bnns_filter_dtor;
    }

    BNNSFilter filter = (BNNSFilter)ctx->node_cache[node->id];
    if (!filter) return -1;

    size_t in_stride  = (size_t)C_in  * H * W;
    size_t out_stride = (size_t)C_out * outH * outW;

    for (int n = 0; n < N_batch; n++) {
        int status = BNNSFilterApply(filter,
                                     x + (size_t)n * in_stride,
                                     y + (size_t)n * out_stride);
        if (status != 0) return -1;

        /* Residual add (BNNS Conv doesn't support fused residual) */
        if (residual) {
            float *y_n = y + (size_t)n * out_stride;
            const float *r_n = residual + (size_t)n * out_stride;
            vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)out_stride);
            /* Apply deferred activation: when a residual exists, ReLU was NOT
             * baked into the BNNS filter (to preserve correct order), so apply
             * it now after the residual has been accumulated. */
            if (attr.fused_activation == 1)
                fused_activation_pass(y_n, out_stride, 1);
        }

        /* SiLU not supported by BNNS natively — apply manually */
        if (attr.fused_activation == 2) {
            fused_activation_pass(y + (size_t)n * out_stride, out_stride, 2);
        }
    }

    return 0;
}

#pragma clang diagnostic pop

#endif /* __APPLE__ */
