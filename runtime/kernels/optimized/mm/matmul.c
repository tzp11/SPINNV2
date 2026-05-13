#include "simd_kernels.h"
#include "simd_sgemm.h"

#ifdef __AVX2__

#include <string.h>

int kernel_matmul_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    if (a_rec->rank < 2 || b_rec->rank < 2 || y_rec->rank < 2)
        return -11;

    int M = (int)y_rec->shape[y_rec->rank - 2];
    int N = (int)y_rec->shape[y_rec->rank - 1];
    int K = (int)a_rec->shape[a_rec->rank - 1];
    size_t batch = simd_elem_count(y_rec) / ((size_t)M * N);

    for (size_t bi = 0; bi < batch; bi++) {
        /* compute batch indices with broadcasting */
        uint32_t y_bc[8] = {0};
        size_t tmp = bi;
        for (int d = (int)y_rec->rank - 3; d >= 0; d--) {
            y_bc[d] = (uint32_t)(tmp % y_rec->shape[d]);
            tmp /= y_rec->shape[d];
        }

        size_t a_batch = 0;
        for (uint16_t d = 0; d + 2 < a_rec->rank; d++) {
            int yd = (int)y_rec->rank - 2 - ((int)a_rec->rank - 2) + d;
            uint32_t coord = yd >= 0 ? y_bc[yd] : 0;
            a_batch = a_batch * a_rec->shape[d] +
                      (a_rec->shape[d] == 1 ? 0 : coord);
        }
        size_t b_batch = 0;
        for (uint16_t d = 0; d + 2 < b_rec->rank; d++) {
            int yd = (int)y_rec->rank - 2 - ((int)b_rec->rank - 2) + d;
            uint32_t coord = yd >= 0 ? y_bc[yd] : 0;
            b_batch = b_batch * b_rec->shape[d] +
                      (b_rec->shape[d] == 1 ? 0 : coord);
        }

        const float *a_base = a + a_batch * M * K;
        const float *b_base = b + b_batch * K * N;
        float *y_base = y + bi * M * N;

        memset(y_base, 0, (size_t)M * N * sizeof(float));
        sgemm_nn(M, N, K, a_base, K, b_base, N, y_base, N);
    }
    return 0;
}


#endif /* __AVX2__ */
