#include "reference_kernels.h"

#include <stdlib.h>

/* Heap entry: keep a heap of size K. */
typedef struct { float v; uint32_t idx; } TopkPair;

/* Sift-down for min-heap (used when largest=1: root is the smallest of top K). */
static inline void topk_sift_down_min(TopkPair *h, size_t n, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < n && h[l].v < h[m].v) m = l;
        if (r < n && h[r].v < h[m].v) m = r;
        if (m == i) break;
        TopkPair t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

/* Sift-down for max-heap (used when largest=0: root is the largest of bottom K). */
static inline void topk_sift_down_max(TopkPair *h, size_t n, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < n && h[l].v > h[m].v) m = l;
        if (r < n && h[r].v > h[m].v) m = r;
        if (m == i) break;
        TopkPair t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

/* Comparator for sorted output: descending if largest, ascending otherwise. */
static int topk_cmp_desc(const void *a, const void *b) {
    float av = ((const TopkPair *)a)->v, bv = ((const TopkPair *)b)->v;
    if (av < bv) return 1;
    if (av > bv) return -1;
    /* Tie-break by smaller index (matches typical ONNX behavior). */
    return ((const TopkPair *)a)->idx < ((const TopkPair *)b)->idx ? -1 : 1;
}
static int topk_cmp_asc(const void *a, const void *b) {
    float av = ((const TopkPair *)a)->v, bv = ((const TopkPair *)b)->v;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return ((const TopkPair *)a)->idx < ((const TopkPair *)b)->idx ? -1 : 1;
}

int kernel_topk(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *k_rec = ctx->tensors[node->inputs[1]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *k_data = (const float *)ctx->tensors[node->inputs[1]].data;
    float *values = (float *)ctx->tensors[node->outputs[0]].data;
    float *indices = (float *)ctx->tensors[node->outputs[1]].data;
    (void)k_rec;
    int axis = spkv2_kernel_normalize_axis(attr.axis, x_rec->rank);
    if (axis < 0 || axis >= (int)x_rec->rank) return -11;
    size_t k = (size_t)k_data[0];
    size_t dim = x_rec->shape[axis];
    if (k == 0 || dim == 0) return 0;
    if (k > dim) k = dim;

    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= x_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < x_rec->rank; i++) inner *= x_rec->shape[i];

    /* Use scratch if it can hold k pairs; otherwise allocate. */
    TopkPair *heap = NULL;
    size_t need_bytes = k * sizeof(TopkPair);
    int owned = 0;
    if (scratch && need_bytes <= (size_t)0xFFFFFFFF) {
        heap = (TopkPair *)scratch;
    } else {
        heap = (TopkPair *)malloc(need_bytes);
        if (!heap) return -1;
        owned = 1;
    }

    for (size_t o = 0; o < outer; o++) {
        for (size_t in = 0; in < inner; in++) {
            /* Build initial heap from first k elements. */
            for (size_t d = 0; d < k; d++) {
                heap[d].v   = x[(o * dim + d) * inner + in];
                heap[d].idx = (uint32_t)d;
            }
            if (attr.largest) {
                /* Min-heap (root = smallest of top k). */
                for (long i = (long)(k / 2) - 1; i >= 0; i--)
                    topk_sift_down_min(heap, k, (size_t)i);
                for (size_t d = k; d < dim; d++) {
                    float v = x[(o * dim + d) * inner + in];
                    if (v > heap[0].v) {
                        heap[0].v = v;
                        heap[0].idx = (uint32_t)d;
                        topk_sift_down_min(heap, k, 0);
                    }
                }
            } else {
                /* Max-heap (root = largest of bottom k). */
                for (long i = (long)(k / 2) - 1; i >= 0; i--)
                    topk_sift_down_max(heap, k, (size_t)i);
                for (size_t d = k; d < dim; d++) {
                    float v = x[(o * dim + d) * inner + in];
                    if (v < heap[0].v) {
                        heap[0].v = v;
                        heap[0].idx = (uint32_t)d;
                        topk_sift_down_max(heap, k, 0);
                    }
                }
            }
            /* Sort heap into output order (sorted=1: descending if largest). */
            qsort(heap, k, sizeof(TopkPair),
                  attr.largest ? topk_cmp_desc : topk_cmp_asc);
            for (size_t out_i = 0; out_i < k; out_i++) {
                values[(o * k + out_i) * inner + in]  = heap[out_i].v;
                indices[(o * k + out_i) * inner + in] = (float)heap[out_i].idx;
            }
        }
    }
    if (owned) free(heap);
    return 0;
}

