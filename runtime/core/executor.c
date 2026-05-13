#include "context.h"
#include "spkv2_platform.h"
#include "spkv2_runtime.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* --- Per-op profiling (env SPKV2_PROFILE=1) --- */
#define SPKV2_PROF_MAX_OPS 64
#define SPKV2_PROF_MAX_NODES 4096
static int    s_prof_enabled   = -1;   /* -1 = uninit, 0 = off, 1 = on */
static double s_prof_total[SPKV2_PROF_MAX_OPS];
static int    s_prof_count[SPKV2_PROF_MAX_OPS];
static double s_prof_node_total[SPKV2_PROF_MAX_NODES];
static int    s_prof_node_count[SPKV2_PROF_MAX_NODES];
static int    s_prof_runs      = 0;
static const Spkv2Context *s_prof_last_ctx = NULL;

static const char *op_name(uint16_t op) {
    switch (op) {
        case SPKV2_OP_ADD: return "Add";
        case SPKV2_OP_CONV: return "Conv";
        case SPKV2_OP_MUL: return "Mul";
        case SPKV2_OP_SUB: return "Sub";
        case SPKV2_OP_DIV: return "Div";
        case SPKV2_OP_RELU: return "Relu";
        case SPKV2_OP_SIGMOID: return "Sigmoid";
        case SPKV2_OP_GEMM: return "Gemm";
        case SPKV2_OP_MATMUL: return "MatMul";
        case SPKV2_OP_CONCAT: return "Concat";
        case SPKV2_OP_SPLIT: return "Split";
        case SPKV2_OP_RESHAPE: return "Reshape";
        case SPKV2_OP_RESIZE: return "Resize";
        case SPKV2_OP_TRANSPOSE: return "Transpose";
        case SPKV2_OP_MAXPOOL: return "MaxPool";
        case SPKV2_OP_SOFTMAX: return "Softmax";
        case SPKV2_OP_TOPK: return "TopK";
        case SPKV2_OP_FLATTEN: return "Flatten";
        case SPKV2_OP_UNSQUEEZE: return "Unsqueeze";
        case SPKV2_OP_TILE: return "Tile";
        case SPKV2_OP_GATHERELEMENTS: return "GatherElements";
        case SPKV2_OP_REDUCEMAX: return "ReduceMax";
        case SPKV2_OP_REDUCEMEAN: return "ReduceMean";
        case SPKV2_OP_CAST: return "Cast";
        case SPKV2_OP_MOD: return "Mod";
        default: return "Unknown";
    }
}

static double prof_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static const char *kernel_kind_name(uint16_t kind) {
    switch (kind) {
        case SPKV2_KERNEL_REFERENCE: return "reference";
        case SPKV2_KERNEL_DIRECT: return "direct";
        case SPKV2_KERNEL_IM2COL_GEMM: return "im2col_gemm";
        case SPKV2_KERNEL_PACKED_GEMM: return "packed_gemm";
        case SPKV2_KERNEL_POINTWISE_1X1: return "pointwise_1x1";
        case SPKV2_KERNEL_DEPTHWISE_DIRECT: return "depthwise_direct";
        case SPKV2_KERNEL_WINOGRAD_3X3S1: return "winograd_3x3s1";
        case SPKV2_KERNEL_CONV3X3S2_DIRECT: return "conv3x3s2_direct";
        default: return "unknown";
    }
}

static const Spkv2KernelSpecRecord *prof_kernel_spec(const Spkv2Context *ctx, const Spkv2NodeRecord *node) {
    if (!ctx || !node || node->kernel_spec_id == 0xFFFFFFFFu || node->kernel_spec_id >= ctx->kernel_spec_count) {
        return NULL;
    }
    return &ctx->kernel_spec_records[node->kernel_spec_id];
}

static int prof_get_attr(const Spkv2Context *ctx, const Spkv2NodeRecord *node, Spkv2AttrRecord *attr) {
    if (!ctx || !node || !attr) return -1;
    if (node->attr_offset > ctx->attrs_size ||
        node->attr_size > ctx->attrs_size - node->attr_offset ||
        node->attr_size < sizeof(Spkv2AttrRecord)) {
        return -10;
    }
    memcpy(attr, ctx->attrs + node->attr_offset, sizeof(Spkv2AttrRecord));
    return 0;
}

static void prof_init(void) {
    if (s_prof_enabled >= 0) return;
    const char *v = getenv("SPKV2_PROFILE");
    s_prof_enabled = (v && v[0] && v[0] != '0') ? 1 : 0;
    memset(s_prof_total, 0, sizeof(s_prof_total));
    memset(s_prof_count, 0, sizeof(s_prof_count));
    memset(s_prof_node_total, 0, sizeof(s_prof_node_total));
    memset(s_prof_node_count, 0, sizeof(s_prof_node_count));
    s_prof_runs = 0;
}

static void prof_dump(const Spkv2Context *ctx) {
    fprintf(stderr, "\n[SPKV2_PROFILE] per-op totals over %d runs:\n", s_prof_runs);
    fprintf(stderr, "  %-20s %8s %12s %12s\n", "op", "count", "total_ms", "avg_ms");
    /* sort indices by total_ms desc */
    int idx[SPKV2_PROF_MAX_OPS];
    int n = 0;
    for (int i = 0; i < SPKV2_PROF_MAX_OPS; i++) {
        if (s_prof_count[i] > 0) idx[n++] = i;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (s_prof_total[idx[j]] > s_prof_total[idx[i]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
    }
    double grand = 0.0;
    for (int i = 0; i < n; i++) grand += s_prof_total[idx[i]];
    for (int i = 0; i < n; i++) {
        int op = idx[i];
        fprintf(stderr, "  %-20s %8d %12.3f %12.4f (%5.1f%%)\n",
                op_name((uint16_t)op),
                s_prof_count[op] / (s_prof_runs > 0 ? s_prof_runs : 1),
                s_prof_total[op],
                s_prof_total[op] / (s_prof_count[op] > 0 ? s_prof_count[op] : 1),
                grand > 0 ? 100.0 * s_prof_total[op] / grand : 0.0);
    }
    fprintf(stderr, "  %-20s %8s %12.3f\n", "TOTAL", "", grand);

    if (!ctx) return;
    fprintf(stderr, "\n[SPKV2_PROFILE] conv nodes over %d runs:\n", s_prof_runs);
    fprintf(stderr,
            "  %-5s %-18s %8s %10s %22s %12s %10s\n",
            "node", "algo", "count", "avg_ms", "N,C,H,W->OC,k,s,p,g", "total_ms", "scratch");
    for (uint32_t i = 0; i < ctx->header.num_nodes && i < SPKV2_PROF_MAX_NODES; i++) {
        const Spkv2NodeRecord *node = &ctx->node_records[i];
        if (node->op_type != SPKV2_OP_CONV || s_prof_node_count[i] == 0) continue;
        if (node->input_count < 2 || node->output_count < 1) continue;
        const Spkv2TensorRecord *x = ctx->tensors[node->inputs[0]].record;
        const Spkv2TensorRecord *w = ctx->tensors[node->inputs[1]].record;
        if (!x || !w || x->rank != 4 || w->rank != 4) continue;
        Spkv2AttrRecord attr;
        memset(&attr, 0, sizeof(attr));
        if (prof_get_attr(ctx, node, &attr) != 0) continue;
        const Spkv2KernelSpecRecord *spec = prof_kernel_spec(ctx, node);
        uint16_t kind = spec ? spec->kernel_kind : SPKV2_KERNEL_REFERENCE;
        char shape[128];
        snprintf(shape, sizeof(shape), "%u,%u,%u,%u->%u,%ux%u,%dx%d,%d/%d/%d/%d,%d",
                 x->shape[0], x->shape[1], x->shape[2], x->shape[3],
                 w->shape[0], w->shape[2], w->shape[3],
                 attr.strides[0], attr.strides[1],
                 attr.pads[0], attr.pads[1], attr.pads[2], attr.pads[3],
                 attr.group);
        fprintf(stderr, "  %-5u %-18s %8d %10.4f %22s %12.3f %10u\n",
                node->id,
                kernel_kind_name(kind),
                s_prof_node_count[i] / (s_prof_runs > 0 ? s_prof_runs : 1),
                s_prof_node_total[i] / (s_prof_node_count[i] > 0 ? s_prof_node_count[i] : 1),
                shape,
                s_prof_node_total[i],
                node->scratch_bytes);
    }
}

static uint64_t align16(uint64_t value) {
    return (value + 15u) & ~15u;
}

enum { PREP_SGEMM_MR = 6 };

static void *pack_conv_weight_for_sgemm(int M, int K, const float *weight, int lda) {
    int m_blocks = (M + PREP_SGEMM_MR - 1) / PREP_SGEMM_MR;
    size_t total = (size_t)m_blocks * (size_t)K * PREP_SGEMM_MR;
    float *packed = (float *)spkv2_platform_malloc(total * sizeof(float));
    if (!packed) return NULL;

    float *dst = packed;
    for (int m0 = 0; m0 < M; m0 += PREP_SGEMM_MR) {
        int cm = M - m0 < PREP_SGEMM_MR ? M - m0 : PREP_SGEMM_MR;
        for (int k = 0; k < K; k++) {
            int m = 0;
            for (; m < cm; m++) {
                dst[m] = weight[(size_t)(m0 + m) * (size_t)lda + (size_t)k];
            }
            for (; m < PREP_SGEMM_MR; m++) {
                dst[m] = 0.0f;
            }
            dst += PREP_SGEMM_MR;
        }
    }
    return packed;
}

static int prepare_conv_weight_caches(Spkv2Context *ctx) {
    if (!ctx || !ctx->node_cache) return 0;

    for (uint32_t i = 0; i < ctx->header.num_nodes; i++) {
        const Spkv2NodeRecord *node = &ctx->node_records[i];
        if (node->op_type != SPKV2_OP_CONV || node->input_count < 2) continue;
        if (node->id >= ctx->node_cache_count || ctx->node_cache[node->id]) continue;

        const Spkv2KernelSpecRecord *spec = prof_kernel_spec(ctx, node);
        if (!spec || spec->backend != SPKV2_BACKEND_SIMD) continue;
        if (spec->kernel_kind != SPKV2_KERNEL_IM2COL_GEMM &&
            spec->kernel_kind != SPKV2_KERNEL_POINTWISE_1X1) {
            continue;
        }

        Spkv2AttrRecord attr;
        if (prof_get_attr(ctx, node, &attr) != 0) continue;
        if (attr.group != 1) continue;

        const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
        const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
        if (!w_rec || !w || w_rec->rank != 4) continue;

        int out_c = (int)w_rec->shape[0];
        int k = (int)(w_rec->shape[1] * w_rec->shape[2] * w_rec->shape[3]);
        if (out_c <= 0 || k <= 0) continue;

        ctx->node_cache[node->id] = pack_conv_weight_for_sgemm(out_c, k, w, k);
        if (!ctx->node_cache[node->id]) return -1;
    }
    return 0;
}

int spkv2_prepare(Spkv2Context *ctx, void *arena, size_t arena_size) {
    return spkv2_prepare_with_scratch(ctx, arena, arena_size, NULL, 0);
}

int spkv2_prepare_with_scratch(
    Spkv2Context *ctx,
    void *arena,
    size_t arena_size,
    void *scratch,
    size_t scratch_size) {
    if (!ctx) return -1;

    uint64_t required = ctx->header.activation_arena_bytes;
    uint64_t scratch_required = ctx->header.scratch_arena_bytes;

    if (arena) {
        if (arena_size < required) return -2;
        ctx->owned_arena = NULL;
        ctx->arena_size = arena_size;
    } else {
        arena = spkv2_platform_calloc(1, (size_t)required);
        if (!arena && required > 0) return -1;
        ctx->owned_arena = (uint8_t *)arena;
        ctx->arena_size = (size_t)required;
    }

    if (scratch_required > 0) {
        if (scratch) {
            if (scratch_size < scratch_required) return -2;
            ctx->owned_scratch = NULL;
            ctx->scratch = (uint8_t *)scratch;
            ctx->scratch_size = scratch_size;
        } else {
            scratch = spkv2_platform_calloc(1, (size_t)scratch_required);
            if (!scratch) return -1;
            ctx->owned_scratch = (uint8_t *)scratch;
            ctx->scratch = (uint8_t *)scratch;
            ctx->scratch_size = (size_t)scratch_required;
        }
    }

    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role == SPKV2_ROLE_WEIGHT) {
            if (record->data_offset > ctx->weights_size ||
                record->size_bytes > ctx->weights_size - record->data_offset) {
                return -3;
            }
            ctx->tensors[i].data = (uint8_t *)(ctx->weights + record->data_offset);
        } else if (record->memory_class == SPKV2_MEMORY_EXTERNAL) {
            ctx->tensors[i].data = NULL;
        } else {
            uint64_t end = record->data_offset + record->size_bytes;
            if (end > required) return -4;
            ctx->tensors[i].data = ((uint8_t *)arena) + record->data_offset;
        }
    }
    int cache_rc = prepare_conv_weight_caches(ctx);
    if (cache_rc != 0) return cache_rc;
    return 0;
}

int spkv2_set_input(Spkv2Context *ctx, int index, const void *data, size_t size) {
    if (!ctx || !data || index < 0) return -1;
    int current = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_INPUT) continue;
        if (current == index) {
            if (size != record->size_bytes) return -2;
            memcpy(ctx->tensors[i].data, data, size);
            return 0;
        }
        current++;
    }
    return -3;
}

int spkv2_bind_input(Spkv2Context *ctx, int index, void *data, size_t size) {
    if (!ctx || !data || index < 0) return -1;
    int current = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_INPUT) continue;
        if (current == index) {
            if (size != record->size_bytes) return -2;
            ctx->tensors[i].data = (uint8_t *)data;
            return 0;
        }
        current++;
    }
    return -3;
}

static int output_tensor_id(const Spkv2Context *ctx, int index, uint32_t *tensor_id) {
    if (!ctx || !tensor_id || index < 0) return -1;
    if (ctx->output_ids && (size_t)index < ctx->output_count) {
        uint32_t id = ctx->output_ids[index];
        if (id >= ctx->header.num_tensors) return -3;
        if (ctx->tensor_records[id].role != SPKV2_ROLE_OUTPUT) return -3;
        *tensor_id = id;
        return 0;
    }

    int current = 0;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_OUTPUT) continue;
        if (current == index) {
            *tensor_id = i;
            return 0;
        }
        current++;
    }
    return -3;
}

int spkv2_bind_output(Spkv2Context *ctx, int index, void *data, size_t size) {
    if (!ctx || !data || index < 0) return -1;
    uint32_t tensor_id = 0;
    int rc = output_tensor_id(ctx, index, &tensor_id);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *record = &ctx->tensor_records[tensor_id];
    if (size != record->size_bytes) return -2;
    ctx->tensors[tensor_id].data = (uint8_t *)data;
    return 0;
}

int spkv2_get_output(Spkv2Context *ctx, int index, void *data, size_t size) {
    if (!ctx || !data || index < 0) return -1;
    uint32_t tensor_id = 0;
    int rc = output_tensor_id(ctx, index, &tensor_id);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *record = &ctx->tensor_records[tensor_id];
    if (size != record->size_bytes) return -2;
    memcpy(data, ctx->tensors[tensor_id].data, size);
    return 0;
}

int spkv2_get_output_size(Spkv2Context *ctx, int index, size_t *out_size) {
    if (!ctx || !out_size || index < 0) return -1;
    uint32_t tensor_id = 0;
    int rc = output_tensor_id(ctx, index, &tensor_id);
    if (rc != 0) return rc;
    *out_size = (size_t)ctx->tensor_records[tensor_id].size_bytes;
    return 0;
}

int spkv2_run(Spkv2Context *ctx) {
    if (!ctx) return -1;
    prof_init();
    s_prof_last_ctx = ctx;
    for (uint32_t i = 0; i < ctx->header.num_tensors; i++) {
        const Spkv2TensorRecord *record = &ctx->tensor_records[i];
        if (record->role != SPKV2_ROLE_WEIGHT && ctx->tensors[i].data == NULL) {
            return -5;
        }
    }
    if (s_prof_enabled) {
        for (uint32_t i = 0; i < ctx->header.num_nodes; i++) {
            const Spkv2NodeRecord *nd = &ctx->node_records[i];
            double t0 = prof_now_ms();
            int rc = spkv2_execute_node(ctx, nd);
            double dt = prof_now_ms() - t0;
            if (nd->op_type < SPKV2_PROF_MAX_OPS) {
                s_prof_total[nd->op_type] += dt;
                s_prof_count[nd->op_type]++;
            }
            if (nd->id < SPKV2_PROF_MAX_NODES) {
                s_prof_node_total[nd->id] += dt;
                s_prof_node_count[nd->id]++;
            }
            if (rc != 0) return rc;
        }
        s_prof_runs++;
    } else {
        for (uint32_t i = 0; i < ctx->header.num_nodes; i++) {
            int rc = spkv2_execute_node(ctx, &ctx->node_records[i]);
            if (rc != 0) return rc;
        }
    }
    return 0;
}

void spkv2_profile_dump(void) {
    if (s_prof_enabled > 0 && s_prof_runs > 0) prof_dump(s_prof_last_ctx);
}
