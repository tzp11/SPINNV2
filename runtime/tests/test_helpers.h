#ifndef SPKV2_TEST_HELPERS_H
#define SPKV2_TEST_HELPERS_H

#include "spkv2_format.h"
#include "context.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_test_pass_count = 0;
static int g_test_fail_count = 0;

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                        \
        if ((a) != (b)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s != %s (%d != %d)\n", __FILE__,     \
                    __LINE__, #a, #b, (int)(a), (int)(b));                      \
            g_test_fail_count++;                                                \
            return 1;                                                           \
        }                                                                       \
        g_test_pass_count++;                                                    \
    } while (0)

#define ASSERT_NEAR(a, b, eps)                                                 \
    do {                                                                        \
        float _a = (float)(a), _b = (float)(b);                                \
        if (fabsf(_a - _b) > (float)(eps)) {                                   \
            fprintf(stderr, "FAIL %s:%d: %s != %s (%.8f != %.8f, eps=%.8f)\n", \
                    __FILE__, __LINE__, #a, #b, _a, _b, (float)(eps));         \
            g_test_fail_count++;                                                \
            return 1;                                                           \
        }                                                                       \
        g_test_pass_count++;                                                    \
    } while (0)

#define ASSERT_OK(expr)                                                        \
    do {                                                                        \
        int _rc = (expr);                                                       \
        if (_rc != 0) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s returned %d\n", __FILE__,          \
                    __LINE__, #expr, _rc);                                      \
            g_test_fail_count++;                                                \
            return 1;                                                           \
        }                                                                       \
        g_test_pass_count++;                                                    \
    } while (0)

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "FAIL %s:%d: %s is false\n", __FILE__, __LINE__,   \
                    #cond);                                                     \
            g_test_fail_count++;                                                \
            return 1;                                                           \
        }                                                                       \
        g_test_pass_count++;                                                    \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                        \
        printf("  %-50s ", #fn);                                               \
        if (fn() == 0) {                                                        \
            printf("PASS\n");                                                  \
        } else {                                                                \
            printf("FAIL\n");                                                  \
            g_test_fail_count++;                                                \
        }                                                                       \
    } while (0)

#define TEST_SUMMARY()                                                         \
    do {                                                                        \
        printf("\n%d assertions, %d failures\n", g_test_pass_count,            \
               g_test_fail_count);                                             \
        return g_test_fail_count > 0 ? 1 : 0;                                 \
    } while (0)

static inline void init_tensor_record(Spkv2TensorRecord *rec, uint32_t id,
                                       uint16_t dtype, uint16_t role,
                                       uint16_t rank, const uint32_t shape[],
                                       uint64_t size_bytes) {
    memset(rec, 0, sizeof(*rec));
    rec->id = id;
    rec->dtype = dtype;
    rec->role = role;
    rec->rank = rank;
    for (uint16_t i = 0; i < rank && i < 8; i++)
        rec->shape[i] = shape[i];
    rec->size_bytes = size_bytes;
}

static inline void init_attr_record(Spkv2AttrRecord *attr, uint16_t op_type) {
    memset(attr, 0, sizeof(*attr));
    attr->op_type = op_type;
}

static inline void init_node_record(Spkv2NodeRecord *node, uint32_t id,
                                     uint16_t op_type, uint16_t input_count,
                                     const uint32_t inputs[],
                                     uint16_t output_count,
                                     const uint32_t outputs[]) {
    memset(node, 0, sizeof(*node));
    node->id = id;
    node->op_type = op_type;
    node->input_count = input_count;
    node->output_count = output_count;
    for (uint16_t i = 0; i < input_count && i < 8; i++)
        node->inputs[i] = inputs[i];
    for (uint16_t i = 0; i < output_count && i < 4; i++)
        node->outputs[i] = outputs[i];
}

typedef struct {
    Spkv2Context ctx;
    Spkv2TensorState tensor_states[16];
    Spkv2TensorRecord tensor_records[16];
} TestContext;

static inline void init_test_context(TestContext *tc, int num_tensors,
                                      Spkv2AttrRecord *attrs,
                                      size_t attrs_size) {
    memset(tc, 0, sizeof(*tc));
    tc->ctx.tensors = tc->tensor_states;
    tc->ctx.attrs = (const uint8_t *)attrs;
    tc->ctx.attrs_size = attrs_size;
    for (int i = 0; i < num_tensors; i++) {
        tc->tensor_states[i].record = &tc->tensor_records[i];
    }
}

static inline void set_tensor_data(TestContext *tc, int index, void *data) {
    tc->tensor_states[index].data = (uint8_t *)data;
}

#endif /* SPKV2_TEST_HELPERS_H */
