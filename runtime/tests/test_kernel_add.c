#include "test_helpers.h"
#include "reference_kernels.h"

static int test_add_same_shape(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_ADD);
    init_test_context(&tc, 3, &attr, sizeof(attr));

    uint32_t shape[] = {1, 4, 2, 2};
    size_t n = 1 * 4 * 2 * 2;
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, n * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, n * sizeof(float));
    init_tensor_record(&tc.tensor_records[2], 2, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, n * sizeof(float));

    float a[16], b[16], y[16];
    for (int i = 0; i < 16; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 10);
    }
    memset(y, 0, sizeof(y));
    set_tensor_data(&tc, 0, a);
    set_tensor_data(&tc, 1, b);
    set_tensor_data(&tc, 2, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0, 1};
    uint32_t outputs[] = {2};
    init_node_record(&node, 0, SPKV2_OP_ADD, 2, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_add(&tc.ctx, &node, NULL));

    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR(y[i], a[i] + b[i], 1e-6f);
    }
    return 0;
}

static int test_add_broadcast(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_ADD);
    init_test_context(&tc, 3, &attr, sizeof(attr));

    uint32_t shape_a[] = {1, 4, 2, 2};
    uint32_t shape_b[] = {1, 4, 1, 1};
    uint32_t shape_y[] = {1, 4, 2, 2};
    size_t na = 16, nb = 4, ny = 16;
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape_a, na * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape_b, nb * sizeof(float));
    init_tensor_record(&tc.tensor_records[2], 2, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape_y, ny * sizeof(float));

    float a[16], b[4], y[16];
    for (int i = 0; i < 16; i++) a[i] = (float)i;
    for (int i = 0; i < 4; i++) b[i] = 100.0f * (float)(i + 1);
    memset(y, 0, sizeof(y));
    set_tensor_data(&tc, 0, a);
    set_tensor_data(&tc, 1, b);
    set_tensor_data(&tc, 2, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0, 1};
    uint32_t outputs[] = {2};
    init_node_record(&node, 0, SPKV2_OP_ADD, 2, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_add(&tc.ctx, &node, NULL));

    for (int c = 0; c < 4; c++) {
        for (int hw = 0; hw < 4; hw++) {
            int idx = c * 4 + hw;
            ASSERT_NEAR(y[idx], a[idx] + b[c], 1e-6f);
        }
    }
    return 0;
}

int main(void) {
    printf("test_kernel_add:\n");
    RUN_TEST(test_add_same_shape);
    RUN_TEST(test_add_broadcast);
    TEST_SUMMARY();
}
