#include "test_helpers.h"
#include "reference_kernels.h"

static int test_transpose_nchw_to_nhwc(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_TRANSPOSE);
    attr.extra_count = 4;
    attr.extra[0] = 0;
    attr.extra[1] = 2;
    attr.extra[2] = 3;
    attr.extra[3] = 1;
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t x_shape[] = {1, 2, 3, 4};
    uint32_t y_shape[] = {1, 3, 4, 2};
    size_t n = 1 * 2 * 3 * 4;
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, x_shape, n * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, y_shape, n * sizeof(float));

    float x[24], y[24];
    for (int i = 0; i < 24; i++) x[i] = (float)i;
    memset(y, 0, sizeof(y));
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_TRANSPOSE, 1, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_transpose(&tc.ctx, &node, NULL));

    for (int n_i = 0; n_i < 1; n_i++) {
        for (int c = 0; c < 2; c++) {
            for (int h = 0; h < 3; h++) {
                for (int w = 0; w < 4; w++) {
                    int x_idx = n_i * 2 * 3 * 4 + c * 3 * 4 + h * 4 + w;
                    int y_idx = n_i * 3 * 4 * 2 + h * 4 * 2 + w * 2 + c;
                    ASSERT_NEAR(y[y_idx], x[x_idx], 1e-7f);
                }
            }
        }
    }
    return 0;
}

static int test_transpose_swap_middle(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_TRANSPOSE);
    attr.extra_count = 4;
    attr.extra[0] = 0;
    attr.extra[1] = 2;
    attr.extra[2] = 1;
    attr.extra[3] = 3;
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t x_shape[] = {2, 3, 4, 5};
    uint32_t y_shape[] = {2, 4, 3, 5};
    size_t n = 2 * 3 * 4 * 5;
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, x_shape, n * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, y_shape, n * sizeof(float));

    float x[120], y[120];
    for (int i = 0; i < 120; i++) x[i] = (float)i * 0.1f;
    memset(y, 0, sizeof(y));
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_TRANSPOSE, 1, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_transpose(&tc.ctx, &node, NULL));

    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 5; k++) {
                    int x_idx = b * 3 * 4 * 5 + i * 4 * 5 + j * 5 + k;
                    int y_idx = b * 4 * 3 * 5 + j * 3 * 5 + i * 5 + k;
                    ASSERT_NEAR(y[y_idx], x[x_idx], 1e-6f);
                }
            }
        }
    }
    return 0;
}

int main(void) {
    printf("test_kernel_transpose:\n");
    RUN_TEST(test_transpose_nchw_to_nhwc);
    RUN_TEST(test_transpose_swap_middle);
    TEST_SUMMARY();
}
