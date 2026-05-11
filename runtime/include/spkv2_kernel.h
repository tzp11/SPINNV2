#ifndef SPKV2_KERNEL_H
#define SPKV2_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Spkv2Tensor Spkv2Tensor;

typedef int (*Spkv2KernelFn)(
    const Spkv2Tensor *inputs,
    int input_count,
    Spkv2Tensor *outputs,
    int output_count,
    const void *attrs,
    void *scratch);

typedef struct {
    uint16_t op_type;
    uint16_t backend;
    uint16_t dtype;
    uint16_t layout;
    uint16_t flags;
    int priority;
    Spkv2KernelFn fn;
} Spkv2KernelReg;

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_KERNEL_H */

