#include "spkv2_platform.h"

#include <stdlib.h>

void *spkv2_platform_malloc(size_t size) {
    return malloc(size);
}

void *spkv2_platform_calloc(size_t count, size_t size) {
    return calloc(count, size);
}

void spkv2_platform_free(void *ptr) {
    free(ptr);
}
