#ifndef SPKV2_PLATFORM_H
#define SPKV2_PLATFORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *spkv2_platform_malloc(size_t size);
void *spkv2_platform_calloc(size_t count, size_t size);
void spkv2_platform_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_PLATFORM_H */
