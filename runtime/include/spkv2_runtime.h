#ifndef SPKV2_RUNTIME_H
#define SPKV2_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Spkv2Context Spkv2Context;

const char *spkv2_runtime_version(void);

int spkv2_load_memory(const void *data, size_t size, Spkv2Context **out_ctx);
int spkv2_prepare(Spkv2Context *ctx, void *arena, size_t arena_size);
int spkv2_run(Spkv2Context *ctx);
void spkv2_free(Spkv2Context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_RUNTIME_H */

