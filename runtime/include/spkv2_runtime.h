#ifndef SPKV2_RUNTIME_H
#define SPKV2_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Spkv2Context Spkv2Context;

const char *spkv2_runtime_version(void);

int spkv2_load_file(const char *path, Spkv2Context **out_ctx);
int spkv2_load_memory(const void *data, size_t size, Spkv2Context **out_ctx);
int spkv2_prepare(Spkv2Context *ctx, void *arena, size_t arena_size);
int spkv2_prepare_with_scratch(
    Spkv2Context *ctx,
    void *arena,
    size_t arena_size,
    void *scratch,
    size_t scratch_size);
int spkv2_set_input(Spkv2Context *ctx, int index, const void *data, size_t size);
int spkv2_bind_input(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_bind_output(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_get_output_size(Spkv2Context *ctx, int index, size_t *out_size);
int spkv2_get_output(Spkv2Context *ctx, int index, void *data, size_t size);
int spkv2_run(Spkv2Context *ctx);
void spkv2_profile_dump(void);
void spkv2_free(Spkv2Context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* SPKV2_RUNTIME_H */
