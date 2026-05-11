#include "spkv2_runtime.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned char *read_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    unsigned char *data = (unsigned char *)malloc((size_t)size);
    if (!data) {
        fclose(fp);
        return NULL;
    }
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *out_size = (size_t)size;
    return data;
}

static int write_file(const char *path, const unsigned char *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    int ok = fwrite(data, 1, size, fp) == size;
    fclose(fp);
    return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s model.spk input.bin output.bin\n", argv[0]);
        return 2;
    }

    Spkv2Context *ctx = NULL;
    if (spkv2_load_file(argv[1], &ctx) != 0) {
        fprintf(stderr, "failed to load SPK\n");
        return 1;
    }
    if (spkv2_prepare(ctx, NULL, 0) != 0) {
        fprintf(stderr, "failed to prepare runtime\n");
        spkv2_free(ctx);
        return 1;
    }

    size_t input_size = 0;
    unsigned char *input = read_file(argv[2], &input_size);
    if (!input) {
        fprintf(stderr, "failed to read input\n");
        spkv2_free(ctx);
        return 1;
    }
    if (spkv2_set_input(ctx, 0, input, input_size) != 0) {
        fprintf(stderr, "failed to set input\n");
        free(input);
        spkv2_free(ctx);
        return 1;
    }
    free(input);

    if (spkv2_run(ctx) != 0) {
        fprintf(stderr, "failed to run model\n");
        spkv2_free(ctx);
        return 1;
    }

    size_t output_size = 0;
    if (spkv2_get_output_size(ctx, 0, &output_size) != 0) {
        fprintf(stderr, "failed to get output size\n");
        spkv2_free(ctx);
        return 1;
    }
    unsigned char *output = (unsigned char *)malloc(output_size);
    if (!output) {
        spkv2_free(ctx);
        return 1;
    }
    if (spkv2_get_output(ctx, 0, output, output_size) != 0) {
        fprintf(stderr, "failed to get output\n");
        free(output);
        spkv2_free(ctx);
        return 1;
    }

    if (write_file(argv[3], output, output_size) != 0) {
        fprintf(stderr, "failed to write output\n");
        free(output);
        spkv2_free(ctx);
        return 1;
    }

    free(output);
    spkv2_free(ctx);
    return 0;
}
