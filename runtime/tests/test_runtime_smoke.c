#include "spkv2_format.h"
#include "spkv2_runtime.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    const char *version = spkv2_runtime_version();
    if (version == NULL || strcmp(version, "0.0.0-m0") != 0) {
        fprintf(stderr, "unexpected runtime version\n");
        return 1;
    }

    Spkv2Header header;
    memset(&header, 0, sizeof(header));
    header.magic = SPKV2_MAGIC;
    header.version_major = SPKV2_VERSION_MAJOR;
    header.version_minor = SPKV2_VERSION_MINOR;

    if (header.magic != SPKV2_MAGIC) {
        fprintf(stderr, "unexpected SPK magic\n");
        return 1;
    }

    return 0;
}

