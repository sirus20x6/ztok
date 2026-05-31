/* Self-contained ztok consumer smoke test.
 *
 * Deliberately needs NO vocab file or other runtime input: it only
 * exercises that <ztok.h> is on the include path and that libztok links
 * and loads. We call ztok_version() (always present, no setup) and also
 * construct + immediately free a trivial byte-id pipeline so the test
 * touches a real allocating entry point, not just a string constant the
 * linker might satisfy from a stale symbol.
 *
 * Built by tests/consumer/run.sh once via CMake find_package(ztok) and
 * once via pkg-config, against both an absolute and a relative install
 * prefix. The relative-prefix run is the regression guard for the
 * baked-in-relative-path bug.
 *
 * Exit code: 0 on success, non-zero on any error.
 */

#include <ztok.h>

#include <stdio.h>
#include <string.h>

int main(void) {
    const char* v = ztok_version();
    if (v == NULL || v[0] == '\0') {
        fprintf(stderr, "ztok_version() returned empty\n");
        return 1;
    }
    printf("ztok consumer ok: version %s\n", v);

    /* Touch an allocating C ABI entry point so we exercise more than a
     * single exported string. A byte-id pipeline needs no vocab file. */
    ztok_pipeline_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.normalizer = ZTOK_NORMALIZER_IDENTITY;
    cfg.pre_tokenizer = ZTOK_PRETOK_IDENTITY;
    cfg.model = ZTOK_MODEL_BYTE_ID;
    cfg.decoder = ZTOK_DECODER_CONCAT;

    ztok_status st = ZTOK_OK;
    ztok_pipeline* p = ztok_pipeline_new(&cfg, &st);
    if (p == NULL || st != ZTOK_OK) {
        fprintf(stderr, "ztok_pipeline_new failed: status=%d\n", (int)st);
        return 1;
    }
    ztok_pipeline_free(p);

    return 0;
}
