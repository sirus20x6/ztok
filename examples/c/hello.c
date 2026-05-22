/* Minimal ztok C consumer.
 *
 * Builds against an installed ztok via either CMake (`find_package(ztok)`)
 * or pkg-config (`pkg-config --cflags --libs ztok`). See README.md.
 *
 * Loads a .tiktoken vocab, encodes one line, prints the resulting ids,
 * decodes them back, prints the round-trip string, and tears down.
 *
 * Usage:
 *   hello <path/to/cl100k_base.tiktoken> "text to encode"
 *
 * Exit code: 0 on success, non-zero on any error.
 */

#include <ztok.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* status_str(ztok_status s) {
    switch (s) {
        case ZTOK_OK:                    return "ok";
        case ZTOK_ERR_OUT_OF_MEMORY:     return "out of memory";
        case ZTOK_ERR_INVALID_INPUT:     return "invalid input";
        case ZTOK_ERR_BUFFER_TOO_SMALL:  return "buffer too small";
        case ZTOK_ERR_INTERNAL:          return "internal error";
        default:                         return "unknown";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <cl100k_base.tiktoken> <text>\n", argv[0]);
        return 2;
    }
    const char* vocab_path = argv[1];
    const char* text       = argv[2];
    size_t text_len        = strlen(text);

    printf("ztok %s\n", ztok_version());

    ztok_status st = ZTOK_OK;
    ztok_pipeline* p = ztok_pipeline_new_bpe_from_tiktoken(vocab_path, NULL, &st);
    if (!p) {
        fprintf(stderr, "load failed: %s\n", status_str(st));
        return 1;
    }

    /* ztok_encode's per-span worst-case bookkeeping needs the output
     * buffer to fit `sum(maxTokensFor(span_len))` ids, not just the
     * actual id count. The probe path (`out=NULL`) returns the exact
     * count via `*out_len + ZTOK_ERR_BUFFER_TOO_SMALL`, but allocating
     * exactly that risks the per-span early-exit re-tripping. Bound by
     * `input_len` (one id per input byte is the absolute ceiling) and
     * call once.
     */
    size_t cap = text_len + 16;
    ztok_token_id* ids = (ztok_token_id*)malloc(cap * sizeof(*ids));
    if (!ids) {
        fprintf(stderr, "malloc failed\n");
        ztok_pipeline_free(p);
        return 1;
    }
    size_t n = 0;
    st = ztok_encode(p, text, text_len, ids, cap, &n);
    if (st != ZTOK_OK) {
        fprintf(stderr, "encode failed: %s\n", status_str(st));
        free(ids);
        ztok_pipeline_free(p);
        return 1;
    }

    printf("ids (%zu):", n);
    for (size_t i = 0; i < n; i++) printf(" %u", ids[i]);
    printf("\n");

    /* Round-trip decode. text_len * 4 covers any reasonable expansion. */
    size_t dec_cap = text_len * 4 + 16;
    char* round = (char*)malloc(dec_cap);
    size_t r_len = 0;
    st = ztok_decode(p, ids, n, round, dec_cap, &r_len);
    if (st != ZTOK_OK) {
        fprintf(stderr, "decode failed: %s\n", status_str(st));
        free(round); free(ids); ztok_pipeline_free(p);
        return 1;
    }
    printf("round-trip: %.*s\n", (int)r_len, round);

    free(round);
    free(ids);
    ztok_pipeline_free(p);
    return 0;
}
