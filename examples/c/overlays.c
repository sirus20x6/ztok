/* ztok overlay-channel C consumer.
 *
 * Demonstrates ztok_encode_with_overlays: encode one string and pull
 * back per-token annotation channels aligned 1:1 with the id stream.
 * Requests the four cheap channels (BYTE_START, BYTE_END, BOUNDARY,
 * PROVENANCE) and prints a table.
 *
 * Builds against an installed ztok via CMake (`find_package(ztok)`) or
 * pkg-config (`pkg-config --cflags --libs ztok`). See README.md.
 *
 * Usage:
 *   overlays <path/to/cl100k_base.tiktoken> "text to encode"
 *
 * Exit code: 0 on success, non-zero on any error.
 */

#include <ztok.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

/* BOUNDARY is a bitset: 0x1 = starts a pre-tok chunk, 0x2 = starts a
 * UTF-8 codepoint. Render as a compact two-letter tag. */
static void boundary_tag(uint32_t b, char out[3]) {
    out[0] = (b & 0x1) ? 'C' : '-';   /* chunk-start    */
    out[1] = (b & 0x2) ? 'U' : '-';   /* codepoint-start */
    out[2] = '\0';
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

    /* The channels we want, in the order their values come back. */
    enum { NCH = 4 };
    ztok_overlay_channel ch[NCH] = {
        { ZTOK_OVERLAY_BYTE_START, NULL, 0 },
        { ZTOK_OVERLAY_BYTE_END,   NULL, 0 },
        { ZTOK_OVERLAY_BOUNDARY,   NULL, 0 },
        { ZTOK_OVERLAY_PROVENANCE, NULL, 0 },
    };

    /* Pass 1 (probe): out_ids = NULL returns the token count via
     * *out_len + ZTOK_ERR_BUFFER_TOO_SMALL. Channel buffers are ignored
     * on the probe, so leaving them NULL is fine. */
    size_t n = 0;
    st = ztok_encode_with_overlays(p, text, text_len, NULL, 0, ch, NCH, &n);
    if (st != ZTOK_ERR_BUFFER_TOO_SMALL) {
        fprintf(stderr, "probe failed: %s\n", status_str(st));
        ztok_pipeline_free(p);
        return 1;
    }

    /* Allocate ids + one uint32 buffer per channel, all length n. */
    ztok_token_id* ids = (ztok_token_id*)malloc(n * sizeof(*ids));
    uint32_t* chbuf[NCH];
    int ok = ids != NULL;
    for (int i = 0; i < NCH; i++) {
        chbuf[i] = (uint32_t*)malloc(n * sizeof(uint32_t));
        ch[i].out = chbuf[i];
        ch[i].out_cap = n;
        ok = ok && (chbuf[i] != NULL);
    }
    if (!ok) {
        fprintf(stderr, "malloc failed\n");
        free(ids);
        for (int i = 0; i < NCH; i++) free(chbuf[i]);
        ztok_pipeline_free(p);
        return 1;
    }

    /* Pass 2 (fill): every listed channel's buffer must hold >= n. */
    size_t n2 = 0;
    st = ztok_encode_with_overlays(p, text, text_len, ids, n, ch, NCH, &n2);
    if (st != ZTOK_OK) {
        fprintf(stderr, "encode failed: %s\n", status_str(st));
        free(ids);
        for (int i = 0; i < NCH; i++) free(chbuf[i]);
        ztok_pipeline_free(p);
        return 1;
    }

    const uint32_t* bstart = chbuf[0];
    const uint32_t* bend   = chbuf[1];
    const uint32_t* bound  = chbuf[2];
    const uint32_t* prov   = chbuf[3];

    printf("%-4s %-8s %-10s %-8s %-10s %s\n",
           "#", "id", "bytes", "bound", "prov", "text");
    for (size_t i = 0; i < n2; i++) {
        char tag[3];
        boundary_tag(bound[i], tag);
        const char* prov_s = (prov[i] == 1) ? "special" : "text";
        int seg_len = (int)(bend[i] - bstart[i]);
        printf("%-4zu %-8u %3u..%-5u %-8s %-10s '%.*s'\n",
               i, ids[i], bstart[i], bend[i], tag, prov_s,
               seg_len, text + bstart[i]);
    }

    free(ids);
    for (int i = 0; i < NCH; i++) free(chbuf[i]);
    ztok_pipeline_free(p);
    return 0;
}
