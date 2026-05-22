/* bench_c_api.c — measure C ABI overhead of ztok vs in-process Zig.
 *
 * Three scenarios:
 *   single_small:  100K calls to ztok_encode with 50-byte inputs.
 *   single_large:  1 call to ztok_encode on a 10 MB corpus.
 *   batch_pooled:  1 call to ztok_encode_batch_pooled with 10K x 1 KB inputs.
 *
 * Usage:
 *   bench_c_api --model PATH_TO_TIKTOKEN [--corpus PATH] [--scenario NAME]
 *
 * If --corpus is missing or smaller than what a scenario needs, the bench
 * generates synthetic ASCII text so the harness is always runnable.
 */

#define _POSIX_C_SOURCE 200809L

#include <ztok.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t nanos_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    *out_len = n;
    return buf;
}

static void fill_pseudo(char *buf, size_t len, uint32_t seed) {
    /* Deterministic mixed-printable ASCII; cheap LCG. Keeps cl100k from
     * collapsing the whole thing into one mega-token. */
    static const char alpha[] = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ "
                                "0123456789 .,;:!? \n\n";
    uint32_t s = seed ? seed : 1;
    for (size_t i = 0; i < len; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = alpha[(s >> 16) % (sizeof(alpha) - 1)];
    }
}

static void run_single_small(ztok_pipeline *p, uint32_t iters) {
    /* 50-byte input, repeated `iters` times. Reuse a single output buffer
     * since callers in hot loops usually do. */
    char input[50];
    fill_pseudo(input, sizeof(input), 0xC0FFEE);

    ztok_token_id out[256];
    size_t out_len = 0;
    uint64_t total_ids = 0;

    uint64_t t0 = nanos_now();
    for (uint32_t k = 0; k < iters; k++) {
        ztok_status st = ztok_encode(p, input, sizeof(input), out, sizeof(out)/sizeof(out[0]), &out_len);
        if (st != ZTOK_OK) {
            fprintf(stderr, "single_small: encode failed st=%d\n", (int)st);
            exit(2);
        }
        total_ids += out_len;
    }
    uint64_t dt = nanos_now() - t0;

    double secs = (double)dt / 1e9;
    double per_call_us = (double)dt / 1000.0 / (double)iters;
    printf("single_small  iters=%u  time=%.3f ms  per-call=%.3f us  ids/call=%.1f  ns/op=%.1f\n",
           iters,
           (double)dt / 1e6,
           per_call_us,
           (double)total_ids / (double)iters,
           (double)dt / (double)iters);
    (void)secs;
}

static void run_single_large(ztok_pipeline *p, const char *corpus, size_t n) {
    ztok_token_id *out = (ztok_token_id *)malloc(sizeof(ztok_token_id) * n);
    if (!out) { fprintf(stderr, "OOM allocating output buffer\n"); exit(2); }
    size_t out_len = 0;

    uint64_t t0 = nanos_now();
    ztok_status st = ztok_encode(p, corpus, n, out, n, &out_len);
    uint64_t dt = nanos_now() - t0;
    if (st != ZTOK_OK) {
        fprintf(stderr, "single_large: encode failed st=%d (out_len=%zu)\n", (int)st, out_len);
        free(out);
        exit(2);
    }
    double mb_per_sec = ((double)n / (double)dt) * 1e3;  /* bytes/ns * 1e9 / 1e6 == bytes/ns * 1e3 */
    printf("single_large  bytes=%zu  time=%.2f ms  MB/s=%.1f  ids=%zu\n",
           n,
           (double)dt / 1e6,
           mb_per_sec,
           out_len);
    free(out);
}

static void run_batch_pooled(ztok_pipeline *p, ztok_batch_pool *pool,
                             const char *corpus, size_t corpus_len,
                             size_t n_inputs, size_t per_len)
{
    if (corpus_len < n_inputs * per_len) {
        fprintf(stderr, "batch_pooled: corpus too small (%zu < %zu)\n",
                corpus_len, n_inputs * per_len);
        exit(2);
    }
    const char **inputs = (const char **)malloc(sizeof(char *) * n_inputs);
    size_t *lens = (size_t *)malloc(sizeof(size_t) * n_inputs);
    ztok_token_id **out_ids = (ztok_token_id **)calloc(n_inputs, sizeof(ztok_token_id *));
    size_t *out_lens = (size_t *)calloc(n_inputs, sizeof(size_t));
    if (!inputs || !lens || !out_ids || !out_lens) {
        fprintf(stderr, "OOM allocating batch arrays\n");
        exit(2);
    }
    for (size_t i = 0; i < n_inputs; i++) {
        inputs[i] = corpus + i * per_len;
        lens[i] = per_len;
    }

    uint64_t t0 = nanos_now();
    ztok_status st = ztok_encode_batch_pooled(p, pool, inputs, lens, n_inputs, out_ids, out_lens);
    uint64_t dt = nanos_now() - t0;
    if (st != ZTOK_OK) {
        fprintf(stderr, "batch_pooled: encode failed st=%d\n", (int)st);
        exit(2);
    }

    uint64_t total_bytes = (uint64_t)n_inputs * (uint64_t)per_len;
    uint64_t total_ids = 0;
    for (size_t i = 0; i < n_inputs; i++) total_ids += out_lens[i];

    double mb_per_sec = ((double)total_bytes / (double)dt) * 1e3;
    printf("batch_pooled  n=%zu  per=%zu  time=%.2f ms  MB/s=%.1f  ids=%llu\n",
           n_inputs, per_len,
           (double)dt / 1e6,
           mb_per_sec,
           (unsigned long long)total_ids);

    for (size_t i = 0; i < n_inputs; i++) ztok_ids_free(out_ids[i]);
    free(out_ids); free(out_lens); free(inputs); free(lens);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *corpus_path = NULL;
    const char *scenario = "all";
    uint32_t small_iters = 100000;
    size_t large_bytes = 10 * 1024 * 1024;
    size_t batch_n = 10000;
    size_t batch_per = 1024;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--scenario") && i + 1 < argc) scenario = argv[++i];
        else if (!strcmp(argv[i], "--small-iters") && i + 1 < argc) small_iters = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--large-bytes") && i + 1 < argc) large_bytes = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--batch-n") && i + 1 < argc) batch_n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--batch-per") && i + 1 < argc) batch_per = (size_t)atol(argv[++i]);
    }
    if (!model_path) {
        fprintf(stderr, "usage: %s --model PATH [--corpus PATH] [--scenario all|single_small|single_large|batch_pooled] "
                "[--small-iters N] [--large-bytes N] [--batch-n N] [--batch-per N]\n", argv[0]);
        return 2;
    }

    /* Load/synthesize corpus. */
    char *corpus = NULL;
    size_t corpus_len = 0;
    if (corpus_path) {
        corpus = read_file(corpus_path, &corpus_len);
        if (!corpus) { fprintf(stderr, "failed to read %s\n", corpus_path); return 2; }
    }
    size_t need = large_bytes;
    if (batch_n * batch_per > need) need = batch_n * batch_per;
    if (corpus_len < need) {
        size_t old = corpus_len;
        corpus = (char *)realloc(corpus, need);
        if (!corpus) { fprintf(stderr, "OOM growing corpus\n"); return 2; }
        if (old < need) fill_pseudo(corpus + old, need - old, 0xDECAFBADu);
        corpus_len = need;
    }

    ztok_status st = ZTOK_OK;
    ztok_pipeline *p = ztok_pipeline_new_bpe_from_tiktoken(model_path, NULL, &st);
    if (!p) { fprintf(stderr, "pipeline load failed st=%d\n", (int)st); free(corpus); return 2; }

    ztok_batch_pool *pool = ztok_batch_pool_new(0, &st);
    if (!pool) { fprintf(stderr, "pool create failed st=%d\n", (int)st); ztok_pipeline_free(p); free(corpus); return 2; }

    printf("ztok %s  workers=%zu  corpus=%zu\n",
           ztok_version(),
           ztok_batch_pool_worker_count(pool),
           corpus_len);

    int do_all = !strcmp(scenario, "all");
    if (do_all || !strcmp(scenario, "single_small"))
        run_single_small(p, small_iters);
    if (do_all || !strcmp(scenario, "single_large"))
        run_single_large(p, corpus, large_bytes);
    if (do_all || !strcmp(scenario, "batch_pooled"))
        run_batch_pooled(p, pool, corpus, corpus_len, batch_n, batch_per);

    ztok_batch_pool_free(pool);
    ztok_pipeline_free(p);
    free(corpus);
    return 0;
}
