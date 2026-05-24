package ztok

// Engram n-gram hashing — deterministic multi-head token-n-gram hashes
// for conditional-memory addressing (see src/ngram.zig). These operate
// on raw token ids and need no Pipeline; the output is row-major
// [position][head] raw uint64 hashes which the caller masks to its own
// table width.

/*
#include <ztok.h>
#include <stdlib.h>

// Parallel-array shims for the batch path. id_arrays must point at
// C-owned memory (cgo forbids storing Go pointers inside a C array that
// is then handed to C), so HashNGramsBatch copies each id stream into a
// malloc'd buffer below.
static const ztok_token_id** ztok_go_ng_alloc_idarr(size_t n) {
    return (const ztok_token_id**)calloc(n, sizeof(ztok_token_id*));
}
static void ztok_go_ng_free_idarr(const ztok_token_id** a) { free((void*)a); }
static void ztok_go_ng_set_idarr(const ztok_token_id** a, size_t i, ztok_token_id* p) { a[i] = p; }

static ztok_token_id* ztok_go_ng_alloc_ids(size_t n) {
    return (ztok_token_id*)malloc(n * sizeof(ztok_token_id));
}
static void ztok_go_ng_free_ids(ztok_token_id* a) { free(a); }
static void ztok_go_ng_set_id(ztok_token_id* a, size_t i, ztok_token_id v) { a[i] = v; }

static size_t* ztok_go_ng_alloc_sz(size_t n) { return (size_t*)calloc(n, sizeof(size_t)); }
static void ztok_go_ng_free_sz(size_t* a) { free(a); }
static void ztok_go_ng_set_sz(size_t* a, size_t i, size_t v) { a[i] = v; }
static size_t ztok_go_ng_get_sz(const size_t* a, size_t i) { return a[i]; }

static uint64_t** ztok_go_ng_alloc_hasharr(size_t n) {
    return (uint64_t**)calloc(n, sizeof(uint64_t*));
}
static void ztok_go_ng_free_hasharr(uint64_t** a) { free((void*)a); }
static uint64_t* ztok_go_ng_get_hash(uint64_t** a, size_t i) { return a[i]; }
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// HashNGrams hashes every length-`n` window of `ids` under `heads`
// independent hash functions, returning the row-major [position][head]
// uint64 hashes (positions = len(ids)-n+1, or 0 if the stream is shorter
// than one window). Mask each hash to your table width
// (hash & ((1<<bits)-1)). Deterministic: the same ids always yield the
// same hashes. Returns nil when there is nothing to hash.
func HashNGrams(ids []uint32, n, heads uint32) ([]uint64, error) {
	if len(ids) == 0 || n == 0 || heads == 0 || len(ids) < int(n) {
		return nil, nil
	}
	positions := len(ids) - int(n) + 1
	want := positions * int(heads)
	if want == 0 {
		return nil, nil
	}

	out := make([]uint64, want)
	var outLen C.size_t
	rc := C.ztok_ngram_hash(
		(*C.ztok_token_id)(unsafe.Pointer(&ids[0])), C.size_t(len(ids)),
		C.uint32_t(n), C.uint32_t(heads),
		(*C.uint64_t)(unsafe.Pointer(&out[0])), C.size_t(want), &outLen,
	)
	runtime.KeepAlive(ids)

	switch int(rc) {
	case int(cStatusOK):
		return out[:int(outLen)], nil
	case int(cStatusBufferTooSmall):
		// Should not happen — we size exactly — but honor the contract.
		grown := make([]uint64, int(outLen))
		if int(outLen) == 0 {
			return nil, nil
		}
		rc2 := C.ztok_ngram_hash(
			(*C.ztok_token_id)(unsafe.Pointer(&ids[0])), C.size_t(len(ids)),
			C.uint32_t(n), C.uint32_t(heads),
			(*C.uint64_t)(unsafe.Pointer(&grown[0])), outLen, &outLen,
		)
		runtime.KeepAlive(ids)
		if int(rc2) != int(cStatusOK) {
			return nil, statusToError(int(rc2), "ztok_ngram_hash")
		}
		return grown[:int(outLen)], nil
	default:
		return nil, statusToError(int(rc), "ztok_ngram_hash")
	}
}

// HashNGramsBatch hashes many id streams in parallel across `pool`.
// results[i] holds the row-major hashes for streams[i] (nil for a stream
// shorter than one window). Equivalent to calling HashNGrams on each
// stream, but fanned out across the pool's workers.
func HashNGramsBatch(pool *BatchPool, streams [][]uint32, n, heads uint32) ([][]uint64, error) {
	if pool == nil || pool.closed || pool.handle == nil {
		return nil, ErrClosed
	}
	nDocs := len(streams)
	if nDocs == 0 {
		return nil, nil
	}

	cIDArr := C.ztok_go_ng_alloc_idarr(C.size_t(nDocs))
	defer C.ztok_go_ng_free_idarr(cIDArr)
	cLens := C.ztok_go_ng_alloc_sz(C.size_t(nDocs))
	defer C.ztok_go_ng_free_sz(cLens)
	cOutHashes := C.ztok_go_ng_alloc_hasharr(C.size_t(nDocs))
	defer C.ztok_go_ng_free_hasharr(cOutHashes)
	cOutLens := C.ztok_go_ng_alloc_sz(C.size_t(nDocs))
	defer C.ztok_go_ng_free_sz(cOutLens)

	// Copy each id stream into C memory (cgo forbids handing C an array
	// of Go pointers). Tracked so the deferred free runs after the call.
	idBufs := make([]*C.ztok_token_id, nDocs)
	defer func() {
		for _, b := range idBufs {
			if b != nil {
				C.ztok_go_ng_free_ids(b)
			}
		}
	}()
	for i, s := range streams {
		C.ztok_go_ng_set_sz(cLens, C.size_t(i), C.size_t(len(s)))
		if len(s) == 0 {
			continue
		}
		buf := C.ztok_go_ng_alloc_ids(C.size_t(len(s)))
		idBufs[i] = buf
		for j, v := range s {
			C.ztok_go_ng_set_id(buf, C.size_t(j), C.ztok_token_id(v))
		}
		C.ztok_go_ng_set_idarr(cIDArr, C.size_t(i), buf)
	}

	rc := C.ztok_ngram_hash_batch(
		(*C.ztok_batch_pool)(pool.handle),
		cIDArr, cLens, C.size_t(nDocs),
		C.uint32_t(n), C.uint32_t(heads),
		cOutHashes, cOutLens,
	)

	// Always materialize + free, even on error, to avoid leaks on a
	// partially-populated batch.
	results := make([][]uint64, nDocs)
	for i := 0; i < nDocs; i++ {
		ptr := C.ztok_go_ng_get_hash(cOutHashes, C.size_t(i))
		length := int(C.ztok_go_ng_get_sz(cOutLens, C.size_t(i)))
		results[i] = materializeHashes(unsafe.Pointer(ptr), length)
	}

	if err := statusToError(int(rc), "ztok_ngram_hash_batch"); err != nil {
		return nil, err
	}
	return results, nil
}

// materializeHashes copies a C-owned uint64 buffer into a Go []uint64,
// then frees it via ztok_u64s_free (the only safe path — the buffer
// carries a length-prefix header, see src/c_api.zig::allocU64Buf).
func materializeHashes(ptr unsafe.Pointer, n int) []uint64 {
	if ptr == nil {
		return nil
	}
	if n <= 0 {
		C.ztok_u64s_free((*C.uint64_t)(ptr))
		return nil
	}
	view := unsafe.Slice((*uint64)(ptr), n)
	out := make([]uint64, n)
	copy(out, view)
	C.ztok_u64s_free((*C.uint64_t)(ptr))
	return out
}
