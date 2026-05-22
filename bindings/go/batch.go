package ztok

/*
#include <ztok.h>
#include <stdlib.h>

// Tiny C shims for the parallel-array layout that
// ztok_encode_batch_pooled expects. cgo can't index `**char` /
// `void**` arrays directly, so we centralize the unsafe pointer
// arithmetic here. Each .go file is its own translation unit, so these
// must live alongside batch.go (where they're called).

static const char** ztok_go_alloc_str_arr(size_t n) {
    return (const char**)calloc(n, sizeof(const char*));
}
static void ztok_go_free_str_arr(const char** arr) {
    free((void*)arr);
}
static size_t* ztok_go_alloc_sz_arr(size_t n) {
    return (size_t*)calloc(n, sizeof(size_t));
}
static void ztok_go_free_sz_arr(size_t* arr) {
    free((void*)arr);
}
static void** ztok_go_alloc_voidp_arr(size_t n) {
    return (void**)calloc(n, sizeof(void*));
}
static void ztok_go_free_voidp_arr(void** arr) {
    free((void*)arr);
}
static void ztok_go_set_str(const char** arr, size_t i, const char* s) {
    arr[i] = s;
}
static void ztok_go_set_sz(size_t* arr, size_t i, size_t v) {
    arr[i] = v;
}
static void* ztok_go_get_ids_ptr(void** arr, size_t i) {
    return arr[i];
}
static size_t ztok_go_get_sz(const size_t* arr, size_t i) {
    return arr[i];
}
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// BatchPool is a persistent multithreaded worker pool. Create once,
// reuse across many EncodeBatch calls — each pool owns its own arenas
// and worker threads, so creating one per batch wastes setup work.
type BatchPool struct {
	handle unsafe.Pointer
	closed bool
}

// BatchPoolOptions configures NewBatchPool.
type BatchPoolOptions struct {
	// Workers is the number of worker threads. 0 = auto-detect cpu count.
	Workers uint32
}

// NewBatchPool builds a persistent worker pool. opts may be nil.
func NewBatchPool(opts *BatchPoolOptions) (*BatchPool, error) {
	workers := uint32(0)
	if opts != nil {
		workers = opts.Workers
	}
	var status C.ztok_status
	h := unsafe.Pointer(C.ztok_batch_pool_new(C.uint32_t(workers), &status))
	if err := statusToError(int(status), "ztok_batch_pool_new"); err != nil {
		return nil, err
	}
	if h == nil {
		return nil, &InternalError{&StatusError{Status: -1, Op: "ztok_batch_pool_new returned NULL"}}
	}
	p := &BatchPool{handle: h}
	runtime.SetFinalizer(p, func(p *BatchPool) {
		if p.handle != nil {
			C.ztok_batch_pool_free((*C.ztok_batch_pool)(p.handle))
			p.handle = nil
		}
	})
	return p, nil
}

// Workers returns the actual worker count (resolves the 0=auto request
// to the detected cpu count).
func (p *BatchPool) Workers() int {
	if p.closed || p.handle == nil {
		return 0
	}
	return int(C.ztok_batch_pool_worker_count((*C.ztok_batch_pool)(p.handle)))
}

// Close releases the pool's worker threads and arenas. Idempotent.
func (p *BatchPool) Close() error {
	if p.closed {
		return nil
	}
	p.closed = true
	runtime.SetFinalizer(p, nil)
	if p.handle != nil {
		C.ztok_batch_pool_free((*C.ztok_batch_pool)(p.handle))
		p.handle = nil
	}
	return nil
}

// EncodeBatch encodes many strings in parallel via a persistent
// BatchPool. Each per-input id slice is materialized into a Go []uint32
// after the call returns; the C-owned buffers are then freed through
// ztok_ids_free (the only safe path — they carry a length-prefix
// header, see src/c_api.zig::allocIdBuf).
func (p *Pipeline) EncodeBatch(pool *BatchPool, inputs []string) ([][]uint32, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	if pool == nil || pool.closed || pool.handle == nil {
		return nil, ErrClosed
	}
	n := len(inputs)
	if n == 0 {
		return nil, nil
	}

	// Layout: allocate parallel C arrays for inputs[], input_lens[],
	// out_ids[], out_lens[]. Each input is a separate CString that we
	// free after the call returns.
	cInputs := C.ztok_go_alloc_str_arr(C.size_t(n))
	defer C.ztok_go_free_str_arr(cInputs)
	cLens := C.ztok_go_alloc_sz_arr(C.size_t(n))
	defer C.ztok_go_free_sz_arr(cLens)
	cOutIDs := C.ztok_go_alloc_voidp_arr(C.size_t(n))
	defer C.ztok_go_free_voidp_arr(cOutIDs)
	cOutLens := C.ztok_go_alloc_sz_arr(C.size_t(n))
	defer C.ztok_go_free_sz_arr(cOutLens)

	// Per-input CStrings, tracked in a Go slice so the deferred free
	// runs in LIFO order before the array containers above.
	stringHandles := make([]*C.char, n)
	defer func() {
		for _, s := range stringHandles {
			if s != nil {
				C.free(unsafe.Pointer(s))
			}
		}
	}()
	for i, s := range inputs {
		cs := C.CString(s)
		stringHandles[i] = cs
		C.ztok_go_set_str(cInputs, C.size_t(i), cs)
		C.ztok_go_set_sz(cLens, C.size_t(i), C.size_t(len(s)))
	}

	rc := C.ztok_encode_batch_pooled(
		(*C.ztok_pipeline)(p.handle),
		(*C.ztok_batch_pool)(pool.handle),
		cInputs, cLens, C.size_t(n),
		(**C.ztok_token_id)(unsafe.Pointer(cOutIDs)),
		cOutLens,
	)

	// Always materialize + free, even on error, to avoid leaks on
	// partially-populated batches.
	results := make([][]uint32, n)
	for i := 0; i < n; i++ {
		ptr := unsafe.Pointer(C.ztok_go_get_ids_ptr(cOutIDs, C.size_t(i)))
		length := int(C.ztok_go_get_sz(cOutLens, C.size_t(i)))
		results[i] = materializeIDs(ptr, length)
	}

	if err := statusToError(int(rc), "ztok_encode_batch_pooled"); err != nil {
		return nil, err
	}
	return results, nil
}
