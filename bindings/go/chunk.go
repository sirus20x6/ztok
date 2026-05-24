package ztok

// Token-window chunking for embedding / late-chunking pipelines (see
// src/chunk.zig). Chunk splits text into overlapping windows of token
// ids, each carrying its byte- and token-index range in the original
// input so a downstream embedder's id stream and chunk boundaries agree.

/*
#include <ztok.h>
#include <stdlib.h>

// Allocate / index a C array of chunk records. The records are written
// by ztok_chunk; each record's `ids` is a ztok-allocated buffer freed by
// ztok_chunks_free. We read fields back through these accessors so the
// Go side never depends on the C struct's layout.
static ztok_chunk_rec* ztok_go_chunk_alloc(size_t n) {
    return (ztok_chunk_rec*)calloc(n, sizeof(ztok_chunk_rec));
}
static void ztok_go_chunk_free_arr(ztok_chunk_rec* a) { free(a); }
static ztok_token_id* ztok_go_chunk_ids(const ztok_chunk_rec* a, size_t i) { return a[i].ids; }
static size_t ztok_go_chunk_ids_len(const ztok_chunk_rec* a, size_t i) { return a[i].ids_len; }
static uint32_t ztok_go_chunk_byte_start(const ztok_chunk_rec* a, size_t i) { return a[i].byte_start; }
static uint32_t ztok_go_chunk_byte_end(const ztok_chunk_rec* a, size_t i) { return a[i].byte_end; }
static uint32_t ztok_go_chunk_token_start(const ztok_chunk_rec* a, size_t i) { return a[i].token_start; }
static uint32_t ztok_go_chunk_token_end(const ztok_chunk_rec* a, size_t i) { return a[i].token_end; }
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// ChunkBoundary selects where chunk edges are allowed to fall. Mirrors
// the C enum ztok_chunk_boundary / Zig chunk.Boundary.
type ChunkBoundary uint32

const (
	// BoundaryToken: pure token-count windows (default).
	BoundaryToken ChunkBoundary = C.ZTOK_CHUNK_BOUNDARY_TOKEN
	// BoundaryCodepoint: snap to a UTF-8 codepoint boundary.
	BoundaryCodepoint ChunkBoundary = C.ZTOK_CHUNK_BOUNDARY_CODEPOINT
	// BoundaryWord: snap to a whitespace word boundary.
	BoundaryWord ChunkBoundary = C.ZTOK_CHUNK_BOUNDARY_WORD
	// BoundaryWordDict: snap to a dictionary word boundary (CJK/Thai/...).
	BoundaryWordDict ChunkBoundary = C.ZTOK_CHUNK_BOUNDARY_WORD_DICT
	// BoundarySentence: snap to a sentence boundary.
	BoundarySentence ChunkBoundary = C.ZTOK_CHUNK_BOUNDARY_SENTENCE
	// BoundaryParagraph: snap to a paragraph break (\n\n).
	BoundaryParagraph ChunkBoundary = C.ZTOK_CHUNK_BOUNDARY_PARAGRAPH
)

// ChunkOptions configures Pipeline.Chunk. MaxTokens must be > 0 and
// Overlap must be < MaxTokens (stride = MaxTokens - Overlap). Boundary
// defaults to BoundaryToken (its zero value).
type ChunkOptions struct {
	MaxTokens uint32
	Overlap   uint32
	Boundary  ChunkBoundary
}

// Chunk is one token window with its position in the original input.
type Chunk struct {
	// IDs are the token ids in this chunk (copied out of C memory).
	IDs []uint32
	// ByteStart/ByteEnd is the half-open byte range this chunk covers in
	// the ORIGINAL input.
	ByteStart uint32
	ByteEnd   uint32
	// TokenStart/TokenEnd is the half-open token-index range in the full
	// encoding.
	TokenStart uint32
	TokenEnd   uint32
}

// Chunk splits `text` into overlapping token windows. Each returned Chunk
// holds its token ids plus byte/token ranges. Returns nil (no error) for
// empty input. The C-owned id buffers are materialized into Go slices and
// freed before returning, so the result is fully owned by Go.
func (p *Pipeline) Chunk(text string, opts ChunkOptions) ([]Chunk, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	if opts.MaxTokens == 0 || opts.Overlap >= opts.MaxTokens {
		return nil, &StatusError{
			Status: int(cStatusInvalidInput),
			Op:     "Chunk: MaxTokens must be > 0 and Overlap < MaxTokens",
		}
	}

	var cText *C.char
	if len(text) > 0 {
		cText = (*C.char)(unsafe.Pointer(unsafe.StringData(text)))
	}
	cTextLen := C.size_t(len(text))
	pipe := (*C.ztok_pipeline)(p.handle)

	// Sizing pass: no buffer -> *out_len = chunk count.
	var need C.size_t
	rc := C.ztok_chunk(
		pipe, cText, cTextLen,
		C.uint32_t(opts.MaxTokens), C.uint32_t(opts.Overlap), C.uint32_t(opts.Boundary),
		nil, 0, &need,
	)
	switch int(rc) {
	case int(cStatusOK):
		// Zero chunks (empty input). KeepAlive guards the borrowed bytes.
		runtime.KeepAlive(text)
		if int(need) == 0 {
			return nil, nil
		}
		// Non-zero count reported as OK with a buffer is impossible from
		// the sizing pass; fall through to the fill pass to be safe.
	case int(cStatusBufferTooSmall):
		// Expected: count is in `need`, proceed to fill.
	default:
		runtime.KeepAlive(text)
		return nil, statusToError(int(rc), "ztok_chunk (sizing)")
	}

	n := int(need)
	if n == 0 {
		runtime.KeepAlive(text)
		return nil, nil
	}

	recs := C.ztok_go_chunk_alloc(C.size_t(n))
	defer C.ztok_go_chunk_free_arr(recs)

	var got C.size_t
	rc = C.ztok_chunk(
		pipe, cText, cTextLen,
		C.uint32_t(opts.MaxTokens), C.uint32_t(opts.Overlap), C.uint32_t(opts.Boundary),
		recs, C.size_t(n), &got,
	)
	runtime.KeepAlive(text)
	if err := statusToError(int(rc), "ztok_chunk"); err != nil {
		return nil, err
	}
	// ztok_chunk allocated an id buffer per record; release them once we
	// have copied the ids into Go memory.
	defer C.ztok_chunks_free(recs, got)

	out := make([]Chunk, int(got))
	for i := 0; i < int(got); i++ {
		idx := C.size_t(i)
		idsPtr := unsafe.Pointer(C.ztok_go_chunk_ids(recs, idx))
		idsLen := int(C.ztok_go_chunk_ids_len(recs, idx))
		var ids []uint32
		if idsPtr != nil && idsLen > 0 {
			view := unsafe.Slice((*uint32)(idsPtr), idsLen)
			ids = make([]uint32, idsLen)
			copy(ids, view)
		}
		out[i] = Chunk{
			IDs:        ids,
			ByteStart:  uint32(C.ztok_go_chunk_byte_start(recs, idx)),
			ByteEnd:    uint32(C.ztok_go_chunk_byte_end(recs, idx)),
			TokenStart: uint32(C.ztok_go_chunk_token_start(recs, idx)),
			TokenEnd:   uint32(C.ztok_go_chunk_token_end(recs, idx)),
		}
	}
	return out, nil
}
