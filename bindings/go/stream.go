package ztok

/*
#include <ztok.h>
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// StreamOptions configures EncodeStream.
type StreamOptions struct {
	// ChunkSize is the per-feed byte count. The encoder defers a
	// trailing partial UTF-8 codepoint / pre-tokenizer span up to a
	// soft cap of 1 MiB (see src/stream.zig); past that it force-cuts at
	// the nearest codepoint boundary. Zero defaults to 64 KiB.
	ChunkSize int
}

// EncodeStream feeds `data` (interpreted as UTF-8 bytes) to the encoder
// in chunks of opts.ChunkSize bytes. For each non-empty batch of ids
// the encoder emits, onIDs is invoked. If onIDs returns an error,
// streaming halts and the error propagates back.
//
// The final ztok_stream_finish call drains any remaining carry; empty
// batches are skipped so onIDs only sees non-empty slices.
func (p *Pipeline) EncodeStream(data string, onIDs func(ids []uint32) error, opts *StreamOptions) error {
	return p.EncodeStreamBytes([]byte(data), onIDs, opts)
}

// EncodeStreamBytes is EncodeStream over raw bytes.
func (p *Pipeline) EncodeStreamBytes(data []byte, onIDs func(ids []uint32) error, opts *StreamOptions) error {
	if err := p.checkOpen(); err != nil {
		return err
	}
	if onIDs == nil {
		return &InvalidInputError{&StatusError{
			Status: int(cStatusInvalidInput),
			Op:     "EncodeStream: onIDs callback is nil",
		}}
	}
	chunkSize := 64 * 1024
	if opts != nil && opts.ChunkSize > 0 {
		chunkSize = opts.ChunkSize
	}

	var status C.ztok_status
	streamH := unsafe.Pointer(C.ztok_stream_new((*C.ztok_pipeline)(p.handle), &status))
	if err := statusToError(int(status), "ztok_stream_new"); err != nil {
		return err
	}
	if streamH == nil {
		return &InternalError{&StatusError{Status: -1, Op: "ztok_stream_new returned NULL"}}
	}
	defer C.ztok_stream_free((*C.ztok_stream)(streamH))

	feed := func(chunk []byte) error {
		var outIDs *C.ztok_token_id
		var outN C.size_t
		var cBytes *C.char
		var cLen C.size_t
		if len(chunk) > 0 {
			cBytes = (*C.char)(unsafe.Pointer(&chunk[0]))
			cLen = C.size_t(len(chunk))
		}
		rc := C.ztok_stream_feed(
			(*C.ztok_stream)(streamH),
			cBytes, cLen,
			&outIDs, &outN,
		)
		runtime.KeepAlive(chunk)
		if err := statusToError(int(rc), "ztok_stream_feed"); err != nil {
			if outIDs != nil {
				cFreeIDs(unsafe.Pointer(outIDs))
			}
			return err
		}
		ids := materializeIDs(unsafe.Pointer(outIDs), int(outN))
		if len(ids) > 0 {
			return onIDs(ids)
		}
		return nil
	}

	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}
		if err := feed(data[i:end]); err != nil {
			return err
		}
	}

	// Empty input: no feed loop runs, but still call finish for symmetry.
	if len(data) == 0 {
		if err := feed(nil); err != nil {
			return err
		}
	}

	// Final flush.
	var outIDs *C.ztok_token_id
	var outN C.size_t
	rc := C.ztok_stream_finish((*C.ztok_stream)(streamH), &outIDs, &outN)
	if err := statusToError(int(rc), "ztok_stream_finish"); err != nil {
		if outIDs != nil {
			cFreeIDs(unsafe.Pointer(outIDs))
		}
		return err
	}
	ids := materializeIDs(unsafe.Pointer(outIDs), int(outN))
	if len(ids) > 0 {
		return onIDs(ids)
	}
	return nil
}
