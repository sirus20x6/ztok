package ztok

// Tokenizer fingerprint — a deterministic 32-byte SHA-256 digest over a
// pipeline's encoding behavior on a fixed canonical input set plus a
// model-kind tag and vocab size (see src/fingerprint.zig). Two pipelines
// that return equal fingerprints produce bit-identical id streams for
// any input; use it as a cache key, KV-store discriminator, or
// training-pipeline guard.

/*
#include <ztok.h>
*/
import "C"

import (
	"encoding/hex"
	"runtime"
	"unsafe"
)

// FingerprintSize is the fixed digest length in bytes.
const FingerprintSize = 32

// Fingerprint is a 32-byte deterministic tokenizer fingerprint. Two
// pipelines that produce equal Fingerprints emit bit-identical id
// streams for any input. The zero value is not a valid fingerprint.
type Fingerprint [FingerprintSize]byte

// Hex returns the lowercase 64-char hexadecimal form (no separators).
func (f Fingerprint) Hex() string {
	return hex.EncodeToString(f[:])
}

// String implements fmt.Stringer.
func (f Fingerprint) String() string {
	return "Fingerprint(" + f.Hex() + ")"
}

// Fingerprint computes the pipeline's tokenizer fingerprint. Two
// pipelines that return equal fingerprints produce bit-identical id
// streams for any input.
func (p *Pipeline) Fingerprint() (Fingerprint, error) {
	var fp Fingerprint
	if err := p.checkOpen(); err != nil {
		return fp, err
	}
	pipe := (*C.ztok_pipeline)(p.handle)
	// ztok_fingerprint writes exactly 32 bytes into out_32. We hand it a
	// pointer to the Go array's backing storage; the call is synchronous
	// and copies nothing past return, so KeepAlive guards it.
	rc := C.ztok_fingerprint(pipe, (*C.uint8_t)(unsafe.Pointer(&fp[0])))
	runtime.KeepAlive(p)
	if err := statusToError(int(rc), "ztok_fingerprint"); err != nil {
		return Fingerprint{}, err
	}
	return fp, nil
}
