package ztok

// ztok — cgo bridge to libztok.
//
// Build prerequisites:
//
//   - libztok.so (or .dylib / ztok.dll) on the dynamic loader path
//     (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or system /usr/lib).
//   - ztok.h on the C preprocessor include path (`C_INCLUDE_PATH`).
//
// The default build wires pkg-config (`cgo_pkgconfig.go`); install via
// `zig build -p prefix` and drop `prefix/lib/pkgconfig` on
// `PKG_CONFIG_PATH` to pick everything up automatically. Build with
// `-tags nopkgconfig` to switch to a bare `-lztok` link directive
// (`cgo_nopkgconfig.go`) and supply paths via `CGO_CFLAGS` /
// `CGO_LDFLAGS` instead.
//
// All id buffers returned by `ztok_encode_batch_pooled` and
// `ztok_stream_*` carry a length-prefix header (see
// src/c_api.zig::allocIdBuf). The only safe free is `ztok_ids_free` —
// we never reach into the C buffer past its declared length and we
// never call `free(3)` on it.

/*
#include <ztok.h>
*/
import "C"

import (
	"unsafe"
)

// Status codes mirror enum ztok_status in include/ztok.h. Exposed here
// (lowercase) so internal callers don't have to repeat the C-side
// constant names; the public API surfaces them via typed error values
// in errors.go.
const (
	cStatusOK             = C.ZTOK_OK
	cStatusOutOfMemory    = C.ZTOK_ERR_OUT_OF_MEMORY
	cStatusInvalidInput   = C.ZTOK_ERR_INVALID_INPUT
	cStatusBufferTooSmall = C.ZTOK_ERR_BUFFER_TOO_SMALL
	cStatusInternal       = C.ZTOK_ERR_INTERNAL
)

// Enum kinds mirror the *_kind enums in include/ztok.h. Callers pass
// these to Config.{Normalizer,PreTokenizer,Decoder}.
const (
	NormalizerIdentity  uint32 = C.ZTOK_NORMALIZER_IDENTITY
	NormalizerNFC       uint32 = C.ZTOK_NORMALIZER_NFC
	NormalizerNFD       uint32 = C.ZTOK_NORMALIZER_NFD
	NormalizerNFKC      uint32 = C.ZTOK_NORMALIZER_NFKC
	NormalizerNFKD      uint32 = C.ZTOK_NORMALIZER_NFKD
	NormalizerByteLevel uint32 = C.ZTOK_NORMALIZER_BYTE_LEVEL

	PretokIdentity uint32 = C.ZTOK_PRETOK_IDENTITY
	PretokCL100K   uint32 = C.ZTOK_PRETOK_CL100K

	ModelByteID uint32 = C.ZTOK_MODEL_BYTE_ID

	DecoderConcat    uint32 = C.ZTOK_DECODER_CONCAT
	DecoderWordPiece uint32 = C.ZTOK_DECODER_WORDPIECE
	DecoderByteLevel uint32 = C.ZTOK_DECODER_BYTE_LEVEL
)

// Overlay channel kinds mirror enum ztok_overlay_kind in include/ztok.h.
// Pass these to Pipeline.EncodeWithOverlays. Cheap channels
// (BYTE_START/BYTE_END/BOUNDARY/PROVENANCE) carry encoder-derived values;
// domain channels (OPCODE/OPERAND/SYMBOL_REF/HUNK) come back zero-filled
// until a domain plugin populates them.
const (
	OverlayByteStart  uint32 = C.ZTOK_OVERLAY_BYTE_START
	OverlayByteEnd    uint32 = C.ZTOK_OVERLAY_BYTE_END
	OverlayBoundary   uint32 = C.ZTOK_OVERLAY_BOUNDARY
	OverlayOpcode     uint32 = C.ZTOK_OVERLAY_OPCODE
	OverlayOperand    uint32 = C.ZTOK_OVERLAY_OPERAND
	OverlaySymbolRef  uint32 = C.ZTOK_OVERLAY_SYMBOL_REF
	OverlayHunk       uint32 = C.ZTOK_OVERLAY_HUNK
	OverlayProvenance uint32 = C.ZTOK_OVERLAY_PROVENANCE
	OverlayUserBase   uint32 = C.ZTOK_OVERLAY_USER_BASE
)

// Format is the auto-detected on-disk vocab format returned by
// `ztok_auto_detect`. Values mirror enum ztok_format.
type Format uint32

const (
	FormatUnknown       Format = C.ZTOK_FORMAT_UNKNOWN
	FormatTiktoken      Format = C.ZTOK_FORMAT_TIKTOKEN
	FormatHFJSON        Format = C.ZTOK_FORMAT_HF_JSON
	FormatSentencePiece Format = C.ZTOK_FORMAT_SP_MODEL
	FormatZTM           Format = C.ZTOK_FORMAT_ZTM
	FormatRWKV          Format = C.ZTOK_FORMAT_RWKV
)

// String renders a Format as the lower-snake string the other bindings
// expose ("tiktoken", "hf_json", "sentencepiece", "ztm", "unknown").
func (f Format) String() string {
	switch f {
	case FormatTiktoken:
		return "tiktoken"
	case FormatHFJSON:
		return "hf_json"
	case FormatSentencePiece:
		return "sentencepiece"
	case FormatZTM:
		return "ztm"
	case FormatRWKV:
		return "rwkv"
	default:
		return "unknown"
	}
}

// cFreeIDs releases a C-allocated id buffer via ztok_ids_free. This is
// the ONLY safe free path — the buffers carry a length-prefix header
// (src/c_api.zig::allocIdBuf), so calling C.free on them corrupts the
// libztok allocator.
func cFreeIDs(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}
	C.ztok_ids_free((*C.ztok_token_id)(ptr))
}

// materializeIDs copies a C-owned uint32 buffer into a Go []uint32, then
// frees it via ztok_ids_free. We always copy because the C buffer's
// lifetime is bounded by ztok_ids_free, and Go's GC can't reach into
// foreign memory.
func materializeIDs(ptr unsafe.Pointer, n int) []uint32 {
	if ptr == nil {
		return nil
	}
	if n <= 0 {
		cFreeIDs(ptr)
		return nil
	}
	view := unsafe.Slice((*uint32)(ptr), n)
	out := make([]uint32, n)
	copy(out, view)
	cFreeIDs(ptr)
	return out
}
