// Package ztok provides Go bindings for libztok — a fast multithreaded
// tokenizer library written in Zig.
//
// Quickstart:
//
//	pipe, err := ztok.OpenTiktoken("cl100k_base.tiktoken", nil)
//	if err != nil { log.Fatal(err) }
//	defer pipe.Close()
//
//	ids, _ := pipe.Encode("hello world")
//	text, _ := pipe.Decode(ids)
//
//	// Auto-detect format.
//	pipe2, _ := ztok.Open("tokenizer.json")
//	defer pipe2.Close()
//
//	// Multithreaded batch encode.
//	pool, _ := ztok.NewBatchPool(&ztok.BatchPoolOptions{Workers: 8})
//	defer pool.Close()
//	results, _ := pipe.EncodeBatch(pool, []string{"foo", "bar", "baz"})
//
//	// Streaming.
//	pipe.EncodeStream(largeText, func(ids []uint32) error {
//	    return process(ids)
//	}, nil)
//
//	fmt.Println(ztok.Version())
//
// Lifetime: Pipeline / BatchPool / streamEncoder all wrap C handles.
// Always call Close() — finalizers fire on GC schedule, not when you'd
// like. After Close(), every method returns ErrClosed.
package ztok

/*
#include <ztok.h>
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// ErrClosed is returned by methods called on a Pipeline / BatchPool /
// stream that has already been closed.
var ErrClosed = errors.New("ztok: handle is closed")

// Version returns the libztok version string (e.g. "1.20.0").
func Version() string {
	raw := C.ztok_version()
	if raw == nil {
		return ""
	}
	return C.GoString(raw)
}

// DetectFormat sniffs `path` for a known tokenizer file format. Returns
// FormatUnknown on any I/O error or unrecognized magic — matches the C
// contract (best-effort, never raises).
func DetectFormat(path string) Format {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	return Format(C.ztok_auto_detect(cpath))
}

// --- Config --------------------------------------------------------------

// Config configures pipeline construction. All fields are optional; a
// nil *Config applies the sensible defaults documented in include/ztok.h.
type Config struct {
	// Normalizer kind (NormalizerIdentity, NormalizerNFC, ...).
	Normalizer uint32

	// Pre-tokenizer kind (PretokIdentity, PretokCL100K).
	PreTokenizer uint32

	// Decoder kind (DecoderConcat, DecoderWordPiece, DecoderByteLevel).
	Decoder uint32

	// CL100K toggles the cl100k_base pre-tokenizer for tiktoken / hf_json
	// loaders. Convenience flag layered on top of PreTokenizer.
	//
	// Tiktoken loaders default to true (cl100k); hf_json loaders default
	// to false (identity).
	CL100K bool

	// UnkID for WordPiece / SentencePiece loaders.
	UnkID uint32
}

// buildCConfig fills a C `ztok_pipeline_config` struct from the Go
// Config + a default pre-tokenizer. The pre-tokenizer comes from the
// caller because tiktoken and hf_json loaders have different defaults.
func (c *Config) buildCConfig(defaultPreTok uint32, defaultDecoder uint32) C.ztok_pipeline_config {
	cfg := C.ztok_pipeline_config{
		normalizer:    C.ztok_normalizer_kind(NormalizerIdentity),
		pre_tokenizer: C.ztok_pretok_kind(defaultPreTok),
		model:         C.ztok_model_kind(ModelByteID),
		decoder:       C.ztok_decoder_kind(defaultDecoder),
	}
	if c == nil {
		return cfg
	}
	if c.Normalizer != 0 {
		cfg.normalizer = C.ztok_normalizer_kind(c.Normalizer)
	}
	if c.PreTokenizer != 0 {
		cfg.pre_tokenizer = C.ztok_pretok_kind(c.PreTokenizer)
	}
	if c.Decoder != 0 {
		cfg.decoder = C.ztok_decoder_kind(c.Decoder)
	}
	return cfg
}

// --- Pipeline ------------------------------------------------------------

// Pipeline is a loaded tokenizer pipeline. Construct via NewByteID,
// OpenTiktoken, OpenHFJSON, OpenWordPiece, OpenSentencePiece,
// OpenMonster, or Open (auto-detect). Always Close() when done.
type Pipeline struct {
	handle unsafe.Pointer
	closed bool
}

// NewByteID builds the baseline byte_id pipeline (each input byte maps
// to its own id). cfg may be nil.
func NewByteID(cfg *Config) (*Pipeline, error) {
	c := cfg.buildCConfig(PretokIdentity, DecoderConcat)
	var status C.ztok_status
	h := unsafe.Pointer(C.ztok_pipeline_new(&c, &status))
	if err := statusToError(int(status), "ztok_pipeline_new"); err != nil {
		return nil, err
	}
	if h == nil {
		return nil, &InternalError{&StatusError{Status: -1, Op: "ztok_pipeline_new returned NULL"}}
	}
	return wrapPipeline(h), nil
}

// OpenTiktoken loads a .tiktoken vocab into a byte-level BPE pipeline.
// cfg.CL100K defaults to true.
func OpenTiktoken(path string, cfg *Config) (*Pipeline, error) {
	defaultPre := uint32(PretokCL100K)
	if cfg != nil && !cfg.CL100K && cfg.PreTokenizer == 0 {
		// Only override when caller explicitly opted out via CL100K=false
		// AND didn't set a custom PreTokenizer. The CL100K flag exists
		// precisely so callers don't have to know about PretokCL100K.
		defaultPre = PretokIdentity
	}
	return openFromFile(path, "ztok_pipeline_new_bpe_from_tiktoken",
		cfg, defaultPre, DecoderConcat)
}

// OpenHFJSON loads a HuggingFace tokenizer.json BPE model. cfg.CL100K
// defaults to false.
func OpenHFJSON(path string, cfg *Config) (*Pipeline, error) {
	defaultPre := uint32(PretokIdentity)
	if cfg != nil && cfg.CL100K {
		defaultPre = PretokCL100K
	}
	return openFromFile(path, "ztok_pipeline_new_bpe_from_hf_json",
		cfg, defaultPre, DecoderConcat)
}

// OpenWordPiece loads a HuggingFace WordPiece model from tokenizer.json.
// cfg.UnkID is required (no sensible default for an unknown-token id).
func OpenWordPiece(path string, cfg *Config) (*Pipeline, error) {
	if cfg == nil {
		return nil, &InvalidInputError{&StatusError{
			Status: int(cStatusInvalidInput),
			Op:     "OpenWordPiece: cfg.UnkID required",
		}}
	}
	c := cfg.buildCConfig(PretokIdentity, DecoderWordPiece)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var status C.ztok_status
	h := unsafe.Pointer(C.ztok_pipeline_new_wordpiece_from_hf_json(
		cpath, C.uint32_t(cfg.UnkID), &c, &status,
	))
	if err := statusToError(int(status), "ztok_pipeline_new_wordpiece_from_hf_json"); err != nil {
		return nil, err
	}
	if h == nil {
		return nil, &InternalError{&StatusError{Status: -1, Op: "wordpiece loader returned NULL"}}
	}
	return wrapPipeline(h), nil
}

// OpenSentencePiece loads a SentencePiece .model (Unigram) file. cfg may
// be nil; cfg.UnkID defaults to 0.
func OpenSentencePiece(path string, cfg *Config) (*Pipeline, error) {
	c := cfg.buildCConfig(PretokIdentity, DecoderConcat)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var status C.ztok_status
	unkID := uint32(0)
	if cfg != nil {
		unkID = cfg.UnkID
	}
	h := unsafe.Pointer(C.ztok_pipeline_new_unigram_from_sp_model(
		cpath, C.uint32_t(unkID), &c, &status,
	))
	if err := statusToError(int(status), "ztok_pipeline_new_unigram_from_sp_model"); err != nil {
		return nil, err
	}
	if h == nil {
		return nil, &InternalError{&StatusError{Status: -1, Op: "sentencepiece loader returned NULL"}}
	}
	return wrapPipeline(h), nil
}

// OpenMonster loads a ztok TokenMonster .ztm vocab file.
func OpenMonster(path string, cfg *Config) (*Pipeline, error) {
	return openFromFile(path, "ztok_pipeline_new_monster_from_file",
		cfg, PretokIdentity, DecoderConcat)
}

// Open auto-detects `path`'s format and dispatches to the right loader.
//
//   - .tiktoken            → OpenTiktoken with CL100K=true
//   - tokenizer.json       → OpenHFJSON
//   - .model               → OpenSentencePiece (UnkID=0)
//   - .ztm                 → OpenMonster
//
// For WordPiece (which lives inside tokenizer.json but needs a specific
// UnkID), call OpenWordPiece directly.
func Open(path string) (*Pipeline, error) {
	fmt := DetectFormat(path)
	switch fmt {
	case FormatTiktoken:
		return OpenTiktoken(path, &Config{CL100K: true})
	case FormatHFJSON:
		return OpenHFJSON(path, nil)
	case FormatSentencePiece:
		return OpenSentencePiece(path, nil)
	case FormatZTM:
		return OpenMonster(path, nil)
	default:
		return nil, &InvalidInputError{&StatusError{
			Status: int(cStatusInvalidInput),
			Op:     "Open: could not auto-detect tokenizer format for " + path,
		}}
	}
}

// openFromFile is the shared body for the simple file loaders that take
// only (path, cfg, status) — the BPE-from-tiktoken / BPE-from-hf_json /
// monster-from-file trio. WordPiece and SentencePiece take an extra
// unk_id and have their own loaders.
func openFromFile(path, fnName string, cfg *Config, defaultPre, defaultDecoder uint32) (*Pipeline, error) {
	c := cfg.buildCConfig(defaultPre, defaultDecoder)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var status C.ztok_status
	var h unsafe.Pointer
	switch fnName {
	case "ztok_pipeline_new_bpe_from_tiktoken":
		h = unsafe.Pointer(C.ztok_pipeline_new_bpe_from_tiktoken(cpath, &c, &status))
	case "ztok_pipeline_new_bpe_from_hf_json":
		h = unsafe.Pointer(C.ztok_pipeline_new_bpe_from_hf_json(cpath, &c, &status))
	case "ztok_pipeline_new_monster_from_file":
		h = unsafe.Pointer(C.ztok_pipeline_new_monster_from_file(cpath, &c, &status))
	default:
		return nil, &InternalError{&StatusError{Status: -1, Op: "unknown loader: " + fnName}}
	}
	if err := statusToError(int(status), fnName); err != nil {
		return nil, err
	}
	if h == nil {
		return nil, &InternalError{&StatusError{Status: -1, Op: fnName + " returned NULL"}}
	}
	return wrapPipeline(h), nil
}

func wrapPipeline(h unsafe.Pointer) *Pipeline {
	p := &Pipeline{handle: h}
	runtime.SetFinalizer(p, func(p *Pipeline) {
		if p.handle != nil {
			C.ztok_pipeline_free((*C.ztok_pipeline)(p.handle))
			p.handle = nil
		}
	})
	return p
}

// Close releases the pipeline's C handle. Idempotent.
func (p *Pipeline) Close() error {
	if p.closed {
		return nil
	}
	p.closed = true
	runtime.SetFinalizer(p, nil)
	if p.handle != nil {
		C.ztok_pipeline_free((*C.ztok_pipeline)(p.handle))
		p.handle = nil
	}
	return nil
}

func (p *Pipeline) checkOpen() error {
	if p.closed || p.handle == nil {
		return ErrClosed
	}
	return nil
}

// --- encode / decode -----------------------------------------------------

// Encode tokenizes `text` (interpreted as UTF-8 bytes) into a slice of
// token ids.
func (p *Pipeline) Encode(text string) ([]uint32, error) {
	return p.EncodeBytes([]byte(text))
}

// EncodeBytes tokenizes raw bytes. Use this for non-UTF-8 inputs (the
// underlying tokenizer treats inputs as a byte stream).
func (p *Pipeline) EncodeBytes(data []byte) ([]uint32, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}

	// Encode is a write-through API: we allocate a Go-side []uint32,
	// pass its base pointer, and grow on BUFFER_TOO_SMALL. The C ABI's
	// per-span `maxTokensFor` bound is conservative, so the initial size
	// is generous (input bytes + 16, min 64) to skip most resize loops.
	cap := len(data) + 16
	if cap < 64 {
		cap = 64
	}
	cInput := (*C.char)(unsafe.Pointer(&data[0]))
	cInputLen := C.size_t(len(data))
	pipe := (*C.ztok_pipeline)(p.handle)

	for attempt := 0; attempt < 8; attempt++ {
		out := make([]uint32, cap)
		var outLen C.size_t
		rc := C.ztok_encode(
			pipe, cInput, cInputLen,
			(*C.ztok_token_id)(unsafe.Pointer(&out[0])),
			C.size_t(cap), &outLen,
		)
		switch int(rc) {
		case int(cStatusOK):
			runtime.KeepAlive(data)
			return out[:int(outLen)], nil
		case int(cStatusBufferTooSmall):
			next := cap * 2
			if int(outLen)+16 > next {
				next = int(outLen) + 16
			}
			cap = next
		default:
			runtime.KeepAlive(data)
			return nil, statusToError(int(rc), "ztok_encode")
		}
	}
	return nil, &InternalError{&StatusError{
		Status: int(cStatusBufferTooSmall),
		Op:     "ztok_encode: BUFFER_TOO_SMALL after 8 grow attempts",
	}}
}

// EncodeWithOverlays tokenizes `text` and returns the ids plus a map of
// per-token annotation channels aligned 1:1 with the id stream. Each
// requested overlay kind (the Overlay* constants) maps to a []uint32 of
// len(ids) values.
//
// Requesting overlays never changes tokenization — the returned ids are
// identical to Encode. Cheap channels (BYTE_START/BYTE_END/BOUNDARY/
// PROVENANCE) carry encoder-derived values; domain channels
// (OPCODE/OPERAND/SYMBOL_REF/HUNK) come back zero-filled until a domain
// plugin populates them.
func (p *Pipeline) EncodeWithOverlays(text string, channels []uint32) ([]uint32, map[uint32][]uint32, error) {
	return p.EncodeBytesWithOverlays([]byte(text), channels)
}

// EncodeBytesWithOverlays is the raw-bytes form of EncodeWithOverlays.
// Use this for non-UTF-8 inputs.
func (p *Pipeline) EncodeBytesWithOverlays(data []byte, channels []uint32) ([]uint32, map[uint32][]uint32, error) {
	if err := p.checkOpen(); err != nil {
		return nil, nil, err
	}

	// Reject duplicate kinds — the result map would silently collapse
	// them, which is almost certainly a caller bug.
	seen := make(map[uint32]struct{}, len(channels))
	for _, k := range channels {
		if _, dup := seen[k]; dup {
			return nil, nil, &InvalidInputError{&StatusError{
				Status: int(cStatusInvalidInput),
				Op:     "EncodeWithOverlays: duplicate overlay kind requested",
			}}
		}
		seen[k] = struct{}{}
	}

	if len(data) == 0 {
		overlays := make(map[uint32][]uint32, len(channels))
		for _, k := range channels {
			overlays[k] = []uint32{}
		}
		return nil, overlays, nil
	}

	cInput := (*C.char)(unsafe.Pointer(&data[0]))
	cInputLen := C.size_t(len(data))
	pipe := (*C.ztok_pipeline)(p.handle)
	nCh := len(channels)

	// cgo forbids passing a Go pointer that points to other Go memory.
	// The channels[] array's `out` fields must point at C-allocated
	// buffers, so we allocate the channel struct array and every
	// per-channel buffer in C heap memory and copy results back into Go.
	const chanSize = C.size_t(unsafe.Sizeof(C.ztok_overlay_channel{}))

	// chanArrayAt returns a typed pointer to the i-th channel struct in a
	// C-allocated array.
	chanArrayAt := func(base unsafe.Pointer, i int) *C.ztok_overlay_channel {
		return (*C.ztok_overlay_channel)(unsafe.Pointer(uintptr(base) + uintptr(i)*uintptr(chanSize)))
	}

	// --- Sizing pass: out_ids = NULL queries the token count. ---
	var sizeArr unsafe.Pointer
	if nCh > 0 {
		sizeArr = C.malloc(C.size_t(nCh) * chanSize)
		defer C.free(sizeArr)
		for i, k := range channels {
			c := chanArrayAt(sizeArr, i)
			c.kind = C.ztok_overlay_kind(k)
			c.out = nil
			c.out_cap = 0
		}
	}
	var outLen C.size_t
	rc := C.ztok_encode_with_overlays(
		pipe, cInput, cInputLen,
		nil, 0,
		(*C.ztok_overlay_channel)(sizeArr), C.size_t(nCh),
		&outLen,
	)
	if int(rc) != int(cStatusOK) && int(rc) != int(cStatusBufferTooSmall) {
		runtime.KeepAlive(data)
		return nil, nil, statusToError(int(rc), "ztok_encode_with_overlays (sizing)")
	}

	count := int(outLen)
	if count == 0 {
		runtime.KeepAlive(data)
		overlays := make(map[uint32][]uint32, nCh)
		for _, k := range channels {
			overlays[k] = []uint32{}
		}
		return nil, overlays, nil
	}

	// --- Fill pass: allocate the id buffer + one C uint32 buffer per
	// channel, each sized to the exact token count from the sizing pass. ---
	idBuf := C.malloc(C.size_t(count) * C.size_t(unsafe.Sizeof(C.ztok_token_id(0))))
	defer C.free(idBuf)

	chanBufs := make([]unsafe.Pointer, nCh)
	bufBytes := C.size_t(count) * C.size_t(unsafe.Sizeof(C.uint32_t(0)))
	var fillArr unsafe.Pointer
	if nCh > 0 {
		fillArr = C.malloc(C.size_t(nCh) * chanSize)
		defer C.free(fillArr)
		for i, k := range channels {
			chanBufs[i] = C.malloc(bufBytes)
			defer C.free(chanBufs[i])
			c := chanArrayAt(fillArr, i)
			c.kind = C.ztok_overlay_kind(k)
			c.out = (*C.uint32_t)(chanBufs[i])
			c.out_cap = C.size_t(count)
		}
	}

	outLen = 0
	rc = C.ztok_encode_with_overlays(
		pipe, cInput, cInputLen,
		(*C.ztok_token_id)(idBuf), C.size_t(count),
		(*C.ztok_overlay_channel)(fillArr), C.size_t(nCh),
		&outLen,
	)
	runtime.KeepAlive(data)
	if err := statusToError(int(rc), "ztok_encode_with_overlays"); err != nil {
		return nil, nil, err
	}

	n := int(outLen)
	ids := make([]uint32, n)
	copy(ids, unsafe.Slice((*uint32)(idBuf), n))
	overlays := make(map[uint32][]uint32, nCh)
	for i, k := range channels {
		dst := make([]uint32, n)
		copy(dst, unsafe.Slice((*uint32)(chanBufs[i]), n))
		overlays[k] = dst
	}
	return ids, overlays, nil
}

// Decode reverses Encode — converts ids back to a string. Invalid UTF-8
// is preserved by Go's []byte→string conversion (no replacement).
func (p *Pipeline) Decode(ids []uint32) (string, error) {
	b, err := p.DecodeBytes(ids)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// DecodeBytes converts ids to raw bytes (no UTF-8 round-tripping).
func (p *Pipeline) DecodeBytes(ids []uint32) ([]byte, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, nil
	}
	pipe := (*C.ztok_pipeline)(p.handle)
	cIDs := (*C.ztok_token_id)(unsafe.Pointer(&ids[0]))
	cIDsLen := C.size_t(len(ids))

	// Sizing pass — pass out_cap=0 so the call writes the required size
	// into out_len and returns BUFFER_TOO_SMALL.
	var sized C.size_t
	rc := C.ztok_decode(pipe, cIDs, cIDsLen, nil, 0, &sized)
	if int(rc) != int(cStatusOK) && int(rc) != int(cStatusBufferTooSmall) {
		runtime.KeepAlive(ids)
		return nil, statusToError(int(rc), "ztok_decode (sizing)")
	}
	if sized == 0 {
		runtime.KeepAlive(ids)
		return nil, nil
	}

	out := make([]byte, int(sized))
	var written C.size_t
	rc = C.ztok_decode(
		pipe, cIDs, cIDsLen,
		(*C.char)(unsafe.Pointer(&out[0])),
		sized, &written,
	)
	runtime.KeepAlive(ids)
	if err := statusToError(int(rc), "ztok_decode"); err != nil {
		return nil, err
	}
	return out[:int(written)], nil
}
