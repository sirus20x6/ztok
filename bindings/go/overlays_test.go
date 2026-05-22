package ztok

import "testing"

// TestEncodeWithOverlaysIDsMatch checks that requesting overlays never
// changes tokenization — the id stream is identical to plain Encode.
func TestEncodeWithOverlaysIDsMatch(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	text := "hello world"
	plain, err := pipe.Encode(text)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	ids, overlays, err := pipe.EncodeWithOverlays(text, []uint32{OverlayByteStart, OverlayByteEnd})
	if err != nil {
		t.Fatalf("EncodeWithOverlays: %v", err)
	}
	if !equalU32(plain, ids) {
		t.Errorf("overlay ids mismatch plain encode:\n  plain=%v\n  overlay=%v", plain, ids)
	}
	if len(overlays) != 2 {
		t.Errorf("expected 2 channels, got %d", len(overlays))
	}
	if _, ok := overlays[OverlayByteStart]; !ok {
		t.Error("missing OverlayByteStart channel")
	}
	if _, ok := overlays[OverlayByteEnd]; !ok {
		t.Error("missing OverlayByteEnd channel")
	}
}

// TestEncodeWithOverlaysChannelLengths checks every channel array has the
// same length as the id stream.
func TestEncodeWithOverlaysChannelLengths(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	ids, overlays, err := pipe.EncodeWithOverlays("the quick brown fox", []uint32{
		OverlayByteStart, OverlayByteEnd, OverlayBoundary, OverlayProvenance,
	})
	if err != nil {
		t.Fatalf("EncodeWithOverlays: %v", err)
	}
	for kind, values := range overlays {
		if len(values) != len(ids) {
			t.Errorf("channel %d length %d != ids length %d", kind, len(values), len(ids))
		}
	}
}

// TestEncodeWithOverlaysByteSpans checks BYTE_START/BYTE_END give sensible,
// tiling spans over the input.
func TestEncodeWithOverlaysByteSpans(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	text := "hello world"
	_, overlays, err := pipe.EncodeWithOverlays(text, []uint32{OverlayByteStart, OverlayByteEnd})
	if err != nil {
		t.Fatalf("EncodeWithOverlays: %v", err)
	}
	starts := overlays[OverlayByteStart]
	ends := overlays[OverlayByteEnd]
	n := uint32(len([]byte(text)))
	if len(starts) == 0 {
		t.Fatal("no spans produced")
	}
	for i := range starts {
		if !(starts[i] < ends[i] && ends[i] <= n) {
			t.Errorf("bad span (%d, %d) for input of %d bytes", starts[i], ends[i], n)
		}
	}
	if starts[0] != 0 {
		t.Errorf("first span should start at 0, got %d", starts[0])
	}
	if ends[len(ends)-1] != n {
		t.Errorf("last span should end at %d, got %d", n, ends[len(ends)-1])
	}
	// Spans tile left-to-right: each picks up where the previous ended.
	for i := 1; i < len(starts); i++ {
		if starts[i] != ends[i-1] {
			t.Errorf("span gap: starts[%d]=%d != ends[%d]=%d", i, starts[i], i-1, ends[i-1])
		}
	}
}

// TestEncodeWithOverlaysByteIDSingleByteSpans checks the byte_id pipeline
// produces one-byte spans.
func TestEncodeWithOverlaysByteIDSingleByteSpans(t *testing.T) {
	pipe, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer pipe.Close()

	ids, overlays, err := pipe.EncodeWithOverlays("hi", []uint32{OverlayByteStart, OverlayByteEnd})
	if err != nil {
		t.Fatalf("EncodeWithOverlays: %v", err)
	}
	if !equalU32(ids, []uint32{0x68, 0x69}) {
		t.Errorf("ids = %v, want [104 105]", ids)
	}
	if !equalU32(overlays[OverlayByteStart], []uint32{0, 1}) {
		t.Errorf("byte_start = %v, want [0 1]", overlays[OverlayByteStart])
	}
	if !equalU32(overlays[OverlayByteEnd], []uint32{1, 2}) {
		t.Errorf("byte_end = %v, want [1 2]", overlays[OverlayByteEnd])
	}
}

// TestEncodeWithOverlaysOpcodeAllZero checks an OPCODE domain channel comes
// back zero-filled when no domain plugin is configured.
func TestEncodeWithOverlaysOpcodeAllZero(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	ids, overlays, err := pipe.EncodeWithOverlays("hello world", []uint32{OverlayOpcode})
	if err != nil {
		t.Fatalf("EncodeWithOverlays: %v", err)
	}
	opcode := overlays[OverlayOpcode]
	if len(opcode) != len(ids) {
		t.Fatalf("opcode length %d != ids length %d", len(opcode), len(ids))
	}
	for i, v := range opcode {
		if v != 0 {
			t.Errorf("opcode[%d] = %d, want 0 (no domain plugin)", i, v)
		}
	}
}

// TestEncodeWithOverlaysEmptyAndNoChannels covers the edge cases.
func TestEncodeWithOverlaysEmptyAndNoChannels(t *testing.T) {
	pipe, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer pipe.Close()

	// Empty input -> empty ids + empty per-channel arrays.
	ids, overlays, err := pipe.EncodeWithOverlays("", []uint32{OverlayByteStart, OverlayOpcode})
	if err != nil {
		t.Fatalf("EncodeWithOverlays(empty): %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("empty input ids = %v, want empty", ids)
	}
	for _, k := range []uint32{OverlayByteStart, OverlayOpcode} {
		if v, ok := overlays[k]; !ok || len(v) != 0 {
			t.Errorf("channel %d = %v, want present+empty", k, v)
		}
	}

	// No channels requested -> just ids, empty map.
	plain, err := pipe.Encode("hello world")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	ids2, overlays2, err := pipe.EncodeWithOverlays("hello world", nil)
	if err != nil {
		t.Fatalf("EncodeWithOverlays(no channels): %v", err)
	}
	if !equalU32(plain, ids2) {
		t.Errorf("ids mismatch: %v vs %v", plain, ids2)
	}
	if len(overlays2) != 0 {
		t.Errorf("expected empty overlay map, got %v", overlays2)
	}

	// Duplicate kinds are rejected.
	if _, _, err := pipe.EncodeWithOverlays("hi", []uint32{OverlayByteStart, OverlayByteStart}); err == nil {
		t.Error("expected error on duplicate overlay kinds")
	}
}
