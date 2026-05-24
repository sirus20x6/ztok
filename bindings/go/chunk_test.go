package ztok

// Smoke tests for Pipeline.Chunk over a byte_id pipeline (each input
// byte = one token), so chunk boundaries are predictable: "abcdefghij"
// is 10 tokens, one per ASCII byte.

import (
	"testing"
)

func TestChunkNonOverlapping(t *testing.T) {
	p, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer p.Close()

	chunks, err := p.Chunk("abcdefghij", ChunkOptions{MaxTokens: 4, Overlap: 0})
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	// 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(chunks))
	}
	want := []struct{ ts, te, bs, be, n uint32 }{
		{0, 4, 0, 4, 4},
		{4, 8, 4, 8, 4},
		{8, 10, 8, 10, 2},
	}
	for i, w := range want {
		c := chunks[i]
		if c.TokenStart != w.ts || c.TokenEnd != w.te {
			t.Errorf("chunk %d token range = [%d,%d), want [%d,%d)", i, c.TokenStart, c.TokenEnd, w.ts, w.te)
		}
		if c.ByteStart != w.bs || c.ByteEnd != w.be {
			t.Errorf("chunk %d byte range = [%d,%d), want [%d,%d)", i, c.ByteStart, c.ByteEnd, w.bs, w.be)
		}
		if uint32(len(c.IDs)) != w.n {
			t.Errorf("chunk %d has %d ids, want %d", i, len(c.IDs), w.n)
		}
	}
}

func TestChunkOverlap(t *testing.T) {
	p, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer p.Close()

	chunks, err := p.Chunk("abcdefghij", ChunkOptions{MaxTokens: 4, Overlap: 2})
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want >= 2", len(chunks))
	}
	// stride = 2, so chunk[i+1] starts 2 tokens after chunk[i]; the last
	// 2 ids of chunk[i] equal the first 2 ids of chunk[i+1].
	for i := 0; i+1 < len(chunks); i++ {
		a, b := chunks[i], chunks[i+1]
		if len(a.IDs) < 2 || len(b.IDs) < 2 {
			continue
		}
		if a.IDs[len(a.IDs)-2] != b.IDs[0] || a.IDs[len(a.IDs)-1] != b.IDs[1] {
			t.Errorf("overlap mismatch between chunk %d and %d: %v vs %v", i, i+1, a.IDs, b.IDs)
		}
	}
}

func TestChunkEmptyAndBadArgs(t *testing.T) {
	p, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer p.Close()

	// Empty input -> no chunks, no error.
	chunks, err := p.Chunk("", ChunkOptions{MaxTokens: 4})
	if err != nil {
		t.Fatalf("Chunk empty: %v", err)
	}
	if chunks != nil {
		t.Fatalf("empty input gave %d chunks, want nil", len(chunks))
	}

	// MaxTokens == 0 is invalid.
	if _, err := p.Chunk("abc", ChunkOptions{MaxTokens: 0}); err == nil {
		t.Fatal("MaxTokens=0 should error")
	}
	// Overlap >= MaxTokens is invalid.
	if _, err := p.Chunk("abc", ChunkOptions{MaxTokens: 4, Overlap: 4}); err == nil {
		t.Fatal("Overlap>=MaxTokens should error")
	}
}
