package ztok

import (
	"reflect"
	"testing"
)

func TestHashNGramsDeterministic(t *testing.T) {
	ids := []uint32{7, 42, 1000, 3, 99, 7, 42}
	a, err := HashNGrams(ids, 3, 2)
	if err != nil {
		t.Fatalf("HashNGrams: %v", err)
	}
	// positions = 7 - 3 + 1 = 5, heads = 2 -> 10 hashes
	if len(a) != 10 {
		t.Fatalf("expected 10 hashes, got %d", len(a))
	}
	b, err := HashNGrams(ids, 3, 2)
	if err != nil {
		t.Fatalf("HashNGrams (2nd): %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Errorf("hashes not deterministic across calls")
	}
}

func TestHashNGramsShortInput(t *testing.T) {
	out, err := HashNGrams([]uint32{1, 2}, 3, 4)
	if err != nil {
		t.Fatalf("HashNGrams: %v", err)
	}
	if out != nil {
		t.Errorf("expected nil for stream shorter than window, got %v", out)
	}
}

func TestHashNGramsHeadsIndependent(t *testing.T) {
	out, err := HashNGrams([]uint32{5, 6, 7, 8}, 3, 3)
	if err != nil {
		t.Fatalf("HashNGrams: %v", err)
	}
	// 2 positions x 3 heads = 6
	if len(out) != 6 {
		t.Fatalf("expected 6 hashes, got %d", len(out))
	}
	// The 3 heads for position 0 must differ from each other.
	if out[0] == out[1] || out[1] == out[2] || out[0] == out[2] {
		t.Errorf("heads not independent: %v", out[:3])
	}
}

func TestHashNGramsBatchMatchesSingle(t *testing.T) {
	pool, err := NewBatchPool(&BatchPoolOptions{Workers: 2})
	if err != nil {
		t.Fatalf("NewBatchPool: %v", err)
	}
	defer pool.Close()

	streams := [][]uint32{
		{1, 2, 3, 4, 5},
		{9, 8, 7},
		{1, 1}, // shorter than n -> nil
	}
	got, err := HashNGramsBatch(pool, streams, 3, 2)
	if err != nil {
		t.Fatalf("HashNGramsBatch: %v", err)
	}
	if len(got) != len(streams) {
		t.Fatalf("expected %d results, got %d", len(streams), len(got))
	}
	for i, s := range streams {
		want, err := HashNGrams(s, 3, 2)
		if err != nil {
			t.Fatalf("HashNGrams(%d): %v", i, err)
		}
		if !reflect.DeepEqual(want, got[i]) {
			t.Errorf("stream %d: batch=%v single=%v", i, got[i], want)
		}
	}
}
