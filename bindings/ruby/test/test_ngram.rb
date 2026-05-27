# frozen_string_literal: true

require_relative "helper"

# Engram n-gram hashing tests for the Ruby binding. Mirrors
# bindings/python/tests/test_ngram.py and src/ngram.zig's contract:
# deterministic multi-head token-n-gram hashes, row-major
# [position][head], with positions = ids.length - n + 1.
class TestNgram < Minitest::Test
  include ZtokTestFixtures

  def test_ngram_length_math
    ids = [1, 2, 3, 4, 5]
    # 5 ids, n=2 -> 4 positions; heads=3 -> 12 hashes.
    out = Ztok.ngram_hash(ids, n: 2, heads: 3)
    assert_equal 4 * 3, out.length
  end

  def test_ngram_deterministic
    ids = [7, 8, 9, 10, 11, 12]
    a = Ztok.ngram_hash(ids, n: 3, heads: 4)
    b = Ztok.ngram_hash(ids, n: 3, heads: 4)
    assert_equal a, b
    assert(a.all? { |h| h.is_a?(Integer) && h >= 0 })
  end

  def test_ngram_head_independence
    # The heads of a single position should not all collide.
    out = Ztok.ngram_hash([42, 43, 44], n: 2, heads: 4)
    first_position = out[0, 4]
    assert_operator first_position.uniq.length, :>, 1
  end

  def test_ngram_short_and_bad_args
    # Stream shorter than one window -> empty.
    assert_equal [], Ztok.ngram_hash([1, 2], n: 3, heads: 2)
    # Degenerate args -> empty (no error).
    assert_equal [], Ztok.ngram_hash([], n: 1, heads: 1)
    assert_equal [], Ztok.ngram_hash([1, 2, 3], n: 0, heads: 1)
    assert_equal [], Ztok.ngram_hash([1, 2, 3], n: 2, heads: 0)
  end

  def test_ngram_batch_matches_single
    streams = [
      [1, 2, 3, 4],
      [],            # empty -> no hashes
      [9],           # shorter than window -> no hashes
      [5, 6, 7, 8, 9],
    ]
    batched = nil
    Ztok::BatchPool.open(workers: 2) do |pool|
      batched = Ztok.ngram_hash_batch(pool, streams, n: 2, heads: 3)
    end
    assert_equal streams.length, batched.length
    streams.zip(batched).each do |s, got|
      assert_equal Ztok.ngram_hash(s, n: 2, heads: 3), got
    end
  end
end
