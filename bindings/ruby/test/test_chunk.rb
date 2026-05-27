# frozen_string_literal: true

require_relative "helper"

# Token-window chunking tests for the Ruby binding. Mirrors
# bindings/python/tests/test_chunk.py. Run over a byte_id pipeline (each
# input byte = one token) so chunk boundaries are predictable:
# "abcdefghij" is 10 tokens, one per byte.
class TestChunk < Minitest::Test
  include ZtokTestFixtures

  def with_byte_pipeline
    pipe = Ztok::Pipeline.byte_id
    begin
      yield pipe
    ensure
      pipe.close
    end
  end

  def test_chunk_non_overlapping
    with_byte_pipeline do |pipe|
      chunks = pipe.chunk("abcdefghij", max_tokens: 4, overlap: 0)
      # 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
      assert_equal 3, chunks.length
      # [token_start, token_end, byte_start, byte_end, n_ids]
      want = [[0, 4, 0, 4, 4], [4, 8, 4, 8, 4], [8, 10, 8, 10, 2]]
      chunks.zip(want).each do |c, (ts, te, bs, be, n)|
        assert_equal [ts, te], [c.token_start, c.token_end]
        assert_equal [bs, be], [c.byte_start, c.byte_end]
        assert_equal n, c.ids.length
      end
    end
  end

  def test_chunk_overlap
    with_byte_pipeline do |pipe|
      chunks = pipe.chunk("abcdefghij", max_tokens: 4, overlap: 2)
      assert_operator chunks.length, :>=, 2
      # stride = 2, so the last 2 ids of chunk[i] equal the first 2 of
      # chunk[i+1].
      chunks.each_cons(2) do |a, b|
        if a.ids.length >= 2 && b.ids.length >= 2
          assert_equal a.ids[-2, 2], b.ids[0, 2]
        end
      end
    end
  end

  def test_chunk_empty_and_bad_args
    with_byte_pipeline do |pipe|
      assert_equal [], pipe.chunk("", max_tokens: 4)
      assert_raises(Ztok::InvalidInputError) do
        pipe.chunk("abc", max_tokens: 0)
      end
      assert_raises(Ztok::InvalidInputError) do
        pipe.chunk("abc", max_tokens: 4, overlap: 4)
      end
    end
  end
end
