# frozen_string_literal: true

require_relative "helper"

class TestStream < Minitest::Test
  include ZtokTestFixtures

  def setup
    @pipe = Ztok::Pipeline.from_tiktoken(ZtokTestFixtures.tiktoken_fixture, cl100k: true)
  end

  def teardown
    @pipe&.close
  end

  def collect(enum)
    out = []
    enum.each do |batch|
      assert_kind_of Array, batch
      out.concat(batch)
    end
    out
  end

  def test_stream_single_chunk_matches_encode
    text = "hello world the quick brown fox"
    want = @pipe.encode(text)
    got = collect(@pipe.encode_stream(text))
    assert_equal want, got
  end

  def test_stream_many_small_chunks_matches_encode
    text = "hello world the quick brown fox hello world the quick brown fox"
    want = @pipe.encode(text)
    got = collect(@pipe.encode_stream(text, chunk_size: 4))
    assert_equal want, got
  end

  def test_stream_mid_utf8_codepoint_is_deferred
    # "héllo" — é is 0xC3 0xA9 (2 bytes). chunk_size=2 chops mid-é.
    Ztok::Pipeline.byte_id do |bpipe|
      text = "héllo"
      want = bpipe.encode(text)
      got = collect(bpipe.encode_stream(text, chunk_size: 2))
      assert_equal want, got
      assert_equal 6, text.bytesize # "h" + 0xC3 0xA9 + "llo"
    end
  end

  def test_stream_empty_input_yields_nothing
    Ztok::Pipeline.byte_id do |bpipe|
      out = bpipe.encode_stream("").to_a
      assert_equal [], out
    end
  end

  def test_stream_multiline_matches_encode
    text = "hello world\nthe quick brown fox\nfoo bar baz\n"
    want = @pipe.encode(text)
    got = collect(@pipe.encode_stream(text, chunk_size: 8))
    assert_equal want, got
  end

  def test_stream_encoder_low_level
    text = "hello world the quick brown fox"
    want = @pipe.encode(text)

    encoder = Ztok::StreamEncoder.new(@pipe)
    got = []
    begin
      # Feed two halves with arbitrary cut.
      mid = text.bytesize / 2
      got.concat(encoder.feed(text.byteslice(0, mid)))
      got.concat(encoder.feed(text.byteslice(mid, text.bytesize - mid)))
      got.concat(encoder.finish)
    ensure
      encoder.close
    end
    assert_equal want, got
  end
end
