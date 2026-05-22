# frozen_string_literal: true

require_relative "helper"

class TestBasic < Minitest::Test
  include ZtokTestFixtures

  def test_version_is_nonempty_string
    v = Ztok.version
    assert_kind_of String, v
    assert_includes v, "."
    major, minor = v.split(".")[0, 2]
    assert_match(/^\d+$/, major)
    assert_match(/^\d+$/, minor)
  end

  def test_byte_id_encode_decode_roundtrip
    pipe = Ztok::Pipeline.byte_id
    begin
      ids = pipe.encode("hi")
      assert_equal [0x68, 0x69], ids
      assert_equal "hi", pipe.decode(ids)
    ensure
      pipe.close
    end
  end

  def test_bpe_encode_decode_roundtrip
    pipe = Ztok::Pipeline.from_tiktoken(ZtokTestFixtures.tiktoken_fixture, cl100k: true)
    begin
      text = "hello world"
      ids = pipe.encode(text)
      assert_operator ids.length, :>, 0
      assert_equal text, pipe.decode(ids)
    ensure
      pipe.close
    end
  end

  def test_100_line_roundtrip_stress
    pipe = Ztok::Pipeline.from_tiktoken(ZtokTestFixtures.tiktoken_fixture, cl100k: true)
    snippets = [
      "hello world",
      "the quick brown fox",
      "foo bar baz",
      "hello there hello world",
      "the the the",
      "  the  ",
      "foo",
      "hello",
      " world",
      "bar baz",
    ]
    begin
      100.times do |i|
        line = snippets[i % snippets.length]
        ids = pipe.encode(line)
        assert_equal line, pipe.decode(ids), "round-trip failed for #{line.inspect}"
      end
    ensure
      pipe.close
    end
  end

  def test_empty_input_returns_empty_ids
    pipe = Ztok::Pipeline.byte_id
    begin
      assert_equal [], pipe.encode("")
      assert_equal "", pipe.decode([])
    ensure
      pipe.close
    end
  end

  def test_close_is_idempotent_and_postclose_ops_raise
    pipe = Ztok::Pipeline.byte_id
    pipe.close
    pipe.close # second close must be a no-op
    assert pipe.closed?
    err = assert_raises(Ztok::Error) { pipe.encode("z") }
    assert_match(/closed/i, err.message)
  end

  def test_invalid_input_raises_typed_error
    assert_raises(Ztok::InvalidInputError) do
      Ztok::Pipeline.byte_id(normalizer: 999)
    end
  end

  def test_decode_bytes_returns_raw_bytes
    pipe = Ztok::Pipeline.byte_id
    begin
      ids = pipe.encode("ab")
      raw = pipe.decode_bytes(ids)
      assert_equal Encoding::ASCII_8BIT, raw.encoding
      assert_equal "ab".b, raw
    ensure
      pipe.close
    end
  end

  def test_block_form_closes_pipeline
    pipe_ref = nil
    Ztok::Pipeline.byte_id do |pipe|
      pipe_ref = pipe
      assert_equal [ord_x], pipe.encode("x")
    end
    refute_nil pipe_ref
    assert pipe_ref.closed?, "block-form should auto-close"
  end

  def ord_x
    "x".ord
  end
end
