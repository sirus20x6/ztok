# frozen_string_literal: true

require_relative "helper"

# Tests for Pipeline#encode_with_overlays (ztok_encode_with_overlays).
# Mirrors bindings/python/tests/test_overlays.py.
class TestOverlays < Minitest::Test
  include ZtokTestFixtures

  def bpe_pipeline
    Ztok::Pipeline.from_tiktoken(ZtokTestFixtures.tiktoken_fixture, cl100k: true)
  end

  def test_ids_match_plain_encode
    pipe = bpe_pipeline
    begin
      text = "hello world"
      plain = pipe.encode(text)
      ids, overlays = pipe.encode_with_overlays(
        text, [Ztok::FFI::OVERLAY_BYTE_START, Ztok::FFI::OVERLAY_BYTE_END]
      )
      assert_equal plain, ids, "requesting overlays must not change tokenization"
      assert_equal [Ztok::FFI::OVERLAY_BYTE_START, Ztok::FFI::OVERLAY_BYTE_END].sort,
                   overlays.keys.sort
    ensure
      pipe.close
    end
  end

  def test_channel_lengths_equal_ids
    pipe = bpe_pipeline
    begin
      ids, overlays = pipe.encode_with_overlays(
        "the quick brown fox",
        [
          Ztok::FFI::OVERLAY_BYTE_START,
          Ztok::FFI::OVERLAY_BYTE_END,
          Ztok::FFI::OVERLAY_BOUNDARY,
          Ztok::FFI::OVERLAY_PROVENANCE,
        ]
      )
      overlays.each do |kind, values|
        assert_equal ids.length, values.length, "channel #{kind} length mismatch"
      end
    ensure
      pipe.close
    end
  end

  def test_byte_spans_are_sensible
    pipe = bpe_pipeline
    begin
      text = "hello world"
      _ids, overlays = pipe.encode_with_overlays(
        text, [Ztok::FFI::OVERLAY_BYTE_START, Ztok::FFI::OVERLAY_BYTE_END]
      )
      starts = overlays[Ztok::FFI::OVERLAY_BYTE_START]
      ends = overlays[Ztok::FFI::OVERLAY_BYTE_END]
      n = text.bytesize
      refute_empty starts
      starts.zip(ends).each do |s, e|
        assert s < e && e <= n, "bad span (#{s}, #{e}) for input of #{n} bytes"
      end
      assert_equal 0, starts.first
      assert_equal n, ends.last
      # Spans tile left-to-right: each picks up where the previous ended.
      ends.each_with_index do |prev_end, i|
        next if i + 1 >= starts.length

        assert_equal prev_end, starts[i + 1], "span gap at #{i}"
      end
    ensure
      pipe.close
    end
  end

  def test_byte_id_single_byte_spans
    pipe = Ztok::Pipeline.byte_id
    begin
      ids, overlays = pipe.encode_with_overlays(
        "hi", [Ztok::FFI::OVERLAY_BYTE_START, Ztok::FFI::OVERLAY_BYTE_END]
      )
      assert_equal [0x68, 0x69], ids
      assert_equal [0, 1], overlays[Ztok::FFI::OVERLAY_BYTE_START]
      assert_equal [1, 2], overlays[Ztok::FFI::OVERLAY_BYTE_END]
    ensure
      pipe.close
    end
  end

  def test_opcode_domain_channel_is_all_zero
    pipe = bpe_pipeline
    begin
      ids, overlays = pipe.encode_with_overlays("hello world", [Ztok::FFI::OVERLAY_OPCODE])
      opcode = overlays[Ztok::FFI::OVERLAY_OPCODE]
      assert_equal ids.length, opcode.length
      assert(opcode.all?(&:zero?), "OPCODE must be zero-filled without a domain plugin")
    ensure
      pipe.close
    end
  end

  def test_empty_input_returns_empty_channels
    pipe = Ztok::Pipeline.byte_id
    begin
      ids, overlays = pipe.encode_with_overlays(
        "", [Ztok::FFI::OVERLAY_BYTE_START, Ztok::FFI::OVERLAY_OPCODE]
      )
      assert_equal [], ids
      assert_equal({ Ztok::FFI::OVERLAY_BYTE_START => [], Ztok::FFI::OVERLAY_OPCODE => [] }, overlays)
    ensure
      pipe.close
    end
  end

  def test_no_channels_returns_just_ids
    pipe = bpe_pipeline
    begin
      ids, overlays = pipe.encode_with_overlays("hello world", [])
      assert_equal pipe.encode("hello world"), ids
      assert_equal({}, overlays)
    ensure
      pipe.close
    end
  end

  def test_duplicate_kinds_raises
    pipe = Ztok::Pipeline.byte_id
    begin
      assert_raises(Ztok::InvalidInputError) do
        pipe.encode_with_overlays("hi", [Ztok::FFI::OVERLAY_BYTE_START, Ztok::FFI::OVERLAY_BYTE_START])
      end
    ensure
      pipe.close
    end
  end
end
