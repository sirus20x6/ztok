# frozen_string_literal: true

require_relative "helper"

# Tokenizer-fingerprint tests for the Ruby binding. Mirror the
# rust/dotnet/java fingerprint tests: 32-byte length + determinism, plus
# a cross-binding golden value for the byte_id pipeline.
class TestFingerprint < Minitest::Test
  # Golden fingerprint for the default byte_id pipeline. Computed directly
  # from libztok's ztok_fingerprint and shared across the
  # rust/dotnet/java/python/nodejs/go bindings to confirm agreement.
  GOLDEN_BYTE_ID =
    "201ecf86554b5a970471e0189d7e78dc2c3df24519f7d5ebb0caacc86701e77c"

  def with_byte_pipeline
    pipe = Ztok::Pipeline.byte_id
    begin
      yield pipe
    ensure
      pipe.close
    end
  end

  def test_fingerprint_is_32_bytes
    with_byte_pipeline do |pipe|
      fp = pipe.fingerprint
      assert_instance_of Ztok::Fingerprint, fp
      assert_equal 32, fp.bytes.bytesize
      assert_equal 64, fp.hex.length
      refute fp.bytes.bytes.all?(&:zero?), "fingerprint must not be all zeros"
    end
  end

  def test_fingerprint_is_deterministic
    a = Ztok::Pipeline.byte_id
    b = Ztok::Pipeline.byte_id
    begin
      assert_equal a.fingerprint, b.fingerprint
      assert_equal a.fingerprint.hex, b.fingerprint.hex
    ensure
      a.close
      b.close
    end
  end

  def test_fingerprint_golden_byte_id
    with_byte_pipeline do |pipe|
      assert_equal GOLDEN_BYTE_ID, pipe.fingerprint.hex
    end
  end
end
