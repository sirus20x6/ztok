# frozen_string_literal: true

require_relative "helper"

# Mistral Tekken tokenizer tests for the Ruby binding. Loads the real
# mistral_nemo_tekken.json fixture (skipped when absent) and checks ztok
# reproduces the golden ids verified against mistral_common 1.8.6.
class TestTekken < Minitest::Test
  include ZtokTestFixtures

  # bindings/ruby/test/ -> repo root -> bench/vocabs/...
  VOCAB = File.expand_path(
    "../../../bench/vocabs/mistral_nemo_tekken.json", __dir__
  )

  # Golden id sequences verified against mistral_common 1.8.6 on
  # bench/vocabs/mistral_nemo_tekken.json.
  GOLDEN = [
    ["Hello, world!", [22177, 1044, 4304, 1033]],
    ["The quick brown fox", [1784, 7586, 22980, 94137]],
    [" and the", [1321, 1278]],
  ].freeze

  def with_tekken_pipeline
    skip "Tekken vocab fixture not present at #{VOCAB}" unless File.exist?(VOCAB)
    pipe = Ztok::Pipeline.from_tekken(VOCAB)
    begin
      yield pipe
    ensure
      pipe.close
    end
  end

  def test_tekken_matches_reference
    with_tekken_pipeline do |pipe|
      GOLDEN.each do |text, want|
        assert_equal want, pipe.encode(text), "encode mismatch for #{text.inspect}"
      end
    end
  end

  def test_tekken_auto_detects_via_from_path
    skip "Tekken vocab fixture not present at #{VOCAB}" unless File.exist?(VOCAB)
    assert_equal :tekken, Ztok::FFI.detect_format(VOCAB)
    pipe = Ztok::Pipeline.from_path(VOCAB)
    begin
      assert_equal [22177, 1044, 4304, 1033], pipe.encode("Hello, world!")
    ensure
      pipe.close
    end
  end
end
