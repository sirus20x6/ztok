# frozen_string_literal: true

require_relative "helper"

# RWKV "World" tokenizer tests for the Ruby binding. Mirrors
# bindings/python/tests/test_rwkv.py. Loads the real
# rwkv_vocab_v20230424.txt fixture (skipped when absent) and checks ztok
# reproduces the canonical reference encodings, matching the in-tree gate
# in src/rwkv_world.zig.
class TestRwkv < Minitest::Test
  include ZtokTestFixtures

  # bindings/ruby/test/ -> repo root -> bench/vocabs/...
  VOCAB = File.expand_path(
    "../../../bench/vocabs/rwkv_vocab_v20230424.txt", __dir__
  )

  # Golden id sequences captured from BlinkDL's canonical reference
  # tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
  GOLDEN = [
    ["Hello, world!", [33155, 45, 40213, 34]],
    ["emoji \u{1F600}\u{1F680}\u{2728} test",
     [34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223]],
    ["0 1 2 10 99 100", [49, 284, 285, 3483, 3572, 3483, 49]],
  ].freeze

  def with_rwkv_pipeline
    skip "RWKV vocab fixture not present at #{VOCAB}" unless File.exist?(VOCAB)
    pipe = Ztok::Pipeline.from_rwkv(VOCAB)
    begin
      yield pipe
    ensure
      pipe.close
    end
  end

  def test_rwkv_matches_reference
    with_rwkv_pipeline do |pipe|
      GOLDEN.each do |text, want|
        assert_equal want, pipe.encode(text), "encode mismatch for #{text.inspect}"
      end
    end
  end

  def test_rwkv_round_trips
    with_rwkv_pipeline do |pipe|
      GOLDEN.each do |text, _want|
        ids = pipe.encode(text)
        assert_equal text, pipe.decode(ids), "round-trip failed for #{text.inspect}"
      end
    end
  end
end
