# frozen_string_literal: true

# PRNG-driven round-trip fuzz harness for the ztok Ruby binding.
#
# Mirrors fuzz/encode_decode.zig in shape: a deterministic Random(SEED)
# generates byte inputs; encode-then-decode must round-trip exactly.
# The pipeline is byte_id (identity normalizer + identity pre-tok +
# byte_id model + concat decoder) — by construction each input byte
# maps to one id and decoding concatenates them back, so for ANY byte
# sequence we must have decode_bytes(encode(x)) == x.
#
# Skips cleanly if libztok cannot be loaded (mirrors the
# fixture-missing skip pattern in test_loaders.rb).

require_relative "helper"

begin
  # Force the loader to resolve the shared library up front so a
  # missing libztok skips the whole suite cleanly instead of erroring
  # mid-test.
  Ztok.version
rescue Ztok::LibraryNotFoundError => e
  $stderr.puts "skipping fuzz suite: #{e.message}"
  Minitest::Test.send(:define_method, :skip_all_libztok_missing) do
    skip "libztok not available: #{e.message}"
  end
end

class TestFuzz < Minitest::Test
  # Deterministic seed: same value as the Python/Node harnesses so
  # failures at a given iteration are cross-language reproducible. The
  # fixed seed may be overridden per-run via FUZZ_SEED env (hex like
  # "0xdeadbeef" or decimal); ZTOK_FUZZ_ITERS scales the iteration count
  # for nightly fuzz workflows.
  DEFAULT_SEED = 0xFEEDB0B
  DEFAULT_ITERATIONS = 1000
  MAX_LEN = 256

  def self.env_int(name, fallback)
    raw = ENV[name]
    return fallback if raw.nil? || raw.empty?

    Integer(raw, raw.start_with?("0x", "0X") ? 16 : 10)
  rescue ArgumentError
    fallback
  end

  SEED = env_int("FUZZ_SEED", DEFAULT_SEED)
  ITERATIONS = env_int("ZTOK_FUZZ_ITERS", DEFAULT_ITERATIONS)

  def test_byte_id_roundtrip_fuzz_1000_iterations
    skip "libztok not available" unless Ztok.respond_to?(:version) && lib_ok?

    rng = Random.new(SEED)
    failures = []

    pipe = Ztok::Pipeline.byte_id
    begin
      ITERATIONS.times do |i|
        data = random_bytes(rng, MAX_LEN)
        ids = pipe.encode(data)
        # byte_id maps 1:1 — id count must equal input byte length.
        assert_equal(
          data.bytesize, ids.length,
          "iter #{i}: byte_id produced #{ids.length} ids for #{data.bytesize} bytes"
        )
        roundtrip = pipe.decode_bytes(ids)
        next if roundtrip == data

        failures << [i, data.unpack1("H*"), roundtrip.unpack1("H*")]
        break if failures.length >= 5 # don't flood
      end
    ensure
      # Release the native handle deterministically; the finalizer would
      # eventually free it on GC, but explicit close keeps the test
      # isolated and matches the rest of the suite's convention.
      pipe.close
    end

    return if failures.empty?

    msg = failures.map { |i, inp, out| "  iter #{i}: in=#{inp} out=#{out}" }.join("\n")
    flunk("byte_id round-trip mismatches:\n#{msg}")
  end

  private

  def random_bytes(rng, max_len)
    n = rng.rand(0..max_len)
    return String.new(encoding: Encoding::BINARY) if n.zero?

    rng.bytes(n) # binary-encoded String of n random bytes
  end

  def lib_ok?
    Ztok.version
    true
  rescue Ztok::LibraryNotFoundError
    false
  end
end
