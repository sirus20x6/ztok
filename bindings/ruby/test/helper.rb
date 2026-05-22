# frozen_string_literal: true

# Shared test scaffolding — mirrors bindings/python/tests/conftest.py
# and bindings/nodejs/test/fixture.js. Writes a tiny synthetic
# .tiktoken vocab covering all 256 single bytes plus a handful of
# merges into a per-process temp dir.

require "fileutils"
require "minitest/autorun"
require "tmpdir"

# Ruby 3.4 dropped base64 from default gems. Inline the tiny chunk we
# need (strict_encode64) so tests don't drag in another gem dep.
module ZtokB64
  def self.encode(bytes)
    [bytes].pack("m0")
  end
end

# Load the binding from lib/ relative to this file so `ruby -Ilib
# test/test_*.rb` works without bundler.
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "ztok"

module ZtokTestFixtures
  EXTRAS = [
    "he", "hel", "hell", "hello",
    " w", " wo", " wor", " worl", " world",
    "th", "the", " th", " the",
    "fo", "foo", "bar", "baz",
    " quick", " brown", " fox",
  ].freeze

  module_function

  def temp_dir
    @temp_dir ||= Dir.mktmpdir("ztok-ruby-")
  end

  def write_tiktoken_vocab(path)
    File.open(path, "w") do |f|
      rank = 0
      256.times do |b|
        f.puts "#{ZtokB64.encode([b].pack("C"))} #{rank}"
        rank += 1
      end
      EXTRAS.each do |e|
        f.puts "#{ZtokB64.encode(e)} #{rank}"
        rank += 1
      end
    end
  end

  def tiktoken_fixture
    path = File.join(temp_dir, "synthetic_cl100k.tiktoken")
    write_tiktoken_vocab(path) unless File.exist?(path)
    path
  end
end
