# frozen_string_literal: true

require_relative "lib/ztok/version"

Gem::Specification.new do |spec|
  spec.name          = "ztok"
  spec.version       = Ztok::VERSION
  spec.authors       = ["ztok contributors"]
  spec.summary       = "Ruby bindings for the ztok tokenizer library"
  spec.description   = <<~DESC
    Thin ffi-gem wrapper around libztok (Zig 0.16). Mirrors the C ABI:
    Pipeline, BatchPool, StreamEncoder. Auto-detects .tiktoken / HF
    tokenizer.json / SentencePiece .model / TokenMonster .ztm via the
    C ABI's ztok_auto_detect. Requires libztok to be built separately
    (via `zig build`) — no native compilation in the gem install path.
  DESC
  spec.homepage      = "https://github.com/sirus20x6/ztok"
  spec.license       = "AGPL-3.0-only"
  spec.required_ruby_version = ">= 3.0.0"

  spec.files = Dir[
    "lib/**/*.rb",
    "README.md",
    "ztok.gemspec",
  ]
  spec.require_paths = ["lib"]

  spec.add_dependency "ffi", "~> 1.15"

  spec.metadata = {
    "source_code_uri" => "https://github.com/sirus20x6/ztok",
    "bug_tracker_uri" => "https://github.com/sirus20x6/ztok/issues",
    "rubygems_mfa_required" => "true",
  }
end
