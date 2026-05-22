# frozen_string_literal: true

module Ztok
  # Tracks the libztok release this binding was shipped with. The
  # runtime `Ztok.version` value is read straight from `ztok_version()`
  # in the loaded shared library; this constant is the gem-side
  # advertisement.
  VERSION = "1.20.0"
end
