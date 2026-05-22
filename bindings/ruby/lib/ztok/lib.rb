# frozen_string_literal: true

require "rbconfig"

module Ztok
  # libztok loader. Mirrors bindings/python/ztok/_lib.py and
  # bindings/nodejs/lib.js so the three bindings agree on where the
  # shared library lives.
  #
  # Resolution order:
  #   1. ENV["ZTOK_LIB_PATH"]                 (explicit override)
  #   2. <gem>/ext/libztok.{so,dylib,dll}     (native gem install)
  #   3. <gem>/../../../../zig-out/lib/libztok.{so,dylib,dll}  (in-tree dev)
  #   4. Standard system paths: /usr/local/lib, /usr/lib, /usr/lib64,
  #      /opt/homebrew/lib on macOS.
  #
  # Raises Ztok::LibraryNotFoundError with the full list of tried paths
  # if nothing works.
  module Lib
    class << self
      def shared_lib_basename
        case RbConfig::CONFIG["host_os"]
        when /darwin/ then "libztok.dylib"
        when /mswin|mingw|cygwin/ then "ztok.dll"
        else "libztok.so"
        end
      end

      def candidate_paths
        base = shared_lib_basename
        here = __dir__ # bindings/ruby/lib/ztok
        paths = []

        # 2. Gem-installed native ext layout.
        paths << File.join(here, "..", "..", "ext", base)
        paths << File.join(here, "..", "..", base)

        # 3. In-tree dev layout: bindings/ruby/lib/ztok ->
        #    repo root -> zig-out/lib/<base>
        paths << File.join(here, "..", "..", "..", "..", "zig-out", "lib", base)

        # 4. Standard system search.
        case RbConfig::CONFIG["host_os"]
        when /linux/
          paths << "/usr/local/lib/#{base}"
          paths << "/usr/lib/#{base}"
          paths << "/usr/lib64/#{base}"
        when /darwin/
          paths << "/usr/local/lib/#{base}"
          paths << "/opt/homebrew/lib/#{base}"
        end

        paths.map { |p| File.expand_path(p) }
      end

      # Returns the absolute path to libztok, raising
      # Ztok::LibraryNotFoundError with a useful message on failure.
      def resolve
        override = ENV["ZTOK_LIB_PATH"]
        if override && !override.empty?
          unless File.exist?(override)
            raise LibraryNotFoundError,
                  "ZTOK_LIB_PATH=#{override.inspect} does not point to an existing file."
          end
          return File.expand_path(override)
        end

        tried = []
        candidate_paths.each do |p|
          tried << p
          return p if File.exist?(p)
        end

        raise LibraryNotFoundError, <<~MSG
          Could not locate libztok. Build it with `zig build` and either:
            - install it system-wide (e.g. cp zig-out/lib/libztok.so /usr/local/lib/),
            - place it next to the ztok Ruby gem (under ext/), or
            - set ZTOK_LIB_PATH=/absolute/path/to/libztok.so.
          Tried: #{tried.join("; ")}
        MSG
      end
    end
  end
end
