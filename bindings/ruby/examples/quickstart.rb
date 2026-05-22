# frozen_string_literal: true

# Self-contained quickstart: load a tokenizer, encode, decode.
#
# Run from the repo root after `zig build`:
#
#   ZTOK_LIB_PATH=zig-out/lib/libztok.so \
#     ruby -Ibindings/ruby/lib bindings/ruby/examples/quickstart.rb [path]
#
# Without an argument we synthesize a tiny .tiktoken file in the system
# temp dir so the example always exits 0 with meaningful output. Pass a
# real .tiktoken / tokenizer.json / .model / .ztm path to use it
# directly — Pipeline.from_path auto-detects the format via the C ABI.

require "tmpdir"
require "ztok"

# Ruby 3.4 dropped base64 from default gems; the inline helper keeps
# this example zero-extra-deps.
def b64(bytes)
  [bytes].pack("m0")
end

def synthesize_tiktoken
  extras = %w[
    he hel hell hello
    \ w \ wo \ wor \ worl \ world
  ].map { |s| s.gsub("\\ ", " ") }
  path = File.join(Dir.tmpdir, "ztok-quickstart-ruby.tiktoken")
  File.open(path, "w") do |f|
    rank = 0
    256.times do |b|
      f.puts "#{b64([b].pack("C"))} #{rank}"
      rank += 1
    end
    extras.each do |e|
      f.puts "#{b64(e)} #{rank}"
      rank += 1
    end
  end
  path
end

path = ARGV[0] || synthesize_tiktoken
pipe = Ztok::Pipeline.from_path(path)
begin
  ids = pipe.encode("hello world")
  puts "ztok #{Ztok.version}: #{ids.length} ids -> #{ids.inspect}"
  puts "decoded: #{pipe.decode(ids).inspect}"

  Ztok::BatchPool.open(workers: 4) do |pool|
    batch = pipe.encode_batch(pool, ["hello world", "foo bar", "baz"])
    puts "batch ids: #{batch.map(&:length).inspect} tokens per input"
  end

  print "streamed: "
  pipe.encode_stream("hello world hello world", chunk_size: 8) do |chunk|
    print "#{chunk.inspect} "
  end
  puts
ensure
  pipe.close
end
