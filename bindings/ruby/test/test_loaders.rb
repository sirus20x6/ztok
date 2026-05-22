# frozen_string_literal: true

require_relative "helper"

class TestLoaders < Minitest::Test
  include ZtokTestFixtures

  # --- format detection sanity checks -----------------------------------
  #
  # Post-1.18 the format detection lives in the C ABI (ztok_auto_detect).
  # We hit it through both the symbol-returning Ruby helper and the raw
  # int-code path so signature drift surfaces immediately.

  def test_detect_tiktoken
    path = ZtokTestFixtures.tiktoken_fixture
    assert_equal :tiktoken, Ztok::FFI.detect_format(path)
    assert_equal Ztok::FFI::FORMAT_TIKTOKEN, Ztok::FFI.ztok_auto_detect(path)
  end

  def test_detect_hf_json
    Dir.mktmpdir do |dir|
      p = File.join(dir, "tokenizer.json")
      File.write(p, '{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}')
      assert_equal :hf_json, Ztok::FFI.detect_format(p)
      assert_equal Ztok::FFI::FORMAT_HF_JSON, Ztok::FFI.ztok_auto_detect(p)
    end
  end

  def test_detect_sentencepiece
    sp = "/thearray/git/ztok/bench/vocabs/llama2.model"
    skip "llama2.model fixture missing" unless File.exist?(sp)
    assert_equal :sentencepiece, Ztok::FFI.detect_format(sp)
    assert_equal Ztok::FFI::FORMAT_SP_MODEL, Ztok::FFI.ztok_auto_detect(sp)
  end

  def test_detect_ztm
    Dir.mktmpdir do |dir|
      p = File.join(dir, "v.ztm")
      File.binwrite(p, "ZTM\x01" + ("\x00" * 60))
      assert_equal :ztm, Ztok::FFI.detect_format(p)
      assert_equal Ztok::FFI::FORMAT_ZTM, Ztok::FFI.ztok_auto_detect(p)
    end
  end

  def test_detect_unknown_for_missing_file
    Dir.mktmpdir do |dir|
      missing = File.join(dir, "no_such_file.bin")
      assert_equal :unknown, Ztok::FFI.detect_format(missing)
      assert_equal Ztok::FFI::FORMAT_UNKNOWN, Ztok::FFI.ztok_auto_detect(missing)
    end
  end

  # --- end-to-end loader dispatch ---------------------------------------

  def test_from_path_loads_tiktoken
    Ztok::Pipeline.from_path(ZtokTestFixtures.tiktoken_fixture) do |pipe|
      ids = pipe.encode("hello world")
      assert_equal "hello world", pipe.decode(ids)
    end
  end

  def test_from_path_loads_sentencepiece
    sp = "/thearray/git/ztok/bench/vocabs/llama2.model"
    skip "llama2.model fixture missing" unless File.exist?(sp)
    Ztok::Pipeline.from_path(sp, unk_id: 0) do |pipe|
      ids = pipe.encode("hello world")
      assert_operator ids.length, :>, 0
    end
  end

  def test_from_path_loads_ztm
    zm = "/thearray/git/ztok/bench/vocabs/tm_englishcode_32k.ztm"
    skip "tm_englishcode_32k.ztm fixture missing" unless File.exist?(zm)
    Ztok::Pipeline.from_path(zm) do |pipe|
      ids = pipe.encode("hello world")
      assert_operator ids.length, :>, 0
    end
  end

  def test_from_path_unknown_format_raises
    Dir.mktmpdir do |dir|
      p = File.join(dir, "mystery.bin")
      File.binwrite(p, "\xFF\xFE\xFD\xFC not a tokenizer")
      assert_raises(Ztok::InvalidInputError) do
        Ztok::Pipeline.from_path(p)
      end
    end
  end

  def test_explicit_from_tiktoken_no_cl100k
    Ztok::Pipeline.from_tiktoken(ZtokTestFixtures.tiktoken_fixture, cl100k: false) do |pipe|
      ids = pipe.encode("hello world")
      assert_equal "hello world", pipe.decode(ids)
    end
  end
end
