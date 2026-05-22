# frozen_string_literal: true

require_relative "helper"

class TestBatch < Minitest::Test
  include ZtokTestFixtures

  def setup
    @pipe = Ztok::Pipeline.from_tiktoken(ZtokTestFixtures.tiktoken_fixture, cl100k: true)
  end

  def teardown
    @pipe&.close
  end

  def test_workers_resolves_auto
    pool = Ztok::BatchPool.new(workers: 0)
    begin
      assert_operator pool.workers, :>=, 1
    ensure
      pool.close
    end
  end

  def test_explicit_worker_count
    pool = Ztok::BatchPool.new(workers: 3)
    begin
      assert_equal 3, pool.workers
    ensure
      pool.close
    end
  end

  def test_closed_pool_raises
    pool = Ztok::BatchPool.new(workers: 2)
    pool.close
    assert pool.closed?
    assert_raises(Ztok::Error) { pool.workers }
  end

  def test_batch_encode_matches_single_encode
    inputs = ["hello world", " the quick brown fox", "foo bar baz"]
    expected = inputs.map { |s| @pipe.encode(s) }
    Ztok::BatchPool.open(workers: 4) do |pool|
      got = @pipe.encode_batch(pool, inputs)
      assert_equal expected, got
    end
  end

  def test_batch_encode_1000_strings_no_leaks
    inputs = Array.new(1000, "hello world")
    results = nil
    Ztok::BatchPool.open(workers: 8) do |pool|
      results = @pipe.encode_batch(pool, inputs)
    end
    assert_equal 1000, results.length
    first = results.first
    assert(results.all? { |r| r == first })

    # The C-owned id buffers were freed inside encode_batch (via
    # ztok_ids_free). Force a couple of GC cycles to flush our Ruby
    # finalizers and confirm we don't crash or leak.
    results = nil
    3.times { GC.start }
  end

  def test_batch_encode_empty_inputs
    Ztok::BatchPool.open(workers: 2) do |pool|
      assert_equal [], @pipe.encode_batch(pool, [])
    end
  end

  def test_batch_encode_with_empty_string
    inputs = ["hello", "", "world"]
    Ztok::BatchPool.open(workers: 2) do |pool|
      results = @pipe.encode_batch(pool, inputs)
      assert_equal 3, results.length
      assert_equal [], results[1] # empty input -> empty ids
      assert_equal @pipe.encode("hello"), results[0]
      assert_equal @pipe.encode("world"), results[2]
    end
  end

  def test_block_form_closes_pool
    captured = nil
    Ztok::BatchPool.open(workers: 2) do |pool|
      captured = pool
      assert_operator pool.workers, :>=, 1
    end
    assert captured.closed?, "block-form must close pool on exit"
  end

  def test_finalizer_releases_pool_on_gc
    # Drop without explicit close. The ObjectSpace finalizer should
    # tear down the C worker pool when Ruby collects the wrapper. We
    # don't have a portable "did the finalizer fire?" probe in MRI, but
    # we exercise the path so leaks surface as crashes under valgrind.
    _pool = Ztok::BatchPool.new(workers: 2)
    _pool = nil
    3.times { GC.start }
  end
end
