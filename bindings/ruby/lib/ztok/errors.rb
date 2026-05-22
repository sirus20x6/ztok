# frozen_string_literal: true

module Ztok
  # Base class for every error raised by the binding. Catch this if you
  # want to handle "anything went wrong inside ztok" without enumerating
  # the four concrete subtypes.
  class Error < StandardError; end

  # Raised when the bound library cannot be located on disk. See
  # `lib/ztok/lib.rb` for the resolution order — the message lists every
  # path that was tried.
  class LibraryNotFoundError < Error; end

  # ZTOK_ERR_OUT_OF_MEMORY (status 1).
  class OutOfMemoryError < Error; end

  # ZTOK_ERR_INVALID_INPUT (status 2). Typically a bad config value, an
  # unparseable vocab file, or a NULL/oversize argument.
  class InvalidInputError < Error; end

  # ZTOK_ERR_BUFFER_TOO_SMALL (status 3). The wrapper retries
  # transparently for `encode`/`decode`; this only escapes if the
  # encoder reports it more than 8 times in a row.
  class BufferTooSmallError < Error; end

  # ZTOK_ERR_INTERNAL (status 99) or any status code the C ABI grows in
  # the future that this binding doesn't yet know about.
  class InternalError < Error; end

  # Numeric-status -> exception class. Keep in lockstep with
  # ztok_status in include/ztok.h.
  STATUS_OK                  = 0
  STATUS_ERR_OUT_OF_MEMORY   = 1
  STATUS_ERR_INVALID_INPUT   = 2
  STATUS_ERR_BUFFER_TOO_SMALL = 3
  STATUS_ERR_INTERNAL        = 99

  ERROR_MAP = {
    STATUS_ERR_OUT_OF_MEMORY    => OutOfMemoryError,
    STATUS_ERR_INVALID_INPUT    => InvalidInputError,
    STATUS_ERR_BUFFER_TOO_SMALL => BufferTooSmallError,
    STATUS_ERR_INTERNAL         => InternalError,
  }.freeze

  module_function

  def raise_for_status(status, ctx)
    return if status == STATUS_OK
    klass = ERROR_MAP[status] || InternalError
    raise klass, "#{ctx}: ztok status #{status}"
  end
end
