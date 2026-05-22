package ztok

import "fmt"

// StatusError is the base error type returned by every ztok call. Each
// status-specific error type (OutOfMemoryError, InvalidInputError, etc.)
// embeds *StatusError, so callers can:
//
//	var oom *ztok.OutOfMemoryError
//	if errors.As(err, &oom) { ... oom.Status ... }
type StatusError struct {
	// Status mirrors the C-side ztok_status integer.
	Status int
	// Op is the calling site ("ztok_encode", "ztok_pipeline_new", ...).
	Op string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s: ztok status %d", e.Op, e.Status)
}

// OutOfMemoryError maps ZTOK_ERR_OUT_OF_MEMORY (status 1).
type OutOfMemoryError struct{ *StatusError }

// InvalidInputError maps ZTOK_ERR_INVALID_INPUT (status 2).
type InvalidInputError struct{ *StatusError }

// BufferTooSmallError maps ZTOK_ERR_BUFFER_TOO_SMALL (status 3). Most
// callers should never see this — the binding grows buffers
// automatically.
type BufferTooSmallError struct{ *StatusError }

// InternalError maps ZTOK_ERR_INTERNAL (status 99) or any unknown
// non-zero status code.
type InternalError struct{ *StatusError }

// LibraryLoadError is returned when libztok itself can't be located /
// dlopened. It's distinct because no pipeline exists at the failure
// point — there is no C status to report.
type LibraryLoadError struct {
	Path   string
	Reason string
}

func (e *LibraryLoadError) Error() string {
	if e.Path == "" {
		return fmt.Sprintf("ztok: library load failed: %s", e.Reason)
	}
	return fmt.Sprintf("ztok: failed to load %s: %s", e.Path, e.Reason)
}

// statusToError maps a non-OK C status to the typed Go error. Returns
// nil when status is ZTOK_OK.
func statusToError(status int, op string) error {
	if status == int(cStatusOK) {
		return nil
	}
	base := &StatusError{Status: status, Op: op}
	switch status {
	case int(cStatusOutOfMemory):
		return &OutOfMemoryError{StatusError: base}
	case int(cStatusInvalidInput):
		return &InvalidInputError{StatusError: base}
	case int(cStatusBufferTooSmall):
		return &BufferTooSmallError{StatusError: base}
	default:
		return &InternalError{StatusError: base}
	}
}
