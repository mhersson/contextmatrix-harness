package llm

import (
	"errors"
	"io"
	"sync/atomic"
	"time"
)

// errStreamStalled marks a stream the stall deadline tore down: no bytes at
// all - data or keepalive - for the whole window. Distinguished from an
// ordinary transport error so the retry log names the real cause.
var errStreamStalled = errors.New("llm: stream stalled - no bytes within the stall timeout")

// stallReader closes its body when a single Read blocks past timeout. The
// deadline re-arms on every delivered byte, so it bounds SILENCE, not total
// stream duration - a long turn on a healthy connection streams keepalives
// and deltas and never trips it.
type stallReader struct {
	body    io.ReadCloser
	timer   *time.Timer
	timeout time.Duration
	stalled atomic.Bool
}

func newStallReader(body io.ReadCloser, timeout time.Duration) *stallReader {
	sr := &stallReader{body: body, timeout: timeout}
	sr.timer = time.AfterFunc(timeout, func() {
		sr.stalled.Store(true)
		body.Close()
	})

	return sr
}

func (sr *stallReader) Read(p []byte) (int, error) {
	n, err := sr.body.Read(p)
	if n > 0 {
		sr.timer.Reset(sr.timeout)
	}

	if err != nil && sr.stalled.Load() {
		return n, errStreamStalled
	}

	return n, err
}

func (sr *stallReader) Close() error {
	sr.timer.Stop()

	return sr.body.Close()
}
