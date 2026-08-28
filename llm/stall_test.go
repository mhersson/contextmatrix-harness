package llm

import (
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStallReaderClosesStalledBody(t *testing.T) {
	pr, pw := io.Pipe()
	sr := newStallReader(pr, 30*time.Millisecond)

	go func() {
		_, _ = pw.Write([]byte("data: {}\n")) // one healthy read
		// then silence - never closed by the writer
	}()

	buf := make([]byte, 64)
	n, err := sr.Read(buf)
	require.NoError(t, err)
	require.Positive(t, n)

	// The next read blocks until the stall deadline closes the body.
	start := time.Now()
	_, err = sr.Read(buf)
	require.ErrorIs(t, err, errStreamStalled)
	assert.Less(t, time.Since(start), time.Second)
}

func TestStallReaderHealthyStreamUnaffected(t *testing.T) {
	pr, pw := io.Pipe()
	sr := newStallReader(pr, 50*time.Millisecond)

	go func() {
		for range 5 {
			_, _ = pw.Write([]byte(": keepalive\n"))

			time.Sleep(20 * time.Millisecond) // always under the deadline
		}

		pw.Close()
	}()

	got, err := io.ReadAll(sr)
	require.NoError(t, err)
	assert.Contains(t, string(got), "keepalive")
}
