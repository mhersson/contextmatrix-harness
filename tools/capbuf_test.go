package tools

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCapWriter(t *testing.T) {
	w := &capWriter{limit: 10}
	n, err := w.Write([]byte("0123456789ABCDEF")) // 16 bytes into a 10-byte cap
	require.NoError(t, err)
	assert.Equal(t, 16, n) // reports full consumption so the pipe copier keeps draining
	assert.Equal(t, "0123456789", w.buf.String())
	assert.Contains(t, w.String(), "[output truncated]")
}

func TestCapWriterExactFit(t *testing.T) {
	w := &capWriter{limit: 10}
	n, err := w.Write([]byte("0123456789")) // exactly fills the cap, nothing dropped
	require.NoError(t, err)
	assert.Equal(t, 10, n)
	assert.Equal(t, "0123456789", w.buf.String())
	assert.Equal(t, "0123456789", w.String())
	assert.NotContains(t, w.String(), "[output truncated]")
}
