package tools

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequireStringMissingWithEcho(t *testing.T) {
	args := map[string]any{
		"path":    "/tmp/foo.txt",
		"old_str": "hello",
		"new_str": "world",
	}

	_, err := requireString(args, "edit", "nonexistent", "a required argument that is missing")
	require.Error(t, err)

	msg := err.Error()

	// The message must contain the tool name.
	assert.Contains(t, msg, `tool "edit"`)

	// The message must contain the missing key name.
	assert.Contains(t, msg, `"nonexistent"`)

	// The message must contain the received key names.
	assert.Contains(t, msg, "path")
	assert.Contains(t, msg, "old_str")
	assert.Contains(t, msg, "new_str")
}

func TestRequireStringMissingWithEchoNoOtherArgs(t *testing.T) {
	args := map[string]any{}

	_, err := requireString(args, "write", "path", "file path relative to the workspace root")
	require.Error(t, err)

	msg := err.Error()
	assert.Contains(t, msg, `tool "write"`)
	assert.Contains(t, msg, `"path"`)
	// With no other args, there should be no "received:" section.
	assert.NotContains(t, msg, "received")
}

func TestRequireStringSuccess(t *testing.T) {
	args := map[string]any{
		"path": "/tmp/foo.txt",
	}

	val, err := requireString(args, "edit", "path", "file path relative to the workspace root")
	require.NoError(t, err)
	assert.Equal(t, "/tmp/foo.txt", val)
}

func TestRequireStringWrongType(t *testing.T) {
	args := map[string]any{
		"path": 42,
	}

	_, err := requireString(args, "edit", "path", "file path relative to the workspace root")
	require.Error(t, err)

	msg := err.Error()
	assert.Contains(t, msg, `tool "edit"`)
	assert.Contains(t, msg, `"path"`)
	assert.Contains(t, msg, "must be a string")
	assert.Contains(t, msg, "path (int)")
	assert.NotContains(t, msg, "42")
}

func TestEcholessArgs(t *testing.T) {
	args := map[string]any{
		"path": "/tmp/foo.txt",
		"data": strings.Repeat("x", 1000),
	}

	result := echolessArgs(args)
	assert.Contains(t, result, "path")
	assert.Contains(t, result, "12 B") // len("/tmp/foo.txt") == 12
	assert.Contains(t, result, "1000 B")
}

func TestRequireStringByteLengthsInMessage(t *testing.T) {
	args := map[string]any{
		"old_string": strings.Repeat("x", 1421),
		"new_string": strings.Repeat("y", 893),
	}

	_, err := requireString(args, "edit", "path", "file path relative to the workspace root")
	require.Error(t, err)

	msg := err.Error()
	assert.Contains(t, msg, "1421 B")
	assert.Contains(t, msg, "893 B")
	assert.Contains(t, msg, "old_string")
	assert.Contains(t, msg, "new_string")
}
