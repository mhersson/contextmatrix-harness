package tools

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriteToolCreatesAndOverwrites(t *testing.T) {
	root := t.TempDir()
	wt := NewWriteTool(root)

	out, err := wt.Execute(context.Background(), map[string]any{"path": "a.txt", "content": "one\ntwo\n"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "created a.txt")

	b, _ := os.ReadFile(filepath.Join(root, "a.txt"))
	assert.Equal(t, "one\ntwo\n", string(b))

	out, err = wt.Execute(context.Background(), map[string]any{"path": "a.txt", "content": "one\nTWO\n"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "-two")
	assert.Contains(t, out.Text, "+TWO")
}

func TestWriteToolRequiresPathAndContent(t *testing.T) {
	root := t.TempDir()
	_, err := NewWriteTool(root).Execute(context.Background(), map[string]any{"content": "x"})
	require.Error(t, err)
	_, err = NewWriteTool(root).Execute(context.Background(), map[string]any{"path": "a.txt"})
	require.Error(t, err)
}

func TestWriteToolNormalizesTrailingNewline(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{"missing newline gets one", "one\ntwo", "one\ntwo\n"},
		{"single newline preserved", "one\ntwo\n", "one\ntwo\n"},
		{"excess newlines trimmed to one", "one\ntwo\n\n\n", "one\ntwo\n"},
		{"empty content becomes one newline", "", "\n"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			_, err := NewWriteTool(root).Execute(context.Background(), map[string]any{
				"path": "f.txt", "content": tt.content,
			})
			require.NoError(t, err)

			b, err := os.ReadFile(filepath.Join(root, "f.txt"))
			require.NoError(t, err)
			assert.Equal(t, tt.want, string(b))
		})
	}
}

func TestWriteToolSchemaDocumentsNewlineNormalization(t *testing.T) {
	desc := NewWriteTool("/w").Schema().Function.Description
	assert.Contains(t, desc, "trailing newline",
		"models must be told about the normalization so the extra byte is expected")
}

func TestWriteCreatesParentDirs(t *testing.T) {
	root := t.TempDir()
	w := NewWriteTool(root)

	res, err := w.Execute(context.Background(), map[string]any{
		"path":    "internal/deep/nested/file.go",
		"content": "package nested",
	})
	require.NoError(t, err, "write must create missing parent directories without create_dirs")
	assert.NotEmpty(t, res.Text)

	got, err := os.ReadFile(filepath.Join(root, "internal/deep/nested/file.go"))
	require.NoError(t, err)
	assert.Equal(t, "package nested\n", string(got))
}
