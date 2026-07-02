package tools

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEditToolReplace(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "f.txt")
	require.NoError(t, os.WriteFile(p, []byte("hello world"), 0o644))

	et := NewEditTool(root)

	_, err := et.Execute(context.Background(), map[string]any{"path": "f.txt", "old_string": "world", "new_string": "gophers"})
	require.NoError(t, err)

	b, _ := os.ReadFile(p)
	assert.Equal(t, "hello gophers", string(b))
}

func TestEditToolAmbiguousWithoutReplaceAll(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("x x x"), 0o644))
	et := NewEditTool(root)

	_, err := et.Execute(context.Background(), map[string]any{"path": "f.txt", "old_string": "x", "new_string": "y"})
	require.Error(t, err) // appears 3 times, replace_all not set

	_, err = et.Execute(context.Background(), map[string]any{"path": "f.txt", "old_string": "x", "new_string": "y", "replace_all": true})
	require.NoError(t, err)
}

func TestEditToolNotFound(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("abc"), 0o644))
	_, err := NewEditTool(root).Execute(context.Background(), map[string]any{"path": "f.txt", "old_string": "zzz", "new_string": "y"})
	require.Error(t, err)
}

func TestEditToolRejectsEmptyOldString(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "f.txt")
	require.NoError(t, os.WriteFile(p, []byte("abc"), 0o644))

	et := NewEditTool(root)

	tests := []struct {
		name string
		args map[string]any
	}{
		{"without replace_all", map[string]any{"path": "f.txt", "old_string": "", "new_string": "X"}},
		{"with replace_all", map[string]any{"path": "f.txt", "old_string": "", "new_string": "X", "replace_all": true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := et.Execute(context.Background(), tt.args)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "old_string must be non-empty")
			assert.NotContains(t, err.Error(), "replace_all",
				"error must not steer the model toward the corrupting replace_all retry")

			b, rerr := os.ReadFile(p)
			require.NoError(t, rerr)
			assert.Equal(t, "abc", string(b), "file must be untouched")
		})
	}
}
