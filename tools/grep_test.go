package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGrepTool(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "a.go"), []byte("package x\nfunc Encode() {}\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "b.txt"), []byte("nothing here\n"), 0o644))

	out, err := NewGrepTool(root).Execute(context.Background(), map[string]any{"pattern": "Encode"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "a.go")
	assert.Contains(t, out.Text, "Encode")

	out, err = NewGrepTool(root).Execute(context.Background(), map[string]any{"pattern": "no-such-token-xyz"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "no matches")
}

func TestGrepToolDashPattern(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "c.txt"), []byte("a-b-c\n"), 0o644))

	out, err := NewGrepTool(root).Execute(context.Background(), map[string]any{"pattern": "-b"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "a-b-c")
}

func TestGrepCapsOutputLines(t *testing.T) {
	root := t.TempDir()

	var b strings.Builder
	for i := range 300 {
		fmt.Fprintf(&b, "needle line %d\n", i)
	}

	require.NoError(t, os.WriteFile(filepath.Join(root, "big.txt"), []byte(b.String()), 0o644))

	g := NewGrepTool(root)
	res, err := g.Execute(context.Background(), map[string]any{"pattern": "needle"})
	require.NoError(t, err)

	lines := strings.Split(strings.TrimRight(res.Text, "\n"), "\n")
	require.Len(t, lines, grepMaxLines+1, "200 match lines + 1 truncation note")
	assert.Contains(t, lines[len(lines)-1], "100 more matching lines")
	assert.Contains(t, lines[len(lines)-1], "narrow the pattern")
}
