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

func TestGrepEmptyWithBRE(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "test.txt"), []byte("hello world\n"), 0o644))

	// Pattern a\|b uses GNU BRE alternation; in ripgrep \| matches a literal pipe.
	out, err := NewGrepTool(root).Execute(context.Background(), map[string]any{"pattern": `a\|b`})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "no matches")
	assert.Contains(t, out.Text, "ripgrep (Rust regex)")
	assert.Contains(t, out.Text, "alternation")
}

func TestGrepEmptyWithValidRustEscapeNoBREHint(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "test.txt"), []byte("hello world\n"), 0o644))

	// \( \) \{ \+ are all valid Rust-regex literal escapes (matching a literal
	// paren/brace/plus) - a model correctly escaping a literal paren that finds
	// no matches must not be told grep's syntax works differently. Only \|
	// (the observed failure class) should trigger the note.
	for _, pattern := range []string{`a\(b`, `a\)b`, `a\{b`, `a\+b`} {
		out, err := NewGrepTool(root).Execute(context.Background(), map[string]any{"pattern": pattern})
		require.NoError(t, err)
		assert.Equal(t, "no matches", out.Text, "pattern %q should not trigger the BRE note", pattern)
	}
}

func TestGrepStripDoesNotCorruptSecondaryRootInfixPath(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

	tmp := t.TempDir()
	workspace := filepath.Join(tmp, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	// Build a secondary root whose absolute path embeds the workspace's
	// absolute path as an INFIX (not as its own leading prefix): e.g. a mirror
	// or backup tree that happens to reproduce the workspace's path
	// underneath it. strings.TrimPrefix on the separator keeps filepath.Join
	// from collapsing the embedded leading slash.
	secondary := filepath.Join(tmp, "mirror", strings.TrimPrefix(workspace, string(filepath.Separator)))
	require.NoError(t, os.MkdirAll(secondary, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(secondary, "b.go"), []byte("needle in secondary\n"), 0o644))

	g := NewGrepTool(workspace).WithReadRoots([]string{secondary})

	out, err := g.Execute(context.Background(), map[string]any{"pattern": "needle", "path": secondary})
	require.NoError(t, err)
	// The matched line's own path field is under `secondary`, which does not
	// start with the workspace prefix - a whole-output ReplaceAll of
	// workspace+"/" would still corrupt the embedded infix occurrence,
	// mashing "mirror" directly into "b.go" with no separator between them.
	assert.Contains(t, out.Text, filepath.Join(secondary, "b.go"))
	assert.NotContains(t, out.Text, "mirrorb.go")
}

func TestGrepEmptyPlain(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "test.txt"), []byte("hello world\n"), 0o644))

	out, err := NewGrepTool(root).Execute(context.Background(), map[string]any{"pattern": "zzzz"})
	require.NoError(t, err)
	assert.Equal(t, "no matches", out.Text)
}

func TestGrepCapsOutputLines(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep (rg) not installed")
	}

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
