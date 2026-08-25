package tools

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// readRootFixture lays out a workspace and a separate dependency tree, each
// holding one file with the same searchable contents.
type readRootFixture struct {
	workspace string
	dep       string
	depFile   string
	outside   string
}

func newReadRootFixture(t *testing.T) readRootFixture {
	t.Helper()

	base := t.TempDir()
	f := readRootFixture{
		workspace: filepath.Join(base, "workspace"),
		dep:       filepath.Join(base, "dep"),
		outside:   filepath.Join(base, "elsewhere"),
	}

	for _, d := range []string{f.workspace, f.dep, f.outside} {
		require.NoError(t, os.MkdirAll(d, 0o755))
	}

	f.depFile = filepath.Join(f.dep, "api.go")
	require.NoError(t, os.WriteFile(f.depFile, []byte("package dep // NEEDLE\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(f.outside, "secret.go"), []byte("package elsewhere // NEEDLE\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(f.workspace, "main.go"), []byte("package main // NEEDLE\n"), 0o644))

	return f
}

// TestExtraReadRootsPerTool is the boundary this card exists to draw: read
// anywhere the caller permits, write only in the workspace.
func TestExtraReadRootsPerTool(t *testing.T) {
	f := newReadRootFixture(t)
	outsideFile := filepath.Join(f.outside, "secret.go")

	tools := []struct {
		name      string
		exec      func(path string) error
		readsDeps bool // configured with the dependency tree as an extra read root
		needsRg   bool
	}{
		{
			name:      "read",
			readsDeps: true,
			exec: func(path string) error {
				_, err := NewReadTool(f.workspace).WithReadRoots([]string{f.dep}).
					Execute(context.Background(), map[string]any{"path": path})

				return err
			},
		},
		{
			name:      "grep",
			readsDeps: true,
			needsRg:   true,
			exec: func(path string) error {
				_, err := NewGrepTool(f.workspace).WithReadRoots([]string{f.dep}).
					Execute(context.Background(), map[string]any{"pattern": "NEEDLE", "path": path})

				return err
			},
		},
		{
			name:      "glob",
			readsDeps: true,
			needsRg:   true,
			exec: func(path string) error {
				_, err := NewGlobTool(f.workspace).WithReadRoots([]string{f.dep}).
					Execute(context.Background(), map[string]any{"pattern": "*.go", "path": path})

				return err
			},
		},
		{
			name: "edit",
			exec: func(path string) error {
				_, err := NewEditTool(f.workspace).Execute(context.Background(), map[string]any{
					"path": path, "old_string": "package dep", "new_string": "package hacked",
				})

				return err
			},
		},
		{
			name: "write",
			exec: func(path string) error {
				_, err := NewWriteTool(f.workspace).Execute(context.Background(), map[string]any{
					"path": path, "content": "overwritten",
				})

				return err
			},
		},
	}

	for _, tc := range tools {
		t.Run(tc.name+" in an extra read root", func(t *testing.T) {
			if tc.needsRg {
				if _, err := exec.LookPath("rg"); err != nil {
					t.Skip("ripgrep (rg) not installed")
				}
			}

			target := f.depFile
			if tc.name == "glob" || tc.name == "grep" {
				target = f.dep
			}

			err := tc.exec(target)
			if tc.readsDeps {
				assert.NoError(t, err, "a configured read root must resolve")

				return
			}

			require.Error(t, err, "writes stay confined to the workspace")
			assert.Contains(t, err.Error(), target)
		})

		t.Run(tc.name+" outside every root", func(t *testing.T) {
			if tc.needsRg {
				if _, err := exec.LookPath("rg"); err != nil {
					t.Skip("ripgrep (rg) not installed")
				}
			}

			target := outsideFile
			if tc.name == "glob" || tc.name == "grep" {
				target = f.outside
			}

			err := tc.exec(target)
			require.Error(t, err, "a path in no permitted root is refused")
			// A jail refusal specifically, not merely the path: edit also fails with
			// "old_string not found in <path>", so Contains(path) passes even with
			// containment removed entirely. Single-root tools say "escapes workspace
			// root"; a tool carrying extra roots names them all.
			assert.Regexp(t, `escapes workspace root|outside the workspace and every permitted read-only root`, err.Error())
		})
	}
}

func TestExtraReadRootsRejectedAtConstruction(t *testing.T) {
	f := newReadRootFixture(t)
	parent := filepath.Dir(f.workspace)

	// A root that only reaches the workspace's parent through a symlink: the
	// containment check resolves symlinks, so this gate must resolve them too.
	linkToParent := filepath.Join(parent, "dep-link")
	require.NoError(t, os.Symlink(parent, linkToParent))

	linkToSlash := filepath.Join(parent, "root-link")
	require.NoError(t, os.Symlink("/", linkToSlash))

	tests := []struct {
		name string
		root string
	}{
		{name: "empty", root: ""},
		{name: "filesystem root", root: "/"},
		{name: "a prefix of the workspace", root: parent},
		{name: "a symlink to a prefix of the workspace", root: linkToParent},
		{name: "a symlink to the filesystem root", root: linkToSlash},
		{name: "a relative path", root: "dep"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Empty(t, sanitizeReadRoots(f.workspace, []string{tc.root}),
				"a root that would widen past the workspace is dropped")

			// And the tool built with it still refuses a path it would have opened.
			_, err := NewReadTool(f.workspace).WithReadRoots([]string{tc.root}).
				Execute(context.Background(), map[string]any{"path": filepath.Join(f.outside, "secret.go")})
			require.Error(t, err)
		})
	}
}

func TestNoExtraReadRootsIsUnchanged(t *testing.T) {
	f := newReadRootFixture(t)

	// Same tool, with and without an empty WithReadRoots call: identical answers,
	// including the error text for a path outside the workspace.
	plain := NewReadTool(f.workspace)
	configured := NewReadTool(f.workspace).WithReadRoots(nil)

	inside := filepath.Join(f.workspace, "main.go")

	got1, err1 := plain.Execute(context.Background(), map[string]any{"path": inside})
	require.NoError(t, err1)

	got2, err2 := configured.Execute(context.Background(), map[string]any{"path": inside})
	require.NoError(t, err2)
	assert.Equal(t, got1.Text, got2.Text)

	_, err1 = plain.Execute(context.Background(), map[string]any{"path": f.depFile})
	_, err2 = configured.Execute(context.Background(), map[string]any{"path": f.depFile})

	require.Error(t, err1)
	require.Error(t, err2)
	assert.Equal(t, err1.Error(), err2.Error(), "the refusal message is unchanged")
	assert.Contains(t, err1.Error(), "escapes workspace root", "the single-root message is byte-identical")
}

// TestUnresolvableReadRootIsDropped pins that a root which cannot be resolved
// when the tool is built grants nothing later. A worker that builds its tools
// before the dependency tree is mounted would otherwise carry a root that is
// judged only by a per-call re-resolution - and the path that appears later can
// point anywhere, including at the workspace's own parent.
func TestUnresolvableReadRootIsDropped(t *testing.T) {
	base := t.TempDir()
	ws := filepath.Join(base, "workspace")
	require.NoError(t, os.MkdirAll(ws, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(base, "secret.txt"), []byte("TOPSECRET"), 0o644))

	deps := filepath.Join(base, "deps") // not mounted yet

	assert.Empty(t, sanitizeReadRoots(ws, []string{deps}), "an unresolvable root is dropped at construction")

	tool := NewReadTool(ws).WithReadRoots([]string{deps})

	// The path appears later, as a symlink to the workspace's parent.
	require.NoError(t, os.Symlink(base, deps))

	_, err := tool.Execute(context.Background(), map[string]any{"path": filepath.Join(base, "secret.txt")})
	require.Error(t, err, "a root dropped at construction cannot be revived by the filesystem")
}
