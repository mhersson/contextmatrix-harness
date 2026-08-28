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
		name   string
		root   string
		reason DropReason
	}{
		{name: "empty", root: "", reason: DropReasonRelative},
		{name: "filesystem root", root: "/", reason: DropReasonFilesystemRoot},
		{name: "a prefix of the workspace", root: parent, reason: DropReasonWorkspaceParent},
		{name: "a symlink to a prefix of the workspace", root: linkToParent, reason: DropReasonWorkspaceParent},
		{name: "a symlink to the filesystem root", root: linkToSlash, reason: DropReasonFilesystemRoot},
		{name: "a relative path", root: "dep", reason: DropReasonRelative},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := sanitizeReadRoots(f.workspace, []string{tc.root})
			assert.Empty(t, result.Effective, "a root that would widen past the workspace is dropped")
			require.Len(t, result.Dropped, 1)
			assert.Equal(t, tc.root, result.Dropped[0].Root)
			assert.Equal(t, tc.reason, result.Dropped[0].Reason)

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

	result := sanitizeReadRoots(ws, []string{deps})
	assert.Empty(t, result.Effective, "an unresolvable root is dropped at construction")
	require.Equal(t, []DroppedReadRoot{{Root: deps, Reason: DropReasonNonexistent}}, result.Dropped)

	tool := NewReadTool(ws).WithReadRoots([]string{deps})
	assert.Equal(t, result, tool.ReadRoots(), "the tool exposes the same sanitize outcome the caller can query")

	// The path appears later, as a symlink to the workspace's parent.
	require.NoError(t, os.Symlink(base, deps))

	_, err := tool.Execute(context.Background(), map[string]any{"path": filepath.Join(base, "secret.txt")})
	require.Error(t, err, "a root dropped at construction cannot be revived by the filesystem")
}

// TestReadRootsAccessorReflectsConfiguration pins the observability decision
// for grep and glob too, not just read: the caller can ask each tool, after
// construction, which extra roots survived and which were dropped and why.
func TestReadRootsAccessorReflectsConfiguration(t *testing.T) {
	f := newReadRootFixture(t)

	roots := []string{f.dep, "relative"}

	grepRoots := NewGrepTool(f.workspace).WithReadRoots(roots).ReadRoots()
	globRoots := NewGlobTool(f.workspace).WithReadRoots(roots).ReadRoots()

	for _, got := range []ReadRoots{grepRoots, globRoots} {
		require.Len(t, got.Effective, 1)
		assert.Contains(t, got.Effective[0], filepath.Base(f.dep))
		require.Len(t, got.Dropped, 1)
		assert.Equal(t, "relative", got.Dropped[0].Root)
		assert.Equal(t, DropReasonRelative, got.Dropped[0].Reason)
	}
}

// TestSchemaDescriptionMentionsRootsWhenConfigured pins the discoverability
// decision: read/grep/glob's schema description names the extra read-only
// roots when WithReadRoots configured any that survived, and says nothing
// about them otherwise - the plain, unconfigured tool's description is
// unchanged.
func TestSchemaDescriptionMentionsRootsWhenConfigured(t *testing.T) {
	f := newReadRootFixture(t)

	cases := []struct {
		name       string
		plainDesc  string
		configured string
	}{
		{
			name:       "read",
			plainDesc:  NewReadTool(f.workspace).Schema().Function.Description,
			configured: NewReadTool(f.workspace).WithReadRoots([]string{f.dep}).Schema().Function.Description,
		},
		{
			name:       "grep",
			plainDesc:  NewGrepTool(f.workspace).Schema().Function.Description,
			configured: NewGrepTool(f.workspace).WithReadRoots([]string{f.dep}).Schema().Function.Description,
		},
		{
			name:       "glob",
			plainDesc:  NewGlobTool(f.workspace).Schema().Function.Description,
			configured: NewGlobTool(f.workspace).WithReadRoots([]string{f.dep}).Schema().Function.Description,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.NotContains(t, tc.plainDesc, f.dep, "an unconfigured tool's schema says nothing about extra roots")
			assert.Contains(t, tc.configured, f.dep, "a configured tool's schema names the permitted root")
		})
	}
}
