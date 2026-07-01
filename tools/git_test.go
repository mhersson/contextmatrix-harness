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

func TestGitToolReadOnly(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}

	root := t.TempDir()

	for _, c := range [][]string{{"init"}, {"config", "user.email", "a@b.c"}, {"config", "user.name", "x"}} {
		cmd := exec.Command("git", c...)
		cmd.Dir = root
		require.NoError(t, cmd.Run())
	}

	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("hi\n"), 0o644))

	gt := NewGitTool(root)
	out, err := gt.Execute(context.Background(), map[string]any{"subcommand": "status"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "f.txt")

	_, err = gt.Execute(context.Background(), map[string]any{"subcommand": "commit", "args": []any{"-m", "x"}})
	require.Error(t, err)
	_, err = gt.Execute(context.Background(), map[string]any{"subcommand": "push"})
	require.Error(t, err)
}

func TestValidateGitArgs(t *testing.T) {
	tests := []struct {
		name    string
		sub     string
		args    []string
		wantErr bool
	}{
		{"diff stat ok", "diff", []string{"--stat"}, false},
		{"show rev ok", "show", []string{"HEAD~1"}, false},
		{"branch list ok", "branch", []string{"--list"}, false},
		{"diff output escapes", "diff", []string{"--output=../x"}, true},
		{"log output eq form", "log", []string{"-o", "/tmp/x"}, true},
		{"branch create positional", "branch", []string{"injected"}, true},
		{"branch delete", "branch", []string{"-D", "wip"}, true},
		{"branch move", "branch", []string{"-m", "old", "new"}, true},
		{"branch set-upstream inline", "branch", []string{"--set-upstream-to=origin/main"}, true},
		{"branch set-upstream fused", "branch", []string{"-uorigin/main"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateGitArgs(tt.sub, tt.args)
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
